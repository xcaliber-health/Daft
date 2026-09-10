"""Commit-hardening tests for rewrite_data_files: OCC retry, partial-progress, idempotency."""

from __future__ import annotations

import logging
import random
import string

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

from pyiceberg.exceptions import CommitFailedException
from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC
from pyiceberg.schema import Schema
from pyiceberg.table import Transaction
from pyiceberg.types import LongType, NestedField, StringType

from daft.catalog import Table
from daft.io.iceberg import RewriteConflict


def _random_strings(n: int, width: int, rng: random.Random) -> list[str]:
    """Generate `n` random length-`width` ASCII strings (poorly compressible).

    Used to inflate parquet file size past the bin-packer's group cap without needing
    a partitioned table (which the current writer doesn't yet support).
    """
    alphabet = string.ascii_letters + string.digits
    return ["".join(rng.choices(alphabet, k=width)) for _ in range(n)]


def _make_multifile_table(local_catalog, name: str, n_files: int = 4, rows_per_file: int = 25_000):
    """Create an unpartitioned table with `n_files` parquet files each ~700KB.

    With target/max-group set to 1 MiB (the validator minimum), two of these files
    can't share a bin-pack group, so the planner emits one group per file — giving
    exactly `n_files` groups for partial-progress to slice.
    """
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "blob", StringType(), required=False),
    )
    table = local_catalog.create_table(
        identifier=name,
        schema=schema,
        partition_spec=UNPARTITIONED_PARTITION_SPEC,
    )
    rng = random.Random(17)
    for f in range(n_files):
        ids = list(range(f * rows_per_file, (f + 1) * rows_per_file))
        blobs = _random_strings(rows_per_file, 32, rng)
        table.append(
            pa.table(
                {
                    "id": pa.array(ids, type=pa.int64()),
                    "blob": pa.array(blobs, type=pa.string()),
                }
            )
        )
    return table


_MULTI_OPTS = {
    "rewrite-all": True,
    "min-input-files": 2,
    "target-file-size-bytes": 1024 * 1024,
    "max-file-group-size-bytes": 1024 * 1024,
}


from tests.io.iceberg.actions._helpers import snapshot_count as _snapshot_count_raw


def _snapshot_count(table) -> int:
    table.refresh()
    return _snapshot_count_raw(table)


def _patch_commit(monkeypatch, behavior):
    """Wrap `Transaction.commit_transaction` so `behavior(call_index, real_commit, self)` runs each time."""
    real_commit = Transaction.commit_transaction
    counter = {"n": 0}

    def wrapper(self):
        counter["n"] += 1
        return behavior(counter["n"], real_commit, self)

    monkeypatch.setattr(Transaction, "commit_transaction", wrapper)
    return counter


def test_occ_retry_succeeds_on_transient_conflict(make_tiny_table, monkeypatch):
    table = make_tiny_table(name="default.t_occ_ok", n_files=6, rows_per_file=3)
    pre_snaps = _snapshot_count(table)

    def behavior(n, real, tx):
        if n == 1:
            raise CommitFailedException("simulated transient conflict")
        return real(tx)

    counter = _patch_commit(monkeypatch, behavior)

    dt = Table.from_iceberg(table)
    result = dt.compact_files(options={"rewrite-all": True, "min-input-files": 2})

    assert counter["n"] == 2, "expected one failure + one success"
    assert result.commits == 1
    assert result.rewritten_files == 6
    assert _snapshot_count(table) == pre_snaps + 1


def test_occ_retry_exhausts_and_raises(make_tiny_table, monkeypatch):
    from daft.io.iceberg import CommitRetryExhausted, RewriteFailedException

    table = make_tiny_table(name="default.t_occ_exhaust", n_files=6, rows_per_file=3)
    pre_snaps = _snapshot_count(table)

    def behavior(n, real, tx):
        raise CommitFailedException(f"always fails (attempt {n})")

    counter = _patch_commit(monkeypatch, behavior)

    dt = Table.from_iceberg(table)
    with pytest.raises(RewriteFailedException) as exc_info:
        dt.compact_files(options={"rewrite-all": True, "min-input-files": 2})

    assert "partial-progress" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, CommitRetryExhausted)
    assert isinstance(exc_info.value.__cause__.__cause__, CommitFailedException)
    assert counter["n"] >= 4, "expected at least num-retries+1 attempts"
    assert _snapshot_count(table) == pre_snaps, "no snapshot must land on full failure"


def test_occ_aborts_when_input_files_vanish(local_catalog, make_tiny_table, monkeypatch):
    # Arrange: the first commit attempt fails, and before the retry another
    # writer removes one of the rewrite's inputs by deleting its rows.
    table = make_tiny_table(name="default.t_occ_conflict", n_files=4, rows_per_file=3)
    competitor = local_catalog.load_table("default.t_occ_conflict")

    def behavior(n, real, tx):
        if n == 1:
            competitor.refresh()
            competitor.delete("id < 3")
            raise CommitFailedException("first attempt")
        return real(tx)

    _patch_commit(monkeypatch, behavior)

    # Act / Assert: the retry sees the removal in the history since the plan.
    dt = Table.from_iceberg(table)
    with pytest.raises(RewriteConflict, match="removed input files"):
        dt.compact_files(options={"rewrite-all": True, "min-input-files": 2})


def test_partial_progress_creates_n_commits(local_catalog):
    # 4 large files → 4 singleton planner groups → 2 batches of 2 groups (max-commits=2).
    table = _make_multifile_table(local_catalog, "default.t_pp_n", n_files=4)
    pre_snaps = _snapshot_count(table)

    dt = Table.from_iceberg(table)
    result = dt.compact_files(
        options={
            **_MULTI_OPTS,
            "partial-progress.enabled": True,
            "partial-progress.max-commits": 2,
        }
    )

    assert result.commits == 2
    assert _snapshot_count(table) == pre_snaps + 2
    assert len(result.snapshot_ids) == 2
    assert result.failed_groups == 0


def _data_files_on_disk(table) -> set[str]:
    from pathlib import Path

    location = table.location().replace("file://", "")
    return {p.name for p in Path(location, "data").glob("*.parquet")}


def test_partial_progress_failed_batch_aggregated(local_catalog, monkeypatch, caplog):
    table = _make_multifile_table(local_catalog, "default.t_pp_fail", n_files=4)
    pre_snaps = _snapshot_count(table)
    originals = _data_files_on_disk(table)

    def behavior(n, real, tx):
        # First batch (call 1) succeeds. Second batch (calls 2..5) all raise → exhaust.
        if n == 1:
            return real(tx)
        raise CommitFailedException(f"batch-2 always fails (call {n})")

    _patch_commit(monkeypatch, behavior)

    dt = Table.from_iceberg(table)
    with caplog.at_level(logging.WARNING, logger="daft.io.iceberg._compact"):
        result = dt.compact_files(
            options={
                **_MULTI_OPTS,
                "partial-progress.enabled": True,
                "partial-progress.max-commits": 2,
            }
        )

    # The first batch landed; the second was refused and its outputs removed.
    # On disk that leaves the originals (replaced inputs are reclaimed later by
    # snapshot expiry) plus the first batch's outputs, and nothing else.
    assert result.commits == 1, "only the first batch should have committed"
    assert result.failed_groups == 2
    assert result.failed_data_files == 2
    assert _snapshot_count(table) == pre_snaps + 1
    table.refresh()
    referenced = {t.file.file_path.rsplit("/", 1)[-1] for t in table.scan().plan_files()}
    assert _data_files_on_disk(table) == originals | referenced, "the failed batch's outputs should be gone"
    assert any("outputs are removed" in rec.message for rec in caplog.records)


def test_partial_progress_commits_a_batch_before_later_groups_are_rewritten(local_catalog, monkeypatch):
    # 4 singleton groups, 2 commits: the first batch must land while the
    # second is still being written.
    table = _make_multifile_table(local_catalog, "default.t_pp_streaming", n_files=4)
    pre_snaps = _snapshot_count(table)
    from daft.io.iceberg import _compact

    real = _compact._rewrite_group
    snapshots_seen: list[int] = []

    def observing(*args, **kwargs):
        kwargs["table"].refresh()
        snapshots_seen.append(len(kwargs["table"].metadata.snapshots))
        return real(*args, **kwargs)

    monkeypatch.setattr(_compact, "_rewrite_group", observing)
    dt = Table.from_iceberg(table)
    # One group at a time makes the moment of each commit observable.
    result = dt.compact_files(
        options={
            **_MULTI_OPTS,
            "partial-progress.enabled": True,
            "partial-progress.max-commits": 2,
            "max-concurrent-file-group-rewrites": 1,
        }
    )

    assert result.commits == 2
    # The third and fourth groups saw the first batch's snapshot already committed.
    assert snapshots_seen[:2] == [pre_snaps, pre_snaps]
    assert snapshots_seen[2:] == [pre_snaps + 1, pre_snaps + 1]


def test_partial_progress_skips_a_group_that_fails_to_rewrite(local_catalog, monkeypatch):
    table = _make_multifile_table(local_catalog, "default.t_pp_group_fail", n_files=4)
    pre_snaps = _snapshot_count(table)
    from daft.io.iceberg import _compact

    real = _compact._rewrite_group
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected rewrite failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(_compact, "_rewrite_group", flaky)
    dt = Table.from_iceberg(table)
    result = dt.compact_files(
        options={**_MULTI_OPTS, "partial-progress.enabled": True, "partial-progress.max-commits": 2}
    )

    # Three groups landed across two commits; the failed one is reported.
    assert result.commits == 2
    assert result.rewritten_files == 3
    assert result.failed_groups == 1
    assert result.failed_data_files == 1
    assert _snapshot_count(table) == pre_snaps + 2


def test_atomic_rewrite_removes_its_outputs_when_the_commit_cannot_land(local_catalog, monkeypatch):
    table = _make_multifile_table(local_catalog, "default.t_atomic_abort", n_files=4)
    before = _data_files_on_disk(table)

    def behavior(n, real, tx):
        raise CommitFailedException(f"always fails (attempt {n})")

    _patch_commit(monkeypatch, behavior)
    dt = Table.from_iceberg(table)
    with pytest.raises(Exception):
        dt.compact_files(options=_MULTI_OPTS)

    assert _data_files_on_disk(table) == before, "an aborted rewrite leaves nothing behind"


def test_atomic_rewrite_removes_its_outputs_when_a_group_fails(local_catalog, monkeypatch):
    table = _make_multifile_table(local_catalog, "default.t_atomic_group_abort", n_files=4)
    before = _data_files_on_disk(table)
    from daft.io.iceberg import _compact

    real = _compact._rewrite_group
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("injected rewrite failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(_compact, "_rewrite_group", flaky)
    dt = Table.from_iceberg(table)
    with pytest.raises(RuntimeError, match="injected"):
        dt.compact_files(options=_MULTI_OPTS)

    assert _data_files_on_disk(table) == before, "the groups written before the failure are removed"


def test_idempotent_replay_across_partial_progress(local_catalog):
    table = _make_multifile_table(local_catalog, "default.t_pp_idemp", n_files=4)
    rid = "pp-replay-me"
    common_opts = {
        **_MULTI_OPTS,
        "partial-progress.enabled": True,
        "partial-progress.max-commits": 2,
        "rewrite-id": rid,
    }

    dt = Table.from_iceberg(table)
    r1 = dt.compact_files(options=common_opts)
    snaps_after_first = _snapshot_count(table)
    assert r1.commits == 2

    r2 = dt.compact_files(options=common_opts)
    assert _snapshot_count(table) == snaps_after_first, "replay must not create new snapshots"
    assert r2.rewrite_id == rid
    assert r2.commits == r1.commits == 2
    assert sorted(r2.snapshot_ids) == sorted(r1.snapshot_ids)


def test_groups_are_rewritten_concurrently_within_the_bound(local_catalog, monkeypatch):
    # Arrange: four singleton groups and a bound of two; observe how many
    # group rewrites are inside the engine at once.
    import threading

    table = _make_multifile_table(local_catalog, "default.t_concurrent_groups", n_files=4)
    from daft.io.iceberg import _compact

    real = _compact._rewrite_group
    lock = threading.Lock()
    state = {"active": 0, "peak": 0}

    def observing(*args, **kwargs):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        try:
            return real(*args, **kwargs)
        finally:
            with lock:
                state["active"] -= 1

    monkeypatch.setattr(_compact, "_rewrite_group", observing)
    before = _snapshot_count(table)

    # Act
    result = Table.from_iceberg(table).compact_files(options={**_MULTI_OPTS, "max-concurrent-file-group-rewrites": 2})

    # Assert: overlap happened, stayed within the bound, and the commit is whole.
    assert state["peak"] == 2
    assert result.rewritten_files == 4
    assert result.commits == 1
    assert _snapshot_count(table) == before + 1
