"""End-to-end tests for IcebergTable.remove_orphan_files()."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import pytest

pytest.importorskip("pyiceberg")

from daft.catalog import Table
from daft.io.iceberg import (
    PrefixMismatchError,
    RemoveOrphanResult,
)
from tests.io.iceberg.actions._helpers import (
    read_ids as _read_ids,
)
from tests.io.iceberg.actions._helpers import (
    scan_paths as _scan_paths,
)
from tests.io.iceberg.actions._helpers import (
    strip_scheme as _strip_scheme,
)


def test_default_scheme_equivalence_collapses_s3_variants():
    from daft.io.iceberg._remove_orphan import _build_canonicalizer

    canon = _build_canonicalizer({})
    assert canon.canonical("s3a://b/x/f.parquet") == canon.canonical("s3://b/x/f.parquet")
    assert canon.canonical("s3n://b/x/f.parquet") == canon.canonical("s3://b/x/f.parquet")


def test_custom_scheme_and_authority_equivalence_collapse():
    from daft.io.iceberg._remove_orphan import _build_canonicalizer

    canon = _build_canonicalizer(
        {
            "equal-schemes": {"myfs": "s3"},
            "equal-authorities": {"bucket.vpce-123": "bucket"},
        }
    )
    assert canon.canonical("myfs://bucket.vpce-123/t/f.parquet") == canon.canonical("s3://bucket/t/f.parquet")


def _patch_frames(monkeypatch, *, reachable: list[str], listed: list[str]) -> None:
    """Drive the orphan computation with controlled reachable/listed path frames."""
    import daft
    from daft.io.iceberg import _remove_orphan as mod

    monkeypatch.setattr(mod, "_reachable_frame", lambda _table: daft.from_pydict({"path": reachable}))
    monkeypatch.setattr(mod, "_listed_frame", lambda *_a, **_k: daft.from_pydict({"path": listed}))


def test_equal_authorities_keeps_live_file_listed_under_alias_host(make_tiny_table, monkeypatch):
    """A reachable file listed under a declared-equivalent host is not an orphan."""
    table = make_tiny_table(name="default.t_orf_eq_auth", n_files=2, rows_per_file=2)
    _patch_frames(
        monkeypatch,
        reachable=["s3://real-host/t/data/live.parquet"],
        listed=["s3://alias-host/t/data/live.parquet"],
    )

    result = Table.from_iceberg(table).remove_orphan_files(
        dry_run=True,
        prefix_mismatch_mode="error",
        options={"equal-authorities": {"alias-host": "real-host"}},
    )
    # The alias host canonicalizes to the reachable host, so the live file is
    # neither a prefix mismatch nor an orphan.
    assert result.orphan_files_count == 0
    assert result.skipped_prefix_mismatch_count == 0


def test_unknown_authority_still_flagged_as_mismatch(make_tiny_table, monkeypatch):
    """A truly foreign authority is still caught by error mode."""
    table = make_tiny_table(name="default.t_orf_unknown", n_files=2, rows_per_file=2)
    _patch_frames(
        monkeypatch,
        reachable=["s3://real-host/t/data/live.parquet"],
        listed=["s3://other-host/t/data/stray.parquet"],
    )

    with pytest.raises(PrefixMismatchError):
        Table.from_iceberg(table).remove_orphan_files(dry_run=True, prefix_mismatch_mode="error")


def _plant_file(directory: str, name: str, content: bytes = b"orphan") -> str:
    path = os.path.join(directory, name)
    os.makedirs(directory, exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)
    # Backdate so the listing filter (mtime < cutoff) never races the test's
    # subsequent older_than=now() call.
    backdated = time.time() - 60
    os.utime(path, (backdated, backdated))
    return path


def test_orphan_in_data_and_metadata_dirs_deleted(make_tiny_table):
    table = make_tiny_table(name="default.t_orf_basic", n_files=3, rows_per_file=2)
    base = _strip_scheme(table.location())
    planted_data = _plant_file(os.path.join(base, "data"), "stray.parquet")
    planted_meta = _plant_file(os.path.join(base, "metadata"), "stray.avro")

    pre_ids = _read_ids(table)

    dt = Table.from_iceberg(table)
    result = dt.remove_orphan_files(
        older_than=datetime.now(tz=timezone.utc),
        options={"allow-recent": True},
    )

    assert isinstance(result, RemoveOrphanResult)
    assert result.orphan_files_count == 2
    assert result.deleted_files_count == 2
    assert result.failed_deletes == 0
    assert not os.path.exists(planted_data)
    assert not os.path.exists(planted_meta)
    assert _read_ids(table) == pre_ids


def test_dry_run_reports_without_deleting(make_tiny_table):
    table = make_tiny_table(name="default.t_orf_dry", n_files=2, rows_per_file=2)
    base = _strip_scheme(table.location())
    planted = _plant_file(os.path.join(base, "data"), "stray.parquet")

    dt = Table.from_iceberg(table)
    result = dt.remove_orphan_files(
        older_than=datetime.now(tz=timezone.utc),
        dry_run=True,
        options={"allow-recent": True},
    )

    assert result.orphan_files_count == 1
    assert result.deleted_files_count == 0
    assert os.path.exists(planted)
    assert any(p.endswith("stray.parquet") for p in result.sample_paths)


def test_clean_table_reports_zero(make_tiny_table):
    table = make_tiny_table(name="default.t_orf_clean", n_files=2, rows_per_file=2)
    dt = Table.from_iceberg(table)
    result = dt.remove_orphan_files(
        older_than=datetime.now(tz=timezone.utc),
        options={"allow-recent": True},
    )
    assert result.orphan_files_count == 0
    assert result.deleted_files_count == 0


def test_recent_cutoff_rejected(make_tiny_table):
    table = make_tiny_table(name="default.t_orf_recent", n_files=1, rows_per_file=1)
    dt = Table.from_iceberg(table)
    with pytest.raises(ValueError, match="cutoff newer than 24 hours"):
        dt.remove_orphan_files(older_than=datetime.now(tz=timezone.utc))


def test_default_older_than_three_days(make_tiny_table):
    """Without options, default cutoff is 3 days ago — no recent file is touched."""
    table = make_tiny_table(name="default.t_orf_default", n_files=2, rows_per_file=2)
    base = _strip_scheme(table.location())
    planted = _plant_file(os.path.join(base, "data"), "stray.parquet")
    pre_ids = _read_ids(table)

    dt = Table.from_iceberg(table)
    result = dt.remove_orphan_files()

    assert result.orphan_files_count == 0
    assert os.path.exists(planted)
    assert _read_ids(table) == pre_ids


def test_gc_enabled_false_rejected(make_tiny_table):
    table = make_tiny_table(name="default.t_orf_gc", n_files=1, rows_per_file=1)
    with table.transaction() as tx:
        tx.set_properties(**{"gc.enabled": "false"})
    table.refresh()

    dt = Table.from_iceberg(table)
    with pytest.raises(ValueError, match="gc.enabled=false"):
        dt.remove_orphan_files(
            older_than=datetime.now(tz=timezone.utc),
            options={"allow-recent": True},
        )


def test_location_must_be_subpath(make_tiny_table):
    table = make_tiny_table(name="default.t_orf_loc", n_files=1, rows_per_file=1)
    dt = Table.from_iceberg(table)
    with pytest.raises(ValueError, match="not a subpath"):
        dt.remove_orphan_files(
            location="file:///elsewhere",
            older_than=datetime.now(tz=timezone.utc),
            options={"allow-recent": True},
        )


def test_invalid_prefix_mismatch_mode_rejected(make_tiny_table):
    table = make_tiny_table(name="default.t_orf_mode", n_files=1, rows_per_file=1)
    dt = Table.from_iceberg(table)
    with pytest.raises(ValueError, match="prefix_mismatch_mode"):
        dt.remove_orphan_files(prefix_mismatch_mode="bogus")


def test_file_list_view_used_instead_of_listing(make_tiny_table, monkeypatch):
    """A supplied inventory replaces object-store listing and finds orphans in it."""
    import daft

    table = make_tiny_table(name="default.t_orf_flv", n_files=2, rows_per_file=2)
    orphan = f"{table.location().rstrip('/')}/data/ghost-0000.parquet"
    old_ms = int(time.time() * 1000) - 10 * 24 * 60 * 60 * 1000
    view = daft.from_pydict({"file_path": [orphan], "last_modified": [old_ms]})

    def _no_glob(*_a, **_k):
        raise AssertionError("from_glob_path must not be called when a view is given")

    monkeypatch.setattr(daft, "from_glob_path", _no_glob)

    result = Table.from_iceberg(table).remove_orphan_files(file_list_view=view, dry_run=True)
    assert result.orphan_files_count == 1
    assert orphan in result.sample_paths


def test_file_list_view_bad_schema_rejected(make_tiny_table):
    import daft

    table = make_tiny_table(name="default.t_orf_flv_bad", n_files=2, rows_per_file=2)
    view = daft.from_pydict({"path": ["x"], "mtime": [0]})  # wrong column names

    with pytest.raises(ValueError, match="file_path"):
        Table.from_iceberg(table).remove_orphan_files(file_list_view=view, dry_run=True)


def test_prefix_listing_flag_accepted(make_tiny_table):
    table = make_tiny_table(name="default.t_orf_pfx", n_files=2, rows_per_file=2)
    result = Table.from_iceberg(table).remove_orphan_files(
        prefix_listing=True,
        dry_run=True,
        options={"allow-recent": True},
        older_than=datetime.now(tz=timezone.utc),
    )
    # No orphans planted: a clean table reports none regardless of listing mode.
    assert result.orphan_files_count == 0


def test_python_canonical_matches_engine_expression():
    import daft
    from daft.io.iceberg._engine import canon_path_expr
    from daft.io.iceberg._remove_orphan import _build_canonicalizer

    spec = _build_canonicalizer({"equal-schemes": {"myfs": "s3"}, "equal-authorities": {"h1": "h2"}})
    paths = [
        "s3a://b/k/f.parquet",
        "s3://b/k/f.parquet",
        "myfs://h1/t/x.parquet",
        "s3://h1/y",
        "file:///t/a.parquet",
        "no-scheme/path",
    ]
    engine = daft.from_pydict({"path": paths}).with_column("c", canon_path_expr("path", spec)).to_pydict()["c"]
    assert engine == [spec.canonical(p) for p in paths]


def test_prefix_mismatch_error_raises(make_tiny_table, monkeypatch):
    """If the reachable set has a different scheme than the listing, ERROR mode raises."""
    table = make_tiny_table(name="default.t_orf_mm_err", n_files=2, rows_per_file=2)
    _patch_frames(
        monkeypatch,
        reachable=["s3://fake-bucket/t/data/live.parquet"],
        listed=["file:///t/data/stray.parquet"],
    )

    dt = Table.from_iceberg(table)
    with pytest.raises(PrefixMismatchError):
        dt.remove_orphan_files(dry_run=True, prefix_mismatch_mode="error")


def test_prefix_mismatch_ignore_skips_and_counts(make_tiny_table, monkeypatch):
    table = make_tiny_table(name="default.t_orf_mm_ign", n_files=2, rows_per_file=2)
    _patch_frames(
        monkeypatch,
        reachable=["s3://fake-bucket/t/data/live.parquet"],
        listed=[
            "file:///t/data/stray-1.parquet",
            "file:///t/data/stray-2.parquet",
        ],
    )

    dt = Table.from_iceberg(table)
    result = dt.remove_orphan_files(dry_run=True, prefix_mismatch_mode="ignore")
    # Every listed file uses a scheme absent from the reachable set, so all are
    # skipped as mismatches and none counts as an orphan.
    assert result.orphan_files_count == 0
    assert result.skipped_prefix_mismatch_count == 2


def test_non_current_snapshot_file_is_reachable(make_tiny_table):
    """Files referenced only by older (but not expired) snapshots must NOT be deleted."""
    table = make_tiny_table(name="default.t_orf_history", n_files=4, rows_per_file=3)
    older_snapshot_paths = _scan_paths(table)
    assert older_snapshot_paths

    # Compaction switches the current snapshot to point at new files but the old
    # data files are still referenced by older retained snapshots.
    dt = Table.from_iceberg(table)
    dt.compact_files(
        options={
            "target-file-size-bytes": 64 * 1024 * 1024,
            "min-input-files": 2,
            "rewrite-all": True,
        }
    )
    table.refresh()
    current_paths = _scan_paths(table)
    assert older_snapshot_paths.isdisjoint(current_paths)

    result = dt.remove_orphan_files(
        older_than=datetime.now(tz=timezone.utc),
        options={"allow-recent": True},
    )
    assert result.orphan_files_count == 0
    for p in older_snapshot_paths:
        assert os.path.exists(_strip_scheme(p)), (
            f"pre-rewrite file removed despite being reachable from a retained snapshot: {p}"
        )


def test_sample_limit_caps_returned_paths(make_tiny_table):
    table = make_tiny_table(name="default.t_orf_sample", n_files=1, rows_per_file=1)
    base = _strip_scheme(table.location())
    for i in range(5):
        _plant_file(os.path.join(base, "data"), f"stray-{i}.parquet")

    dt = Table.from_iceberg(table)
    result = dt.remove_orphan_files(
        older_than=datetime.now(tz=timezone.utc),
        dry_run=True,
        options={"allow-recent": True, "sample-limit": 2},
    )
    assert result.orphan_files_count == 5
    assert len(result.sample_paths) == 2
