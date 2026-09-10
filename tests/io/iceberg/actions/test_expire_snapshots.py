"""End-to-end tests for IcebergTable.expire_snapshots()."""

from __future__ import annotations

import os
import threading
import time

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

from pyiceberg.exceptions import CommitFailedException

from daft.catalog import Table
from daft.io.iceberg import ExpireResult
from daft.io.iceberg._expire import plan_expiry  # internal
from tests.io.iceberg.actions._helpers import (
    scan_paths as _data_file_paths,
)
from tests.io.iceberg.actions._helpers import (
    strip_scheme as _strip_scheme,
)


def _snapshot_ids(table) -> list[int]:
    return [s.snapshot_id for s in table.metadata.snapshots]


def _future() -> int:
    """A cutoff every existing snapshot is older than, so retention is by count alone."""
    return int(time.time() * 1000) + 60_000


def _all_files_exist(paths) -> bool:
    return all(os.path.exists(_strip_scheme(p)) for p in paths)


def _any_file_missing(paths) -> bool:
    return any(not os.path.exists(_strip_scheme(p)) for p in paths)


def _metadata_json_files(table) -> list[str]:
    import glob

    meta_dir = os.path.join(_strip_scheme(table.location()), "metadata")
    return sorted(glob.glob(os.path.join(meta_dir, "*.metadata.json")))


def _capped_metadata_table(local_catalog, simple_schema, name: str, n: int):
    from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC

    table = local_catalog.create_table(
        identifier=name,
        schema=simple_schema,
        partition_spec=UNPARTITIONED_PARTITION_SPEC,
        properties={
            # Keep the metadata log short so each commit drops an older entry,
            # but leave the physical files on disk for cleanup to reclaim.
            "write.metadata.previous-versions-max": "1",
            "write.metadata.delete-after-commit.enabled": "false",
        },
    )
    for i in range(n):
        table.append(
            pa.table(
                {
                    "id": pa.array(list(range(i * 3, i * 3 + 3)), type=pa.int64()),
                    "label": pa.array([f"r{i}"] * 3, type=pa.string()),
                }
            )
        )
    return table


def test_clean_expired_metadata_deletes_stale_metadata_json(local_catalog, simple_schema):
    table = _capped_metadata_table(local_catalog, simple_schema, "default.t_exp_md", n=5)
    before = set(_metadata_json_files(table))
    assert len(before) >= 3

    dt = Table.from_iceberg(table)
    result = dt.expire_snapshots(older_than=_future(), retain_last=1, clean_expired_metadata=True)

    table.refresh()
    after = set(_metadata_json_files(table))
    # At least one previously-present metadata file was reclaimed.
    assert result.deleted_metadata_files_count >= 1
    assert before - after, "expected some pre-existing metadata files to be deleted"
    # The live metadata pointer is never deleted.
    assert os.path.exists(_strip_scheme(table.metadata_location))


def test_clean_expired_metadata_default_keeps_metadata_json(local_catalog, simple_schema):
    table = _capped_metadata_table(local_catalog, simple_schema, "default.t_exp_md_off", n=5)
    before = set(_metadata_json_files(table))

    dt = Table.from_iceberg(table)
    result = dt.expire_snapshots(older_than=_future(), retain_last=1)

    table.refresh()
    after = set(_metadata_json_files(table))
    # No metadata cleanup: every pre-existing metadata file survives.
    assert result.deleted_metadata_files_count == 0
    assert before <= after


def test_older_than_removes_old_keeps_new(make_tiny_table):
    table = make_tiny_table(name="default.t_exp_older", n_files=5, rows_per_file=3)
    snaps = list(table.metadata.snapshots)
    assert len(snaps) == 5
    # Cut between snapshot 2 (idx 1) and snapshot 3 (idx 2).
    cutoff_ms = (snaps[1].timestamp_ms + snaps[2].timestamp_ms) // 2

    dt_table = Table.from_iceberg(table)
    result = dt_table.expire_snapshots(older_than=cutoff_ms)

    table.refresh()
    remaining = _snapshot_ids(table)
    assert snaps[0].snapshot_id not in remaining
    assert snaps[1].snapshot_id not in remaining
    assert snaps[2].snapshot_id in remaining
    assert snaps[-1].snapshot_id in remaining
    assert isinstance(result, ExpireResult)


def test_retain_last_keeps_n_most_recent(make_tiny_table):
    table = make_tiny_table(name="default.t_exp_retain", n_files=6, rows_per_file=3)
    pre_ids = _snapshot_ids(table)
    assert len(pre_ids) == 6

    dt_table = Table.from_iceberg(table)
    dt_table.expire_snapshots(older_than=_future(), retain_last=3)

    table.refresh()
    post_ids = _snapshot_ids(table)
    assert len(post_ids) == 3
    # The three retained must be the three most-recent.
    assert set(post_ids) == set(pre_ids[-3:])


def test_explicit_snapshot_ids_bypass_retention(make_tiny_table):
    table = make_tiny_table(name="default.t_exp_ids", n_files=5, rows_per_file=3)
    snaps = list(table.metadata.snapshots)
    target = snaps[1].snapshot_id  # not protected (not current)

    dt_table = Table.from_iceberg(table)
    dt_table.expire_snapshots(snapshot_ids=[target])

    table.refresh()
    assert target not in _snapshot_ids(table)


def test_branch_head_protected(local_catalog, simple_schema):
    from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC

    table = local_catalog.create_table(
        "default.t_exp_branch",
        schema=simple_schema,
        partition_spec=UNPARTITIONED_PARTITION_SPEC,
    )
    for k in range(4):
        table.append(
            pa.table(
                {
                    "id": pa.array(list(range(k * 5, k * 5 + 5)), type=pa.int64()),
                    "label": pa.array([f"r{i}" for i in range(k * 5, k * 5 + 5)]),
                }
            )
        )
    snap0 = table.metadata.snapshots[0].snapshot_id

    with table.manage_snapshots() as ms:
        ms.create_branch(snapshot_id=snap0, branch_name="protect_me")

    # Try to expire everything: snapshot 0 must survive because the branch refs it.
    dt_table = Table.from_iceberg(table)
    dt_table.expire_snapshots(older_than=int(time.time() * 1000) + 60_000)

    table.refresh()
    surviving = set(_snapshot_ids(table))
    assert snap0 in surviving


def test_explicit_protected_id_rejected(local_catalog, simple_schema):
    from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC

    table = local_catalog.create_table(
        "default.t_exp_reject",
        schema=simple_schema,
        partition_spec=UNPARTITIONED_PARTITION_SPEC,
    )
    table.append(pa.table({"id": pa.array([1], type=pa.int64()), "label": pa.array(["a"])}))
    snap = table.current_snapshot().snapshot_id

    with table.manage_snapshots() as ms:
        ms.create_branch(snapshot_id=snap, branch_name="prot")

    dt_table = Table.from_iceberg(table)
    with pytest.raises(ValueError, match="protected"):
        dt_table.expire_snapshots(snapshot_ids=[snap])


def test_min_snapshots_to_keep_property_is_the_default_floor(make_tiny_table):
    # Arrange: the table asks to keep five; the caller names no count.
    table = make_tiny_table(name="default.t_exp_floor", n_files=6, rows_per_file=2)
    with table.transaction() as tx:
        tx.set_properties(**{"history.expire.min-snapshots-to-keep": "5"})
    table.refresh()

    # Act
    Table.from_iceberg(table).expire_snapshots(older_than=_future())

    # Assert: the property applies when the argument is absent.
    table.refresh()
    assert len(table.metadata.snapshots) == 5


def test_retain_last_overrides_the_min_snapshots_property(make_tiny_table):
    # Arrange: the same table property, but the caller names a count.
    table = make_tiny_table(name="default.t_exp_override", n_files=6, rows_per_file=2)
    with table.transaction() as tx:
        tx.set_properties(**{"history.expire.min-snapshots-to-keep": "5"})
    table.refresh()

    # Act
    Table.from_iceberg(table).expire_snapshots(older_than=_future(), retain_last=2)

    # Assert: an explicit count replaces the table default rather than being floored by it.
    table.refresh()
    assert len(table.metadata.snapshots) == 2


def test_gc_enabled_false_refuses(make_tiny_table):
    table = make_tiny_table(name="default.t_exp_gc", n_files=3, rows_per_file=2)
    with table.transaction() as tx:
        tx.set_properties(**{"gc.enabled": "false"})

    dt_table = Table.from_iceberg(table)
    with pytest.raises(ValueError, match="gc.enabled"):
        dt_table.expire_snapshots(retain_last=1)


def test_clean_expired_files_false_is_metadata_only(make_tiny_table):
    table = make_tiny_table(name="default.t_exp_metaonly", n_files=4, rows_per_file=3)
    # Capture all data file paths that exist before expire.
    snaps = list(table.metadata.snapshots)
    all_paths: set[str] = set()
    for s in snaps:
        for m in s.manifests(table.io):
            for e in m.fetch_manifest_entry(table.io, discard_deleted=False):
                all_paths.add(e.data_file.file_path)

    dt_table = Table.from_iceberg(table)
    result = dt_table.expire_snapshots(older_than=_future(), retain_last=1, clean_expired_files=False)

    table.refresh()
    assert len(table.metadata.snapshots) == 1
    assert result == ExpireResult()  # all-zero counts
    # No files removed from disk.
    assert _all_files_exist(all_paths)


def test_result_counts_match_actual_deletions(make_tiny_table):
    # Append-only tables don't orphan data files (later snapshots still reach them via
    # inherited manifests), but each expired snapshot orphans its own manifest list.
    table = make_tiny_table(name="default.t_exp_counts", n_files=4, rows_per_file=3)

    dt_table = Table.from_iceberg(table)
    result = dt_table.expire_snapshots(older_than=_future(), retain_last=1)

    table.refresh()
    surviving = _data_file_paths(table)
    assert _all_files_exist(surviving)
    # 4 appends → 1 retained → 3 expired snapshots → 3 manifest-list orphans.
    assert result.deleted_manifest_lists_count == 3
    # No data files removed because nothing was overwritten or deleted.
    assert result.deleted_data_files_count == 0


def test_result_counts_after_overwrite(make_tiny_table):
    """Overwrite produces data-file orphans once the prior snapshots are expired."""
    table = make_tiny_table(name="default.t_exp_overwrite", n_files=3, rows_per_file=3)
    overwritten = _data_file_paths(table)
    table.overwrite(
        pa.table(
            {
                "id": pa.array([999], type=pa.int64()),
                "label": pa.array(["final"]),
            }
        )
    )
    table.refresh()

    dt_table = Table.from_iceberg(table)
    result = dt_table.expire_snapshots(older_than=_future(), retain_last=1)

    table.refresh()
    surviving = _data_file_paths(table)
    # Surviving file is the post-overwrite one.
    assert _all_files_exist(surviving)
    # Overwritten files have been physically deleted.
    assert _any_file_missing(overwritten)
    assert result.deleted_data_files_count >= len(overwritten)


def test_idempotent_rerun_is_noop(make_tiny_table):
    table = make_tiny_table(name="default.t_exp_idem", n_files=4, rows_per_file=3)
    dt_table = Table.from_iceberg(table)
    dt_table.expire_snapshots(older_than=_future(), retain_last=1)

    table.refresh()
    result2 = dt_table.expire_snapshots(older_than=_future(), retain_last=1)
    assert result2 == ExpireResult()


def test_parallel_delete_observable(make_tiny_table, monkeypatch):
    table = make_tiny_table(name="default.t_exp_parallel", n_files=6, rows_per_file=3)
    active = {"current": 0, "peak": 0}
    lock = threading.Lock()
    real_delete = type(table.io).delete

    def slow_delete(self, location):
        with lock:
            active["current"] += 1
            active["peak"] = max(active["peak"], active["current"])
        try:
            time.sleep(0.1)
            return real_delete(self, location)
        finally:
            with lock:
                active["current"] -= 1

    monkeypatch.setattr(type(table.io), "delete", slow_delete)

    dt_table = Table.from_iceberg(table)
    dt_table.expire_snapshots(
        older_than=_future(),
        retain_last=1,
        options={"max-concurrent-deletes": 4},
    )
    assert active["peak"] >= 2, f"expected >= 2 concurrent deletes, peak={active['peak']}"


def test_notfound_during_delete_is_success(make_tiny_table, monkeypatch):
    table = make_tiny_table(name="default.t_exp_nf", n_files=4, rows_per_file=3)
    real_delete = type(table.io).delete
    state = {"fail_path": None}

    def flaky(self, location):
        path_str = location if isinstance(location, str) else getattr(location, "location", "")
        if state["fail_path"] is None:
            state["fail_path"] = path_str
            raise FileNotFoundError(path_str)
        return real_delete(self, location)

    monkeypatch.setattr(type(table.io), "delete", flaky)

    dt_table = Table.from_iceberg(table)
    result = dt_table.expire_snapshots(older_than=_future(), retain_last=1)
    # FileNotFoundError on the first delete was suppressed; rest proceeded.
    total = result.deleted_data_files_count + result.deleted_manifest_files_count + result.deleted_manifest_lists_count
    assert total >= 1


def test_manifest_read_tolerates_aws_resource_not_found(make_tiny_table, monkeypatch):
    from pyiceberg.manifest import ManifestFile

    table = make_tiny_table(name="default.t_exp_s3_nf", n_files=4, rows_per_file=3)
    real_fetch = ManifestFile.fetch_manifest_entry
    state = {"raised": False}

    def flaky_fetch(self, io, discard_deleted=False):
        if not state["raised"]:
            state["raised"] = True
            raise OSError(
                f"AWS Error RESOURCE_NOT_FOUND during GetObject operation: "
                f"No response body. (key: {self.manifest_path})"
            )
        return real_fetch(self, io, discard_deleted=discard_deleted)

    monkeypatch.setattr(ManifestFile, "fetch_manifest_entry", flaky_fetch)

    dt_table = Table.from_iceberg(table)
    result = dt_table.expire_snapshots(older_than=_future(), retain_last=1)
    assert state["raised"]
    assert isinstance(result, ExpireResult)


def test_collect_paths_tolerates_aws_resource_not_found_on_manifests_list(make_tiny_table, monkeypatch):
    from pyiceberg.table.snapshots import Snapshot

    table = make_tiny_table(name="default.t_exp_s3_nf_ml", n_files=4, rows_per_file=3)
    real_manifests = Snapshot.manifests
    state = {"raised": False}

    def flaky_manifests(self, io):
        if not state["raised"]:
            state["raised"] = True
            raise OSError("AWS Error RESOURCE_NOT_FOUND during GetObject operation: manifest list gone")
        return real_manifests(self, io)

    monkeypatch.setattr(Snapshot, "manifests", flaky_manifests)

    dt_table = Table.from_iceberg(table)
    result = dt_table.expire_snapshots(older_than=_future(), retain_last=1)
    assert state["raised"]
    assert isinstance(result, ExpireResult)


def test_commit_conflict_retries(make_tiny_table, monkeypatch):
    table = make_tiny_table(name="default.t_exp_commit", n_files=4, rows_per_file=3)

    # Reach into pyiceberg's ExpireSnapshots.commit and fail the first call only.
    from pyiceberg.table.update.snapshot import ExpireSnapshots as _PyExpireSnapshots

    real_commit = _PyExpireSnapshots.commit
    calls = {"n": 0}

    def flaky_commit(self):
        calls["n"] += 1
        if calls["n"] == 1:
            raise CommitFailedException("transient")
        return real_commit(self)

    monkeypatch.setattr(_PyExpireSnapshots, "commit", flaky_commit)

    dt_table = Table.from_iceberg(table)
    dt_table.expire_snapshots(older_than=_future(), retain_last=1)

    table.refresh()
    assert calls["n"] >= 2
    assert len(table.metadata.snapshots) == 1


def test_retain_last_below_one_rejected(make_tiny_table):
    table = make_tiny_table(name="default.t_exp_bad", n_files=2, rows_per_file=2)
    dt_table = Table.from_iceberg(table)
    with pytest.raises(ValueError, match="retain_last"):
        dt_table.expire_snapshots(older_than=_future(), retain_last=0)


def test_no_args_falls_back_to_max_age_property(make_tiny_table):
    table = make_tiny_table(name="default.t_exp_noargs", n_files=4, rows_per_file=2)
    # Configure a very short max-age so every snapshot is older than the threshold.
    with table.transaction() as tx:
        tx.set_properties(**{"history.expire.max-snapshot-age-ms": "1"})
    table.refresh()

    dt_table = Table.from_iceberg(table)
    dt_table.expire_snapshots()

    table.refresh()
    # The current snapshot is protected by the main ref; everything else expired.
    assert len(table.metadata.snapshots) == 1


def _branch_table(local_catalog, simple_schema, name: str, n: int):
    from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC

    table = local_catalog.create_table(name, schema=simple_schema, partition_spec=UNPARTITIONED_PARTITION_SPEC)
    for k in range(n):
        table.append(pa.table({"id": pa.array([k], type=pa.int64()), "label": pa.array([f"r{k}"])}))
    return table


def test_retain_last_alone_keeps_snapshots_younger_than_the_table_age(make_tiny_table):
    # Arrange: every snapshot is seconds old; the table's maximum age is five days.
    table = make_tiny_table(name="default.t_exp_young", n_files=4, rows_per_file=2)

    # Act: a count alone sets a floor, not a ceiling.
    result = Table.from_iceberg(table).expire_snapshots(retain_last=1)

    # Assert
    table.refresh()
    assert len(table.metadata.snapshots) == 4
    assert result == ExpireResult()


def test_older_than_and_retain_last_combine_as_cutoff_and_floor(make_tiny_table):
    # Arrange: a cutoff between snapshots 1 and 2, and a floor of 3.
    table = make_tiny_table(name="default.t_exp_combined", n_files=5, rows_per_file=2)
    snaps = list(table.metadata.snapshots)
    cutoff = (snaps[1].timestamp_ms + snaps[2].timestamp_ms) // 2

    # Act
    Table.from_iceberg(table).expire_snapshots(older_than=cutoff, retain_last=3)

    # Assert: the floor keeps the three most recent; the cutoff alone would have
    # expired snapshots 0 and 1, and the floor does not expire 2.
    table.refresh()
    assert _snapshot_ids(table) == [s.snapshot_id for s in snaps[2:]]

    # A wider floor keeps snapshots the cutoff would expire.
    table2 = make_tiny_table(name="default.t_exp_combined_floor", n_files=5, rows_per_file=2)
    snaps2 = list(table2.metadata.snapshots)
    Table.from_iceberg(table2).expire_snapshots(older_than=_future(), retain_last=4)
    table2.refresh()
    assert _snapshot_ids(table2) == [s.snapshot_id for s in snaps2[1:]]


def test_plan_walks_each_branch_with_its_own_settings(local_catalog, simple_schema):
    # Arrange: main has five snapshots; a branch off the second keeps two
    # regardless of age, and its own snapshots are separate ancestry.
    table = _branch_table(local_catalog, simple_schema, "default.t_exp_branch_walk", n=5)
    snaps = list(table.metadata.snapshots)
    with table.manage_snapshots() as ms:
        ms.create_branch(snapshot_id=snaps[1].snapshot_id, branch_name="audit", min_snapshots_to_keep=2)
    table.refresh()

    # Act: expire everything older than now, keeping one on main.
    plan = plan_expiry(table, older_than=_future(), retain_last=1, snapshot_ids=None, now_ms=int(time.time() * 1000))

    # Assert: main keeps its head; the branch keeps its head and its parent.
    kept = {s.snapshot_id for s in snaps} - set(plan.snapshot_ids)
    assert kept == {snaps[4].snapshot_id, snaps[1].snapshot_id, snaps[0].snapshot_id}
    assert plan.protected_ids == frozenset({snaps[4].snapshot_id, snaps[1].snapshot_id})
    assert plan.ref_names == ()


def test_plan_keeps_an_unreferenced_snapshot_younger_than_the_cutoff(local_catalog, simple_schema):
    # Arrange: overwrite a branch head so its old snapshot is no longer reachable
    # from any reference, then plan with a cutoff in the past.
    table = _branch_table(local_catalog, simple_schema, "default.t_exp_unreferenced", n=3)
    snaps = list(table.metadata.snapshots)
    with table.manage_snapshots() as ms:
        ms.create_tag(snapshot_id=snaps[0].snapshot_id, tag_name="old")
    table.refresh()
    with table.manage_snapshots() as ms:
        ms.remove_tag("old")
    table.refresh()
    with table.manage_snapshots() as ms:
        ms.create_branch(snapshot_id=snaps[2].snapshot_id, branch_name="fork")
    table.refresh()
    # Rewrite main's history so snapshot 1 is reachable only from the fork.
    with table.manage_snapshots() as ms:
        ms.set_current_snapshot(snapshot_id=snaps[0].snapshot_id)
    table.refresh()
    table.append(pa.table({"id": pa.array([99], type=pa.int64()), "label": pa.array(["new"])}))
    table.refresh()
    with table.manage_snapshots() as ms:
        ms.remove_branch("fork")
    table.refresh()
    reachable = {
        s.snapshot_id
        for s in table.metadata.snapshots
        if s.snapshot_id in (snaps[0].snapshot_id, table.current_snapshot().snapshot_id)
    }
    unreferenced = {s.snapshot_id for s in table.metadata.snapshots} - reachable
    assert unreferenced

    # Act: cutoff in the past keeps young unreferenced snapshots; a future cutoff drops them.
    past = plan_expiry(
        table, older_than=snaps[0].timestamp_ms - 1, retain_last=1, snapshot_ids=None, now_ms=int(time.time() * 1000)
    )
    future = plan_expiry(table, older_than=_future(), retain_last=1, snapshot_ids=None, now_ms=int(time.time() * 1000))

    # Assert
    assert unreferenced.isdisjoint(past.snapshot_ids)
    assert unreferenced <= set(future.snapshot_ids)


def test_an_aged_tag_is_removed_and_its_snapshot_expires(local_catalog, simple_schema):
    # Arrange: a tag that ages out after one millisecond.
    table = _branch_table(local_catalog, simple_schema, "default.t_exp_aged_tag", n=3)
    snaps = list(table.metadata.snapshots)
    with table.manage_snapshots() as ms:
        ms.create_tag(snapshot_id=snaps[0].snapshot_id, tag_name="stale", max_ref_age_ms=1)
    table.refresh()
    time.sleep(0.01)

    # Act
    Table.from_iceberg(table).expire_snapshots(older_than=_future(), retain_last=1)

    # Assert: the tag is gone and the snapshot it held expired with the rest.
    table.refresh()
    assert "stale" not in table.metadata.refs
    assert _snapshot_ids(table) == [snaps[2].snapshot_id]


def test_a_live_tag_keeps_its_snapshot(local_catalog, simple_schema):
    table = _branch_table(local_catalog, simple_schema, "default.t_exp_live_tag", n=3)
    snaps = list(table.metadata.snapshots)
    with table.manage_snapshots() as ms:
        ms.create_tag(snapshot_id=snaps[0].snapshot_id, tag_name="keep")
    table.refresh()

    Table.from_iceberg(table).expire_snapshots(older_than=_future(), retain_last=1)

    table.refresh()
    assert "keep" in table.metadata.refs
    assert set(_snapshot_ids(table)) == {snaps[0].snapshot_id, snaps[2].snapshot_id}


def test_explicit_ids_expire_regardless_of_retention(make_tiny_table):
    table = make_tiny_table(name="default.t_exp_explicit_plan", n_files=5, rows_per_file=2)
    snaps = list(table.metadata.snapshots)

    plan = plan_expiry(
        table,
        older_than=None,
        retain_last=None,
        snapshot_ids=[snaps[3].snapshot_id],
        now_ms=int(time.time() * 1000),
    )

    assert set(plan.snapshot_ids) == {snaps[3].snapshot_id}
    assert snaps[-1].snapshot_id in plan.protected_ids
