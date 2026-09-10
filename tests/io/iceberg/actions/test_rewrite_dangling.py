"""Which delete files a rewrite may drop, and when.

A position delete applies to data files whose sequence number is at or below
its own, so it is dangling only when every live data file in its partition is
newer. An equality delete applies only to strictly older data files, so one at
the partition's lowest data sequence number is already dangling. Dropping a
delete that still applies brings the rows it removed back.

Two moments shed deletes. The rewrite commit itself drops any delete older than
every data file left live in the whole table, as any merging commit does. The
optional post-pass then judges the rest partition by partition.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import IdentityTransform
from pyiceberg.typedef import Record
from pyiceberg.types import LongType, NestedField, StringType

from daft.catalog import Table
from daft.io.iceberg._compact import _delete_is_dangling, _remove_dangling_deletes  # internal
from tests.io.iceberg.actions._helpers import (
    commit_positional_deletes,
    commit_row_delta,
    make_seeded_table,
    read_ids,
    scan_paths,
)

_REWRITE = {"rewrite-all": True, "min-input-files": 2, "remove-dangling-deletes": True}


def _live_delete_paths(table) -> set[str]:
    table.refresh()
    return {d.file_path for task in table.scan().plan_files() for d in task.delete_files}


def _rows(n: int, start: int, label: str) -> pa.Table:
    return pa.table(
        {
            "id": pa.array(list(range(start, start + n)), type=pa.int64()),
            "label": pa.array([label] * n, type=pa.string()),
        }
    )


def test_remove_dangling_deletes_option_is_safe_noop(make_tiny_table):
    table = make_tiny_table(name="default.t_dangling", n_files=6, rows_per_file=3)
    dt = Table.from_iceberg(table)
    result = dt.compact_files(options=_REWRITE)
    table.refresh()
    # No deletes to sweep → removed_delete_files unchanged from main rewrite path.
    assert result.removed_delete_files == 0
    assert result.added_files >= 1


def test_remove_dangling_deletes_helper_on_clean_table(make_tiny_table):
    table = make_tiny_table(name="default.t_dangling_helper", n_files=4, rows_per_file=2)
    snapshots_before = len(table.metadata.snapshots)

    removed, snapshot_id = _remove_dangling_deletes(table, branch=None, rewrite_id="r1", strategy="binpack")

    # Nothing dangling: nothing removed and no snapshot committed.
    assert (removed, snapshot_id) == (0, None)
    assert len(table.refresh().metadata.snapshots) == snapshots_before


@pytest.mark.parametrize(
    ("content", "sequence_number", "min_data_sequence_number", "expected"),
    [
        pytest.param("position", 1, 2, True, id="position-below-minimum"),
        pytest.param("position", 2, 2, False, id="position-at-minimum-still-applies"),
        pytest.param("position", 3, 2, False, id="position-above-minimum"),
        pytest.param("equality", 1, 2, True, id="equality-below-minimum"),
        pytest.param("equality", 2, 2, True, id="equality-at-minimum-cannot-apply"),
        pytest.param("equality", 3, 2, False, id="equality-above-minimum"),
    ],
)
def test_dangling_rule_by_delete_kind(content, sequence_number, min_data_sequence_number, expected):
    from pyiceberg.manifest import DataFileContent

    kind = DataFileContent.POSITION_DELETES if content == "position" else DataFileContent.EQUALITY_DELETES

    assert _delete_is_dangling(kind, sequence_number, min_data_sequence_number) is expected


def test_position_delete_written_with_its_data_file_is_kept(local_catalog):
    # Arrange: a data file and a delete over it land in one snapshot, so they
    # share the partition's lowest live data sequence number.
    table = make_seeded_table(local_catalog, "default.t_dangling_row_delta", n_files=2, rows_per_file=10)
    _, delete_path = commit_row_delta(table, _rows(10, 1_000, "delta"), deleted_positions=[0, 1, 2])
    visible_before = read_ids(table)

    # Act
    removed, snapshot_id = _remove_dangling_deletes(table, branch=None, rewrite_id="r1", strategy="binpack")

    # Assert: the delete still applies, so it stays and the rows stay removed.
    assert (removed, snapshot_id) == (0, None)
    assert delete_path in _live_delete_paths(table)
    assert read_ids(table) == visible_before
    assert set(range(1_000, 1_003)).isdisjoint(read_ids(table))


def test_rewrite_keeps_rows_deleted_in_the_same_snapshot_as_their_file(local_catalog):
    # Arrange: seed files to rewrite, plus a newer file with a delete over it
    # that the rewrite leaves alone. The outputs take the plan snapshot's
    # sequence number, which equals the delete's.
    table = make_seeded_table(local_catalog, "default.t_dangling_rewrite_scope", n_files=4, rows_per_file=10)
    delta_path, delete_path = commit_row_delta(table, _rows(10, 1_000, "delta"), deleted_positions=[0, 1, 2])
    visible_before = read_ids(table)
    dt = Table.from_iceberg(table)

    # Act: rewrite only the seed rows' files.
    result = dt.rewrite_data_files("binpack", where="label = 'seed'", options=_REWRITE)

    # Assert: the delta file was not rewritten, its delete survived, and no
    # deleted row came back.
    assert result.rewritten_files == 4
    assert delta_path in scan_paths(table)
    assert delete_path in _live_delete_paths(table)
    assert read_ids(table) == visible_before


def test_rewrite_commit_drops_a_delete_older_than_every_live_data_file(local_catalog):
    # Arrange: a delete over a seed file, then a later append, so the plan
    # snapshot's sequence number exceeds the delete's.
    table = make_seeded_table(local_catalog, "default.t_dangling_genuine", n_files=3, rows_per_file=10)
    seed_path = sorted(scan_paths(table))[0]
    delete_path = commit_positional_deletes(table, {seed_path: [0, 1]})
    table.append(_rows(10, 2_000, "later"))
    visible_before = read_ids(table)
    dt = Table.from_iceberg(table)

    # Act: every input is replaced at the plan snapshot's sequence number, so
    # the delete ends up older than every live data file in the table.
    result = dt.rewrite_data_files("binpack", options={"rewrite-all": True, "min-input-files": 2})

    # Assert: the rewrite snapshot itself sheds the delete; no post-pass ran.
    assert result.rewritten_files == 4
    assert result.commits == 1
    assert delete_path not in _live_delete_paths(table)
    rewrite = table.metadata.snapshot_by_id(result.snapshot_ids[0])
    assert rewrite.summary.additional_properties["removed-delete-files"] == "1"
    assert rewrite.summary.additional_properties["daft.rewrite-dropped-delete-files"] == "1"
    assert read_ids(table) == visible_before


def _make_partitioned_table(catalog, name: str):
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "label", StringType(), required=False),
        NestedField(3, "region", StringType(), required=False),
    )
    spec = PartitionSpec(PartitionField(source_id=3, field_id=1000, transform=IdentityTransform(), name="region"))
    return catalog.create_table(identifier=name, schema=schema, partition_spec=spec)


def _region_rows(n: int, start: int, region: str) -> pa.Table:
    return pa.table(
        {
            "id": pa.array(list(range(start, start + n)), type=pa.int64()),
            "label": pa.array([region] * n, type=pa.string()),
            "region": pa.array([region] * n, type=pa.string()),
        }
    )


def test_post_pass_reclaims_a_delete_dangling_in_its_partition_only(local_catalog):
    # Arrange: an old partition keeps the table-wide minimum sequence number
    # low, so the rewrite commit cannot shed the delete; only the partition
    # rule of the post-pass can.
    table = _make_partitioned_table(local_catalog, "default.t_dangling_partition")
    table.append(_region_rows(10, 0, "cold"))
    for k in range(3):
        table.append(_region_rows(10, 100 + k * 10, "hot"))
    hot_path = sorted(p for p in scan_paths(table) if "region=hot" in p)[0]
    delete_path = commit_positional_deletes(table, {hot_path: [0, 1]}, partition=Record("hot"))
    table.append(_region_rows(10, 2_000, "hot"))
    visible_before = read_ids(table)
    dt = Table.from_iceberg(table)

    # Act
    result = dt.rewrite_data_files("binpack", where="region = 'hot'", options=_REWRITE)

    # Assert: reclaimed by a replace snapshot of its own, accounted for in the result.
    assert result.rewritten_files == 4
    assert result.commits == 2
    assert len(result.snapshot_ids) == 2
    assert delete_path not in _live_delete_paths(table)
    rewrite = table.metadata.snapshot_by_id(result.snapshot_ids[0])
    assert rewrite.summary.additional_properties["daft.rewrite-dropped-delete-files"] == "0"
    reclaim = table.metadata.snapshot_by_id(result.snapshot_ids[-1])
    assert reclaim.summary.operation.value == "replace"
    assert reclaim.summary.additional_properties["daft.rewrite-batch"] == "dangling"
    assert result.removed_delete_files >= 1
    assert read_ids(table) == visible_before


def test_reclaim_replays_with_its_rewrite(local_catalog):
    # Arrange: a rewrite whose post-pass reclaims a delete, under an explicit id.
    table = _make_partitioned_table(local_catalog, "default.t_dangling_replay")
    table.append(_region_rows(10, 0, "cold"))
    for k in range(3):
        table.append(_region_rows(10, 100 + k * 10, "hot"))
    hot_path = sorted(p for p in scan_paths(table) if "region=hot" in p)[0]
    commit_positional_deletes(table, {hot_path: [0, 1]}, partition=Record("hot"))
    table.append(_region_rows(10, 2_000, "hot"))
    dt = Table.from_iceberg(table)
    options = {**_REWRITE, "rewrite-id": "replayed"}
    first = dt.rewrite_data_files("binpack", where="region = 'hot'", options=options)
    snapshots_after_first = len(table.refresh().metadata.snapshots)

    # Act: the same call again.
    second = dt.rewrite_data_files("binpack", where="region = 'hot'", options=options)

    # Assert: recognized as a replay, including the reclaiming snapshot; nothing new lands.
    assert first.commits == 2
    assert second.commits == 2
    assert second.snapshot_ids == first.snapshot_ids
    assert second.removed_delete_files == first.removed_delete_files
    assert (second.rewritten_files, second.added_files) == (first.rewritten_files, first.added_files)
    assert (second.bytes_rewritten, second.bytes_added) == (first.bytes_rewritten, first.bytes_added)
    assert len(table.refresh().metadata.snapshots) == snapshots_after_first


def test_a_format_version_one_table_rewrites_without_sequence_rules(local_catalog, simple_schema):
    # Arrange: version 1 carries neither sequence numbers nor delete files.
    from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC

    table = local_catalog.create_table(
        "default.t_dangling_v1",
        schema=simple_schema,
        partition_spec=UNPARTITIONED_PARTITION_SPEC,
        properties={"format-version": "1"},
    )
    for k in range(3):
        table.append(_rows(5, k * 5, "seed"))
    visible_before = read_ids(table)

    # Act
    result = Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE)

    # Assert
    assert result.rewritten_files == 3
    assert result.commits == 1
    assert read_ids(table) == visible_before
