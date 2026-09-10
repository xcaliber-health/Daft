"""Packing position delete files.

Position deletes accumulate one small file per commit. The rewrite packs the
live position delete files of each partition into target-sized files, drops
rows naming a data file that is no longer live, and keeps the packed files at
the sequence number the old ones carried, so the same rows stay deleted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("pyiceberg")

import daft
from daft.catalog import Table
from daft.io.iceberg import RewritePositionDeletesResult
from tests.io.iceberg.actions._helpers import (
    commit_positional_deletes,
    make_seeded_table,
    scan_paths,
)

if TYPE_CHECKING:
    from pyiceberg.table import Table as PyIcebergTable

_PACK = {"rewrite-all": True, "min-input-files": 2}


def _delete_entries(table: PyIcebergTable) -> list[tuple[str, int, int]]:
    """Return ``(path, sequence number, record count)`` of every live position delete file."""
    table.refresh()
    out: list[tuple[str, int, int]] = []
    for manifest in table.current_snapshot().manifests(table.io):
        for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=True):
            if int(entry.data_file.content) == 1:
                out.append(
                    (entry.data_file.file_path, int(entry.sequence_number or 0), int(entry.data_file.record_count))
                )
    return out


def _ids(table: PyIcebergTable) -> list[int]:
    table.refresh()
    return sorted(daft.read_iceberg(table).to_pydict()["id"])


def _seeded_with_deletes(
    local_catalog, name: str, files: int = 4, rows: int = 25, *, properties: dict[str, str] | None = None
) -> tuple[PyIcebergTable, set[int]]:
    """Return a table with two position delete files per data file, and the ids they remove."""
    table = make_seeded_table(local_catalog, name, n_files=files, rows_per_file=rows)
    if properties:
        with table.transaction() as tx:
            tx.set_properties(**properties)
    removed: set[int] = set()
    for index, path in enumerate(sorted(scan_paths(table))):
        for position in (0, 1):
            commit_positional_deletes(table, {path: [position]})
            removed.add(index * rows + position)
    return table, removed


def test_delete_files_are_packed_one_per_data_file(local_catalog):
    table, removed = _seeded_with_deletes(local_catalog, "default.t_pos_pack")
    assert len(_delete_entries(table)) == 8

    result = Table.from_iceberg(table).rewrite_position_delete_files(options=_PACK)

    assert result.rewritten_delete_files == 8
    assert result.added_delete_files == 4, "file granularity: one packed file per data file"
    assert result.commits == 1 and result.failed_groups == 0
    assert sorted(count for _, _, count in _delete_entries(table)) == [2, 2, 2, 2]
    assert _ids(table) == sorted(set(range(100)) - removed)


def test_partition_granularity_packs_into_one_file(local_catalog):
    table, removed = _seeded_with_deletes(
        local_catalog, "default.t_pos_partition_granularity", properties={"write.delete.granularity": "partition"}
    )

    result = Table.from_iceberg(table).rewrite_position_delete_files(options=_PACK)

    assert result.added_delete_files == 1
    [(_, _, count)] = _delete_entries(table)
    assert count == 8
    assert _ids(table) == sorted(set(range(100)) - removed)


def test_the_packed_file_keeps_the_newest_sequence_number(local_catalog):
    table, _ = _seeded_with_deletes(local_catalog, "default.t_pos_seq")
    newest = max(seq for _, seq, _ in _delete_entries(table))

    Table.from_iceberg(table).rewrite_position_delete_files(options=_PACK)

    assert {seq for _, seq, _ in _delete_entries(table)} == {newest}, "a packed delete applies to what the old ones did"


def test_rows_naming_a_vanished_data_file_are_dropped(local_catalog):
    table, removed = _seeded_with_deletes(local_catalog, "default.t_pos_dangling")
    # A data rewrite of every file leaves the deletes pointing at files that are gone.
    Table.from_iceberg(table).rewrite_data_files("binpack", options={**_PACK, "remove-dangling-deletes": False})
    before = _delete_entries(table)
    assert before and all(count == 1 for _, _, count in before)

    result = Table.from_iceberg(table).rewrite_position_delete_files(options=_PACK)

    assert result.rewritten_delete_files == len(before)
    assert result.added_delete_files == 0, "every row named a vanished file"
    assert _delete_entries(table) == []
    assert _ids(table) == sorted(set(range(100)) - removed)


def test_the_packed_rows_are_sorted_by_file_and_position(local_catalog):
    table, _ = _seeded_with_deletes(local_catalog, "default.t_pos_sorted")

    Table.from_iceberg(table).rewrite_position_delete_files(options=_PACK)

    for path, _, _ in _delete_entries(table):
        rows = pq.read_table(path.removeprefix("file://")).to_pylist()
        assert rows == sorted(rows, key=lambda r: (r["file_path"], r["pos"])) and len(rows) == 2
        assert pq.read_schema(path.removeprefix("file://")).field("pos").metadata[b"PARQUET:field_id"] == b"2147483545"


def test_a_partitioned_table_packs_each_partition_on_its_own(local_catalog, simple_schema):
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.transforms import IdentityTransform
    from pyiceberg.typedef import Record

    spec = PartitionSpec(PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="label"))
    table = local_catalog.create_table(
        identifier="default.t_pos_partitioned", schema=simple_schema, partition_spec=spec
    )
    for start in (0, 10, 20):
        table.append(
            pa.table(
                {"id": pa.array(range(start, start + 10), type=pa.int64()), "label": pa.array(["a"] * 5 + ["b"] * 5)}
            )
        )
    for path in sorted(scan_paths(table)):
        label = "a" if "label=a" in path else "b"
        for position in (0, 1):
            commit_positional_deletes(table, {path: [position]}, partition=Record(label))
    assert len(_delete_entries(table)) == 12

    result = Table.from_iceberg(table).rewrite_position_delete_files(options=_PACK)

    assert result.rewritten_delete_files == 12
    assert result.added_delete_files == 6, "one packed file per data file, each within its partition"
    assert len(_ids(table)) == 18


def test_too_few_inputs_are_left_alone(local_catalog):
    table, _ = _seeded_with_deletes(local_catalog, "default.t_pos_few", files=2)

    result = Table.from_iceberg(table).rewrite_position_delete_files(options={"min-input-files": 5})

    assert result.rewritten_delete_files == 0 and result.commits == 0
    assert len(_delete_entries(table)) == 4


def test_replaying_the_same_rewrite_returns_the_recorded_result(local_catalog):
    table, _ = _seeded_with_deletes(local_catalog, "default.t_pos_replay")
    first = Table.from_iceberg(table).rewrite_position_delete_files(options={**_PACK, "rewrite-id": "pack-1"})
    snapshots = len(table.metadata.snapshots)

    again = Table.from_iceberg(table).rewrite_position_delete_files(options={**_PACK, "rewrite-id": "pack-1"})

    assert again == first
    assert len(table.metadata.snapshots) == snapshots


def test_an_unknown_option_is_refused(local_catalog):
    table, _ = _seeded_with_deletes(local_catalog, "default.t_pos_option")

    with pytest.raises(ValueError, match="nonsense-option"):
        Table.from_iceberg(table).rewrite_position_delete_files(options={"nonsense-option": 1})


def test_a_table_without_position_deletes_is_a_no_op(local_catalog):
    table = make_seeded_table(local_catalog, "default.t_pos_none", n_files=2, rows_per_file=10)

    result = Table.from_iceberg(table).rewrite_position_delete_files(options=_PACK)

    assert isinstance(result, RewritePositionDeletesResult)
    assert result.commits == 0 and result.added_delete_files == 0


def test_delete_file_properties_override_the_data_ones(local_catalog):
    table, _ = _seeded_with_deletes(
        local_catalog,
        "default.t_pos_delete_codec",
        properties={"write.parquet.compression-codec": "zstd", "write.delete.parquet.compression-codec": "snappy"},
    )

    Table.from_iceberg(table).rewrite_position_delete_files(options=_PACK)

    for path, _, _ in _delete_entries(table):
        codecs = {
            pq.ParquetFile(path.removeprefix("file://")).metadata.row_group(0).column(c).compression for c in range(2)
        }
        assert codecs == {"SNAPPY"}
