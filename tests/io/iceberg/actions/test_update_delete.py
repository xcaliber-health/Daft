"""Changing and removing rows by condition."""

from __future__ import annotations

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

import daft
from daft import col, lit
from daft.catalog.__iceberg import IcebergTable
from daft.io.iceberg import RowLevelFailedException

from ._helpers import make_seeded_table, read_ids, scan_paths

MODES = ["copy-on-write", "merge-on-read"]


@pytest.fixture(params=MODES)
def mode(request):
    return request.param


def _table(catalog, mode, *, name="default.rows", n_files=3, rows_per_file=4):
    table = make_seeded_table(catalog, name, n_files=n_files, rows_per_file=rows_per_file)
    with table.transaction() as transaction:
        transaction.set_properties(**{"write.update.mode": mode, "write.delete.mode": mode, "format-version": "2"})
    return table.refresh()


def _rows(table):
    frame = daft.read_iceberg(table.refresh()).sort("id").to_pydict()
    return list(zip(frame["id"], frame["label"]))


def test_update_changes_only_the_rows_the_condition_selects(local_catalog, mode):
    table = _table(local_catalog, mode)

    result = IcebergTable.from_iceberg(table).update_where(col("id") < 3, {"label": lit("changed")})

    rows = dict(_rows(table))
    assert result.rows_updated == 3
    assert [rows[i] for i in range(3)] == ["changed"] * 3
    assert rows[3] == "seed"
    assert len(rows) == 12


def test_delete_removes_only_the_rows_the_condition_selects(local_catalog, mode):
    table = _table(local_catalog, mode)

    result = IcebergTable.from_iceberg(table).delete_where(col("id") % 2 == 0)

    assert result.rows_deleted == 6
    assert sorted(read_ids(table.refresh())) == [1, 3, 5, 7, 9, 11]


def test_a_removal_covering_whole_files_reads_nothing(local_catalog, mode):
    table = _table(local_catalog, mode, n_files=3, rows_per_file=4)
    before = set(scan_paths(table))

    result = IcebergTable.from_iceberg(table).delete_where(col("id") < 4)

    assert result.files_dropped == 1, "the first file holds exactly the removed rows"
    assert result.added_data_files == 0 and result.added_delete_files == 0
    assert result.operation == "delete"
    after = set(scan_paths(table.refresh()))
    assert len(before - after) == 1
    assert sorted(read_ids(table.refresh())) == list(range(4, 12))


def test_a_removal_covering_no_row_commits_nothing(local_catalog, mode):
    table = _table(local_catalog, mode)
    before = table.current_snapshot().snapshot_id

    result = IcebergTable.from_iceberg(table).delete_where(col("id") > 1000)

    assert result.operation == "none"
    assert table.refresh().current_snapshot().snapshot_id == before


def test_update_without_assignments_is_refused(local_catalog, mode):
    table = _table(local_catalog, mode)

    with pytest.raises(RowLevelFailedException, match="at least one column"):
        IcebergTable.from_iceberg(table).update_where(col("id") < 3, {})


def test_upsert_replaces_matched_rows_and_adds_the_rest(local_catalog, mode):
    table = _table(local_catalog, mode)
    source = daft.from_pydict({"id": [1, 99], "label": ["new", "new"]})

    result = IcebergTable.from_iceberg(table).upsert(source, keys=["id"])

    rows = dict(_rows(table))
    assert (result.rows_updated, result.rows_inserted) == (1, 1)
    assert rows[1] == "new" and rows[99] == "new" and rows[2] == "seed"


def test_upsert_without_keys_needs_the_table_to_declare_them(local_catalog, mode):
    table = _table(local_catalog, mode)
    source = daft.from_pydict({"id": [1], "label": ["new"]})

    with pytest.raises(ValueError, match="declares which columns identify a row"):
        IcebergTable.from_iceberg(table).upsert(source)


def test_removed_rows_stay_removed_after_a_later_update(local_catalog, mode):
    table = _table(local_catalog, mode)
    handle = IcebergTable.from_iceberg(table)

    handle.delete_where(col("id") == 5)
    handle.update_where(col("id") < 3, {"label": lit("changed")})

    ids = sorted(read_ids(table.refresh()))
    assert 5 not in ids, "an earlier removal is not undone by a later change"
    assert len(ids) == 11


def test_rows_added_after_a_removal_are_kept(local_catalog, mode):
    table = _table(local_catalog, mode)
    handle = IcebergTable.from_iceberg(table)

    handle.delete_where(col("id") == 0)
    table.refresh().append(pa.table({"id": pa.array([0], type=pa.int64()), "label": pa.array(["again"])}))

    rows = dict(_rows(table))
    assert rows[0] == "again", "a removal does not apply to rows written after it"
