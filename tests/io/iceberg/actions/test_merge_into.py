"""Merging rows of a source into a table.

Covers both ways a table records a change: rewriting the files that hold the old
rows, and recording the positions of the rows that left.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

import daft
from daft import col
from daft.catalog.__iceberg import IcebergTable
from daft.io.iceberg import MergeCardinalityError

from ._helpers import make_seeded_table, read_ids, scan_paths

MODES = ["copy-on-write", "merge-on-read"]


@pytest.fixture(params=MODES)
def mode(request):
    return request.param


def _table(catalog, mode, *, name="default.merge", n_files=3, rows_per_file=4):
    table = make_seeded_table(catalog, name, n_files=n_files, rows_per_file=rows_per_file)
    with table.transaction() as transaction:
        transaction.set_properties(**{"write.merge.mode": mode, "format-version": "2"})
    return table.refresh()


def _labelled(table, start, count, label):
    """Append rows the merge can recognise by label."""
    table.append(
        pa.table(
            {
                "id": pa.array(list(range(start, start + count)), type=pa.int64()),
                "label": pa.array([label] * count, type=pa.string()),
            }
        )
    )
    return table.refresh()


def _rows(table):
    frame = daft.read_iceberg(table.refresh()).sort("id").to_pydict()
    return list(zip(frame["id"], frame["label"]))


def _handle(table):
    return IcebergTable.from_iceberg(table)


def test_updates_matched_rows_and_inserts_the_rest(local_catalog, mode):
    table = _table(local_catalog, mode)
    source = daft.from_pydict({"id": [1, 5, 99], "label": ["new"] * 3})

    result = (
        _handle(table)
        .merge_into(source, on=col("target.id") == col("source.id"))
        .when_matched()
        .update({"label": col("source.label")})
        .when_not_matched()
        .insert_all()
        .execute()
    )

    assert (result.rows_updated, result.rows_inserted, result.rows_deleted) == (2, 1, 0)
    rows = dict(_rows(table))
    assert rows[1] == "new" and rows[5] == "new" and rows[99] == "new"
    assert rows[0] == "seed", "rows the source did not match keep their values"
    assert len(rows) == 13


def test_deletes_matched_rows(local_catalog, mode):
    table = _table(local_catalog, mode)
    source = daft.from_pydict({"id": [2, 3]})

    result = (
        _handle(table).merge_into(source, on=col("target.id") == col("source.id")).when_matched().delete().execute()
    )

    assert result.rows_deleted == 2
    assert sorted(read_ids(table.refresh())) == [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]


def test_first_rule_whose_condition_holds_decides_the_row(local_catalog, mode):
    table = _table(local_catalog, mode)
    source = daft.from_pydict({"id": [1, 2], "op": ["delete", "update"], "label": ["x", "kept"]})

    (
        _handle(table)
        .merge_into(source, on=col("target.id") == col("source.id"))
        .when_matched(col("source.op") == "delete")
        .delete()
        .when_matched()
        .update({"label": col("source.label")})
        .execute()
    )

    rows = dict(_rows(table))
    assert 1 not in rows, "the first rule claimed the row"
    assert rows[2] == "kept"


def test_rules_for_rows_the_source_did_not_match(local_catalog, mode):
    table = _table(local_catalog, mode)
    source = daft.from_pydict({"id": [0, 1]})

    result = (
        _handle(table)
        .merge_into(source, on=col("target.id") == col("source.id"))
        .when_not_matched_by_source()
        .update({"label": daft.lit("stale")})
        .execute()
    )

    rows = dict(_rows(table))
    assert result.rows_updated == 10
    assert rows[0] == "seed" and rows[1] == "seed", "matched rows are untouched without a matched rule"
    assert rows[11] == "stale"


def test_a_row_matched_twice_is_refused(local_catalog, mode):
    table = _table(local_catalog, mode)
    source = daft.from_pydict({"id": [1, 1], "label": ["a", "b"]})

    with pytest.raises(MergeCardinalityError) as raised:
        (
            _handle(table)
            .merge_into(source, on=col("target.id") == col("source.id"))
            .when_matched()
            .update({"label": col("source.label")})
            .execute()
        )

    assert "more than one source row" in str(raised.value)
    assert sorted(read_ids(table.refresh())) == list(range(12)), "the table is unchanged"


def test_duplicates_are_allowed_when_no_rule_depends_on_which_row_wins(local_catalog, mode):
    table = _table(local_catalog, mode)
    source = daft.from_pydict({"id": [1, 1]})

    result = (
        _handle(table).merge_into(source, on=col("target.id") == col("source.id")).when_matched().delete().execute()
    )

    assert result.rows_deleted >= 1
    assert 1 not in read_ids(table.refresh())


def test_a_merge_that_changes_nothing_commits_nothing(local_catalog, mode):
    table = _table(local_catalog, mode)
    before = table.current_snapshot().snapshot_id
    source = daft.from_pydict({"id": [], "label": []}).select(
        col("id").cast(daft.DataType.int64()), col("label").cast(daft.DataType.string())
    )

    result = (
        _handle(table)
        .merge_into(source, on=col("target.id") == col("source.id"))
        .when_matched()
        .update({"label": col("source.label")})
        .execute()
    )

    assert result.operation == "none"
    assert table.refresh().current_snapshot().snapshot_id == before


def test_only_the_files_holding_matched_rows_are_rewritten(local_catalog):
    table = _table(local_catalog, "copy-on-write", n_files=4, rows_per_file=3)
    before = set(scan_paths(table))
    source = daft.from_pydict({"id": [0], "label": ["new"]})

    (
        _handle(table)
        .merge_into(source, on=col("target.id") == col("source.id"))
        .when_matched()
        .update({"label": col("source.label")})
        .execute()
    )

    after = set(scan_paths(table.refresh()))
    assert len(before - after) == 1, "one file held the matched row"
    assert len(before & after) == 3, "the other files are untouched"


def test_rows_are_removed_by_position_under_merge_on_read(local_catalog):
    from pyiceberg.manifest import DataFileContent

    table = _table(local_catalog, "merge-on-read")
    before = set(scan_paths(table))
    source = daft.from_pydict({"id": [1, 5], "label": ["new", "new"]})

    result = (
        _handle(table)
        .merge_into(source, on=col("target.id") == col("source.id"))
        .when_matched()
        .update({"label": col("source.label")})
        .execute()
    )

    assert result.added_delete_files > 0
    assert before <= set(scan_paths(table.refresh())), "the files holding the old rows stay"
    deletes = [
        entry.data_file
        for snapshot in [table.refresh().current_snapshot()]
        for manifest in snapshot.manifests(table.io)
        for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=True)
        if entry.data_file.content == DataFileContent.POSITION_DELETES
    ]
    assert deletes, "the removed rows are named by a delete file"
    assert dict(_rows(table))[1] == "new"
    assert len(_rows(table)) == 12, "a replaced row appears once"


def test_repeating_a_merge_under_the_same_name_changes_nothing_twice(local_catalog, mode):
    table = _table(local_catalog, mode)
    source = daft.from_pydict({"id": [1], "label": ["new"]})

    def run():
        return (
            _handle(table.refresh())
            .merge_into(source, on=col("target.id") == col("source.id"), options={"merge-id": "fixed"})
            .when_matched()
            .update({"label": col("source.label")})
            .execute()
        )

    first = run()
    second = run()

    assert first.snapshot_id == second.snapshot_id
    assert second.rows_updated == first.rows_updated
    assert len(_rows(table)) == 12


def test_a_condition_beyond_key_equality_decides_which_rows_pair(local_catalog, mode):
    table = _table(local_catalog, mode)
    _labelled(table, 100, 2, "recent")
    source = daft.from_pydict({"id": [1, 100], "label": ["new", "new"], "floor": [5, 5]})

    # Only the row whose id clears the floor is paired; the other is unmatched.
    result = (
        _handle(table)
        .merge_into(
            source,
            on=(col("target.id") == col("source.id")) & (col("target.id") > col("source.floor")),
        )
        .when_matched()
        .update({"label": col("source.label")})
        .when_not_matched()
        .insert({"id": col("source.id"), "label": col("source.label")})
        .execute()
    )

    rows = _rows(table)
    assert result.rows_updated == 1
    assert (100, "new") in rows, "the pair that satisfies the whole condition is updated"
    assert (1, "seed") in rows, "the pair that fails it leaves the table's row alone"
    assert (1, "new") in rows, "and the source row it left unmatched is added instead"
    assert result.rows_inserted == 1
