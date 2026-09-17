"""What a row-level write does when another writer commits underneath it.

A write reads a snapshot, decides what to change and commits later. These cover
what happens when that gap is not quiet.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

import daft
from daft import col, lit
from daft.catalog.__iceberg import IcebergTable
from daft.io.iceberg import MergeFailedException

from ._helpers import inject_around_row_level, make_seeded_table, read_ids

MODES = ["copy-on-write", "merge-on-read"]


@pytest.fixture(params=MODES)
def mode(request):
    return request.param


def _table(catalog, mode, isolation="serializable", *, name="default.races"):
    table = make_seeded_table(catalog, name, n_files=3, rows_per_file=4)
    with table.transaction() as transaction:
        transaction.set_properties(
            **{
                "write.merge.mode": mode,
                "write.update.mode": mode,
                "write.delete.mode": mode,
                "write.merge.isolation-level": isolation,
                "write.update.isolation-level": isolation,
                "write.delete.isolation-level": isolation,
                "format-version": "2",
            }
        )
    return table.refresh()


def _append(table, start=500, rows=2, label="late"):
    def _run():
        live = table.catalog.load_table(table.name())
        live.append(
            pa.table(
                {
                    "id": pa.array(list(range(start, start + rows)), type=pa.int64()),
                    "label": pa.array([label] * rows, type=pa.string()),
                }
            )
        )

    return _run


def _labels(table):
    """Return the table's rows as ids and labels, in id order."""
    frame = daft.read_iceberg(table.refresh()).sort("id").to_pydict()
    return frame["id"], frame["label"]


def _merge(table, **options):
    source = daft.from_pydict({"id": [1], "label": ["new"]})
    return (
        IcebergTable.from_iceberg(table)
        .merge_into(source, on=col("target.id") == col("source.id"), options=options or None)
        .when_matched()
        .update({"label": col("source.label")})
        .execute()
    )


def test_rows_added_under_a_strict_write_make_it_start_over(local_catalog, mode, monkeypatch):
    table = _table(local_catalog, mode)
    state = inject_around_row_level(monkeypatch, _append(table))

    result = _merge(table)

    assert state["attempts"] == 2, "the merge planned again against what the other writer left"
    assert result.rows_updated == 1
    ids = read_ids(table.refresh())
    assert 500 in ids, "the competing write survives"
    assert len(ids) == 14


def test_a_strict_write_gives_up_when_the_interference_does_not_stop(local_catalog, mode, monkeypatch):
    table = _table(local_catalog, mode)
    state = inject_around_row_level(monkeypatch, _append(table, start=600), every_attempt=True)

    with pytest.raises(MergeFailedException, match="gave up"):
        _merge(table)

    assert state["attempts"] > 1, "it tried more than once before giving up"
    assert dict(zip(read_ids(table.refresh()), read_ids(table.refresh())))


def test_rows_added_under_a_relaxed_write_are_allowed(local_catalog, mode, monkeypatch):
    table = _table(local_catalog, mode)
    state = inject_around_row_level(monkeypatch, _append(table))

    result = _merge(table, **{"isolation-level": "snapshot"})

    assert state["attempts"] == 1, "rows it would not have touched do not make it start over"
    assert result.rows_updated == 1
    ids = read_ids(table.refresh())
    assert 500 in ids, "the competing write survives"
    assert len(ids) == 14


def test_a_change_to_rows_being_rewritten_is_refused(local_catalog, monkeypatch):
    table = _table(local_catalog, "merge-on-read", isolation="snapshot")
    handle = IcebergTable.from_iceberg(table)

    def remove_a_row():
        IcebergTable.from_iceberg(table.catalog.load_table(table.name())).delete_where(col("id") == 2)

    state = inject_around_row_level(monkeypatch, remove_a_row)

    handle.update_where(col("id") < 3, {"label": lit("changed")})

    assert state["attempts"] == 2, "the update planned again once the row it read was removed"
    rows = dict(zip(*_labels(table)))
    assert 2 not in rows, "the competing removal stands"
    assert rows[1] == "changed"


def test_a_removal_tolerates_rows_removed_underneath_it(local_catalog, monkeypatch):
    table = _table(local_catalog, "merge-on-read", isolation="snapshot")
    handle = IcebergTable.from_iceberg(table)

    def remove_a_row():
        IcebergTable.from_iceberg(table.catalog.load_table(table.name())).delete_where(col("id") == 2)

    state = inject_around_row_level(monkeypatch, remove_a_row)

    result = handle.delete_where(col("id") < 3)

    assert state["attempts"] == 1, "removing the same row twice loses nothing"
    assert result.rows_deleted >= 1
    assert sorted(read_ids(table.refresh())) == list(range(3, 12))


def test_a_file_replaced_underneath_a_write_is_refused(local_catalog, mode, monkeypatch):
    table = _table(local_catalog, mode, isolation="snapshot")

    def compact():
        IcebergTable.from_iceberg(table.catalog.load_table(table.name())).compact_files(options={"rewrite-all": True})

    state = inject_around_row_level(monkeypatch, compact)

    result = _merge(table)

    assert state["attempts"] == 2, "the merge planned again against the files the compaction left"
    assert result.rows_updated == 1
    assert sorted(read_ids(table.refresh())) == list(range(12))
