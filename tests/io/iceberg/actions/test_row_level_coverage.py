"""Row-level writes over table states the plain cases do not produce.

Two of those states matter in production and are easy to get wrong: a table
whose rows are already masked by deletes written by a streaming pipeline, and a
line of work kept on a branch until someone decides to publish it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyiceberg")

import daft
from daft import col, lit
from daft.catalog.__iceberg import IcebergTable

from ._helpers import commit_equality_deletes, commit_positional_deletes, make_seeded_table, read_ids

MODES = ["copy-on-write", "merge-on-read"]


@pytest.fixture(params=MODES)
def mode(request):
    return request.param


def _table(catalog, mode, name):
    table = make_seeded_table(catalog, name, n_files=3, rows_per_file=10)
    with table.transaction() as transaction:
        transaction.set_properties(
            **{
                "format-version": "2",
                "write.merge.mode": mode,
                "write.update.mode": mode,
                "write.delete.mode": mode,
            }
        )
    return table.refresh()


def _rows(table, branch=None):
    """Return the table's rows, read the way the engine reads them.

    The catalog library refuses a table carrying equality deletes, so the rows
    are read through the engine, which applies every kind of delete.
    """
    frame = daft.read_iceberg(table.refresh()) if branch is None else daft.read_iceberg(table.refresh(), branch=branch)
    data = frame.to_pydict()
    return dict(zip(data["id"], data["label"]))


def test_a_merge_ignores_rows_an_equality_delete_already_removed(local_catalog, mode) -> None:
    """Rows masked by a delete on their key are neither matched nor carried through."""
    table = _table(local_catalog, mode, f"default.eq_{mode.replace('-', '_')}")
    commit_equality_deletes(table, "id", [1, 2, 3])
    table = table.refresh()
    source = daft.from_pydict({"id": [2, 4], "label": ["merged", "merged"]})

    result = (
        IcebergTable.from_iceberg(table)
        .merge_into(source, on=col("target.id") == col("source.id"))
        .when_matched()
        .update({"label": col("source.label")})
        .when_not_matched()
        .insert_all()
        .execute()
    )

    rows = _rows(table)
    assert 1 not in rows and 3 not in rows, "rows the delete removed stay removed"
    assert rows[4] == "merged", "a row the delete left alone is changed"
    assert rows[2] == "merged", "the removed key comes back only as the row the source brought"
    assert result.rows_updated + result.rows_inserted == 2


def test_a_merge_ignores_rows_a_position_delete_already_removed(local_catalog, mode) -> None:
    """Rows masked by recorded positions are not seen by the merge either."""
    import pyarrow.parquet as pq

    from ._helpers import strip_scheme

    table = _table(local_catalog, mode, f"default.pos_{mode.replace('-', '_')}")
    first_file = sorted(task.file.file_path for task in table.scan().plan_files())[0]
    # The rows removed are the first two of that file, whichever ids they hold.
    removed = set(pq.read_table(strip_scheme(first_file)).column("id").to_pylist()[:2])
    commit_positional_deletes(table, {first_file: [0, 1]})
    table = table.refresh()
    assert removed & set(_rows(table)) == set(), "the fixture removed the rows it meant to"

    returning = sorted(removed)[0]
    live = max(set(_rows(table)))
    source = daft.from_pydict({"id": [returning, live], "label": ["merged", "merged"]})
    (
        IcebergTable.from_iceberg(table)
        .merge_into(source, on=col("target.id") == col("source.id"))
        .when_matched()
        .update({"label": col("source.label")})
        .when_not_matched()
        .insert_all()
        .execute()
    )

    rows = _rows(table)
    assert rows[live] == "merged", "a row the delete left alone is changed"
    assert rows[returning] == "merged", "the removed row comes back only as the row the source brought"
    assert set(removed) - {returning} & set(rows) == set(), "the other removed row stays removed"


def test_changes_on_a_branch_leave_the_main_line_alone(local_catalog, mode) -> None:
    """A write aimed at a branch is visible there and nowhere else."""
    table = _table(local_catalog, mode, f"default.branch_{mode.replace('-', '_')}")
    main_snapshot = table.current_snapshot().snapshot_id
    table.manage_snapshots().create_branch(snapshot_id=main_snapshot, branch_name="audit").commit()
    table = table.refresh()
    source = daft.from_pydict({"id": [1], "label": ["on-branch"]})

    result = (
        IcebergTable.from_iceberg(table)
        .merge_into(source, on=col("target.id") == col("source.id"), branch="audit")
        .when_matched()
        .update({"label": col("source.label")})
        .execute()
    )

    table = table.refresh()
    branch_rows = _rows(table, branch="audit")
    assert result.rows_updated == 1
    assert branch_rows[1] == "on-branch"
    assert _rows(table)[1] == "seed", "the main line is untouched"
    assert table.current_snapshot().snapshot_id == main_snapshot


def test_a_removal_on_a_branch_leaves_the_main_line_alone(local_catalog, mode) -> None:
    """The same holds for removing rows by condition."""
    table = _table(local_catalog, mode, f"default.branch_delete_{mode.replace('-', '_')}")
    main_snapshot = table.current_snapshot().snapshot_id
    table.manage_snapshots().create_branch(snapshot_id=main_snapshot, branch_name="audit").commit()
    table = table.refresh()

    IcebergTable.from_iceberg(table).delete_where(col("id") < 5, branch="audit")

    table = table.refresh()
    branch_ids = sorted(_rows(table, branch="audit"))
    assert min(branch_ids) >= 5, "the branch lost the rows the condition covers"
    assert min(read_ids(table)) == 0, "the main line still holds them"


def test_a_change_by_condition_on_a_branch_is_recorded_there(local_catalog, mode) -> None:
    """A conditional change follows the branch it was aimed at."""
    table = _table(local_catalog, mode, f"default.branch_update_{mode.replace('-', '_')}")
    main_snapshot = table.current_snapshot().snapshot_id
    table.manage_snapshots().create_branch(snapshot_id=main_snapshot, branch_name="audit").commit()
    table = table.refresh()

    result = IcebergTable.from_iceberg(table).update_where(col("id") < 3, {"label": lit("changed")}, branch="audit")

    table = table.refresh()
    branch_rows = _rows(table, branch="audit")
    assert result.rows_updated == 3
    assert branch_rows[0] == "changed"
    assert _rows(table)[0] == "seed"
