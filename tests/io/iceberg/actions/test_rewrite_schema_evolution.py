"""Rewrites over a table whose schema moved on without a snapshot.

A rename, a drop, or an add changes the schema without producing a snapshot, so
the snapshot a rewrite starts from can name columns the table no longer uses.
Reading under those stale names and writing under the current ones is how a
column gets silently emptied, which is what these tests pin down.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

from pyiceberg.types import StringType

from daft.catalog import Table
from tests.io.iceberg.actions._helpers import read_ids as _read_ids
from tests.io.iceberg.actions._helpers import scan_file_count as _scan_file_count

_COMPACT_ALL = {
    "target-file-size-bytes": 64 * 1024 * 1024,
    "min-input-files": 2,
    "rewrite-all": True,
}


def test_rewrite_after_rename_preserves_the_renamed_column(make_tiny_table):
    table = make_tiny_table(name="default.t_renamed", n_files=6, rows_per_file=4)
    with table.update_schema() as update:
        update.rename_column("label", "tag")
    table.refresh()
    expected = table.scan().to_arrow().sort_by("id").to_pylist()
    assert all(row["tag"] is not None for row in expected)

    Table.from_iceberg(table).rewrite_data_files("binpack", options=_COMPACT_ALL)

    table.refresh()
    # Compaction reorders rows, so compare by identity rather than position.
    assert table.scan().to_arrow().sort_by("id").to_pylist() == expected
    assert _read_ids(table) == list(range(24))
    assert _scan_file_count(table) < 6


def test_rewrite_after_add_column_preserves_both_columns(make_tiny_table):
    table = make_tiny_table(name="default.t_added", n_files=6, rows_per_file=4)
    with table.update_schema() as update:
        update.add_column("note", StringType())
    table.refresh()
    table.append(
        pa.table(
            {
                "id": pa.array([100, 101], type=pa.int64()),
                "label": pa.array(["row-100", "row-101"], type=pa.string()),
                "note": pa.array(["n0", "n1"], type=pa.string()),
            }
        )
    )
    before = table.scan().to_arrow().sort_by("id").to_pylist()

    Table.from_iceberg(table).rewrite_data_files("binpack", options=_COMPACT_ALL)

    table.refresh()
    assert table.scan().to_arrow().sort_by("id").to_pylist() == before


def test_rewrite_after_drop_column_keeps_the_remaining_column(make_tiny_table):
    table = make_tiny_table(name="default.t_dropped", n_files=6, rows_per_file=4)
    with table.update_schema() as update:
        update.delete_column("label")
    table.refresh()

    Table.from_iceberg(table).rewrite_data_files("binpack", options=_COMPACT_ALL)

    table.refresh()
    assert _read_ids(table) == list(range(24))
    assert table.scan().to_arrow().column_names == ["id"]


def test_time_travel_still_reads_the_snapshot_schema(make_tiny_table):
    """The rewrite reads the current schema; a plain read as of a snapshot must not."""
    import daft

    table = make_tiny_table(name="default.t_travel", n_files=3, rows_per_file=4)
    snapshot_id = table.current_snapshot().snapshot_id
    with table.update_schema() as update:
        update.rename_column("label", "tag")
    table.refresh()

    as_of = daft.read_iceberg(table, snapshot_id=snapshot_id)
    assert "label" in as_of.schema().column_names()
    assert "tag" not in as_of.schema().column_names()
    assert sorted(as_of.to_pydict()["label"]) == sorted(f"row-{i}" for i in range(12))


def test_time_travel_filter_on_a_renamed_column_still_prunes(make_tiny_table):
    """Predicates carry the snapshot's names; the scan binds the current ones."""
    import daft

    table = make_tiny_table(name="default.t_travel_filter", n_files=3, rows_per_file=4)
    snapshot_id = table.current_snapshot().snapshot_id
    with table.update_schema() as update:
        update.rename_column("label", "tag")
    table.refresh()

    as_of = daft.read_iceberg(table, snapshot_id=snapshot_id).where(daft.col("label") == "row-5")
    assert as_of.to_pydict()["id"] == [5]


def test_writing_data_that_both_lacks_and_adds_columns_is_refused():
    """A backstop under the rewrite's read path: never pad one column while dropping another.

    Padding and dropping together is what a stale column name looks like from the
    writer's side, and it empties the column rather than failing.
    """
    from pyiceberg.schema import Schema
    from pyiceberg.types import LongType, NestedField

    from daft.io.iceberg.iceberg_write import get_missing_columns

    target = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "tag", StringType(), required=False),
    )
    stale = pa.schema([("id", pa.int64()), ("label", pa.string())])

    with pytest.raises(ValueError, match="do not line up"):
        get_missing_columns(stale, target, require_matching_columns=True)


def test_writing_data_missing_only_columns_still_pads_them():
    """The refusal is narrow: a genuinely absent column is still filled with nulls."""
    from pyiceberg.schema import Schema
    from pyiceberg.types import LongType, NestedField

    from daft.io.iceberg.iceberg_write import get_missing_columns

    target = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "tag", StringType(), required=False),
    )

    added = get_missing_columns(pa.schema([("id", pa.int64())]), target)

    assert [expr.name() for expr in added] == ["tag"]
