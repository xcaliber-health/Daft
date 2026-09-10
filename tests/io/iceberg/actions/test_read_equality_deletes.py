"""Reading a table that carries equality deletes.

An equality delete names column values rather than row positions and applies
to every strictly older data file in its partition, or in every partition when
written under an unpartitioned spec. A read applies each delete to exactly the
files it covers, alongside any position deletes, so the rows returned are the
rows a reference reader returns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

import daft
from daft.catalog import Table
from tests.io.iceberg.actions._helpers import (
    commit_equality_delete_rows,
    commit_equality_deletes,
    commit_equality_upsert,
    commit_positional_deletes,
    make_seeded_table,
    scan_paths,
)

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog
    from pyiceberg.table import Table as PyIcebergTable


def _rows(table: PyIcebergTable, **read_options: int) -> dict[int, str | None]:
    """Return ``id -> label`` as the reader returns them."""
    table.refresh()
    result = daft.read_iceberg(table, **read_options).to_pydict()
    id_column = "id" if "id" in result else next(name for name in result if name != "label")
    ids = result[id_column]
    assert len(ids) == len(set(ids)), "a row was duplicated"
    return dict(zip(ids, result["label"]))


def test_rows_an_equality_delete_removes_are_not_read(local_catalog):
    table = make_seeded_table(local_catalog, "default.r_eq_apply", n_files=4, rows_per_file=25)
    commit_equality_deletes(table, "id", [1, 2, 3])

    assert set(_rows(table)) == set(range(100)) - {1, 2, 3}


def test_rows_committed_beside_the_delete_keep_their_new_values(local_catalog):
    # Arrange: an upsert writes the new rows and the delete of their keys in one
    # commit, at one sequence number, so the delete covers only older files.
    table = make_seeded_table(local_catalog, "default.r_eq_upsert", n_files=4, rows_per_file=25)
    commit_equality_upsert(
        table,
        pa.table({"id": pa.array([5, 6], type=pa.int64()), "label": pa.array(["new-5", "new-6"])}),
        ["id"],
    )

    rows = _rows(table)

    assert set(rows) == set(range(100))
    assert rows[5] == "new-5" and rows[6] == "new-6"
    assert rows[7] == "seed"


def test_a_second_upsert_of_the_same_key_wins(local_catalog):
    table = make_seeded_table(local_catalog, "default.r_eq_twice", n_files=2, rows_per_file=10)
    commit_equality_upsert(table, pa.table({"id": pa.array([5], type=pa.int64()), "label": pa.array(["v2"])}), ["id"])
    commit_equality_upsert(table, pa.table({"id": pa.array([5], type=pa.int64()), "label": pa.array(["v3"])}), ["id"])

    rows = _rows(table)

    assert rows[5] == "v3"
    assert len(rows) == 20


def test_a_null_delete_value_matches_null_rows(local_catalog):
    table = make_seeded_table(local_catalog, "default.r_eq_null", n_files=2, rows_per_file=10)
    table.append(pa.table({"id": pa.array([100, 101], type=pa.int64()), "label": pa.array([None, "kept"])}))
    commit_equality_delete_rows(table, pa.table({"label": pa.array([None], type=pa.string())}), ["label"])

    rows = _rows(table)

    assert 100 not in rows
    assert rows[101] == "kept"


def test_a_delete_with_a_null_key_and_a_value_key_matches_both_together(local_catalog):
    table = make_seeded_table(local_catalog, "default.r_eq_mixed_null", n_files=1, rows_per_file=10)
    table.append(pa.table({"id": pa.array([200], type=pa.int64()), "label": pa.array([None], type=pa.string())}))
    commit_equality_delete_rows(
        table,
        pa.table({"id": pa.array([1, 200], type=pa.int64()), "label": pa.array([None, None], type=pa.string())}),
        ["id", "label"],
    )

    rows = _rows(table)

    assert 200 not in rows
    assert rows[1] == "seed"


def _regional_table(catalog: Catalog, name: str) -> PyIcebergTable:
    """Return a table partitioned by region after an unpartitioned start, with rows under both specs."""
    from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC
    from pyiceberg.schema import Schema
    from pyiceberg.types import LongType, NestedField, StringType

    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "label", StringType(), required=False),
        NestedField(3, "region", StringType(), required=False),
    )
    table = catalog.create_table(identifier=name, schema=schema, partition_spec=UNPARTITIONED_PARTITION_SPEC)
    table.append(_regional_rows(0, 10))
    with table.update_spec() as update:
        update.add_identity("region")
    table.refresh()
    table.append(_regional_rows(10, 20))
    return table


def _regional_rows(start: int, end: int) -> pa.Table:
    return pa.table(
        {
            "id": pa.array(list(range(start, end)), type=pa.int64()),
            "label": pa.array(["seed"] * (end - start)),
            "region": pa.array(["a" if i % 2 == 0 else "b" for i in range(start, end)]),
        }
    )


def test_a_delete_under_the_unpartitioned_spec_covers_every_partition(local_catalog):
    table = _regional_table(local_catalog, "default.r_eq_global")
    commit_equality_delete_rows(table, pa.table({"id": pa.array([2, 3, 12, 13], type=pa.int64())}), ["id"], spec_id=0)

    assert set(_rows(table)) == set(range(20)) - {2, 3, 12, 13}


def test_a_partition_scoped_delete_leaves_other_partitions_alone(local_catalog):
    from pyiceberg.typedef import Record

    table = _regional_table(local_catalog, "default.r_eq_partition")
    # Region "a" holds the even ids; a delete over region "a" naming an odd id
    # from region "b" must not reach it.
    commit_equality_delete_rows(
        table,
        pa.table({"id": pa.array([12, 13], type=pa.int64())}),
        ["id"],
        partition=Record("a"),
        spec_id=table.spec().spec_id,
    )

    rows = _rows(table)

    assert 12 not in rows
    assert 13 in rows, "the delete is scoped to region a"
    assert 2 in rows, "the unpartitioned file is not in the delete's partition"


def test_deletes_scoped_to_two_partitions_each_stay_in_their_own(local_catalog):
    from pyiceberg.typedef import Record

    # Arrange: region a deletes 12 (even, in a); region b deletes 12 as well,
    # which names no row of b, and 13 (odd, in b).
    table = _regional_table(local_catalog, "default.r_eq_two_partitions")
    spec_id = table.spec().spec_id
    commit_equality_delete_rows(
        table, pa.table({"id": pa.array([12], type=pa.int64())}), ["id"], partition=Record("a"), spec_id=spec_id
    )
    commit_equality_delete_rows(
        table, pa.table({"id": pa.array([12, 13], type=pa.int64())}), ["id"], partition=Record("b"), spec_id=spec_id
    )

    rows = _rows(table)

    assert 12 not in rows and 13 not in rows
    assert set(rows) == set(range(20)) - {12, 13}


def test_a_renamed_column_still_matches_by_field_id(local_catalog):
    table = make_seeded_table(local_catalog, "default.r_eq_rename", n_files=2, rows_per_file=10)
    commit_equality_deletes(table, "id", [4])
    with table.update_schema() as update:
        update.rename_column("id", "ident")

    assert 4 not in _rows(table)


def test_position_and_equality_deletes_apply_together(local_catalog):
    table = make_seeded_table(local_catalog, "default.r_eq_both", n_files=2, rows_per_file=10)
    first = sorted(scan_paths(table))[0]
    commit_positional_deletes(table, {first: [0, 1]})
    commit_equality_deletes(table, "id", [15])

    rows = _rows(table)

    assert 15 not in rows
    assert len(rows) == 17


def test_an_older_snapshot_still_shows_the_rows(local_catalog):
    table = make_seeded_table(local_catalog, "default.r_eq_snapshot", n_files=2, rows_per_file=10)
    before = table.current_snapshot().snapshot_id
    commit_equality_deletes(table, "id", [4])

    assert 4 in _rows(table, snapshot_id=before)
    assert 4 not in _rows(table)


def test_a_filter_still_prunes_after_the_deletes(local_catalog):
    table = make_seeded_table(local_catalog, "default.r_eq_filter", n_files=4, rows_per_file=25)
    commit_equality_deletes(table, "id", [60, 61])

    ids = daft.read_iceberg(table).where(daft.col("id") >= 50).to_pydict()["id"]

    assert sorted(ids) == sorted(set(range(50, 100)) - {60, 61})


def test_the_catalog_handle_reads_the_same_rows(local_catalog):
    table = make_seeded_table(local_catalog, "default.r_eq_handle", n_files=2, rows_per_file=10)
    commit_equality_deletes(table, "id", [3])

    ids = Table.from_iceberg(table).read().to_pydict()["id"]

    assert sorted(ids) == sorted(set(range(20)) - {3})


def test_a_table_without_equality_deletes_keeps_the_plain_scan(local_catalog, capsys):
    table = make_seeded_table(local_catalog, "default.r_eq_none", n_files=2, rows_per_file=10)

    df = daft.read_iceberg(table)
    df.explain(show_all=True)

    plan = capsys.readouterr().out
    assert "IcebergScanOperator" in plan and "IcebergFileGroupScanOperator" not in plan
    assert df.count_rows() == 20
