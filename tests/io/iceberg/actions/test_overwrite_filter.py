"""An overwrite that replaces only the rows a filter selects.

The rows on which the filter is true are removed and the new rows written, in
one snapshot; a row on which it is false or null is kept. Every new row must
satisfy the filter. The write is reported by file, and a concurrent commit that
overlaps it refuses it without committing anything.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC, PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table.snapshots import Operation
from pyiceberg.transforms import BucketTransform, IdentityTransform
from pyiceberg.types import LongType, NestedField, StringType

import daft
from daft import col, lit
from daft.io.iceberg import WriteConflict
from tests.io.iceberg.actions._helpers import (
    commit_equality_delete_rows,
    commit_equality_deletes,
    commit_positional_deletes,
    snapshot_count,
)

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog
    from pyiceberg.table import Table

SCHEMA = Schema(
    NestedField(1, "id", LongType(), required=False),
    NestedField(2, "region", StringType(), required=False),
    NestedField(3, "v", LongType(), required=False),
)

SPECS = {
    "unpartitioned": UNPARTITIONED_PARTITION_SPEC,
    "identity": PartitionSpec(PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="region")),
    "bucket": PartitionSpec(PartitionField(source_id=1, field_id=1000, transform=BucketTransform(4), name="id_bucket")),
}

Row = tuple[int | None, str | None, int | None]


def _arrow(rows: list[Row]) -> pa.Table:
    return pa.table(
        {
            "id": pa.array([r[0] for r in rows], type=pa.int64()),
            "region": pa.array([r[1] for r in rows], type=pa.string()),
            "v": pa.array([r[2] for r in rows], type=pa.int64()),
        }
    )


def _table(catalog: Catalog, spec: PartitionSpec, files: list[list[Row]], name: str = "default.t") -> Table:
    table = catalog.create_table(name, schema=SCHEMA, partition_spec=spec, properties={"format-version": "2"})
    for rows in files:
        _frame(rows).write_iceberg(table, mode="append")
    return table


def _rows(table: Table) -> list[Row]:
    table.refresh()
    data = daft.read_iceberg(table).to_pydict()
    return sorted(zip(data["id"], data["region"], data["v"]), key=repr)


def _frame(rows: list[Row]) -> daft.DataFrame:
    return daft.from_arrow(_arrow(rows))


SEED = [
    [(1, "eu", 10), (2, "eu", 20)],
    [(3, "us", 30), (4, "us", 40)],
    [(5, "eu", 50), (6, "us", 60)],
]


@pytest.mark.parametrize("spec", list(SPECS.values()), ids=list(SPECS))
@pytest.mark.parametrize(
    ("scope", "new_rows", "kept"),
    [
        pytest.param(
            col("region") == "eu",
            [(7, "eu", 70)],
            [(3, "us", 30), (4, "us", 40), (6, "us", 60)],
            id="a partition's rows",
        ),
        pytest.param(
            col("id") >= 4,
            [(4, "us", 41), (9, "us", 90)],
            [(1, "eu", 10), (2, "eu", 20), (3, "us", 30)],
            id="rows across files",
        ),
        pytest.param(
            col("v") > 1_000,
            [],
            [row for rows in SEED for row in rows],
            id="no rows",
        ),
    ],
)
def test_only_the_rows_the_filter_selects_are_replaced(
    local_catalog: Catalog, spec: PartitionSpec, scope: daft.Expression, new_rows: list[Row], kept: list[Row]
) -> None:
    table = _table(local_catalog, spec, SEED)

    _frame(new_rows).write_iceberg(table, mode="overwrite", overwrite_filter=scope)

    assert _rows(table) == sorted(kept + new_rows, key=repr)


@pytest.mark.parametrize(
    ("scope", "new_rows", "kept"),
    [
        pytest.param(lit(True), [(7, "eu", 70)], [], id="true replaces every row"),
        pytest.param(lit(False), [], [row for rows in SEED for row in rows], id="false replaces none"),
    ],
)
def test_a_constant_filter_reads_no_data_file(
    local_catalog: Catalog,
    monkeypatch: pytest.MonkeyPatch,
    scope: daft.Expression,
    new_rows: list[Row],
    kept: list[Row],
) -> None:
    from daft.io.iceberg import _overwrite

    table = _table(local_catalog, UNPARTITIONED_PARTITION_SPEC, SEED)
    counted: list[str] = []
    real_count = _overwrite._count_rows

    def count_rows(table: Table, plan: _overwrite.ScanPlan, paths: list[str], *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        counted.extend(paths)
        return real_count(table, plan, paths, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_overwrite, "_count_rows", count_rows)

    _frame(new_rows).write_iceberg(table, mode="overwrite", overwrite_filter=scope)

    assert _rows(table) == sorted(kept + new_rows, key=repr)
    assert counted == []


def test_a_row_the_filter_is_null_on_is_kept(local_catalog: Catalog) -> None:
    table = _table(local_catalog, UNPARTITIONED_PARTITION_SPEC, [[(None, "b", 2), (10, "a", 1), (3, "a", 1)]])
    scope = (col("id") > 5) | (col("id").is_null() & (col("v") == 1))

    _frame([]).write_iceberg(table, mode="overwrite", overwrite_filter=scope)

    assert _rows(table) == sorted([(None, "b", 2), (3, "a", 1)], key=repr)


def test_a_filter_the_table_cannot_express_is_still_applied(local_catalog: Catalog) -> None:
    table = _table(local_catalog, UNPARTITIONED_PARTITION_SPEC, SEED)

    _frame([(8, "eu", 80)]).write_iceberg(table, mode="overwrite", overwrite_filter=(col("id") % 2) == 0)

    assert _rows(table) == sorted([(1, "eu", 10), (3, "us", 30), (5, "eu", 50), (8, "eu", 80)], key=repr)


def test_the_write_is_one_overwrite_snapshot(local_catalog: Catalog) -> None:
    table = _table(local_catalog, UNPARTITIONED_PARTITION_SPEC, SEED)
    before = snapshot_count(table)

    _frame([(7, "eu", 70)]).write_iceberg(
        table, mode="overwrite", overwrite_filter=col("id") >= 4, snapshot_properties={"run": "r1"}
    )

    table.refresh()
    summary = table.current_snapshot().summary
    assert snapshot_count(table) == before + 1
    assert summary.operation == Operation.OVERWRITE
    assert summary["run"] == "r1"


def test_the_result_states_each_files_own_facts(local_catalog: Catalog) -> None:
    table = _table(local_catalog, UNPARTITIONED_PARTITION_SPEC, SEED)

    result = (
        _frame([(4, "us", 41), (9, "us", 90)])
        .write_iceberg(table, mode="overwrite", overwrite_filter=col("id") >= 4)
        .to_pydict()
    )

    files = list(zip(result["operation"], result["rows"], result["rows_removed"], result["rewritten"]))
    removed = sorted((rows, rows_removed) for op, rows, rows_removed, _ in files if op == "DELETE")
    added = sorted((rows, rewritten) for op, rows, _, rewritten in files if op == "ADD")
    # Of the second file one row is kept, and the third file is removed whole.
    assert removed == [(2, 1), (2, 2)]
    assert added == [(1, True), (2, False)]
    assert all(rows_removed is None for op, _, rows_removed, _ in files if op == "ADD")
    assert all(rewritten is None for op, _, _, rewritten in files if op == "DELETE")


@pytest.mark.parametrize(
    "new_rows",
    [
        pytest.param([(1, "us", 99)], id="the filter is false"),
        pytest.param([(1, None, 99)], id="the filter is null"),
    ],
)
def test_a_new_row_outside_the_filter_fails_the_write_and_commits_nothing(
    local_catalog: Catalog, new_rows: list[Row]
) -> None:
    table = _table(local_catalog, UNPARTITIONED_PARTITION_SPEC, SEED)
    before = (snapshot_count(table), _rows(table))

    with pytest.raises(ValueError, match="must satisfy its filter"):
        _frame([(7, "eu", 70), *new_rows]).write_iceberg(
            table, mode="overwrite", overwrite_filter=col("region") == "eu"
        )

    assert (snapshot_count(table), _rows(table)) == before


def test_row_level_deletes_stay_applied_and_are_not_counted(local_catalog: Catalog) -> None:
    table = _table(local_catalog, UNPARTITIONED_PARTITION_SPEC, [[(1, "eu", 10), (2, "eu", 20), (3, "eu", 30)]])
    path = next(iter(table.scan().plan_files())).file.file_path
    commit_positional_deletes(table, {path: [0]})
    commit_equality_deletes(table, "id", [3])

    result = _frame([]).write_iceberg(table, mode="overwrite", overwrite_filter=col("id") >= 1).to_pydict()

    assert _rows(table) == []
    assert [n for op, n in zip(result["operation"], result["rows_removed"]) if op == "DELETE"] == [1]


def test_a_row_level_delete_outside_the_filter_still_applies_to_the_kept_rows(local_catalog: Catalog) -> None:
    table = _table(local_catalog, UNPARTITIONED_PARTITION_SPEC, [[(1, "eu", 10), (2, "eu", 20), (4, "eu", 40)]])
    # Its bounds lie outside the filter, so a plan by the filter would leave it out.
    commit_equality_delete_rows(table, pa.table({"id": pa.array([1], type=pa.int64())}), ["id"], bounds=True)

    _frame([]).write_iceberg(table, mode="overwrite", overwrite_filter=col("id") >= 4)

    assert _rows(table) == [(2, "eu", 20)]


def test_running_the_same_overwrite_twice_leaves_the_same_rows(local_catalog: Catalog) -> None:
    table = _table(local_catalog, UNPARTITIONED_PARTITION_SPEC, SEED)

    def run() -> None:
        _frame([(4, "us", 41)]).write_iceberg(table, mode="overwrite", overwrite_filter=col("id") >= 4)

    run()
    once = _rows(table)
    run()

    assert _rows(table) == once


def _inject_before_commit(monkeypatch: pytest.MonkeyPatch, action: Callable[[Table], None], table: Table) -> None:
    """Run ``action`` once, after the overwrite is planned and written and before it commits."""
    from daft.io.iceberg import _overwrite

    real_commit = _overwrite._commit

    def commit_after_action(*args: object, **kwargs: object) -> None:
        action(table)
        monkeypatch.setattr(_overwrite, "_commit", real_commit)
        real_commit(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_overwrite, "_commit", commit_after_action)


def _append(rows: list[Row]) -> Callable[[Table], None]:
    def append(table: Table) -> None:
        table.refresh()
        table.append(_arrow(rows))

    return append


def _remove_us(table: Table) -> None:
    table.refresh()
    table.delete(delete_filter="region = 'us'")


@pytest.mark.parametrize(
    ("isolation", "foreign", "conflicts"),
    [
        pytest.param("serializable", _append([(8, "eu", 80)]), True, id="serializable: rows the filter selects"),
        pytest.param("serializable", _append([(0, "us", 80)]), False, id="serializable: rows outside the filter"),
        pytest.param("snapshot", _append([(8, "eu", 80)]), False, id="snapshot: rows the filter selects"),
        pytest.param("serializable", _remove_us, True, id="serializable: a replaced file removed"),
        pytest.param("snapshot", _remove_us, True, id="snapshot: a replaced file removed"),
    ],
)
def test_a_concurrent_commit_that_overlaps_refuses_the_overwrite(
    local_catalog: Catalog,
    monkeypatch: pytest.MonkeyPatch,
    isolation: str,
    foreign: Callable[[Table], None],
    conflicts: bool,
) -> None:
    spec = SPECS["identity"]
    table = _table(local_catalog, spec, SEED)
    with table.transaction() as tx:
        tx.set_properties({"write.delete.isolation-level": isolation})
    _inject_before_commit(monkeypatch, foreign, table)
    # Every eu row and the us row with id 6 are replaced; the us file holding 3 and 4 is untouched.
    scope = (col("region") == "eu") | (col("id") >= 6)

    def overwrite() -> None:
        _frame([(7, "eu", 70)]).write_iceberg(table, mode="overwrite", overwrite_filter=scope)

    if conflicts:
        with pytest.raises(WriteConflict):
            overwrite()
        assert table.current_snapshot().summary.operation != Operation.OVERWRITE
        assert (7, "eu", 70) not in _rows(table)
    else:
        overwrite()
        assert (7, "eu", 70) in _rows(table)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        pytest.param({"mode": "append"}, ValueError, "requires mode='overwrite'", id="in append mode"),
        pytest.param({"overwrite_filter": col("v") + 1}, ValueError, "must be boolean", id="not boolean"),
        pytest.param({"overwrite_filter": col("missing") == 1}, ValueError, "must refer only to", id="unknown column"),
        pytest.param({"overwrite_filter": "id > 1"}, TypeError, "wrong input type", id="not an expression"),
    ],
)
def test_an_ill_formed_overwrite_is_refused(
    local_catalog: Catalog, kwargs: dict[str, object], error: type[Exception], match: str
) -> None:
    table = _table(local_catalog, UNPARTITIONED_PARTITION_SPEC, SEED)
    options: dict[str, object] = {"mode": "overwrite", "overwrite_filter": col("id") > 1, **kwargs}

    with pytest.raises(error, match=match):
        _frame([(7, "eu", 70)]).write_iceberg(table, **options)  # type: ignore[arg-type]


def test_a_plain_write_reports_its_files_the_same_way(local_catalog: Catalog) -> None:
    table = _table(local_catalog, UNPARTITIONED_PARTITION_SPEC, [[(1, "eu", 10), (2, "eu", 20)]])
    path = next(iter(table.scan().plan_files())).file.file_path
    commit_positional_deletes(table, {path: [0]})

    appended = _frame([(3, "eu", 30)]).write_iceberg(table, mode="append").to_pydict()
    overwritten = _frame([(4, "eu", 40)]).write_iceberg(table, mode="overwrite").to_pydict()

    assert (appended["rewritten"], appended["rows_removed"]) == ([False], [None])
    deleted = sorted(n for op, n in zip(overwritten["operation"], overwritten["rows_removed"]) if op == "DELETE")
    assert deleted == [1, 1]
    assert _rows(table) == [(4, "eu", 40)]
