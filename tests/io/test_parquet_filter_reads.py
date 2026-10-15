"""A filter pushed into a Parquet read returns exactly the rows it keeps.

Each row group evaluates the filter over its own columns, then decodes the
others either through a selection of the kept rows or whole and masked,
whichever its kept rows make cheaper. Whatever the choice, the rows returned
are the rows the filter keeps, in file order.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import daft
from daft import Expression, col

ROWS = 20_000
ROW_GROUP_ROWS = 1_000


def _table() -> pa.Table:
    keys = [row % 200 for row in range(ROWS)]
    return pa.table(
        {
            "k": pa.array(keys, pa.int64()),
            "n": pa.array([None if row % 7 == 0 else row % 50 for row in range(ROWS)], pa.int64()),
            "v": pa.array(range(ROWS), pa.int64()),
            "text": pa.array([f"row {row} " + "x" * (200 + row % 300) for row in range(ROWS)]),
            "items": pa.array([[{"code": f"c{row % 9}", "value": float(row)}] for row in range(ROWS)]),
        }
    )


@pytest.fixture(scope="module", params=["clustered", "scattered"])
def table_and_path(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> tuple[pa.Table, str]:
    table = _table()
    if request.param == "clustered":
        table = table.sort_by("k")
    path = Path(tmp_path_factory.mktemp(request.param)) / "rows.parquet"
    pq.write_table(table, path, row_group_size=ROW_GROUP_ROWS)
    return table, str(path)


Row = dict[str, object]

FILTERS: dict[str, tuple[Expression, Callable[[Row], bool]]] = {
    "a few keys": (col("k").is_in([3, 4, 5]), lambda row: row["k"] in (3, 4, 5)),
    "one row in a hundred": ((col("v") % 100) == 0, lambda row: row["v"] % 100 == 0),
    "every other row": ((col("v") % 2) == 0, lambda row: row["v"] % 2 == 0),
    "a predicate that is null for some rows": (
        col("n") > 40,
        lambda row: row["n"] is not None and row["n"] > 40,
    ),
    "no row": (col("k") > 1_000, lambda row: False),
    "every row": (col("v") >= 0, lambda row: True),
}


def _kept(table: pa.Table, keeps: Callable[[Row], bool], columns: list[str] | None = None) -> list[Row]:
    rows = [row for row in table.to_pylist() if keeps(row)]
    if columns is None:
        return rows
    return [{column: row[column] for column in columns} for row in rows]


@pytest.mark.parametrize(
    "columns",
    [None, ["v", "text", "items"], ["k", "v"]],
    ids=["every column", "without the filter's column", "narrow columns, decoded whole when kept rows are dense"],
)
@pytest.mark.parametrize("daft_filter,keeps", FILTERS.values(), ids=FILTERS.keys())
def test_a_filtered_read_returns_exactly_the_rows_kept(
    table_and_path: tuple[pa.Table, str],
    daft_filter: Expression,
    keeps: Callable[[Row], bool],
    columns: list[str] | None,
) -> None:
    table, path = table_and_path

    frame = daft.read_parquet(path).where(daft_filter)
    if columns is not None:
        frame = frame.select(*columns)

    assert frame.to_arrow().to_pylist() == _kept(table, keeps, columns)


@pytest.mark.parametrize("daft_filter,keeps", FILTERS.values(), ids=FILTERS.keys())
def test_a_filtered_read_with_a_limit_returns_the_first_rows_kept(
    table_and_path: tuple[pa.Table, str],
    daft_filter: Expression,
    keeps: Callable[[Row], bool],
) -> None:
    table, path = table_and_path

    answer = daft.read_parquet(path).where(daft_filter).limit(25).to_arrow()

    assert answer.to_pylist() == _kept(table, keeps)[:25]
