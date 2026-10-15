"""A filtered read of a file with a page index returns exactly the rows kept.

With the page index, the reader skips the pages of the filter's columns that
their statistics rule out, and fetches of every other column only the pages
holding a kept row. The rows returned are those the filter keeps, in file
order, whether the page index is used or not.
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

Row = dict[str, object]


def _table(order: str) -> pa.Table:
    rows = range(ROWS)
    table = pa.table(
        {
            "k": pa.array([row // 100 if order == "sorted" else (row * 7_919) % 200 for row in rows], pa.int64()),
            "n": pa.array([None if row % 3 == 0 or 4_000 <= row < 6_000 else row for row in rows], pa.int64()),
            "label": pa.array([f"label-{row % 37}" for row in rows]),
            "text": pa.array([f"row {row} " + "x" * (row % 200) for row in rows]),
            "items": pa.array([[{"code": f"c{row % 9}", "value": float(row)}] for row in rows]),
        }
    )
    return table


@pytest.fixture(scope="module", params=["sorted", "scattered"])
def table_and_path(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> tuple[pa.Table, str]:
    table = _table(request.param)
    path = Path(tmp_path_factory.mktemp(request.param)) / "rows.parquet"
    pq.write_table(table, path, row_group_size=5_000, data_page_size=4_096, write_page_index=True)
    return table, str(path)


FILTERS: dict[str, tuple[Expression, Callable[[Row], bool]]] = {
    "one key": (col("k") == 37, lambda row: row["k"] == 37),
    "a range of keys": ((col("k") >= 40) & (col("k") < 43), lambda row: 40 <= row["k"] < 43),
    "a filter on a column with null pages": (
        col("n") > 4_500,
        lambda row: row["n"] is not None and row["n"] > 4_500,
    ),
    "two columns": (
        (col("k") == 12) & (col("label") == "label-5"),
        lambda row: row["k"] == 12 and row["label"] == "label-5",
    ),
    "a dictionary string": (col("label") == "label-3", lambda row: row["label"] == "label-3"),
    "no row": (col("k") > 1_000, lambda row: False),
}


def _kept(table: pa.Table, keeps: Callable[[Row], bool]) -> list[Row]:
    return [row for row in table.to_pylist() if keeps(row)]


@pytest.mark.parametrize("page_index", [True, False], ids=["with the page index", "without it"])
@pytest.mark.parametrize("daft_filter,keeps", FILTERS.values(), ids=FILTERS.keys())
def test_a_filtered_read_returns_exactly_the_rows_kept(
    table_and_path: tuple[pa.Table, str],
    daft_filter: Expression,
    keeps: Callable[[Row], bool],
    page_index: bool,
) -> None:
    table, path = table_and_path

    with daft.execution_config_ctx(parquet_page_index=page_index):
        answer = daft.read_parquet(path).where(daft_filter).to_arrow()

    assert answer.to_pylist() == _kept(table, keeps)


@pytest.mark.parametrize("daft_filter,keeps", FILTERS.values(), ids=FILTERS.keys())
def test_a_filtered_read_without_the_filters_column_returns_the_rows_kept(
    table_and_path: tuple[pa.Table, str],
    daft_filter: Expression,
    keeps: Callable[[Row], bool],
) -> None:
    table, path = table_and_path

    answer = daft.read_parquet(path).where(daft_filter).select("text", "items").to_arrow()

    assert answer.to_pylist() == [{"text": row["text"], "items": row["items"]} for row in _kept(table, keeps)]


@pytest.mark.parametrize("daft_filter,keeps", FILTERS.values(), ids=FILTERS.keys())
def test_a_filtered_read_with_a_limit_returns_the_first_rows_kept(
    table_and_path: tuple[pa.Table, str],
    daft_filter: Expression,
    keeps: Callable[[Row], bool],
) -> None:
    table, path = table_and_path

    answer = daft.read_parquet(path).where(daft_filter).limit(17).to_arrow()

    assert answer.to_pylist() == _kept(table, keeps)[:17]


def test_a_file_without_a_page_index_reads_as_before(tmp_path: Path) -> None:
    table = _table("sorted")
    path = tmp_path / "rows.parquet"
    pq.write_table(table, path, row_group_size=5_000, write_page_index=False)

    answer = daft.read_parquet(str(path)).where(col("k") == 37).to_arrow()

    assert answer.to_pylist() == _kept(table, lambda row: row["k"] == 37)


def test_the_setting_defaults_to_on() -> None:
    assert daft.context.get_context().daft_execution_config.parquet_page_index is True
