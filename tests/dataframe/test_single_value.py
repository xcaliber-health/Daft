from __future__ import annotations

import datetime
import decimal
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import daft
from daft import DataType, Window, col
from daft.exceptions import DaftCardinalityError
from daft.functions import single_value


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        pytest.param([1, None, 3], DataType.int64(), id="int64"),
        pytest.param([1.5, None, 3.5], DataType.float64(), id="float64"),
        pytest.param(["a", None, "c"], DataType.string(), id="string"),
        pytest.param([True, None, False], DataType.bool(), id="bool"),
        pytest.param([datetime.date(2026, 1, 1), None, datetime.date(2026, 3, 1)], DataType.date(), id="date"),
        pytest.param(
            [decimal.Decimal("1.25"), None, decimal.Decimal("3.75")], DataType.decimal128(10, 2), id="decimal"
        ),
        pytest.param([[1, 2], None, [3]], DataType.list(DataType.int64()), id="list"),
        pytest.param([{"x": 1}, None, {"x": 3}], DataType.struct({"x": DataType.int64()}), id="struct"),
    ],
)
def test_each_group_answers_its_only_row(values: list[object], dtype: DataType) -> None:
    df = daft.from_pydict({"k": [1, 2, 3], "v": values}).with_column("v", col("v").cast(dtype))

    answer = df.groupby("k").agg(single_value(col("v"))).sort("k")

    assert answer.schema()["v"].dtype == dtype
    assert answer.to_pydict()["v"] == df.sort("k").to_pydict()["v"]


@pytest.mark.parametrize(
    "offending",
    [
        pytest.param(["a", "b"], id="distinct values"),
        pytest.param(["a", "a"], id="equal values"),
        pytest.param([None, None], id="two nulls"),
    ],
)
def test_a_group_of_two_rows_fails_the_query(offending: list[str | None]) -> None:
    df = daft.from_pydict({"k": [1, 2, 2], "v": ["x", *offending]})

    with pytest.raises(DaftCardinalityError, match="single_value"):
        df.groupby("k").agg(single_value(col("v"))).collect()


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        pytest.param([7], [7], id="one row"),
        pytest.param([], [None], id="no rows"),
    ],
)
def test_a_whole_frame_answers_its_only_row(values: list[int], expected: list[int | None]) -> None:
    df = daft.from_pydict({"v": values}).with_column("v", col("v").cast(DataType.int64()))

    assert df.agg(single_value(col("v"))).to_pydict()["v"] == expected


def test_a_whole_frame_of_two_rows_fails_the_query() -> None:
    df = daft.from_pydict({"v": [7, 7]})

    with pytest.raises(DaftCardinalityError):
        df.agg(single_value(col("v"))).collect()


def _write_files(directory: pathlib.Path, rows_per_file: list[dict[str, list[object]]]) -> str:
    # Each file is read as its own unit, so its rows reach their own partial aggregation.
    for i, rows in enumerate(rows_per_file):
        pq.write_table(pa.table(rows), directory / f"part-{i}.parquet")
    return str(directory / "*.parquet")


def test_a_group_whose_rows_are_in_different_files_fails_the_query(tmp_path: pathlib.Path) -> None:
    # Alone, each file holds a single row for the key.
    path = _write_files(tmp_path, [{"k": [1], "v": [10]}, {"k": [1], "v": [20]}])

    with pytest.raises(DaftCardinalityError):
        daft.read_parquet(path).groupby("k").agg(single_value(col("v"))).collect()


def test_groups_spread_over_files_answer_their_rows(tmp_path: pathlib.Path) -> None:
    path = _write_files(
        tmp_path,
        [{"k": [1, 2], "v": ["a", "b"]}, {"k": [3], "v": ["c"]}, {"k": [4], "v": ["d"]}],
    )

    answer = daft.read_parquet(path).groupby("k").agg(single_value(col("v"))).sort("k").to_pydict()

    assert answer == {"k": [1, 2, 3, 4], "v": ["a", "b", "c", "d"]}


def test_a_whole_frame_answers_its_row_when_the_other_files_hold_none(tmp_path: pathlib.Path) -> None:
    # The filter leaves most files without rows, and their partial aggregations answer
    # nothing; only the file holding the row may supply the value.
    path = _write_files(tmp_path, [{"k": [i] * 100, "v": list(range(i * 100, i * 100 + 100))} for i in range(4)])
    df = daft.read_parquet(path).where((col("k") == 3) & (col("v") == 350))

    answers = [df.agg(single_value(col("v"))).to_pydict()["v"] for _ in range(5)]

    assert answers == [[350]] * 5


def test_a_whole_frame_whose_rows_are_in_different_files_fails_the_query(tmp_path: pathlib.Path) -> None:
    path = _write_files(tmp_path, [{"v": [10]}, {"v": [20]}])

    with pytest.raises(DaftCardinalityError):
        daft.read_parquet(path).agg(single_value(col("v"))).collect()


@pytest.mark.parametrize(
    "v",
    [pytest.param([10, 20, 30], id="numeric"), pytest.param(["a", "b", "c"], id="string")],
)
def test_it_combines_with_other_aggregations(v: list[object]) -> None:
    df = daft.from_pydict({"k": [1, 2, 3], "v": v})

    answer = df.groupby("k").agg(single_value(col("v")).alias("one"), col("v").count().alias("n")).sort("k").to_pydict()

    assert answer == {"k": [1, 2, 3], "one": v, "n": [1, 1, 1]}


def test_a_filter_that_reads_the_value_still_fails_on_a_group_of_two_rows() -> None:
    df = daft.from_pydict({"k": [1, 2, 2], "v": ["a", "b", "c"]})

    with pytest.raises(DaftCardinalityError):
        df.groupby("k").agg(single_value(col("v"))).where(col("v") == "a").collect()


def test_a_partition_window_answers_its_only_row() -> None:
    df = daft.from_pydict({"k": [1, 2], "v": ["a", "b"]})

    answer = df.select("k", single_value(col("v")).over(Window().partition_by("k")).alias("w")).sort("k")

    assert answer.to_pydict() == {"k": [1, 2], "w": ["a", "b"]}


def test_a_partition_window_of_two_rows_fails_the_query() -> None:
    df = daft.from_pydict({"k": [1, 1], "v": ["a", "b"]})

    with pytest.raises(DaftCardinalityError):
        df.select(single_value(col("v")).over(Window().partition_by("k")).alias("w")).collect()


def test_the_frame_methods_take_each_groups_only_row() -> None:
    df = daft.from_pydict({"k": [1, 2], "v": ["a", "b"]})

    assert df.groupby("k").single_value("v").sort("k").to_pydict() == {"k": [1, 2], "v": ["a", "b"]}
    assert df.where(col("k") == 2).single_value("v").to_pydict() == {"v": ["b"]}
    assert df.groupby("k").agg(col("v").single_value()).sort("k").to_pydict()["v"] == ["a", "b"]
