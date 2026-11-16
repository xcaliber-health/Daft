from __future__ import annotations

import io
import re

import pytest

import daft
from daft import DataType, col, lit
from daft.functions import coalesce, when

from . import test_approx_count_distinct


@pytest.mark.parametrize("data_and_expected", test_approx_count_distinct.TESTS)
@pytest.mark.parametrize("partition_size", [None, 2, 3])
def test_count_distinct(data_and_expected, partition_size):
    data, expected = data_and_expected
    df = test_approx_count_distinct.make_df(data)
    if partition_size:
        df = df.into_partitions(partition_size).collect()
    df = df.agg(col("a").count_distinct()).collect()
    test_approx_count_distinct.assert_equal(df, expected)


ROWS = {
    "x": [None, 1, None, 2, 2],
    "k": ["a", "b", "a", None, "c"],
    "b": [True, False, True, None, False],
    "g": [1] * 5,
}


def _expected(values: list) -> int:
    return len({v for v in values if v is not None})


def _evaluated(expr, frame: daft.DataFrame) -> list:
    return frame.select(expr.alias("v")).to_pydict()["v"]


# Expressions the engine's scheduler counts as free over plain columns; an ungrouped
# count_distinct of each used to plan a read of a column that did not exist.
SHAPES = [
    pytest.param(lambda: when(col("x").is_null(), col("k")).otherwise(None), id="conditional"),
    pytest.param(lambda: col("x").cast(DataType.string()), id="cast"),
    pytest.param(lambda: ~col("b"), id="not"),
    pytest.param(lambda: col("x").is_null(), id="is-null"),
    pytest.param(lambda: col("x").fill_null(0), id="fill-null"),
    pytest.param(lambda: coalesce(col("x"), lit(0)), id="coalesce"),
    pytest.param(lambda: lit(1), id="literal"),
    pytest.param(lambda: (~col("b")).alias("b"), id="not-named-like-its-column"),
    pytest.param(lambda: col("x").cast(DataType.string()).alias("x"), id="cast-named-like-its-column"),
    pytest.param(lambda: col("k"), id="plain-column"),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("grouped", [False, True], ids=["ungrouped", "grouped"])
def test_count_distinct_of_any_expression(shape, grouped) -> None:
    frame = daft.from_pydict(ROWS)
    expected = _expected(_evaluated(shape(), frame))

    aggregated = frame.groupby("g") if grouped else frame
    answered = aggregated.agg(shape().count_distinct().alias("n")).to_pydict()["n"]

    assert answered == [expected]


@pytest.mark.parametrize(
    "values, expected",
    [pytest.param([None, 1, 1, 2], 2, id="partly-null"), pytest.param([None, None], 0, id="all-null")],
)
def test_count_distinct_counts_only_non_null_values(values, expected) -> None:
    frame = daft.from_pydict({"x": values}).select(col("x").cast(DataType.int64()))

    answered = frame.agg(col("x").cast(DataType.string()).count_distinct().alias("n")).to_pydict()["n"]

    assert answered == [expected]


def test_two_count_distincts_beside_a_sum() -> None:
    frame = daft.from_pydict(ROWS)

    answered = frame.agg(
        col("x").is_null().count_distinct().alias("a"),
        col("x").cast(DataType.string()).count_distinct().alias("b"),
        col("x").sum().alias("s"),
    ).to_pydict()

    assert answered == {"a": [2], "b": [2], "s": [5]}


def test_count_distinct_of_a_plain_column_adds_no_projection() -> None:
    frame = daft.from_pydict(ROWS).agg(col("k").count_distinct().alias("n"))
    printed = io.StringIO()

    frame.explain(show_all=True, file=printed)

    text = printed.getvalue()
    optimized = text[text.index("== Optimized Logical Plan ==") : text.index("== Physical Plan ==")]
    projections = [line for line in optimized.splitlines() if line.startswith("* Project:")]
    # Only column pruning: every projection selects bare columns and computes nothing.
    assert all(re.fullmatch(r"\* Project: col\(\w+\)(, col\(\w+\))*", line) for line in projections)
