from __future__ import annotations

import datetime
import io
from decimal import Decimal

import pytest

import daft
from daft import DataType, col, lit
from daft.functions import power, uuid


def _optimized_projects(frame: daft.DataFrame) -> int:
    """How many Project nodes the optimized logical plan holds."""
    printed = io.StringIO()
    frame.explain(show_all=True, file=printed)
    text = printed.getvalue()
    optimized = text[text.index("== Optimized Logical Plan ==") : text.index("== Physical Plan ==")]
    return optimized.count("* Project")


def _frame() -> daft.DataFrame:
    return daft.from_pydict({"x": [10, 20], "y": [1, 2]})


TWO_INT64 = lit(2).cast(DataType.int64())


@pytest.mark.parametrize(
    "exprs, expected",
    [
        pytest.param(
            [power(col("x"), TWO_INT64).alias("p"), (col("x") + TWO_INT64).alias("q")],
            {"p": [100.0, 400.0], "q": [12, 22]},
            id="power-beside-a-sum",
        ),
        pytest.param(
            [power(col("x"), TWO_INT64).alias("p"), TWO_INT64.alias("q")],
            {"p": [100.0, 400.0], "q": [2, 2]},
            id="power-beside-the-constant",
        ),
        pytest.param(
            [power(col("x"), TWO_INT64).alias("p"), power(col("y"), TWO_INT64).alias("q")],
            {"p": [100.0, 400.0], "q": [1.0, 4.0]},
            id="two-functions-needing-the-constant",
        ),
        pytest.param(
            [
                power(col("x"), lit(2).cast(DataType.int32())).alias("p"),
                (col("x") + lit(2).cast(DataType.int32())).alias("q"),
            ],
            {"p": [100.0, 400.0], "q": [12, 22]},
            id="int32-constant",
        ),
        pytest.param(
            [power(col("x"), lit(2) + lit(0)).alias("p"), (col("y") + (lit(2) + lit(0))).alias("q")],
            {"p": [100.0, 400.0], "q": [3, 4]},
            id="nested-constant",
        ),
    ],
)
def test_a_repeated_constant_stays_a_constant(exprs, expected) -> None:
    assert _frame().select(*exprs).to_pydict() == expected


def test_a_decimal_constant_used_twice() -> None:
    rate = lit(Decimal("1.50")).cast(DataType.decimal128(10, 2))

    answered = _frame().select((col("x") * rate).alias("a"), (col("y") * rate).alias("b")).to_pydict()

    assert answered == {"a": [Decimal("15.00"), Decimal("30.00")], "b": [Decimal("1.50"), Decimal("3.00")]}


def test_a_date_constant_from_text_used_twice() -> None:
    start = lit("2020-01-01").cast(DataType.date())
    frame = daft.from_pydict({"d": [datetime.date(2019, 12, 31), datetime.date(2020, 1, 2)]})

    answered = frame.select((col("d") > start).alias("after"), (col("d") == start).alias("on")).to_pydict()

    assert answered == {"after": [False, True], "on": [False, False]}


def test_a_repeated_constant_adds_no_projection() -> None:
    frame = _frame().select(power(col("x"), TWO_INT64).alias("p"), (col("x") + TWO_INT64).alias("q"))

    assert _optimized_projects(frame) == 1


def test_a_repeated_column_expression_is_still_computed_once() -> None:
    shared = power(col("x") + 1, 2)
    frame = _frame().select(shared.alias("p"), (shared + 1).alias("q"))

    assert frame.to_pydict() == {"p": [121.0, 441.0], "q": [122.0, 442.0]}
    assert _optimized_projects(frame) == 2


def test_a_non_deterministic_expression_is_never_shared() -> None:
    first, second = uuid(), uuid()
    frame = _frame().select(first.alias("a"), second.alias("b"))

    answered = frame.to_pydict()

    assert all(a != b for a, b in zip(answered["a"], answered["b"]))
