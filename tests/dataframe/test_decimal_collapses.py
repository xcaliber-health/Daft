from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

import pytest

import daft
from daft import DataType, Expression, Window, col
from daft.exceptions import DaftComputeError

# 33 digits before the point at scale 2: a value of decimal(38, 2) that cannot be
# widened to decimal(38, 6), the type a mean or percentile of it answers.
WIDE = "9" * 33 + ".99"
# The largest decimal(38, 0).
MAX_38 = "9" * 38


def _decimals(
    values: list[str | None], precision: int = 38, scale: int = 2, groups: list[int] | None = None
) -> daft.DataFrame:
    data: dict[str, list[str | None] | list[int]] = {"v": values}
    if groups is not None:
        data["g"] = groups
    return daft.from_pydict(data).with_column("v", col("v").cast(DataType.decimal128(precision, scale)))


# --- a collapse answers exactly where its answer fits ---


@pytest.mark.parametrize(
    ("collapse", "values", "expected"),
    [
        pytest.param(lambda v: v.sum(), ["1.10", "2.20", None], Decimal("3.30"), id="sum"),
        # 5.00 / 3 = 1.666666.., rounded half away from zero at scale 6.
        pytest.param(lambda v: v.mean(), ["1.00", "2.00", "2.00"], Decimal("1.666667"), id="mean rounds up"),
        pytest.param(lambda v: v.mean(), ["-1.00", "-2.00", "-2.00"], Decimal("-1.666667"), id="mean rounds away"),
        # The median of x and -x is 0: it fits, although widening either value would not.
        pytest.param(lambda v: v.median(), [WIDE, "-" + WIDE], Decimal("0.000000"), id="median of wide values"),
        pytest.param(lambda v: v.percentile(0.5), ["6.21", "6.35"], Decimal("6.280000"), id="percentile"),
    ],
)
def test_a_decimal_collapse_answers_exactly_where_it_fits(
    collapse: Callable[[Expression], Expression], values: list[str | None], expected: Decimal
) -> None:
    answered = _decimals(values).agg(collapse(col("v")).alias("a")).to_pydict()["a"]

    assert answered == [expected]


def test_a_decimal_sum_whose_running_total_passes_128_bits_answers_exactly() -> None:
    df = _decimals([MAX_38, MAX_38, "-" + MAX_38], precision=38, scale=0)

    assert df.agg(col("v").sum().alias("s")).to_pydict()["s"] == [Decimal(MAX_38)]


# --- a collapse refuses where its answer does not fit, never wrapping or answering null ---


@pytest.mark.parametrize(
    ("collapse", "values", "scale", "named"),
    [
        pytest.param(lambda v: v.sum(), [MAX_38, MAX_38], 0, "sum", id="sum wraps no more"),
        pytest.param(lambda v: v.mean(), [WIDE, WIDE], 2, "mean", id="mean"),
        pytest.param(lambda v: v.median(), [WIDE, WIDE], 2, "percentile", id="median"),
        pytest.param(lambda v: v.percentile(0.5), [WIDE, WIDE], 2, "percentile", id="percentile"),
    ],
)
def test_a_decimal_collapse_that_does_not_fit_is_refused(
    collapse: Callable[[Expression], Expression], values: list[str], scale: int, named: str
) -> None:
    df = _decimals(values, scale=scale)

    with pytest.raises(DaftComputeError, match=f"The {named} of these decimals does not fit"):
        df.agg(collapse(col("v")).alias("a")).collect()


def test_a_grouped_collapse_refuses_only_where_a_group_does_not_fit() -> None:
    df = _decimals([MAX_38, MAX_38, "1", "2"], scale=0, groups=[1, 1, 2, 2])

    with pytest.raises(DaftComputeError, match="The sum of these decimals does not fit"):
        df.groupby("g").agg(col("v").sum().alias("s")).collect()

    fitting = df.where(col("g") == 2).groupby("g").agg(col("v").sum().alias("s")).to_pydict()
    assert fitting == {"g": [2], "s": [Decimal("3")]}


def test_a_group_without_values_still_answers_null() -> None:
    df = _decimals([None, None, "1.00"], groups=[1, 1, 2])

    answered = df.groupby("g").agg(col("v").mean().alias("m")).sort("g").to_pydict()

    assert answered == {"g": [1, 2], "m": [None, Decimal("1.000000")]}


# --- a product of decimals answers a float ---


def test_a_decimal_product_answers_a_float() -> None:
    df = _decimals(["2.00", "3.00"])

    answered = df.agg(col("v").product().alias("p"))

    assert answered.schema()["p"].dtype == DataType.float64()
    assert answered.to_pydict() == {"p": [6.0]}


# --- window frames follow the same rules ---

_RUNNING = Window().partition_by("g").order_by("o").rows_between(Window.unbounded_preceding, Window.current_row)


def test_a_decimal_running_sum_that_does_not_fit_is_refused() -> None:
    df = _decimals([MAX_38, MAX_38], scale=0, groups=[1, 1]).with_column("o", daft.lit(1))

    with pytest.raises(DaftComputeError, match="The sum of these decimals does not fit"):
        df.select(col("v").sum().over(_RUNNING).alias("s")).collect()


def test_a_decimal_running_mean_rounds_half_away_from_zero() -> None:
    df = daft.from_pydict({"g": [1, 1, 1], "o": [1, 2, 3], "v": ["1.00", "2.00", "2.00"]}).with_column(
        "v", col("v").cast(DataType.decimal128(10, 2))
    )

    answered = df.select("o", col("v").mean().over(_RUNNING).alias("m")).sort("o").to_pydict()["m"]

    assert answered == [Decimal("1.000000"), Decimal("1.500000"), Decimal("1.666667")]
