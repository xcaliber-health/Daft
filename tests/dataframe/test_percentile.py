from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

import pytest

import daft
from daft import DataType, Expression, col


@pytest.mark.parametrize(
    ("percentage", "expected"),
    [
        (0.0, 1.0),
        (0.25, 1.75),
        (0.5, 2.5),
        (0.75, 3.25),
        (1.0, 4.0),
    ],
)
def test_percentile_global(percentage, expected):
    df = daft.from_pydict({"values": [1.0, 2.0, 3.0, 4.0]})

    actual = df.agg(col("values").percentile(percentage).alias("percentile")).collect().to_pydict()
    assert actual == {"percentile": [expected]}


@pytest.mark.parametrize(
    ("percentage", "expected"),
    [
        (0.25, {"id": [1, 2], "percentile": [1.5, 4.5]}),
        (0.5, {"id": [1, 2], "percentile": [2.0, 5.0]}),
        (0.75, {"id": [1, 2], "percentile": [6.0, 5.5]}),
    ],
)
def test_percentile_groupby(percentage, expected):
    df = daft.from_pydict(
        {
            "id": [1, 1, 1, 2, 2],
            "values": [1.0, 2.0, 10.0, 4.0, 6.0],
        }
    )

    actual = (
        df.groupby("id").agg(col("values").percentile(percentage).alias("percentile")).sort("id").collect().to_pydict()
    )
    assert actual == expected


def test_percentile_integer_input_casts_to_float64():
    df = daft.from_pydict({"values": [1, 2, 3, 4]})

    actual = df.agg(col("values").percentile(0.25).alias("p25")).collect().to_pydict()
    assert actual == {"p25": [1.75]}


def test_percentile_50_is_median():
    df = daft.from_pydict({"values": [1.0, None, 2.0, 10.0]})

    actual = (
        df.agg(
            [
                col("values").percentile(0.5).alias("p50"),
                col("values").median().alias("median"),
            ]
        )
        .collect()
        .to_pydict()
    )
    assert actual == {"p50": [2.0], "median": [2.0]}


def test_percentile_all_nulls():
    df = daft.from_pydict({"values": [None, None]})

    actual = (
        df.agg(
            [
                col("values").cast(DataType.float64()).percentile(0.5).alias("p50"),
                col("values").cast(DataType.float64()).median().alias("median"),
            ]
        )
        .collect()
        .to_pydict()
    )
    assert actual == {"p50": [None], "median": [None]}


def test_percentile_empty_input():
    df = daft.from_pydict({"values": [1.0, 2.0]}).limit(0)

    actual = (
        df.agg(
            [
                col("values").percentile(0.5).alias("p50"),
                col("values").median().alias("median"),
            ]
        )
        .collect()
        .to_pydict()
    )
    assert actual == {"p50": [None], "median": [None]}


def test_percentile_sql_invalid_percentage_raises():
    df = daft.from_pydict({"values": [1.0, 2.0, 3.0]})

    with pytest.raises(Exception) as excinfo:
        daft.sql("SELECT percentile(values, 1.5) FROM df", df=df).collect()

    assert "between 0.0 and 1.0" in str(excinfo.value)


def test_percentile_sql_integer_percentage_raises():
    df = daft.from_pydict({"values": [1.0, 2.0, 3.0]})

    with pytest.raises(Exception) as excinfo:
        daft.sql("SELECT percentile(values, 1) FROM df", df=df).collect()

    assert "float literal" in str(excinfo.value)


_DECIMALS = daft.from_pydict({"g": [1, 1, 2, 2], "v": ["1.25", "2.50", "3.00", "5.00"]}).with_column(
    "v", col("v").cast(DataType.decimal128(18, 2))
)


def _decimals(values: list[str], precision: int = 18, scale: int = 2) -> daft.DataFrame:
    return daft.from_pydict({"v": values}).with_column("v", col("v").cast(DataType.decimal128(precision, scale)))


@pytest.mark.parametrize("partitions", [1, 3])
@pytest.mark.parametrize(
    "aggregate",
    [pytest.param(lambda v: v.percentile(0.5), id="percentile"), pytest.param(lambda v: v.median(), id="median")],
)
def test_a_percentile_of_a_decimal_is_an_exact_decimal(
    partitions: int, aggregate: Callable[[Expression], Expression]
) -> None:
    df = _DECIMALS.where(col("g") == 1).into_partitions(partitions)

    answered = df.agg(aggregate(col("v")).alias("p50"))

    # Typed as the decimal's mean is: decimal(38, s + 4).
    assert answered.schema()["p50"].dtype == DataType.decimal128(38, 6)
    assert answered.to_pydict() == {"p50": [Decimal("1.875000")]}


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        # As floats these interpolate to 6.404999999999999 and 6.279999999999999.
        pytest.param(["6.14", "6.67"], Decimal("6.405000"), id="no float drift below"),
        pytest.param(["6.21", "6.35"], Decimal("6.280000"), id="a threshold is met exactly"),
        # -1.00 + 0.5 x (0.01 - -1.00) = -0.495 exactly, well within scale 6.
        pytest.param(["-1.00", "0.01"], Decimal("-0.495000"), id="negative values"),
    ],
)
def test_a_decimal_median_is_exact(values: list[str], expected: Decimal) -> None:
    answered = _decimals(values).agg(col("v").median().alias("m")).to_pydict()["m"]

    assert answered == [expected]


def test_a_decimal_percentile_reads_the_percentage_as_written() -> None:
    # 0.9 x 10 is exactly rank 9; read as its binary value it would interpolate past 9.00.
    df = _decimals([f"{i}.00" for i in range(11)])

    answered = df.agg(col("v").percentile(0.9).alias("p90")).to_pydict()["p90"]

    assert answered == [Decimal("9.000000")]


def test_a_decimal_percentile_cuts_toward_zero_at_the_widened_scale() -> None:
    # 1/3 of the way from 0.0000 to 0.0001 is 0.0000333..., cut to 0.00003333 at scale 4 + 4.
    df = _decimals(["0.0000", "0.0001", "0.0001", "0.0001"], precision=10, scale=4)

    answered = df.agg(col("v").percentile(1 / 9).alias("p")).to_pydict()["p"]

    assert answered == [Decimal("0.00003333")]


@pytest.mark.parametrize("partitions", [1, 3])
def test_percentile_of_a_decimal_by_group(partitions: int) -> None:
    df = _DECIMALS.into_partitions(partitions)

    actual = df.groupby("g").agg(col("v").percentile(0.5).alias("p50")).sort("g").to_pydict()

    assert actual == {"g": [1, 2], "p50": [Decimal("1.875000"), Decimal("4.000000")]}


def test_percentile_of_decimal_lists_is_an_exact_decimal() -> None:
    df = daft.from_pydict({"v": [["1.25", "2.50"], ["3.00"]]}).with_column(
        "v", col("v").cast(DataType.list(DataType.decimal128(18, 2)))
    )

    answered = df.agg(col("v").percentile(0.5).alias("p50")).to_pydict()["p50"]

    assert answered == [Decimal("2.500000")]


def test_percentile_of_a_float_is_still_a_float() -> None:
    df = daft.from_pydict({"v": [1.25, 2.5]})

    answered = df.agg(col("v").percentile(0.5).alias("p50"))

    assert answered.schema()["p50"].dtype == DataType.float64()
    assert answered.to_pydict() == {"p50": [1.875]}


@pytest.mark.parametrize("partitions", [1, 3])
def test_approx_percentile_of_a_decimal_is_near_the_exact_one(partitions: int) -> None:
    df = _DECIMALS.repartition(partitions)

    actual = df.agg(col("v").approx_percentiles(0.5).alias("p50")).to_pydict()["p50"][0]

    assert actual == pytest.approx(2.5, rel=0.02)
