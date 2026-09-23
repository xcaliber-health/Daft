from __future__ import annotations

import pytest

import daft
from daft import DataType, col


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


@pytest.mark.parametrize("partitions", [1, 3])
def test_percentile_of_a_decimal_is_a_float(partitions: int) -> None:
    df = _DECIMALS.where(col("g") == 1).repartition(partitions)

    actual = df.agg(col("v").percentile(0.5).alias("p50")).to_pydict()

    assert actual == {"p50": [1.875]}
    assert df.agg(col("v").percentile(0.5)).schema()["v"].dtype == DataType.float64()


@pytest.mark.parametrize("partitions", [1, 3])
def test_percentile_of_a_decimal_by_group(partitions: int) -> None:
    df = _DECIMALS.repartition(partitions)

    actual = df.groupby("g").agg(col("v").percentile(0.5).alias("p50")).sort("g").to_pydict()

    assert actual == {"g": [1, 2], "p50": [1.875, 4.0]}


@pytest.mark.parametrize("partitions", [1, 3])
def test_approx_percentile_of_a_decimal_is_near_the_exact_one(partitions: int) -> None:
    df = _DECIMALS.repartition(partitions)

    actual = df.agg(col("v").approx_percentiles(0.5).alias("p50")).to_pydict()["p50"][0]

    assert actual == pytest.approx(2.5, rel=0.02)
