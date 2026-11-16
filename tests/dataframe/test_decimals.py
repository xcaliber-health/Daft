from __future__ import annotations

import decimal
import itertools
import random
import tempfile
from collections.abc import Iterator

import pyarrow as pa
import pytest

import daft


def test_decimal_parquet_roundtrip() -> None:
    python_decimals = [decimal.Decimal("-2.010"), decimal.Decimal("0.000"), decimal.Decimal("2.010")]
    data = {
        "decimal128": pa.array(python_decimals),
        # Not supported yet.
        # "decimal256": pa.array([decimal.Decimal('1234567890.1234567890123456789012345678901234567890')]),
    }

    df = daft.from_pydict(data)

    with tempfile.TemporaryDirectory() as dirname:
        df.write_parquet(dirname)
        df_readback = daft.read_parquet(dirname).collect()

        assert str(df.to_pydict()["decimal128"]) == str(df_readback.to_pydict()["decimal128"])


def test_arrow_decimal() -> None:
    # Test roundtrip of Arrow decimals.
    pa_table = pa.Table.from_pydict(
        {"decimal128": pa.array([decimal.Decimal("-1.010"), decimal.Decimal("0.000"), decimal.Decimal("1.010")])}
    )

    df = daft.from_arrow(pa_table)

    assert df.to_arrow() == pa_table


def test_python_decimal() -> None:
    # Test roundtrip of Python decimals.
    python_decimals = [decimal.Decimal("-1.010"), decimal.Decimal("0.000"), decimal.Decimal("1.010")]
    df = daft.from_pydict({"decimal128": python_decimals})

    res = df.to_pydict()["decimal128"]
    assert str(res) == str(python_decimals)


@pytest.mark.parametrize("prec, partitions", itertools.product([5, 30], [1, 2]))
def test_decimal_sum(prec, partitions) -> None:
    python_decimals = [decimal.Decimal("-1.010"), decimal.Decimal("99.001"), decimal.Decimal("10.010")]
    df = daft.from_pydict({"decimal128": python_decimals}).repartition(partitions)
    df = df.with_column("decimal128", df["decimal128"].cast(daft.DataType.decimal128(prec, 3)))
    res = df.sum().collect()
    assert res.to_pydict()["decimal128"] == [decimal.Decimal("108.001")]

    schema = res.schema()
    assert schema["decimal128"].dtype == daft.DataType.decimal128(38, 3)


@pytest.mark.parametrize("prec, partitions", itertools.product([5, 30], [1, 2]))
def test_decimal_mean(prec, partitions) -> None:
    python_decimals = [decimal.Decimal("-1.010"), decimal.Decimal("99.001"), decimal.Decimal("10.010")]
    df = daft.from_pydict({"decimal128": python_decimals}).repartition(partitions)
    df = df.with_column("decimal128", df["decimal128"].cast(daft.DataType.decimal128(prec, 3)))
    res = df.mean().collect()
    assert res.to_pydict()["decimal128"] == [decimal.Decimal("36.0003333")]

    schema = res.schema()
    assert schema["decimal128"].dtype == daft.DataType.decimal128(38, 7)


@pytest.mark.parametrize("prec, partitions", itertools.product([5, 30], [1, 2]))
def test_decimal_stddev(prec, partitions) -> None:
    python_decimals = [decimal.Decimal("-1.010"), decimal.Decimal("99.001"), decimal.Decimal("10.010")]
    df = daft.from_pydict({"decimal128": python_decimals}).repartition(partitions)
    df = df.with_column("decimal128", df["decimal128"].cast(daft.DataType.decimal128(prec, 3)))
    res = df.agg(df["decimal128"].stddev())  ## TODO: we can't do `select(df['x'].stddev())` make ticket
    assert pytest.approx(res.to_pydict()["decimal128"][0]) == 54.83769780300165

    schema = res.schema()
    assert schema["decimal128"].dtype == daft.DataType.float64()


@pytest.mark.parametrize("prec, partitions", itertools.product([5, 30], [1, 2]))
def test_decimal_grouped_sum(prec, partitions) -> None:
    python_decimals = [decimal.Decimal("-1.010"), decimal.Decimal("99.001"), decimal.Decimal("10.010"), None]
    group = [0, 1, 0, 1]

    df = daft.from_pydict({"decimal128": python_decimals, "group": group}).repartition(partitions)
    df = df.with_column("decimal128", df["decimal128"].cast(daft.DataType.decimal128(prec, 3)))
    res = df.groupby("group").sum().sort("group").collect()
    assert res.to_pydict() == {"group": [0, 1], "decimal128": [decimal.Decimal("9.000"), decimal.Decimal("99.001")]}
    schema = res.schema()
    assert schema["decimal128"].dtype == daft.DataType.decimal128(38, 3)


@pytest.mark.parametrize("prec, partitions", itertools.product([5, 30], [1, 2]))
def test_decimal_grouped_mean(prec, partitions) -> None:
    python_decimals = [decimal.Decimal("-1.010"), decimal.Decimal("99.001"), decimal.Decimal("10.010"), None]
    group = [0, 1, 0, 1]

    df = daft.from_pydict({"decimal128": python_decimals, "group": group}).repartition(partitions)
    df = df.with_column("decimal128", df["decimal128"].cast(daft.DataType.decimal128(prec, 3)))
    res = df.groupby("group").mean().sort("group").collect()
    assert res.to_pydict() == {"group": [0, 1], "decimal128": [decimal.Decimal("4.500"), decimal.Decimal("99.001")]}
    schema = res.schema()
    assert schema["decimal128"].dtype == daft.DataType.decimal128(38, 7)


@pytest.mark.parametrize("prec, partitions", itertools.product([5, 30], [1, 2]))
def test_decimal_grouped_stddev(prec, partitions) -> None:
    python_decimals = [decimal.Decimal("-1.010"), decimal.Decimal("99.001"), decimal.Decimal("10.010"), None]
    group = [0, 1, 0, 1]

    df = daft.from_pydict({"decimal128": python_decimals, "group": group}).repartition(partitions)
    df = df.with_column("decimal128", df["decimal128"].cast(daft.DataType.decimal128(prec, 3)))
    res = df.groupby("group").stddev().sort("group").collect()
    assert res.to_pydict() == {"group": [0, 1], "decimal128": [pytest.approx(7.792316), None]}
    schema = res.schema()
    assert schema["decimal128"].dtype == daft.DataType.float64()


DecimalType = tuple[int, int]


def _frame(
    left: list[str | int | None],
    left_type: DecimalType | pa.DataType,
    right: list[str | int | None],
    right_type: DecimalType | pa.DataType,
) -> daft.DataFrame:
    def column(values: list[str | int | None], dtype: DecimalType | pa.DataType) -> pa.Array:
        if isinstance(dtype, pa.DataType):
            return pa.array(values, dtype)
        return pa.array([None if v is None else decimal.Decimal(v) for v in values], pa.decimal128(*dtype))

    return daft.from_arrow(pa.table({"a": column(left, left_type), "b": column(right, right_type)}))


def _exact(value: decimal.Decimal, dtype: daft.DataType) -> decimal.Decimal | None:
    """The exact answer at a decimal type: rounded half away from zero, or None when it does not fit."""
    arrow_type = dtype.to_arrow_dtype()
    rounded = value.quantize(decimal.Decimal(1).scaleb(-arrow_type.scale), rounding=decimal.ROUND_HALF_UP)
    return rounded if len(rounded.as_tuple().digits) <= arrow_type.precision or rounded.is_zero() else None


@pytest.fixture
def wide_context() -> Iterator[None]:
    """Python decimals precise enough to hold any product or quotient exactly before rounding."""
    with decimal.localcontext() as context:
        context.prec = 100
        yield


@pytest.mark.parametrize(
    "a, a_type, b, b_type, expected",
    [
        pytest.param("6.00", (18, 2), "2.00", (18, 2), decimal.Decimal(3), id="six-by-two"),
        pytest.param("1.50", (18, 2), "0.25", (18, 2), decimal.Decimal(6), id="one-fifty-by-quarter"),
        pytest.param("1.500000", (18, 6), "0.25", (18, 2), decimal.Decimal(6), id="mixed-scales"),
        pytest.param("9.20", (18, 2), "9.20", (18, 2), decimal.Decimal(1), id="by-itself"),
        pytest.param("10", (10, 0), "4", (10, 0), decimal.Decimal("2.5"), id="narrow"),
    ],
)
def test_decimal_division_answers_the_exact_quotient(a, a_type, b, b_type, expected) -> None:
    frame = _frame([a], a_type, [b], b_type)

    (quotient,) = frame.select((daft.col("a") / daft.col("b")).alias("q")).to_pydict()["q"]

    assert quotient == expected


def test_decimal_multiplication_answers_the_exact_product() -> None:
    frame = _frame(["9000000000000000.00", "-1.50"], (18, 2), ["9000000000000000.00", "2.00"], (18, 2))

    products = frame.select((daft.col("a") * daft.col("b")).alias("p")).to_pydict()["p"]

    assert products == [decimal.Decimal(81000000000000000000000000000000), decimal.Decimal(-3)]


@pytest.mark.parametrize(
    "a, b, expected",
    [
        pytest.param("2", "3", decimal.Decimal("0.6666666666667"), id="rounds-up"),
        pytest.param("-2", "3", decimal.Decimal("-0.6666666666667"), id="rounds-negative-away"),
        pytest.param("1", "3", decimal.Decimal("0.3333333333333"), id="rounds-down"),
    ],
)
def test_decimal_division_rounds_half_away_from_zero(a, b, expected) -> None:
    frame = _frame([a], (10, 2), [b], (10, 2))

    (quotient,) = frame.select((daft.col("a") / daft.col("b")).alias("q")).to_pydict()["q"]

    assert quotient == expected


@pytest.mark.parametrize(
    "a_type, b_type, operator, expected",
    [
        pytest.param((18, 2), (18, 2), "/", daft.DataType.decimal128(37, 21), id="divide"),
        pytest.param((18, 6), (18, 2), "/", daft.DataType.decimal128(37, 25), id="divide-mixed"),
        pytest.param((18, 2), (18, 2), "*", daft.DataType.decimal128(36, 4), id="multiply"),
        pytest.param((10, 2), pa.int32(), "*", daft.DataType.decimal128(20, 2), id="multiply-integer"),
    ],
)
def test_decimal_result_types_are_unchanged(a_type, b_type, operator, expected) -> None:
    frame = _frame([], a_type, [], b_type)
    expression = daft.col("a") / daft.col("b") if operator == "/" else daft.col("a") * daft.col("b")

    assert frame.select(expression.alias("r")).schema()["r"].dtype == expected


def test_decimal_division_by_zero_is_null() -> None:
    frame = _frame(["1.00", "1.00"], (10, 2), ["0", "4"], (10, 2))

    quotients = frame.select((daft.col("a") / daft.col("b")).alias("q")).to_pydict()["q"]

    assert quotients == [None, decimal.Decimal("0.2500000000000")]


def test_a_guarded_decimal_division_by_zero_answers() -> None:
    frame = _frame(["1.00", "1.00"], (10, 2), ["0", "4"], (10, 2))
    guarded = daft.functions.when(daft.col("b") != 0, daft.col("a") / daft.col("b")).otherwise(None)

    quotients = frame.select(guarded.alias("q")).to_pydict()["q"]

    assert quotients == [None, decimal.Decimal("0.2500000000000")]


@pytest.mark.parametrize(
    "expression, expected",
    [
        pytest.param(daft.col("a") / daft.col("b"), [decimal.Decimal("0.5"), None, None], id="nulls-on-either-side"),
        pytest.param(
            daft.col("a") / daft.lit(decimal.Decimal("2.00")).cast(daft.DataType.decimal128(10, 2)),
            [decimal.Decimal("0.5"), None, decimal.Decimal(1)],
            id="scalar-divisor",
        ),
        pytest.param(
            daft.lit(decimal.Decimal("3.00")).cast(daft.DataType.decimal128(10, 2)) / daft.col("b"),
            [decimal.Decimal("1.5"), decimal.Decimal(1), None],
            id="scalar-dividend",
        ),
        pytest.param(daft.col("a") * daft.col("b"), [decimal.Decimal(2), None, None], id="product-nulls"),
        pytest.param(
            daft.col("a") * daft.lit(None).cast(daft.DataType.decimal128(10, 2)), [None, None, None], id="null-scalar"
        ),
    ],
)
def test_decimal_arithmetic_carries_nulls_and_scalars(expression, expected) -> None:
    frame = _frame(["1.00", None, "2.00"], (10, 2), ["2.00", "3.00", None], (10, 2))

    answers = frame.select(expression.alias("r")).to_pydict()["r"]

    assert answers == expected


def test_an_integer_operand_divides_exactly() -> None:
    frame = _frame([2, -2], pa.int32(), ["3.00", "3.00"], (10, 2))

    quotients = frame.select((daft.col("a") / daft.col("b")).alias("q")).to_pydict()["q"]

    assert quotients == [decimal.Decimal("0.66666666667"), decimal.Decimal("-0.66666666667")]


def _random_values(generator: random.Random, dtype: DecimalType | pa.DataType, count: int) -> list[str | int]:
    if isinstance(dtype, pa.DataType):
        return [generator.randint(-(2**31), 2**31 - 1) for _ in range(count)]
    precision, scale = dtype
    values = []
    for _ in range(count):
        digits = generator.randint(1, precision)
        unscaled = generator.randint(-(10**digits) + 1, 10**digits - 1)
        values.append(str(decimal.Decimal(unscaled).scaleb(-scale)))
    return values


@pytest.mark.parametrize(
    "a_type, b_type",
    [
        pytest.param((10, 2), (10, 2), id="10-2"),
        pytest.param((18, 2), (18, 2), id="18-2"),
        pytest.param((18, 6), (18, 2), id="18-6-by-18-2"),
        pytest.param((5, 2), (10, 2), id="5-2-by-10-2"),
        pytest.param(pa.int32(), (10, 2), id="int32-by-10-2"),
    ],
)
@pytest.mark.parametrize("operator", ["/", "*"])
def test_decimal_arithmetic_matches_exact_arithmetic(a_type, b_type, operator, wide_context) -> None:
    generator = random.Random(53)
    left = _random_values(generator, a_type, 500)
    right = _random_values(generator, b_type, 500)
    frame = _frame(left, a_type, right, b_type)
    expression = daft.col("a") / daft.col("b") if operator == "/" else daft.col("a") * daft.col("b")
    result = frame.select(expression.alias("r"))

    answers = result.to_pydict()["r"]

    dtype = result.schema()["r"].dtype
    for a, b, answer in zip(left, right, answers):
        a, b = decimal.Decimal(a), decimal.Decimal(b)
        if operator == "/" and b.is_zero():
            expected = None
        else:
            expected = _exact(a / b if operator == "/" else a * b, dtype)
        assert answer == expected, f"{a} {operator} {b}"
