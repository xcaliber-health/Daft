from __future__ import annotations

import itertools

import pyarrow as pa
import pytest

import daft
from daft import col
from daft.functions import when

SIGNED = [pa.int8(), pa.int16(), pa.int32(), pa.int64()]
UNSIGNED = [pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64()]
FLOATS = [pa.float32(), pa.float64()]


def _frame(dividends: list, divisors: list, dtype: pa.DataType) -> daft.DataFrame:
    return daft.from_arrow(pa.table({"a": pa.array(dividends, dtype), "b": pa.array(divisors, dtype)}))


def _divide(frame: daft.DataFrame) -> dict[str, list]:
    return frame.select((col("a") // col("b")).alias("q"), (col("a") % col("b")).alias("r")).to_pydict()


@pytest.mark.parametrize(
    "a, b, quotient, remainder",
    [
        pytest.param(-7, 2, -4, 1, id="negative-dividend"),
        pytest.param(7, -2, -4, -1, id="negative-divisor"),
        pytest.param(-7, -2, 3, -1, id="both-negative"),
        pytest.param(7, 2, 3, 1, id="both-positive"),
    ],
)
def test_integer_division_floors(a, b, quotient, remainder) -> None:
    answered = _divide(_frame([a], [b], pa.int64()))

    assert answered == {"q": [quotient], "r": [remainder]}


@pytest.mark.parametrize(
    "a, b, quotient, remainder",
    [
        pytest.param(-7.0, 2.0, -4.0, 1.0, id="negative-dividend"),
        pytest.param(7.0, -2.0, -4.0, -1.0, id="negative-divisor"),
    ],
)
def test_float_division_floors(a, b, quotient, remainder) -> None:
    answered = _divide(_frame([a], [b], pa.float64()))

    assert answered == {"q": [quotient], "r": [remainder]}


@pytest.mark.parametrize("dtype", SIGNED + UNSIGNED + FLOATS, ids=str)
def test_quotient_and_remainder_rebuild_the_dividend(dtype) -> None:
    values = [-9, -7, -1, 1, 7, 9] if dtype in SIGNED + FLOATS else [1, 7, 9, 100]
    divisors = [-4, -2, -1, 1, 2, 4] if dtype in SIGNED + FLOATS else [1, 2, 4, 7]
    pairs = list(itertools.product(values, divisors))
    frame = _frame([a for a, _ in pairs], [b for _, b in pairs], dtype)

    answered = _divide(frame)

    for (a, b), q, r in zip(pairs, answered["q"], answered["r"]):
        assert b * q + r == a, f"{a} // {b}"
        assert r == 0 or (r < 0) == (b < 0), f"{a} % {b} takes the divisor's sign"


@pytest.mark.parametrize("dtype", UNSIGNED, ids=str)
def test_unsigned_division_is_unchanged(dtype) -> None:
    answered = _divide(_frame([7, 9], [2, 4], dtype))

    assert answered == {"q": [3, 2], "r": [1, 1]}


@pytest.mark.parametrize("dtype", SIGNED + UNSIGNED, ids=str)
def test_a_zero_integer_divisor_is_null(dtype) -> None:
    answered = _divide(_frame([7, 9, None], [0, 2, 0], dtype))

    assert answered == {"q": [None, 4, None], "r": [None, 1, None]}


def test_a_guarded_zero_divisor_answers() -> None:
    frame = _frame([7, 9], [0, 2], pa.int64())
    guarded = when(col("b") != 0, col("a") // col("b")).otherwise(None)

    assert frame.select(guarded.alias("q")).to_pydict() == {"q": [None, 4]}


@pytest.mark.parametrize("dtype", SIGNED, ids=str)
def test_the_lowest_value_by_minus_one_wraps(dtype) -> None:
    lowest = -(2 ** (dtype.bit_width - 1))

    answered = _divide(_frame([lowest], [-1], dtype))

    assert answered == {"q": [lowest], "r": [0]}


def test_a_scalar_on_either_side_floors() -> None:
    frame = _frame([-7, 7], [2, -2], pa.int64())

    answered = frame.select(
        (col("a") // 2).alias("by_two"),
        (col("a") % 2).alias("mod_two"),
        (daft.lit(-7) // col("b")).alias("seven_by"),
        (col("a") // 0).alias("by_zero"),
    ).to_pydict()

    assert answered == {"by_two": [-4, 3], "mod_two": [1, 1], "seven_by": [-4, 3], "by_zero": [None, None]}


def test_series_floor_too() -> None:
    a = daft.Series.from_pylist([-7, 7, 1])
    b = daft.Series.from_pylist([2, -2, 0])

    assert (a // b).to_pylist() == [-4, -4, None]
    assert (a % b).to_pylist() == [1, -1, None]


def test_sql_modulo_and_integer_division_truncate() -> None:
    frame = _frame([-7, 7, 1], [2, -2, 0], pa.int64())

    answered = daft.sql("SELECT a % b AS r, MOD(a, b) AS m, a // b AS q FROM frame", frame=frame).to_pydict()

    assert answered == {"r": [-1, 1, None], "m": [-1, 1, None], "q": [-3, -3, None]}


def test_sql_float_division_keeps_flooring_and_modulo_keeps_truncating() -> None:
    frame = _frame([-7.0], [2.0], pa.float64())

    answered = daft.sql("SELECT a // b AS q, a % b AS r FROM frame", frame=frame).to_pydict()

    assert answered == {"q": [-4.0], "r": [-1.0]}
