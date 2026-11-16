from __future__ import annotations

import operator
from decimal import Decimal

import pyarrow as pa
import pytest

import daft
from daft import col

BIG = "9" * 36 + ".99"  # 36 integer digits: fits (38, 2), not (38, 6)
MAX_6 = "9" * 32 + ".999999"  # the largest (38, 6) value

COMPARISONS = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def _column(values: list[str | None], precision: int, scale: int) -> pa.Array:
    return pa.array([None if v is None else Decimal(v) for v in values], pa.decimal128(precision, scale))


def _frame(
    left: list[str | None], left_type: tuple[int, int], right: list[str | None], right_type: tuple[int, int]
) -> daft.DataFrame:
    return daft.from_arrow(pa.table({"l": _column(left, *left_type), "r": _column(right, *right_type)}))


def _compare(frame: daft.DataFrame, symbol: str, reversed_operands: bool) -> list[bool | None]:
    left, right = (col("r"), col("l")) if reversed_operands else (col("l"), col("r"))
    return frame.select(COMPARISONS[symbol](left, right).alias("c")).to_pydict()["c"]


ROWS = [
    pytest.param("10.00", "3.333333", id="total-against-mean"),
    pytest.param("1.12", "1.123456", id="differs-past-the-second-place"),
    pytest.param("1.12", "1.120000", id="equal-at-the-larger-scale"),
    pytest.param(BIG, MAX_6, id="past-the-other-type"),
    pytest.param("-" + BIG, "-" + MAX_6, id="negative-past-the-other-type"),
    pytest.param("-1.00", "0.000001", id="negative-against-positive"),
    pytest.param("0.00", "-0.000000", id="zeros"),
]


@pytest.mark.parametrize("symbol", COMPARISONS)
@pytest.mark.parametrize("reversed_operands", [False, True], ids=["left-first", "right-first"])
@pytest.mark.parametrize("left, right", ROWS)
def test_decimals_past_38_digits_compare_exactly(left, right, symbol, reversed_operands) -> None:
    frame = _frame([left], (38, 2), [right], (38, 6))
    a, b = (Decimal(right), Decimal(left)) if reversed_operands else (Decimal(left), Decimal(right))

    assert _compare(frame, symbol, reversed_operands) == [COMPARISONS[symbol](a, b)]


def test_a_full_width_column_equals_a_constant_of_higher_scale() -> None:
    frame = daft.from_arrow(pa.table({"l": _column(["1.50", "1.51"], 38, 2)}))
    constant = daft.lit(Decimal("1.500000")).cast(daft.DataType.decimal128(18, 6))

    assert frame.select((col("l") == constant).alias("c")).to_pydict()["c"] == [True, False]


def test_rows_inside_and_past_128_bits_answer_in_order() -> None:
    left = ["1.00", BIG, "-2.50", "-" + BIG, "3.00"]
    right = ["1.000000", MAX_6, "-2.500001", "0.000001", "2.999999"]
    frame = _frame(left, (38, 2), right, (38, 6))

    answered = _compare(frame, ">", reversed_operands=False)

    assert answered == [Decimal(a) > Decimal(b) for a, b in zip(left, right)]


@pytest.mark.parametrize("symbol", ["==", "<"])
def test_nulls_on_either_side_are_null(symbol) -> None:
    frame = _frame([None, "1.00", None], (38, 2), ["1.000000", None, None], (38, 6))

    assert _compare(frame, symbol, reversed_operands=False) == [None, None, None]


def test_null_safe_equality_compares_nulls() -> None:
    frame = _frame([None, "1.00", None, "1.12"], (38, 2), [None, None, "1.000000", "1.120000"], (38, 6))

    answered = frame.select(col("l").eq_null_safe(col("r")).alias("c")).to_pydict()["c"]

    assert answered == [True, False, False, True]


def test_between_bounds_of_three_types() -> None:
    frame = daft.from_arrow(
        pa.table(
            {
                "v": _column(["10.00", "1.12", None, BIG], 38, 2),
                "lo": _column(["3.333333", "1.120000", "0.000000", "1.000000"], 38, 6),
                "hi": _column(["99.9999", "1.1200", "5.0000", "2.0000"], 30, 4),
            }
        )
    )

    answered = frame.select(col("v").between(col("lo"), col("hi")).alias("c")).to_pydict()["c"]

    assert answered == [True, True, None, False]


def test_is_in_compares_members_exactly() -> None:
    frame = daft.from_arrow(pa.table({"v": _column(["1.12", "1.13", None, BIG], 38, 2)}))
    members = [Decimal("1.120000"), Decimal("1.130001")]

    answered = frame.select(
        col("v").is_in([daft.lit(m).cast(daft.DataType.decimal128(18, 6)) for m in members]).alias("c")
    ).to_pydict()["c"]

    assert answered == [True, False, None, False]


def test_comparisons_within_38_digits_are_unchanged() -> None:
    frame = _frame(["1.12", "2.50"], (18, 2), ["1.120000", "2.400000"], (18, 6))

    assert _compare(frame, "==", reversed_operands=False) == [True, False]
    assert _compare(frame, ">", reversed_operands=False) == [False, True]


def test_clip_of_decimals_past_38_digits_still_refuses() -> None:
    frame = _frame(["1.12"], (38, 2), ["1.000000"], (38, 6))

    with pytest.raises(Exception, match="38"):
        frame.select(col("l").clip(col("r"), col("r"))).collect()
