from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

import pytest

import daft
from daft import DataType, Expression, Window, col
from daft.functions import dense_rank, lag, lead, rank, row_number

ORDERED = Window().partition_by("g").order_by("o")
FRAMES = {
    "running rows": ORDERED.rows_between(Window.unbounded_preceding, Window.current_row),
    "whole partition": ORDERED.rows_between(Window.unbounded_preceding, Window.unbounded_following),
    "running range": ORDERED.range_between(Window.unbounded_preceding, Window.current_row),
    "sliding rows": ORDERED.rows_between(-1, 1),
}


def _decimals() -> daft.DataFrame:
    return daft.from_pydict(
        {"g": [1, 1, 2, 2, 2], "o": [1, 2, 1, 2, 3], "v": ["1.10", "2.20", "1.10", "3.30", "5.50"]}
    ).with_column("v", col("v").cast(DataType.decimal128(18, 2)))


@pytest.mark.parametrize("frame", list(FRAMES), ids=list(FRAMES))
@pytest.mark.parametrize(
    "aggregate",
    [pytest.param(lambda v: v.sum(), id="sum"), pytest.param(lambda v: v.mean(), id="mean")],
)
def test_a_decimal_aggregates_over_a_frame_as_it_does_as_a_float(
    frame: str, aggregate: Callable[[Expression], Expression]
) -> None:
    df = _decimals()

    as_decimal = df.select("g", "o", aggregate(col("v")).over(FRAMES[frame]).alias("w")).sort(["g", "o"])
    as_float = df.select("g", "o", aggregate(col("v").cast(DataType.float64())).over(FRAMES[frame]).alias("w")).sort(
        ["g", "o"]
    )

    assert as_decimal.schema()["w"].dtype == df.select(aggregate(col("v")).alias("w")).schema()["w"].dtype
    observed = [float(value) for value in as_decimal.to_pydict()["w"]]
    assert observed == pytest.approx(as_float.to_pydict()["w"])


def test_a_decimal_running_total_is_exact() -> None:
    df = _decimals()

    running = df.select("g", "o", col("v").sum().over(FRAMES["running rows"]).alias("w")).sort(["g", "o"])

    assert running.to_pydict()["w"] == [
        Decimal("1.10"),
        Decimal("3.30"),
        Decimal("1.10"),
        Decimal("4.40"),
        Decimal("9.90"),
    ]


@pytest.mark.parametrize("frame", list(FRAMES), ids=list(FRAMES))
@pytest.mark.parametrize(
    ("positional", "named"),
    [
        pytest.param(row_number, "row_number", id="row_number"),
        pytest.param(rank, "rank", id="rank"),
        pytest.param(dense_rank, "dense_rank", id="dense_rank"),
        pytest.param(lambda: lag(col("v"), 1), "lag", id="lag"),
        pytest.param(lambda: lead(col("v"), 1), "lead", id="lead"),
    ],
)
def test_a_positional_function_given_a_frame_is_refused(
    frame: str, positional: Callable[[], Expression], named: str
) -> None:
    df = _decimals()

    with pytest.raises(ValueError, match=rf"{named}\(\) does not take a frame"):
        df.select(positional().over(FRAMES[frame]).alias("w")).collect()


@pytest.mark.parametrize(
    "positional",
    [pytest.param(row_number, id="row_number"), pytest.param(lambda: lag(col("v"), 1), id="lag")],
)
def test_a_positional_function_without_a_frame_still_answers(positional: Callable[[], Expression]) -> None:
    df = _decimals()

    result = df.select(positional().over(ORDERED).alias("w")).to_pydict()

    assert len(result["w"]) == 5
