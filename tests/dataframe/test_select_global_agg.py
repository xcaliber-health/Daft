from __future__ import annotations

from collections.abc import Callable

import pytest

import daft
from daft import col, lit


def test_select_global_agg_returns_single_row() -> None:
    df = daft.from_pydict({"a": [1, 2, 3], "b": [4, 5, 6]})

    res = df.select(col("a").sum().alias("sum_a")).collect().to_pydict()

    assert res == {"sum_a": [6]}


def test_select_global_agg_allows_literals() -> None:
    df = daft.from_pydict({"a": [1, 2, 3]})

    res = (
        df.select(
            col("a").sum().alias("sum_a"),
            lit(1).alias("one"),
        )
        .collect()
        .to_pydict()
    )

    assert res == {"sum_a": [6], "one": [1]}


def test_select_global_agg_allows_multiple_aggs() -> None:
    df = daft.from_pydict({"a": [1, 2, 3]})

    res = (
        df.select(
            col("a").sum().alias("sum_a"),
            col("a").count().alias("cnt_a"),
        )
        .collect()
        .to_pydict()
    )

    assert res == {"sum_a": [6], "cnt_a": [3]}


def test_select_global_agg_without_alias() -> None:
    df = daft.from_pydict({"a": [1, 2, 3], "b": [4, 5, 6]})

    res = df.select(col("a").sum()).collect().to_pydict()

    assert res == {"a": [6]}


def test_select_global_agg_rejects_non_agg_column_reference() -> None:
    df = daft.from_pydict({"a": [1, 2, 3]})

    with pytest.raises(ValueError, match="Expressions in aggregations"):
        df.select(col("a").sum().alias("sum_a"), col("a")).collect()


@pytest.mark.parametrize("partitions", [1, 3])
@pytest.mark.parametrize(
    "read",
    [
        pytest.param(lambda paired: paired.select("s", "m"), id="both sides"),
        pytest.param(lambda paired: paired.select("s"), id="first side only"),
        pytest.param(lambda paired: paired.select("m"), id="second side only"),
        pytest.param(lambda paired: paired.select(lit(1).alias("one")), id="neither side"),
    ],
)
def test_an_unread_global_aggregate_is_still_one_row(
    partitions: int, read: Callable[[daft.DataFrame], daft.DataFrame]
) -> None:
    first = daft.from_pydict({"a": list(range(8))}).into_partitions(partitions).agg(col("a").sum().alias("s"))
    second = daft.from_pydict({"b": [1, 2]}).into_partitions(partitions).agg(col("b").max().alias("m"))
    paired = first.join(second, how="cross")

    assert read(paired).count_rows() == 1
    assert len(read(paired).to_arrow()) == 1


def test_a_global_aggregate_read_by_nothing_counts_one_row() -> None:
    df = daft.from_pydict({"a": [1, 2, 3]}).agg(col("a").sum().alias("s"))

    assert df.count_rows() == 1
