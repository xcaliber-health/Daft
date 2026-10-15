"""A restriction to some groups of an aggregate, a window or a distinct.

A filter, semi join or anti join that reads only the columns rows are grouped by
keeps or drops each group whole, so the engine runs it before the grouping:
only the selected groups' rows are grouped, and a filter can reach the scan.
Each case's answer is the one the restriction gives when applied after the
grouping; the placement is read from the optimized plan.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import daft
from daft import DataFrame, Window, col, lit

ROWS = {
    "k": [1, 1, 2, 2, 3, None, None],
    "v": [1, 2, 3, 4, 5, 6, 7],
}


def _rows() -> DataFrame:
    return daft.from_pydict(ROWS)


def _sorted(frame: DataFrame, by: list[str]) -> dict[str, list[object]]:
    return frame.sort(by, nulls_first=[True] * len(by)).to_pydict()


def _totals() -> DataFrame:
    return _rows().groupby("k").agg(col("v").sum().alias("total"))


def _running() -> DataFrame:
    by_key = Window().partition_by("k").order_by("v").rows_between(Window.unbounded_preceding, Window.current_row)
    return _rows().with_column("running", col("v").sum().over(by_key))


CASES: dict[str, tuple[Callable[[], DataFrame], list[str], dict[str, list[object]]]] = {
    "filter on the group key": (
        lambda: _totals().where(col("k") == 2),
        ["k"],
        {"k": [2], "total": [7]},
    ),
    "filter on a null group key": (
        lambda: _totals().where(col("k").is_null()),
        ["k"],
        {"k": [None], "total": [13]},
    ),
    "filter mixing the key and a total": (
        lambda: _totals().where((col("k") >= 2) & (col("total") > 5)),
        ["k"],
        {"k": [2], "total": [7]},
    ),
    "filter that keeps nothing over a global total": (
        lambda: _rows().agg(col("v").sum().alias("total")).where(lit(False)),
        ["total"],
        {"total": []},
    ),
    "filter on the partition key of an ordered, framed window": (
        lambda: _running().where(col("k") == 1),
        ["k", "v"],
        {"k": [1, 1], "v": [1, 2], "running": [1, 3]},
    ),
    "semi join on the group key": (
        lambda: _totals().join(daft.from_pydict({"k": [1, 3, 9]}), on="k", how="semi"),
        ["k"],
        {"k": [1, 3], "total": [3, 5]},
    ),
    "anti join on the group key": (
        lambda: _totals().join(daft.from_pydict({"k": [1, 3]}), on="k", how="anti"),
        ["k"],
        {"k": [None, 2], "total": [13, 7]},
    ),
    "semi join on the partition key": (
        lambda: _running().join(daft.from_pydict({"k": [2]}), on="k", how="semi"),
        ["k", "v"],
        {"k": [2, 2], "v": [3, 4], "running": [3, 7]},
    ),
    "filter over distinct rows": (
        lambda: daft.from_pydict({"k": [1, 1, 2], "v": [1, 1, 2]}).distinct().where(col("v") == 1),
        ["k"],
        {"k": [1], "v": [1]},
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_restriction_to_some_groups_answers_as_after_the_grouping(case: str) -> None:
    frame, by, expected = CASES[case]

    assert _sorted(frame(), by) == expected


def _optimized(frame: DataFrame) -> list[str]:
    """The optimized plan's operators, from the root down to the scan."""
    written = io.StringIO()
    frame.explain(show_all=True, file=written)
    plan = written.getvalue().split("== Optimized Logical Plan ==")[1].split("== Physical Plan ==")[0]
    return [line.strip("|* ").split(":")[0] for line in plan.splitlines() if "* " in line]


@pytest.fixture
def stored(tmp_path: Path) -> DataFrame:
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"k": list(range(10)) * 10, "v": list(range(100))}), path)
    return daft.read_parquet(str(path))


PLACED: dict[str, tuple[Callable[[DataFrame], DataFrame], str, str]] = {
    "filter below an aggregate": (
        lambda rows: rows.groupby("k").agg(col("v").sum()).where(col("k").is_in([1, 2])),
        "Filter",
        "Aggregation",
    ),
    "filter below a window": (
        lambda rows: rows.with_column("t", col("v").sum().over(Window().partition_by("k"))).where(col("k") == 1),
        "Filter",
        "Window",
    ),
    "semi join below a window": (
        lambda rows: rows.with_column("t", col("v").sum().over(Window().partition_by("k"))).join(
            daft.from_pydict({"k": [1]}), on="k", how="semi"
        ),
        "Join",
        "Window",
    ),
}


@pytest.mark.parametrize("case", sorted(PLACED))
def test_a_restriction_on_grouping_columns_runs_below_the_grouping(case: str, stored: DataFrame) -> None:
    restricted, restriction, grouping = PLACED[case]

    operators = _optimized(restricted(stored))

    # A filter the scan absorbed is below everything.
    assert restriction not in operators or operators.index(restriction) > operators.index(grouping), operators


def test_a_semi_join_on_the_group_key_stays_above_the_aggregate(stored: DataFrame) -> None:
    # Below, it would probe every row rather than one per group, which costs more
    # than the rows it saves aggregating.
    restricted = stored.groupby("k").agg(col("v").sum()).join(daft.from_pydict({"k": [1]}), on="k", how="semi")

    operators = _optimized(restricted)

    assert operators.index("Join") < operators.index("Aggregation"), operators


def test_a_filter_below_an_aggregate_reaches_the_scan(stored: DataFrame) -> None:
    operators = _optimized(stored.groupby("k").agg(col("v").sum()).where(col("k") == 1))

    assert "Filter" not in operators, operators
