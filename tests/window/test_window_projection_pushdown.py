from __future__ import annotations

import contextlib
import io
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pytest

import daft
from daft import Window, col
from daft.functions import lag, row_number
from daft.functions import sum as total

COLUMNS = ["id", "g", "o", "a", "b", "c", "d", "e"]
ORDERED = Window().partition_by("g").order_by("o")
RUNNING = ORDERED.rows_between(Window.unbounded_preceding, Window.current_row)


def _pushed_columns(frame: daft.DataFrame) -> set[str] | None:
    """Return the columns the optimized plan reads from its scan, or None when it reads all."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        frame.explain(show_all=True)
    plan = printed.getvalue()
    optimized = plan[plan.find("== Optimized") : plan.find("== Physical")]
    for line in optimized.splitlines():
        if "Projection pushdown" in line:
            return set(line.split("=", 1)[1].strip().strip("[]").split(", "))
    return None


@pytest.fixture
def table(tmp_path: Path) -> Callable[[], daft.DataFrame]:
    rows = {name: [i % 3 if name == "g" else i for i in range(6)] for name in COLUMNS}
    daft.from_arrow(pa.table(rows)).write_parquet(str(tmp_path))
    return lambda: daft.read_parquet(str(tmp_path))


@pytest.mark.parametrize(
    ("query", "reads"),
    [
        pytest.param(
            lambda df: df.select("id", row_number().over(ORDERED).alias("w")), {"id", "g", "o"}, id="row number"
        ),
        pytest.param(
            lambda df: df.select("id", lag(col("a"), 1).over(ORDERED).alias("w")),
            {"id", "g", "o", "a"},
            id="prior value",
        ),
        pytest.param(
            lambda df: df.select("id", total(col("a")).over(RUNNING).alias("w")),
            {"id", "g", "o", "a"},
            id="running total",
        ),
        pytest.param(
            lambda df: df.select("id", total(col("a")).over(Window().partition_by("g")).alias("w")),
            {"id", "g", "a"},
            id="partition total",
        ),
        pytest.param(
            lambda df: df.with_column("w", total(col("a")).over(Window().partition_by("g"))).select("id"),
            {"id"},
            id="unread window",
        ),
    ],
)
def test_a_window_reads_only_the_columns_it_needs(
    table: Callable[[], daft.DataFrame], query: Callable[[daft.DataFrame], daft.DataFrame], reads: set[str]
) -> None:
    assert _pushed_columns(query(table())) == reads


@pytest.mark.parametrize(
    "query",
    [
        pytest.param(lambda df: df.select("id", row_number().over(ORDERED).alias("w")), id="row number"),
        pytest.param(lambda df: df.select("id", total(col("a")).over(RUNNING).alias("w")), id="running total"),
        pytest.param(
            lambda df: df.with_columns(
                {"w1": total(col("a")).over(ORDERED.rows_between(-1, 1)), "w2": lag(col("b"), 1).over(ORDERED)}
            ).select("id", "w1"),
            id="one of two results read",
        ),
    ],
)
def test_pruning_under_a_window_keeps_its_answers(
    table: Callable[[], daft.DataFrame], query: Callable[[daft.DataFrame], daft.DataFrame]
) -> None:
    # The same query over the table held in memory is never pruned.
    unpruned = daft.from_arrow(table().to_arrow())

    expected = query(unpruned).sort("id").to_pydict()
    observed = query(table()).sort("id").to_pydict()

    assert observed == expected
