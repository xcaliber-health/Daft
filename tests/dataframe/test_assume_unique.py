"""Stating that a source's rows are unique by some columns helps plan, and never changes answers."""

from __future__ import annotations

import io
from collections.abc import Callable

import pytest

import daft
from daft import DataFrame, col
from daft.context import planning_config_ctx

# Events, each naming a subject; subjects, each held twice, so the stated key is false.
EVENTS = {"event_id": list(range(40)), "subject": [i % 8 for i in range(40)], "v": [i % 5 for i in range(40)]}
SUBJECTS = {"subject": [s for s in range(8) for _ in range(2)], "tier": [s % 3 for s in range(8) for _ in range(2)]}


def optimized_plan(df: DataFrame) -> str:
    out = io.StringIO()
    df.explain(show_all=True, file=out)
    return out.getvalue().split("Optimized Logical Plan")[1].split("Physical Plan")[0]


def rows(df: DataFrame) -> list[tuple[object, ...]]:
    return sorted(zip(*df.to_pydict().values()), key=repr)


def test_a_stated_key_shows_in_the_plan() -> None:
    df = daft.from_pydict(EVENTS).assume_unique("event_id").assume_unique("subject", "v")

    assert "Assumed unique by = [event_id], [subject, v]" in optimized_plan(df)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        pytest.param(lambda: daft.from_pydict(EVENTS).assume_unique(), "at least one column", id="no_column"),
        pytest.param(lambda: daft.from_pydict(EVENTS).assume_unique("nope"), "no column", id="unknown_column"),
        pytest.param(
            lambda: daft.from_pydict(EVENTS).where(col("v") > 1).assume_unique("event_id"),
            "read directly from a source",
            id="not_a_source",
        ),
    ],
)
def test_a_key_is_stated_only_on_existing_columns_of_a_source(build: Callable[[], DataFrame], message: str) -> None:
    with pytest.raises(Exception, match=message):
        build()


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param("lookup", id="lookup"),
        pytest.param("repeating", id="repeating_join"),
        pytest.param("filtering", id="filtering_join"),
    ],
)
def test_a_false_key_never_changes_an_answer(shape: str) -> None:
    def build(state_keys: bool) -> DataFrame:
        events = daft.from_pydict(EVENTS)
        subjects = daft.from_pydict(SUBJECTS)
        if state_keys:
            # False: every subject is held twice, and events share subjects.
            subjects = subjects.assume_unique("subject")
            events = events.assume_unique("subject")
        if shape == "repeating":
            subjects = daft.from_pydict(SUBJECTS).assume_unique("tier") if state_keys else subjects
        if shape == "filtering":
            subjects = subjects.where(col("subject") < 2)
        return (
            events.join(subjects, on="subject")
            .groupby("tier")
            .agg(col("v").sum().alias("s"), col("v").count().alias("n"))
        )

    stated = rows(build(True))

    assert stated == rows(build(False))
    with planning_config_ctx(disable_partial_aggregate_pushdown=True):
        assert stated == rows(build(True))
