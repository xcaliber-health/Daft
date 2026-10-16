"""A query whose join's build side fails reports that failure when it happens.

The probe side of a join waits for the build side's result. When the build
side stops without one, because what it reads failed or it failed itself, the
probe side stops waiting at once, and the query reports the build side's own
failure, with its class, instead of waiting out the failed pipeline's grace.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

import daft
from daft import DataFrame, col, lit
from daft.exceptions import DaftCardinalityError, DaftComputeError
from daft.functions import single_value
from tests.conftest import get_tests_daft_runner_name

pytestmark = pytest.mark.skipif(
    get_tests_daft_runner_name() != "native", reason="the wind-down grace is the native runner's"
)

# Well under the 10 s a failed pipeline's remaining tasks are given to end.
PROMPT_SECONDS = 2.0


@daft.func(return_dtype=daft.DataType.string())
def refuse(value: str) -> str:
    raise ValueError(f"refused {value}")


def _subjects() -> DataFrame:
    return daft.from_pydict({"subject": ["s1", "s2", "s3"], "t": [1, 2, 3]})


def _held() -> DataFrame:
    return daft.from_pydict({"subject": ["s1", "s2", "s2", "s3"], "code": ["a", "b", "c", "d"], "t": [1, 2, 2, 3]})


def _claimed() -> DataFrame:
    """One code per subject, claimed of a subject holding two."""
    return _held().groupby("subject").agg(single_value(col("code")).alias("code")).with_column("t", lit(2))


def _refused() -> DataFrame:
    return _held().with_column("code", refuse(col("code")))


FAILURES: dict[str, tuple[Callable[[], DataFrame], type[Exception]]] = {
    "a claim of one row meeting two": (_claimed, DaftCardinalityError),
    "a function raising": (_refused, DaftComputeError),
}

Shape = Callable[[DataFrame, DataFrame], DataFrame]

SHAPES: dict[str, Shape] = {
    "left join, failing on the right": lambda rows, failing: rows.join(failing, on="subject", how="left"),
    "right join, failing on the left": lambda rows, failing: failing.join(rows, on="subject", how="right"),
    "inner join, failing on the right": lambda rows, failing: rows.join(failing, on="subject"),
    "inner join, failing on the left": lambda rows, failing: failing.join(rows, on="subject"),
    "outer join, failing on the right": lambda rows, failing: rows.join(failing, on="subject", how="outer"),
    "outer join, failing on the left": lambda rows, failing: failing.join(rows, on="subject", how="outer"),
    "cross join, failing on the right": lambda rows, failing: rows.join(failing.select("code"), how="cross"),
    "cross join, failing on the left": lambda rows, failing: failing.select("code").join(rows, how="cross"),
    "asof join, failing on the right": lambda rows, failing: rows.join_asof(failing, on="t", by="subject"),
    "asof join, failing on the left": lambda rows, failing: failing.join_asof(rows, on="t", by="subject"),
}


def _failure_and_seconds(query: DataFrame) -> tuple[BaseException, float]:
    began = time.perf_counter()
    with pytest.raises(Exception) as raised:
        query.to_arrow()
    return raised.value, time.perf_counter() - began


@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
@pytest.mark.parametrize("failing,expected", FAILURES.values(), ids=FAILURES.keys())
def test_a_failing_side_of_a_join_is_reported_promptly_with_its_class(
    shape: Shape, failing: Callable[[], DataFrame], expected: type[Exception]
) -> None:
    failure, seconds = _failure_and_seconds(shape(_subjects(), failing()))

    assert isinstance(failure, expected)
    assert seconds < PROMPT_SECONDS


@pytest.mark.parametrize("failing,expected", FAILURES.values(), ids=FAILURES.keys())
def test_the_failure_reported_is_always_the_build_sides_own(
    failing: Callable[[], DataFrame], expected: type[Exception]
) -> None:
    reported = [_failure_and_seconds(_subjects().join(failing(), on="subject", how="left"))[0] for _ in range(25)]

    assert [type(failure) for failure in reported] == [expected] * 25
