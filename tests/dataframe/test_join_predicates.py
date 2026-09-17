"""Joins whose predicate says more than key equality.

The extra condition is checked on the pairs key equality allows, and a row whose
every candidate fails it has matched nothing, which is what an outer join has to
report.
"""

from __future__ import annotations

import pytest

import daft
from daft import col
from daft.daft import JoinType
from daft.dataframe import DataFrame

JOIN_TYPES = {
    "inner": JoinType.Inner,
    "left": JoinType.Left,
    "right": JoinType.Right,
    "outer": JoinType.Outer,
}


def _join(left, right, predicate, how):
    return DataFrame(left._builder.join_on(right._builder, predicate, JOIN_TYPES[how]))


@pytest.fixture
def sides():
    left = daft.from_pydict({"k": [1, 1, 2, 3], "lv": [10, 11, 20, 30]})
    right = daft.from_pydict({"k2": [1, 1, 2, 4], "rv": [5, 12, 25, 40]})
    return left, right


def _ordered(rows):
    """Return rows in a fixed order, so only their content is compared."""
    return sorted(rows, key=lambda row: tuple((value is None, value) for value in row))


def _rows(frame):
    """Return the joined rows, in a fixed order."""
    out = frame.to_pydict()
    return _ordered(zip(out["k"], out["lv"], out["k2"], out["rv"]))


def test_inner_join_keeps_only_pairs_the_condition_allows(sides):
    left, right = sides

    rows = _rows(_join(left, right, (col("k") == col("k2")) & (col("rv") > col("lv")), "inner"))

    assert rows == _ordered([(1, 10, 1, 12), (1, 11, 1, 12), (2, 20, 2, 25)])


def test_left_join_reports_a_row_whose_every_pair_fails_as_unmatched(sides):
    left, right = sides

    rows = _rows(_join(left, right, (col("k") == col("k2")) & (col("rv") > col("lv")), "left"))

    assert rows == _ordered([(1, 10, 1, 12), (1, 11, 1, 12), (2, 20, 2, 25), (3, 30, None, None)])


def test_right_join_reports_the_other_side_the_same_way(sides):
    left, right = sides

    rows = _rows(_join(left, right, (col("k") == col("k2")) & (col("rv") > col("lv")), "right"))

    assert rows == _ordered(
        [
            (None, None, 1, 5),
            (None, None, 4, 40),
            (1, 10, 1, 12),
            (1, 11, 1, 12),
            (2, 20, 2, 25),
        ]
    ), "the row whose only key match fails the condition is unmatched, not dropped"


def test_outer_join_reports_both_sides(sides):
    left, right = sides

    rows = _rows(_join(left, right, (col("k") == col("k2")) & (col("rv") > col("lv")), "outer"))

    assert rows == _ordered(
        [
            (None, None, 1, 5),
            (None, None, 4, 40),
            (1, 10, 1, 12),
            (1, 11, 1, 12),
            (2, 20, 2, 25),
            (3, 30, None, None),
        ]
    )


def test_a_condition_no_pair_satisfies_leaves_every_row_unmatched(sides):
    left, right = sides

    rows = _rows(_join(left, right, (col("k") == col("k2")) & (col("rv") < col("lv") - 100), "left"))

    assert rows == _ordered([(1, 10, None, None), (1, 11, None, None), (2, 20, None, None), (3, 30, None, None)])


def test_key_equality_alone_is_unaffected(sides):
    left, right = sides

    rows = _rows(_join(left, right, col("k") == col("k2"), "inner"))

    assert rows == _ordered([(1, 10, 1, 5), (1, 10, 1, 12), (1, 11, 1, 5), (1, 11, 1, 12), (2, 20, 2, 25)])
