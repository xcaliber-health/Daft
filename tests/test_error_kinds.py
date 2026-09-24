from __future__ import annotations

from collections.abc import Callable

import pytest

import daft
from daft import DataType, Window, col
from daft.exceptions import (
    DaftCardinalityError,
    DaftComputeError,
    DaftCoreException,
    DaftFieldNotFoundError,
    DaftTypeError,
    DaftValueError,
)
from daft.functions import first_value, single_value, year


def _frame() -> daft.DataFrame:
    return daft.from_pydict({"g": [1, 1], "o": [1, 2], "a": [1, 2], "s": ["x", "y"]})


@pytest.mark.parametrize(
    ("fail", "kind"),
    [
        pytest.param(
            lambda df: df.select(col("missing")), DaftFieldNotFoundError, id="column not found while planning"
        ),
        pytest.param(lambda df: df.select(year(col("s"))), DaftTypeError, id="wrong type while planning"),
        pytest.param(
            lambda df: df.select(first_value(col("a")).over(Window().partition_by("g").order_by("o"))).collect(),
            DaftValueError,
            id="refused value while optimizing",
        ),
        pytest.param(
            lambda df: df.select(col("s").cast(DataType.struct({"x": DataType.int64()}))).collect(),
            DaftTypeError,
            id="wrong type while running",
        ),
        pytest.param(
            lambda df: df.groupby("g").agg(single_value(col("a"))).collect(),
            DaftCardinalityError,
            id="several rows where one was required",
        ),
    ],
)
def test_a_failure_arrives_as_its_kind(
    fail: Callable[[daft.DataFrame], daft.DataFrame], kind: type[DaftCoreException]
) -> None:
    with pytest.raises(kind):
        fail(_frame())


@pytest.mark.parametrize(
    "kind",
    [DaftFieldNotFoundError, DaftTypeError, DaftValueError, DaftCardinalityError],
)
def test_every_kind_is_still_a_core_exception_and_a_value_error(kind: type[DaftCoreException]) -> None:
    assert issubclass(kind, DaftCoreException)
    assert issubclass(kind, ValueError)


def test_a_failure_keeps_its_message() -> None:
    with pytest.raises(DaftFieldNotFoundError, match="missing"):
        _frame().select(col("missing"))


def test_a_cardinality_failure_is_a_compute_failure() -> None:
    assert issubclass(DaftCardinalityError, DaftComputeError)
