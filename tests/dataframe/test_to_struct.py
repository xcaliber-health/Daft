from __future__ import annotations

from collections.abc import Callable

import pytest

import daft
import daft.exceptions
from daft import Expression, col
from daft.functions import to_struct


def test_to_struct_empty_structs():
    df = daft.from_pydict({"a": [1, 2, 3]})

    with pytest.raises(daft.exceptions.DaftCoreException, match="Cannot call struct with no inputs"):
        df.select(to_struct()).collect()


# there was a bug with pushdowns onto to_struct previously
def test_to_struct_pushdown():
    df = daft.from_pydict(
        {
            "a": [1, 2, 3, 4, None, 6, None],
            "b": ["a", "b", "c", "", "e", None, None],
        }
    )
    df = df.select(to_struct(col("a"), col("b")))
    df = df.select(df["struct"].get("a"))
    assert df.to_pydict() == {"a": [1, 2, 3, 4, None, 6, None]}


def test_to_struct_with_literal_materialized():
    df = daft.from_pydict({"a": [1, 2, 3]})
    df = df.with_column("b", daft.lit("hello"))
    result = df.select(to_struct(col("a"), col("b"))).to_pydict()
    assert result["struct"] == [
        {"a": 1, "b": "hello"},
        {"a": 2, "b": "hello"},
        {"a": 3, "b": "hello"},
    ]


def test_to_struct_with_literal_inline():
    df = daft.from_pydict({"a": [1, 2, 3]})
    result = df.select(to_struct(col("a"), daft.lit("hello").alias("b"))).to_pydict()
    assert result["struct"] == [
        {"a": 1, "b": "hello"},
        {"a": 2, "b": "hello"},
        {"a": 3, "b": "hello"},
    ]


def test_to_struct_empty_partition_with_literal():
    df = daft.from_pydict({"a": []})
    result = df.select(to_struct(col("a"), daft.lit("hello").alias("b"))).to_pydict()
    assert result["struct"] == []


@pytest.mark.parametrize(
    "parts",
    [
        pytest.param(lambda: [col("v") + 1, col("v") * 2], id="derived from one column"),
        pytest.param(lambda: [col("v").alias("x"), (col("v") * 2).alias("x")], id="aliased alike"),
        pytest.param(lambda: [daft.lit(1), daft.lit(2)], id="two constants"),
    ],
)
def test_to_struct_refuses_two_fields_of_one_name(parts: Callable[[], list[Expression]]) -> None:
    df = daft.from_pydict({"v": [2]})

    with pytest.raises(daft.exceptions.DaftCoreException, match="two fields named"):
        df.select(to_struct(*parts()).alias("s")).collect()


def test_to_struct_keeps_every_field_named_apart() -> None:
    df = daft.from_pydict({"v": [2]})

    result = df.select(to_struct(col("v").alias("x"), (col("v") * 2).alias("y")).alias("s")).to_pydict()

    assert result == {"s": [{"x": 2, "y": 4}]}
