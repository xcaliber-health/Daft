"""A membership test over a list of values answers the same however it is run.

A list of literal values is built once per expression and reused for every
batch, so the answer must not depend on how the input is split into batches,
on which column the same expression is later applied to, on whether the
expression was sent to another process, or on whether statistics let a scan
skip row groups.
"""

from __future__ import annotations

import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import daft
from daft import col
from daft.pickle import dumps, loads

# Lists both shorter and longer than the 64 values statistics judge one by one.
values_lists = st.lists(st.integers(min_value=-60, max_value=60), max_size=80)
column_values = st.lists(st.one_of(st.none(), st.integers(min_value=-50, max_value=50)), max_size=300)


def _membership(values: list[int | None], items: list[int]) -> list[bool | None]:
    """Whether each value is listed: unknown for a null, except that nothing is in an empty list."""
    if not items:
        return [False] * len(values)
    listed = set(items)
    return [None if value is None else value in listed for value in values]


@settings(max_examples=int(os.getenv("HYPOTHESIS_MAX_EXAMPLES", "60")), deadline=None)
@given(values=column_values, items=values_lists, batch_rows=st.sampled_from([1, 7, 64, 4096]))
def test_membership_does_not_depend_on_how_the_input_is_batched(
    values: list[int | None], items: list[int], batch_rows: int
) -> None:
    frame = daft.from_pydict({"x": pa.array(values, pa.int64())})

    with daft.execution_config_ctx(default_morsel_size=batch_rows):
        tested = frame.select(col("x").is_in(items).alias("m")).to_pydict()["m"]
        kept = frame.where(col("x").is_in(items)).to_pydict()["x"]

    assert tested == _membership(values, items)
    assert kept == [value for value, member in zip(values, _membership(values, items)) if member]


@pytest.mark.parametrize("column_type", [pa.int8(), pa.int32(), pa.int64(), pa.uint16(), pa.float64()])
def test_one_expression_tests_columns_of_any_numeric_type(column_type: pa.DataType) -> None:
    membership = col("x").is_in(list(range(0, 200, 3)))
    first = daft.from_pydict({"x": pa.array(range(100), pa.int64())})
    second = daft.from_pydict({"x": pa.array(range(100), column_type)})

    first_answer = first.select(membership.alias("m")).to_pydict()["m"]
    second_answer = second.select(membership.alias("m")).to_pydict()["m"]

    assert first_answer == second_answer == [value % 3 == 0 for value in range(100)]


def test_an_expression_sent_to_another_process_tests_the_same_values() -> None:
    membership = col("x").is_in([1, 5, 9])
    frame = daft.from_pydict({"x": list(range(12))})
    frame.select(membership.alias("m")).collect()

    received = loads(dumps(membership))

    assert frame.select(received.alias("m")).to_pydict()["m"] == [v in (1, 5, 9) for v in range(12)]


@pytest.mark.parametrize("listed", [3, 64, 65, 1_000, 10_000])
@pytest.mark.parametrize("page_index", [False, True], ids=["row groups", "pages"])
def test_skipping_row_groups_by_statistics_never_drops_a_listed_row(tmp_path, listed: int, page_index: bool) -> None:
    rows = 50_000
    path = str(tmp_path / "sorted.parquet")
    pq.write_table(
        pa.table({"x": pa.array(range(rows), pa.int64()), "y": pa.array(range(rows, 0, -1), pa.int64())}),
        path,
        row_group_size=1_000,
        data_page_size=1_024,
        write_page_index=True,
    )
    # Spread across the file and past both ends of it.
    items = sorted({(i * 7_919) % (rows + 2_000) - 1_000 for i in range(listed)})

    with daft.execution_config_ctx(parquet_page_index=page_index):
        read = daft.read_parquet(path).where(col("x").is_in(items)).sort("x").to_pydict()
    in_memory = daft.from_arrow(pq.read_table(path)).where(col("x").is_in(items)).sort("x").to_pydict()

    assert read == in_memory
    assert read["x"] == [x for x in items if 0 <= x < rows]
