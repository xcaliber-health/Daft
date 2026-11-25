"""A part of a plan written twice is computed once, with the answers it gives computed twice.

The native runner finds a part that does real work (a join, aggregation, window or sort)
written more than once over the same source, runs it once, and hands its rows to each
reader. The distributed runner plans each occurrence on its own.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator

import duckdb
import pyarrow as pa
import pytest

import daft
from daft import DataFrame, Window, col, lit
from daft.context import execution_config_ctx
from daft.functions import row_number
from daft.io.pushdowns import Pushdowns
from daft.io.source import DataSource, DataSourceTask
from daft.recordbatch import RecordBatch
from daft.schema import Schema
from tests.conftest import get_tests_daft_runner_name

BATCHES = 8
ROWS_PER_BATCH = 500
GROUPS = 50

native_only = pytest.mark.skipif(
    get_tests_daft_runner_name() != "native", reason="only the native runner shares a part"
)


class _Batch(DataSourceTask):
    def __init__(self, source: CountingSource, index: int) -> None:
        self._source = source
        self._index = index

    @property
    def schema(self) -> Schema:
        return self._source.schema

    def read(self) -> Iterator[RecordBatch]:
        self._source.batches_read += 1
        first = self._index * ROWS_PER_BATCH
        values = list(range(first, first + ROWS_PER_BATCH))
        yield RecordBatch.from_arrow_table(
            pa.table({"k": [v % GROUPS for v in values], "v": values, "t": [v % 7 for v in values]})
        )


class CountingSource(DataSource):
    """A source of `BATCHES` batches that counts how many it hands out."""

    def __init__(self) -> None:
        self.batches_read = 0

    @property
    def name(self) -> str:
        return "counting"

    @property
    def schema(self) -> Schema:
        return Schema.from_pyarrow_schema(pa.schema([("k", pa.int64()), ("v", pa.int64()), ("t", pa.int64())]))

    async def get_tasks(self, pushdowns: Pushdowns) -> AsyncIterator[DataSourceTask]:
        del pushdowns  # Every batch is handed out whole; the engine filters it.
        for index in range(BATCHES):
            yield _Batch(self, index)


def sum_by_k(rows: DataFrame) -> DataFrame:
    return rows.groupby("k").agg(col("v").sum().alias("s"))


def repro(rows: DataFrame, reader_filter: bool, same_object: bool) -> DataFrame:
    """An aggregation read on both sides of a join, one side filtered on the group key."""
    first = sum_by_k(rows)
    second = first if same_object else sum_by_k(rows)
    left = first.where(col("k") % lit(2) == lit(1)) if reader_filter else first
    return left.join(second, on="k", suffix="_all")


def count_beside_choice(rows: DataFrame) -> DataFrame:
    """A count of the records of a join, beside the first record of each key, joined back."""
    dim = rows.select(col("k").alias("k2"), col("t").alias("t2"))

    def matched() -> DataFrame:
        return rows.join(dim, left_on="k", right_on="k2").where(col("t") == col("t2"))

    counts = matched().groupby("k").agg(col("v").count().alias("n"))
    first = (
        matched()
        .with_column("rn", row_number().over(Window().partition_by("k").order_by("v")))
        .where(col("rn") == 1)
        .select("k", col("v").alias("first_v"))
    )
    return counts.join(first, on="k")


def concat_of_aggregates(rows: DataFrame) -> DataFrame:
    return sum_by_k(rows).concat(sum_by_k(rows).with_column("s", col("s") + 1))


def top_rows(rows: DataFrame) -> DataFrame:
    ordered = lambda: rows.sort("v", desc=True)  # noqa: E731
    return ordered().limit(3).concat(ordered().limit(5))


def ranked(rows: DataFrame) -> DataFrame:
    """A window over the part, read by two aggregations joined back."""

    def numbered() -> DataFrame:
        return rows.with_column("r", row_number().over(Window().partition_by("k").order_by("v")))

    firsts = numbered().where(col("r") == 1).groupby("k").agg(col("v").min().alias("a"))
    lasts = numbered().groupby("k").agg(col("r").max().alias("b"))
    return firsts.join(lasts, on="k")


def rows_of(df: DataFrame) -> list[tuple[object, ...]]:
    table = df.to_pydict()
    return sorted(zip(*table.values()))


def run(build: Callable[[DataFrame], DataFrame], **config: object) -> tuple[list[tuple[object, ...]], int]:
    """The rows `build` gives over a fresh counting source, and how many batches it read."""
    source = CountingSource()
    with execution_config_ctx(**config):
        answer = rows_of(build(source.read()))
    return answer, source.batches_read


SHAPES = [
    pytest.param(lambda rows: repro(rows, True, False), id="repro, two builds"),
    pytest.param(lambda rows: repro(rows, True, True), id="repro, one object"),
    pytest.param(lambda rows: repro(rows, False, False), id="repro, no reader filter"),
    pytest.param(count_beside_choice, id="count beside a choice"),
    pytest.param(concat_of_aggregates, id="concat of aggregations"),
    pytest.param(top_rows, id="limits over a sort"),
    pytest.param(ranked, id="window read by aggregations"),
]


@pytest.mark.parametrize("build", SHAPES)
def test_a_shared_part_answers_as_computing_it_twice_does(build: Callable[[DataFrame], DataFrame]) -> None:
    shared, _ = run(build)
    separate, _ = run(build, disable_subplan_sharing=True)

    assert shared == separate


@native_only
@pytest.mark.parametrize("build", SHAPES)
def test_a_shared_part_reads_its_source_half_as_often(build: Callable[[DataFrame], DataFrame]) -> None:
    _, shared = run(build)
    _, separate = run(build, disable_subplan_sharing=True)

    assert shared * 2 == separate


def test_the_repro_answers_as_another_engine_does() -> None:
    answer, _ = run(lambda rows: repro(rows, True, False))
    values = list(range(BATCHES * ROWS_PER_BATCH))
    connection = duckdb.connect()
    connection.register("records", pa.table({"k": [v % GROUPS for v in values], "v": values}))
    expected = connection.sql(
        """
        with part as (select k, sum(v) as s from records group by k)
        select a.k, a.s, b.s as s_all from part a join part b using (k) where a.k % 2 = 1
        """
    ).fetchall()

    assert answer == sorted(expected)


@native_only
def test_readers_that_drain_the_part_never_spill(tmp_path_factory: pytest.TempPathFactory) -> None:
    # A spill would need the scratch directory, which does not exist.
    missing = str(tmp_path_factory.mktemp("spill") / "missing" / "dir")
    answer, batches = run(count_beside_choice, spill_dirs=[missing])
    separate, separate_batches = run(count_beside_choice, disable_subplan_sharing=True)

    assert answer == separate
    assert batches * 2 == separate_batches


@native_only
@pytest.mark.parametrize("build", [SHAPES[0], SHAPES[3]])
def test_a_waiting_reader_spills_past_a_small_buffer_and_answers_alike(
    build: Callable[[DataFrame], DataFrame],
) -> None:
    shared, batches = run(build, shared_part_buffer_bytes=1)
    separate, separate_batches = run(build, disable_subplan_sharing=True)

    assert shared == separate
    assert batches * 2 == separate_batches


@native_only
def test_readers_that_stop_early_read_no_more_than_one_reader_alone() -> None:
    def joined(rows: DataFrame) -> DataFrame:
        return rows.join(rows.select(col("k").alias("k2")).distinct(), left_on="k", right_on="k2")

    answer, shared = run(lambda rows: joined(rows).limit(3).concat(joined(rows).limit(5)))
    _, alone = run(lambda rows: joined(rows).limit(5))

    assert len(answer) == 8
    assert shared <= alone


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda rows: rows.where(col("v") > 10).join(rows.where(col("v") > 10), on="k", suffix="_r"),
            id="a filtered scan",
        ),
        pytest.param(
            lambda rows: sum_by_k(rows.where(col("v") > 10)).join(sum_by_k(rows.where(col("v") > 20)), on="k"),
            id="parts that differ",
        ),
        pytest.param(
            lambda rows: sum_by_k(rows.with_column("v", _python_plus_one(col("v")))).join(
                sum_by_k(rows.with_column("v", _python_plus_one(col("v")))), on="k", suffix="_r"
            ),
            id="a python function",
        ),
    ],
)
def test_parts_that_are_not_shared_read_their_source_per_occurrence(
    build: Callable[[DataFrame], DataFrame],
) -> None:
    shared, batches = run(build)
    separate, separate_batches = run(build, disable_subplan_sharing=True)

    assert shared == separate
    assert batches == separate_batches


@daft.func(return_dtype=daft.DataType.int64())
def _python_plus_one(v: int) -> int:
    return v + 1


@native_only
def test_explain_shows_the_shared_part(capsys: pytest.CaptureFixture[str]) -> None:
    repro(CountingSource().read(), True, False).explain(show_all=True)

    assert "Shared #0" in capsys.readouterr().out


def test_the_distributed_plan_shares_nothing() -> None:
    plan = repro(CountingSource().read(), True, False)._builder

    distributed = plan.optimize(daft.context.get_context().daft_execution_config)

    assert "Shared" not in distributed.pretty_print()


def test_sharing_is_off_when_spilling_is() -> None:
    plan = repro(CountingSource().read(), True, False)._builder
    config = daft.context.get_context().daft_execution_config.with_config_values(enable_spilling=False)

    assert "Shared" not in plan.optimize(config, local=True).pretty_print()


@native_only
def test_a_frame_held_in_memory_is_shared_across_its_reads() -> None:
    # Each read of a materialized frame plans a new scan of the same partitions.
    rows = daft.from_pydict({"k": [n % 10 for n in range(1_000)], "v": list(range(1_000))})
    plan = repro(rows, True, False)._builder

    optimized = plan.optimize(daft.context.get_context().daft_execution_config, local=True)

    assert optimized.pretty_print().count("Shared #0") == 2
