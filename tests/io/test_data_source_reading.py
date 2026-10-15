"""How the engine reads a Python data source's batches.

A source may wait for its next batch, for as long as it needs, without holding
the engine's IO workers: more sources than there are workers can wait at once,
and every other read keeps moving. A source may read synchronously or
asynchronously, and an asynchronous one runs only a bounded way ahead of the
engine reading it.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import textwrap
import time
from collections.abc import AsyncIterator, Callable, Iterator

import pyarrow as pa
import pytest

import daft
from daft.io import DataSource, DataSourceTask
from daft.io.pushdowns import Pushdowns
from daft.recordbatch import RecordBatch
from daft.schema import Schema
from tests.conftest import get_tests_daft_runner_name

SCHEMA = pa.schema([pa.field("x", pa.int64())])

# Twice the most IO workers the engine starts, so that waiting sources would hold
# every worker if each held one.
SOURCES_WAITING_TOGETHER = 16

# How long a source waits for the others before failing, in seconds.
PATIENCE = 30

Batches = Iterator[RecordBatch] | AsyncIterator[RecordBatch]

# These read one process's IO workers and buffers, with sources sharing that
# process's state, so they run on the native runner only.
native_only = pytest.mark.skipif(
    get_tests_daft_runner_name() != "native", reason="reads one process's IO workers and buffers"
)


def _batch(values: list[int]) -> RecordBatch:
    return RecordBatch.from_arrow_record_batches([pa.record_batch([pa.array(values)], schema=SCHEMA)], SCHEMA)


class _Source(DataSource):
    """One task, whose batches come from ``read``."""

    def __init__(self, read: Callable[[], Batches]) -> None:
        self._read = read

    @property
    def name(self) -> str:
        return "test"

    @property
    def schema(self) -> Schema:
        return Schema.from_pyarrow_schema(SCHEMA)

    async def get_tasks(self, pushdowns: Pushdowns) -> AsyncIterator[DataSourceTask]:
        yield _Task(self._read)


class _Task(DataSourceTask):
    def __init__(self, read: Callable[[], Batches]) -> None:
        self._read = read

    @property
    def schema(self) -> Schema:
        return Schema.from_pyarrow_schema(SCHEMA)

    def read(self) -> Batches:
        return self._read()


def _synchronously() -> Iterator[RecordBatch]:
    for values in ([1, 2], [3], [4, 5, 6]):
        yield _batch(values)


async def _asynchronously() -> AsyncIterator[RecordBatch]:
    for values in ([1, 2], [3], [4, 5, 6]):
        await asyncio.sleep(0)
        yield _batch(values)


@pytest.mark.parametrize("reading", [_synchronously, _asynchronously], ids=["sync", "async"])
def test_a_source_reads_its_batches(reading: Callable[[], Batches]) -> None:
    rows = _Source(reading).read().sort("x").to_pydict()

    assert rows == {"x": [1, 2, 3, 4, 5, 6]}


def _failing_synchronously() -> Iterator[RecordBatch]:
    yield _batch([1])
    raise ValueError("the source broke")


async def _failing_asynchronously() -> AsyncIterator[RecordBatch]:
    yield _batch([1])
    raise ValueError("the source broke")


@pytest.mark.parametrize("reading", [_failing_synchronously, _failing_asynchronously], ids=["sync", "async"])
def test_a_source_failing_fails_its_query_with_its_exception(reading: Callable[[], Batches]) -> None:
    with pytest.raises(ValueError, match="the source broke"):
        _Source(reading).read().to_pydict()


def test_a_source_yielding_something_else_fails_its_query() -> None:
    def other() -> Iterator[RecordBatch]:
        yield pa.record_batch([pa.array([1])], schema=SCHEMA)  # type: ignore[misc]

    with pytest.raises(AttributeError):
        _Source(other).read().to_pydict()


# Sources that wait together, each in a query of its own, run in a fresh
# interpreter: an engine whose waiting sources hold its IO workers never ends
# them, which must fail the case rather than hang the test run.
WAITING = textwrap.dedent(
    """
    import asyncio, json, sys, threading, time
    import pyarrow as pa
    import daft
    from daft.io.source import DataSource, DataSourceTask
    from daft.recordbatch import RecordBatch
    from daft.schema import Schema

    how, sources, patience = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    SCHEMA = pa.schema([pa.field("x", pa.int64())])
    together = threading.Barrier(sources, timeout=patience)
    released = threading.Event()
    state = {"arrived": 0, "all": None}

    def batch():
        return RecordBatch.from_arrow_record_batches(
            [pa.record_batch([pa.array([1])], schema=SCHEMA)], SCHEMA
        )

    class Task(DataSourceTask):
        schema = Schema.from_pyarrow_schema(SCHEMA)

        def read(self):
            if how == "async":
                return self.read_async()
            return self.read_sync()

        def read_sync(self):
            if how == "sync":
                together.wait()
            else:
                released.wait(patience)
            yield batch()

        async def read_async(self):
            # Every asynchronous source runs on one event loop, so the event is shared.
            if state["all"] is None:
                state["all"] = asyncio.Event()
            state["arrived"] += 1
            if state["arrived"] == sources:
                state["all"].set()
            await asyncio.wait_for(state["all"].wait(), patience)
            yield batch()

    class Source(DataSource):
        name = "waiting"
        schema = Schema.from_pyarrow_schema(SCHEMA)

        async def get_tasks(self, pushdowns):
            yield Task()

    answers = [None] * sources

    def run(index):
        try:
            answers[index] = Source().read().to_pydict()["x"]
        except Exception as failure:
            answers[index] = f"{type(failure).__name__}: {failure}"

    runners = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(sources)]
    for runner in runners:
        runner.start()
    beside = None
    if how == "beside":
        started = time.monotonic()
        daft.from_pydict({"x": [1, 2, 3]}).where(daft.col("x") > 1).to_pydict()
        beside = time.monotonic() - started
        released.set()
    for runner in runners:
        runner.join()
    print(json.dumps({"answers": answers, "beside": beside}))
    """
)


def _waiting(how: str) -> dict[str, list[object] | float | None]:
    ran = subprocess.run(
        [sys.executable, "-c", WAITING, how, str(SOURCES_WAITING_TOGETHER), str(PATIENCE)],
        capture_output=True,
        text=True,
        timeout=4 * PATIENCE,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr[-2000:]
    return json.loads(ran.stdout.strip().splitlines()[-1])


@native_only
@pytest.mark.parametrize("how", ["sync", "async"])
def test_more_sources_than_io_workers_wait_together(how: str) -> None:
    waited = _waiting(how)

    assert waited["answers"] == [[1]] * SOURCES_WAITING_TOGETHER


@native_only
def test_other_reads_move_while_sources_wait() -> None:
    waited = _waiting("beside")

    assert waited["answers"] == [[1]] * SOURCES_WAITING_TOGETHER
    assert isinstance(waited["beside"], float)
    assert waited["beside"] < PATIENCE / 2


@native_only
def test_an_asynchronous_source_runs_a_bounded_way_ahead_of_its_reader() -> None:
    # Batches of 8 MB each, so the source's own bound holds it to a few.
    batches, rows = 60, 1_000_000
    batch = _batch(list(range(rows)))
    produced = 0
    leads: list[int] = []

    async def counted() -> AsyncIterator[RecordBatch]:
        nonlocal produced
        for _ in range(batches):
            produced += 1
            yield batch

    @daft.func.batch(return_dtype=daft.DataType.int64())
    def slowly(x: daft.Series) -> daft.Series:
        # Batches the source has made but the reader has not yet reached.
        leads.append(produced - (len(leads) + 1))
        time.sleep(0.02)
        return x

    total = _Source(counted).read().select(slowly(daft.col("x")).alias("x")).count_rows()

    assert total == batches * rows
    # The engine holds a few batches of its own between the source and the
    # function; an unbounded source would be read whole, all 60 ahead.
    assert max(leads) < batches // 2, leads
