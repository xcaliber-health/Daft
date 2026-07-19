"""Streaming under pressure: slow consumers, large results, slot release.

Exercises the bounded result buffer end to end — a consumer that drains
slowly must not wedge the server, and abandoning a long computation must
free its execution slot for the next query.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest

import daft
from daft.runners.remote_runner import RemoteRunner
from daft.serve import ServeSettings, start_server

from .conftest import collect_via


@pytest.fixture()
def single_slot_runner() -> Iterator[RemoteRunner]:
    """A runner against a dedicated single-slot server with a short queue."""
    server = start_server(ServeSettings(host="127.0.0.1", port=0, max_concurrent_queries=1, queue_timeout_secs=5))
    yield RemoteRunner(server.address())
    server.shutdown(drain_timeout_secs=10)


def test_large_result_with_minimal_buffer_completes(remote_runner: RemoteRunner) -> None:
    n = 2_000_000
    df = daft.from_pydict({"v": list(range(n))}).into_partitions(8)

    total_rows = 0
    for part in remote_runner.run_iter_tables(df._builder, results_buffer_size=1):
        total_rows += len(part)
    assert total_rows == n


def test_slow_consumer_does_not_wedge_the_stream(remote_runner: RemoteRunner) -> None:
    df = daft.from_pydict({"v": list(range(100_000))}).into_partitions(8)

    seen = 0
    for part in remote_runner.run_iter_tables(df._builder, results_buffer_size=1):
        seen += len(part)
        time.sleep(0.05)
    assert seen == 100_000


def test_abandoned_long_query_frees_the_slot(single_slot_runner: RemoteRunner) -> None:
    """Dropping a result stream mid-execution must release its slot."""

    @daft.func(return_dtype=daft.DataType.int64())
    def slow(v: int) -> int:
        time.sleep(0.05)
        return v

    from daft import col

    slow_df = daft.from_pydict({"v": list(range(200))}).into_partitions(8).select(slow(col("v")))
    stream = single_slot_runner.run_iter_tables(slow_df._builder, results_buffer_size=1)
    next(stream)
    stream.close()

    # With max_concurrent_queries=1 and a 5s queue timeout, this only
    # succeeds if the abandoned query's slot was actually released.
    quick = daft.from_pydict({"a": [1, 2, 3]})
    assert collect_via(single_slot_runner, quick) == {"a": [1, 2, 3]}
