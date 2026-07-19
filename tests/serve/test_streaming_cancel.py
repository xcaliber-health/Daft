"""Streaming semantics: early stop, cancellation, and concurrency."""

from __future__ import annotations

import concurrent.futures
import time
from typing import TYPE_CHECKING

import pytest

import daft

from .conftest import collect_via

if TYPE_CHECKING:
    from daft.daft import DaftServeServer
    from daft.runners.remote_runner import RemoteRunner


def test_streamed_partitions_arrive_incrementally(remote_runner: RemoteRunner) -> None:
    df = daft.from_pydict({"a": list(range(1000))}).into_partitions(8)
    parts = list(remote_runner.run_iter_tables(df._builder))
    assert sum(len(p) for p in parts) == 1000


def test_abandoned_iterator_releases_server_slot(remote_runner: RemoteRunner, serve_server: DaftServeServer) -> None:
    df = daft.from_pydict({"a": list(range(1000))}).into_partitions(8)
    gen = remote_runner.run_iter_tables(df._builder)
    next(gen)
    gen.close()

    deadline = time.monotonic() + 10
    while serve_server.active_queries() > 0:
        assert time.monotonic() < deadline, "server did not release the abandoned query"
        time.sleep(0.05)


def test_cancel_unknown_query_is_noop(remote_runner: RemoteRunner) -> None:
    assert remote_runner.client.cancel_query("no-such-query") is False


def test_concurrent_queries_are_isolated(remote_runner: RemoteRunner) -> None:
    def run_one(n: int) -> list[int]:
        df = daft.from_pydict({"v": list(range(n))}).sum("v")
        return collect_via(remote_runner, df)["v"]

    sizes = [10, 100, 1000, 5000]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run_one, sizes))

    assert results == [[sum(range(n))] for n in sizes]


def test_failing_query_does_not_poison_the_server(remote_runner: RemoteRunner) -> None:
    @daft.func(return_dtype=daft.DataType.int64())
    def boom(v: int) -> int:
        raise RuntimeError("intentional failure")

    bad = daft.from_pydict({"v": [1, 2, 3]}).select(boom(daft.col("v")))
    with pytest.raises(Exception, match="intentional failure"):
        collect_via(remote_runner, bad)

    good = daft.from_pydict({"v": [1, 2, 3]})
    assert collect_via(remote_runner, good) == {"v": [1, 2, 3]}


def test_registry_is_empty_after_queries_finish(remote_runner: RemoteRunner, serve_server: DaftServeServer) -> None:
    df = daft.from_pydict({"a": [1, 2, 3]})
    collect_via(remote_runner, df)
    deadline = time.monotonic() + 10
    while serve_server.active_queries() > 0:
        assert time.monotonic() < deadline, "query registry leaked entries"
        time.sleep(0.05)
