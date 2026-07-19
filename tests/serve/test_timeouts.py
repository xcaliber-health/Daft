"""Per-query execution time limits.

A server configured with a query timeout must cancel overrunning queries
with a clear, typed error and release their execution slots.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest

import daft
from daft import col
from daft.runners.remote_runner import RemoteRunner
from daft.serve import ServeSettings, start_server

from .conftest import collect_via


@pytest.fixture()
def timeout_runner() -> Iterator[RemoteRunner]:
    """A runner against a single-slot server with a 2-second query limit."""
    server = start_server(ServeSettings(host="127.0.0.1", port=0, max_concurrent_queries=1, query_timeout_secs=2))
    yield RemoteRunner(server.address())
    server.shutdown(drain_timeout_secs=10)


def _slow_df() -> daft.DataFrame:
    @daft.func(return_dtype=daft.DataType.int64())
    def crawl(v: int) -> int:
        time.sleep(0.5)
        return v

    return daft.from_pydict({"v": list(range(64))}).into_partitions(8).select(crawl(col("v")))


def test_overrunning_query_is_cancelled_with_timeout_error(timeout_runner: RemoteRunner) -> None:
    with pytest.raises(Exception, match="(?i)timeout|time limit"):
        collect_via(timeout_runner, _slow_df())


def test_slot_is_released_after_timeout(timeout_runner: RemoteRunner) -> None:
    with pytest.raises(Exception, match="(?i)timeout|time limit"):
        collect_via(timeout_runner, _slow_df())

    # The single slot must be free again for a fast query.
    quick = daft.from_pydict({"a": [1, 2, 3]})
    assert collect_via(timeout_runner, quick) == {"a": [1, 2, 3]}


def test_fast_query_unaffected_by_timeout_config(timeout_runner: RemoteRunner) -> None:
    quick = daft.from_pydict({"a": [1, 2]}).select((col("a") * 2).alias("a"))
    assert collect_via(timeout_runner, quick) == {"a": [2, 4]}
