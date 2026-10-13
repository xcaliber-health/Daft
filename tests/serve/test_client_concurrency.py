"""One client connection shared by many threads.

A query waiting for an execution slot must not hold up the same client's
other calls: another query, a cancel, or a question to the server.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import pytest

import daft
from daft import col
from daft.runners.remote_runner import RemoteRunner
from daft.serve import ServeSettings, start_server

if TYPE_CHECKING:
    import pathlib

    from daft.daft import DaftServeClient, DaftServeServer

# A held query runs for about ROWS * DELAY_SECS seconds; every call under test
# must return well inside that window.
ROWS = 16
DELAY_SECS = 0.3
PROMPT_SECS = 2.0


@pytest.fixture
def one_slot_server() -> Iterator[DaftServeServer]:
    server = start_server(ServeSettings(host="127.0.0.1", port=0, max_concurrent_queries=1, queue_timeout_secs=30))
    yield server
    server.shutdown(drain_timeout_secs=10)


def _slow_df(tmp_path: pathlib.Path) -> daft.DataFrame:
    path = str(tmp_path / "slow.parquet")
    daft.from_pydict({"v": list(range(ROWS))}).write_parquet(path, write_mode="overwrite")

    @daft.func(return_dtype=daft.DataType.int64())
    def crawl(v: int) -> int:
        time.sleep(DELAY_SECS)
        return v

    return daft.read_parquet(path).select(crawl(col("v")).alias("v"))


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda client: client.server_info(), id="server info"),
        pytest.param(lambda client: client.health(), id="health"),
        pytest.param(lambda client: client.cancel_query("no-such-query"), id="cancel"),
    ],
)
def test_a_call_returns_while_the_same_clients_query_waits_for_a_slot(
    one_slot_server: DaftServeServer,
    tmp_path: pathlib.Path,
    call: Callable[[DaftServeClient], object],
) -> None:
    client = RemoteRunner(one_slot_server.address()).client
    df = _slow_df(tmp_path)
    holder = client.run_plan(df._builder._builder, {}, "holder", results_buffer_size=1)

    queued_done = threading.Event()

    def submit_queued() -> None:
        queued = client.run_plan(df._builder._builder, {}, "queued", results_buffer_size=1)
        del queued
        queued_done.set()

    waiter = threading.Thread(target=submit_queued, daemon=True)
    waiter.start()
    # Give the second query time to reach the server's queue.
    time.sleep(0.5)
    assert not queued_done.is_set()

    started = time.monotonic()
    call(client)
    elapsed = time.monotonic() - started

    assert elapsed < PROMPT_SECS

    del holder
    waiter.join(timeout=30)
    assert queued_done.is_set()
