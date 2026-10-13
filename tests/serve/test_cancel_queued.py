"""Cancelling a query while it waits for an execution slot.

A queued query is cancellable by its id from the moment it is accepted: the
cancel is accepted, the query's client is answered at once, and the query
never runs. Ownership is unchanged, so another caller cannot cancel it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

import daft
from daft import col
from daft.runners.remote_runner import RemoteRunner
from daft.serve import ServeSettings, TenantSpec, start_server

if TYPE_CHECKING:
    import pathlib

    from daft.daft import DaftServeClient, DaftServeQueryResult, DaftServeServer


@pytest.fixture
def one_slot_server() -> Iterator[DaftServeServer]:
    server = start_server(
        ServeSettings(
            host="127.0.0.1",
            port=0,
            queue_timeout_secs=30,
            tenants=(
                TenantSpec(name="alpha", token="tok-alpha", max_concurrent_queries=1, queue_timeout_secs=30),
                TenantSpec(name="beta", token="tok-beta"),
            ),
        )
    )
    yield server
    server.shutdown(drain_timeout_secs=10)


def _holding_df(tmp_path: pathlib.Path) -> daft.DataFrame:
    """A query that holds its slot for several seconds."""
    path = str(tmp_path / "hold.parquet")
    daft.from_pydict({"v": list(range(16))}).write_parquet(path, write_mode="overwrite")

    @daft.func(return_dtype=daft.DataType.int64())
    def crawl(v: int) -> int:
        time.sleep(0.3)
        return v

    return daft.read_parquet(path).select(crawl(col("v")).alias("v"))


def _marking_df(tmp_path: pathlib.Path, marker: pathlib.Path) -> daft.DataFrame:
    """A query that leaves `marker` behind if it ever runs."""
    path = str(tmp_path / "mark.parquet")
    daft.from_pydict({"v": [1]}).write_parquet(path, write_mode="overwrite")
    marker_path = str(marker)

    @daft.func(return_dtype=daft.DataType.int64())
    def mark(v: int) -> int:
        with open(marker_path, "w") as written:
            written.write("ran")
        return v

    return daft.read_parquet(path).select(mark(col("v")).alias("v"))


def _tiny_df(tmp_path: pathlib.Path) -> daft.DataFrame:
    path = str(tmp_path / "tiny.parquet")
    daft.from_pydict({"v": [1]}).write_parquet(path, write_mode="overwrite")
    return daft.read_parquet(path)


def _queue(
    client: DaftServeClient, df: daft.DataFrame, query_id: str
) -> tuple[threading.Thread, dict[str, BaseException | DaftServeQueryResult]]:
    """Submit `df` from a thread; the call waits until the query is admitted or refused."""
    outcome: dict[str, BaseException | DaftServeQueryResult] = {}

    def submit() -> None:
        try:
            outcome["result"] = client.run_plan(df._builder._builder, {}, query_id, results_buffer_size=1)
        except Exception as raised:
            outcome["error"] = raised

    thread = threading.Thread(target=submit, daemon=True)
    thread.start()
    # Give the query time to reach the server's queue.
    time.sleep(0.5)
    return thread, outcome


def test_a_queued_query_is_cancelled_and_never_runs(one_slot_server: DaftServeServer, tmp_path: pathlib.Path) -> None:
    alpha = RemoteRunner(one_slot_server.address(), token="tok-alpha").client
    marker = tmp_path / "ran"
    holder = alpha.run_plan(_holding_df(tmp_path)._builder._builder, {}, "holder", results_buffer_size=1)
    thread, outcome = _queue(alpha, _marking_df(tmp_path, marker), "queued")

    accepted = RemoteRunner(one_slot_server.address(), token="tok-alpha").client.cancel_query("queued")
    thread.join(timeout=10)

    assert accepted is True
    assert "result" not in outcome
    assert "cancelled" in str(outcome["error"]).lower()
    del holder
    time.sleep(0.5)
    assert not marker.exists()


def test_another_caller_cannot_cancel_a_queued_query(one_slot_server: DaftServeServer, tmp_path: pathlib.Path) -> None:
    alpha = RemoteRunner(one_slot_server.address(), token="tok-alpha").client
    holder = alpha.run_plan(_holding_df(tmp_path)._builder._builder, {}, "holder", results_buffer_size=1)
    thread, outcome = _queue(alpha, _tiny_df(tmp_path), "queued")

    accepted = RemoteRunner(one_slot_server.address(), token="tok-beta").client.cancel_query("queued")

    assert accepted is False
    del holder
    thread.join(timeout=30)
    assert "result" in outcome


def test_cancelling_a_finished_query_answers_false(one_slot_server: DaftServeServer, tmp_path: pathlib.Path) -> None:
    alpha = RemoteRunner(one_slot_server.address(), token="tok-alpha").client
    result = alpha.run_plan(_tiny_df(tmp_path)._builder._builder, {}, "done", results_buffer_size=1)
    list(result)

    assert alpha.cancel_query("done") is False
