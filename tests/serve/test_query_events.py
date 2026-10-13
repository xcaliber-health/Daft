"""A served query reports its lifecycle to the server's subscribers.

A served query reports what the same query run in-process reports, under the
query id its client chose, plus one event of its own: that it was accepted and
queued for an execution slot. It reports its end on every exit, with the rows
and bytes handed to its reader.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import pytest

import daft
from daft import col
from daft.daft import QueryEndState
from daft.runners.remote_runner import RemoteRunner
from daft.serve import ServedRequest, ServeSettings, TenantSpec, start_server
from daft.subscribers import Subscriber
from daft.subscribers.events import (
    Event,
    ExecutionFinished,
    ExecutionStarted,
    OptimizationCompleted,
    OptimizationStarted,
    QueryFinished,
    QueryQueued,
    QueryStarted,
)

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Collection

    from daft.daft import DaftServeClient, DaftServeQueryResult, DaftServeServer

LIFECYCLE = (
    QueryQueued,
    QueryStarted,
    OptimizationStarted,
    OptimizationCompleted,
    ExecutionStarted,
    ExecutionFinished,
    QueryFinished,
)


class Recorder(Subscriber):
    """Keeps every lifecycle event it is sent, in arrival order."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self._lock = threading.Lock()

    def on_event(self, event: Event) -> None:
        if isinstance(event, LIFECYCLE):
            with self._lock:
                self.events.append(event)

    def of(self, query_id: str) -> list[Event]:
        with self._lock:
            return [event for event in self.events if getattr(event, "query_id", None) == query_id]

    def kinds(self, query_id: str) -> list[str]:
        return [type(event).__name__ for event in self.of(query_id)]

    def run_in_process(self, df: daft.DataFrame) -> str:
        """Collects `df` in-process and returns the id its query reported."""
        seen = len(self.events)
        df.collect()
        with self._lock:
            return next(event.query_id for event in self.events[seen:] if isinstance(event, QueryStarted))

    def finished(self, query_id: str, within_secs: float = 30) -> QueryFinished:
        """The query's end event, once the server has reported it."""
        deadline = time.monotonic() + within_secs
        while time.monotonic() < deadline:
            ended = [event for event in self.of(query_id) if isinstance(event, QueryFinished)]
            if ended:
                return ended[0]
            time.sleep(0.05)
        raise AssertionError(f"query {query_id} never reported its end: {self.kinds(query_id)}")


class EchoHost:
    """Answers a request with one row holding its body."""

    @property
    def kinds(self) -> Collection[str]:
        return ("echo",)

    def plan(self, request: ServedRequest) -> daft.DataFrame:
        return daft.from_pydict({"body": [request.body.decode()]})


@pytest.fixture
def recorder() -> Iterator[Recorder]:
    recording = Recorder()
    with daft.with_subscriber("served-events", recording):
        yield recording


@pytest.fixture
def server() -> Iterator[DaftServeServer]:
    serving = start_server(
        ServeSettings(
            host="127.0.0.1",
            port=0,
            tenants=(TenantSpec(name="svc", token="tok-svc", max_concurrent_queries=1, queue_timeout_secs=30),),
        ),
        request_host=EchoHost(),
    )
    yield serving
    serving.shutdown(drain_timeout_secs=10)


def _client(server: DaftServeServer, token: str | None = "tok-svc") -> DaftServeClient:
    return RemoteRunner(server.address(), token=token).client


def _written(tmp_path: pathlib.Path, rows: int = 3) -> str:
    path = str(tmp_path / f"table-{rows}.parquet")
    daft.from_pydict({"v": list(range(rows))}).write_parquet(path, write_mode="overwrite")
    return path


def _table(tmp_path: pathlib.Path, rows: int = 3) -> daft.DataFrame:
    return daft.read_parquet(_written(tmp_path, rows))


def _crawling(tmp_path: pathlib.Path, rows: int = 16) -> daft.DataFrame:
    """A query that runs for several seconds, one row at a time."""

    @daft.func(return_dtype=daft.DataType.int64())
    def crawl(v: int) -> int:
        time.sleep(0.3)
        return v

    return _table(tmp_path, rows).select(crawl(col("v")).alias("v"))


def _drain(result: DaftServeQueryResult) -> int:
    return sum(len(part) for part in result)


def test_a_served_query_reports_what_an_in_process_one_does_after_queueing(
    recorder: Recorder, server: DaftServeServer, tmp_path: pathlib.Path
) -> None:
    path = _written(tmp_path)
    in_process_id = recorder.run_in_process(daft.read_parquet(path).where(col("v") > 0))
    df = daft.read_parquet(path).where(col("v") > 0)

    _drain(_client(server).run_plan(df._builder._builder, {}, "q-served", results_buffer_size=1))
    recorder.finished("q-served")

    assert recorder.kinds("q-served") == ["QueryQueued", *recorder.kinds(in_process_id)]


@pytest.mark.parametrize(
    ("send", "lane", "kind"),
    [
        pytest.param(
            lambda client, df: client.run_plan(df._builder._builder, {}, "q-lane", results_buffer_size=1),
            "plan",
            None,
            id="plan",
        ),
        pytest.param(lambda client, _: client.run_sql("SELECT 1 AS one", "q-lane"), "sql", None, id="sql"),
        pytest.param(lambda client, _: client.run_request("echo", b"hi", "q-lane"), "request", "echo", id="request"),
    ],
)
def test_a_queued_event_names_its_lane_and_caller(
    recorder: Recorder,
    server: DaftServeServer,
    tmp_path: pathlib.Path,
    send: Callable[[DaftServeClient, daft.DataFrame], DaftServeQueryResult],
    lane: str,
    kind: str | None,
) -> None:
    _drain(send(_client(server), _table(tmp_path)))
    recorder.finished("q-lane")

    queued = recorder.of("q-lane")[0]
    assert isinstance(queued, QueryQueued)
    assert (queued.lane, queued.kind, queued.caller) == (lane, kind, "svc")


def test_the_caller_is_none_without_named_callers(recorder: Recorder, tmp_path: pathlib.Path) -> None:
    serving = start_server(ServeSettings(host="127.0.0.1", port=0, token="shared"))
    try:
        df = _table(tmp_path)
        _drain(_client(serving, "shared").run_plan(df._builder._builder, {}, "q-anon", results_buffer_size=1))
        recorder.finished("q-anon")
    finally:
        serving.shutdown(drain_timeout_secs=10)

    queued = recorder.of("q-anon")[0]
    assert isinstance(queued, QueryQueued)
    assert queued.caller is None


def test_a_served_query_reports_its_wait_and_what_it_returned(
    recorder: Recorder, server: DaftServeServer, tmp_path: pathlib.Path
) -> None:
    path = _written(tmp_path, rows=5)
    in_process_id = recorder.run_in_process(daft.read_parquet(path))
    df = daft.read_parquet(path)
    in_process = recorder.finished(in_process_id)

    rows = _drain(_client(server).run_plan(df._builder._builder, {}, "q-counts", results_buffer_size=1))
    served = recorder.finished("q-counts")
    started = next(event for event in recorder.of("q-counts") if isinstance(event, QueryStarted))

    assert started.metadata.queued_seconds is not None
    assert started.metadata.queued_seconds >= 0
    assert served.result.end_state == QueryEndState.Finished
    assert (served.result.rows_returned, served.result.bytes_returned) == (
        in_process.result.rows_returned,
        in_process.result.bytes_returned,
    )
    assert served.result.rows_returned == rows == 5


def test_waiting_for_a_slot_is_reported_as_queued_time(
    recorder: Recorder, server: DaftServeServer, tmp_path: pathlib.Path
) -> None:
    holder = _client(server).run_plan(
        _crawling(tmp_path, rows=4)._builder._builder, {}, "holder", results_buffer_size=1
    )
    waiting: dict[str, int] = {}

    def submit() -> None:
        df = _table(tmp_path)
        waiting["rows"] = _drain(_client(server).run_plan(df._builder._builder, {}, "q-waits", results_buffer_size=1))

    thread = threading.Thread(target=submit, daemon=True)
    thread.start()
    time.sleep(0.5)
    _drain(holder)
    thread.join(timeout=30)
    recorder.finished("q-waits")

    started = next(event for event in recorder.of("q-waits") if isinstance(event, QueryStarted))
    assert started.metadata.queued_seconds is not None
    assert started.metadata.queued_seconds >= 0.4


def test_a_failed_query_reports_its_failure(
    recorder: Recorder, server: DaftServeServer, tmp_path: pathlib.Path
) -> None:
    @daft.func(return_dtype=daft.DataType.int64())
    def explode(v: int) -> int:
        raise ValueError("the row was refused")

    df = _table(tmp_path).select(explode(col("v")).alias("v"))

    with pytest.raises(Exception, match="the row was refused"):
        _drain(_client(server).run_plan(df._builder._builder, {}, "q-fails", results_buffer_size=1))
    ended = recorder.finished("q-fails")

    assert ended.result.end_state == QueryEndState.Failed
    assert "the row was refused" in (ended.result.error_message or "")
    assert recorder.kinds("q-fails")[:2] == ["QueryQueued", "QueryStarted"]


def test_a_query_cancelled_while_queued_never_reports_a_start(
    recorder: Recorder, server: DaftServeServer, tmp_path: pathlib.Path
) -> None:
    holder = _client(server).run_plan(_crawling(tmp_path)._builder._builder, {}, "holder", results_buffer_size=1)
    outcome: dict[str, BaseException] = {}

    def submit() -> None:
        try:
            df = _table(tmp_path)
            _client(server).run_plan(df._builder._builder, {}, "q-queued", results_buffer_size=1)
        except Exception as raised:
            outcome["error"] = raised

    thread = threading.Thread(target=submit, daemon=True)
    thread.start()
    time.sleep(0.5)
    assert _client(server).cancel_query("q-queued") is True
    thread.join(timeout=10)
    ended = recorder.finished("q-queued")
    del holder

    assert "error" in outcome
    assert recorder.kinds("q-queued") == ["QueryQueued", "QueryFinished"]
    assert ended.result.end_state == QueryEndState.Canceled
    assert ended.result.rows_returned == 0


def test_a_running_query_cancelled_reports_cancelled(
    recorder: Recorder, server: DaftServeServer, tmp_path: pathlib.Path
) -> None:
    result = _client(server).run_plan(_crawling(tmp_path)._builder._builder, {}, "q-running", results_buffer_size=1)
    time.sleep(0.5)

    assert _client(server).cancel_query("q-running") is True
    with pytest.raises(Exception, match="(?i)cancel"):
        _drain(result)
    ended = recorder.finished("q-running")

    assert ended.result.end_state == QueryEndState.Canceled


def test_a_timed_out_query_reports_its_failure(recorder: Recorder, tmp_path: pathlib.Path) -> None:
    serving = start_server(ServeSettings(host="127.0.0.1", port=0, query_timeout_secs=1))
    try:
        df = _crawling(tmp_path)
        with pytest.raises(Exception, match="(?i)timeout|time limit"):
            _drain(_client(serving, None).run_plan(df._builder._builder, {}, "q-slow", results_buffer_size=1))
        ended = recorder.finished("q-slow")
    finally:
        serving.shutdown(drain_timeout_secs=10)

    assert ended.result.end_state == QueryEndState.Failed


def test_a_query_whose_client_left_reports_its_end(
    recorder: Recorder, server: DaftServeServer, tmp_path: pathlib.Path
) -> None:
    # One row per batch, so the server is still streaming when its client leaves.
    df = _crawling(tmp_path).into_batches(1)
    result = _client(server).run_plan(df._builder._builder, {}, "q-left", results_buffer_size=1)
    next(iter(result))
    del result

    ended = recorder.finished("q-left")

    assert ended.result.end_state == QueryEndState.Canceled
    assert (ended.result.rows_returned or 0) >= 1
