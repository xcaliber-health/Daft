"""A request planned past its time limit keeps its slot until its planning returns.

A request host's planning cannot be interrupted. When a request's time limit
passes while its host still plans, the client hears `Timeout` at once, and the
slot stays taken until that planning returns, so no more host plans run at
once than the server has slots. Planning that returns after its query ended
reports nothing and never runs.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Collection, Iterator
from typing import TYPE_CHECKING

import pytest

import daft
from daft.exceptions import DaftServeError
from daft.runners.remote_runner import RemoteRunner
from daft.serve import ServedRequest, ServeSettings, start_server
from daft.subscribers import Subscriber
from daft.subscribers.events import Event, QueryFinished, QueryQueued, QueryStarted

if TYPE_CHECKING:
    from daft.daft import DaftServeServer

PLAN_SECONDS = 1.5
LIMIT_SECONDS = 1
REQUESTS = 3


class SlowHost:
    """Plans slowly, counting the plans running at once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._planning = 0
        self.most = 0
        self.returned = 0

    @property
    def kinds(self) -> Collection[str]:
        return ("slow",)

    def plan(self, request: ServedRequest) -> daft.DataFrame:
        with self._lock:
            self._planning += 1
            self.most = max(self.most, self._planning)
        time.sleep(PLAN_SECONDS)
        with self._lock:
            self._planning -= 1
            self.returned += 1
        return daft.from_pydict({"n": [1]})

    def wait_until_all_returned(self, count: int) -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            with self._lock:
                if self.returned >= count:
                    return
            time.sleep(0.05)
        raise AssertionError(f"only {self.returned} of {count} plans returned")


class Recorder(Subscriber):
    def __init__(self) -> None:
        self.events: list[Event] = []
        self._lock = threading.Lock()

    def on_event(self, event: Event) -> None:
        if isinstance(event, (QueryQueued, QueryStarted, QueryFinished)):
            with self._lock:
                self.events.append(event)

    def kinds(self, query_id: str) -> list[str]:
        with self._lock:
            return [type(event).__name__ for event in self.events if getattr(event, "query_id", None) == query_id]


@pytest.fixture
def host() -> SlowHost:
    return SlowHost()


@pytest.fixture
def server(host: SlowHost) -> Iterator[DaftServeServer]:
    # The engine's context is set up before the burst arrives, so a fresh process's
    # first use (a separate defect, with its own test) cannot fail these requests.
    daft.context.get_context()
    serving = start_server(
        ServeSettings(
            host="127.0.0.1", port=0, max_concurrent_queries=1, query_timeout_secs=LIMIT_SECONDS, queue_timeout_secs=30
        ),
        request_host=host,
    )
    yield serving
    serving.shutdown(drain_timeout_secs=10)


def _at_once(send: Callable[[str], None]) -> dict[str, BaseException | None]:
    ended: dict[str, BaseException | None] = {}

    def run(query_id: str) -> None:
        try:
            send(query_id)
            ended[query_id] = None
        except BaseException as failure:
            ended[query_id] = failure

    threads = [threading.Thread(target=run, args=(f"q-{n}",)) for n in range(REQUESTS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    return ended


@pytest.mark.parametrize(
    "send",
    [
        pytest.param(lambda client, query_id: list(client.run_request("slow", b"", query_id)), id="a request"),
        pytest.param(lambda client, query_id: client.explain_request("slow", b"", query_id), id="an explanation"),
    ],
)
def test_planning_past_the_time_limit_stays_within_the_slots(
    server: DaftServeServer, host: SlowHost, send: Callable[[object, str], object]
) -> None:
    client = RemoteRunner(server.address()).client

    ended = _at_once(lambda query_id: send(client, query_id))
    host.wait_until_all_returned(REQUESTS)

    assert host.most == 1
    assert all(isinstance(failure, DaftServeError) and failure.error_class == "Timeout" for failure in ended.values())


def test_the_client_hears_its_timeout_without_waiting_for_planning(server: DaftServeServer, host: SlowHost) -> None:
    client = RemoteRunner(server.address()).client
    started = time.monotonic()

    with pytest.raises(DaftServeError, match="time limit"):
        list(client.run_request("slow", b"", "q-prompt"))

    assert time.monotonic() - started < PLAN_SECONDS
    host.wait_until_all_returned(1)


def test_planning_that_returns_after_its_query_ended_reports_nothing(server: DaftServeServer, host: SlowHost) -> None:
    recorder = Recorder()
    with daft.with_subscriber("late-planning", recorder):
        with pytest.raises(DaftServeError):
            list(RemoteRunner(server.address()).client.run_request("slow", b"", "q-late"))
        host.wait_until_all_returned(1)
        # Planning has returned; give any report it would make time to arrive.
        time.sleep(0.5)

    assert recorder.kinds("q-late") == ["QueryQueued", "QueryFinished"]
