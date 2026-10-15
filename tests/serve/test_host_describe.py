"""A request host describes how a frame it planned failed while running.

The description reaches the client as the failure's ``detail``, beside a class
and message the engine left unchanged. The host is asked only for a failure of
its frame while it ran, within the query's slot and time limit; if it cannot
describe it, the failure reaches the client as the engine raised it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Collection, Iterator
from typing import TYPE_CHECKING

import pytest

import daft
from daft import col
from daft.exceptions import DaftCardinalityError, DaftServeError
from daft.runners.remote_runner import RemoteRunner
from daft.serve import DescribingHost, ServedRequest, ServeSettings, start_server
from daft.subscribers import Subscriber
from daft.subscribers.events import Event, QueryFinished

if TYPE_CHECKING:
    from daft.daft import DaftServeServer

DESCRIBED = "a group by g held 3"


@daft.func(return_dtype=daft.DataType.int64())
def broken(value: int) -> int:
    raise ValueError(f"no value for {value}")


class Host:
    """Plans a frame for each kind, and describes failures as it is told to."""

    def __init__(self, answer: str | None = DESCRIBED, raises: bool = False, sleep: float = 0.0) -> None:
        self._answer = answer
        self._raises = raises
        self._sleep = sleep
        self.asked: list[tuple[ServedRequest, Exception]] = []
        self.describing = threading.Event()

    @property
    def kinds(self) -> Collection[str]:
        return ("collapse", "fine", "refused", "udf")

    def plan(self, request: ServedRequest) -> daft.DataFrame:
        if request.kind == "refused":
            raise PermissionError("the request was refused by its host")
        rows = daft.from_pydict({"g": [1, 1, 1], "v": [1, 2, 3]})
        if request.kind == "collapse":
            return rows.groupby("g").agg(col("v").single_value())
        if request.kind == "udf":
            return rows.select(broken(col("v")).alias("v"))
        return rows

    def describe(self, request: ServedRequest, failure: Exception) -> str | None:
        self.asked.append((request, failure))
        self.describing.set()
        time.sleep(self._sleep)
        if self._raises:
            raise RuntimeError("the description could not be made")
        return self._answer


class Ended(Subscriber):
    """Records the queries whose end has been reported."""

    def __init__(self) -> None:
        self.query_ids: set[str] = set()

    def on_event(self, event: Event) -> None:
        if isinstance(event, QueryFinished):
            self.query_ids.add(event.query_id)


class EndWatchingHost(Host):
    """Notes, when asked to describe, whether the query's end was already reported."""

    def __init__(self, ended: Ended) -> None:
        super().__init__()
        self._ended = ended
        self.ended_before_describing: list[bool] = []

    def describe(self, request: ServedRequest, failure: Exception) -> str | None:
        self.ended_before_describing.append(request.query_id in self._ended.query_ids)
        return super().describe(request, failure)


class PlanningOnlyHost(Host):
    """A host with no ``describe``."""

    describe = None  # type: ignore[assignment]


def _serving(host: object, **settings: object) -> Iterator[DaftServeServer]:
    server = start_server(ServeSettings(host="127.0.0.1", port=0, **settings), request_host=host)
    try:
        yield server
    finally:
        server.shutdown(drain_timeout_secs=10)


def _run(server: DaftServeServer, kind: str, query_id: str = "q-1") -> int:
    result = RemoteRunner(server.address()).client.run_request(kind, b"body", query_id)
    return sum(len(part) for part in result)


def test_a_failure_while_running_is_described_with_its_class_and_message_unchanged() -> None:
    host = Host()
    for server in _serving(host):
        with pytest.raises(DaftCardinalityError) as raised:
            _run(server, "collapse", "q-collapse")

    assert raised.value.detail == DESCRIBED
    assert "more than one row" in str(raised.value)
    assert DESCRIBED not in str(raised.value)
    assert raised.value.query_id == "q-collapse"
    [(request, failure)] = host.asked
    assert (request.kind, request.body, request.query_id) == ("collapse", b"body", "q-collapse")
    assert isinstance(failure, DaftCardinalityError)
    assert failure.query_id == "q-collapse"


def test_a_python_failure_while_running_is_described_too() -> None:
    host = Host()
    for server in _serving(host):
        with pytest.raises(Exception, match="no value for") as raised:
            _run(server, "udf")

    assert raised.value.detail == DESCRIBED
    assert len(host.asked) == 1


@pytest.mark.parametrize("kind", ["fine", "refused"], ids=["a success", "a failure plan raised"])
def test_the_host_is_not_asked_outside_a_failure_while_running(kind: str) -> None:
    host = Host()
    for server in _serving(host):
        try:
            _run(server, kind)
        except DaftServeError as failure:
            assert failure.error_class == "Host"
            assert failure.detail is None

    assert host.asked == []


@pytest.mark.parametrize(
    "host",
    [Host(answer=None), Host(raises=True), PlanningOnlyHost()],
    ids=["describes nothing", "raises", "has no describe"],
)
def test_a_failure_not_described_reaches_the_client_as_raised(host: Host) -> None:
    for server in _serving(host):
        with pytest.raises(DaftCardinalityError, match="more than one row") as raised:
            _run(server, "collapse")
        rows_after = _run(server, "fine", "q-2")

    assert raised.value.detail is None
    assert rows_after == 3


def test_a_description_past_the_time_limit_sends_the_failure_as_raised() -> None:
    host = Host(sleep=6.0)
    for server in _serving(host, query_timeout_secs=2):
        started = time.monotonic()
        with pytest.raises(DaftCardinalityError) as raised:
            _run(server, "collapse")
        took = time.monotonic() - started

    assert raised.value.detail is None
    assert took < 5.0


def test_a_long_description_is_bounded() -> None:
    host = Host(answer="é" * 3_000)
    for server in _serving(host):
        with pytest.raises(DaftCardinalityError) as raised:
            _run(server, "collapse")

    detail = raised.value.detail
    assert len(detail.encode()) <= 2048
    assert detail.endswith("…")
    assert detail[:-1] == "é" * len(detail[:-1])


def test_describing_holds_the_querys_slot() -> None:
    host = Host(sleep=1.5)
    for server in _serving(host, max_concurrent_queries=1):
        failing = threading.Thread(target=lambda: pytest.raises(DaftCardinalityError, _run, server, "collapse"))
        failing.start()
        assert host.describing.wait(30)
        held = server.active_queries()
        failing.join()

    assert held == 1


def test_the_host_describes_before_the_querys_end_is_reported() -> None:
    ended = Ended()
    host = EndWatchingHost(ended)
    with daft.with_subscriber("describe-order", ended):
        for server in _serving(host):
            with pytest.raises(DaftCardinalityError):
                _run(server, "collapse", "q-order")

    assert host.ended_before_describing == [False]
    assert "q-order" in ended.query_ids


def test_a_host_with_describe_is_a_describing_host() -> None:
    assert isinstance(Host(), DescribingHost)
    assert not isinstance(PlanningOnlyHost(), DescribingHost)
