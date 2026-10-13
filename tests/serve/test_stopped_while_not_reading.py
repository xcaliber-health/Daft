"""A query stopped while its client is not reading ends at once.

A client that holds a result without reading it leaves the server waiting for
room to send. A query cancelled, or past its time limit, ends regardless: it
frees its slot at once, reports its end, and discards what its client has not
read, so the client's next read raises why it ended. A server's drain, which
cancels whatever still runs, completes too, closing connections whose clients
do not read.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Collection, Iterator
from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

import daft
from daft.daft import QueryEndState
from daft.exceptions import DaftServeError
from daft.runners.remote_runner import RemoteRunner
from daft.serve import ServedRequest, ServeSettings, start_server
from daft.subscribers import Subscriber
from daft.subscribers.events import Event, QueryFinished

if TYPE_CHECKING:
    from daft.daft import DaftServeClient, DaftServeQueryResult, DaftServeServer

# Far more than the server holds for a client, so a query waits on a client not reading.
ROWS = 4_000_000
LIMIT_SECONDS = 2
# How long a stopped query may take to end, past the moment it is stopped.
PATIENCE_SECONDS = 5
# The parts the server holds for a client; what it holds unread is discarded at the stop.
HELD_PARTS = 16


class Host:
    """Answers a result larger than the server holds for one client, or a small one."""

    def __init__(self) -> None:
        self._rows = pa.table({"n": pa.array(range(ROWS), pa.int64())})

    @property
    def kinds(self) -> Collection[str]:
        return ("large", "small")

    def plan(self, request: ServedRequest) -> daft.DataFrame:
        if request.kind == "large":
            return daft.from_arrow(self._rows)
        return daft.from_pydict({"n": [1, 2, 3]})


class Recorder(Subscriber):
    def __init__(self) -> None:
        self.finished: dict[str, QueryFinished] = {}
        self._lock = threading.Lock()

    def on_event(self, event: Event) -> None:
        if isinstance(event, QueryFinished):
            with self._lock:
                self.finished[event.query_id] = event

    def wait_for(self, query_id: str, seconds: float) -> QueryFinished | None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            with self._lock:
                if query_id in self.finished:
                    return self.finished[query_id]
            time.sleep(0.02)
        return None


@pytest.fixture(params=["cancel", "time limit"], ids=["cancelled by its id", "past its time limit"])
def stop(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def server(stop: str) -> Iterator[DaftServeServer]:
    daft.context.get_context()
    serving = start_server(
        ServeSettings(
            host="127.0.0.1",
            port=0,
            max_concurrent_queries=1,
            query_timeout_secs=LIMIT_SECONDS if stop == "time limit" else 0,
            queue_timeout_secs=30,
        ),
        request_host=Host(),
    )
    yield serving
    serving.shutdown(drain_timeout_secs=1)


@pytest.fixture
def client(server: DaftServeServer) -> DaftServeClient:
    # Held for the whole test: a client let go closes its connection, which ends its queries.
    return RemoteRunner(server.address()).client


def _stopped(client: DaftServeClient, stop: str) -> DaftServeQueryResult:
    """Start a large result, read one part, stop its query, and read no more."""
    held = client.run_request("large", b"", "q-held", results_buffer_size=HELD_PARTS)
    next(held)
    if stop == "cancel":
        client.cancel_query("q-held")
    return held


def _patience(stop: str) -> float:
    return PATIENCE_SECONDS + (LIMIT_SECONDS if stop == "time limit" else 0)


def test_its_slot_is_free_for_the_next_query_while_its_client_holds_it(
    server: DaftServeServer, client: DaftServeClient, stop: str
) -> None:
    held = _stopped(client, stop)
    started = time.monotonic()

    answered = sum(len(part) for part in client.run_request("small", b"", "q-next"))

    assert answered == 3
    assert time.monotonic() - started < _patience(stop)
    assert server.active_queries() == 0
    del held


def test_its_end_is_reported_while_its_client_holds_it(client: DaftServeClient, stop: str) -> None:
    recorder = Recorder()
    with daft.with_subscriber("stopped-while-not-reading", recorder):
        held = _stopped(client, stop)
        ended = recorder.wait_for("q-held", _patience(stop))

    assert ended is not None
    assert ended.result.end_state == (QueryEndState.Canceled if stop == "cancel" else QueryEndState.Failed)
    del held


def test_its_clients_next_read_raises_why_it_ended(server: DaftServeServer, client: DaftServeClient, stop: str) -> None:
    held = _stopped(client, stop)
    deadline = time.monotonic() + _patience(stop)
    while server.active_queries() and time.monotonic() < deadline:
        time.sleep(0.02)

    read_after = 0
    with pytest.raises(DaftServeError) as raised:
        for _ in held:
            read_after += 1

    assert raised.value.error_class == ("Cancelled" if stop == "cancel" else "Timeout")
    assert raised.value.query_id == "q-held"
    # Only what was already on its way to the client arrives; the server's unread parts do not.
    assert read_after < HELD_PARTS, read_after


# Each drain runs in its own interpreter, which then exits, so what the server leaves
# running past its drain shows as a hang or a failure at exit.
DRAIN = textwrap.dedent(
    """
    import json, sys, time
    import pyarrow as pa
    import daft
    from daft.runners.remote_runner import RemoteRunner
    from daft.serve import ServeSettings, start_server

    class Host:
        kinds = ("large",)

        def __init__(self):
            self.rows = pa.table({"n": pa.array(range(ROWS), pa.int64())})

        def plan(self, request):
            return daft.from_arrow(self.rows)

    daft.context.get_context()
    server = start_server(ServeSettings(host="127.0.0.1", port=0), request_host=Host())
    client = RemoteRunner(server.address()).client
    held = client.run_request("large", b"", "q-held")
    next(held)
    if sys.argv[1] == "cancelled":
        client.cancel_query("q-held")
    started = time.monotonic()
    server.shutdown(drain_timeout_secs=1)
    server.wait()
    waited = time.monotonic() - started
    try:
        for _ in held:
            pass
        read = "finished"
    except Exception as failure:
        read = type(failure).__name__
    print(json.dumps({"waited": waited, "active": server.active_queries(), "read": read}))
    """
).replace("ROWS", str(ROWS))


@pytest.mark.parametrize("held", ["cancelled", "running"], ids=["held after its cancel", "held while running"])
def test_a_drain_completes_while_a_client_holds_a_result_unread(held: str) -> None:
    ran = subprocess.run([sys.executable, "-c", DRAIN, held], capture_output=True, text=True, timeout=120, check=False)

    assert ran.returncode == 0, ran.stderr
    assert "panicked" not in ran.stderr
    drained = json.loads(ran.stdout.strip().splitlines()[-1])
    # The drain window, then a short grace for clients to hear their queries ended.
    assert drained["waited"] < 1 + PATIENCE_SECONDS
    assert drained["active"] == 0
    assert drained["read"] != "finished"
