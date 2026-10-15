"""Answers a client holds unread hold up none of its other calls.

A caller may send several requests and read their answers later. What the
server sends ahead of an answer's reader is bounded for each answer on its
own, so answers left unread never stop the same client's next answer, and
each reads in full once its reader takes it.
"""

from __future__ import annotations

import threading
from collections.abc import Collection, Iterator
from typing import TYPE_CHECKING

import pytest

import daft
from daft.runners.remote_runner import RemoteRunner
from daft.serve import ServedRequest, ServeSettings, start_server

if TYPE_CHECKING:
    from daft.daft import DaftServeClient, DaftServeQueryResult, DaftServeServer

# Each answer is 8 MB of int64; together they are far more than a connection's
# default window.
ROWS = 1_000_000
UNREAD = 16
PROMPT_SECS = 10.0


class Host:
    """Answers many rows, or one."""

    @property
    def kinds(self) -> Collection[str]:
        return ("many", "one")

    def plan(self, request: ServedRequest) -> daft.DataFrame:
        return daft.range(ROWS if request.kind == "many" else 1)


@pytest.fixture
def server() -> Iterator[DaftServeServer]:
    serving = start_server(
        ServeSettings(host="127.0.0.1", port=0, max_concurrent_queries=2 * UNREAD),
        request_host=Host(),
    )
    yield serving
    serving.shutdown(drain_timeout_secs=1)


@pytest.fixture
def client(server: DaftServeServer) -> DaftServeClient:
    return RemoteRunner(server.address()).client


def _rows(answer: DaftServeQueryResult) -> int:
    return sum(len(part) for part in answer)


def _answered_within(client: DaftServeClient, seconds: float) -> int | None:
    """Return the rows of a one-row request, or None if it has not answered in time."""
    answered: list[int] = []
    asker = threading.Thread(
        target=lambda: answered.append(_rows(client.run_request("one", b"", "q-one"))), daemon=True
    )
    asker.start()
    asker.join(seconds)
    return answered[0] if answered else None


def test_a_request_answers_while_the_same_client_holds_many_answers_unread(client: DaftServeClient) -> None:
    held = [client.run_request("many", b"", f"q-held-{number}") for number in range(UNREAD)]

    assert _answered_within(client, PROMPT_SECS) == 1
    assert [_rows(answer) for answer in held] == [ROWS] * UNREAD
