"""A served result's schema is its plan's, handed over once planning returns.

The schema reaches the client before any of the query runs, so a query stopped
before its first part still hands it over before why it ended. It is exactly
the schema of the result's rows: every column, type, time zone, precision and
nested field reads as a local result's does.
"""

from __future__ import annotations

import time
from collections.abc import Collection, Iterator
from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

import daft
from daft import col
from daft.exceptions import DaftServeError
from daft.runners.remote_runner import RemoteRunner
from daft.serve import ServedRequest, ServeSettings, start_server

from .test_equality import CASES
from .test_types import TYPE_CASES

if TYPE_CHECKING:
    from daft.daft import DaftServeClient, DaftServeServer

LIMIT_SECONDS = 1
# Far longer than the time limit, so the query is stopped before its first part.
WORK_SECONDS = 4.0
# The schema arrives once the query is planned, long before its work ends.
PLANNED_WITHIN_SECONDS = 1.0


def _slow() -> daft.DataFrame:
    @daft.func(return_dtype=daft.DataType.int64())
    def wait(n: int) -> int:
        time.sleep(WORK_SECONDS)
        return n

    return daft.from_pydict({"n": [1]}).select(
        wait(col("n")).alias("n"), col("n").cast(daft.DataType.string()).alias("s")
    )


def _frame(kind: str) -> daft.DataFrame:
    if kind == "slow":
        return _slow()
    if kind in CASES:
        build, _ = CASES[kind]
        return build()
    return daft.from_arrow(pa.table({"c": TYPE_CASES[kind]}))


KINDS = sorted([*CASES, *TYPE_CASES])


class Host:
    """Answers each plan of the equality and type matrices, and a slow one."""

    @property
    def kinds(self) -> Collection[str]:
        return (*KINDS, "slow")

    def plan(self, request: ServedRequest) -> daft.DataFrame:
        return _frame(request.kind)


@pytest.fixture(scope="module")
def server() -> Iterator[DaftServeServer]:
    serving = start_server(ServeSettings(host="127.0.0.1", port=0), request_host=Host())
    yield serving
    serving.shutdown(drain_timeout_secs=1)


@pytest.mark.parametrize("kind", KINDS)
def test_its_schema_is_exactly_the_local_results(server: DaftServeServer, kind: str) -> None:
    local = _frame(kind).to_arrow().schema
    served = RemoteRunner(server.address()).client.run_request(kind, b"", f"q-{kind}")

    assert pa.schema(served) == local
    assert pa.table(served).schema == local


@pytest.fixture(params=["cancel", "time limit"], ids=["cancelled", "past its time limit"])
def stop(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def limited(stop: str) -> Iterator[DaftServeServer]:
    serving = start_server(
        ServeSettings(host="127.0.0.1", port=0, query_timeout_secs=LIMIT_SECONDS if stop == "time limit" else 0),
        request_host=Host(),
    )
    yield serving
    serving.shutdown(drain_timeout_secs=1)


@pytest.fixture
def client(limited: DaftServeServer) -> DaftServeClient:
    # Held for the whole test: a client let go closes its connection, which ends its queries.
    return RemoteRunner(limited.address()).client


def test_a_query_stopped_before_its_first_part_hands_over_its_schema_first(client: DaftServeClient, stop: str) -> None:
    started = time.monotonic()
    answer = client.run_request("slow", b"", "q-slow")

    schema = pa.schema(answer)
    planned = time.monotonic() - started
    if stop == "cancel":
        client.cancel_query("q-slow")
    with pytest.raises(pa.ArrowException):
        pa.table(answer)

    assert schema == pa.schema([("n", pa.int64()), ("s", pa.large_string())])
    assert planned < PLANNED_WITHIN_SECONDS
    assert isinstance(answer.failure, DaftServeError)
    assert answer.failure.error_class == ("Cancelled" if stop == "cancel" else "Timeout")
