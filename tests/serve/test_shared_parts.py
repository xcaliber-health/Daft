"""A served query computes a part written twice once, whichever way its plan arrives.

A request host plans in the server's own process, so the occurrences of a part keep the
source objects they share. A plan sent as bytes loses them, so the client marks the
shared parts before sending it.
"""

from __future__ import annotations

import threading
from collections.abc import Collection, Iterator
from typing import TYPE_CHECKING

import pytest

import daft
from daft import DataFrame, col, lit
from daft.runners.remote_runner import RemoteRunner
from daft.serve import ServedRequest, ServeSettings, start_server
from daft.subscribers import Subscriber
from daft.subscribers.events import Event, OptimizationCompleted
from tests.dataframe.test_shared_parts import BATCHES, CountingSource, repro
from tests.serve.conftest import collect_via

if TYPE_CHECKING:
    from daft.daft import DaftServeServer
    from daft.runners.native_runner import NativeRunner


class RepeatingHost:
    """Answers a request with an aggregation read on both sides of a join, over one counting source."""

    def __init__(self) -> None:
        self.source = CountingSource()

    @property
    def kinds(self) -> Collection[str]:
        return ("repeat",)

    def plan(self, request: ServedRequest) -> DataFrame:
        del request
        return repro(self.source.read(), reader_filter=True, same_object=False)


class OptimizedPlans(Subscriber):
    """Keeps the optimized plan each query reports."""

    def __init__(self) -> None:
        self.plans: dict[str, str] = {}
        self._lock = threading.Lock()

    def on_event(self, event: Event) -> None:
        if isinstance(event, OptimizationCompleted):
            with self._lock:
                self.plans[event.query_id] = event.optimized_plan


@pytest.fixture
def host() -> RepeatingHost:
    return RepeatingHost()


@pytest.fixture
def server(host: RepeatingHost) -> Iterator[DaftServeServer]:
    serving = start_server(ServeSettings(host="127.0.0.1", port=0), request_host=host)
    yield serving
    serving.shutdown(drain_timeout_secs=10)


def test_a_request_planned_by_its_host_reads_its_source_once(server: DaftServeServer, host: RepeatingHost) -> None:
    rows = sum(len(part) for part in RemoteRunner(server.address()).client.run_request("repeat", b"", "q-repeat"))

    assert rows > 0
    assert host.source.batches_read == BATCHES


def test_a_sent_plan_shares_its_repeated_part(server: DaftServeServer, native_runner: NativeRunner) -> None:
    rows = daft.from_pydict({"k": [n % 10 for n in range(1_000)], "v": list(range(1_000))})
    part = lambda: rows.groupby("k").agg(col("v").sum().alias("s"))  # noqa: E731
    df = part().where(col("k") % lit(2) == lit(1)).join(part(), on="k", suffix="_all")
    plans = OptimizedPlans()

    with daft.with_subscriber("shared-parts", plans):
        got = collect_via(RemoteRunner(server.address()), df)

    assert sorted(zip(*got.values())) == sorted(zip(*collect_via(native_runner, df).values()))
    assert any('"Shared"' in plan for plan in plans.plans.values())
