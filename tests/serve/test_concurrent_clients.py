"""Multiple distinct clients against one server.

Verifies isolation of concurrently submitted queries and that one client's
in-memory data never bleeds into another client's results.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import daft
from daft import col
from daft.runners.remote_runner import RemoteRunner

from .conftest import collect_via

if TYPE_CHECKING:
    from daft.daft import DaftServeServer


def test_distinct_clients_get_isolated_results(serve_server: DaftServeServer) -> None:
    client_a = RemoteRunner(serve_server.address())
    client_b = RemoteRunner(serve_server.address())

    df_a = daft.from_pydict({"v": [1, 2, 3]}).select((col("v") * 10).alias("v"))
    df_b = daft.from_pydict({"v": [7, 8]}).select((col("v") + 1).alias("v"))

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(collect_via, client_a, df_a),
            pool.submit(collect_via, client_b, df_b),
        ]
        got_a, got_b = (f.result(timeout=120) for f in futures)

    assert got_a == {"v": [10, 20, 30]}
    assert got_b == {"v": [8, 9]}


def test_interleaved_queries_from_many_clients(serve_server: DaftServeServer) -> None:
    runners = [RemoteRunner(serve_server.address()) for _ in range(4)]

    def run(i: int) -> dict[str, list[object]]:
        df = daft.from_pydict({"v": list(range(i + 1))}).select((col("v") + i).alias("v"))
        return collect_via(runners[i], df)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(4)))

    for i, got in enumerate(results):
        assert got == {"v": [v + i for v in range(i + 1)]}
