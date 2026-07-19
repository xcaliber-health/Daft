"""Multiple named tenants sharing one server.

Each tenant authenticates with its own token and gets its own admission
slots, execution time limit, and payload cap; one tenant can neither starve
nor cancel another.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

import daft
from daft import col
from daft.runners.remote_runner import RemoteRunner
from daft.serve import ServeSettings, TenantSpec, start_server

from .conftest import collect_via

if TYPE_CHECKING:
    import pathlib

    from daft.daft import DaftServeQueryResult, DaftServeServer


@pytest.fixture(scope="module")
def tenant_server() -> Iterator[DaftServeServer]:
    """A server with three tenants.

    `alpha` has one dedicated slot, a short queue, and a small payload cap;
    `beta` inherits the server defaults; `gamma` has a two-second execution
    limit.
    """
    server = start_server(
        ServeSettings(
            host="127.0.0.1",
            port=0,
            tenants=(
                TenantSpec(
                    name="alpha",
                    token="tok-alpha",
                    max_concurrent_queries=1,
                    queue_timeout_secs=1,
                    max_pset_bytes=64 * 1024,
                ),
                TenantSpec(name="beta", token="tok-beta"),
                TenantSpec(name="gamma", token="tok-gamma", query_timeout_secs=2),
            ),
        )
    )
    yield server
    server.shutdown(drain_timeout_secs=10)


def _slow_scan_df(tmp_path: pathlib.Path, rows: int = 16, delay_secs: float = 0.3) -> daft.DataFrame:
    """A file-backed query that runs for roughly ``rows * delay_secs``.

    File-backed so it references no in-memory data and can be submitted
    through the connection's client with a chosen query id.
    """
    path = str(tmp_path / f"slow-{rows}-{int(delay_secs * 1000)}.parquet")
    daft.from_pydict({"v": list(range(rows))}).write_parquet(path, write_mode="overwrite")

    @daft.func(return_dtype=daft.DataType.int64())
    def crawl(v: int) -> int:
        time.sleep(delay_secs)
        return v

    return daft.read_parquet(path).select(crawl(col("v")).alias("v"))


def _submit(runner: RemoteRunner, df: daft.DataFrame, query_id: str) -> DaftServeQueryResult:
    """Submit a plan under a chosen id; returns once the query is admitted.

    The returned result stream holds the query's execution slot until it is
    drained or dropped.
    """
    return runner.client.run_plan(df._builder._builder, {}, query_id, results_buffer_size=1)


def test_each_tenant_executes_with_its_own_token(tenant_server: DaftServeServer) -> None:
    alpha = RemoteRunner(tenant_server.address(), token="tok-alpha")
    beta = RemoteRunner(tenant_server.address(), token="tok-beta")

    df = daft.from_pydict({"a": [1, 2, 3]}).select((col("a") * 2).alias("a"))
    assert collect_via(alpha, df) == {"a": [2, 4, 6]}
    assert collect_via(beta, df) == {"a": [2, 4, 6]}


@pytest.mark.parametrize("token", ["wrong", "", None])
def test_non_tenant_tokens_are_rejected(tenant_server: DaftServeServer, token: str | None) -> None:
    with pytest.raises(Exception, match="(?i)unauthenticated|authorization|token"):
        RemoteRunner(tenant_server.address(), token=token)


def test_saturated_tenant_does_not_block_other_tenant(tenant_server: DaftServeServer, tmp_path: pathlib.Path) -> None:
    alpha = RemoteRunner(tenant_server.address(), token="tok-alpha")
    beta = RemoteRunner(tenant_server.address(), token="tok-beta")

    # Occupy alpha's single slot with a slow query.
    held = _submit(alpha, _slow_scan_df(tmp_path), "alpha-holds-slot")

    # Beta's queries are admitted immediately through the default pool.
    quick = daft.from_pydict({"b": [7, 8]})
    assert collect_via(beta, quick) == {"b": [7, 8]}
    del held


def test_saturated_tenant_gets_capacity_error(tenant_server: DaftServeServer, tmp_path: pathlib.Path) -> None:
    alpha = RemoteRunner(tenant_server.address(), token="tok-alpha")

    held = _submit(alpha, _slow_scan_df(tmp_path), "alpha-holds-slot-2")
    with pytest.raises(Exception, match="(?i)at capacity"):
        collect_via(alpha, daft.from_pydict({"a": [1]}))
    del held


def test_tenant_query_timeout_applies_only_to_that_tenant(
    tenant_server: DaftServeServer, tmp_path: pathlib.Path
) -> None:
    gamma = RemoteRunner(tenant_server.address(), token="tok-gamma")
    beta = RemoteRunner(tenant_server.address(), token="tok-beta")

    with pytest.raises(Exception, match="(?i)timeout|time limit"):
        collect_via(gamma, _slow_scan_df(tmp_path))

    # Beta has no execution limit; a shorter slow query completes.
    got = collect_via(beta, _slow_scan_df(tmp_path, rows=4, delay_secs=0.1))
    assert sorted(got["v"]) == list(range(4))


def test_tenant_payload_cap_applies_only_to_that_tenant(tenant_server: DaftServeServer) -> None:
    alpha = RemoteRunner(tenant_server.address(), token="tok-alpha")
    beta = RemoteRunner(tenant_server.address(), token="tok-beta")

    big = daft.from_pydict({"a": list(range(100_000))})
    with pytest.raises(Exception, match="(?i)exceeds the server cap"):
        collect_via(alpha, big)
    assert len(collect_via(beta, big)["a"]) == 100_000


def test_cancel_is_scoped_to_the_owning_tenant(tenant_server: DaftServeServer, tmp_path: pathlib.Path) -> None:
    alpha = RemoteRunner(tenant_server.address(), token="tok-alpha")
    beta = RemoteRunner(tenant_server.address(), token="tok-beta")

    query_id = "tenancy-cancel-scope"
    held = _submit(alpha, _slow_scan_df(tmp_path), query_id)

    # Another tenant cannot cancel it; the owner can.
    assert beta.client.cancel_query(query_id) is False
    assert alpha.client.cancel_query(query_id) is True
    del held
