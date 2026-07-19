"""Authentication, payload policy, and admission limits at the endpoint."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import daft
from daft.runners.remote_runner import RemoteRunner
from daft.serve import ServeSettings, start_server

from .conftest import collect_via


@pytest.fixture(scope="module")
def token_server() -> Iterator[object]:
    server = start_server(ServeSettings(host="127.0.0.1", port=0, token="s3cr3t"))
    yield server
    server.shutdown(drain_timeout_secs=5)


def test_correct_token_is_accepted(token_server: object) -> None:
    runner = RemoteRunner(token_server.address(), token="s3cr3t")
    df = daft.from_pydict({"a": [1, 2]})
    assert collect_via(runner, df) == {"a": [1, 2]}


@pytest.mark.parametrize("token", [None, "wrong", ""])
def test_bad_token_is_rejected(token_server: object, token: str | None) -> None:
    with pytest.raises(Exception, match="(?i)unauthenticated|authorization|token"):
        RemoteRunner(token_server.address(), token=token)


def test_plan_payloads_can_be_disabled() -> None:
    server = start_server(ServeSettings(host="127.0.0.1", port=0, disable_plan_payload=True))
    try:
        runner = RemoteRunner(server.address())
        df = daft.from_pydict({"a": [1]})
        with pytest.raises(Exception, match="(?i)plan payloads are disabled"):
            collect_via(runner, df)
        # The textual lane stays available on a plan-disabled server.
        result = runner.client.run_sql("select 1 as one", "q-sql-lane")
        parts = list(result)
        assert len(parts) == 1
    finally:
        server.shutdown(drain_timeout_secs=5)


def test_oversized_in_memory_data_is_rejected() -> None:
    server = start_server(ServeSettings(host="127.0.0.1", port=0, max_pset_bytes=1024))
    try:
        runner = RemoteRunner(server.address())
        df = daft.from_pydict({"a": list(range(100_000))})
        with pytest.raises(Exception, match="(?i)exceeds the server cap"):
            collect_via(runner, df)
    finally:
        server.shutdown(drain_timeout_secs=5)


def test_single_slot_server_serializes_queries() -> None:
    server = start_server(ServeSettings(host="127.0.0.1", port=0, max_concurrent_queries=1))
    try:
        runner = RemoteRunner(server.address())
        df = daft.from_pydict({"a": list(range(10))})
        # Two sequential queries both succeed through the single slot.
        assert collect_via(runner, df)["a"] == list(range(10))
        assert collect_via(runner, df)["a"] == list(range(10))
    finally:
        server.shutdown(drain_timeout_secs=5)


def test_server_info_reports_configuration() -> None:
    server = start_server(ServeSettings(host="127.0.0.1", port=0, max_concurrent_queries=7, max_pset_bytes=12345))
    try:
        runner = RemoteRunner(server.address())
        info = runner.client.server_info()
        assert info.max_concurrent_queries == 7
        assert info.max_pset_bytes == 12345
        assert info.plan_payload_enabled is True
        assert info.catalogs == []
    finally:
        server.shutdown(drain_timeout_secs=5)
