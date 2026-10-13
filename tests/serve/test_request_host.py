"""A registered request host: requests of its kinds, planned on the server.

The host is handed each request with its query id and its authenticated
caller, after admission. Requests of undeclared kinds or oversized bodies are
refused before admission; SQL keeps its own lane beside the host's.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

import daft
from daft import col
from daft.runners.remote_runner import RemoteRunner
from daft.serve import ServedRequest, ServeSettings, TenantSpec, start_server

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Collection

    from daft.daft import DaftServeClient, DaftServeQueryResult, DaftServeServer


class RecordingHost:
    """Answers each request with one row describing it, and keeps every request it was handed."""

    def __init__(self, kinds: Collection[str] = ("echo",), fail: bool = False) -> None:
        self._kinds = frozenset(kinds)
        self._fail = fail
        self.handed: list[ServedRequest] = []

    @property
    def kinds(self) -> Collection[str]:
        return self._kinds

    def plan(self, request: ServedRequest) -> daft.DataFrame:
        self.handed.append(request)
        if self._fail:
            raise PermissionError("the request was refused by its host")
        return daft.from_pydict({"kind": [request.kind], "body": [request.body.decode()]})


def _rows(result: Iterator[object]) -> int:
    return sum(len(part) for part in result)


@pytest.fixture
def host() -> RecordingHost:
    return RecordingHost()


@pytest.fixture
def tenant_server(host: RecordingHost) -> Iterator[DaftServeServer]:
    server = start_server(
        ServeSettings(
            host="127.0.0.1",
            port=0,
            max_request_body_bytes=64,
            tenants=(
                TenantSpec(name="svc", token="tok-svc", max_concurrent_queries=1, queue_timeout_secs=1),
                TenantSpec(name="big", token="tok-big", max_request_body_bytes=1024),
            ),
        ),
        request_host=host,
    )
    yield server
    server.shutdown(drain_timeout_secs=10)


def _client(server: DaftServeServer, token: str | None = None) -> DaftServeClient:
    return RemoteRunner(server.address(), token=token).client


def _hold_slot(server: DaftServeServer, token: str, tmp_path: pathlib.Path) -> DaftServeQueryResult:
    """Take the caller's slot with a query that runs for seconds; the slot frees when the result is dropped."""
    path = str(tmp_path / "hold.parquet")
    daft.from_pydict({"v": list(range(8))}).write_parquet(path, write_mode="overwrite")

    @daft.func(return_dtype=daft.DataType.int64())
    def crawl(v: int) -> int:
        time.sleep(0.5)
        return v

    held = daft.read_parquet(path).select(crawl(col("v")).alias("v"))
    return _client(server, token).run_plan(held._builder._builder, {}, "holder", results_buffer_size=1)


def test_the_host_is_handed_the_request_its_query_id_and_its_caller(
    tenant_server: DaftServeServer, host: RecordingHost
) -> None:
    rows = _rows(_client(tenant_server, "tok-svc").run_request("echo", b"hello", "q-23"))

    assert rows == 1
    assert host.handed == [ServedRequest(kind="echo", body=b"hello", query_id="q-23", caller="svc")]


def test_the_caller_is_none_without_named_callers(host: RecordingHost) -> None:
    server = start_server(ServeSettings(host="127.0.0.1", port=0, token="shared"), request_host=host)
    try:
        _rows(_client(server, "shared").run_request("echo", b"x", "q-anon"))
    finally:
        server.shutdown(drain_timeout_secs=10)

    assert [request.caller for request in host.handed] == [None]


def test_an_undeclared_kind_is_refused_before_admission(
    tenant_server: DaftServeServer, host: RecordingHost, tmp_path: pathlib.Path
) -> None:
    # The caller's one slot is taken, so a refusal that waited for admission would fail on capacity instead.
    holder = _hold_slot(tenant_server, "tok-svc", tmp_path)

    with pytest.raises(Exception, match="no host for request kind `unknown`"):
        _client(tenant_server, "tok-svc").run_request("unknown", b"x", "q-unknown")

    assert host.handed == []
    del holder


@pytest.mark.parametrize(
    ("token", "body_bytes", "refused"),
    [
        pytest.param("tok-svc", 64, False, id="at the server cap"),
        pytest.param("tok-svc", 65, True, id="over the server cap"),
        pytest.param("tok-big", 1024, False, id="within a tenant's own cap"),
    ],
)
def test_a_request_body_is_capped(tenant_server: DaftServeServer, token: str, body_bytes: int, refused: bool) -> None:
    send = lambda: _rows(_client(tenant_server, token).run_request("echo", b"x" * body_bytes, f"q-{body_bytes}"))  # noqa: E731

    if refused:
        with pytest.raises(Exception, match="exceeds the server cap"):
            send()
    else:
        assert send() == 1


def test_sql_keeps_its_own_lane_beside_the_host(tenant_server: DaftServeServer) -> None:
    rows = _rows(_client(tenant_server, "tok-big").run_sql("SELECT 1 AS one", "q-sql"))

    assert rows == 1


def test_a_server_without_sql_still_serves_its_host(host: RecordingHost) -> None:
    server = start_server(ServeSettings(host="127.0.0.1", port=0, disable_sql=True), request_host=host)
    try:
        client = _client(server)
        with pytest.raises(Exception, match="no session configured for textual queries"):
            _rows(client.run_sql("SELECT 1", "q-sql"))
        assert _rows(client.run_request("echo", b"x", "q-host")) == 1
    finally:
        server.shutdown(drain_timeout_secs=10)


def test_disabling_sql_with_catalogs_is_refused() -> None:
    from daft.serve import CatalogSpec

    settings = ServeSettings(port=0, disable_sql=True, catalogs=(CatalogSpec(name="c", properties={}),))

    with pytest.raises(ValueError, match="disable_sql"):
        start_server(settings)


def test_an_explanation_is_planned_by_the_host_without_running(tmp_path: pathlib.Path) -> None:
    marker = tmp_path / "ran"

    class MarkingHost(RecordingHost):
        def plan(self, request: ServedRequest) -> daft.DataFrame:
            self.handed.append(request)
            marker_path = str(marker)

            @daft.func(return_dtype=daft.DataType.int64())
            def mark(v: int) -> int:
                with open(marker_path, "w") as written:
                    written.write("ran")
                return v

            return daft.from_pydict({"v": [1]}).select(mark(col("v")).alias("v"))

    marking = MarkingHost()
    server = start_server(
        ServeSettings(host="127.0.0.1", port=0, tenants=(TenantSpec(name="svc", token="tok-svc"),)),
        request_host=marking,
    )
    try:
        explained = _client(server, "tok-svc").explain_request("echo", b"x", "q-explain")
    finally:
        server.shutdown(drain_timeout_secs=10)

    assert explained
    assert marking.handed == [ServedRequest(kind="echo", body=b"x", query_id="q-explain", caller="svc")]
    assert not marker.exists()


def test_an_explanation_takes_a_slot(tenant_server: DaftServeServer, tmp_path: pathlib.Path) -> None:
    holder = _hold_slot(tenant_server, "tok-svc", tmp_path)

    with pytest.raises(Exception, match="(?i)capacity"):
        _client(tenant_server, "tok-svc").explain_request("echo", b"x", "q-explain")
    del holder


def test_a_host_failure_fails_the_query() -> None:
    failing = RecordingHost(fail=True)
    server = start_server(ServeSettings(host="127.0.0.1", port=0), request_host=failing)
    try:
        with pytest.raises(Exception, match="refused by its host"):
            _rows(_client(server).run_request("echo", b"x", "q-fail"))
    finally:
        server.shutdown(drain_timeout_secs=10)


def test_a_connection_runs_a_request_into_a_frame() -> None:
    # Connecting sets the process-wide runner, so this runs in its own process.
    script = textwrap.dedent(
        """
        import daft
        from daft.serve import ServeSettings, start_server

        class Host:
            kinds = ("echo",)

            def plan(self, request):
                return daft.from_pydict({"kind": [request.kind], "body": [request.body.decode()]})

        server = start_server(ServeSettings(host="127.0.0.1", port=0), request_host=Host())
        frame = daft.connect(server.address()).request("echo", b"hello")
        assert frame.to_pydict() == {"kind": ["echo"], "body": ["hello"]}, frame.to_pydict()
        server.shutdown(drain_timeout_secs=5)
        print("REQUEST_OK")
        """
    )

    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300, check=False)

    assert "REQUEST_OK" in out.stdout, out.stdout + out.stderr
