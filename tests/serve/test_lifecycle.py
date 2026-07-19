"""Process lifecycle: out-of-process serving, connect API, crash handling."""

from __future__ import annotations

import re
import signal
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator

import pytest

import daft
from daft.runners.remote_runner import RemoteRunner

from .conftest import collect_via

_ADDRESS_PATTERN = re.compile(r"serving on (grpc://\S+)")


@pytest.fixture()
def subprocess_server() -> Iterator[tuple[subprocess.Popen[str], str]]:
    """A real server subprocess bound to an ephemeral port."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "daft.serve", "--host", "127.0.0.1", "--port", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    address = None
    deadline = time.monotonic() + 60
    assert proc.stdout is not None
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        match = _ADDRESS_PATTERN.search(line)
        if match:
            address = match.group(1)
            break
    if address is None:
        proc.kill()
        pytest.fail("server subprocess did not report its address")
    yield proc, address
    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=30)


def test_cross_process_query_execution(subprocess_server: tuple[subprocess.Popen[str], str]) -> None:
    _, address = subprocess_server
    runner = RemoteRunner(address)
    df = daft.from_pydict({"a": [1, 2, 3]})
    assert collect_via(runner, df) == {"a": [1, 2, 3]}


def test_killed_server_raises_connection_error_not_hang(
    subprocess_server: tuple[subprocess.Popen[str], str],
) -> None:
    proc, address = subprocess_server
    runner = RemoteRunner(address)
    proc.kill()
    proc.wait(timeout=30)

    df = daft.from_pydict({"a": [1, 2, 3]})
    with pytest.raises(Exception, match="(?i)connect|transport|status|error|unavailable"):
        collect_via(runner, df)


def test_sigterm_drains_and_exits(subprocess_server: tuple[subprocess.Popen[str], str]) -> None:
    proc, address = subprocess_server
    runner = RemoteRunner(address)
    df = daft.from_pydict({"a": [1, 2, 3]})
    assert collect_via(runner, df) == {"a": [1, 2, 3]}

    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=60) == 0


def test_connect_sets_global_runner_in_dedicated_process() -> None:
    """`daft.connect` wires the process-global runner to the server."""
    script = textwrap.dedent(
        """
        import daft
        from daft.serve import ServeSettings, start_server

        server = start_server(ServeSettings(host="127.0.0.1", port=0))
        conn = daft.connect(server.address())
        df = daft.from_pydict({"a": [1, 2, 3]})
        assert df.where(df["a"] > 1).collect().to_pydict() == {"a": [2, 3]}
        assert daft.get_or_infer_runner_type() == "remote"
        info = conn.server_info()
        assert info.plan_payload_enabled
        server.shutdown(drain_timeout_secs=5)
        print("CONNECT_OK")
        """
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300, check=False)
    assert "CONNECT_OK" in out.stdout, out.stdout + out.stderr


def test_connect_sql_helper_materializes_results() -> None:
    script = textwrap.dedent(
        """
        import daft
        from daft.serve import ServeSettings, start_server

        server = start_server(ServeSettings(host="127.0.0.1", port=0))
        conn = daft.connect(server.address())
        df = conn.sql("select 40 + 2 as answer")
        assert df.to_pydict() == {"answer": [42]}
        server.shutdown(drain_timeout_secs=5)
        print("SQL_OK")
        """
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300, check=False)
    assert "SQL_OK" in out.stdout, out.stdout + out.stderr
