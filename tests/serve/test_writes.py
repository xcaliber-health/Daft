"""Write paths through the serving endpoint.

Write methods execute eagerly on the process-global runner, so genuinely
remote writes are exercised in a dedicated process whose global runner is
the connection. The in-process cases cover the manifest round trip: a
completed write's result set ships with the next plan touching it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest

import daft

from .conftest import collect_via

if TYPE_CHECKING:
    import pathlib

    from daft.runners.native_runner import NativeRunner
    from daft.runners.remote_runner import RemoteRunner


def _run_connected(script_body: str) -> None:
    """Run a script in a fresh process wired to a dedicated server."""
    script = (
        textwrap.dedent(
            """
        import daft
        from daft.serve import ServeSettings, start_server

        server = start_server(ServeSettings(host="127.0.0.1", port=0))
        daft.connect(server.address())
        """
        )
        + textwrap.dedent(script_body)
        + textwrap.dedent(
            """
        server.shutdown(drain_timeout_secs=5)
        print("WRITE_OK")
        """
        )
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600, check=False)
    assert "WRITE_OK" in out.stdout, out.stdout + out.stderr


def test_write_parquet_executes_through_connection(tmp_path: pathlib.Path) -> None:
    target = str(tmp_path / "out")
    _run_connected(
        f"""
        df = daft.from_pydict({{"a": [1, 2, 3], "b": ["x", "y", "z"]}})
        manifest = df.write_parquet({target!r})
        paths = manifest.to_pydict()["path"]
        assert len(paths) >= 1, paths
        """
    )
    written = pq.read_table(target)
    assert written.num_rows == 3
    assert sorted(written.column("a").to_pylist()) == [1, 2, 3]


def test_write_csv_executes_through_connection(tmp_path: pathlib.Path) -> None:
    target = str(tmp_path / "csv_out")
    _run_connected(
        f"""
        df = daft.from_pydict({{"a": [1, 2]}})
        manifest = df.write_csv({target!r})
        assert len(manifest.to_pydict()["path"]) >= 1
        back = daft.read_csv({target!r}).sort("a").to_pydict()
        assert back == {{"a": [1, 2]}}, back
        """
    )


def test_write_iceberg_end_to_end(tmp_path: pathlib.Path) -> None:
    """A table-format write executed end to end through the connection.

    Files are written by the serving path, the manifest streams back, and
    the snapshot commit happens on the client.
    """
    pytest.importorskip("pyiceberg")
    _run_connected(
        f"""
        from pyiceberg.catalog.sql import SqlCatalog

        warehouse = {str(tmp_path)!r}
        catalog = SqlCatalog(
            "wtest", uri=f"sqlite:///{{warehouse}}/catalog.db", warehouse=f"file://{{warehouse}}"
        )
        catalog.create_namespace("db")

        df = daft.from_pydict({{"id": [1, 2, 3], "name": ["a", "b", "c"]}})
        table = catalog.create_table("db.people", schema=df.to_arrow().schema)
        df.write_iceberg(table)

        back = daft.read_iceberg(catalog.load_table("db.people")).sort("id").to_pydict()
        assert back == {{"id": [1, 2, 3], "name": ["a", "b", "c"]}}, back
        """
    )


def test_write_manifest_round_trips_with_next_plan(tmp_path: pathlib.Path, remote_runner: RemoteRunner) -> None:
    """A completed write's manifest round-trips with the next plan.

    The manifest is cached in memory; a remote plan over it ships the
    cached set and returns it unchanged.
    """
    target = str(tmp_path / "manifest")
    df = daft.from_pydict({"a": [1, 2, 3]}).write_parquet(target)

    manifest = collect_via(remote_runner, df)

    assert len(manifest["path"]) >= 1
    assert pq.read_table(target).num_rows == 3


def test_write_then_read_round_trip(
    tmp_path: pathlib.Path, remote_runner: RemoteRunner, native_runner: NativeRunner
) -> None:
    target = str(tmp_path / "roundtrip")
    daft.from_pydict({"n": list(range(20))}).write_parquet(target)

    df = daft.read_parquet(target).sort("n")
    got = collect_via(remote_runner, df)
    expected = collect_via(native_runner, df)
    assert got == expected
