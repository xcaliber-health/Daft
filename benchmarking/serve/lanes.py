"""Execution lanes: the four engine configurations under comparison.

Each lane exposes the same interface — run a named TPC-H query and return
row count plus wall-clock seconds — so the bench driver and the concurrency
harness treat all engines identically:

- ``daft-inprocess``: DataFrame queries on the local engine.
- ``daft-serve``: the same DataFrame queries through the query server.
- ``duckdb-inprocess``: stock TPC-H SQL over table-format scans.
- ``duckdb-quack``: the same SQL through a quack server process.
"""

from __future__ import annotations

import logging
import pathlib
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog

logger = logging.getLogger(__name__)

QUERY_NAMES = tuple(f"q{number}" for number in range(1, 23))
_TPCH_QUERY_IDS = {name: int(name[1:]) for name in QUERY_NAMES}


class Lane(Protocol):
    """One engine configuration under benchmark."""

    name: str

    def run(self, query: str) -> tuple[int, float]:
        """Execute a named query; returns (row_count, elapsed_seconds)."""
        ...

    def rows(self, query: str) -> list[tuple[object, ...]]:
        """Execute a named query and return its full sorted result rows."""
        ...

    def close(self) -> None:
        """Release lane resources."""
        ...


def _load_catalog(warehouse: pathlib.Path) -> Catalog:
    from pyiceberg.catalog.sql import SqlCatalog

    return SqlCatalog(
        "bench",
        uri=f"sqlite:///{warehouse}/catalog.db",
        warehouse=f"file://{warehouse}",
    )


class _DaftLane:
    """Shared implementation for the DataFrame lanes."""

    def __init__(self, warehouse: pathlib.Path, runner: object) -> None:
        self._catalog = _load_catalog(warehouse)
        self._runner = runner
        self._tables: dict[str, object] = {}

    def _get_df(self, name: str) -> object:
        import daft

        if name not in self._tables:
            self._tables[name] = self._catalog.load_table(f"tpch.{name}")
        return daft.read_iceberg(self._tables[name])

    def _collect(self, query: str) -> list[dict[str, list[object]]]:
        from benchmarking.tpch import answers

        df = getattr(answers, query)(self._get_df)
        parts = list(self._runner.run_iter_tables(df._builder))
        return [part.to_pydict() for part in parts]

    def run(self, query: str) -> tuple[int, float]:
        start = time.perf_counter()
        parts = self._collect(query)
        elapsed = time.perf_counter() - start
        rows = sum(len(next(iter(p.values()), [])) for p in parts)
        return rows, elapsed

    def rows(self, query: str) -> list[tuple[object, ...]]:
        merged: dict[str, list[object]] = {}
        for part in self._collect(query):
            for column, values in part.items():
                merged.setdefault(column, []).extend(values)
        if not merged:
            return []
        # Both engines emit the query-defined column order; compare
        # positionally since output column names differ across dialects.
        return sorted(
            zip(*merged.values()),
            key=lambda row: tuple((v is None, str(v)) for v in row),
        )

    def close(self) -> None:
        return None


class DaftInProcessLane(_DaftLane):
    """DataFrame queries executed by the local engine."""

    name = "daft-inprocess"

    def __init__(self, warehouse: pathlib.Path) -> None:
        from daft.runners.native_runner import NativeRunner

        super().__init__(warehouse, NativeRunner())


class DaftServeLane(_DaftLane):
    """DataFrame queries shipped to a query server."""

    name = "daft-serve"

    def __init__(self, warehouse: pathlib.Path, address: str, token: str | None = None) -> None:
        from daft.runners.remote_runner import RemoteRunner

        super().__init__(warehouse, RemoteRunner(address, token))


def _tpch_sql(query: str) -> str:
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch")
    number = _TPCH_QUERY_IDS[query]
    row = con.execute("SELECT query FROM tpch_queries() WHERE query_nr = ?", [number]).fetchone()
    if row is None:
        raise ValueError(f"unknown TPC-H query {query}")
    return str(row[0])


def _create_views(con: object, locations: dict[str, str]) -> None:
    for name, metadata in locations.items():
        con.execute(  # type: ignore[attr-defined]
            f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM iceberg_scan('{metadata}')"
        )


class DuckDBInProcessLane:
    """Stock TPC-H SQL over table-format scans, in process."""

    name = "duckdb-inprocess"

    def __init__(self, locations: dict[str, str]) -> None:
        import duckdb

        self._con = duckdb.connect()
        self._con.execute("INSTALL iceberg; LOAD iceberg; INSTALL tpch; LOAD tpch")
        self._con.execute("SET unsafe_enable_version_guessing=true")
        _create_views(self._con, locations)

    def run(self, query: str) -> tuple[int, float]:
        sql = _tpch_sql(query)
        start = time.perf_counter()
        result = self._con.execute(sql).fetchall()
        return len(result), time.perf_counter() - start

    def rows(self, query: str) -> list[tuple[object, ...]]:
        sql = _tpch_sql(query)
        rows = [tuple(row) for row in self._con.execute(sql).fetchall()]
        return sorted(rows, key=lambda row: tuple((v is None, str(v)) for v in row))

    def close(self) -> None:
        self._con.close()


_QUACK_SERVER_SCRIPT = """
import json, sys, time
import duckdb

locations = json.loads(sys.argv[1])
port = int(sys.argv[2])
con = duckdb.connect()
con.execute("INSTALL quack; LOAD quack; INSTALL iceberg; LOAD iceberg")
con.execute("SET unsafe_enable_version_guessing=true")
for name, metadata in locations.items():
    con.execute(
        f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM iceberg_scan('{metadata}')"
    )
con.execute(f"CALL quack_serve('quack:localhost:{port}', token = 'bench')")
print("QUACK_READY", flush=True)
while True:
    time.sleep(1)
"""


def start_quack_server(locations: dict[str, str], port: int) -> subprocess.Popen[str]:
    """Start a quack server process serving views over the given tables.

    Returns the server process once it reports readiness; the caller owns
    its lifetime.
    """
    import json

    proc = subprocess.Popen(
        [sys.executable, "-c", _QUACK_SERVER_SCRIPT, json.dumps(locations), str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    deadline = time.monotonic() + 120
    lines: list[str] = []
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        lines.append(line)
        if "QUACK_READY" in line:
            return proc
    proc.kill()
    raise RuntimeError(f"quack server did not start: {''.join(lines)}")


class DuckDBQuackLane:
    """Stock TPC-H SQL executed on a quack server in another process.

    When constructed without an existing port, a dedicated server process is
    started and owned by the lane.
    """

    name = "duckdb-quack"

    def __init__(
        self,
        locations: dict[str, str] | None = None,
        port: int = 9595,
        existing_server: bool = False,
    ) -> None:
        import duckdb

        self._proc: subprocess.Popen[str] | None = None
        if not existing_server:
            if locations is None:
                raise ValueError("locations are required when starting a server")
            self._proc = start_quack_server(locations, port)

        self._con = duckdb.connect()
        self._con.execute("INSTALL quack; LOAD quack; INSTALL tpch; LOAD tpch")
        self._con.execute("CREATE SECRET quack_secret (TYPE quack, TOKEN 'bench')")
        self._con.execute(f"ATTACH 'quack:localhost:{port}' AS remote")

    def _execute(self, query: str) -> object:
        sql = _tpch_sql(query).replace("'", "''")
        # The remote query function ships the statement whole, executing it
        # server-side in a single round trip.
        return self._con.execute(f"SELECT * FROM remote.query('{sql}')")

    def run(self, query: str) -> tuple[int, float]:
        start = time.perf_counter()
        result = self._execute(query).fetchall()
        return len(result), time.perf_counter() - start

    def rows(self, query: str) -> list[tuple[object, ...]]:
        rows = [tuple(row) for row in self._execute(query).fetchall()]
        return sorted(rows, key=lambda row: tuple((v is None, str(v)) for v in row))

    def close(self) -> None:
        self._con.close()
        if self._proc is not None:
            self._proc.kill()
            self._proc.wait(timeout=30)
