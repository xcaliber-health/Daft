from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

from daft.recordbatch import MicroPartition
from daft.runners.native_runner import NativeRunner
from daft.runners.remote_runner import RemoteRunner
from daft.serve import ServeSettings, start_server

if TYPE_CHECKING:
    from daft.daft import DaftServeServer
    from daft.dataframe import DataFrame
    from daft.runners.runner import Runner


@pytest.fixture(scope="session")
def serve_server() -> Iterator[DaftServeServer]:
    """A serving process bound to an ephemeral loopback port."""
    server = start_server(ServeSettings(host="127.0.0.1", port=0))
    yield server
    server.shutdown(drain_timeout_secs=10)


@pytest.fixture(scope="session")
def remote_runner(serve_server: DaftServeServer) -> RemoteRunner:
    """A runner executing on the session server."""
    return RemoteRunner(serve_server.address())


@pytest.fixture(scope="session")
def native_runner() -> NativeRunner:
    """An in-process runner used as ground truth."""
    return NativeRunner()


def collect_via(runner: Runner[MicroPartition], df: DataFrame) -> dict[str, list[object]]:
    """Execute a dataframe's plan on a specific runner and materialize it.

    Bypasses the process-global runner so remote and local execution can be
    compared within one process.
    """
    parts = list(runner.run_iter_tables(df._builder))
    if not parts:
        return {column: [] for column in df.column_names}
    merged: dict[str, list[object]] = {column: [] for column in parts[0].column_names()}
    for part in parts:
        for column, values in part.to_pydict().items():
            merged[column].extend(values)
    return merged


def assert_remote_matches_native(
    remote: RemoteRunner,
    native: NativeRunner,
    df: DataFrame,
    sort_key: str | None = None,
) -> None:
    """Assert a plan produces identical results remotely and locally.

    When ``sort_key`` is given, rows are aligned on it first so plans with
    non-deterministic row order compare stably.
    """
    got = collect_via(remote, df)
    expected = collect_via(native, df)
    assert set(got.keys()) == set(expected.keys())
    if sort_key is not None:
        got = _sort_columns(got, sort_key)
        expected = _sort_columns(expected, sort_key)
    assert got == expected


def _sort_columns(columns: dict[str, list[object]], key: str) -> dict[str, list[object]]:
    order = sorted(range(len(columns[key])), key=lambda i: (columns[key][i] is None, columns[key][i]))
    return {name: [values[i] for i in order] for name, values in columns.items()}
