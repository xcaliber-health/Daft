"""Writes execute server-side; the client only receives the write manifest."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow.parquet as pq

import daft

from .conftest import collect_via

if TYPE_CHECKING:
    import pathlib

    from daft.runners.native_runner import NativeRunner
    from daft.runners.remote_runner import RemoteRunner


def test_write_parquet_runs_on_server(tmp_path: pathlib.Path, remote_runner: RemoteRunner) -> None:
    target = str(tmp_path / "out")
    df = daft.from_pydict({"a": [1, 2, 3], "b": ["x", "y", "z"]}).write_parquet(target)

    manifest = collect_via(remote_runner, df)

    assert len(manifest["path"]) >= 1
    written = pq.read_table(target)
    assert written.num_rows == 3
    assert sorted(written.column("a").to_pylist()) == [1, 2, 3]


def test_write_then_read_round_trip(
    tmp_path: pathlib.Path, remote_runner: RemoteRunner, native_runner: NativeRunner
) -> None:
    target = str(tmp_path / "roundtrip")
    collect_via(remote_runner, daft.from_pydict({"n": list(range(20))}).write_parquet(target))

    df = daft.read_parquet(target).sort("n")
    got = collect_via(remote_runner, df)
    expected = collect_via(native_runner, df)
    assert got == expected


def test_write_csv_runs_on_server(tmp_path: pathlib.Path, remote_runner: RemoteRunner) -> None:
    target = str(tmp_path / "csv_out")
    df = daft.from_pydict({"a": [1, 2]}).write_csv(target)

    manifest = collect_via(remote_runner, df)

    assert len(manifest["path"]) >= 1
