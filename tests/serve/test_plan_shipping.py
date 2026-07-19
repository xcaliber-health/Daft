"""Visibility of where scan planning runs for a shipped plan.

Sources that cannot travel force scan planning onto the client; that
fallback must be surfaced to the user rather than silent.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import daft
from daft import col

from .conftest import collect_via

if TYPE_CHECKING:
    import pathlib

    from daft.runners.remote_runner import RemoteRunner


def test_native_scan_fallback_is_logged(
    tmp_path: pathlib.Path, remote_runner: RemoteRunner, caplog: logging.LogCaptureFixture
) -> None:
    path = str(tmp_path / "data.parquet")
    daft.from_pydict({"n": list(range(10))}).write_parquet(path)
    df = daft.read_parquet(path).where(col("n") > 3)

    with caplog.at_level(logging.INFO, logger="daft.runners.remote_runner"):
        got = collect_via(remote_runner, df)

    assert sorted(got["n"]) == [4, 5, 6, 7, 8, 9]
    assert any("scan planning ran" in record.message for record in caplog.records)


def test_in_memory_plans_ship_without_fallback(remote_runner: RemoteRunner, caplog: logging.LogCaptureFixture) -> None:
    df = daft.from_pydict({"a": [1, 2, 3]}).where(col("a") > 1)

    with caplog.at_level(logging.INFO, logger="daft.runners.remote_runner"):
        got = collect_via(remote_runner, df)

    assert got == {"a": [2, 3]}
    assert not any("scan planning ran" in record.message for record in caplog.records)


def test_explain_notes_client_side_planning(tmp_path: pathlib.Path, remote_runner: RemoteRunner) -> None:
    path = str(tmp_path / "data.parquet")
    daft.from_pydict({"n": [1, 2, 3]}).write_parquet(path)
    df = daft.read_parquet(path)

    explained = remote_runner.client.explain_plan(df._builder._builder)
    assert "scan planning" in explained
