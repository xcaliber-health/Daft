"""The textual query lane: planned and executed entirely server-side."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from daft.recordbatch import MicroPartition

if TYPE_CHECKING:
    from daft.runners.remote_runner import RemoteRunner


def _run_sql(runner: RemoteRunner, query: str) -> dict[str, list[object]]:
    result = runner.client.run_sql(query, f"sql-{abs(hash(query)) % 10**9}")
    merged: dict[str, list[object]] = {}
    for part in result:
        for column, values in MicroPartition._from_pymicropartition(part).to_pydict().items():
            merged.setdefault(column, []).extend(values)
    return merged


def test_literal_select(remote_runner: RemoteRunner) -> None:
    assert _run_sql(remote_runner, "select 1 as one, 'x' as s") == {"one": [1], "s": ["x"]}


def test_expression_evaluation(remote_runner: RemoteRunner) -> None:
    got = _run_sql(remote_runner, "select 2 + 3 as total")
    assert got == {"total": [5]}


def test_malformed_query_returns_clean_error(remote_runner: RemoteRunner) -> None:
    with pytest.raises(Exception, match="(?i)sql|parse|syntax|error"):
        _run_sql(remote_runner, "selectt oops from nowhere")


def test_unknown_table_returns_clean_error(remote_runner: RemoteRunner) -> None:
    with pytest.raises(Exception, match="(?i)table|catalog|not found|missing|error"):
        _run_sql(remote_runner, "select * from missing_catalog.missing_table")
