"""Rewrites racing a merge-on-read delete.

A merge-on-read delete does not remove the data file it applies to. It adds a
delete file naming that file and the row positions inside it. A rewrite that
replaces the data file leaves the delete matching nothing, so the rows it
removed come back unless the rewrite refuses to commit.

Each test lands exactly one delete between the rewrite's read and its commit,
so the outcome is deterministic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ParamSpec, TypeVar, cast

import pytest

pytest.importorskip("pyiceberg")

from daft.catalog import Table
from daft.io.iceberg import (
    IcebergMaintenanceOptions,
    RewriteConflict,
    _compact,  # internal helper, monkeypatched below
)
from tests.io.iceberg.actions._helpers import commit_positional_deletes, make_seeded_table

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyiceberg.table import Table as PyIcebergTable

_P = ParamSpec("_P")
_R = TypeVar("_R")
_ROWS_PER_FILE = 100
_DELETED_POSITIONS = list(range(10))


def _rewrite_options(isolation: str) -> IcebergMaintenanceOptions:
    return {
        "rewrite-all": True,
        "min-input-files": 2,
        "conflict-isolation": isolation,
    }


def _delete_from_the_first_input_file(table: PyIcebergTable) -> None:
    table.refresh()
    first = sorted(task.file.file_path for task in table.scan().plan_files())[0]
    commit_positional_deletes(table, {first: _DELETED_POSITIONS})


def _inject_after_the_group_is_read(
    monkeypatch: pytest.MonkeyPatch, action: Callable[[PyIcebergTable], None]
) -> dict[str, int]:
    """Run ``action`` once, after the first group is read but before it commits."""
    state = {"fired": 0}

    def instrument(real_rewrite_group: Callable[_P, _R]) -> Callable[_P, _R]:
        def instrumented(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            out = real_rewrite_group(*args, **kwargs)
            if state["fired"] == 0:
                state["fired"] = 1
                action(cast("PyIcebergTable", kwargs["table"]))
            return out

        return instrumented

    monkeypatch.setattr(_compact, "_rewrite_group", instrument(_compact._rewrite_group))
    return state


def _live_ids(table: PyIcebergTable) -> set[int]:
    table.refresh()
    return {int(v) for v in table.scan().to_arrow().column("id").to_pylist()}


@pytest.mark.parametrize("isolation", ["serializable", "snapshot"])
def test_a_concurrent_delete_is_never_undone(local_catalog, monkeypatch, isolation):
    """Neither level may resurrect rows a delete has already removed."""
    table = make_seeded_table(local_catalog, f"default.t_del_{isolation}", n_files=6, rows_per_file=_ROWS_PER_FILE)
    before = _live_ids(table)
    state = _inject_after_the_group_is_read(monkeypatch, _delete_from_the_first_input_file)
    dt = Table.from_iceberg(table)

    with pytest.raises(RewriteConflict):
        dt.rewrite_data_files("binpack", options=_rewrite_options(isolation))

    assert state["fired"] == 1
    after = _live_ids(table)
    assert len(after) == len(before) - len(_DELETED_POSITIONS)
    assert after < before


def test_a_delete_on_an_untouched_file_does_not_block_the_rewrite(local_catalog, monkeypatch):
    """The refusal is aimed at the files being replaced, not at deletes in general."""
    table = make_seeded_table(local_catalog, "default.t_del_untouched", n_files=6, rows_per_file=_ROWS_PER_FILE)
    table.refresh()
    untouched = sorted(task.file.file_path for task in table.scan().plan_files())[-1]
    commit_positional_deletes(table, {untouched: _DELETED_POSITIONS})
    before = _live_ids(table)
    dt = Table.from_iceberg(table)

    result = dt.rewrite_data_files("binpack", options=_rewrite_options("snapshot"))

    assert result.added_files >= 1
    assert _live_ids(table) == before


def test_a_delete_outside_the_rewrite_scope_does_not_block_it(local_catalog, monkeypatch):
    """A delete is only a conflict for the files the rewrite actually replaces."""
    table = make_seeded_table(local_catalog, "default.t_del_out_of_scope", n_files=6, rows_per_file=_ROWS_PER_FILE)
    table.refresh()
    in_scope = {task.file.file_path for task in table.scan(row_filter="id < 300").plan_files()}
    out_of_scope = sorted(
        task.file.file_path for task in table.scan().plan_files() if task.file.file_path not in in_scope
    )
    assert out_of_scope, "the fixture must leave files outside the filtered rewrite"

    state = _inject_after_the_group_is_read(
        monkeypatch, lambda t: commit_positional_deletes(t, {out_of_scope[-1]: _DELETED_POSITIONS})
    )
    dt = Table.from_iceberg(table)

    result = dt.rewrite_data_files("binpack", where="id < 300", options=_rewrite_options("snapshot"))

    assert state["fired"] == 1
    assert result.added_files >= 1
    # The delete stayed in force: its rows are gone and nothing was resurrected.
    assert len(_live_ids(table)) == 600 - len(_DELETED_POSITIONS)
