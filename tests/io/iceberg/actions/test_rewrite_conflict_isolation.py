"""Conflict-isolation option for rewrite_data_files.

Snapshot isolation (the default) permits a concurrent append into a partition
the rewrite touches and rejects only when one of the rewrite's own input files
is removed or a row-level delete lands on one. Serializable isolation also
rejects the concurrent append.

Each test injects exactly one foreign operation from inside ``_rewrite_group``
-- after the plan snapshot is taken but before the commit -- so the outcome is
deterministic and does not depend on a background thread's cadence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

from daft.catalog import Table
from daft.io.iceberg import RewriteConflict
from tests.io.iceberg.actions._helpers import _row_count, inject_once_around_rewrite, make_seeded_table

if TYPE_CHECKING:
    from pyiceberg.table import Table as PyIcebergTable

_FOREIGN_ROWS = 20


def _append_foreign_rows(table: PyIcebergTable) -> None:
    table.refresh()
    table.append(
        pa.table(
            {
                "id": pa.array(list(range(5_000_000, 5_000_000 + _FOREIGN_ROWS)), type=pa.int64()),
                "label": pa.array(["foreign"] * _FOREIGN_ROWS, type=pa.string()),
            }
        )
    )


def _delete_all_seed_rows(table: PyIcebergTable) -> None:
    table.refresh()
    table.delete(delete_filter="label = 'seed'")


def test_snapshot_isolation_allows_concurrent_partition_append(local_catalog, monkeypatch):
    # Arrange
    table = make_seeded_table(local_catalog, "default.t_iso_snapshot_ok", n_files=6)
    seed_rows = _row_count(table)
    state = inject_once_around_rewrite(monkeypatch, _append_foreign_rows, before=True)
    dt = Table.from_iceberg(table)

    # Act
    result = dt.rewrite_data_files(
        strategy="binpack",
        options={
            "rewrite-all": True,
            "min-input-files": 2,
            "conflict-isolation": "snapshot",
        },
    )

    # Assert: rewrite committed, the foreign append survived, no rows lost.
    assert state["fired"] == 1
    assert result.added_files >= 1
    assert _row_count(table) == seed_rows + _FOREIGN_ROWS


def test_default_isolation_allows_concurrent_partition_append(local_catalog, monkeypatch):
    # Arrange
    table = make_seeded_table(local_catalog, "default.t_iso_default_ok", n_files=6)
    seed_rows = _row_count(table)
    state = inject_once_around_rewrite(monkeypatch, _append_foreign_rows, before=True)
    dt = Table.from_iceberg(table)

    # Act: no isolation level named.
    result = dt.rewrite_data_files(
        strategy="binpack",
        options={"rewrite-all": True, "min-input-files": 2},
    )

    # Assert: the default behaves as snapshot isolation.
    assert state["fired"] == 1
    assert result.commits == 1
    assert _row_count(table) == seed_rows + _FOREIGN_ROWS


def test_serializable_isolation_rejects_concurrent_partition_append(local_catalog, monkeypatch):
    # Arrange
    table = make_seeded_table(local_catalog, "default.t_iso_serializable_conflict", n_files=6)
    state = inject_once_around_rewrite(monkeypatch, _append_foreign_rows, before=True)
    dt = Table.from_iceberg(table)

    # Act / Assert: the strict level rejects the same concurrent append.
    with pytest.raises(RewriteConflict):
        dt.rewrite_data_files(
            strategy="binpack",
            options={
                "rewrite-all": True,
                "min-input-files": 2,
                "conflict-isolation": "serializable",
            },
        )
    assert state["fired"] == 1


def test_snapshot_isolation_still_rejects_vanished_inputs(local_catalog, monkeypatch):
    # Arrange: the foreign op removes the very files the rewrite is replacing.
    table = make_seeded_table(local_catalog, "default.t_iso_snapshot_vanished", n_files=6)
    state = inject_once_around_rewrite(monkeypatch, _delete_all_seed_rows, before=False)
    dt = Table.from_iceberg(table)

    # Act / Assert: snapshot isolation does not mask a vanished-input conflict.
    with pytest.raises(RewriteConflict):
        dt.rewrite_data_files(
            strategy="binpack",
            options={
                "rewrite-all": True,
                "min-input-files": 2,
                "conflict-isolation": "snapshot",
            },
        )
    assert state["fired"] == 1


@pytest.mark.parametrize("bad_value", ["serial", "SNAPSHOT", "", "none"])
def test_invalid_conflict_isolation_value_rejected(local_catalog, bad_value):
    # Arrange
    table = make_seeded_table(local_catalog, f"default.t_iso_bad_{abs(hash(bad_value))}", n_files=2)
    dt = Table.from_iceberg(table)

    # Act / Assert
    with pytest.raises(ValueError, match="conflict-isolation"):
        dt.rewrite_data_files(options={"conflict-isolation": bad_value})
