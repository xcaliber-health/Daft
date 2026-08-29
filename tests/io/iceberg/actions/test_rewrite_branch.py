"""Rewrite against a non-default branch ref commits to that branch only."""

from __future__ import annotations

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

from daft.catalog import Table


def test_rewrite_on_branch_leaves_main_untouched(local_catalog, simple_schema):
    from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC

    table = local_catalog.create_table(
        "default.t_branch",
        schema=simple_schema,
        partition_spec=UNPARTITIONED_PARTITION_SPEC,
    )
    for k in range(6):
        table.append(
            pa.table(
                {
                    "id": pa.array(list(range(k * 5, k * 5 + 5)), type=pa.int64()),
                    "label": pa.array([f"r{i}" for i in range(k * 5, k * 5 + 5)]),
                }
            )
        )
    main_snap_before = table.current_snapshot().snapshot_id

    # Create a new branch pointing at the current snapshot.
    branch_name = "feature_x"
    with table.manage_snapshots() as ms:
        ms.create_branch(snapshot_id=main_snap_before, branch_name=branch_name)

    # Sanity: branch exists at the same snapshot.
    assert table.snapshot_by_name(branch_name).snapshot_id == main_snap_before

    dt = Table.from_iceberg(table)
    result = dt.compact_files(
        branch=branch_name,
        options={"rewrite-all": True, "min-input-files": 2},
    )
    table.refresh()

    # Main is untouched.
    assert table.current_snapshot().snapshot_id == main_snap_before
    # Branch advanced to a new snapshot.
    new_branch_snap = table.snapshot_by_name(branch_name).snapshot_id
    assert new_branch_snap != main_snap_before
    assert result.commits >= 1
    assert result.snapshot_ids == [new_branch_snap]


def _rows_on(table, branch: str | None = None) -> int:
    table.refresh()
    snapshot = table.snapshot_by_name(branch) if branch else table.current_snapshot()
    return table.scan(snapshot_id=snapshot.snapshot_id).to_arrow().num_rows


def _diverged_table(catalog, simple_schema, name: str):
    """A branch that has taken its own commit while main has advanced past it."""
    from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC

    def batch(start: int, n: int = 200):
        return pa.table(
            {
                "id": pa.array(range(start, start + n), type=pa.int64()),
                "label": pa.array(["x"] * n, type=pa.large_string()),
            }
        )

    table = catalog.create_table(name, schema=simple_schema, partition_spec=UNPARTITIONED_PARTITION_SPEC)
    for k in range(6):
        table.append(batch(k * 200))
    table.refresh()
    table.manage_snapshots().create_branch(table.current_snapshot().snapshot_id, "work").commit()
    table.refresh()
    table.append(batch(10_000), branch="work")
    table.append(batch(20_000))
    table.refresh()
    assert table.snapshot_by_name("work").snapshot_id != table.current_snapshot().snapshot_id
    return table


@pytest.mark.parametrize("isolation", ["serializable", "snapshot"])
def test_rewrite_on_a_diverged_branch_succeeds(local_catalog, simple_schema, isolation):
    """A branch exists in order to diverge, so divergence must not read as a conflict.

    The commit-time checks resolve the branch being rewritten rather than the
    table's default reference; reading main's ancestry treats every snapshot on
    it as a foreign writer the moment the branch takes a commit of its own.
    """
    table = _diverged_table(local_catalog, simple_schema, f"default.t_diverged_{isolation}")
    main_rows = _rows_on(table)
    work_rows = _rows_on(table, "work")

    result = Table.from_iceberg(table).rewrite_data_files(
        "binpack",
        branch="work",
        options={"rewrite-all": True, "min-input-files": 2, "conflict-isolation": isolation},
    )

    assert result.rewritten_files > 0
    assert _rows_on(table, "work") == work_rows
    assert _rows_on(table) == main_rows, "main must be untouched"
