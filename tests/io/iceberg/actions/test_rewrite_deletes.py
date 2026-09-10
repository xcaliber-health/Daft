"""Delete-handling tests for rewrite_data_files."""

from __future__ import annotations

import pytest

pytest.importorskip("pyiceberg")

from daft.catalog import Table
from daft.daft import _iceberg as _rust_iceberg


def test_equality_deletes_count_toward_the_delete_file_threshold():
    """A file at the target is selected only for its deletes, of either kind."""
    candidates = [
        {
            "path": "/x/a.parquet",
            "size_bytes": 64 * 1024 * 1024,
            "partition_key": "{}",
            "partition_spec_id": 0,
            "positional_delete_paths": [],
            "equality_delete_paths": ["/x/eq.parquet"],
        },
    ]

    groups = _rust_iceberg.plan_file_groups_py(candidates, {"delete-file-threshold": 1}, 0)

    assert [f["path"] for f in groups[0]["files"]] == ["/x/a.parquet"]


def test_positional_deletes_are_merged(make_tiny_table):
    """Run rewrite on a table after a partial delete; deleted rows must not reappear."""
    from pyiceberg.expressions import EqualTo

    table = make_tiny_table(name="default.t_pos_del", n_files=8, rows_per_file=4)
    # Delete a specific row by id; pyiceberg emits a delete via overwrite.
    table.delete(EqualTo("id", 5))
    table.refresh()
    pre_ids = sorted(int(r["id"]) for r in table.scan().to_arrow().to_pylist())
    assert 5 not in pre_ids

    dt = Table.from_iceberg(table)
    result = dt.compact_files(options={"rewrite-all": True, "min-input-files": 2})

    table.refresh()
    post_ids = sorted(int(r["id"]) for r in table.scan().to_arrow().to_pylist())
    assert post_ids == pre_ids
    assert result.added_files >= 1


def test_a_file_scoped_delete_counts_toward_the_delete_ratio(local_catalog):
    """A delete whose path bounds name one file attributes its rows to that file.

    Files inside the size band are picked up on their delete ratio alone, as
    the reference picks them up; a delete spanning several files attributes
    nothing, since its rows cannot be split among them.
    """
    from daft.catalog import Table
    from tests.io.iceberg.actions._helpers import commit_positional_deletes, make_seeded_table, scan_paths

    table = make_seeded_table(local_catalog, "default.t_delete_ratio_scoped", n_files=3, rows_per_file=10)
    target = sorted(scan_paths(table))[0]
    commit_positional_deletes(table, {target: [0, 1, 2, 3, 4]})

    result = Table.from_iceberg(table).rewrite_data_files(
        "binpack",
        options={
            # Nothing is selected on size: every file sits inside the band.
            "min-file-size-bytes": 1,
            "min-input-files": 5,
            "delete-ratio-threshold": 0.3,
        },
    )

    assert result.rewritten_files == 1, "only the half-deleted file is rewritten, on its delete ratio"
    assert target not in scan_paths(table)
