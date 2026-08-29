"""Behavior of a rewrite that only appears on a distributed runner.

A distributed write emits one file per partition, and the scan starts with one
partition per input file. Left alone, a group of many small files is written as
many independent streams, each leaving its own undersized remainder, so a
rewrite meant to reduce the file count can raise it instead. A single-node run
never sees this: the whole group streams through one writer that rolls by size.

These assertions hold on either runner. What differs is that only the
distributed one can fail them.
"""

from __future__ import annotations

import random
from typing import Any

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

from daft.catalog import Table

_MIB = 1024 * 1024
_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def _many_small_files(catalog, name: str, *, n_files: int, rows_per_file: int) -> Any:
    """Seed a table whose files are all well under the rewrite target."""
    rng = random.Random(23)
    table = catalog.create_table(
        identifier=name,
        schema=pa.schema([("id", pa.int64()), ("pad", pa.string())]),
        properties={"format-version": "2"},
    )
    for batch in range(n_files):
        start = batch * rows_per_file
        table.append(
            pa.table(
                {
                    "id": pa.array(range(start, start + rows_per_file), type=pa.int64()),
                    "pad": pa.array(
                        ["".join(rng.choices(_ALPHABET, k=64)) for _ in range(rows_per_file)],
                        type=pa.string(),
                    ),
                }
            )
        )
    table.refresh()
    return table


def _sizes(table: Any) -> list[int]:
    table.refresh()
    return sorted(task.file.file_size_in_bytes for task in table.scan().plan_files())


def test_a_rewrite_never_increases_the_file_count(local_catalog):
    """The one thing a compaction must not do is fragment the table further."""
    table = _many_small_files(local_catalog, "default.t_dist_count", n_files=24, rows_per_file=7_000)
    before = _sizes(table)

    Table.from_iceberg(table).rewrite_data_files(
        "binpack",
        options={"target-file-size-bytes": 2 * _MIB, "min-input-files": 2, "rewrite-all": True},
    )

    after = _sizes(table)
    assert len(after) < len(before), f"{len(before)} files became {len(after)}"
    assert sum(after) > 0


def test_merged_files_do_not_leave_undersized_remainders(local_catalog):
    """One remainder for the group, not one per input file."""
    table = _many_small_files(local_catalog, "default.t_dist_remainder", n_files=24, rows_per_file=7_000)

    Table.from_iceberg(table).rewrite_data_files(
        "binpack",
        options={"target-file-size-bytes": 2 * _MIB, "min-input-files": 2, "rewrite-all": True},
    )

    sizes = _sizes(table)
    undersized = [s for s in sizes if s < _MIB]
    assert len(undersized) <= 1, f"at most one remainder is expected, got {undersized} of {sizes}"


def test_rows_survive_the_distributed_write(local_catalog):
    """Repartitioning before the write must not drop or duplicate a row."""
    table = _many_small_files(local_catalog, "default.t_dist_rows", n_files=12, rows_per_file=5_000)
    expected = sorted(table.scan().to_arrow().column("id").to_pylist())

    Table.from_iceberg(table).rewrite_data_files(
        "binpack",
        options={"target-file-size-bytes": 2 * _MIB, "min-input-files": 2, "rewrite-all": True},
    )

    table.refresh()
    assert sorted(table.scan().to_arrow().column("id").to_pylist()) == expected


def test_forcing_a_rewrite_of_at_target_files_keeps_every_row(local_catalog):
    """D052: rewriting files already at the target may add files, never lose rows.

    A distributed write emits a file per partition, so a partition larger than
    the target rolls and leaves its own remainder, and remainders cannot be
    merged across partitions. Forcing this with ``rewrite-all`` on files that
    are already well sized can therefore raise the file count rather than lower
    it. The planner does not select such files on its own. What must hold either
    way is that every row survives.
    """
    table = _many_small_files(local_catalog, "default.t_dist_attarget", n_files=6, rows_per_file=45_000)
    before = _sizes(table)
    assert min(before) > 2 * _MIB * 0.9, "the inputs must already be near the target"
    expected = sorted(table.scan().to_arrow().column("id").to_pylist())

    Table.from_iceberg(table).rewrite_data_files(
        "binpack",
        options={"target-file-size-bytes": 2 * _MIB, "min-input-files": 2, "rewrite-all": True},
    )

    table.refresh()
    assert sorted(table.scan().to_arrow().column("id").to_pylist()) == expected
    assert sum(_sizes(table)) > 0
