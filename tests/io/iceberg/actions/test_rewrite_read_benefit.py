"""Whether a compacted table is actually cheaper to read.

Every other check here asks whether the rewrite preserved the data and recorded
the right metadata. None of them ask the question the operator is paying for:
that the table costs less to read afterwards.

Cost is measured as work a reader must do -- files opened, bytes stored -- and
not as elapsed time. Wall clock belongs in the benchmark, where it can be run
deliberately on a quiet machine; asserting on it here would make the suite fail
for reasons that have nothing to do with the code.
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
_COMPACT = {"target-file-size-bytes": 8 * _MIB, "min-input-files": 2}


def _fragmented_table(catalog, name: str, *, n_files: int = 60, rows_per_file: int = 1_500) -> Any:
    """The table an operator actually has: many small files from many appends."""
    rng = random.Random(31)
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
                        ["".join(rng.choices(_ALPHABET, k=48)) for _ in range(rows_per_file)],
                        type=pa.string(),
                    ),
                }
            )
        )
    table.refresh()
    return table


def _live_files(table: Any) -> list[Any]:
    table.refresh()
    return [task.file for task in table.scan().plan_files()]


def _files_opened_for(table: Any, low: int, high: int) -> int:
    """Files a range query over ``id`` cannot prune away.

    Counted from the committed bounds, which is what any reader's planner uses,
    so this measures pruning rather than one engine's execution.
    """
    from pyiceberg.conversions import from_bytes
    from pyiceberg.types import LongType

    field_id = table.schema().find_field("id").field_id
    opened = 0
    for data_file in _live_files(table):
        lower = from_bytes(LongType(), data_file.lower_bounds[field_id])
        upper = from_bytes(LongType(), data_file.upper_bounds[field_id])
        if lower <= high and upper >= low:
            opened += 1
    return opened


def test_compaction_reduces_the_files_a_full_scan_opens(local_catalog):
    """The direct cost of the small-file problem: one open per file."""
    table = _fragmented_table(local_catalog, "default.t_read_files")
    before = len(_live_files(table))

    Table.from_iceberg(table).rewrite_data_files("binpack", options=_COMPACT)

    after = len(_live_files(table))
    assert after * 4 < before, f"{before} files became {after}"


def test_compaction_does_not_grow_the_table_on_disk(local_catalog):
    """Merging small files into large ones compresses at least as well."""
    table = _fragmented_table(local_catalog, "default.t_read_bytes")
    before = sum(f.file_size_in_bytes for f in _live_files(table))

    Table.from_iceberg(table).rewrite_data_files("binpack", options=_COMPACT)

    after = sum(f.file_size_in_bytes for f in _live_files(table))
    assert after <= before, f"table grew: {before} -> {after} bytes"


def test_a_selective_range_query_still_prunes_after_compaction(local_catalog):
    """Fewer, larger files must not cost a selective reader its pruning.

    Compaction preserves row order within a group, so the committed bounds stay
    narrow and a range query opens a fraction of the table rather than all of it.
    """
    table = _fragmented_table(local_catalog, "default.t_read_prune")
    rows = 60 * 1_500

    # A target small enough that the group becomes several files, since pruning
    # between files is what is under test and one file cannot show it.
    Table.from_iceberg(table).rewrite_data_files(
        "binpack", options={"target-file-size-bytes": 1024 * 1024, "min-input-files": 2}
    )

    total = len(_live_files(table))
    assert total > 1, "the fixture must produce several files for pruning to mean anything"
    opened = _files_opened_for(table, 0, rows // 20)
    assert opened < total, f"a 5% range opened every one of {total} files"


def test_every_row_survives_the_rewrite(local_catalog):
    """The cost only matters if the answer is still right."""
    table = _fragmented_table(local_catalog, "default.t_read_rows")
    expected = sorted(table.scan().to_arrow().column("id").to_pylist())

    Table.from_iceberg(table).rewrite_data_files("binpack", options=_COMPACT)

    table.refresh()
    assert sorted(table.scan().to_arrow().column("id").to_pylist()) == expected
