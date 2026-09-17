"""Provenance of rows read for a row-level write.

Every row must name the data file it came from and its position in that file,
and those must stay correct once the file's own deletes are applied. This covers
the read seam the row-level operations are built on; their own behaviour is
covered through the table API.
"""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest

pytest.importorskip("pyiceberg")

from daft.daft import IOConfig
from daft.io.iceberg._deletes import plan_files
from daft.io.iceberg._row_level import FileTable, provenance_scan

from ._helpers import commit_positional_deletes, make_seeded_table


@pytest.fixture
def seeded(local_catalog):
    return make_seeded_table(local_catalog, "default.provenance", n_files=3, rows_per_file=5)


def _read(table, columns=None):
    plan = plan_files(table, table.scan())
    file_table = FileTable.from_plan(plan)
    frame = provenance_scan(
        table=table,
        plan=plan,
        file_table=file_table,
        paths=file_table.paths,
        io_config=IOConfig(),
    )
    if columns is not None:
        frame = frame.select(*columns)
    return plan, file_table, frame.to_pydict()


def test_rows_carry_their_file_and_position(seeded):
    _plan, file_table, rows = _read(seeded)

    assert len(rows["id"]) == 15
    by_file: dict[int, list[int]] = {}
    for file_index, position in zip(rows["_file_idx"], rows["_pos"]):
        by_file.setdefault(file_index, []).append(position)
    assert len(by_file) == 3, "three appends produce three data files"
    for positions in by_file.values():
        assert sorted(positions) == list(range(5)), "positions are ordinals within one file"
    assert set(by_file) == set(range(len(file_table))), "file indices address the file table"


def test_file_index_resolves_to_the_file_that_holds_the_row(seeded):
    _plan, file_table, rows = _read(seeded)

    ids_by_path: dict[str, set[int]] = {}
    for row_id, file_index in zip(rows["id"], rows["_file_idx"]):
        ids_by_path.setdefault(file_table.entry(file_index).path, set()).add(row_id)

    for path, ids in ids_by_path.items():
        written = pq.read_table(path.replace("file://", "")).column("id").to_pylist()
        assert ids == set(written)


def test_positions_survive_existing_deletes(seeded):
    plan, file_table, _rows = _read(seeded)
    first = file_table.entry(0).path
    commit_positional_deletes(seeded, {first: [1, 3]})

    _plan, file_table, rows = _read(seeded.refresh())

    kept = [
        position
        for position, file_index in zip(rows["_pos"], rows["_file_idx"])
        if file_table.entry(file_index).path == first
    ]
    assert sorted(kept) == [0, 2, 4], "a deleted row shifts no other row's position"
    assert len(rows["id"]) == 13


def test_selecting_keys_keeps_provenance_available(seeded):
    _plan, _file_table, rows = _read(seeded, columns=["id", "_file_idx"])

    assert set(rows) == {"id", "_file_idx"}
    assert len(rows["id"]) == 15


def test_plan_is_empty_for_an_empty_table(local_catalog, simple_schema):
    from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC

    table = local_catalog.create_table(
        identifier="default.empty",
        schema=simple_schema,
        partition_spec=UNPARTITIONED_PARTITION_SPEC,
    )

    plan, file_table, rows = _read(table)

    assert len(plan.tasks) == 0
    assert len(file_table) == 0
    assert rows["id"] == []
