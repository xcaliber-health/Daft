"""What a rewrite records in the snapshot it commits.

A rewrite changes how rows are stored, not what they are. Three things in the
committed metadata have to say so: the snapshot is a replace rather than an
overwrite, its file and record totals are recomputed, the added entries carry
the sequence number of the snapshot the rewrite read from, and each manifest
declares the partitioning its files were actually written under.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pyiceberg")

from pyiceberg.manifest import ManifestEntryStatus
from pyiceberg.transforms import IdentityTransform

from daft.catalog import Table
from tests.io.iceberg.actions._helpers import make_seeded_table

_REWRITE_ALL = {"rewrite-all": True, "min-input-files": 2}


def _added_entry_sequence_numbers(table: Any) -> set[int | None]:
    snapshot = table.current_snapshot()
    return {
        entry.sequence_number
        for manifest in snapshot.manifests(table.io)
        for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=False)
        if entry.status == ManifestEntryStatus.ADDED
    }


def _spec_ids_of_added_manifests(table: Any) -> set[int]:
    """Spec ids of the manifests holding this snapshot's added entries.

    A rewrite also writes manifests marking the replaced files deleted, and
    those keep the spec their files were written under, so they are excluded.
    """
    snapshot = table.current_snapshot()
    return {
        manifest.partition_spec_id
        for manifest in snapshot.manifests(table.io)
        if any(
            entry.status == ManifestEntryStatus.ADDED
            for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=False)
        )
    }


def test_a_rewrite_commits_a_replace_snapshot(local_catalog):
    """A replace says the data did not change, which is what a rewrite means."""
    table = make_seeded_table(local_catalog, "default.t_replace", n_files=6)

    Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL)

    table.refresh()
    assert table.current_snapshot().summary.operation.value == "replace"


def test_the_snapshot_totals_are_recomputed(local_catalog):
    """A rewrite does change file counts and sizes, so the totals must follow."""
    table = make_seeded_table(local_catalog, "default.t_totals", n_files=6, rows_per_file=100)

    Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL)

    table.refresh()
    summary = table.current_snapshot().summary
    live_files = list(table.scan().plan_files())
    assert int(summary["total-records"]) == 600
    assert int(summary["total-data-files"]) == len(live_files)
    assert int(summary["total-files-size"]) == sum(task.file.file_size_in_bytes for task in live_files)


def test_added_files_carry_the_starting_sequence_number(local_catalog):
    """Rewritten files keep the sequence number of the data they replace.

    A row-level delete applies to data at or below its own sequence number, so
    advancing the rewritten files past it would stop it applying.
    """
    table = make_seeded_table(local_catalog, "default.t_seq", n_files=6)
    table.refresh()
    starting = table.current_snapshot().sequence_number

    Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL)

    table.refresh()
    assert table.current_snapshot().sequence_number > starting
    assert _added_entry_sequence_numbers(table) == {starting}


def test_the_starting_sequence_number_can_be_turned_off(local_catalog):
    """Turning it off leaves the commit assigning the new snapshot's number."""
    table = make_seeded_table(local_catalog, "default.t_seq_off", n_files=6)
    table.refresh()
    starting = table.current_snapshot().sequence_number

    Table.from_iceberg(table).rewrite_data_files(
        "binpack", options={**_REWRITE_ALL, "use-starting-sequence-number": False}
    )

    table.refresh()
    committed = table.current_snapshot().sequence_number
    assert committed > starting
    assert _added_entry_sequence_numbers(table) == {committed}


def _make_two_spec_table(catalog, name: str) -> tuple[Any, int]:
    """Seed a table under one partition spec, then evolve it to another."""
    import pyarrow as pa

    table = catalog.create_table(
        identifier=name,
        schema=pa.schema([("id", pa.int64()), ("g", pa.int64())]),
        properties={"format-version": "2"},
    )
    with table.update_spec() as update:
        update.add_field("g", IdentityTransform(), "g_id")
    table.refresh()
    seeded_spec_id = table.spec().spec_id
    for i in range(4):
        table.append(
            pa.table(
                {
                    "id": pa.array(range(i * 10, i * 10 + 10), type=pa.int64()),
                    "g": pa.array([i % 2] * 10, type=pa.int64()),
                }
            )
        )
    with table.update_spec() as update:
        update.add_field("id", IdentityTransform(), "id_id")
    table.refresh()
    return table, seeded_spec_id


def test_a_rewrite_can_target_a_partition_spec_that_is_no_longer_current(local_catalog):
    """The option chooses the output partitioning, and the commit has to honour it."""
    table, older_spec_id = _make_two_spec_table(local_catalog, "default.t_spec")
    assert older_spec_id != table.spec().spec_id
    before = sorted(table.scan().to_arrow().column("id").to_pylist())

    result = Table.from_iceberg(table).rewrite_data_files(
        "binpack", options={**_REWRITE_ALL, "output-spec-id": older_spec_id}
    )

    table.refresh()
    assert result.added_files >= 1
    assert sorted(table.scan().to_arrow().column("id").to_pylist()) == before
    assert _spec_ids_of_added_manifests(table) == {older_spec_id}


def test_a_rewrite_defaults_to_the_current_partition_spec(local_catalog):
    """Without the option the output is written under the table's current spec."""
    table, older_spec_id = _make_two_spec_table(local_catalog, "default.t_spec_default")
    current = table.spec().spec_id

    Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL)

    table.refresh()
    assert current != older_spec_id
    assert _spec_ids_of_added_manifests(table) == {current}
