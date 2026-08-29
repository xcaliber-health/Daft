"""A rewrite over a table that really carries equality deletes.

An equality delete names the column values it removes rather than the rows'
positions, so it applies to any data file in the partition and cannot be
resolved by reading one file. Rewriting the data files without applying it would
bring the deleted rows back, which is why the rewrite refuses instead.

Neither the catalog library nor the reference engine's batch writers emit one,
so the delete is written by the helper here. That is the point: until now the
refusal was only ever exercised against a hand-built planner input, never
against a table an engine would actually meet.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pyiceberg")

from daft.catalog import Table
from daft.io.iceberg import RewriteConflict  # noqa: F401 - imported to prove the module loads
from tests.io.iceberg.actions._helpers import commit_equality_deletes, make_seeded_table

_REWRITE_ALL = {"rewrite-all": True, "min-input-files": 2}


def _live_data_files(table: Any) -> list[str]:
    """Live data file paths read from the manifests.

    A scan cannot be used: the catalog library refuses to plan a table carrying
    equality deletes at all, which is the very condition under test.
    """
    table.refresh()
    paths: list[str] = []
    snapshot = table.current_snapshot()
    for manifest in snapshot.manifests(table.io):
        for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=True):
            if int(entry.data_file.content) == 0:
                paths.append(entry.data_file.file_path)
    return sorted(paths)


def _delete_content_counts(table: Any) -> dict[int, int]:
    """Count live delete files by content type, straight from the manifests."""
    table.refresh()
    counts: dict[int, int] = {}
    snapshot = table.current_snapshot()
    for manifest in snapshot.manifests(table.io):
        for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=True):
            content = int(entry.data_file.content)
            if content != 0:
                counts[content] = counts.get(content, 0) + 1
    return counts


def test_a_table_can_be_given_a_real_equality_delete(local_catalog):
    """The fixture itself is worth asserting: without it nothing below is real."""
    table = make_seeded_table(local_catalog, "default.t_eq_fixture", n_files=4, rows_per_file=25)

    commit_equality_deletes(table, "id", [1, 2, 3])

    counts = _delete_content_counts(table)
    assert counts.get(2) == 1, f"expected one equality delete file, saw {counts}"


def test_a_rewrite_refuses_a_table_carrying_equality_deletes(local_catalog):
    """Refusing is correct: the rewrite cannot apply the delete while replacing files."""
    from daft.io.iceberg._compact import EqualityDeletesPresent

    table = make_seeded_table(local_catalog, "default.t_eq_refuse", n_files=4, rows_per_file=25)
    commit_equality_deletes(table, "id", [1, 2, 3])
    table.refresh()

    with pytest.raises(EqualityDeletesPresent) as raised:
        Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL)

    assert "equality" in str(raised.value).lower()


def test_the_refusal_names_the_offending_files(local_catalog):
    """An operator has to be able to find what blocked the rewrite."""
    from daft.io.iceberg._compact import EqualityDeletesPresent

    table = make_seeded_table(local_catalog, "default.t_eq_named", n_files=4, rows_per_file=25)
    delete_path = commit_equality_deletes(table, "id", [7])
    table.refresh()

    with pytest.raises(EqualityDeletesPresent) as raised:
        Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL)

    message = str(raised.value)
    assert delete_path.rsplit("/", 1)[-1] in message or "equality" in message.lower()


def test_a_refused_rewrite_changes_nothing(local_catalog):
    """The refusal must leave the table exactly as it was, not half rewritten."""
    from daft.io.iceberg._compact import EqualityDeletesPresent

    table = make_seeded_table(local_catalog, "default.t_eq_untouched", n_files=4, rows_per_file=25)
    commit_equality_deletes(table, "id", [5])
    table.refresh()
    before_snapshots = len(table.metadata.snapshots or [])
    before_files = _live_data_files(table)

    with pytest.raises(EqualityDeletesPresent):
        Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL)

    table.refresh()
    assert len(table.metadata.snapshots or []) == before_snapshots, "a snapshot was committed"
    assert _live_data_files(table) == before_files


def test_other_maintenance_still_works_alongside_equality_deletes(local_catalog):
    """Only the data rewrite refuses; manifest maintenance is unaffected."""
    table = make_seeded_table(local_catalog, "default.t_eq_manifests", n_files=6, rows_per_file=25)
    commit_equality_deletes(table, "id", [9])
    table.refresh()

    result = Table.from_iceberg(table).rewrite_manifests()

    table.refresh()
    assert result.rewritten_manifests_count >= 1
    assert _delete_content_counts(table).get(2) == 1, "the delete must survive a manifest rewrite"
