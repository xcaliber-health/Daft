"""The manifests a rewrite commit leaves behind.

A manifest declares whether it lists data files or delete files, and readers
trust the declaration, so a rewrite that removes some entries of a delete
manifest must rewrite it as a delete manifest. The commit also merges small
manifests when the table's commit properties ask for it, as any merging
commit does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

from daft.catalog import Table
from tests.io.iceberg.actions._helpers import (
    _commit_files,
    _write_equality_delete_file,
    commit_equality_upsert,
    make_seeded_table,
)

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog
    from pyiceberg.table import Table as PyIcebergTable

_REWRITE_ALL_RECLAIMING = {"rewrite-all": True, "min-input-files": 2, "remove-dangling-deletes": True}


def _manifests(table: PyIcebergTable) -> list[tuple[int, list[int]]]:
    """Return ``(manifest content, [entry content ...])`` for every live manifest."""
    table.refresh()
    out = []
    for manifest in table.current_snapshot().manifests(table.io):
        entries = manifest.fetch_manifest_entry(table.io, discard_deleted=True)
        out.append((int(manifest.content), [int(entry.data_file.content) for entry in entries]))
    return out


def _regional_table(catalog: Catalog, name: str) -> PyIcebergTable:
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.schema import Schema
    from pyiceberg.transforms import IdentityTransform
    from pyiceberg.types import LongType, NestedField, StringType

    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "region", StringType(), required=False),
    )
    spec = PartitionSpec(PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="region"))
    table = catalog.create_table(identifier=name, schema=schema, partition_spec=spec)
    for start in (0, 100):
        table.append(
            pa.table(
                {
                    "id": pa.array(list(range(start, start + 20)), type=pa.int64()),
                    "region": pa.array(["a" if i % 2 == 0 else "b" for i in range(start, start + 20)]),
                }
            )
        )
    return table


def test_a_partially_reclaimed_delete_manifest_stays_a_delete_manifest(local_catalog):
    # Arrange: one delete manifest holding a delete for each partition.
    from pyiceberg.typedef import Record

    table = _regional_table(local_catalog, "default.t_manifest_content")
    deletes = [
        _write_equality_delete_file(
            table, pa.table({"id": pa.array([0, 2], type=pa.int64())}), ["id"], partition=Record("a")
        ),
        _write_equality_delete_file(
            table, pa.table({"id": pa.array([1, 3], type=pa.int64())}), ["id"], partition=Record("b")
        ),
    ]
    _commit_files(table, [], deletes)
    assert [content for content, _ in _manifests(table) if content == 1] == [1]

    # Act: rewrite one partition, whose delete then covers nothing and is reclaimed.
    Table.from_iceberg(table).rewrite_data_files("binpack", where="region = 'a'", options=_REWRITE_ALL_RECLAIMING)

    # Assert: the other partition's delete survives, in a manifest still declared as deletes.
    manifests = _manifests(table)
    delete_manifests = [entries for content, entries in manifests if content == 1]
    assert delete_manifests == [[2]], f"live manifests: {manifests}"
    assert all(all(entry == 0 for entry in entries) for content, entries in manifests if content == 0)


def test_removed_delete_files_are_recorded_in_a_delete_manifest(local_catalog):
    table = make_seeded_table(local_catalog, "default.t_removed_delete_manifest", n_files=2, rows_per_file=10)
    commit_equality_upsert(table, pa.table({"id": pa.array([1], type=pa.int64()), "label": pa.array(["new"])}), ["id"])

    Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL_RECLAIMING)

    table.refresh()
    for manifest in table.current_snapshot().manifests(table.io):
        for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=False):
            is_delete = int(entry.data_file.content) != 0
            assert (int(manifest.content) == 1) == is_delete, f"{manifest.manifest_path} mixes content kinds"


@pytest.mark.parametrize(
    ("properties", "expected_data_manifests"),
    [
        pytest.param({}, 2, id="default-min-count-keeps-them-apart"),
        pytest.param({"commit.manifest.min-count-to-merge": "2"}, 1, id="min-count-two-merges-them"),
        pytest.param(
            {"commit.manifest.min-count-to-merge": "2", "commit.manifest-merge.enabled": "false"},
            2,
            id="merge-disabled",
        ),
    ],
)
def test_data_manifests_merge_as_the_commit_properties_ask(local_catalog, properties, expected_data_manifests):
    # Arrange: a rewrite over every file leaves one manifest of added files and
    # one of removed entries, a pair the merge rule can act on.
    table = make_seeded_table(local_catalog, f"default.t_merge_{len(properties)}", n_files=4, rows_per_file=10)
    if properties:
        with table.transaction() as transaction:
            transaction.set_properties(**properties)

    # Act
    Table.from_iceberg(table).rewrite_data_files("binpack", options={"rewrite-all": True, "min-input-files": 2})

    # Assert
    data_manifests = [entries for content, entries in _manifests(table) if content == 0]
    assert len(data_manifests) == expected_data_manifests


def test_delete_manifests_merge_into_a_delete_manifest(local_catalog):
    # Arrange: three upserts leave three delete manifests; the two oldest data
    # files are left alone, so every delete still applies and survives.
    table = make_seeded_table(local_catalog, "default.t_merge_deletes", n_files=4, rows_per_file=10)
    for key in (1, 2, 3):
        commit_equality_upsert(
            table, pa.table({"id": pa.array([key], type=pa.int64()), "label": pa.array([f"new-{key}"])}), ["id"]
        )
    with table.transaction() as transaction:
        transaction.set_properties(**{"commit.manifest.min-count-to-merge": "2"})

    # Act
    Table.from_iceberg(table).rewrite_data_files(
        "binpack", where="id >= 20", options={"rewrite-all": True, "min-input-files": 2}
    )

    # Assert
    manifests = _manifests(table)
    assert [entries for content, entries in manifests if content == 1] == [[2, 2, 2]], f"live manifests: {manifests}"
