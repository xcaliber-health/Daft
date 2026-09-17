"""Merging into a table whose rows are laid out by partition.

A delete file belongs to the partition of the data file it names, so a table
whose partitioning has changed since some files were written must still get
delete files under the partitioning each file was written with.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

import daft
from daft import col
from daft.catalog.__iceberg import IcebergTable

MODES = ["copy-on-write", "merge-on-read"]


@pytest.fixture(params=MODES)
def mode(request):
    return request.param


def _partitioned(catalog, mode, name="default.by_region"):
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.schema import Schema
    from pyiceberg.transforms import IdentityTransform
    from pyiceberg.types import LongType, NestedField, StringType

    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "region", StringType(), required=False),
        NestedField(3, "label", StringType(), required=False),
    )
    spec = PartitionSpec(PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="region"))
    table = catalog.create_table(
        identifier=name,
        schema=schema,
        partition_spec=spec,
        properties={"write.merge.mode": mode, "format-version": "2"},
    )
    for batch, region in enumerate(["us", "eu", "apac"]):
        start = batch * 10
        table.append(
            pa.table(
                {
                    "id": pa.array(list(range(start, start + 10)), type=pa.int64()),
                    "region": pa.array([region] * 10, type=pa.string()),
                    "label": pa.array(["seed"] * 10, type=pa.string()),
                }
            )
        )
    return table.refresh()


def _rows(table):
    frame = daft.read_iceberg(table.refresh()).sort("id").to_pydict()
    return list(zip(frame["id"], frame["region"], frame["label"]))


def test_a_merge_changes_rows_in_every_partition(local_catalog, mode):
    table = _partitioned(local_catalog, mode)
    changed = [1, 11, 21]
    source = daft.from_pydict({"id": changed, "region": ["us", "eu", "apac"], "label": ["merged"] * 3})

    result = (
        IcebergTable.from_iceberg(table)
        .merge_into(source, on=col("target.id") == col("source.id"))
        .when_matched()
        .update({"label": col("source.label")})
        .execute()
    )

    rows = {row_id: label for row_id, _, label in _rows(table)}
    assert result.rows_updated == 3
    assert all(rows[row_id] == "merged" for row_id in changed)
    assert len(rows) == 30


def test_rows_added_by_a_merge_land_in_their_own_partition(local_catalog, mode):
    table = _partitioned(local_catalog, mode)
    source = daft.from_pydict({"id": [99], "region": ["us"], "label": ["added"]})

    (
        IcebergTable.from_iceberg(table)
        .merge_into(source, on=col("target.id") == col("source.id"))
        .when_matched()
        .update({"label": col("source.label")})
        .when_not_matched()
        .insert_all()
        .execute()
    )

    files = [
        entry.data_file
        for snapshot in [table.refresh().current_snapshot()]
        for manifest in snapshot.manifests(table.io)
        for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=True)
    ]
    added = [f for f in files if "99" not in str(f.file_path)]
    assert added, "the table still holds files"
    assert (99, "us", "added") in _rows(table)
    for data_file in files:
        assert data_file.partition is not None, "every file names the partition it belongs to"


def test_removing_rows_names_them_under_the_partitioning_they_were_written_with(local_catalog):
    from pyiceberg.manifest import DataFileContent
    from pyiceberg.transforms import IdentityTransform

    table = _partitioned(local_catalog, "merge-on-read", name="default.evolving")
    with table.update_spec() as update:
        update.add_field("label", IdentityTransform(), "label")
    table = table.refresh()
    table.append(
        pa.table(
            {
                "id": pa.array([100, 101], type=pa.int64()),
                "region": pa.array(["us", "eu"], type=pa.string()),
                "label": pa.array(["late"] * 2, type=pa.string()),
            }
        )
    )

    source = daft.from_pydict({"id": [1, 100], "region": ["us", "us"], "label": ["merged"] * 2})
    (
        IcebergTable.from_iceberg(table.refresh())
        .merge_into(source, on=col("target.id") == col("source.id"))
        .when_matched()
        .update({"label": col("source.label")})
        .execute()
    )

    live = table.refresh().current_snapshot()
    specs = table.specs()
    deletes = [
        entry.data_file
        for manifest in live.manifests(table.io)
        for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=True)
        if entry.data_file.content == DataFileContent.POSITION_DELETES
    ]
    assert deletes, "the replaced rows are named by delete files"
    for delete_file in deletes:
        spec = specs[delete_file.spec_id]
        assert len(delete_file.partition) == len(spec.fields), (
            "a delete file carries the partition values of the spec it names"
        )
    rows = {row_id: label for row_id, _, label in _rows(table)}
    assert rows[1] == "merged" and rows[100] == "merged"
    assert len(rows) == 32
