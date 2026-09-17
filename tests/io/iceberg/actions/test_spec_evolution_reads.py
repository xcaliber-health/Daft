"""Reading a table whose partitioning changed after some files were written.

A file carries the partition values of the layout it was written under. A
condition on a field added later says nothing about such a file, so the file
has to be read rather than refused.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

import daft
from daft import col
from daft.catalog.__iceberg import IcebergTable


def _evolving_table(catalog, name="default.evolving_reads"):
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.schema import Schema
    from pyiceberg.transforms import IdentityTransform
    from pyiceberg.types import LongType, NestedField, StringType

    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "region", StringType(), required=False),
        NestedField(3, "tier", StringType(), required=False),
    )
    spec = PartitionSpec(PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="region"))
    table = catalog.create_table(
        identifier=name, schema=schema, partition_spec=spec, properties={"format-version": "2"}
    )
    table.append(
        pa.table(
            {
                "id": pa.array([1, 2, 3], type=pa.int64()),
                "region": pa.array(["us", "us", "eu"], type=pa.string()),
                "tier": pa.array(["gold", "silver", "gold"], type=pa.string()),
            }
        )
    )
    with table.update_spec() as update:
        update.add_field("tier", IdentityTransform(), "tier")
    table = table.refresh()
    table.append(
        pa.table(
            {
                "id": pa.array([4], type=pa.int64()),
                "region": pa.array(["us"], type=pa.string()),
                "tier": pa.array(["gold"], type=pa.string()),
            }
        )
    )
    return table.refresh()


def test_a_condition_on_a_later_field_still_reads_older_files(local_catalog) -> None:
    """Files written before the field existed are read, not skipped."""
    table = _evolving_table(local_catalog)

    rows = daft.read_iceberg(table).where(col("tier") == "gold").sort("id").to_pydict()

    assert rows["id"] == [1, 3, 4]


def test_a_condition_on_an_original_field_still_prunes(local_catalog) -> None:
    """The layout every file shares is still used to skip files."""
    table = _evolving_table(local_catalog, name="default.evolving_pruned")

    rows = daft.read_iceberg(table).where(col("region") == "eu").sort("id").to_pydict()

    assert rows["id"] == [3]


@pytest.mark.parametrize("mode", ["copy-on-write", "merge-on-read"])
def test_a_merge_reaches_rows_in_files_of_either_layout(local_catalog, mode: str) -> None:
    """A change lands on rows whichever layout their file was written under."""
    table = _evolving_table(local_catalog, name=f"default.evolving_merge_{mode.replace('-', '_')}")
    with table.transaction() as transaction:
        transaction.set_properties(**{"write.merge.mode": mode})
    source = daft.from_pydict({"id": [1, 4], "tier": ["platinum", "platinum"]})

    result = (
        IcebergTable.from_iceberg(table.refresh())
        .merge_into(source, on=col("target.id") == col("source.id"))
        .when_matched()
        .update({"tier": col("source.tier")})
        .execute()
    )

    rows = daft.read_iceberg(table.refresh()).sort("id").to_pydict()
    assert result.rows_updated == 2
    assert dict(zip(rows["id"], rows["tier"])) == {1: "platinum", 2: "silver", 3: "gold", 4: "platinum"}
