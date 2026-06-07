"""Compaction must preserve bloom-filtered query correctness.

``rewrite_data_files`` reads through the same scan and writes through the same
writer as normal reads/writes, so compacted files inherit the table's bloom
properties and remain correctly queryable. These tests assert that equality and
membership queries return identical results before and after compaction.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC
from pyiceberg.schema import Schema
from pyiceberg.types import LongType, NestedField, StringType

import daft
from daft import col
from daft.catalog import Table
from tests.io.iceberg.actions._helpers import read_ids as _read_ids
from tests.io.iceberg.actions._helpers import scan_file_count as _scan_file_count

_BLOOM_PROPERTIES = {
    "write.parquet.bloom-filter-enabled.column.id": "true",
    "write.parquet.bloom-filter-enabled.column.label": "true",
    "write.parquet.bloom-filter-fpp.column.id": "0.01",
}


def _bloom_table(local_catalog, name: str, n_files: int, rows_per_file: int):
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "label", StringType(), required=False),
    )
    table = local_catalog.create_table(
        identifier=name,
        schema=schema,
        partition_spec=UNPARTITIONED_PARTITION_SPEC,
        properties=_BLOOM_PROPERTIES,
    )
    for f in range(n_files):
        start = f * rows_per_file
        table.append(
            pa.table(
                {
                    "id": pa.array(range(start, start + rows_per_file), type=pa.int64()),
                    "label": pa.array([f"l{i}" for i in range(start, start + rows_per_file)], type=pa.string()),
                }
            )
        )
    table.refresh()
    return table


def test_compaction_preserves_equality_results(local_catalog) -> None:
    table = _bloom_table(local_catalog, "default.bloom_eq", n_files=6, rows_per_file=100)
    total = 600

    pre = daft.read_iceberg(table)
    assert pre.where(col("id") == 250).to_pydict()["id"] == [250]

    dt = Table.from_iceberg(table)
    result = dt.rewrite_data_files(
        "binpack",
        options={"min-input-files": 2, "rewrite-all": True, "target-file-size-bytes": 64 * 1024 * 1024},
    )
    table.refresh()
    assert result.added_files == _scan_file_count(table)

    read = daft.read_iceberg(table)
    # Present value survives compaction (no wrong drop); absent value pruned.
    assert read.where(col("id") == 250).to_pydict()["id"] == [250]
    assert read.where(col("id") == 9_999_999).to_pydict()["id"] == []
    assert read.where(col("label") == "l42").to_pydict()["label"] == ["l42"]
    assert read.where(col("label") == "absent").to_pydict()["label"] == []
    # All data preserved.
    assert _read_ids(table) == list(range(total))


def test_compaction_preserves_membership_results(local_catalog) -> None:
    table = _bloom_table(local_catalog, "default.bloom_in", n_files=4, rows_per_file=50)

    dt = Table.from_iceberg(table)
    dt.rewrite_data_files(
        "binpack",
        options={"min-input-files": 2, "rewrite-all": True, "target-file-size-bytes": 64 * 1024 * 1024},
    )
    table.refresh()

    read = daft.read_iceberg(table)
    got = sorted(read.where(col("id").is_in([1, 99, 9_999_999])).to_pydict()["id"])
    assert got == [1, 99]
