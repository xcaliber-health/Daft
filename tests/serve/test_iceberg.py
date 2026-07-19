"""Table-format reads execute server-side.

Scan planning, pruning, and data access all happen next to the data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

import daft

from .conftest import assert_remote_matches_native

if TYPE_CHECKING:
    import pathlib

    from pyiceberg.table import Table

    from daft.runners.native_runner import NativeRunner
    from daft.runners.remote_runner import RemoteRunner

pytest.importorskip("pyiceberg")


@pytest.fixture(scope="module")
def iceberg_table(tmp_path_factory: pytest.TempPathFactory) -> Table:
    from pyiceberg.catalog.sql import SqlCatalog

    warehouse: pathlib.Path = tmp_path_factory.mktemp("iceberg-warehouse")
    catalog = SqlCatalog(
        "test",
        uri=f"sqlite:///{warehouse}/catalog.db",
        warehouse=f"file://{warehouse}",
    )
    catalog.create_namespace("db")
    data = pa.table(
        {
            "id": pa.array(list(range(100)), type=pa.int64()),
            "bucket": pa.array([i % 5 for i in range(100)], type=pa.int64()),
            "label": pa.array([f"item-{i}" for i in range(100)], type=pa.large_string()),
        }
    )
    table = catalog.create_table("db.items", schema=data.schema)
    table.append(data)
    return table


def test_full_scan_matches_native(
    iceberg_table: Table, remote_runner: RemoteRunner, native_runner: NativeRunner
) -> None:
    df = daft.read_iceberg(iceberg_table).sort("id")
    assert_remote_matches_native(remote_runner, native_runner, df)


def test_filtered_scan_matches_native(
    iceberg_table: Table, remote_runner: RemoteRunner, native_runner: NativeRunner
) -> None:
    df = daft.read_iceberg(iceberg_table).where(daft.col("bucket") == 3).select("id", "label")
    assert_remote_matches_native(remote_runner, native_runner, df, sort_key="id")


def test_aggregation_over_table_matches_native(
    iceberg_table: Table, remote_runner: RemoteRunner, native_runner: NativeRunner
) -> None:
    df = daft.read_iceberg(iceberg_table).groupby("bucket").agg(daft.col("id").count().alias("n"))
    assert_remote_matches_native(remote_runner, native_runner, df, sort_key="bucket")


def test_snapshot_read_matches_native(
    iceberg_table: Table, remote_runner: RemoteRunner, native_runner: NativeRunner
) -> None:
    snapshot = iceberg_table.current_snapshot()
    assert snapshot is not None
    df = daft.read_iceberg(iceberg_table, snapshot_id=snapshot.snapshot_id).sort("id").limit(5)
    assert_remote_matches_native(remote_runner, native_runner, df)
