"""A lakehouse read plans in the round trips its metadata needs.

Planning reads each manifest list and manifest in one request, without asking
its size first, and reads a snapshot's manifest list once however many checks
consult it. A table never rewrites those files in place, so a later read of
the same snapshot, under the same IO configuration, fetches none of them again.
The files planned are the table's own.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pytest
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.table import Table

import daft
from daft import col
from tests.integration.io.conftest import minio_create_bucket
from tests.integration.io.counting_proxy import Request, counting_proxy, io_config

FILES = 4
ROWS = 1_000


@dataclass
class Store:
    """The store, reached directly and through a proxy noting each request."""

    direct: daft.io.IOConfig
    port: int
    noted: list[Request]
    bucket: str

    def count(self, kind: str) -> int:
        return sum(getattr(request, kind) for request in self.noted)


@pytest.fixture
def store(minio_io_config: daft.io.IOConfig) -> Iterator[Store]:
    with counting_proxy() as (port, noted), minio_create_bucket(minio_io_config) as (_, bucket):
        yield Store(minio_io_config, port, noted, bucket)


def _catalog(store: Store, tmp_path: Path, endpoint: str) -> SqlCatalog:
    """The table's catalog, whose own file IO reaches the store at ``endpoint``."""
    return SqlCatalog(
        "planning",
        uri=f"sqlite:///{tmp_path}/catalog.db",
        warehouse=f"s3://{store.bucket}/warehouse",
        **{
            "s3.endpoint": endpoint,
            "s3.access-key-id": "minioadmin",
            "s3.secret-access-key": "minioadmin",
            "s3.region": "us-east-1",
        },
    )


def _append(table: Table, store: Store, first: int) -> None:
    rows = pa.table({"k": pa.array(range(first, first + ROWS), pa.int64())})
    daft.from_arrow(rows).write_iceberg(table, mode="append", io_config=store.direct)


def _proxied(store: Store, tmp_path: Path) -> Table:
    """The table as loaded through a catalog whose own file IO reaches the store through
    the proxy, so every planning request is noted, whichever file IO makes it.
    """
    return _catalog(store, tmp_path, f"http://127.0.0.1:{store.port}").load_table("ns.rows")


@pytest.fixture
def table(store: Store, tmp_path: Path) -> Table:
    writer = _catalog(store, tmp_path, store.direct.s3.endpoint_url)
    writer.create_namespace("ns")
    table = writer.create_table("ns.rows", schema=pa.schema([("k", pa.int64())]))
    for number in range(FILES):
        _append(table, store, number * ROWS)
    return _proxied(store, tmp_path)


def _read(table: Table, config: daft.io.IOConfig) -> list[int]:
    return sorted(daft.read_iceberg(table, io_config=config).where(col("k") < 5).to_pydict()["k"])


@pytest.mark.integration()
def test_a_read_fetches_its_manifest_list_once_and_asks_no_metadata_size(store: Store, table: Table) -> None:
    answer = _read(table, io_config(store.port))

    assert answer == [0, 1, 2, 3, 4]
    assert store.count("manifest_list") == 1
    assert store.count("manifest") == FILES
    assert store.count("metadata_size") == 0


@pytest.mark.integration()
def test_a_later_read_of_the_same_snapshot_fetches_no_metadata(store: Store, table: Table) -> None:
    _read(table, io_config(store.port))
    store.noted.clear()

    answer = _read(table, io_config(store.port))

    assert answer == [0, 1, 2, 3, 4]
    assert store.count("manifest_list") == 0
    assert store.count("manifest") == 0


@pytest.mark.integration()
def test_another_io_configuration_fetches_the_metadata_for_itself(store: Store, table: Table) -> None:
    _read(table, io_config(store.port))
    store.noted.clear()

    answer = _read(table, io_config(store.port, num_tries=7))

    assert answer == [0, 1, 2, 3, 4]
    assert store.count("manifest_list") == 1
    assert store.count("manifest") == FILES


@pytest.mark.integration()
def test_with_no_cache_each_read_fetches_its_metadata_without_asking_sizes(store: Store, table: Table) -> None:
    with daft.execution_config_ctx(iceberg_metadata_cache_bytes=0):
        _read(table, io_config(store.port))
        store.noted.clear()
        answer = _read(table, io_config(store.port))

    assert answer == [0, 1, 2, 3, 4]
    assert store.count("manifest") == FILES
    assert store.count("metadata_size") == 0


@pytest.mark.integration()
def test_a_new_snapshot_is_planned_with_its_new_files(store: Store, table: Table, tmp_path: Path) -> None:
    _read(table, io_config(store.port))
    _append(_catalog(store, tmp_path, store.direct.s3.endpoint_url).load_table("ns.rows"), store, -ROWS)
    appended = _proxied(store, tmp_path)

    answer = daft.read_iceberg(appended, io_config=io_config(store.port)).count_rows()

    assert answer == (FILES + 1) * ROWS
