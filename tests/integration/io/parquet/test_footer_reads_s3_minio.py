"""A read fetches each file's footer once, and a process keeps the footers of unchanged files.

A read takes the footer schema inference already read, and the size the listing
or the table's manifest already gave, so it asks the store for neither again.
Footers of a table's files, which are never rewritten in place, are kept between
reads under the IO configuration that fetched them. Whatever is reused, the rows
are the file's own.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyarrow import fs as pafs
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.table import Table

import daft
from daft import col
from tests.integration.io.conftest import minio_create_bucket
from tests.integration.io.counting_proxy import Request, counting_proxy, io_config

ROWS = 10_000


@dataclass
class Store:
    """The store, reached directly and through a proxy noting each request."""

    direct: daft.io.IOConfig
    proxied: daft.io.IOConfig
    port: int
    noted: list[Request]
    bucket: str

    def footers(self) -> int:
        return sum(request.parquet_footer for request in self.noted)

    def sizes(self) -> int:
        return sum(request.parquet_size for request in self.noted)


@pytest.fixture
def store(minio_io_config: daft.io.IOConfig) -> Iterator[Store]:
    with counting_proxy() as (port, noted), minio_create_bucket(minio_io_config) as (_, bucket):
        yield Store(minio_io_config, io_config(port), port, noted, bucket)


def _rows(first: int) -> pa.Table:
    return pa.table({"k": pa.array(range(first, first + ROWS), pa.int64())})


def _write_files(store: Store, name: str, files: int) -> str:
    path = f"s3://{store.bucket}/{name}"
    for number in range(files):
        daft.from_arrow(_rows(number * ROWS)).write_parquet(path, write_mode="append", io_config=store.direct)
    return path


def _read(store: Store, path: str, filtered: bool) -> list[int]:
    frame = daft.read_parquet(path, io_config=store.proxied)
    if filtered:
        frame = frame.where(col("k") % 7 == 0)
    return sorted(frame.to_pydict()["k"])


def _expected(files: int, filtered: bool) -> list[int]:
    return [k for k in range(files * ROWS) if not filtered or k % 7 == 0]


@pytest.mark.integration()
@pytest.mark.parametrize("filtered", [False, True], ids=["plain", "filtered"])
@pytest.mark.parametrize("files", [1, 4], ids=["one file", "four files"])
def test_a_read_fetches_each_files_footer_once_and_no_size(store: Store, files: int, filtered: bool) -> None:
    path = _write_files(store, "glob", files)

    answer = _read(store, path, filtered)

    assert answer == _expected(files, filtered)
    assert store.footers() == files
    assert store.sizes() == 0


@pytest.mark.integration()
def test_a_file_rewritten_after_the_frame_is_built_is_read_afresh(store: Store) -> None:
    path = f"s3://{store.bucket}/rewritten/part.parquet"
    pq.write_table(_rows(0), path.removeprefix("s3://"), filesystem=_filesystem(store))
    frame = daft.read_parquet(path, io_config=store.proxied)

    rewritten = pa.table({"k": pa.array(range(-5, 0), pa.int64())})
    pq.write_table(rewritten, path.removeprefix("s3://"), filesystem=_filesystem(store))

    assert frame.to_pydict()["k"] == list(range(-5, 0))


def _filesystem(store: Store) -> pafs.S3FileSystem:
    return pafs.S3FileSystem(
        access_key="minioadmin",
        secret_key="minioadmin",
        endpoint_override=store.direct.s3.endpoint_url,
        scheme="http",
    )


@pytest.fixture
def table(store: Store, tmp_path: Path) -> Table:
    catalog = SqlCatalog(
        "footers",
        uri=f"sqlite:///{tmp_path}/catalog.db",
        warehouse=f"s3://{store.bucket}/warehouse",
        **{
            "s3.endpoint": store.direct.s3.endpoint_url,
            "s3.access-key-id": "minioadmin",
            "s3.secret-access-key": "minioadmin",
            "s3.region": "us-east-1",
        },
    )
    catalog.create_namespace("ns")
    table = catalog.create_table("ns.rows", schema=pa.schema([("k", pa.int64())]))
    for number in range(4):
        daft.from_arrow(_rows(number * ROWS)).write_iceberg(table, mode="append", io_config=store.direct)
    return catalog.load_table("ns.rows")


def _read_table(table: Table, io_config: daft.io.IOConfig, filtered: bool) -> list[int]:
    frame = daft.read_iceberg(table, io_config=io_config)
    if filtered:
        frame = frame.where(col("k") % 7 == 0)
    return sorted(frame.to_pydict()["k"])


@pytest.mark.integration()
@pytest.mark.parametrize("filtered", [False, True], ids=["plain", "filtered"])
def test_a_tables_unchanged_files_are_read_without_their_footers_again(
    store: Store, table: Table, filtered: bool
) -> None:
    first = _read_table(table, store.proxied, filtered)
    first_footers = store.footers()
    store.noted.clear()

    again = _read_table(table, store.proxied, filtered)

    assert first == again == _expected(4, filtered)
    assert first_footers == 4
    assert store.footers() == 0
    assert store.sizes() == 0


@pytest.mark.integration()
def test_another_io_configuration_fetches_the_footers_for_itself(store: Store, table: Table) -> None:
    _read_table(table, store.proxied, filtered=False)
    store.noted.clear()
    answer = _read_table(table, io_config(store.port, num_tries=7), filtered=False)

    assert answer == _expected(4, filtered=False)
    assert store.footers() == 4


@pytest.mark.integration()
def test_with_no_cache_every_read_fetches_its_footers(store: Store, table: Table) -> None:
    with daft.execution_config_ctx(parquet_footer_cache_bytes=0):
        _read_table(table, store.proxied, filtered=False)
        store.noted.clear()
        answer = _read_table(table, store.proxied, filtered=False)

    assert answer == _expected(4, filtered=False)
    assert store.footers() == 4


@pytest.mark.integration()
def test_a_cached_page_index_is_unused_when_the_index_is_turned_off(store: Store, table: Table) -> None:
    _read_table(table, store.proxied, filtered=True)

    with daft.execution_config_ctx(parquet_page_index=False):
        answer = _read_table(table, store.proxied, filtered=True)

    assert answer == _expected(4, filtered=True)


@pytest.mark.integration()
def test_a_column_renamed_since_the_footer_was_kept_reads_under_its_new_name(store: Store, table: Table) -> None:
    _read_table(table, store.proxied, filtered=False)
    with table.update_schema() as update:
        update.rename_column("k", "key")
    renamed = table.refresh()

    answer = daft.read_iceberg(renamed, io_config=store.proxied).to_pydict()

    assert sorted(answer["key"]) == _expected(4, filtered=False)
