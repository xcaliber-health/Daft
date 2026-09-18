"""Writes that change rows in place, against a catalog served over the network.

The unit coverage runs against a catalog in a local file, where committing is a
transaction in a database this process owns. Here the catalog is a service and
the files live in an object store, which is where a commit can be refused for
reasons a local catalog never produces.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

import pytest

pyiceberg = pytest.importorskip("pyiceberg")

import pyarrow as pa

import daft
from daft import col, lit
from daft.catalog import Table as DaftTable
from daft.io.iceberg import MergeCardinalityError

MODES = ["copy-on-write", "merge-on-read"]

#: Where the catalog and store are, overridable so the same tests can run
#: against whichever pair of services is up.
_REST_URI = os.environ.get("DAFT_TEST_ICEBERG_REST_URI", "http://localhost:8181")
_S3_ENDPOINT = os.environ.get("DAFT_TEST_ICEBERG_S3_ENDPOINT", "http://localhost:9000")
_ACCESS_KEY = os.environ.get("DAFT_TEST_ICEBERG_ACCESS_KEY", "admin")
_SECRET_KEY = os.environ.get("DAFT_TEST_ICEBERG_SECRET_KEY", "password")
_NAMESPACE = os.environ.get("DAFT_TEST_ICEBERG_NAMESPACE", "default")


@pytest.fixture(scope="module")
def rest_catalog() -> Any:
    """Return the remote catalog these tests write through, or skip without one."""
    from pyiceberg.catalog import load_catalog

    try:
        catalog = load_catalog(
            "integration",
            **{
                "type": "rest",
                "uri": _REST_URI,
                "s3.endpoint": _S3_ENDPOINT,
                "s3.access-key-id": _ACCESS_KEY,
                "s3.secret-access-key": _SECRET_KEY,
            },
        )
        catalog.list_namespaces()
    except Exception as unreachable:  # noqa: BLE001 - any failure here means no service
        pytest.skip(f"no catalog at {_REST_URI}: {unreachable}")
    return catalog


@contextlib.contextmanager
def _table(catalog: Any, name: str, mode: str):
    """Create a seeded table that records changes the way ``mode`` says."""
    schema = pa.schema([("id", pa.int64()), ("label", pa.large_string())])
    identifier = f"{_NAMESPACE}.{name}"
    with contextlib.suppress(Exception):
        catalog.drop_table(identifier)
    table = catalog.create_table(
        identifier,
        schema=schema,
        properties={
            "format-version": "2",
            "write.merge.mode": mode,
            "write.update.mode": mode,
            "write.delete.mode": mode,
        },
    )
    for batch in range(3):
        start = batch * 100
        table.append(
            pa.table(
                {
                    "id": pa.array(range(start, start + 100), type=pa.int64()),
                    "label": pa.array(["seed"] * 100, type=pa.large_string()),
                }
            )
        )
    try:
        yield table.refresh()
    finally:
        with contextlib.suppress(Exception):
            catalog.drop_table(identifier)


def _rows(table: Any) -> dict[int, str]:
    data = table.refresh().scan().to_arrow().to_pydict()
    return dict(zip(data["id"], data["label"]))


@pytest.mark.integration()
@pytest.mark.parametrize("mode", MODES)
def test_a_merge_changes_and_adds_rows(rest_catalog: Any, mode: str) -> None:
    """Rows the change set matches are replaced and the rest are added."""
    catalog = rest_catalog
    with _table(catalog, f"merge_{mode.replace('-', '_')}", mode) as table:
        source = daft.from_pydict({"id": [5, 150, 999], "label": ["merged", "merged", "added"]})

        result = (
            DaftTable.from_iceberg(table)
            .merge_into(source, on=col("target.id") == col("source.id"))
            .when_matched()
            .update({"label": col("source.label")})
            .when_not_matched()
            .insert_all()
            .execute()
        )

        rows = _rows(table)
        assert (result.rows_updated, result.rows_inserted) == (2, 1)
        assert rows[5] == "merged" and rows[150] == "merged" and rows[999] == "added"
        assert len(rows) == 301


@pytest.mark.integration()
@pytest.mark.parametrize("mode", MODES)
def test_changing_and_removing_rows_by_condition(rest_catalog: Any, mode: str) -> None:
    """A condition decides which rows change and which leave."""
    catalog = rest_catalog
    with _table(catalog, f"update_{mode.replace('-', '_')}", mode) as table:
        changed = DaftTable.from_iceberg(table).update_where(col("id") < 10, {"label": lit("changed")})
        removed = DaftTable.from_iceberg(table.refresh()).delete_where(col("id") >= 250)

        rows = _rows(table)
        assert changed.rows_updated == 10
        assert all(rows[row_id] == "changed" for row_id in range(10))
        assert max(rows) < 250
        assert len(rows) == 250
        assert removed.rows_deleted + removed.files_dropped > 0


@pytest.mark.integration()
def test_a_row_matched_twice_is_refused(rest_catalog: Any) -> None:
    """Nothing is committed when the change set matches one row twice."""
    catalog = rest_catalog
    with _table(catalog, "cardinality", "merge-on-read") as table:
        before = table.current_snapshot().snapshot_id
        source = daft.from_pydict({"id": [7, 7], "label": ["a", "b"]})

        with pytest.raises(MergeCardinalityError):
            (
                DaftTable.from_iceberg(table)
                .merge_into(source, on=col("target.id") == col("source.id"))
                .when_matched()
                .update({"label": col("source.label")})
                .execute()
            )

        assert table.refresh().current_snapshot().snapshot_id == before


@pytest.mark.integration()
def test_a_repeated_merge_under_one_name_applies_once(rest_catalog: Any) -> None:
    """Running the same named merge twice leaves one snapshot and one change."""
    catalog = rest_catalog
    with _table(catalog, "replayed", "merge-on-read") as table:
        source = daft.from_pydict({"id": [1], "label": ["once"]})

        def run() -> Any:
            return (
                DaftTable.from_iceberg(table.refresh())
                .merge_into(source, on=col("target.id") == col("source.id"), options={"merge-id": "fixed"})
                .when_matched()
                .update({"label": col("source.label")})
                .execute()
            )

        first = run()
        second = run()

        assert first.snapshot_id == second.snapshot_id
        assert _rows(table)[1] == "once"
