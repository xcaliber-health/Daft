"""Per-file metadata a rewrite records about the files it writes.

Two things a reader relies on and a Parquet footer cannot supply: which order
the rows are in, and how many NaN values a floating-point column holds. Bounds
deliberately exclude NaN, so without the count a file can be pruned for a
predicate a NaN row would have matched.
"""

from __future__ import annotations

import math
from typing import Any

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

from daft.catalog import Table
from tests.io.iceberg.actions._helpers import make_seeded_table

_REWRITE_ALL = {"rewrite-all": True, "min-input-files": 2}
_SORT_BY_ID = [("id", "asc", "nulls-last")]


def _sort_order_ids(table: Any) -> set[int | None]:
    return {task.file.sort_order_id for task in table.scan().plan_files()}


def _nan_counts(table: Any) -> list[dict[int, int]]:
    return [dict(task.file.nan_value_counts or {}) for task in table.scan().plan_files()]


def _seed_with_nans(catalog, name: str, *, nan_every: int = 5, properties: dict[str, str] | None = None) -> Any:
    table = catalog.create_table(
        identifier=name,
        schema=pa.schema([("id", pa.int64()), ("v", pa.float64())]),
        properties={"format-version": "2", **(properties or {})},
    )
    for batch in range(4):
        start = batch * 10
        table.append(
            pa.table(
                {
                    "id": pa.array(range(start, start + 10), type=pa.int64()),
                    "v": pa.array(
                        [float("nan") if i % nan_every == 0 else float(i) for i in range(10)],
                        type=pa.float64(),
                    ),
                }
            )
        )
    table.refresh()
    return table


def test_a_binpack_records_the_unsorted_order(local_catalog):
    """Bin-packing does not order rows, so it claims no order."""
    table = make_seeded_table(local_catalog, "default.t_so_binpack", n_files=6)

    Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL)

    table.refresh()
    assert _sort_order_ids(table) == {0}


def test_a_sort_matching_a_registered_order_records_it(local_catalog):
    """A rewrite claims an order the table already knows about."""
    from pyiceberg.table.sorting import NullOrder, SortDirection, SortField, SortOrder
    from pyiceberg.transforms import IdentityTransform

    table = make_seeded_table(local_catalog, "default.t_so_sort", n_files=6)
    with table.transaction() as transaction:
        transaction.update_sort_order().asc("id", IdentityTransform(), NullOrder.NULLS_LAST).commit()
    table.refresh()
    registered = [oid for oid in table.sort_orders() if oid != 0]
    assert registered, "the fixture must register a sort order"

    Table.from_iceberg(table).rewrite_data_files("sort", sort_order=_SORT_BY_ID, options=_REWRITE_ALL)

    table.refresh()
    assert _sort_order_ids(table) == {registered[0]}
    assert sorted(table.sort_orders()) == sorted([0, *registered]), "no order should be added"
    _ = (SortOrder, SortField, SortDirection)


def test_a_sort_matching_no_registered_order_records_unsorted(local_catalog):
    """An ad-hoc order is not registered, so the output cannot claim it.

    The reference behaves the same way: it looks for a table sort order equal to
    the one asked for and falls back to unsorted, warning that the files will
    not be marked as sorted. Registering instead would change metadata every
    other writer reads.
    """
    table = make_seeded_table(local_catalog, "default.t_so_adhoc", n_files=6)
    before = sorted(table.sort_orders())

    Table.from_iceberg(table).rewrite_data_files("sort", sort_order=_SORT_BY_ID, options=_REWRITE_ALL)

    table.refresh()
    assert _sort_order_ids(table) == {0}
    assert sorted(table.sort_orders()) == before, "no sort order should be registered"


def test_a_zorder_records_the_unsorted_order(local_catalog):
    """A space-filling curve is not a sequence of per-column sorts."""
    table = local_catalog.create_table(
        identifier="default.t_so_zorder",
        schema=pa.schema([("id", pa.int64()), ("g", pa.int64())]),
        properties={"format-version": "2"},
    )
    for batch in range(4):
        start = batch * 10
        table.append(
            pa.table(
                {
                    "id": pa.array(range(start, start + 10), type=pa.int64()),
                    "g": pa.array([batch] * 10, type=pa.int64()),
                }
            )
        )
    table.refresh()

    Table.from_iceberg(table).rewrite_data_files("zorder", zorder_by=["id", "g"], options=_REWRITE_ALL)

    table.refresh()
    assert _sort_order_ids(table) == {0}


def test_a_rewrite_counts_nan_values_per_column(local_catalog):
    """The count is taken while writing, since a Parquet footer does not carry one."""
    table = _seed_with_nans(local_catalog, "default.t_nan")
    expected = sum(1 for v in table.scan().to_arrow().column("v").to_pylist() if v is not None and math.isnan(v))
    assert expected == 8

    Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL)

    table.refresh()
    counts = _nan_counts(table)
    assert sum(c.get(2, 0) for c in counts) == expected


def test_a_column_without_nan_values_records_none(local_catalog):
    """Absent means none here, not unknown, because the column was counted."""
    table = make_seeded_table(local_catalog, "default.t_nan_absent", n_files=6)

    Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL)

    table.refresh()
    assert all(counts == {} for counts in _nan_counts(table))


def test_disabling_metrics_for_a_column_suppresses_its_nan_count(local_catalog):
    """The metrics configuration decides which columns are measured at all."""
    table = _seed_with_nans(
        local_catalog,
        "default.t_nan_off",
        properties={"write.metadata.metrics.column.v": "none"},
    )

    Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL)

    table.refresh()
    assert all(counts == {} for counts in _nan_counts(table))


def _seed_nested_floats(catalog, name: str) -> Any:
    """Seed a table with floats nested in a struct, a list, and a map."""
    nan = float("nan")
    table = catalog.create_table(
        identifier=name,
        schema=pa.schema(
            [
                ("id", pa.int64()),
                ("s", pa.struct([("inner", pa.float64())])),
                ("l", pa.large_list(pa.float64())),
                ("m", pa.map_(pa.large_string(), pa.float64())),
            ]
        ),
        properties={"format-version": "2"},
    )
    for batch in range(4):
        start = batch * 10
        table.append(
            pa.table(
                {
                    "id": pa.array(range(start, start + 10), type=pa.int64()),
                    "s": pa.array([{"inner": nan if i % 2 else 1.0} for i in range(10)]),
                    "l": pa.array([[nan, 2.0] if i % 5 == 0 else [3.0] for i in range(10)]),
                    "m": pa.array(
                        [[("k", nan if i % 5 == 0 else 4.0)] for i in range(10)],
                        type=pa.map_(pa.large_string(), pa.float64()),
                    ),
                }
            )
        )
    table.refresh()
    return table


def test_a_rewrite_counts_nan_in_nested_columns(local_catalog):
    """A float inside a struct, list, or map is a leaf like any other.

    The reference collects metrics with a visitor that runs on every leaf, so
    leaving nested columns uncounted would prune files a NaN row matches.
    """
    table = _seed_nested_floats(local_catalog, "default.t_nan_nested")
    ids = {field.field_id for field in table.schema().fields}

    Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL)

    table.refresh()
    counts: dict[int, int] = {}
    for entry in _nan_counts(table):
        for field_id, count in entry.items():
            counts[field_id] = counts.get(field_id, 0) + count

    nested = {field_id: count for field_id, count in counts.items() if field_id not in ids}
    assert nested, f"nested floats should carry NaN counts, got {counts}"
    # 20 in the struct (every other row of 40), 8 in the list and 8 in the map
    # (two rows per batch of ten, each carrying one).
    assert sorted(nested.values()) == [8, 8, 20]
