from __future__ import annotations

import math

import pytest

pytest.importorskip("pyiceberg")

from daft.io.writer import (
    _DEFAULT_BLOOM_FPP,
    _DEFAULT_BLOOM_MAX_BYTES,
    _distinct_values_for_byte_budget,
    _iceberg_bloom_filter_options,
    _resolve_iceberg_writer_options,
)


def _bits(ndv: int, fpp: float) -> int:
    """Forward split-block sizing relation: bits for a distinct-value count."""
    return math.ceil(-8.0 * ndv / math.log(1.0 - fpp ** (1.0 / 8.0)))


class TestBloomSizingMath:
    @pytest.mark.parametrize("fpp", [0.1, 0.05, 0.01, 0.001])
    @pytest.mark.parametrize("max_bytes", [32 * 1024, 256 * 1024, _DEFAULT_BLOOM_MAX_BYTES])
    def test_derived_count_fits_byte_budget(self, fpp: float, max_bytes: int) -> None:
        ndv = _distinct_values_for_byte_budget(fpp, max_bytes)

        assert ndv >= 1
        # Validated against an independent forward computation of the relation.
        assert _bits(ndv, fpp) <= max_bytes * 8

    @pytest.mark.parametrize("fpp", [0.1, 0.05, 0.01, 0.001])
    def test_derived_count_is_maximal(self, fpp: float) -> None:
        # One more distinct value than derived should overflow the budget.
        max_bytes = _DEFAULT_BLOOM_MAX_BYTES
        ndv = _distinct_values_for_byte_budget(fpp, max_bytes)

        assert _bits(ndv + 1, fpp) > max_bytes * 8

    def test_smaller_budget_yields_fewer_distinct_values(self) -> None:
        small = _distinct_values_for_byte_budget(0.01, 64 * 1024)
        large = _distinct_values_for_byte_budget(0.01, _DEFAULT_BLOOM_MAX_BYTES)

        assert small < large


class TestBloomOptionParsing:
    def test_no_columns_enabled_returns_none(self) -> None:
        assert _iceberg_bloom_filter_options({"write.parquet.compression-codec": "zstd"}) is None

    def test_enable_flag_is_case_insensitive_and_excludes_false(self) -> None:
        props = {
            "write.parquet.bloom-filter-enabled.column.id": "true",
            "write.parquet.bloom-filter-enabled.column.name": "TRUE",
            "write.parquet.bloom-filter-enabled.column.skip": "false",
        }

        options = _iceberg_bloom_filter_options(props)

        assert options is not None
        assert set(options) == {"id", "name"}

    def test_default_probability_applied_when_unset(self) -> None:
        options = _iceberg_bloom_filter_options({"write.parquet.bloom-filter-enabled.column.id": "true"})

        assert options is not None
        assert options["id"]["fpp"] == _DEFAULT_BLOOM_FPP
        assert options["id"]["ndv"] == _distinct_values_for_byte_budget(_DEFAULT_BLOOM_FPP, _DEFAULT_BLOOM_MAX_BYTES)

    def test_per_column_probability_overrides_default(self) -> None:
        props = {
            "write.parquet.bloom-filter-enabled.column.name": "true",
            "write.parquet.bloom-filter-fpp.column.name": "0.05",
        }

        options = _iceberg_bloom_filter_options(props)

        assert options is not None
        assert options["name"]["fpp"] == 0.05

    def test_byte_budget_controls_distinct_value_count(self) -> None:
        props = {
            "write.parquet.bloom-filter-enabled.column.id": "true",
            "write.parquet.bloom-filter-max-bytes": "524288",
        }

        options = _iceberg_bloom_filter_options(props)

        assert options is not None
        assert options["id"]["ndv"] == _distinct_values_for_byte_budget(_DEFAULT_BLOOM_FPP, 524288)
        assert _bits(options["id"]["ndv"], _DEFAULT_BLOOM_FPP) <= 524288 * 8

    def test_nested_column_path_preserved(self) -> None:
        props = {"write.parquet.bloom-filter-enabled.column.address.zip": "true"}

        options = _iceberg_bloom_filter_options(props)

        assert options is not None
        assert "address.zip" in options


class TestResolveWriterOptions:
    def test_bloom_options_included_when_enabled(self) -> None:
        out = _resolve_iceberg_writer_options({"write.parquet.bloom-filter-enabled.column.id": "true"})

        assert "bloom_filter_options" in out
        assert "id" in out["bloom_filter_options"]

    def test_bloom_options_absent_when_disabled(self) -> None:
        out = _resolve_iceberg_writer_options({"write.parquet.compression-codec": "zstd"})

        assert "bloom_filter_options" not in out


from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC
from pyiceberg.schema import Schema
from pyiceberg.types import LongType, NestedField, StringType

import daft
from daft import col

_BLOOM_TABLE_PROPERTIES = {
    "write.parquet.bloom-filter-enabled.column.id": "true",
    "write.parquet.bloom-filter-enabled.column.name": "true",
    "write.parquet.bloom-filter-fpp.column.id": "0.01",
    "write.parquet.bloom-filter-max-bytes": "1048576",
}


@pytest.fixture
def bloom_catalog(tmp_path):
    catalog = SqlCatalog(
        "default",
        uri=f"sqlite:///{tmp_path}/pyiceberg_catalog.db",
        warehouse=f"file://{tmp_path}",
    )
    catalog.create_namespace("default")
    yield catalog
    catalog.engine.dispose()


def _create_bloom_table(catalog, name: str = "default.bloom"):
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "name", StringType(), required=False),
    )
    return catalog.create_table(
        identifier=name,
        schema=schema,
        partition_spec=UNPARTITIONED_PARTITION_SPEC,
        properties=_BLOOM_TABLE_PROPERTIES,
    )


class TestIcebergBloomEndToEnd:
    """Write through the real writer with bloom properties, then read back.

    A present value must survive (proves the read probe agrees with the writer's
    hashing — a wrong "absent" verdict would drop matching rows); an absent value
    must return nothing.
    """

    def test_equality_round_trips(self, bloom_catalog) -> None:
        table = _create_bloom_table(bloom_catalog)
        rows = 2000
        df = daft.from_pydict({"id": list(range(rows)), "name": [f"n{i}" for i in range(rows)]})
        df.write_iceberg(table)
        table.refresh()

        read = daft.read_iceberg(table)
        assert read.where(col("id") == 1234).to_pydict()["id"] == [1234]
        assert read.where(col("id") == 9_999_999).to_pydict()["id"] == []
        assert read.where(col("name") == "n777").to_pydict()["name"] == ["n777"]
        assert read.where(col("name") == "missing").to_pydict()["name"] == []

    def test_membership_round_trips(self, bloom_catalog) -> None:
        table = _create_bloom_table(bloom_catalog, name="default.bloom_in")
        df = daft.from_pydict({"id": list(range(1000)), "name": [f"n{i}" for i in range(1000)]})
        df.write_iceberg(table)
        table.refresh()

        read = daft.read_iceberg(table)
        got = sorted(read.where(col("id").is_in([5, 17, 9_999_999])).to_pydict()["id"])
        assert got == [5, 17]

    def test_full_scan_unaffected(self, bloom_catalog) -> None:
        table = _create_bloom_table(bloom_catalog, name="default.bloom_full")
        df = daft.from_pydict({"id": list(range(500)), "name": [f"n{i}" for i in range(500)]})
        df.write_iceberg(table)
        table.refresh()

        read = daft.read_iceberg(table)
        assert sorted(read.to_pydict()["id"]) == list(range(500))
