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

    @pytest.mark.parametrize("bad_fpp", ["0", "1", "-0.1", "1.5"])
    def test_probability_out_of_range_raises(self, bad_fpp: str) -> None:
        props = {
            "write.parquet.bloom-filter-enabled.column.id": "true",
            "write.parquet.bloom-filter-fpp.column.id": bad_fpp,
        }

        with pytest.raises(ValueError, match="open interval"):
            _iceberg_bloom_filter_options(props)

    def test_unknown_column_is_ignored_when_schema_known(self) -> None:
        props = {
            "write.parquet.bloom-filter-enabled.column.id": "true",
            "write.parquet.bloom-filter-enabled.column.ghost": "true",
        }

        options = _iceberg_bloom_filter_options(props, valid_columns={"id", "name"})

        assert options is not None
        assert set(options) == {"id"}

    def test_all_unknown_columns_returns_none(self) -> None:
        props = {"write.parquet.bloom-filter-enabled.column.ghost": "true"}

        assert _iceberg_bloom_filter_options(props, valid_columns={"id"}) is None

    def test_explicit_ndv_below_budget_is_honored(self) -> None:
        props = {
            "write.parquet.bloom-filter-enabled.column.id": "true",
            "write.parquet.bloom-filter-ndv.column.id": "1000",
        }

        options = _iceberg_bloom_filter_options(props)

        # A small count sizes the filter below the byte budget, so it is honored as-is.
        assert options is not None
        assert options["id"]["ndv"] == 1000

    def test_explicit_ndv_above_budget_is_capped(self) -> None:
        props = {
            "write.parquet.bloom-filter-enabled.column.id": "true",
            "write.parquet.bloom-filter-ndv.column.id": "100000000000",
        }

        options = _iceberg_bloom_filter_options(props)

        assert options is not None
        assert options["id"]["ndv"] == _distinct_values_for_byte_budget(_DEFAULT_BLOOM_FPP, _DEFAULT_BLOOM_MAX_BYTES)

    @pytest.mark.parametrize("bad_ndv", ["0", "-5"])
    def test_non_positive_ndv_raises(self, bad_ndv: str) -> None:
        props = {
            "write.parquet.bloom-filter-enabled.column.id": "true",
            "write.parquet.bloom-filter-ndv.column.id": bad_ndv,
        }

        with pytest.raises(ValueError, match="must be positive"):
            _iceberg_bloom_filter_options(props)


class TestResolveWriterOptions:
    def test_bloom_options_included_when_enabled(self) -> None:
        out = _resolve_iceberg_writer_options({"write.parquet.bloom-filter-enabled.column.id": "true"})

        assert out.bloom_filter_options is not None
        assert "id" in out.bloom_filter_options

    def test_bloom_options_absent_when_disabled(self) -> None:
        out = _resolve_iceberg_writer_options({"write.parquet.compression-codec": "zstd"})

        assert out.bloom_filter_options is None


import datetime
import decimal

import pyarrow as pa
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC
from pyiceberg.schema import Schema
from pyiceberg.types import (
    BinaryType,
    DateType,
    DecimalType,
    DoubleType,
    IntegerType,
    LongType,
    NestedField,
    StringType,
    TimestampType,
)

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


def _data_file_bytes(table) -> int:
    """Total bytes of the table's data files, summed from scan planning."""
    table.refresh()
    return sum(task.file.file_size_in_bytes for task in table.scan().plan_files())


def _create_long_table(catalog, name: str, properties: dict[str, str]):
    schema = Schema(NestedField(1, "id", LongType(), required=False))
    return catalog.create_table(
        identifier=name,
        schema=schema,
        partition_spec=UNPARTITIONED_PARTITION_SPEC,
        properties=properties,
    )


class TestBloomWritePathActive:
    """Prove a filter is actually written to disk and sized like the reference.

    A conservative reader stays correct even with no filter, so result-only checks
    cannot tell "bloom works" from "bloom is a no-op"; these compare on-disk bytes.
    """

    def test_enabled_file_larger_than_disabled(self, bloom_catalog) -> None:
        data = {"id": list(range(5000))}
        on = _create_long_table(
            bloom_catalog,
            "default.b_on",
            {"write.parquet.bloom-filter-enabled.column.id": "true"},
        )
        off = _create_long_table(bloom_catalog, "default.b_off", {})
        daft.from_pydict(data).write_iceberg(on)
        daft.from_pydict(data).write_iceberg(off)

        assert _data_file_bytes(on) > _data_file_bytes(off)

    def test_probability_does_not_change_filter_size(self, bloom_catalog) -> None:
        # Reference parity: the byte budget sets the size; the probability is a
        # hint that does not change it.
        data = {"id": list(range(5000))}
        loose = _create_long_table(
            bloom_catalog,
            "default.b_loose",
            {
                "write.parquet.bloom-filter-enabled.column.id": "true",
                "write.parquet.bloom-filter-fpp.column.id": "0.01",
            },
        )
        tight = _create_long_table(
            bloom_catalog,
            "default.b_tight",
            {
                "write.parquet.bloom-filter-enabled.column.id": "true",
                "write.parquet.bloom-filter-fpp.column.id": "0.001",
            },
        )
        daft.from_pydict(data).write_iceberg(loose)
        daft.from_pydict(data).write_iceberg(tight)

        assert _data_file_bytes(loose) == _data_file_bytes(tight)

    def test_explicit_ndv_shrinks_filter(self, bloom_catalog) -> None:
        # An explicit distinct-value count well below the byte budget produces a
        # smaller on-disk filter than the budget-sized default, and still prunes.
        data = {"id": list(range(5000))}
        budget = _create_long_table(
            bloom_catalog,
            "default.ndv_budget",
            {"write.parquet.bloom-filter-enabled.column.id": "true"},
        )
        sized = _create_long_table(
            bloom_catalog,
            "default.ndv_sized",
            {
                "write.parquet.bloom-filter-enabled.column.id": "true",
                "write.parquet.bloom-filter-ndv.column.id": "5000",
            },
        )
        daft.from_pydict(data).write_iceberg(budget)
        daft.from_pydict(data).write_iceberg(sized)

        assert _data_file_bytes(sized) < _data_file_bytes(budget)

        sized.refresh()
        read = daft.read_iceberg(sized)
        assert read.where(col("id") == 123).to_pydict()["id"] == [123]
        assert read.where(col("id") == 9_999_999).to_pydict()["id"] == []

    @pytest.mark.parametrize("max_bytes", ["97", "100000", "1000003"])
    def test_small_and_nonpow2_budget_round_trips(self, bloom_catalog, max_bytes: str) -> None:
        # Tiny and non-power-of-two budgets must still produce a valid, probeable
        # filter that prunes absent values without dropping present ones.
        table = _create_long_table(
            bloom_catalog,
            f"default.b_budget_{max_bytes}",
            {
                "write.parquet.bloom-filter-enabled.column.id": "true",
                "write.parquet.bloom-filter-max-bytes": max_bytes,
            },
        )
        daft.from_pydict({"id": list(range(2000))}).write_iceberg(table)
        table.refresh()

        read = daft.read_iceberg(table)
        assert read.where(col("id") == 123).to_pydict()["id"] == [123]
        assert read.where(col("id") == 9_999_999).to_pydict()["id"] == []


_N = 1000
_DATE_BASE = datetime.date(2000, 1, 1)
# Iceberg ``timestamp`` is timezone-naive; build naive datetimes without a constructor.
_TS_BASE = datetime.datetime.fromisoformat("2000-01-01T00:00:00")

# (label, iceberg type, arrow type, values, present, absent)
_TYPE_CASES = [
    ("long", LongType(), pa.int64(), list(range(_N)), 123, 9_999_999),
    ("int", IntegerType(), pa.int32(), list(range(_N)), 123, 9_999_999),
    ("double", DoubleType(), pa.float64(), [x + 0.5 for x in range(_N)], 123.5, 9_999_999.5),
    ("string", StringType(), pa.string(), [f"v{x}" for x in range(_N)], "v123", "missing"),
    (
        "binary",
        BinaryType(),
        pa.binary(),
        [x.to_bytes(4, "big") for x in range(_N)],
        (123).to_bytes(4, "big"),
        (9_999_999).to_bytes(4, "big"),
    ),
    (
        "date",
        DateType(),
        pa.date32(),
        [_DATE_BASE + datetime.timedelta(days=x) for x in range(_N)],
        _DATE_BASE + datetime.timedelta(days=123),
        datetime.date(1850, 1, 1),
    ),
    (
        "decimal",
        DecimalType(9, 2),
        pa.decimal128(9, 2),
        [decimal.Decimal(x).quantize(decimal.Decimal("0.01")) for x in range(_N)],
        decimal.Decimal("123.00"),
        decimal.Decimal("99999.99"),
    ),
    (
        "timestamp",
        TimestampType(),
        pa.timestamp("us"),
        [_TS_BASE + datetime.timedelta(seconds=x) for x in range(_N)],
        _TS_BASE + datetime.timedelta(seconds=123),
        datetime.datetime.fromisoformat("1850-01-01T00:00:00"),
    ),
]


class TestBloomTypeMatrix:
    """Probe writer-produced filters across every supported physical type.

    Cross-writer parity: a present value must survive the probe (a wrong encoding
    would drop it) and an absent value must be pruned.
    """

    @pytest.mark.parametrize("case", _TYPE_CASES, ids=[c[0] for c in _TYPE_CASES])
    def test_type_round_trips(self, bloom_catalog, case) -> None:
        label, iceberg_type, arrow_type, values, present, absent = case
        schema = Schema(NestedField(1, "v", iceberg_type, required=False))
        table = bloom_catalog.create_table(
            identifier=f"default.tm_{label}",
            schema=schema,
            partition_spec=UNPARTITIONED_PARTITION_SPEC,
            properties={"write.parquet.bloom-filter-enabled.column.v": "true"},
        )
        arrow_table = pa.table({"v": pa.array(values, type=arrow_type)})
        daft.from_arrow(arrow_table).write_iceberg(table)
        table.refresh()

        read = daft.read_iceberg(table)
        present_rows = read.where(col("v") == present).to_pydict()["v"]
        absent_rows = read.where(col("v") == absent).to_pydict()["v"]
        assert len(present_rows) == 1, f"present value dropped for {label}"
        assert absent_rows == [], f"absent value not pruned for {label}"
