"""How a rewrite seeds the writer's estimate of data expansion from disk into memory."""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest

pytest.importorskip("pyiceberg")

from daft.catalog import Table
from daft.io.iceberg import _compact  # internal helper, monkeypatched below
from tests.io.iceberg.actions._helpers import make_seeded_table, scan_paths, strip_scheme

_REWRITE = {"rewrite-all": True, "min-input-files": 2}


def test_inflation_is_in_memory_bytes_over_bytes_on_disk(local_catalog):
    # Arrange
    table = make_seeded_table(local_catalog, "default.t_inflation_measure", n_files=2, rows_per_file=500)
    paths = sorted(scan_paths(table))
    in_memory = 0
    on_disk = 0
    for path in paths:
        parquet_file = pq.ParquetFile(strip_scheme(path))
        group = parquet_file.metadata.row_group(0)
        on_disk += sum(group.column(c).total_compressed_size for c in range(group.num_columns))
        in_memory += parquet_file.read_row_group(0).nbytes

    # Act
    measured = _compact._measure_inflation_factor(table, paths)

    # Assert: the ratio the writer compares, not the footer's compression ratio.
    assert measured == pytest.approx(in_memory / on_disk)
    assert measured > 1.0


def _captured_inflation(monkeypatch) -> dict[str, float | None]:
    seen: dict[str, float | None] = {}
    real = _compact._collect_data_files

    def capture(**kwargs):
        seen["inflation_factor"] = kwargs["inflation_factor"]
        return real(**kwargs)

    monkeypatch.setattr(_compact, "_collect_data_files", capture)
    return seen


@pytest.mark.parametrize(
    ("strategy", "extra"),
    [
        pytest.param("binpack", {}, id="binpack"),
        pytest.param("sort", {"sort_order": [("id", "asc", "nulls-last")]}, id="sort"),
        pytest.param("zorder", {"zorder_by": ["id"]}, id="zorder"),
    ],
)
def test_every_strategy_seeds_the_writer_with_the_measured_ratio(local_catalog, monkeypatch, strategy, extra):
    # Arrange
    table = make_seeded_table(local_catalog, f"default.t_inflation_{strategy}", n_files=3, rows_per_file=200)
    expected = _compact._measure_inflation_factor(table, sorted(scan_paths(table)))
    seen = _captured_inflation(monkeypatch)

    # Act
    Table.from_iceberg(table).rewrite_data_files(strategy, options=_REWRITE, **extra)

    # Assert
    assert seen["inflation_factor"] == pytest.approx(expected)


def test_compression_factor_overrides_the_measurement_for_clustering_only(local_catalog, monkeypatch):
    # Arrange: an explicit factor, applied to a sort and to a bin-pack.
    sorted_table = make_seeded_table(local_catalog, "default.t_inflation_cf_sort", n_files=3, rows_per_file=200)
    packed_table = make_seeded_table(local_catalog, "default.t_inflation_cf_pack", n_files=3, rows_per_file=200)
    measured = _compact._measure_inflation_factor(packed_table, sorted(scan_paths(packed_table)))
    options = {**_REWRITE, "compression-factor": 4.5}

    # Act
    seen_sort = _captured_inflation(monkeypatch)
    Table.from_iceberg(sorted_table).rewrite_data_files(
        "sort", sort_order=[("id", "asc", "nulls-last")], options=options
    )
    sort_factor = seen_sort["inflation_factor"]
    seen_pack = _captured_inflation(monkeypatch)
    Table.from_iceberg(packed_table).rewrite_data_files("binpack", options=options)

    # Assert: the clustering strategy takes the caller's factor; bin-pack keeps the measurement.
    assert sort_factor == pytest.approx(4.5)
    assert seen_pack["inflation_factor"] == pytest.approx(measured)
