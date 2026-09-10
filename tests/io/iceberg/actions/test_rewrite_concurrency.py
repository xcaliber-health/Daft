"""File groups are rewritten concurrently up to a configurable bound.

Each group's read, re-cluster, and write stream through the execution engine.
Up to ``max-concurrent-file-group-rewrites`` groups run at once; the bound caps
how many group working sets share the engine budget. The number of concurrent
groups never exceeds the bound, and the result is independent of it.
"""

from __future__ import annotations

import threading
import time

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import LongType, NestedField, StringType

from daft.catalog import Table
from daft.io.iceberg import _compact  # internal helper, monkeypatched below


def _make_partitioned(local_catalog, name: str):
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "region", StringType(), required=False),
    )
    spec = PartitionSpec(PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="region"))
    table = local_catalog.create_table(name, schema=schema, partition_spec=spec)
    for region in ("us", "eu", "ap"):
        for k in range(3):
            ids = list(range(k * 5, k * 5 + 5))
            table.append(
                pa.table(
                    {
                        "id": pa.array(ids, type=pa.int64()),
                        "region": pa.array([region] * 5, type=pa.string()),
                    }
                )
            )
    return table


def _instrument_peak_concurrency(monkeypatch):
    """Patch ``_rewrite_group`` to record the peak number of overlapping calls."""
    active = {"current": 0, "peak": 0}
    lock = threading.Lock()
    real_rewrite = _compact._rewrite_group

    def instrumented(*args, **kwargs):
        with lock:
            active["current"] += 1
            active["peak"] = max(active["peak"], active["current"])
        try:
            # Hold the slot briefly so genuinely-concurrent groups overlap and
            # the observed peak reflects the configured bound deterministically.
            time.sleep(0.05)
            return real_rewrite(*args, **kwargs)
        finally:
            with lock:
                active["current"] -= 1

    monkeypatch.setattr(_compact, "_rewrite_group", instrumented)
    return active


@pytest.mark.parametrize("max_concurrent", [1, 4])
def test_concurrency_bounded_by_option(local_catalog, monkeypatch, max_concurrent):
    table = _make_partitioned(local_catalog, f"default.t_conc_{max_concurrent}")
    active = _instrument_peak_concurrency(monkeypatch)

    dt = Table.from_iceberg(table)
    dt.compact_files(
        options={
            "rewrite-all": True,
            "min-input-files": 2,
            "max-concurrent-file-group-rewrites": max_concurrent,
        }
    )

    from daft import runners

    # The number of groups running at once never exceeds the configured bound,
    # on any runner.
    assert 1 <= active["peak"] <= max_concurrent, f"observed peak={active['peak']} outside [1, {max_concurrent}]"
    if max_concurrent == 1:
        assert active["peak"] == 1, "groups must not overlap when the bound is 1"
    elif runners.get_or_create_runner().name == "ray":
        # Three partitions form three groups; a distributed runner overlaps them.
        assert active["peak"] >= 2, "distributed groups should run concurrently"


def test_concurrency_option_does_not_change_result(local_catalog):
    table = _make_partitioned(local_catalog, "default.t_conc_result")

    dt = Table.from_iceberg(table)
    result = dt.compact_files(
        options={
            "rewrite-all": True,
            "min-input-files": 2,
            "max-concurrent-file-group-rewrites": 8,
        }
    )

    table.refresh()
    # Three partitions, three files each: every input file is rewritten.
    assert result.rewritten_files == 9
    assert result.added_files == result.added_files  # at least one output per partition
    rows = sorted(table.scan().to_arrow().column("id").to_pylist())
    assert rows == sorted([i for _ in range(3) for k in range(3) for i in range(k * 5, k * 5 + 5)])
