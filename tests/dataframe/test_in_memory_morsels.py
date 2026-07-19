"""Large single-chunk in-memory partitions flow through execution correctly.

Oversized in-memory partitions are re-emitted as row-range slices so all
workers participate; results must be identical to the unsliced input,
including row order when order is maintained.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import daft
from daft import col
from daft.context import execution_config_ctx

N = 1_000_000


@pytest.fixture(scope="module")
def big_single_chunk_df() -> daft.DataFrame:
    """One partition holding a single million-row chunk."""
    table = pa.table({"idx": pa.array(range(N)), "val": pa.array([i % 97 for i in range(N)])})
    return daft.from_arrow(table).collect()


def test_passthrough_preserves_order(big_single_chunk_df: daft.DataFrame) -> None:
    got = big_single_chunk_df.select(col("idx")).to_pydict()
    assert got["idx"] == list(range(N))


def test_filter_preserves_order(big_single_chunk_df: daft.DataFrame) -> None:
    got = big_single_chunk_df.where(col("idx") % 10 == 3).to_pydict()
    assert got["idx"] == list(range(3, N, 10))


def test_groupby_matches_unsliced_semantics(big_single_chunk_df: daft.DataFrame) -> None:
    got = big_single_chunk_df.groupby("val").agg(col("idx").count().alias("n")).to_pydict()
    counts = dict(zip(got["val"], got["n"]))
    expected_full = N // 97
    assert len(counts) == 97
    assert all(n in (expected_full, expected_full + 1) for n in counts.values())
    assert sum(counts.values()) == N


def test_limit_returns_leading_rows(big_single_chunk_df: daft.DataFrame) -> None:
    got = big_single_chunk_df.limit(5).to_pydict()
    assert got["idx"] == [0, 1, 2, 3, 4]


def test_unordered_execution_returns_same_multiset(big_single_chunk_df: daft.DataFrame) -> None:
    with execution_config_ctx(maintain_order=False):
        got = big_single_chunk_df.select(col("idx")).to_pydict()
    assert sorted(got["idx"]) == list(range(N))
