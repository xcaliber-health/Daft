"""Execution-time row-group splitting of file scans.

With the flag enabled, a single file's row groups are read by several
workers in parallel; results must be identical to the unsplit read,
including row order when order is maintained.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as papq
import pytest

import daft
from daft import col
from daft.context import execution_config_ctx

if TYPE_CHECKING:
    import pathlib


@pytest.fixture(scope="module")
def multi_row_group_file(tmp_path_factory: pytest.TempPathFactory) -> str:
    """One file holding many small row groups."""
    path = str(tmp_path_factory.mktemp("rg") / "many_groups.parquet")
    n = 100_000
    table = pa.table({"idx": pa.array(range(n), type=pa.int64()), "val": pa.array([i * 2 for i in range(n)])})
    papq.write_table(table, path, row_group_size=5_000)
    return path


@pytest.mark.parametrize("maintain_order", [True, False])
def test_split_read_matches_unsplit(multi_row_group_file: str, maintain_order: bool) -> None:
    def read() -> dict[str, list[int]]:
        df = daft.read_parquet(multi_row_group_file).where(col("idx") % 3 == 0)
        return df.to_pydict()

    with execution_config_ctx(enable_scan_task_row_group_splitting=False, maintain_order=maintain_order):
        expected = read()
    with execution_config_ctx(enable_scan_task_row_group_splitting=True, maintain_order=maintain_order):
        got = read()

    if maintain_order:
        assert got == expected
    else:
        assert sorted(got["idx"]) == sorted(expected["idx"])
        assert sorted(got["val"]) == sorted(expected["val"])


def test_split_read_preserves_row_order(multi_row_group_file: str) -> None:
    with execution_config_ctx(enable_scan_task_row_group_splitting=True, maintain_order=True):
        got = daft.read_parquet(multi_row_group_file).to_pydict()
    assert got["idx"] == list(range(100_000))


def test_split_respects_limit_pushdown(multi_row_group_file: str) -> None:
    """A limit pushdown disables splitting; the result must still be correct."""
    with execution_config_ctx(enable_scan_task_row_group_splitting=True):
        got = daft.read_parquet(multi_row_group_file).limit(7).to_pydict()
    assert got["idx"] == list(range(7))


def test_flag_off_by_default() -> None:
    assert daft.context.get_context().daft_execution_config.enable_scan_task_row_group_splitting is False


def test_split_single_row_group_file_is_unchanged(tmp_path: pathlib.Path) -> None:
    path = str(tmp_path / "one_group.parquet")
    papq.write_table(pa.table({"a": list(range(100))}), path)
    with execution_config_ctx(enable_scan_task_row_group_splitting=True):
        got = daft.read_parquet(path).to_pydict()
    assert got["a"] == list(range(100))
