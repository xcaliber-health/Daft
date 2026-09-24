from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import date, datetime
from pathlib import Path

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import DataScan, FileScanTask, Table
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import DoubleType, LongType, NestedField

import daft
from daft import DataType, Expression, col, lit

ROWS = pa.table(
    {
        "x": pa.array([74.5, 3.0, 120.25], pa.float64()),
        "n": pa.array([1, 5, 200], pa.int64()),
        "v": pa.array(["7.5", "12.0", "0.5"], pa.string()),
        "ts": pa.array(
            [datetime(2024, 1, 1, 9, 30), datetime(2024, 1, 2, 0, 0), datetime(2024, 1, 3, 18, 0)],
            pa.timestamp("us"),
        ),
    }
)


@pytest.fixture
def catalog(tmp_path: Path) -> Iterator[SqlCatalog]:
    held = SqlCatalog("pruning", uri=f"sqlite:///{tmp_path}/catalog.db", warehouse=f"file://{tmp_path}")
    held.create_namespace("n")
    yield held


@pytest.fixture
def one_file_per_row(catalog: SqlCatalog) -> Table:
    """A table whose every row is alone in its file, so a wrongly pruned file loses its row."""
    table = catalog.create_table("n.rows", schema=ROWS.schema)
    for i in range(ROWS.num_rows):
        table.append(ROWS.slice(i, 1))
    return catalog.load_table("n.rows")


@pytest.mark.parametrize(
    "predicate",
    [
        pytest.param(col("x").cast(DataType.int64()) == 74, id="converted column equals"),
        pytest.param(col("x").try_cast(DataType.int64()) == 74, id="try-converted column equals"),
        pytest.param(col("v").cast(DataType.decimal128(18, 2)) < 8.0, id="text read as a decimal"),
        pytest.param(col("ts").cast(DataType.date()) == date(2024, 1, 1), id="moment read as its date"),
        pytest.param(
            (col("x").cast(DataType.int64()) == 74) & (col("n") < 100),
            id="conjunction with one convertible side",
        ),
        pytest.param(
            (col("x").cast(DataType.int64()) == 3) | (col("n") > 100),
            id="disjunction with a converted side",
        ),
        pytest.param(~(col("x").cast(DataType.int64()) == 74), id="negated conversion"),
        pytest.param(col("n") > lit(4).cast(DataType.int64()), id="converted constant"),
    ],
)
def test_a_filter_answers_as_it_does_in_memory(one_file_per_row: Table, predicate: Expression) -> None:
    expected = daft.from_arrow(ROWS).where(predicate).sort("n").to_pydict()

    observed = daft.read_iceberg(one_file_per_row).where(predicate).sort("n").to_pydict()

    assert observed == expected


def test_a_count_under_a_partition_filter_counts_only_the_matching_rows(catalog: SqlCatalog) -> None:
    schema = Schema(
        NestedField(1, "p", LongType(), required=False),
        NestedField(2, "x", DoubleType(), required=False),
    )
    spec = PartitionSpec(PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="p"))
    table = catalog.create_table("n.partitioned", schema=schema, partition_spec=spec)
    table.append(pa.table({"p": pa.array([1, 1, 2], pa.int64()), "x": pa.array([1.0, 2.0, 3.0])}))

    counted = daft.read_iceberg(table).where(col("p") == 1).count_rows()

    assert counted == 2


def test_an_unfiltered_count_counts_every_row(one_file_per_row: Table) -> None:
    assert daft.read_iceberg(one_file_per_row).count_rows() == ROWS.num_rows


def test_building_a_read_plans_no_files(one_file_per_row: Table, monkeypatch: pytest.MonkeyPatch) -> None:
    planned: list[DataScan] = []
    plan_files = DataScan.plan_files

    def counting(scan: DataScan) -> Iterable[FileScanTask]:
        planned.append(scan)
        return plan_files(scan)

    monkeypatch.setattr(DataScan, "plan_files", counting)

    daft.read_iceberg(one_file_per_row)

    assert planned == []
