"""An overwrite that replaces the rows whose key is in a key set.

A row's key is its values on the key columns, all together. The rows whose key
is in the key set K are removed and the new rows written, in one snapshot; a
row with a null in any key column never matches and is kept, and a key of K
with no new row still loses its rows. Every new row must have its key in K.
Only files whose partition and bounds may hold a key of K are read, and a
concurrent commit conflicts only when a file it added may hold one.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable
from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC, PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table.snapshots import Operation
from pyiceberg.transforms import BucketTransform, IdentityTransform
from pyiceberg.types import LongType, NestedField, StringType

import daft
from daft import col
from daft.exceptions import DaftValueError
from daft.io.iceberg import WriteConflict
from tests.io.iceberg.actions._helpers import commit_equality_delete_rows, snapshot_count

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog
    from pyiceberg.table import FileScanTask, Table

SCHEMA = Schema(
    NestedField(1, "id", LongType(), required=False),
    NestedField(2, "region", StringType(), required=False),
    NestedField(3, "v", LongType(), required=False),
)

SPECS = {
    "unpartitioned": UNPARTITIONED_PARTITION_SPEC,
    "by the key": PartitionSpec(PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="id")),
    "bucketed by the key": PartitionSpec(
        PartitionField(source_id=1, field_id=1000, transform=BucketTransform(4), name="id_bucket")
    ),
    "by another column": PartitionSpec(
        PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="region")
    ),
}

Row = tuple[int | None, str | None, int | None]

SEED: list[list[Row]] = [
    [(1, "eu", 10), (2, "eu", 20)],
    [(3, "us", 30), (4, "us", 40)],
    [(5, "eu", 50), (None, "us", 0)],
]


def _frame(rows: list[Row]) -> daft.DataFrame:
    return daft.from_arrow(
        pa.table(
            {
                "id": pa.array([r[0] for r in rows], type=pa.int64()),
                "region": pa.array([r[1] for r in rows], type=pa.string()),
                "v": pa.array([r[2] for r in rows], type=pa.int64()),
            }
        )
    )


def _keys(ids: list[int], *, dtype: pa.DataType | None = None) -> daft.DataFrame:
    return daft.from_arrow(pa.table({"id": pa.array(ids, type=dtype or pa.int64())}))


def _table(catalog: Catalog, spec: PartitionSpec, files: list[list[Row]], name: str = "default.t") -> Table:
    table = catalog.create_table(name, schema=SCHEMA, partition_spec=spec, properties={"format-version": "2"})
    for rows in files:
        _frame(rows).write_iceberg(table, mode="append")
    return table


def _rows(table: Table) -> list[Row]:
    table.refresh()
    data = daft.read_iceberg(table).to_pydict()
    return sorted(zip(data["id"], data["region"], data["v"]), key=repr)


def _sorted(rows: list[Row]) -> list[Row]:
    return sorted(rows, key=repr)


def _counted_files(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the files the overwrite reads to count rows in scope."""
    from daft.io.iceberg import _overwrite

    counted: list[str] = []
    real_count = _overwrite._count_rows

    def count_rows(table, plan, paths, *args, **kwargs):  # type: ignore[no-untyped-def]
        counted.extend(paths)
        return real_count(table, plan, paths, *args, **kwargs)

    monkeypatch.setattr(_overwrite, "_count_rows", count_rows)
    return counted


@pytest.mark.parametrize("spec", list(SPECS.values()), ids=list(SPECS))
@pytest.mark.parametrize(
    ("keys", "new_rows", "kept"),
    [
        pytest.param(
            [2, 4],
            [(2, "eu", 21), (4, "us", 41)],
            [(1, "eu", 10), (3, "us", 30), (5, "eu", 50), (None, "us", 0)],
            id="keys across files",
        ),
        pytest.param(
            [3, 9],
            [(9, "us", 90)],
            [(1, "eu", 10), (2, "eu", 20), (4, "us", 40), (5, "eu", 50), (None, "us", 0)],
            id="a key with no new row loses its rows",
        ),
        pytest.param(
            [1, 2, 3, 4, 5],
            [],
            [(None, "us", 0)],
            id="a null key is never matched",
        ),
        pytest.param(
            [],
            [],
            [row for rows in SEED for row in rows],
            id="no keys",
        ),
    ],
)
def test_only_the_rows_whose_key_is_in_the_key_set_are_replaced(
    local_catalog: Catalog, spec: PartitionSpec, keys: list[int], new_rows: list[Row], kept: list[Row]
) -> None:
    table = _table(local_catalog, spec, SEED)

    _frame(new_rows).write_iceberg(table, mode="overwrite", overwrite_keys=_keys(keys))

    assert _rows(table) == _sorted(kept + new_rows)


def test_a_key_matches_on_every_key_column_together(local_catalog: Catalog) -> None:
    table = _table(local_catalog, UNPARTITIONED_PARTITION_SPEC, [[(1, "eu", 10), (1, "us", 11), (2, "eu", 20)]])
    # Each column's values appear in the key set, but only (1, "eu") is a key of it.
    keys = daft.from_pydict({"id": [1, 2], "region": ["eu", "us"]})

    _frame([(1, "eu", 12)]).write_iceberg(table, mode="overwrite", overwrite_keys=keys)

    assert _rows(table) == _sorted([(1, "eu", 12), (1, "us", 11), (2, "eu", 20)])


def test_the_write_is_one_overwrite_snapshot_stating_each_files_facts(local_catalog: Catalog) -> None:
    table = _table(local_catalog, UNPARTITIONED_PARTITION_SPEC, SEED)
    before = snapshot_count(table)

    result = (
        _frame([(2, "eu", 21)])
        .write_iceberg(table, mode="overwrite", overwrite_keys=_keys([2, 3, 4]), snapshot_properties={"run": "r1"})
        .to_pydict()
    )

    table.refresh()
    assert snapshot_count(table) == before + 1
    assert table.current_snapshot().summary.operation == Operation.OVERWRITE
    files = list(zip(result["operation"], result["rows"], result["rows_removed"], result["rewritten"]))
    # The first file keeps one row, and the second is removed whole.
    assert sorted((rows, removed) for op, rows, removed, _ in files if op == "DELETE") == [(2, 1), (2, 2)]
    assert sorted((rows, rewritten) for op, rows, _, rewritten in files if op == "ADD") == [(1, False), (1, True)]


@pytest.mark.parametrize(
    "new_rows",
    [
        pytest.param([(7, "eu", 70)], id="a key outside the key set"),
        pytest.param([(None, "eu", 70)], id="a null key"),
    ],
)
def test_a_new_row_whose_key_is_not_in_the_key_set_fails_and_commits_nothing(
    local_catalog: Catalog, new_rows: list[Row]
) -> None:
    table = _table(local_catalog, UNPARTITIONED_PARTITION_SPEC, SEED)
    before = (snapshot_count(table), _rows(table))

    with pytest.raises(DaftValueError, match="must have its key"):
        _frame([(2, "eu", 21), *new_rows]).write_iceberg(table, mode="overwrite", overwrite_keys=_keys([2]))

    assert (snapshot_count(table), _rows(table)) == before


def test_an_overwrite_that_changes_nothing_commits_nothing(local_catalog: Catalog) -> None:
    table = _table(local_catalog, UNPARTITIONED_PARTITION_SPEC, SEED)
    before = (snapshot_count(table), _rows(table))

    result = _frame([]).write_iceberg(table, mode="overwrite", overwrite_keys=_keys([]))

    assert result.count_rows() == 0
    assert (snapshot_count(table), _rows(table)) == before


@pytest.mark.parametrize(
    ("dtype", "keys"),
    [
        pytest.param(pa.int32(), [2], id="a narrower integer"),
        pytest.param(pa.float64(), [2.0, 2.5], id="a float, of which a fraction matches nothing"),
    ],
)
def test_a_key_of_another_type_in_the_same_family_compares_by_value(
    local_catalog: Catalog, dtype: pa.DataType, keys: list[float]
) -> None:
    table = _table(local_catalog, UNPARTITIONED_PARTITION_SPEC, SEED)

    _frame([(2, "eu", 21)]).write_iceberg(
        table, mode="overwrite", overwrite_keys=daft.from_arrow(pa.table({"id": pa.array(keys, type=dtype)}))
    )

    assert _rows(table) == _sorted(
        [(1, "eu", 10), (2, "eu", 21), (3, "us", 30), (4, "us", 40), (5, "eu", 50), (None, "us", 0)]
    )


def test_a_long_text_key_is_pruned_and_matched_through_truncated_bounds(
    local_catalog: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema = Schema(NestedField(1, "id", StringType(), required=False), NestedField(2, "v", LongType(), required=False))
    table = local_catalog.create_table("default.long_keys", schema=schema)
    names = [f"customer-{i:04d}-with-a-long-identifier" for i in range(30)]
    for start in range(0, 30, 10):
        daft.from_pydict({"id": names[start : start + 10], "v": list(range(start, start + 10))}).write_iceberg(table)
    counted = _counted_files(monkeypatch)

    daft.from_pydict({"id": [names[12]], "v": [-1]}).write_iceberg(
        table, mode="overwrite", overwrite_keys=daft.from_pydict({"id": [names[12]]})
    )

    rows = daft.read_iceberg(table).to_pydict()
    assert dict(zip(rows["id"], rows["v"]))[names[12]] == -1
    assert len(rows["id"]) == 30
    # Bounds truncated to 16 characters still separate the three files.
    assert len(counted) == 1


def test_bucketing_by_the_key_prunes_where_bounds_cannot(
    local_catalog: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every append holds keys across the whole range, so every file's bounds enclose
    # every key: as random keys do. Only the bucket a key hashes to rules files out.
    appends = [[(i, "eu", run) for i in range(run, 1000, 4)] for run in range(4)]
    table = _table(local_catalog, SPECS["bucketed by the key"], appends)
    key = 501
    files = list(table.scan().plan_files())
    in_its_bucket = [task.file.file_path for task in files if any(r[0] == key for r in _rows_of(task.file.file_path))]
    counted = _counted_files(monkeypatch)

    _frame([]).write_iceberg(table, mode="overwrite", overwrite_keys=_keys([key]))

    assert len(in_its_bucket) == 1 and len(files) > 4
    assert sorted(counted) == sorted(
        task.file.file_path for task in files if task.file.partition[0] == _bucket_of(files, in_its_bucket[0])
    )
    assert all(row[0] != key for row in _rows(table))


def _rows_of(path: str) -> list[tuple[int, str, int]]:
    data = daft.read_parquet(path).to_pydict()
    return list(zip(data["id"], data["region"], data["v"]))


def _bucket_of(files: list[FileScanTask], path: str) -> int:
    return next(task.file.partition[0] for task in files if task.file.file_path == path)


def test_a_row_level_delete_outside_the_keys_still_applies_to_the_kept_rows(local_catalog: Catalog) -> None:
    table = _table(local_catalog, UNPARTITIONED_PARTITION_SPEC, [[(1, "eu", 10), (2, "eu", 20), (4, "eu", 40)]])
    commit_equality_delete_rows(table, pa.table({"id": pa.array([1], type=pa.int64())}), ["id"], bounds=True)

    _frame([]).write_iceberg(table, mode="overwrite", overwrite_keys=_keys([4]))

    assert _rows(table) == [(2, "eu", 20)]


def test_running_the_same_overwrite_twice_leaves_the_same_rows(local_catalog: Catalog) -> None:
    table = _table(local_catalog, UNPARTITIONED_PARTITION_SPEC, SEED)

    def run() -> None:
        _frame([(4, "us", 41)]).write_iceberg(table, mode="overwrite", overwrite_keys=_keys([3, 4]))

    run()
    once = _rows(table)
    run()

    assert _rows(table) == once


def _inject_before_commit(monkeypatch: pytest.MonkeyPatch, action: Callable[[Table], None], table: Table) -> None:
    """Run ``action`` once, after the overwrite is planned and written and before it commits."""
    from daft.io.iceberg import _overwrite

    real_commit = _overwrite._commit

    def commit_after_action(*args: object, **kwargs: object) -> None:
        action(table)
        monkeypatch.setattr(_overwrite, "_commit", real_commit)
        real_commit(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_overwrite, "_commit", commit_after_action)


def _append(ids: list[int]) -> Callable[[Table], None]:
    def append(table: Table) -> None:
        table.refresh()
        table.append(pa.table({"id": pa.array(ids, type=pa.int64()), "region": ["x"] * len(ids), "v": [0] * len(ids)}))

    return append


@pytest.mark.parametrize(
    ("foreign", "conflicts"),
    [
        pytest.param(_append([10]), True, id="a file holding a key"),
        pytest.param(_append([5]), False, id="a file between the keys"),
        pytest.param(_append([2, 9]), False, id="bounds interleaved with the keys but enclosing none"),
        pytest.param(_append([0, 20]), True, id="bounds enclosing a key"),
    ],
)
def test_a_concurrent_commit_conflicts_only_when_a_file_it_added_may_hold_a_key(
    local_catalog: Catalog, monkeypatch: pytest.MonkeyPatch, foreign: Callable[[Table], None], conflicts: bool
) -> None:
    table = _table(local_catalog, UNPARTITIONED_PARTITION_SPEC, [[(1, "eu", 10)], [(10, "eu", 100)]])
    _inject_before_commit(monkeypatch, foreign, table)

    def overwrite() -> None:
        _frame([(1, "eu", 11)]).write_iceberg(table, mode="overwrite", overwrite_keys=_keys([1, 10]))

    if conflicts:
        with pytest.raises(WriteConflict):
            overwrite()
        assert (1, "eu", 11) not in _rows(table)
    else:
        overwrite()
        assert (1, "eu", 11) in _rows(table)


def _moments_table(catalog: Catalog) -> Table:
    from pyiceberg.types import TimestampType, TimestamptzType

    schema = Schema(
        NestedField(1, "wall", TimestampType(), required=False),
        NestedField(2, "instant", TimestamptzType(), required=False),
        NestedField(3, "v", LongType(), required=False),
    )
    table = catalog.create_table("default.moments", schema=schema)
    noon = datetime.datetime(2026, 10, 13, 12, 0)
    daft.from_arrow(
        pa.table(
            {
                "wall": pa.array([noon, noon.replace(hour=13)], type=pa.timestamp("us")),
                "instant": pa.array(
                    [noon.replace(tzinfo=datetime.timezone.utc), noon.replace(hour=13, tzinfo=datetime.timezone.utc)],
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "v": [1, 2],
            }
        )
    ).write_iceberg(table)
    return table


@pytest.mark.parametrize(
    ("column", "key_type"),
    [
        pytest.param("wall", pa.timestamp("us", tz="UTC"), id="a zoned key against a zoneless column"),
        pytest.param("instant", pa.timestamp("us"), id="a zoneless key against a zoned column"),
    ],
)
def test_a_timestamp_key_is_refused_against_a_column_of_other_zone_awareness(
    local_catalog: Catalog, column: str, key_type: pa.DataType
) -> None:
    table = _moments_table(local_catalog)
    before = snapshot_count(table)
    noon = datetime.datetime(2026, 10, 13, 12, 0, tzinfo=datetime.timezone.utc if key_type.tz else None)
    keys = daft.from_arrow(pa.table({column: pa.array([noon], type=key_type)}))

    with pytest.raises(ValueError, match="does not compare"):
        daft.read_iceberg(table).limit(0).write_iceberg(table, mode="overwrite", overwrite_keys=keys)

    assert snapshot_count(table) == before


def test_a_zoned_key_matches_the_same_instant_in_another_zone(local_catalog: Catalog) -> None:
    table = _moments_table(local_catalog)
    # 08:00 in New York, in October, is 12:00 UTC.
    new_york = datetime.timezone(datetime.timedelta(hours=-4))
    keys = daft.from_arrow(
        pa.table(
            {
                "instant": pa.array(
                    [datetime.datetime(2026, 10, 13, 8, 0, tzinfo=new_york)],
                    type=pa.timestamp("us", tz="America/New_York"),
                )
            }
        )
    )

    daft.read_iceberg(table).limit(0).write_iceberg(table, mode="overwrite", overwrite_keys=keys)

    assert daft.read_iceberg(table).to_pydict()["v"] == [2]


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        pytest.param(
            {"overwrite_keys": _keys([1]), "overwrite_filter": col("id") > 0},
            ValueError,
            "cannot be combined",
            id="with a filter",
        ),
        pytest.param(
            {"overwrite_keys": _keys([1]), "mode": "append"}, ValueError, "requires mode", id="in append mode"
        ),
        pytest.param(
            {"overwrite_keys": daft.from_pydict({"nope": [1]})},
            ValueError,
            "not a column of the table",
            id="unknown key",
        ),
        pytest.param(
            {"overwrite_keys": daft.from_pydict({"id": ["1"]})},
            ValueError,
            "does not compare",
            id="text against a number",
        ),
    ],
)
def test_an_ill_formed_key_overwrite_is_refused(
    local_catalog: Catalog, kwargs: dict[str, object], error: type[Exception], match: str
) -> None:
    table = _table(local_catalog, UNPARTITIONED_PARTITION_SPEC, SEED)
    options: dict[str, object] = {"mode": "overwrite", **kwargs}

    with pytest.raises(error, match=match):
        _frame([(1, "eu", 11)]).write_iceberg(table, **options)  # type: ignore[arg-type]
