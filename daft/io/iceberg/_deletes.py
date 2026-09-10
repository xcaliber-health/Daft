"""Row-level delete planning and application shared by reads and rewrites.

Position deletes are applied inside the native reader per data file. Equality
deletes are applied here: every delete file's keys are read once, each data
file carries its own data sequence number and partition scope, and a row is
dropped when a newer delete in scope matches it.
"""

from __future__ import annotations

import datetime
import json
import uuid as _uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import pyarrow as pa
    from pyiceberg.io import FileIO
    from pyiceberg.manifest import DataFile, ManifestEntry
    from pyiceberg.schema import Schema
    from pyiceberg.table import DataScan, FileScanTask
    from pyiceberg.table import Table as PyIcebergTable
    from pyiceberg.typedef import Record

    from daft.daft import IOConfig
    from daft.dataframe import DataFrame
    from daft.datatype import DataType
    from daft.expressions import Expression
    from daft.io.iceberg.iceberg_scan import SchemaSource

#: One value of a partition record: a primitive of the table format, or null.
PartitionValue: TypeAlias = (
    str | int | float | bool | bytes | datetime.date | datetime.datetime | _uuid.UUID | Decimal | None
)

_SOURCE_COL = "__daft_equality_source__"
_SEQUENCE_COL = "__daft_data_sequence__"
_SCOPE_COL = "__daft_delete_scope__"
_LATEST_COL = "__daft_equality_latest__"


@dataclass(frozen=True)
class EqualityDelete:
    """An equality delete file and the scope within which it applies."""

    data_file: DataFile
    sequence_number: int
    spec_id: int
    partition_key: str
    is_global: bool

    @property
    def path(self) -> str:
        return str(self.data_file.file_path)

    @property
    def scope(self) -> str | None:
        """Return the partition the delete is confined to, or ``None`` for every partition."""
        return None if self.is_global else f"{self.spec_id}:{self.partition_key}"

    def applies_to(self, *, sequence_number: int, spec_id: int, partition_key: str) -> bool:
        """Return whether the delete covers a data file with these coordinates.

        An equality delete applies to strictly older data files in its
        partition, or in every partition when written under an unpartitioned
        spec.
        """
        if sequence_number >= self.sequence_number:
            return False
        return self.is_global or (spec_id == self.spec_id and partition_key == self.partition_key)


@dataclass(frozen=True)
class ScanPlan:
    """The data files a scan selected, with the delete files that apply to each."""

    tasks: dict[str, FileScanTask]
    equality_by_path: dict[str, tuple[EqualityDelete, ...]]
    sequence_by_path: dict[str, int]
    scope_by_path: dict[str, str]

    @property
    def has_equality_deletes(self) -> bool:
        return bool(self.equality_by_path)


def plan_files(table: PyIcebergTable, scan: DataScan) -> ScanPlan:
    """Plan the scan's data files with their position and equality deletes."""
    from itertools import chain

    from pyiceberg.expressions import AlwaysTrue
    from pyiceberg.manifest import DataFileContent
    from pyiceberg.table import FileScanTask
    from pyiceberg.table.delete_file_index import DeleteFileIndex
    from pyiceberg.table.metadata import INITIAL_SEQUENCE_NUMBER

    specs = table.specs()
    data_entries: list[ManifestEntry] = []
    position_deletes = DeleteFileIndex()
    equality_deletes: list[EqualityDelete] = []
    for entry in chain.from_iterable(scan.scan_plan_helper()):
        data_file = entry.data_file
        sequence_number = int(entry.sequence_number) if entry.sequence_number is not None else INITIAL_SEQUENCE_NUMBER
        if data_file.content == DataFileContent.DATA:
            data_entries.append(entry)
        elif data_file.content == DataFileContent.POSITION_DELETES:
            position_deletes.add_delete_file(entry, partition_key=data_file.partition)
        elif data_file.content == DataFileContent.EQUALITY_DELETES:
            spec_id = int(data_file.spec_id)
            equality_deletes.append(
                EqualityDelete(
                    data_file=data_file,
                    sequence_number=sequence_number,
                    spec_id=spec_id,
                    partition_key=stable_partition_key(data_file.partition),
                    is_global=specs[spec_id].is_unpartitioned(),
                )
            )

    tasks: dict[str, FileScanTask] = {}
    equality_by_path: dict[str, tuple[EqualityDelete, ...]] = {}
    sequence_by_path: dict[str, int] = {}
    scope_by_path: dict[str, str] = {}
    for entry in data_entries:
        data_file = entry.data_file
        path = str(data_file.file_path)
        sequence_number = int(entry.sequence_number) if entry.sequence_number is not None else INITIAL_SEQUENCE_NUMBER
        spec_id = int(data_file.spec_id)
        partition_key = stable_partition_key(data_file.partition)
        tasks[path] = FileScanTask(
            data_file,
            delete_files=position_deletes.for_data_file(sequence_number, data_file, partition_key=data_file.partition),
            residual=AlwaysTrue(),
        )
        sequence_by_path[path] = sequence_number
        scope_by_path[path] = f"{spec_id}:{partition_key}"
        applicable = tuple(
            delete
            for delete in equality_deletes
            if delete.applies_to(sequence_number=sequence_number, spec_id=spec_id, partition_key=partition_key)
        )
        if applicable:
            equality_by_path[path] = applicable
    return ScanPlan(
        tasks=tasks,
        equality_by_path=equality_by_path,
        sequence_by_path=sequence_by_path,
        scope_by_path=scope_by_path,
    )


def stable_partition_key(record: Record | None) -> str:
    """Return a stable string key for a partition record, whose values are positional."""
    if record is None:
        return "[]"
    try:
        values = [_json_safe(record[i]) for i in range(len(record))]
        return json.dumps(values, default=str)
    except TypeError:
        return json.dumps(str(record))


def _json_safe(v: PartitionValue) -> str | int | float | bool | None:
    """Return ``v`` if it serialises as JSON, otherwise its string form."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def snapshot_has_equality_deletes(table: PyIcebergTable, snapshot_id: int | None) -> bool:
    """Return whether the snapshot's delete manifests hold an equality delete.

    Only delete manifests are opened, so a table without deletes costs nothing.
    """
    from pyiceberg.manifest import DataFileContent, ManifestContent

    snapshot = table.current_snapshot() if snapshot_id is None else table.snapshot_by_id(snapshot_id)
    if snapshot is None or table.metadata.format_version < 2:
        return False
    for manifest in snapshot.manifests(table.io):
        if manifest.content != ManifestContent.DELETES:
            continue
        for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=True):
            if entry.data_file.content == DataFileContent.EQUALITY_DELETES:
                return True
    return False


def read_with_deletes(
    *,
    table: PyIcebergTable,
    plan: ScanPlan,
    paths: list[str],
    snapshot_id: int | None,
    io_config: IOConfig,
    schema_source: SchemaSource = "current",
    ignore_corrupt_files: bool = False,
) -> DataFrame:
    """Build a lazy frame over exactly these data files, with their deletes applied.

    Files sharing a sequence number and partition are read together.
    """
    from daft import runners
    from daft.daft import ScanOperatorHandle, StorageConfig
    from daft.dataframe import DataFrame
    from daft.expressions import lit
    from daft.io.iceberg.iceberg_scan import IcebergFileGroupScanOperator
    from daft.logical.builder import LogicalPlanBuilder

    multithreaded_io = runners.get_or_create_runner().name != "ray"
    storage_config = StorageConfig(multithreaded_io, io_config)

    def _frame(group: list[str]) -> DataFrame:
        tasks = [plan.tasks[path] for path in group]
        operator = IcebergFileGroupScanOperator(
            table,
            snapshot_id=snapshot_id,
            storage_config=storage_config,
            tasks=tasks,
            schema_source=schema_source,
            ignore_corrupt_files=ignore_corrupt_files,
        )
        handle = ScanOperatorHandle.from_python_scan_operator(operator)
        return DataFrame(LogicalPlanBuilder.from_tabular_scan(scan_operator=handle))

    deletes: dict[str, EqualityDelete] = {}
    for path in paths:
        deletes.update((delete.path, delete) for delete in plan.equality_by_path.get(path, ()))
    if not deletes:
        return _frame(paths)

    # A file no delete covers is placed at the newest delete's sequence number.
    untouched = max(delete.sequence_number for delete in deletes.values())
    groups: dict[tuple[int, str], list[str]] = {}
    for path in paths:
        sequence = plan.sequence_by_path[path] if path in plan.equality_by_path else untouched
        groups.setdefault((sequence, plan.scope_by_path[path]), []).append(path)
    frames = [
        _frame(group).with_column(_SEQUENCE_COL, lit(sequence)).with_column(_SCOPE_COL, lit(scope))
        for (sequence, scope), group in groups.items()
    ]
    df = frames[0]
    for other in frames[1:]:
        df = df.concat(other)
    data_types = {field.name: field.dtype for field in df.schema()}
    # Names come from the schema the frame was read under.
    schema = (
        table.schema()
        if schema_source == "current" or snapshot_id is None
        else table.scan(snapshot_id=snapshot_id).projection()
    )
    for columns, keys in read_equality_keys(schema, table.io, list(deletes.values())).items():
        df = apply_equality_deletes(df, keys, deletes, list(columns), data_types)
    return df.exclude(_SEQUENCE_COL, _SCOPE_COL)


def read_equality_keys(schema: Schema, io: FileIO, deletes: list[EqualityDelete]) -> dict[tuple[str, ...], pa.Table]:
    """Read every delete file's equality columns once, grouped by the columns they match on.

    Columns are matched by field id and each row is tagged with its source file.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    tables: dict[tuple[str, ...], list[pa.Table]] = {}
    for delete in deletes:
        field_ids = [int(i) for i in delete.data_file.equality_ids or ()]
        if not field_ids:
            raise ValueError(f"equality delete file {delete.path!r} names no equality columns")
        names: list[str] = []
        for field_id in field_ids:
            name = schema.find_column_name(field_id)
            if name is None or "." in name:
                raise ValueError(
                    f"equality delete file {delete.path!r} matches on field {field_id}, "
                    "which is not a top-level column of the table"
                )
            names.append(name)
        with io.new_input(delete.path).open() as handle:
            parquet_file = pq.ParquetFile(handle)
            by_field_id = {_arrow_field_id(field): field.name for field in parquet_file.schema_arrow}
            missing = [field_id for field_id in field_ids if field_id not in by_field_id]
            if missing:
                raise ValueError(
                    f"equality delete file {delete.path!r} lacks equality columns with field ids {missing}"
                )
            rows = parquet_file.read(columns=[by_field_id[field_id] for field_id in field_ids])
        rows = rows.rename_columns(names).append_column(
            _SOURCE_COL, pa.array([delete.path] * rows.num_rows, type=pa.string())
        )
        tables.setdefault(tuple(names), []).append(rows)
    return {columns: pa.concat_tables(parts, promote_options="permissive") for columns, parts in tables.items()}


def _arrow_field_id(field: pa.Field) -> int | None:
    """Return the field id an arrow field carries in its metadata, if any."""
    metadata = field.metadata or {}
    raw = metadata.get(b"PARQUET:field_id")
    return int(raw) if raw is not None else None


def apply_equality_deletes(
    df: DataFrame,
    keys: pa.Table,
    deletes: dict[str, EqualityDelete],
    columns: list[str],
    data_types: dict[str, DataType],
) -> DataFrame:
    """Drop the rows of ``df`` that a newer delete in scope matches on ``columns``.

    Keys are split by which columns are null, since a join never matches nulls.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    sources = keys[_SOURCE_COL].to_pylist()
    keys = keys.drop([_SOURCE_COL])
    keys = keys.append_column(_LATEST_COL, pa.array([deletes[s].sequence_number for s in sources], type=pa.int64()))
    keys = keys.append_column(_SCOPE_COL, pa.array([deletes[s].scope for s in sources], type=pa.string()))
    is_global = pc.is_null(keys[_SCOPE_COL])
    df = _apply_keys(df, keys.filter(is_global).drop([_SCOPE_COL]), columns, data_types, scoped=False)
    return _apply_keys(df, keys.filter(pc.invert(is_global)), columns, data_types, scoped=True)


def _apply_keys(
    df: DataFrame,
    keys: pa.Table,
    columns: list[str],
    data_types: dict[str, DataType],
    *,
    scoped: bool,
) -> DataFrame:
    """Apply one scope class of keys, one join per pattern of null columns."""
    import pyarrow as pa
    import pyarrow.compute as pc

    import daft
    from daft.expressions import col, lit

    if keys.num_rows == 0:
        return df
    latest_col = f"{_LATEST_COL}_max"
    mask = pa.array([0] * keys.num_rows, type=pa.int64())
    for index, name in enumerate(columns):
        mask = pc.add(mask, pc.multiply(pc.cast(pc.is_null(keys[name]), pa.int64()), 1 << index))
    for pattern in sorted(int(value) for value in pc.unique(mask).to_pylist()):
        null_columns = [name for index, name in enumerate(columns) if pattern & (1 << index)]
        value_columns = [name for name in columns if name not in null_columns]
        join_columns: list[Expression | str] = [*value_columns, *([_SCOPE_COL] if scoped else [])]
        rows = keys.filter(pc.equal(mask, pattern))
        all_null: Expression | None = None
        for name in null_columns:
            all_null = col(name).is_null() if all_null is None else all_null & col(name).is_null()
        if not join_columns:
            assert all_null is not None
            latest = lit(pc.max(rows[_LATEST_COL]).as_py())
            df = df.filter(~(all_null & (latest > col(_SEQUENCE_COL))))
            continue
        grouped = rows.group_by(join_columns).aggregate([(_LATEST_COL, "max")])
        subset = daft.from_arrow(grouped.select([*join_columns, latest_col]))
        subset = subset.select(
            *[subset[name].cast(data_types[name]) for name in value_columns],
            *([subset[_SCOPE_COL]] if scoped else []),
            subset[latest_col],
        )
        matched = col(latest_col).not_null() & (col(latest_col) > col(_SEQUENCE_COL))
        if all_null is not None:
            matched = matched & all_null
        df = df.join(subset, on=join_columns, how="left").filter(~matched).exclude(latest_col)
    return df
