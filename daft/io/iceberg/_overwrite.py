"""Replace the rows of a table that a filter selects, and report every write by file.

An overwrite by filter removes each existing row on which the filter is true and
writes the new rows, in one snapshot. A row on which the filter is false or null
is kept. Files every live row of which matches are removed whole; files only
some of whose rows match are rewritten without them. Every new row must satisfy
the filter, or the write fails before anything is committed.

The commit is checked against the commits made since the overwrite was planned,
under the table's ``write.delete.isolation-level``: ``serializable`` (the
default) also refuses a concurrent commit that added rows the filter may select.
"""

from __future__ import annotations

import logging
import uuid as _uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, TypeAlias

from daft.io.iceberg._common import begin_transaction, commit_with_retry

if TYPE_CHECKING:
    from collections.abc import Callable

    import pyarrow as pa
    from pyiceberg.expressions import BooleanExpression
    from pyiceberg.io import FileIO
    from pyiceberg.manifest import DataFile
    from pyiceberg.table import Table as PyIcebergTable
    from pyiceberg.table import Transaction
    from pyiceberg.table.snapshots import Operation, Snapshot
    from pyiceberg.types import IcebergType

    from daft.daft import IOConfig
    from daft.dataframe import DataFrame
    from daft.datatype import DataType
    from daft.expressions import Expression
    from daft.io.iceberg._deletes import PartitionValue, ScanPlan
    from daft.schema import Schema

logger = logging.getLogger(__name__)

#: Table property naming how an overwrite is checked against concurrent commits.
ISOLATION_LEVEL_KEY = "write.delete.isolation-level"
ISOLATION_SERIALIZABLE = "serializable"
ISOLATION_SNAPSHOT = "snapshot"
_ISOLATION_LEVELS = (ISOLATION_SERIALIZABLE, ISOLATION_SNAPSHOT)

_FILE_COL = "__daft_overwrite_file__"
_MATCHED_COL = "__daft_overwrite_matched__"
_LIVE_COL = "__daft_overwrite_live__"


class WriteConflict(RuntimeError):
    """Raised when a commit made since a write was planned overlaps what the write replaces.

    Nothing of the write is committed, so it may be run again.
    """


@dataclass(frozen=True)
class WrittenFile:
    """A data file a write added.

    Attributes:
    ----------
    data_file
        The file's metadata.
    rewritten
        Whether the file holds rows kept from a file the write replaced, rather
        than the write's new rows.
    """

    data_file: DataFile
    rewritten: bool


@dataclass(frozen=True)
class RemovedFile:
    """A data file a write removed.

    Attributes:
    ----------
    data_file
        The file's metadata.
    rows_removed
        How many of the file's live rows the write removed.
    """

    data_file: DataFile
    rows_removed: int


def write_result(
    table: PyIcebergTable, added: list[WrittenFile], removed: list[RemovedFile], *, like: DataFrame | None = None
) -> DataFrame:
    """Describe a write by file: one ``ADD`` row per file added, one ``DELETE`` row per file removed.

    Parameters
    ----------
    table
        The table written to; its partition fields name the partition values.
    added, removed
        The files the write added and removed.
    like
        A frame whose execution metadata the result carries.

    Returns:
    -------
    DataFrame
        Columns ``operation``, ``rows``, ``file_size``, ``file_name``, the
        ``partitioning`` values when the table is partitioned, ``rows_removed``
        (on ``DELETE`` rows) and ``rewritten`` (on ``ADD`` rows).
    """
    import pyarrow as pa

    from daft import from_pydict

    schema = table.schema()
    partitioning: dict[str, list[PartitionValue]] = {
        schema.find_field(field.source_id).name: [] for field in table.spec().fields
    }
    files = [(file.data_file, "ADD", None, file.rewritten) for file in added] + [
        (file.data_file, "DELETE", file.rows_removed, None) for file in removed
    ]
    for data_file, _, _, _ in files:
        for name, values in partitioning.items():
            values.append(getattr(data_file.partition, name, None))

    columns: dict[str, pa.Array] = {
        "operation": pa.array([operation for _, operation, _, _ in files], type=pa.string()),
        "rows": pa.array([data_file.record_count for data_file, _, _, _ in files], type=pa.int64()),
        "file_size": pa.array([data_file.file_size_in_bytes for data_file, _, _, _ in files], type=pa.int64()),
        "file_name": pa.array([data_file.file_path for data_file, _, _, _ in files], type=pa.string()),
    }
    if partitioning:
        columns["partitioning"] = pa.StructArray.from_arrays(
            [pa.array(values) for values in partitioning.values()], names=list(partitioning)
        )
    columns["rows_removed"] = pa.array([rows_removed for _, _, rows_removed, _ in files], type=pa.int64())
    columns["rewritten"] = pa.array([rewritten for _, _, _, rewritten in files], type=pa.bool_())

    result = from_pydict(columns)
    if like is not None:
        result._metadata = like._metadata
    return result


def removed_whole(
    table: PyIcebergTable, plan: ScanPlan, snapshot_id: int | None, io_config: IOConfig
) -> list[RemovedFile]:
    """Describe the removal of every file ``plan`` selected, each with its live rows.

    A file no row-level delete covers has as many live rows as it records;
    the others are read to count them.
    """
    covered = [path for path, task in plan.tasks.items() if task.delete_files or path in plan.equality_by_path]
    counts = _count_rows(table, plan, covered, snapshot_id, io_config, scope=None)
    removed: list[RemovedFile] = []
    for path, task in plan.tasks.items():
        if path in covered:
            # A file whose rows were all deleted reads as no rows.
            live = counts[path][0] if path in counts else 0
        else:
            live = int(task.file.record_count)
        removed.append(RemovedFile(task.file, live))
    return removed


def overwrite_by_filter(
    df: DataFrame,
    table: PyIcebergTable,
    overwrite_filter: Expression,
    io_config: IOConfig,
    snapshot_properties: dict[str, str],
) -> DataFrame:
    """Replace the rows of ``table`` on which ``overwrite_filter`` is true with the rows of ``df``.

    Parameters
    ----------
    df
        The new rows; every one must satisfy the filter.
    table
        The table to write; its main branch is replaced in.
    overwrite_filter
        A boolean expression over the table's columns, evaluated on the
        existing rows and on the new ones.
    io_config
        Storage access for reading and writing data files.
    snapshot_properties
        Properties set on the snapshot committed.

    Returns:
    -------
    DataFrame
        The files added and removed, as :func:`write_result` describes them.

    Raises:
    ------
    ValueError
        When the filter does not resolve to a boolean over both the table's
        columns and the new rows' columns, when the isolation level is not
        recognized, or when a new row does not satisfy the filter.
    WriteConflict
        When a commit made since the overwrite was planned overlaps it.
    CommitRetryExhausted
        When the commit cannot land within the table's retry budget.
    """
    return _overwrite(df, table, _FilterScope(table, overwrite_filter, df.schema()), io_config, snapshot_properties)


def overwrite_by_keys(
    df: DataFrame,
    table: PyIcebergTable,
    keys: DataFrame,
    io_config: IOConfig,
    snapshot_properties: dict[str, str],
) -> DataFrame:
    """Replace the rows of ``table`` whose key is in the key set ``keys`` with the rows of ``df``.

    A row's key is its values on the key columns, which are ``keys``' columns,
    matched to the table's columns by name, all together. A row with a null in
    any key column never matches and is kept. A key of ``keys`` with no new row
    still loses its existing rows.

    Parameters
    ----------
    df
        The new rows; every one must have its key in ``keys``.
    table
        The table to write; its main branch is replaced in.
    keys
        The key set K. Its distinct rows are read once and held in memory, in
        the calling process, without spilling.
    io_config
        Storage access for reading and writing data files.
    snapshot_properties
        Properties set on the snapshot committed.

    Returns:
    -------
    DataFrame
        The files added and removed, as :func:`write_result` describes them.

    Raises:
    ------
    ValueError
        When a key column is not a column of the table and of ``df``, when a key
        column's type does not compare with the table's, when the isolation
        level is not recognized, or when a new row's key is not in ``keys``.
    WriteConflict
        When a commit made since the overwrite was planned overlaps it.
    CommitRetryExhausted
        When the commit cannot land within the table's retry budget.
    """
    return _overwrite(df, table, _KeyScope(table, keys, df.schema()), io_config, snapshot_properties)


def _overwrite(
    df: DataFrame,
    table: PyIcebergTable,
    scope: _Scope,
    io_config: IOConfig,
    snapshot_properties: dict[str, str],
) -> DataFrame:
    """Replace the rows of ``table`` that ``scope`` selects with the rows of ``df``, in one snapshot."""
    from daft import col
    from daft.io.iceberg._deletes import plan_files, read_with_deletes

    isolation = _isolation_level(table)
    start = table.current_snapshot()
    start_id = int(start.snapshot_id) if start is not None else None
    # Planned without the scope: a plan narrowed by it omits row-level deletes
    # whose bounds miss it, and those still apply to the rows a rewrite keeps.
    plan = plan_files(table, table.scan(snapshot_id=start_id))
    candidates = [path for path, task in plan.tasks.items() if scope.may_hold(task.file)]
    removed, partial = _select_files(table, plan, candidates, scope, start_id, io_config)

    new_files, write_df = _write(scope.checked(df), table, io_config, require_matching_columns=False)
    survivor_files: list[DataFile] = []
    try:
        if partial:
            survivors = (
                scope.matched(
                    read_with_deletes(table=table, plan=plan, paths=partial, snapshot_id=start_id, io_config=io_config)
                )
                .where(~col(_MATCHED_COL))
                .select(*[field.name for field in table.schema().fields])
            )
            survivor_files, _ = _write(survivors, table, io_config, require_matching_columns=True)
        added = [WrittenFile(f, rewritten=False) for f in new_files] + [
            WrittenFile(f, rewritten=True) for f in survivor_files
        ]
        # An overwrite that removes no row and writes no row changes nothing, so commits nothing.
        if added or removed:
            _commit(
                table,
                start_id=start_id,
                added=added,
                removed=removed,
                may_hold=scope.may_hold if isolation == ISOLATION_SERIALIZABLE else None,
                snapshot_properties=snapshot_properties,
            )
    except BaseException:
        _discard(table, [*new_files, *survivor_files])
        raise
    return write_result(table, added, removed, like=write_df)


class _Scope(ABC):
    """The existing rows an overwrite replaces, and the new rows it may write."""

    @abstractmethod
    def may_hold(self, data_file: DataFile) -> bool:
        """Whether a data file's metadata allows a row in scope; false only when it holds none."""

    def holds_only(self, data_file: DataFile) -> bool:
        """Whether a data file's metadata shows every row is in scope, so it is removed unread."""
        return False

    @abstractmethod
    def matched(self, frame: DataFrame) -> DataFrame:
        """``frame`` with a non-null boolean column saying whether each row is in scope."""

    @abstractmethod
    def checked(self, df: DataFrame) -> DataFrame:
        """``df`` failing its query on any row not in scope, with its columns unchanged."""


class _FilterScope(_Scope):
    """The rows on which a boolean filter is true."""

    def __init__(self, table: PyIcebergTable, overwrite_filter: Expression, written: Schema) -> None:
        from pyiceberg.expressions import AlwaysTrue
        from pyiceberg.expressions.visitors import _StrictMetricsEvaluator

        from daft.io.iceberg._expressions import convert_filter
        from daft.io.iceberg._metadata import convert_iceberg_schema

        _require_boolean(overwrite_filter, convert_iceberg_schema(table.schema()), "the table's columns")
        _require_boolean(overwrite_filter, written, "the written rows' columns")
        self._filter = overwrite_filter
        prune = _constant_predicate(overwrite_filter)
        if prune is None:
            prune = convert_filter(overwrite_filter._expr, table.schema()) or AlwaysTrue()
        self._may_hold = _may_hold_rows_of(table, prune)
        strict = _strict_predicate(overwrite_filter, table)
        self._must_match = (
            _StrictMetricsEvaluator(table.schema(), strict, case_sensitive=True).eval if strict is not None else None
        )

    def may_hold(self, data_file: DataFile) -> bool:
        return self._may_hold(data_file)

    def holds_only(self, data_file: DataFile) -> bool:
        return self._must_match is not None and bool(self._must_match(data_file))

    def matched(self, frame: DataFrame) -> DataFrame:
        return frame.with_column(_MATCHED_COL, self._filter.fill_null(False))

    def checked(self, df: DataFrame) -> DataFrame:
        return _checked_against(
            df, self._filter, f"a row written by an overwrite must satisfy its filter {self._filter}"
        )


#: A file's bound on a key column, decoded in the order K's values compare in.
BoundValue: TypeAlias = int | float | str | bytes | bool | Decimal

#: Keys tested at once against a file's bounds on the key columns after the first.
_BOUNDS_CHUNK = 65_536


class _KeyScope(_Scope):
    """The rows whose key, on every key column together, is a key of a key set K.

    K's distinct rows are read once and held in memory, in the calling process,
    without spilling: one copy, sorted on the key columns, cast to the table's
    types. A key with a null in any column, or that the table's column cannot
    hold, never matches and is dropped.

    A file may hold a key of K when its partition, on the fields drawn from key
    columns, is one K's keys fall in, and its bounds, on every key column
    together, enclose a key of K. A file without both bounds on a key column is
    not ruled out by that column. Testing a file costs ``log |K|`` binary search
    steps on the first key column, plus a test of the keys inside those bounds
    on the other key columns that stops at the first key inside every bound:
    ``files * log |K|`` when the first column's bounds separate the files, up to
    ``files * |K|`` when they do not and another column's do.
    """

    def __init__(self, table: PyIcebergTable, keys: DataFrame, written: Schema) -> None:
        from daft import col, from_arrow, lit
        from daft.io.iceberg._metadata import convert_iceberg_schema

        names = keys.column_names
        if not names:
            raise ValueError("overwrite_keys must have at least one key column")
        table_schema = convert_iceberg_schema(table.schema())
        for name in names:
            if name not in table_schema.column_names():
                raise ValueError(f"the key column {name!r} is not a column of the table")
            if name not in written.column_names():
                raise ValueError(f"the key column {name!r} is not a column of the written rows")
            for side, dtype in (("key set", keys.schema()[name].dtype), ("written rows", written[name].dtype)):
                if not _comparable(dtype, table_schema[name].dtype):
                    raise ValueError(
                        f"the key column {name!r} of the {side} is {dtype}, which does not compare with "
                        f"the table's {table_schema[name].dtype}"
                    )
        self._names: list[Expression | str] = list(names)

        # Cast to the table's types; a key the table's column cannot hold matches no row.
        held = keys.select(*names)
        kept = None
        for name in names:
            original = f"__daft_key_original_{name}__"
            target = table_schema[name].dtype
            held = held.with_column(original, col(name)).with_column(name, col(name).cast(target))
            representable = col(name).not_null() & (col(name).cast(keys.schema()[name].dtype) == col(original))
            kept = representable if kept is None else kept & representable
        assert kept is not None
        self._keys = held.where(kept).select(*names).distinct().sort(self._names).to_arrow()
        self._frame = from_arrow(self._keys).with_column(_MATCHED_COL, lit(True))
        self._bounds = [_KeyColumn.of(table, name, self._keys.column(name)) for name in names]
        self._partitions = _key_partitions(table, names, self._keys)

    def may_hold(self, data_file: DataFile) -> bool:
        if self._keys.num_rows == 0:
            return False
        tested = self._partitions.get(int(data_file.spec_id))
        if tested is not None:
            positions, allowed = tested
            if tuple(_canonical(data_file.partition[pos]) for pos in positions) not in allowed:
                return False
        return self._bounds_may_hold(data_file)

    def _bounds_may_hold(self, data_file: DataFile) -> bool:
        import pyarrow.compute as pc

        bounds = [column.bounds(data_file) for column in self._bounds]
        start, stop = 0, self._keys.num_rows
        first = bounds[0]
        if first is not None:
            start, stop = self._bounds[0].slice_within(*first)
            if start >= stop:
                return False
        rest = [
            (column.values, bound)
            for i, (column, bound) in enumerate(zip(self._bounds, bounds))
            if i > 0 and bound is not None and column.values is not None
        ]
        if not rest:
            return True
        for offset in range(start, stop, _BOUNDS_CHUNK):
            length = min(_BOUNDS_CHUNK, stop - offset)
            inside = None
            for column_values, (low, high) in rest:
                values = column_values.slice(offset, length)
                within = pc.and_(pc.greater_equal(values, low), pc.less_equal(values, high))
                inside = within if inside is None else pc.and_(inside, within)
            if inside is not None and pc.any(inside).as_py():
                return True
        return False

    def matched(self, frame: DataFrame) -> DataFrame:
        from daft import col

        return frame.join(self._frame, on=self._names, how="left").with_column(
            _MATCHED_COL, col(_MATCHED_COL).fill_null(False)
        )

    def checked(self, df: DataFrame) -> DataFrame:
        from daft import col

        columns = df.column_names
        marked = _checked_against(
            self.matched(df),
            col(_MATCHED_COL),
            f"a row written by an overwrite must have its key, on {self._names}, in the key set",
        )
        return marked.select(*columns)


@dataclass(frozen=True)
class _KeyColumn:
    """One key column of K, in the order its file bounds compare in, with its field's type."""

    field_id: int
    field_type: IcebergType
    #: K's values in comparison order, or ``None`` when the type's bounds are not compared.
    #: Kept in K's own chunks, so no copy of K is made.
    values: pa.ChunkedArray | None

    @classmethod
    def of(cls, table: PyIcebergTable, name: str, values: pa.ChunkedArray) -> _KeyColumn:
        field = table.schema().find_field(name)
        return cls(field.field_id, field.field_type, _comparison_order(field.field_type, values))

    def bounds(self, data_file: DataFile) -> tuple[BoundValue, BoundValue] | None:
        """The file's lower and upper bound on this column in comparison order, when it records both."""
        from pyiceberg.conversions import from_bytes

        if self.values is None:
            return None
        lower = (data_file.lower_bounds or {}).get(self.field_id)
        upper = (data_file.upper_bounds or {}).get(self.field_id)
        if lower is None or upper is None:
            return None
        return from_bytes(self.field_type, lower), from_bytes(self.field_type, upper)

    def slice_within(self, low: BoundValue, high: BoundValue) -> tuple[int, int]:
        """The range of positions of K's sorted values inside ``[low, high]``."""
        from bisect import bisect_left, bisect_right

        assert self.values is not None
        view = _SortedView(self.values)
        return bisect_left(view, low), bisect_right(view, high)


class _SortedView:
    """Sorted values read one at a time, so a binary search converts only the values it visits."""

    def __init__(self, values: pa.ChunkedArray) -> None:
        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, position: int) -> BoundValue:
        return self._values[position].as_py()


def _comparison_order(field_type: IcebergType, values: pa.ChunkedArray) -> pa.ChunkedArray | None:
    """K's values in the form the table's bounds decode to, or ``None`` when that type's bounds are not compared.

    Dates compare as days and times and timestamps as microseconds; strings in
    code-point order, which is the byte order a truncated bound keeps sound. A
    value already in that form is returned as it is, sharing K's buffers.
    """
    import pyarrow as pa
    from pyiceberg.types import (
        BinaryType,
        BooleanType,
        DateType,
        DecimalType,
        DoubleType,
        FixedType,
        FloatType,
        IntegerType,
        LongType,
        StringType,
        TimestampType,
        TimestamptzType,
        TimeType,
    )

    if isinstance(field_type, (IntegerType, LongType)):
        return values.cast(pa.int64())
    if isinstance(field_type, DateType):
        return values.cast(pa.date32()).cast(pa.int32())
    if isinstance(field_type, (TimestampType, TimestamptzType)):
        return values.cast(pa.timestamp("us", tz=values.type.tz)).cast(pa.int64())
    if isinstance(field_type, TimeType):
        return values.cast(pa.time64("us")).cast(pa.int64())
    if isinstance(field_type, (FloatType, DoubleType)):
        return values.cast(pa.float64())
    if isinstance(field_type, (StringType, BinaryType, FixedType, DecimalType, BooleanType)):
        return values
    return None


def _comparable(key: DataType, column: DataType) -> bool:
    """Whether a key of type ``key`` is compared with a column of type ``column``.

    Types compare within one family: numbers, text, bytes, dates, times,
    timestamps, or truth values. A key is never matched across families, as a
    cast from text to a number would.
    """
    family = _family(key)
    return key == column or (family is not None and family == _family(column))


def _family(dtype: DataType) -> str | None:
    """The family of types ``dtype`` compares within, or ``None`` when it compares only with itself."""
    if dtype.is_numeric():
        return "number"
    if dtype.is_string():
        return "text"
    if dtype.is_binary() or dtype.is_fixed_size_binary():
        return "bytes"
    if dtype.is_date():
        return "date"
    if dtype.is_time():
        return "time"
    if dtype.is_timestamp():
        return "timestamp"
    if dtype.is_boolean():
        return "truth"
    return None


def _key_partitions(
    table: PyIcebergTable, names: list[str], keys: pa.Table
) -> dict[int, tuple[tuple[int, ...], frozenset[tuple[PartitionValue, ...]]]]:
    """Per partition spec, the fields drawn from key columns and the partition tuples K's keys fall in.

    Each field is computed over K with the transforms the writer partitions by,
    so a key's bucket or truncated value is the one its row's file records, and
    the tuples are made distinct by the engine, so only as many reach Python as
    partitions K falls in. A spec with no field drawn from a key column is left out.
    """
    from pyiceberg.transforms import (
        BucketTransform,
        DayTransform,
        HourTransform,
        IdentityTransform,
        MonthTransform,
        TruncateTransform,
        YearTransform,
    )

    from daft.io.iceberg.iceberg_write import partition_field_to_expr

    supported = (
        IdentityTransform,
        YearTransform,
        MonthTransform,
        DayTransform,
        HourTransform,
        BucketTransform,
        TruncateTransform,
    )
    schema = table.schema()
    key_ids = {schema.find_field(name).field_id for name in names}
    from daft import from_arrow

    frame = from_arrow(keys)
    tested: dict[int, tuple[tuple[int, ...], frozenset[tuple[PartitionValue, ...]]]] = {}
    for spec_id, spec in table.specs().items():
        fields = [
            (position, field)
            for position, field in enumerate(spec.fields)
            if field.source_id in key_ids and isinstance(field.transform, supported)
        ]
        if not fields:
            continue
        # Made distinct by the engine: only the partitions K falls in reach Python, not every key.
        computed = (
            frame.select(
                *[
                    partition_field_to_expr(field, schema).alias(f"__daft_partition_{position}")
                    for position, field in fields
                ]
            )
            .distinct()
            .to_pydict()
        )
        columns = [computed[f"__daft_partition_{position}"] for position, _ in fields]
        allowed = frozenset(tuple(_canonical(value) for value in row) for row in zip(*columns))
        tested[int(spec_id)] = (tuple(position for position, _ in fields), allowed)
    return tested


def _canonical(value: PartitionValue) -> PartitionValue:
    """A partition value in the form the writer records it, so computed and recorded values compare."""
    import datetime

    from daft.io.iceberg.iceberg_write import to_partition_representation

    if isinstance(value, datetime.datetime) and value.tzinfo is not None:
        value = value.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return to_partition_representation(value)


def _require_boolean(expression: Expression, schema: Schema, over: str) -> None:
    """Refuse a filter that does not resolve to a boolean over ``schema``."""
    from daft.datatype import DataType

    try:
        field = expression._to_field(schema)
    except ValueError as exc:
        raise ValueError(f"the overwrite filter {expression} must refer only to {over}: {exc}") from exc
    if field.dtype != DataType.bool():
        raise ValueError(f"the overwrite filter {expression} must be boolean, got {field.dtype}")


def _isolation_level(table: PyIcebergTable) -> str:
    level = table.properties.get(ISOLATION_LEVEL_KEY, ISOLATION_SERIALIZABLE)
    if level not in _ISOLATION_LEVELS:
        raise ValueError(f"{ISOLATION_LEVEL_KEY} must be one of {_ISOLATION_LEVELS}, got {level!r}")
    return level


def _select_files(
    table: PyIcebergTable,
    plan: ScanPlan,
    candidates: list[str],
    scope: _Scope,
    snapshot_id: int | None,
    io_config: IOConfig,
) -> tuple[list[RemovedFile], list[str]]:
    """Return which of the ``candidates`` the scope removes, and the paths of those only partly removed.

    A file without row-level deletes whose metadata shows every row is in scope
    is removed without being read; the other candidates are read for the
    columns the scope needs, and kept whole when no row is in scope.
    """
    removed: list[RemovedFile] = []
    to_count: list[str] = []
    for path in candidates:
        task = plan.tasks[path]
        has_deletes = bool(task.delete_files) or path in plan.equality_by_path
        if not has_deletes and scope.holds_only(task.file):
            removed.append(RemovedFile(task.file, int(task.file.record_count)))
        else:
            to_count.append(path)

    partial: list[str] = []
    for path, (live, matched) in _count_rows(table, plan, to_count, snapshot_id, io_config, scope=scope).items():
        if matched == 0:
            continue
        removed.append(RemovedFile(plan.tasks[path].file, matched))
        if matched < live:
            partial.append(path)
    return removed, partial


def _constant_predicate(scope: Expression) -> BooleanExpression | None:
    """Return a constant filter as the table predicate it is, else ``None``.

    A null constant selects no row, as false does.
    """
    from pyiceberg.expressions import AlwaysFalse, AlwaysTrue

    if not scope.is_literal():
        return None
    return AlwaysTrue() if scope.as_py() is True else AlwaysFalse()


def _strict_predicate(scope: Expression, table: PyIcebergTable) -> BooleanExpression | None:
    """Return the filter as a table predicate when it translates exactly, else ``None``."""
    from pyiceberg.exceptions import ValidationError
    from pyiceberg.expressions import BooleanExpression
    from pyiceberg.expressions.visitors import bind

    from daft.io.iceberg._visitors import IcebergPredicateVisitor

    constant = _constant_predicate(scope)
    if constant is not None:
        return constant
    try:
        predicate = IcebergPredicateVisitor(table.schema()).visit(scope)
        if not isinstance(predicate, BooleanExpression):
            return None
        bind(table.schema(), predicate, case_sensitive=True)
    except (ValueError, ValidationError) as exc:
        logger.debug("overwrite filter %s is not an exact table predicate: %s", scope, exc)
        return None
    return predicate


def _count_rows(
    table: PyIcebergTable,
    plan: ScanPlan,
    paths: list[str],
    snapshot_id: int | None,
    io_config: IOConfig,
    *,
    scope: _Scope | None,
) -> dict[str, tuple[int, int]]:
    """Return, per file, its live rows and how many of them are in ``scope``.

    Only the columns the scope needs are read. Without a scope, every live row
    counts as in scope. A file whose rows were all deleted is absent.
    """
    from daft import col
    from daft.datatype import DataType
    from daft.io.iceberg._deletes import read_with_deletes

    if not paths:
        return {}
    df = read_with_deletes(
        table=table,
        plan=plan,
        paths=paths,
        snapshot_id=snapshot_id,
        io_config=io_config,
        file_path_column=_FILE_COL,
    )
    if scope is None:
        # Every live row matches; one table column is read so each row is produced.
        first = col(table.schema().fields[0].name)
        live = first.count("all").alias(_LIVE_COL)
        counts = df.select(col(_FILE_COL), first).groupby(_FILE_COL).agg(live, live.alias(_MATCHED_COL)).to_pydict()
    else:
        counts = (
            scope.matched(df)
            .select(col(_FILE_COL), col(_MATCHED_COL).cast(DataType.int64()))
            .groupby(_FILE_COL)
            .agg(col(_MATCHED_COL).count().alias(_LIVE_COL), col(_MATCHED_COL).sum().alias(_MATCHED_COL))
            .to_pydict()
        )
    return {
        str(path): (int(live), int(matched))
        for path, live, matched in zip(counts[_FILE_COL], counts[_LIVE_COL], counts[_MATCHED_COL])
    }


def _checked_against(df: DataFrame, condition: Expression, message: str) -> DataFrame:
    """Return ``df`` failing its query, with ``message``, on any row where ``condition`` is not true."""
    from daft import col
    from daft.expressions import Expression

    first = df.column_names[0]
    checked = Expression._call_builtin_scalar_fn("require_true", col(first), condition, message=message)
    return df.with_column(first, checked)


def _write(
    df: DataFrame, table: PyIcebergTable, io_config: IOConfig, *, require_matching_columns: bool
) -> tuple[list[DataFile], DataFrame]:
    """Write ``df`` as data files of ``table`` without committing them; return them and the write's frame."""
    from daft.dataframe import DataFrame

    write_df = DataFrame(df._builder.write_iceberg(table, io_config, require_matching_columns=require_matching_columns))
    write_df.collect()
    data_files = write_df.to_pydict().get("data_file", [])
    return [data_file for data_file in data_files if data_file is not None], write_df


def _discard(table: PyIcebergTable, data_files: list[DataFile]) -> None:
    """Delete files no snapshot will reference."""
    for data_file in data_files:
        try:
            table.io.delete(data_file.file_path)
        except OSError as exc:
            logger.warning("overwrite: could not remove uncommitted file %s: %s", data_file.file_path, exc)


def _commit(
    table: PyIcebergTable,
    *,
    start_id: int | None,
    added: list[WrittenFile],
    removed: list[RemovedFile],
    may_hold: Callable[[DataFile], bool] | None,
    snapshot_properties: dict[str, str],
) -> None:
    """Commit one overwrite snapshot, retrying a bare version race and refusing a real overlap.

    ``may_hold``, under serializable isolation, says whether a data file a
    concurrent commit added may hold a row in scope; ``None`` permits them.
    """
    from pyiceberg.table.refs import MAIN_BRANCH
    from pyiceberg.table.snapshots import Operation

    from daft.io.iceberg._compact import _starting_sequence_number
    from daft.io.iceberg._deletes import stable_partition_key

    removed_paths = {str(file.data_file.file_path) for file in removed}
    touched_partitions = {stable_partition_key(file.data_file.partition) for file in removed}
    survivor_paths = frozenset(str(file.data_file.file_path) for file in added if file.rewritten)

    def _attempt(_: int) -> None:
        table.refresh()
        _raise_on_conflict(
            table,
            start_id=start_id,
            removed_paths=removed_paths,
            touched_partitions=touched_partitions,
            may_hold=may_hold,
        )
        tx = begin_transaction(table)
        producer = _overwrite_producer_class()(
            operation=Operation.OVERWRITE,
            transaction=tx,
            io=table.io,
            commit_uuid=_uuid.uuid4(),
            snapshot_properties=snapshot_properties,
            branch=MAIN_BRANCH,
            starting_sequence_number=_starting_sequence_number(table, start_id),
            survivor_paths=survivor_paths,
        )
        for gone in removed:
            producer.delete_data_file(gone.data_file)
        for written in added:
            producer.append_data_file(written.data_file)
        producer.commit()
        tx.commit_transaction()

    commit_with_retry(table, _attempt, op_name="overwrite")


def _raise_on_conflict(
    table: PyIcebergTable,
    *,
    start_id: int | None,
    removed_paths: set[str],
    touched_partitions: set[str],
    may_hold: Callable[[DataFile], bool] | None,
) -> None:
    """Raise :class:`WriteConflict` when a commit since ``start_id`` overlaps the overwrite.

    Any isolation refuses a commit that removed a file the overwrite replaces,
    or added a row-level delete that may cover one. ``may_hold``, given under
    serializable isolation, also refuses a commit that added a data file that
    may hold a row in scope.
    """
    from pyiceberg.manifest import DataFileContent, ManifestEntryStatus

    from daft.io.iceberg._compact import _delete_may_apply, _entries_of

    foreign, reached_start = _snapshots_since(table, start_id)
    if not reached_start:
        raise WriteConflict(
            f"the table's history no longer contains snapshot {start_id}, which the overwrite was planned on"
        )
    for snapshot in foreign:
        snapshot_id = int(snapshot.snapshot_id)
        for data_file in _entries_of(snapshot, table, ManifestEntryStatus.DELETED, conflict=WriteConflict):
            if str(data_file.file_path) in removed_paths:
                raise WriteConflict(
                    f"snapshot {snapshot_id} removed {data_file.file_path!r}, which the overwrite replaces"
                )
        for data_file in _entries_of(snapshot, table, ManifestEntryStatus.ADDED, conflict=WriteConflict):
            if data_file.content != DataFileContent.DATA:
                if _delete_may_apply(data_file, table, removed_paths, touched_partitions):
                    raise WriteConflict(
                        f"snapshot {snapshot_id} added the row-level delete {data_file.file_path!r} "
                        "covering a file the overwrite replaces"
                    )
            elif may_hold is not None and may_hold(data_file):
                raise WriteConflict(
                    f"snapshot {snapshot_id} added {data_file.file_path!r}, which may hold rows "
                    "the overwrite's filter selects"
                )


def _snapshots_since(table: PyIcebergTable, start_id: int | None) -> tuple[list[Snapshot], bool]:
    """Return the main branch's snapshots after ``start_id``, and whether ``start_id`` is still an ancestor.

    With no start, every snapshot of the branch came after it.
    """
    snapshots: list[Snapshot] = []
    snapshot = table.current_snapshot()
    while snapshot is not None and (start_id is None or int(snapshot.snapshot_id) != start_id):
        snapshots.append(snapshot)
        parent_id = snapshot.parent_snapshot_id
        snapshot = table.metadata.snapshot_by_id(parent_id) if parent_id is not None else None
    return snapshots, start_id is None or snapshot is not None


def _may_hold_rows_of(table: PyIcebergTable, scope: BooleanExpression) -> Callable[[DataFile], bool]:
    """Return whether a data file's partition and metrics allow a row the predicate selects."""
    scan = table.scan(row_filter=scope)
    metrics = scan._build_metrics_evaluator()
    partitions: dict[int, Callable[[DataFile], bool]] = {}

    def _may_hold(data_file: DataFile) -> bool:
        spec_id = int(data_file.spec_id)
        if spec_id not in partitions:
            partitions[spec_id] = scan._build_partition_evaluator(spec_id)
        return partitions[spec_id](data_file) and metrics(data_file)

    return _may_hold


_OVERWRITE_PRODUCER_CLASS: type | None = None


def _overwrite_producer_class() -> type:
    """Return the snapshot producer an overwrite commits through, built once."""
    global _OVERWRITE_PRODUCER_CLASS
    if _OVERWRITE_PRODUCER_CLASS is not None:
        return _OVERWRITE_PRODUCER_CLASS

    from daft.io.iceberg._compact import _compaction_producer_class

    class _OverwriteProducer(_compaction_producer_class()):  # type: ignore[misc]
        """Commits an overwrite: kept rows at the sequence number they were read at, new rows at the commit's."""

        def __init__(
            self,
            *,
            operation: Operation,
            transaction: Transaction,
            io: FileIO,
            commit_uuid: _uuid.UUID,
            snapshot_properties: dict[str, str],
            branch: str,
            starting_sequence_number: int | None,
            survivor_paths: frozenset[str],
        ) -> None:
            super().__init__(
                operation=operation,
                transaction=transaction,
                io=io,
                commit_uuid=commit_uuid,
                snapshot_properties=snapshot_properties,
                branch=branch,
                starting_sequence_number=starting_sequence_number,
            )
            self._survivor_paths = survivor_paths

        def _added_entry_sequence_number_for(self, data_file: DataFile) -> int | None:
            """Keep a rewritten file's rows under row-level deletes committed after the plan."""
            if str(data_file.file_path) in self._survivor_paths:
                return self._added_entry_sequence_number()
            return None

    _OVERWRITE_PRODUCER_CLASS = _OverwriteProducer
    return _OVERWRITE_PRODUCER_CLASS
