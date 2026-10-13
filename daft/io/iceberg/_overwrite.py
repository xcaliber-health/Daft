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
from dataclasses import dataclass
from typing import TYPE_CHECKING

from daft.io.iceberg._common import begin_transaction, commit_with_retry

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyiceberg.expressions import BooleanExpression
    from pyiceberg.io import FileIO
    from pyiceberg.manifest import DataFile
    from pyiceberg.table import Table as PyIcebergTable
    from pyiceberg.table import Transaction
    from pyiceberg.table.snapshots import Operation, Snapshot

    from daft.daft import IOConfig
    from daft.dataframe import DataFrame
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
    from pyiceberg.expressions import AlwaysTrue

    from daft.io.iceberg._deletes import plan_files, read_with_deletes
    from daft.io.iceberg._expressions import convert_filter
    from daft.io.iceberg._metadata import convert_iceberg_schema

    _require_boolean(overwrite_filter, convert_iceberg_schema(table.schema()), "the table's columns")
    _require_boolean(overwrite_filter, df.schema(), "the written rows' columns")
    isolation = _isolation_level(table)

    start = table.current_snapshot()
    start_id = int(start.snapshot_id) if start is not None else None
    prune = convert_filter(overwrite_filter._expr, table.schema()) or AlwaysTrue()
    # Planned without the filter: a filtered plan omits row-level deletes whose
    # bounds miss the filter, and those still apply to the rows a rewrite keeps.
    plan = plan_files(table, table.scan(snapshot_id=start_id))
    may_hold = _may_hold_rows_of(table, prune)
    candidates = [path for path, task in plan.tasks.items() if may_hold(task.file)]
    removed, partial = _select_files(table, plan, candidates, overwrite_filter, start_id, io_config)

    in_scope = overwrite_filter.fill_null(False)
    new_files, write_df = _write(
        _checked_against(df, overwrite_filter), table, io_config, require_matching_columns=False
    )
    survivor_files: list[DataFile] = []
    try:
        if partial:
            survivors = read_with_deletes(
                table=table, plan=plan, paths=partial, snapshot_id=start_id, io_config=io_config
            ).where(~in_scope)
            survivor_files, _ = _write(survivors, table, io_config, require_matching_columns=True)
        added = [WrittenFile(f, rewritten=False) for f in new_files] + [
            WrittenFile(f, rewritten=True) for f in survivor_files
        ]
        if added or removed:
            _commit(
                table,
                start_id=start_id,
                added=added,
                removed=removed,
                scope=prune,
                isolation=isolation,
                snapshot_properties=snapshot_properties,
            )
    except BaseException:
        _discard(table, [*new_files, *survivor_files])
        raise
    return write_result(table, added, removed, like=write_df)


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
    scope: Expression,
    snapshot_id: int | None,
    io_config: IOConfig,
) -> tuple[list[RemovedFile], list[str]]:
    """Return which of the ``candidates`` the filter removes, and the paths of those only partly removed.

    A file without row-level deletes whose metrics show every row matches is
    removed without being read; the other candidates are read for the filter's
    columns, and kept whole when no row matches.
    """
    from pyiceberg.expressions.visitors import _StrictMetricsEvaluator

    strict = _strict_predicate(scope, table)
    must_match = (
        _StrictMetricsEvaluator(table.schema(), strict, case_sensitive=True).eval if strict is not None else None
    )
    removed: list[RemovedFile] = []
    to_count: list[str] = []
    for path in candidates:
        task = plan.tasks[path]
        has_deletes = bool(task.delete_files) or path in plan.equality_by_path
        if must_match is not None and not has_deletes and must_match(task.file):
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


def _strict_predicate(scope: Expression, table: PyIcebergTable) -> BooleanExpression | None:
    """Return the filter as a table predicate when it translates exactly, else ``None``."""
    from pyiceberg.exceptions import ValidationError
    from pyiceberg.expressions import BooleanExpression
    from pyiceberg.expressions.visitors import bind

    from daft.io.iceberg._visitors import IcebergPredicateVisitor

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
    scope: Expression | None,
) -> dict[str, tuple[int, int]]:
    """Return, per file, its live rows and how many of them ``scope`` is true on.

    Only the filter's columns are read. Without a filter, every live row counts
    as matching. A file whose rows were all deleted is absent.
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
        matched = scope.fill_null(False).cast(DataType.int64()).alias(_MATCHED_COL)
        counts = (
            df.select(col(_FILE_COL), matched)
            .groupby(_FILE_COL)
            .agg(col(_MATCHED_COL).count().alias(_LIVE_COL), col(_MATCHED_COL).sum().alias(_MATCHED_COL))
            .to_pydict()
        )
    return {
        str(path): (int(live), int(matched))
        for path, live, matched in zip(counts[_FILE_COL], counts[_LIVE_COL], counts[_MATCHED_COL])
    }


def _checked_against(df: DataFrame, scope: Expression) -> DataFrame:
    """Return ``df`` failing its query on any row where ``scope`` is not true."""
    from daft import col
    from daft.expressions import Expression

    first = df.column_names[0]
    checked = Expression._call_builtin_scalar_fn(
        "require_true",
        col(first),
        scope,
        message=f"a row written by an overwrite must satisfy its filter {scope}",
    )
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
    scope: BooleanExpression,
    isolation: str,
    snapshot_properties: dict[str, str],
) -> None:
    """Commit one overwrite snapshot, retrying a bare version race and refusing a real overlap."""
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
            scope=scope,
            isolation=isolation,
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
    scope: BooleanExpression,
    isolation: str,
) -> None:
    """Raise :class:`WriteConflict` when a commit since ``start_id`` overlaps the overwrite.

    Any isolation refuses a commit that removed a file the overwrite replaces,
    or added a row-level delete that may cover one. ``serializable`` also
    refuses a commit that added a data file the filter may select rows of.
    """
    from pyiceberg.manifest import DataFileContent, ManifestEntryStatus

    from daft.io.iceberg._compact import _delete_may_apply, _entries_of

    foreign, reached_start = _snapshots_since(table, start_id)
    if not reached_start:
        raise WriteConflict(
            f"the table's history no longer contains snapshot {start_id}, which the overwrite was planned on"
        )
    may_hold_scope = _may_hold_rows_of(table, scope) if isolation == ISOLATION_SERIALIZABLE else None
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
            elif may_hold_scope is not None and may_hold_scope(data_file):
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
