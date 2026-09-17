"""Changing or removing the rows of a table that satisfy a condition.

Both operations are a merge with nothing to merge against: the rows to act on
are named by a condition rather than found by pairing with a source. They share
the write path with a merge, so the same two ways of recording a change apply.

A removal that covers whole files does not read them at all: the table's own
record of what each file holds is enough to prove no row survives, so the files
are dropped and nothing is written.
"""

from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from daft.datatype import DataType
from daft.io.iceberg._common import ROW_FILE_COLUMN, ROW_POSITION_COLUMN, scalar_options
from daft.io.iceberg._row_level import (
    ACTION_COLUMN,
    ACTION_DELETE,
    ACTION_KEEP,
    ACTION_UPDATE,
    COPY_ON_WRITE,
    DELETE_COUNT_KEYS,
    MERGE_ON_READ,
    SNAPSHOT_PROP_MERGE_ID,
    SNAPSHOT_PROP_MODE,
    SNAPSHOT_PROP_OPERATION,
    UPDATE_COUNT_KEYS,
    DeleteWriterOpener,
    FileTable,
    collect_written_files,
    commit_row_level,
    discard_files,
    distribute,
    find_replayed_snapshot,
    provenance_scan,
    resolve_delete_granularity,
    resolve_delete_target_size,
    resolve_distribution,
    resolve_isolation,
    resolve_mode,
    validate_row_level_commit,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pyiceberg.expressions import BooleanExpression
    from pyiceberg.manifest import DataFile
    from pyiceberg.table import Table as PyIcebergTable

    from daft.dataframe import DataFrame
    from daft.expressions import Expression
    from daft.io.iceberg._common import MaintenanceOptions
    from daft.io.iceberg._row_level import WrittenFiles

_RECOGNISED_OPTIONS = frozenset({"isolation-level", "merge-id"})


class RowLevelFailedException(RuntimeError):
    """Raised when rows cannot be changed or removed."""


@dataclass(frozen=True)
class UpdateResult:
    """What an update changed."""

    snapshot_id: int
    operation: str
    mode: str
    operation_id: str
    rows_updated: int
    rows_copied: int
    added_data_files: int
    removed_data_files: int
    added_delete_files: int


@dataclass(frozen=True)
class DeleteResult:
    """What a removal changed."""

    snapshot_id: int
    operation: str
    mode: str
    operation_id: str
    rows_deleted: int
    rows_copied: int
    added_data_files: int
    removed_data_files: int
    added_delete_files: int
    files_dropped: int


def _as_table_predicate(where: Expression, table: PyIcebergTable) -> tuple[BooleanExpression, bool]:
    """Return the condition as the table understands it, and whether it is exact.

    A condition the table cannot express is reported as one that holds
    everywhere, which is safe: it prunes nothing and assumes any concurrent
    change could matter.
    """
    from pyiceberg.expressions import AlwaysTrue

    from daft.io.iceberg._expressions import convert_expression_to_iceberg

    try:
        return convert_expression_to_iceberg(where, table.schema()), True
    except (ValueError, NotImplementedError):
        return AlwaysTrue(), False


def _whole_file_matches(
    table: PyIcebergTable, predicate: BooleanExpression, files: Sequence[DataFile]
) -> set[str]:
    """Return the files the condition covers entirely.

    A file whose recorded bounds put every row inside the condition needs no
    reading: dropping it removes exactly the rows the condition names.
    """
    from pyiceberg.expressions.visitors import _StrictMetricsEvaluator

    evaluate = _StrictMetricsEvaluator(table.schema(), predicate, case_sensitive=True).eval
    return {str(data_file.file_path) for data_file in files if evaluate(data_file)}


def _tagged_rows(
    frame: DataFrame,
    *,
    where: Expression,
    assignments: Mapping[str, Expression] | None,
    columns: Sequence[str],
    types: Mapping[str, DataType],
    mode: str,
    changed_action: int,
) -> DataFrame:
    """Tag each row with what happens to it, and apply any new values.

    Rewriting whole files keeps the rows that stay, since they have to be written
    again; recording positions keeps only the rows that change.
    """
    from daft.expressions import col, lit
    from daft.functions import when

    action = lit(changed_action).cast(DataType.uint8())
    keep = lit(ACTION_KEEP).cast(DataType.uint8())
    values = []
    for name in columns:
        replacement = (assignments or {}).get(name)
        value = col(name) if replacement is None else when(where, then=replacement).otherwise(col(name))
        values.append(value.cast(types[name]).alias(name))
    tagged = frame.select(
        *values,
        col(ROW_FILE_COLUMN),
        col(ROW_POSITION_COLUMN),
        when(where, then=action).otherwise(keep).alias(ACTION_COLUMN),
    )
    if mode == MERGE_ON_READ:
        return tagged.where(col(ACTION_COLUMN) != lit(ACTION_KEEP).cast(DataType.uint8()))
    return tagged


def _run_row_level(
    table: PyIcebergTable,
    where: Expression,
    assignments: Mapping[str, Expression] | None,
    *,
    verb: str,
    branch: str | None,
    options: MaintenanceOptions | None,
) -> tuple[int, str, str, str, WrittenFiles | None, int]:
    """Apply ``where`` to the table and commit the result.

    Returns the snapshot, its operation, the mode used, the operation's name, what
    was written and how many files were dropped without being read.
    """
    from pyiceberg.table.snapshots import Operation

    from daft.dataframe import DataFrame
    from daft.io.iceberg._common import CommitRetryExhausted
    from daft.io.iceberg._compact import _branch_head, _io_config_for_table
    from daft.io.iceberg._deletes import plan_files, stable_partition_key
    from daft.io.iceberg._merge import _column_types, _table_columns

    settings = scalar_options(options or {})
    unknown = set(settings) - _RECOGNISED_OPTIONS
    if unknown:
        raise RowLevelFailedException(f"unknown option(s): {sorted(unknown)}")

    properties = dict(table.properties)
    mode = resolve_mode(properties, verb)
    isolation = resolve_isolation(
        properties, verb, None if settings.get("isolation-level") is None else str(settings["isolation-level"])
    )
    operation_id = str(settings.get("merge-id") or _uuid.uuid4().hex[:16])
    io_config = _io_config_for_table(table)

    head = _branch_head(table, branch)
    replayed = find_replayed_snapshot(table, operation_id, branch)
    if replayed is not None:
        return (int(replayed.snapshot_id), str(replayed.summary.operation.value), mode, operation_id, None, 0)

    starting_snapshot_id = int(head.snapshot_id) if head is not None else None
    # Only a condition the table's own metadata can reason about narrows the scan
    # or the check against concurrent writers; the rows themselves are always
    # selected by evaluating the condition as given.
    predicate, understood = _as_table_predicate(where, table)
    plan = plan_files(table, table.scan(row_filter=predicate, snapshot_id=starting_snapshot_id))
    file_table = FileTable.from_plan(plan)
    columns = _table_columns(table)
    types = _column_types(table)

    # Files the condition covers entirely are removed without being read, which
    # only applies when nothing is written in their place.
    whole_files: set[str] = set()
    if assignments is None and understood:
        whole_files = _whole_file_matches(
            table, predicate, [file_table.entry(index).data_file for index in range(len(file_table))]
        )
    paths = [path for path in file_table.paths if path not in whole_files]

    written = None
    if paths:
        target = provenance_scan(table=table, plan=plan, file_table=file_table, paths=paths, io_config=io_config)
        tagged = _tagged_rows(
            target,
            where=where,
            assignments=assignments,
            columns=columns,
            types=types,
            mode=mode,
            changed_action=ACTION_UPDATE if assignments is not None else ACTION_DELETE,
        )
        partitioned = bool(table.spec().fields)
        laid_out = distribute(tagged, table, resolve_distribution(properties, verb, mode, partitioned), mode)
        granularity = resolve_delete_granularity(properties)
        written = collect_written_files(
            DataFrame(
                laid_out._builder.write_iceberg_row_delta(
                    table,
                    io_config,
                    action_column=ACTION_COLUMN,
                    file_index_column=ROW_FILE_COLUMN,
                    position_column=ROW_POSITION_COLUMN,
                    delete_writer_factory=(
                        DeleteWriterOpener(table, file_table, io_config) if mode == MERGE_ON_READ else None
                    ),
                    delete_file_paths=list(file_table.paths) if mode == MERGE_ON_READ else None,
                    delete_file_groups=(file_table.delete_groups(granularity) if mode == MERGE_ON_READ else None),
                    delete_target_file_size=(resolve_delete_target_size(properties) if mode == MERGE_ON_READ else None),
                )
            )
        )

    removed: list[DataFile] = [file_table.entry(file_table.index_of(path)).data_file for path in sorted(whole_files)]
    if mode == COPY_ON_WRITE:
        removed.extend(file_table.entry(file_table.index_of(path)).data_file for path in paths)

    added_data = list(written.data_files) if written is not None else []
    added_deletes = list(written.delete_files) if written is not None else []
    if not added_data and not added_deletes and not removed:
        snapshot = int(head.snapshot_id) if head is not None else -1
        return (snapshot, "none", mode, operation_id, written, 0)

    operation = Operation.DELETE if not added_data and not added_deletes else Operation.OVERWRITE
    touched_partitions = {stable_partition_key(file_table.entry(file_table.index_of(path)).partition) for path in paths}

    def _validate() -> None:
        validate_row_level_commit(
            table,
            starting_snapshot_id=starting_snapshot_id,
            merge_id=operation_id,
            branch=branch,
            isolation=isolation,
            # Removing rows twice loses nothing, so only a change to what a row
            # holds has to see the removals another writer recorded.
            checks_deletes=assignments is not None,
            touched_paths=[*paths, *sorted(whole_files)],
            touched_partitions=touched_partitions,
            conflict_filter=predicate,
        )

    counts = written
    try:
        snapshot_id = commit_row_level(
            table,
            operation=operation,
            snapshot_properties=_snapshot_properties(verb, operation_id, mode, counts),
            added_data_files=added_data,
            added_delete_files=added_deletes,
            removed_files=removed,
            validate=_validate,
            replay_key=operation_id,
            branch=branch,
            op_name=f"{verb}_where",
        )
    except CommitRetryExhausted as exhausted:
        discard_files(table, [*added_data, *added_deletes])
        raise RowLevelFailedException(str(exhausted)) from exhausted

    return (snapshot_id, str(operation.value), mode, operation_id, written, len(whole_files))


def _snapshot_properties(
    verb: str, operation_id: str, mode: str, written: WrittenFiles | None
) -> dict[str, str]:
    """Return what the snapshot records about the change."""
    keys = UPDATE_COUNT_KEYS if verb == "update" else DELETE_COUNT_KEYS
    changed = 0 if written is None else (written.rows_updated if verb == "update" else written.rows_deleted)
    copied = 0 if written is None else written.rows_kept
    properties = {
        SNAPSHOT_PROP_OPERATION: f"{verb}-where",
        SNAPSHOT_PROP_MERGE_ID: operation_id,
        SNAPSHOT_PROP_MODE: mode,
        keys["copied"]: str(copied),
    }
    properties[keys["updated" if verb == "update" else "deleted"]] = str(changed)
    return properties


def run_update(
    table: PyIcebergTable,
    where: Expression,
    assignments: Mapping[str, Expression],
    *,
    branch: str | None = None,
    options: MaintenanceOptions | None = None,
) -> UpdateResult:
    """Replace named columns of the rows ``where`` selects.

    Raises:
    ------
    RowLevelFailedException
        If an option is unknown, or if another writer keeps invalidating what
        the update planned against.
    """
    if not assignments:
        raise RowLevelFailedException("an update needs at least one column to change")
    snapshot_id, operation, mode, operation_id, written, _ = _run_row_level(
        table, where, assignments, verb="update", branch=branch, options=options
    )
    return UpdateResult(
        snapshot_id=snapshot_id,
        operation=operation,
        mode=mode,
        operation_id=operation_id,
        rows_updated=0 if written is None else written.rows_updated,
        rows_copied=0 if written is None else written.rows_kept,
        added_data_files=0 if written is None else len(written.data_files),
        removed_data_files=0,
        added_delete_files=0 if written is None else len(written.delete_files),
    )


def run_delete(
    table: PyIcebergTable,
    where: Expression,
    *,
    branch: str | None = None,
    options: MaintenanceOptions | None = None,
) -> DeleteResult:
    """Remove the rows ``where`` selects.

    Raises:
    ------
    RowLevelFailedException
        If an option is unknown, or if another writer keeps invalidating what
        the removal planned against.
    """
    snapshot_id, operation, mode, operation_id, written, dropped = _run_row_level(
        table, where, None, verb="delete", branch=branch, options=options
    )
    return DeleteResult(
        snapshot_id=snapshot_id,
        operation=operation,
        mode=mode,
        operation_id=operation_id,
        rows_deleted=0 if written is None else written.rows_deleted,
        rows_copied=0 if written is None else written.rows_kept,
        added_data_files=0 if written is None else len(written.data_files),
        removed_data_files=0,
        added_delete_files=0 if written is None else len(written.delete_files),
        files_dropped=dropped,
    )
