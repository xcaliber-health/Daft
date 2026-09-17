"""Merging rows of a source into a table.

A merge pairs the table's rows with a source's and decides, per pair, whether to
replace the table's row, remove it, leave it alone, or add a row that has no
counterpart. The rules are given per group: pairs that matched, source rows that
matched nothing, and table rows that matched nothing.

How the change is written depends on the table. Under copy-on-write the files
holding changed rows are rewritten without them and the new rows are written
alongside. Under merge-on-read the old rows stay where they are and their
positions are recorded, which makes the write proportional to the change rather
than to the files it touches.
"""

from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from daft.daft import JoinType, MergeClause, MergeRowsConfig
from daft.datatype import DataType
from daft.io.iceberg._common import ROW_FILE_COLUMN, ROW_POSITION_COLUMN, scalar_options
from daft.io.iceberg._row_level import (
    ACTION_COLUMN,
    ACTION_DELETE,
    COPY_ON_WRITE,
    MERGE_ON_READ,
    SNAPSHOT_PROP_MERGE_ID,
    SNAPSHOT_PROP_MODE,
    SNAPSHOT_PROP_OPERATION,
    SOURCE_PRESENT_COLUMN,
    TARGET_PRESENT_COLUMN,
    DeleteWriterOpener,
    FileTable,
    RowLevelConflict,
    WrittenFiles,
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

    from pyiceberg.manifest import DataFile
    from pyiceberg.table import Table as PyIcebergTable
    from pyiceberg.table.snapshots import Operation, Snapshot

    from daft.daft import IOConfig
    from daft.dataframe import DataFrame
    from daft.expressions import Expression
    from daft.io.iceberg._common import MaintenanceOptions
    from daft.io.iceberg._deletes import ScanPlan


#: Groups a rule can belong to, by which side of a pair exists.
_MATCHED = "matched"
_NOT_MATCHED = "not_matched"
_NOT_MATCHED_BY_SOURCE = "not_matched_by_source"

_RECOGNISED_OPTIONS = frozenset({"isolation-level", "merge-id"})


class MergeFailedException(RuntimeError):
    """Raised when a merge cannot be completed."""


class MergeCardinalityError(MergeFailedException):
    """Raised when one row of the table is matched by more than one source row.

    Which of the matching rows should win is not defined, so the merge refuses
    rather than picking one. Remove the duplicate keys from the source, or narrow
    the condition so each row of the table matches at most once.
    """


@dataclass(frozen=True)
class MergeResult:
    """What a merge changed."""

    snapshot_id: int
    operation: str
    mode: str
    merge_id: str
    rows_updated: int
    rows_deleted: int
    rows_inserted: int
    rows_copied: int
    added_data_files: int
    removed_data_files: int
    added_delete_files: int
    removed_delete_files: int


@dataclass(frozen=True)
class _Rule:
    """One rule of a merge, as the caller stated it."""

    group: str
    action: str
    condition: Expression | None = None
    assignments: Mapping[str, Expression] | None = None
    all_columns: bool = False


@dataclass
class _Rules:
    """The rules of one merge, in the order they were given."""

    matched: list[_Rule] = field(default_factory=list)
    not_matched: list[_Rule] = field(default_factory=list)
    not_matched_by_source: list[_Rule] = field(default_factory=list)

    def add(self, rule: _Rule) -> None:
        getattr(self, rule.group).append(rule)


class MatchedRule:
    """What to do with a pair the condition selected."""

    def __init__(self, builder: MergeIntoBuilder, group: str, condition: Expression | None) -> None:
        self._builder = builder
        self._group = group
        self._condition = condition

    def update(self, assignments: Mapping[str, Expression]) -> MergeIntoBuilder:
        """Replace the named columns of the table's row; the rest keep their values."""
        self._builder._rules.add(
            _Rule(group=self._group, action="update", condition=self._condition, assignments=assignments)
        )
        return self._builder

    def update_all(self) -> MergeIntoBuilder:
        """Replace every column of the table's row with the source column of the same name."""
        self._builder._rules.add(_Rule(group=self._group, action="update", condition=self._condition, all_columns=True))
        return self._builder

    def delete(self) -> MergeIntoBuilder:
        """Remove the table's row."""
        self._builder._rules.add(_Rule(group=self._group, action="delete", condition=self._condition))
        return self._builder


class NotMatchedRule:
    """What to do with a source row that matched nothing."""

    def __init__(self, builder: MergeIntoBuilder, condition: Expression | None) -> None:
        self._builder = builder
        self._condition = condition

    def insert(self, assignments: Mapping[str, Expression]) -> MergeIntoBuilder:
        """Add a row built from the named columns; the rest are left empty."""
        self._builder._rules.add(
            _Rule(group=_NOT_MATCHED, action="insert", condition=self._condition, assignments=assignments)
        )
        return self._builder

    def insert_all(self) -> MergeIntoBuilder:
        """Add a row taking every column from the source column of the same name."""
        self._builder._rules.add(
            _Rule(group=_NOT_MATCHED, action="insert", condition=self._condition, all_columns=True)
        )
        return self._builder


class MergeIntoBuilder:
    """States the rules of a merge, then runs it.

    Rules are tried in the order they are given, and the first whose condition
    holds decides the row. A row no rule claims is left as it is.
    """

    def __init__(
        self,
        table: PyIcebergTable,
        source: DataFrame,
        on: Expression,
        *,
        target_alias: str = "target",
        source_alias: str = "source",
        branch: str | None = None,
        options: MaintenanceOptions | None = None,
    ) -> None:
        self._table = table
        self._source = source
        self._on = on
        self._target_alias = target_alias
        self._source_alias = source_alias
        self._branch = branch
        self._options = options
        self._rules = _Rules()

    def when_matched(self, condition: Expression | None = None) -> MatchedRule:
        """Start a rule for pairs where both sides exist."""
        return MatchedRule(self, _MATCHED, condition)

    def when_not_matched(self, condition: Expression | None = None) -> NotMatchedRule:
        """Start a rule for source rows that matched nothing."""
        return NotMatchedRule(self, condition)

    def when_not_matched_by_source(self, condition: Expression | None = None) -> MatchedRule:
        """Start a rule for the table's rows that matched nothing."""
        return MatchedRule(self, _NOT_MATCHED_BY_SOURCE, condition)

    def execute(self) -> MergeResult:
        """Run the merge and return what it changed."""
        return run_merge(
            self._table,
            self._source,
            self._on,
            self._rules,
            target_alias=self._target_alias,
            source_alias=self._source_alias,
            branch=self._branch,
            options=self._options,
        )


def _table_columns(table: PyIcebergTable) -> list[str]:
    """Return the table's column names, in schema order."""
    return [column.name for column in table.schema().columns]


def _required_columns(table: PyIcebergTable) -> set[str]:
    """Return the columns a new row must supply a value for."""
    return {field.name for field in table.schema().fields if field.required}


def _column_types(table: PyIcebergTable) -> dict[str, DataType]:
    """Return the type each column is written as."""
    from daft.io.iceberg._metadata import convert_iceberg_schema

    return {field.name: field.dtype for field in convert_iceberg_schema(table.schema())}


def _outputs_for(
    rule: _Rule,
    *,
    columns: Sequence[str],
    types: Mapping[str, DataType],
    target_alias: str,
    source_alias: str,
    required: set[str],
) -> list[Expression]:
    """Return one expression per written column, then the row's origin.

    A rule that adds a row has no origin to report, so its origin columns are
    empty; every other rule carries through where the row came from.
    """
    from daft.expressions import col, lit

    def target(name: str) -> Expression:
        return col(f"{target_alias}.{name}")

    def source(name: str) -> Expression:
        return col(f"{source_alias}.{name}")

    assignments = dict(rule.assignments or {})
    unknown = set(assignments) - set(columns)
    if unknown:
        raise MergeFailedException(f"{sorted(unknown)} are not columns of the table")

    values: list[Expression] = []
    for name in columns:
        if name in assignments:
            value = assignments[name]
        elif rule.all_columns:
            value = source(name)
        elif rule.action == "insert":
            if name in required:
                raise MergeFailedException(f"a new row must supply a value for {name!r}")
            value = lit(None)
        else:
            value = target(name)
        values.append(value.cast(types[name]).alias(name))

    if rule.action == "insert":
        values.append(lit(None).cast(DataType.int32()).alias(ROW_FILE_COLUMN))
        values.append(lit(None).cast(DataType.int64()).alias(ROW_POSITION_COLUMN))
    else:
        values.append(col(ROW_FILE_COLUMN))
        values.append(col(ROW_POSITION_COLUMN))
    return values


def _merge_clause(rule: _Rule, outputs: Sequence[Expression]) -> MergeClause:
    """Return the rule as the operator states it."""
    return MergeClause(
        rule.action,
        [expr._expr for expr in outputs],
        rule.condition._expr if rule.condition is not None else None,
    )


def _keep_clause(columns: Sequence[str], types: Mapping[str, DataType], target_alias: str) -> MergeClause:
    """Return the rule that carries a row through unchanged."""
    keep = _Rule(group=_MATCHED, action="keep")
    outputs = _outputs_for(
        keep,
        columns=columns,
        types=types,
        target_alias=target_alias,
        source_alias=target_alias,
        required=set(),
    )
    return MergeClause("keep", [expr._expr for expr in outputs], None)


def _join_type(mode: str, has_not_matched: bool, has_not_matched_by_source: bool) -> str:
    """Return the join that brings in every row the rules can claim.

    Rows of a side are only there to be claimed if some rule can claim them, so
    a merge that adds nothing needs no unmatched source rows, and one that
    rewrites files always needs the table's rows.
    """
    if mode == COPY_ON_WRITE:
        return "outer" if has_not_matched else "left"
    if has_not_matched and has_not_matched_by_source:
        return "outer"
    if has_not_matched:
        return "right"
    if has_not_matched_by_source:
        return "left"
    return "inner"


def _checks_cardinality(rules: _Rules) -> bool:
    """Return whether one row of the table may be claimed by only one source row.

    With no rule for matched pairs nothing depends on which pair wins, and one
    unconditional removal reaches the same outcome however many pairs there are.
    """
    if not rules.matched:
        return False
    if len(rules.matched) == 1:
        only = rules.matched[0]
        return not (only.action == "delete" and only.condition is None)
    return True


def _aliased_target(frame: DataFrame, columns: Sequence[str], alias: str) -> DataFrame:
    """Name the table's columns apart from the source's, and mark its rows present."""
    from daft.expressions import col, lit

    return frame.select(
        *[col(name).alias(f"{alias}.{name}") for name in columns],
        col(ROW_FILE_COLUMN),
        col(ROW_POSITION_COLUMN),
        lit(True).alias(TARGET_PRESENT_COLUMN),
    )


def _aliased_source(frame: DataFrame, alias: str) -> DataFrame:
    """Name the source's columns apart from the table's, and mark its rows present."""
    from daft.expressions import col, lit

    return frame.select(
        *[col(name).alias(f"{alias}.{name}") for name in frame.schema().column_names()],
        lit(True).alias(SOURCE_PRESENT_COLUMN),
    )


def _affected_files(
    *,
    table: PyIcebergTable,
    plan: ScanPlan,
    file_table: FileTable,
    source: DataFrame,
    on: Expression,
    io_config: IOConfig,
    target_alias: str,
    source_alias: str,
) -> list[int]:
    """Return the files holding rows the source can match.

    Only the columns the condition reads are decoded, and the result is a list of
    file indices, so this costs a pass over those columns rather than the table.
    """
    from daft.dataframe import DataFrame
    from daft.expressions import col

    target = provenance_scan(
        table=table,
        plan=plan,
        file_table=file_table,
        paths=file_table.paths,
        io_config=io_config,
    )
    keys = _aliased_target(target, _table_columns(table), target_alias)
    matched = keys._builder.join_on(
        _aliased_source(source, source_alias)._builder,
        on,
        JoinType.Semi,
        build_on_left=False,
    )
    indices = DataFrame(matched).select(col(ROW_FILE_COLUMN)).distinct().to_pydict()[ROW_FILE_COLUMN]
    return sorted(int(index) for index in indices)


def _carried_over_deletes(
    *,
    table: PyIcebergTable,
    plan: ScanPlan,
    file_table: FileTable,
    columns: Sequence[str],
    types: Mapping[str, DataType],
    io_config: IOConfig,
) -> tuple[DataFrame | None, list[DataFile]]:
    """Return the removals already recorded for the scanned files, and their files.

    A delete file that names one data file is replaced rather than added to, so
    the rows it holds are carried into the file that replaces it. Removals
    recorded for a whole partition are left alone.
    """
    import daft
    from daft.expressions import col, lit
    from daft.io.iceberg._compact import _referenced_data_file

    rewritable: dict[str, DataFile] = {}
    for path, task in plan.tasks.items():
        del path
        for delete_file in task.delete_files:
            if _referenced_data_file(delete_file) is not None:
                rewritable[str(delete_file.file_path)] = delete_file
    if not rewritable:
        return None, []

    positions = daft.read_parquet(list(rewritable), io_config=io_config).select(
        col("file_path"), col("pos").alias(ROW_POSITION_COLUMN)
    )
    known = daft.from_pydict({"file_path": list(file_table.paths), ROW_FILE_COLUMN: list(range(len(file_table)))})
    rows = positions.join(known, on="file_path", how="inner").select(
        *[lit(None).cast(types[name]).alias(name) for name in columns],
        col(ROW_FILE_COLUMN).cast(DataType.int32()),
        col(ROW_POSITION_COLUMN).cast(DataType.int64()),
        lit(ACTION_DELETE).cast(DataType.uint8()).alias(ACTION_COLUMN),
    )
    return rows, list(rewritable.values())


def run_merge(
    table: PyIcebergTable,
    source: DataFrame,
    on: Expression,
    rules: _Rules,
    *,
    target_alias: str = "target",
    source_alias: str = "source",
    branch: str | None = None,
    options: MaintenanceOptions | None = None,
) -> MergeResult:
    """Merge ``source`` into ``table`` under ``rules`` and commit the result.

    Raises:
    ------
    MergeFailedException
        If the rules cannot be applied to the table, or if another writer keeps
        invalidating what the merge planned against.
    MergeCardinalityError
        If one row of the table is matched by more than one source row.
    """
    from daft.io.iceberg._row_level import attempt_with_replan

    def _once() -> MergeResult:
        return _merge_once(
            table,
            source,
            on,
            rules,
            target_alias=target_alias,
            source_alias=source_alias,
            branch=branch,
            options=options,
        )

    try:
        return attempt_with_replan(table, _once, op_name="merge_into")
    except RowLevelConflict as conflict:
        raise MergeFailedException(str(conflict)) from conflict


def _merge_once(
    table: PyIcebergTable,
    source: DataFrame,
    on: Expression,
    rules: _Rules,
    *,
    target_alias: str,
    source_alias: str,
    branch: str | None,
    options: MaintenanceOptions | None,
) -> MergeResult:
    """Carry out one attempt of a merge against the branch's current head."""
    from pyiceberg.expressions import AlwaysTrue

    from daft.dataframe import DataFrame
    from daft.io.iceberg._common import CommitRetryExhausted
    from daft.io.iceberg._compact import _branch_head, _io_config_for_table
    from daft.io.iceberg._deletes import plan_files, stable_partition_key

    if not (rules.matched or rules.not_matched or rules.not_matched_by_source):
        raise MergeFailedException("a merge needs at least one rule")
    settings = scalar_options(options or {})
    unknown = set(settings) - _RECOGNISED_OPTIONS
    if unknown:
        raise MergeFailedException(f"unknown option(s): {sorted(unknown)}")

    properties = dict(table.properties)
    mode = resolve_mode(properties, "merge")
    isolation = resolve_isolation(properties, "merge", _option_str(settings, "isolation-level"))
    merge_id = _option_str(settings, "merge-id") or _uuid.uuid4().hex[:16]
    io_config = _io_config_for_table(table)

    head = _branch_head(table, branch)
    replayed = find_replayed_snapshot(table, merge_id, branch)
    if replayed is not None:
        return _result_from_snapshot(replayed, mode, merge_id)

    starting_snapshot_id = int(head.snapshot_id) if head is not None else None
    plan = plan_files(table, table.scan(snapshot_id=starting_snapshot_id))
    file_table = FileTable.from_plan(plan)
    columns = _table_columns(table)
    types = _column_types(table)

    # Under copy-on-write only the files holding matched rows are rewritten, so
    # they are found first from the join keys alone. A rule for the table's
    # unmatched rows can change any row, which leaves nothing to narrow down.
    paths = list(file_table.paths)
    if mode == COPY_ON_WRITE and not rules.not_matched_by_source and paths:
        affected = _affected_files(
            table=table,
            plan=plan,
            file_table=file_table,
            source=source,
            on=on,
            io_config=io_config,
            target_alias=target_alias,
            source_alias=source_alias,
        )
        paths = [file_table.entry(index).path for index in affected]
        if not paths and not rules.not_matched:
            return _unchanged(head, mode, merge_id)

    target = provenance_scan(table=table, plan=plan, file_table=file_table, paths=paths, io_config=io_config)
    joined = DataFrame(
        _aliased_target(target, columns, target_alias)._builder.join_on(
            _aliased_source(source, source_alias)._builder,
            on,
            _JOIN_TYPES[_join_type(mode, bool(rules.not_matched), bool(rules.not_matched_by_source))],
            build_on_left=False,
        )
    )

    config = _merge_config(
        rules,
        columns=columns,
        types=types,
        target_alias=target_alias,
        source_alias=source_alias,
        required=_required_columns(table),
        mode=mode,
    )
    merged = DataFrame(joined._builder.merge_rows(config))

    carried: DataFrame | None = None
    replaced_deletes: list[DataFile] = []
    if mode == MERGE_ON_READ:
        carried, replaced_deletes = _carried_over_deletes(
            table=table, plan=plan, file_table=file_table, columns=columns, types=types, io_config=io_config
        )
        if carried is not None:
            merged = merged.concat(carried)

    partitioned = bool(table.spec().fields)
    laid_out = distribute(merged, table, resolve_distribution(properties, "merge", mode, partitioned), mode)
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

    removed = (
        [file_table.entry(file_table.index_of(path)).data_file for path in paths]
        if mode == COPY_ON_WRITE
        else list(replaced_deletes)
    )
    if not written.data_files and not written.delete_files and not removed:
        return _unchanged(head, mode, merge_id)

    operation = _snapshot_operation(mode, written, removed)
    touched_partitions = {stable_partition_key(file_table.entry(file_table.index_of(path)).partition) for path in paths}

    def _validate() -> None:
        validate_row_level_commit(
            table,
            starting_snapshot_id=starting_snapshot_id,
            merge_id=merge_id,
            branch=branch,
            isolation=isolation,
            checks_deletes=True,
            touched_paths=paths,
            touched_partitions=touched_partitions,
            conflict_filter=AlwaysTrue(),
        )

    try:
        snapshot_id = commit_row_level(
            table,
            operation=operation,
            snapshot_properties=_snapshot_properties(merge_id, mode, written),
            added_data_files=written.data_files,
            added_delete_files=written.delete_files,
            removed_files=removed,
            validate=_validate,
            replay_key=merge_id,
            branch=branch,
            op_name="merge_into",
        )
    except CommitRetryExhausted as exhausted:
        discard_files(table, [*written.data_files, *written.delete_files])
        raise MergeFailedException(str(exhausted)) from exhausted
    except RowLevelConflict:
        discard_files(table, [*written.data_files, *written.delete_files])
        raise

    return MergeResult(
        snapshot_id=snapshot_id,
        operation=str(operation.value),
        mode=mode,
        merge_id=merge_id,
        rows_updated=written.rows_updated,
        rows_deleted=written.rows_deleted,
        rows_inserted=written.rows_inserted,
        rows_copied=written.rows_kept,
        added_data_files=len(written.data_files),
        removed_data_files=len([f for f in removed if _is_data(f)]),
        added_delete_files=len(written.delete_files),
        removed_delete_files=len([f for f in removed if not _is_data(f)]),
    )


#: Join each merge shape needs, by the rows its rules can claim.
_JOIN_TYPES = {
    "inner": JoinType.Inner,
    "left": JoinType.Left,
    "right": JoinType.Right,
    "outer": JoinType.Outer,
}


def _option_str(settings: Mapping[str, str | int | float | bool], key: str) -> str | None:
    """Return a text option, or ``None`` when it was not given."""
    value = settings.get(key)
    return None if value is None else str(value)


def _is_data(data_file: DataFile) -> bool:
    """Return whether the file holds rows rather than the positions of removed rows."""
    from pyiceberg.manifest import DataFileContent

    return data_file.content == DataFileContent.DATA


def _merge_config(
    rules: _Rules,
    *,
    columns: Sequence[str],
    types: Mapping[str, DataType],
    target_alias: str,
    source_alias: str,
    required: set[str],
    mode: str,
) -> MergeRowsConfig:
    """Return the rules as the operator states them.

    Rewriting whole files means every row of a file must come out of the merge,
    so a rule that carries rows through unchanged is added to the groups that can
    hold them. Recording removals instead needs no such rule: a row nothing
    claims simply stays where it is.
    """
    from daft.expressions import col

    def clauses(group: Sequence[_Rule]) -> list[MergeClause]:
        return [
            _merge_clause(
                rule,
                _outputs_for(
                    rule,
                    columns=columns,
                    types=types,
                    target_alias=target_alias,
                    source_alias=source_alias,
                    required=required,
                ),
            )
            for rule in group
        ]

    matched = clauses(rules.matched)
    not_matched = clauses(rules.not_matched)
    not_matched_by_source = clauses(rules.not_matched_by_source)
    if mode == COPY_ON_WRITE:
        keep = _keep_clause(columns, types, target_alias)
        matched.append(keep)
        not_matched_by_source.append(keep)

    row_id = [col(ROW_FILE_COLUMN)._expr, col(ROW_POSITION_COLUMN)._expr] if _checks_cardinality(rules) else []
    return MergeRowsConfig(
        matched,
        not_matched,
        not_matched_by_source,
        col(TARGET_PRESENT_COLUMN).not_null()._expr,
        col(SOURCE_PRESENT_COLUMN).not_null()._expr,
        row_id,
        ACTION_COLUMN,
    )


def _snapshot_operation(mode: str, written: WrittenFiles, removed: Sequence[DataFile]) -> Operation:
    """Return what the commit is: rows added, rows removed, or rows replaced."""
    from pyiceberg.table.snapshots import Operation

    del mode
    if not removed and not written.delete_files:
        return Operation.APPEND
    if not written.data_files:
        return Operation.DELETE
    return Operation.OVERWRITE


def _snapshot_properties(merge_id: str, mode: str, written: WrittenFiles) -> dict[str, str]:
    """Return what the snapshot records about the merge."""
    from daft.io.iceberg._row_level import MERGE_COUNT_KEYS

    return {
        SNAPSHOT_PROP_OPERATION: "merge-into",
        SNAPSHOT_PROP_MERGE_ID: merge_id,
        SNAPSHOT_PROP_MODE: mode,
        MERGE_COUNT_KEYS["copied"]: str(written.rows_kept),
        MERGE_COUNT_KEYS["updated"]: str(written.rows_updated),
        MERGE_COUNT_KEYS["deleted"]: str(written.rows_deleted),
        MERGE_COUNT_KEYS["inserted"]: str(written.rows_inserted),
    }


def _unchanged(head: Snapshot | None, mode: str, merge_id: str) -> MergeResult:
    """Return the result of a merge that changed nothing, which commits nothing."""
    return MergeResult(
        snapshot_id=int(head.snapshot_id) if head is not None else -1,
        operation="none",
        mode=mode,
        merge_id=merge_id,
        rows_updated=0,
        rows_deleted=0,
        rows_inserted=0,
        rows_copied=0,
        added_data_files=0,
        removed_data_files=0,
        added_delete_files=0,
        removed_delete_files=0,
    )


def _result_from_snapshot(snapshot: Snapshot, mode: str, merge_id: str) -> MergeResult:
    """Return what an earlier run of this merge recorded in its snapshot."""
    from daft.io.iceberg._row_level import MERGE_COUNT_KEYS

    summary = snapshot.summary.additional_properties if snapshot.summary is not None else {}

    def count(key: str) -> int:
        return int(summary.get(key, "0"))

    return MergeResult(
        snapshot_id=int(snapshot.snapshot_id),
        operation=str(snapshot.summary.operation.value) if snapshot.summary is not None else "overwrite",
        mode=summary.get(SNAPSHOT_PROP_MODE, mode),
        merge_id=merge_id,
        rows_updated=count(MERGE_COUNT_KEYS["updated"]),
        rows_deleted=count(MERGE_COUNT_KEYS["deleted"]),
        rows_inserted=count(MERGE_COUNT_KEYS["inserted"]),
        rows_copied=count(MERGE_COUNT_KEYS["copied"]),
        added_data_files=int(summary.get("added-data-files", "0")),
        removed_data_files=int(summary.get("deleted-data-files", "0")),
        added_delete_files=int(summary.get("added-delete-files", "0")),
        removed_delete_files=int(summary.get("removed-delete-files", "0")),
    )
