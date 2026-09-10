"""Compact or re-cluster data files: enumerate candidates, plan groups, read+write outputs, commit atomically."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid as _uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

from daft.daft import _iceberg as _rust_iceberg
from daft.io.iceberg._common import (
    CommitRetryExhausted,
    MaintenanceOptions,
    branch_ancestry,
    commit_with_retry,
    manifest_writer_for,
    scalar_options,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from pyiceberg.expressions import BooleanExpression
    from pyiceberg.io import FileIO
    from pyiceberg.manifest import (
        DataFile,
        DataFileContent,
        ManifestContent,
        ManifestEntry,
        ManifestEntryStatus,
        ManifestFile,
        ManifestWriter,
    )
    from pyiceberg.partitioning import PartitionSpec
    from pyiceberg.table import FileScanTask, Transaction
    from pyiceberg.table import Table as PyIcebergTable
    from pyiceberg.table.snapshots import Operation, Snapshot, Summary

    from daft.daft import IOConfig
    from daft.daft._iceberg import CandidateRecord, FileGroupRecord, OptionValue
    from daft.dataframe import DataFrame

from daft.io.iceberg._deletes import ScanPlan, plan_files, read_with_deletes, stable_partition_key

logger = logging.getLogger(__name__)

#: What one group's rewrite produces.
_Output = TypeVar("_Output")


class RewriteConflict(RuntimeError):
    """Concurrent writer removed input files mid-retry; outputs are orphaned."""


SUPPORTED_STRATEGIES = ("binpack", "sort", "zorder")
_VALID_SORT_DIRECTIONS = {"asc", "desc"}
_VALID_NULL_ORDERS = {"nulls-first", "nulls-last"}
_ZORDER_KEY_COL = "__daft_zorder_key__"
SNAPSHOT_PROP_REWRITE_ID = "daft.rewrite-id"
SNAPSHOT_PROP_STRATEGY = "daft.rewrite-strategy"
SNAPSHOT_PROP_INPUT_FILES = "daft.rewrite-input-files"
SNAPSHOT_PROP_OUTPUT_FILES = "daft.rewrite-output-files"
SNAPSHOT_PROP_BATCH = "daft.rewrite-batch"
SNAPSHOT_PROP_DANGLING_REMOVED = "daft.rewrite-dangling-deletes-removed"
SNAPSHOT_PROP_CONSUMED_DELETES = "daft.rewrite-consumed-delete-files"
SNAPSHOT_PROP_DROPPED_DELETES = "daft.rewrite-dropped-delete-files"
#: Batch label of the snapshot that reclaims dangling deletes after a rewrite.
DANGLING_BATCH_LABEL = "dangling"
SNAPSHOT_PROP_MAINTENANCE_OP = "daft.maintenance.op"
SNAPSHOT_PROP_MAINTENANCE_OP_VALUE = "rewrite-data-files"

WRITE_TARGET_FILE_SIZE_BYTES_KEY = "write.target-file-size-bytes"

#: Level of the commit-time conflict check: ``snapshot`` permits concurrent
#: appends into a touched partition, ``serializable`` refuses them.
CONFLICT_ISOLATION_KEY = "conflict-isolation"

#: Caller-supplied identity for an idempotent replay of the same rewrite.
REWRITE_ID_KEY = "rewrite-id"

#: Content marker of a data file, as opposed to a row-level delete file.
_DATA_CONTENT = 0

#: Field id reserved for the data file path column of a position delete.
_DELETE_FILE_PATH_FIELD_ID = 2147483546
CONFLICT_ISOLATION_SERIALIZABLE = "serializable"
CONFLICT_ISOLATION_SNAPSHOT = "snapshot"
_VALID_CONFLICT_ISOLATIONS = (
    CONFLICT_ISOLATION_SERIALIZABLE,
    CONFLICT_ISOLATION_SNAPSHOT,
)


def _parse_conflict_isolation(raw_options: dict[str, OptionValue]) -> str:
    """Pop and validate the conflict-isolation option, defaulting to ``snapshot``."""
    value = raw_options.pop(CONFLICT_ISOLATION_KEY, CONFLICT_ISOLATION_SNAPSHOT)
    if not isinstance(value, str) or value not in _VALID_CONFLICT_ISOLATIONS:
        raise ValueError(f"{CONFLICT_ISOLATION_KEY} must be one of {_VALID_CONFLICT_ISOLATIONS}, got {value!r}")
    return value


def _parse_rewrite_id(raw_options: dict[str, OptionValue]) -> str | None:
    """Pop the caller-supplied rewrite id from ``raw_options``."""
    value = raw_options.pop(REWRITE_ID_KEY, None)
    return str(value) if value else None


@dataclass(frozen=True)
class RewriteResult:
    """Summary of a rewrite_data_files invocation.

    Parameters
    ----------
    strategy
        The strategy applied: ``"binpack"``, ``"sort"``, or ``"zorder"``.
    rewritten_files
        Number of input data files removed by the rewrite.
    added_files
        Number of output data files written.
    bytes_rewritten
        Total size in bytes of the removed data files.
    bytes_added
        Total size in bytes of the written data files.
    removed_delete_files
        Number of positional delete files consumed during read, plus any deletes
        dropped by ``remove-dangling-deletes`` post-processing.
    failed_groups
        Number of file groups that did not land: their rewrite failed, or the
        batch holding them could not commit. Non-zero only when
        ``partial-progress.enabled=true``; without it any failure raises.
    commits
        Number of snapshots produced. Always ``1`` in atomic mode; up to
        ``partial-progress.max-commits`` otherwise.
    snapshot_ids
        Snapshot IDs created by this call, in commit order.
    rewrite_id
        Stable identifier used for idempotent replay.
    failed_data_files
        Same as ``failed_groups`` but counted at file granularity.
    """

    strategy: str
    rewritten_files: int
    added_files: int
    bytes_rewritten: int
    bytes_added: int
    removed_delete_files: int
    failed_groups: int
    commits: int
    snapshot_ids: list[int] = field(default_factory=list)
    rewrite_id: str = ""
    failed_data_files: int = 0


class RewriteFailedException(RuntimeError):
    """Rewrite could not make forward progress."""


def run(
    table: PyIcebergTable,
    strategy: str,
    sort_order: list[tuple[str, str, str]] | None,
    zorder_by: list[str] | None,
    where: str | BooleanExpression | None,
    branch: str | None,
    options: MaintenanceOptions | None,
) -> RewriteResult:
    """Plan, rewrite and commit the data files a strategy selects."""
    from pyiceberg.expressions import AlwaysTrue
    from pyiceberg.manifest import DataFileContent

    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"strategy must be one of {SUPPORTED_STRATEGIES}, got {strategy!r}")
    parsed_sort_order: list[tuple[str, bool, bool]] | None = None
    parsed_zorder_by: list[str] | None = None
    if strategy == "sort":
        parsed_sort_order = _parse_sort_order(sort_order, table)
    elif strategy == "zorder":
        parsed_zorder_by = _parse_zorder_columns(zorder_by, table)
    output_sort_order_id = _resolve_sort_order(table, strategy, parsed_sort_order)

    raw_options: dict[str, OptionValue] = dict(scalar_options(options))
    # The planner validator rejects keys it does not know.
    conflict_isolation = _parse_conflict_isolation(raw_options)
    explicit_rewrite_id = _parse_rewrite_id(raw_options)
    # The table's own target keeps the rewrite and other writers sized alike.
    if "target-file-size-bytes" not in raw_options:
        prop = table.properties.get(WRITE_TARGET_FILE_SIZE_BYTES_KEY)
        if prop is not None:
            raw_options["target-file-size-bytes"] = int(prop)
    normalized = _rust_iceberg.validate_options_py(raw_options)

    row_filter = where if where is not None else AlwaysTrue()
    if branch is not None:
        starting_snapshot = table.snapshot_by_name(branch)
    else:
        starting_snapshot = table.current_snapshot()
    starting_snapshot_id: int | None = int(starting_snapshot.snapshot_id) if starting_snapshot is not None else None
    plan = plan_files(table, table.scan(row_filter=row_filter, snapshot_id=starting_snapshot_id))
    plan_by_path = plan.tasks

    candidates: list[CandidateRecord] = []
    for path, task in plan_by_path.items():
        pos_deletes: list[str] = []
        deleted_rows = 0
        for d in task.delete_files:
            if d.content == DataFileContent.POSITION_DELETES:
                pos_deletes.append(d.file_path)
                # Only a delete scoped to this one file has rows attributable to it.
                if _referenced_data_file(d) == path:
                    deleted_rows += int(d.record_count or 0)
        candidates.append(
            {
                "path": path,
                "size_bytes": int(task.file.file_size_in_bytes),
                "partition_key": stable_partition_key(task.file.partition),
                "partition_spec_id": int(task.file.spec_id),
                "positional_delete_paths": pos_deletes,
                "equality_delete_paths": [d.path for d in plan.equality_by_path.get(path, ())],
                "record_count": int(task.file.record_count or 0),
                "deleted_record_count": deleted_rows,
            }
        )

    current_spec_id = int(table.spec().spec_id)
    groups = _rust_iceberg.plan_file_groups_py(candidates, raw_options, current_spec_id)

    rewrite_id = _resolve_rewrite_id(table, branch, strategy, normalized, candidates, explicit_rewrite_id)
    cached = _lookup_idempotent_result(table, rewrite_id, strategy, branch)
    if cached is not None:
        logger.info("rewrite_data_files: idempotency hit on rewrite_id=%s; skipping", rewrite_id)
        return cached

    if not groups:
        return RewriteResult(
            strategy=strategy,
            rewritten_files=0,
            added_files=0,
            bytes_rewritten=0,
            bytes_added=0,
            removed_delete_files=0,
            failed_groups=0,
            commits=0,
            snapshot_ids=[],
            rewrite_id=rewrite_id,
        )

    io_config = _io_config_for_table(table)
    result = _rewrite_and_commit(
        table=table,
        groups=groups,
        plan=plan,
        snapshot_id=starting_snapshot_id,
        io_config=io_config,
        normalized_options=normalized,
        strategy=strategy,
        sort_order=parsed_sort_order,
        zorder_by=parsed_zorder_by,
        output_sort_order_id=output_sort_order_id,
        rewrite_id=rewrite_id,
        branch=branch,
        conflict_isolation=conflict_isolation,
    )

    if normalized.get("remove-dangling-deletes"):
        removed, snapshot_id = _remove_dangling_deletes(table, branch=branch, rewrite_id=rewrite_id, strategy=strategy)
        if snapshot_id is not None:
            result = _augment_result_with_dangling(result, removed, snapshot_id)
    return result


def _io_config_for_table(table: PyIcebergTable) -> IOConfig:
    """Return the storage configuration recorded on the table, or the process default."""
    from daft.context import get_context
    from daft.io.iceberg._iceberg import (
        _convert_iceberg_file_io_properties_to_io_config,
    )

    io_config = _convert_iceberg_file_io_properties_to_io_config(table.io.properties)
    if io_config is not None:
        return io_config
    return get_context().daft_planning_config.default_io_config


@dataclass
class _Tally:
    """Running totals of what a rewrite has committed and what it has lost."""

    rewritten_files: int = 0
    added_files: int = 0
    bytes_rewritten: int = 0
    bytes_added: int = 0
    removed_delete_files: int = 0
    failed_groups: int = 0
    failed_data_files: int = 0
    failed_batches: int = 0
    snapshot_ids: list[int] = field(default_factory=list)

    def add_commit(self, result: RewriteResult) -> None:
        self.rewritten_files += result.rewritten_files
        self.added_files += result.added_files
        self.bytes_rewritten += result.bytes_rewritten
        self.bytes_added += result.bytes_added
        self.removed_delete_files += result.removed_delete_files
        self.snapshot_ids.extend(result.snapshot_ids)

    def add_failed_groups(self, groups: list[FileGroupRecord]) -> None:
        self.failed_groups += len(groups)
        self.failed_data_files += sum(len(g["files"]) for g in groups)

    def add_failed_batch(self, outputs: list[_GroupOutput]) -> None:
        self.failed_batches += 1
        self.failed_groups += len(outputs)
        self.failed_data_files += sum(len(o.input_data_files) for o in outputs)

    def result(self, strategy: str, rewrite_id: str) -> RewriteResult:
        return RewriteResult(
            strategy=strategy,
            rewritten_files=self.rewritten_files,
            added_files=self.added_files,
            bytes_rewritten=self.bytes_rewritten,
            bytes_added=self.bytes_added,
            removed_delete_files=self.removed_delete_files,
            failed_groups=self.failed_groups,
            commits=len(self.snapshot_ids),
            snapshot_ids=list(self.snapshot_ids),
            rewrite_id=rewrite_id,
            failed_data_files=self.failed_data_files,
        )


def _rewrite_and_commit(
    *,
    table: PyIcebergTable,
    groups: list[FileGroupRecord],
    plan: ScanPlan,
    snapshot_id: int | None,
    io_config: IOConfig,
    normalized_options: dict[str, OptionValue],
    strategy: str,
    sort_order: list[tuple[str, bool, bool]] | None,
    zorder_by: list[str] | None,
    output_sort_order_id: int,
    rewrite_id: str,
    branch: str | None,
    conflict_isolation: str,
) -> RewriteResult:
    """Rewrite the planned groups and commit them as batches finish, in plan order.

    Batches are cut by plan position so a replay finds each under the same label.
    """
    partial = bool(normalized_options.get("partial-progress.enabled"))
    use_starting_sequence_number = bool(normalized_options["use-starting-sequence-number"])
    if partial:
        max_commits = max(1, int(normalized_options["partial-progress.max-commits"]))
        n_batches = min(max_commits, len(groups))
        max_failed_commits = int(normalized_options.get("partial-progress.max-failed-commits", max_commits))
    else:
        n_batches = 1
        max_failed_commits = 0
    groups_per_batch = -(-len(groups) // n_batches)
    batches = [groups[i : i + groups_per_batch] for i in range(0, len(groups), groups_per_batch)]

    def _rewrite_one(group: FileGroupRecord) -> _GroupOutput:
        return _rewrite_group(
            table=table,
            group=group,
            plan=plan,
            snapshot_id=snapshot_id,
            io_config=io_config,
            normalized_options=normalized_options,
            strategy=strategy,
            sort_order=sort_order,
            zorder_by=zorder_by,
            output_sort_order_id=output_sort_order_id,
        )

    def _commit_outputs(
        outputs: list[_GroupOutput], label: str | None
    ) -> tuple[RewriteResult | None, Exception | None]:
        return _commit_batch(
            table=table,
            batch=outputs,
            plan_by_path=plan.tasks,
            rewrite_id=rewrite_id,
            strategy=strategy,
            batch_label=label,
            branch=branch,
            starting_snapshot_id=snapshot_id,
            conflict_isolation=conflict_isolation,
            use_starting_sequence_number=use_starting_sequence_number,
        )

    tally = _Tally()
    pending: list[_GroupOutput] = []
    # The distributed runner cannot run plans from several threads at once.
    max_concurrent = (
        1 if _writes_one_file_per_partition() else max(1, int(normalized_options["max-concurrent-file-group-rewrites"]))
    )
    for index, outputs, failures in _rewrite_batches(batches, _rewrite_one, max_concurrent=max_concurrent):
        if failures and not partial:
            _abort_outputs(table, [*pending, *outputs])
            raise failures[0][1]
        tally.add_failed_groups([group for group, _ in failures])
        if not outputs:
            continue
        if not partial:
            pending.extend(outputs)
            continue
        label = f"{index + 1}/{len(batches)}"
        result, err = _commit_outputs(outputs, label)
        if result is None:
            logger.warning(
                "rewrite_data_files: batch %s of %s could not commit (%s); its outputs are removed",
                index + 1,
                len(batches),
                type(err).__name__ if err else "unknown",
            )
            _abort_outputs(table, outputs)
            tally.add_failed_batch(outputs)
            continue
        tally.add_commit(result)

    if not partial:
        result, err = _commit_outputs(pending, None)
        if result is None:
            _abort_outputs(table, pending)
            assert err is not None
            if isinstance(err, CommitRetryExhausted):
                raise RewriteFailedException(
                    "rewrite_data_files: atomic commit could not land within the "
                    "retry budget. To tolerate concurrent writers, set "
                    "options={'partial-progress.enabled': True}."
                ) from err
            raise err
        return result

    if tally.failed_batches > max_failed_commits:
        raise RewriteFailedException(
            f"rewrite_data_files: {tally.failed_batches} of {len(batches)} batches failed "
            f"(threshold partial-progress.max-failed-commits={max_failed_commits}). "
            f"{len(tally.snapshot_ids)} commit(s) landed; the failed batches' outputs were removed."
        )
    return tally.result(strategy, rewrite_id)


def _rewrite_batches(
    batches: list[list[FileGroupRecord]],
    rewrite_one: Callable[[FileGroupRecord], _Output],
    *,
    max_concurrent: int,
) -> Iterator[tuple[int, list[_Output], list[tuple[FileGroupRecord, BaseException]]]]:
    """Yield each batch's index, finished outputs and ``(group, error)`` failures, in plan order."""
    if max_concurrent == 1 or sum(len(batch) for batch in batches) <= 1:
        for index, batch in enumerate(batches):
            outputs: list[_Output] = []
            failures: list[tuple[FileGroupRecord, BaseException]] = []
            for group in batch:
                try:
                    outputs.append(rewrite_one(group))
                except Exception as exc:
                    logger.error("rewrite_data_files: group rewrite failed: %s", exc)
                    failures.append((group, exc))
            yield index, outputs, failures
        return

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = [[pool.submit(rewrite_one, group) for group in batch] for batch in batches]
        for index, batch in enumerate(batches):
            outputs = []
            failures = []
            for group, future in zip(batch, futures[index]):
                try:
                    outputs.append(future.result())
                except Exception as exc:
                    logger.error("rewrite_data_files: group rewrite failed: %s", exc)
                    failures.append((group, exc))
            yield index, outputs, failures


def _abort_outputs(table: PyIcebergTable, outputs: list[_GroupOutput]) -> None:
    """Delete the files a batch wrote once no snapshot will reference them."""
    for path in _orphan_output_paths(outputs):
        try:
            table.io.delete(path)
        except OSError as exc:
            logger.warning("rewrite_data_files: could not remove aborted output %s: %s", path, exc)


def _augment_result_with_dangling(result: RewriteResult, removed_delete_files: int, snapshot_id: int) -> RewriteResult:
    """Fold the reclaiming snapshot into the result: its count, its commit, its id."""
    return RewriteResult(
        strategy=result.strategy,
        rewritten_files=result.rewritten_files,
        added_files=result.added_files,
        bytes_rewritten=result.bytes_rewritten,
        bytes_added=result.bytes_added,
        removed_delete_files=result.removed_delete_files + removed_delete_files,
        failed_groups=result.failed_groups,
        commits=result.commits + 1,
        snapshot_ids=[*result.snapshot_ids, int(snapshot_id)],
        rewrite_id=result.rewrite_id,
        failed_data_files=result.failed_data_files,
    )


@dataclass
class _GroupOutput:
    """The files one group read and wrote, with their byte totals."""

    input_data_files: list[str]
    input_positional_delete_files: list[str]
    input_equality_delete_files: list[str]
    data_files: list[DataFile]
    bytes_added: int
    bytes_rewritten: int


_ZORDER_SUPPORTED_TYPES = {
    "int",
    "long",
    "boolean",
    "float",
    "double",
    "date",
    "timestamp",
    "timestamptz",
    "string",
    "binary",
    "decimal",
}


def _parse_zorder_columns(
    zorder_by: list[str] | None,
    table: PyIcebergTable,
) -> list[str]:
    """Validate zorder_by columns and reject unsupported types (nested, uuid, fixed)."""
    if not zorder_by:
        raise ValueError("strategy='zorder' requires a non-empty zorder_by")
    schema = table.schema()
    by_name = {f.name: f for f in schema.fields}
    out: list[str] = []
    for c in zorder_by:
        if c not in by_name:
            raise ValueError(f"zorder column {c!r} not in table schema")
        type_str = str(by_name[c].field_type).lower()
        # Strip parameters: "decimal(10,2)" -> "decimal", "timestamptz" stays.
        bare = type_str.split("(")[0].strip()
        if bare not in _ZORDER_SUPPORTED_TYPES:
            raise ValueError(
                f"zorder column {c!r} has unsupported type {type_str!r}; supported: {sorted(_ZORDER_SUPPORTED_TYPES)}"
            )
        if c == _ZORDER_KEY_COL:
            raise ValueError(f"column name {_ZORDER_KEY_COL!r} is reserved by the z-order rewrite")
        out.append(c)
    return out


#: Write property the writer reads to size its first output file.
INFLATION_FACTOR_PROPERTY = "daft.write.inflation-factor"

#: Inputs sampled for the disk-to-memory ratio; a few describe the table as well as all.
_INFLATION_SAMPLE_FILES = 3
#: Rows decoded per sampled input; a whole row group can be the whole file.
_INFLATION_SAMPLE_ROWS = 65_536


def _measure_inflation_factor(table: PyIcebergTable, paths: list[str]) -> float | None:
    """Return the in-memory to on-disk size ratio of the first rows of a few inputs."""
    import pyarrow.parquet as pq

    on_disk = 0.0
    in_memory = 0
    for path in paths[:_INFLATION_SAMPLE_FILES]:
        try:
            with table.io.new_input(path).open() as handle:
                parquet_file = pq.ParquetFile(handle)
                if parquet_file.metadata.num_row_groups == 0:
                    continue
                group = parquet_file.metadata.row_group(0)
                if group.num_rows == 0:
                    continue
                batch = next(parquet_file.iter_batches(batch_size=_INFLATION_SAMPLE_ROWS, row_groups=[0]))
                group_bytes = sum(int(group.column(c).total_compressed_size) for c in range(group.num_columns))
                on_disk += group_bytes * batch.num_rows / group.num_rows
                in_memory += int(batch.nbytes)
        except (OSError, ValueError, StopIteration) as exc:
            logger.debug("rewrite_data_files: could not measure %s: %s", path, exc)
            return None
    if on_disk <= 0 or in_memory <= 0:
        return None
    return in_memory / on_disk


def _resolve_sort_order(
    table: PyIcebergTable,
    strategy: str,
    sort_order: list[tuple[str, bool, bool]] | None,
) -> int:
    """Return the sort order id the output files will declare."""
    from pyiceberg.table.sorting import (
        UNSORTED_SORT_ORDER_ID,
        NullOrder,
        SortDirection,
        SortField,
    )
    from pyiceberg.transforms import IdentityTransform

    if strategy != "sort" or not sort_order:
        return UNSORTED_SORT_ORDER_ID

    schema = table.schema()
    fields = [
        SortField(
            source_id=schema.find_field(name).field_id,
            transform=IdentityTransform(),
            direction=SortDirection.DESC if descending else SortDirection.ASC,
            null_order=NullOrder.NULLS_FIRST if nulls_first else NullOrder.NULLS_LAST,
        )
        for (name, descending, nulls_first) in sort_order
    ]

    for candidate in table.sort_orders().values():
        if list(candidate.fields) == fields:
            return int(candidate.order_id)

    logger.warning(
        "rewrite_data_files: the requested sort order matches none registered on the "
        "table, so the rewritten files will not be marked as sorted"
    )
    return UNSORTED_SORT_ORDER_ID


def _parse_sort_order(
    sort_order: list[tuple[str, str, str]] | None,
    table: PyIcebergTable,
) -> list[tuple[str, bool, bool]]:
    """Validate sort_order and return ``[(column_name, descending, nulls_first)]``."""
    if not sort_order:
        raise ValueError("strategy='sort' requires a non-empty sort_order")
    schema = table.schema()
    schema_names = {f.name for f in schema.fields}
    parsed: list[tuple[str, bool, bool]] = []
    for item in sort_order:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(f"sort_order entries must be (column, asc|desc, nulls-first|nulls-last); got {item!r}")
        col, direction, null_order = item
        if col not in schema_names:
            raise ValueError(f"sort column {col!r} not in table schema")
        if direction not in _VALID_SORT_DIRECTIONS:
            raise ValueError(f"sort direction must be one of {_VALID_SORT_DIRECTIONS}, got {direction!r}")
        if null_order not in _VALID_NULL_ORDERS:
            raise ValueError(f"null order must be one of {_VALID_NULL_ORDERS}, got {null_order!r}")
        parsed.append((col, direction == "desc", null_order == "nulls-first"))
    return parsed


def _rewrite_group(
    *,
    table: PyIcebergTable,
    group: FileGroupRecord,
    plan: ScanPlan,
    snapshot_id: int | None,
    io_config: IOConfig,
    normalized_options: dict[str, OptionValue],
    strategy: str,
    sort_order: list[tuple[str, bool, bool]] | None,
    zorder_by: list[str] | None,
    output_sort_order_id: int,
) -> _GroupOutput:
    """Read one group's files, optionally re-cluster, and write target-sized outputs."""
    input_paths = [f["path"] for f in group["files"]]
    flat_delete_paths = sorted({p for f in group["files"] for p in f["positional_delete_paths"]})
    flat_equality_paths = sorted({d.path for path in input_paths for d in plan.equality_by_path.get(path, ())})
    bytes_rewritten = sum(int(f["size_bytes"]) for f in group["files"])

    target_size = int(normalized_options["target-file-size-bytes"])
    output_spec_id = int(group["output_spec_id"])
    # Only bin-pack reads its outputs from a split of the input.
    split_size = int(group.get("input_split_size") or target_size)

    df = read_with_deletes(table=table, plan=plan, paths=input_paths, snapshot_id=snapshot_id, io_config=io_config)

    # Clustering changes the ratio, so an explicit compression-factor wins there.
    inflation_factor = _measure_inflation_factor(table, input_paths)
    compression_factor = float(normalized_options["compression-factor"])
    if strategy != "binpack" and compression_factor != 1.0:
        inflation_factor = compression_factor

    write_target = split_size
    expected_output_files = int(group.get("expected_output_files") or 1)
    if strategy == "sort":
        assert sort_order is not None
        df = df.sort(
            [name for (name, _, _) in sort_order],
            desc=[descending for (_, descending, _) in sort_order],
            nulls_first=[nulls_first for (_, _, nulls_first) in sort_order],
        )
        write_target = _shuffled_target_size(target_size, normalized_options)
    elif strategy == "zorder":
        assert zorder_by is not None
        df = _apply_zorder(df, zorder_by, normalized_options)
        write_target = _shuffled_target_size(target_size, normalized_options)
    # After the sort, so it samples and runs over every input partition.
    df = _repartition_for_output(df, expected_output_files=expected_output_files, keep_order=strategy != "binpack")
    if _writes_one_file_per_partition():
        # Each partition is written alone, so the larger roll size absorbs its remainder.
        write_target = max(write_target, _write_max_file_size(normalized_options))

    data_files = _collect_data_files(
        df=df,
        table=table,
        io_config=io_config,
        target_size=write_target,
        output_spec_id=output_spec_id,
        output_sort_order_id=output_sort_order_id,
        inflation_factor=inflation_factor,
    )
    bytes_added = sum(int(getattr(d, "file_size_in_bytes", 0)) for d in data_files)

    return _GroupOutput(
        input_data_files=input_paths,
        input_positional_delete_files=flat_delete_paths,
        input_equality_delete_files=flat_equality_paths,
        data_files=data_files,
        bytes_added=bytes_added,
        bytes_rewritten=bytes_rewritten,
    )


def _apply_zorder(
    df: DataFrame,
    zorder_by: list[str],
    normalized_options: dict[str, OptionValue],
) -> DataFrame:
    """Sort rows along a space-filling curve over the given columns."""
    from daft.expressions import col as col_expr
    from daft.io.iceberg._zorder import zorder_key

    var_len = int(normalized_options["var-length-contribution"])
    max_out = int(normalized_options["max-output-size"])
    key = zorder_key([col_expr(c) for c in zorder_by], var_len, max_out)
    return (
        df.with_column(_ZORDER_KEY_COL, key)
        .sort(_ZORDER_KEY_COL, desc=False, nulls_first=True)
        .exclude(_ZORDER_KEY_COL)
    )


_MIN_SHUFFLED_TARGET_BYTES = 1024 * 1024


def _writes_one_file_per_partition() -> bool:
    """Return whether the runner writes one file per partition rather than one stream."""
    from daft import runners

    return runners.get_or_create_runner().name == "ray"


def _write_max_file_size(normalized_options: dict[str, OptionValue]) -> int:
    """Return the largest size an output file may reach while being written."""
    target = int(normalized_options["target-file-size-bytes"])
    maximum = int(normalized_options["max-file-size-bytes"])
    return target + max(0, maximum - target) // 2


def _repartition_for_output(df: DataFrame, *, expected_output_files: int, keep_order: bool) -> DataFrame:
    """Give a group one partition per file it is meant to become.

    Surplus partitions are coalesced without moving rows; a shortfall is shuffled unless ``keep_order`` forbids it.
    """
    from daft import runners
    from daft.dataframe import DataFrame

    if runners.get_or_create_runner().name != "ray":
        return df
    partitions = df.num_partitions()
    if expected_output_files < 1 or partitions is None or partitions == expected_output_files:
        return df
    if partitions > expected_output_files:
        return df.into_partitions(expected_output_files)
    if keep_order:
        return df
    # The random shuffle behind ``repartition``, without its advisory warning.
    return DataFrame(df._builder.random_shuffle(expected_output_files))


def _shuffled_target_size(target_size: int, normalized_options: dict[str, OptionValue]) -> int:
    """Divide the write target by ``shuffle-partitions-per-file``, floored at a usable size."""
    factor = int(normalized_options.get("shuffle-partitions-per-file", 1))
    if factor <= 1:
        return target_size
    return max(_MIN_SHUFFLED_TARGET_BYTES, target_size // factor)


def _collect_data_files(
    *,
    df: DataFrame,
    table: PyIcebergTable,
    io_config: IOConfig,
    target_size: int,
    output_spec_id: int,
    output_sort_order_id: int,
    inflation_factor: float | None,
) -> list[DataFile]:
    """Write the frame as target-sized data files under the chosen spec and return their metadata."""
    from daft.dataframe import DataFrame

    write_builder = df._builder.write_iceberg(
        table,
        io_config,
        target_file_size_bytes=target_size,
        partition_spec_id=output_spec_id,
        require_matching_columns=True,
        sort_order_id=output_sort_order_id,
        inflation_factor=inflation_factor,
    )
    write_df = DataFrame(write_builder)
    write_df.collect()
    result = write_df.to_pydict()
    data_files = result.get("data_file", [])
    return [data_file for data_file in data_files if data_file is not None]


def _resolve_rewrite_id(
    table: PyIcebergTable,
    branch: str | None,
    strategy: str,
    normalized_options: dict[str, OptionValue],
    candidates: list[CandidateRecord],
    explicit: str | None,
) -> str:
    """Return the caller's rewrite id, or a digest of the table, branch, strategy, options and inputs."""
    if explicit:
        return explicit
    payload = {
        "table_uuid": str(table.metadata.table_uuid),
        "branch": branch or "main",
        "strategy": strategy,
        "options": {k: normalized_options[k] for k in sorted(normalized_options)},
        "files": sorted(c["path"] for c in candidates),
    }
    h = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return h[:16]


def _summary_as_dict(summary: Summary | None) -> dict[str, str]:
    """Return a snapshot summary's operation and properties as one string mapping."""
    if summary is None:
        return {}
    out: dict[str, str] = {}
    op = getattr(summary, "operation", None)
    if op is not None:
        out["operation"] = str(op.value if hasattr(op, "value") else op)
    extra = getattr(summary, "additional_properties", None)
    if extra:
        out.update({str(k): str(v) for k, v in extra.items()})
    return out


def _lookup_idempotent_result(
    table: PyIcebergTable,
    rewrite_id: str,
    strategy: str,
    branch: str | None,
) -> RewriteResult | None:
    """Return the result an earlier run of this rewrite committed on the branch, if any."""
    matches: list[tuple[Snapshot, dict[str, str]]] = []
    for snap in reversed(branch_ancestry(table, branch)):
        summary = _summary_as_dict(snap.summary)
        if summary.get(SNAPSHOT_PROP_REWRITE_ID) == rewrite_id:
            matches.append((snap, summary))
    if not matches:
        return None
    # Byte totals count data files only; the reclaim snapshot removes delete files.
    rewrites = [s for _, s in matches if s.get(SNAPSHOT_PROP_BATCH) != DANGLING_BATCH_LABEL]
    rewritten_total = sum(int(s.get(SNAPSHOT_PROP_INPUT_FILES, 0)) for s in rewrites)
    added_total = sum(int(s.get(SNAPSHOT_PROP_OUTPUT_FILES, 0)) for s in rewrites)
    bytes_rewritten_total = sum(int(s.get("removed-files-size", 0)) for s in rewrites)
    bytes_added_total = sum(int(s.get("added-files-size", 0)) for s in rewrites)
    removed_deletes_total = sum(
        int(s.get(SNAPSHOT_PROP_CONSUMED_DELETES, 0)) + int(s.get(SNAPSHOT_PROP_DANGLING_REMOVED, 0))
        for _, s in matches
    )
    recorded_strategy = matches[-1][1].get(SNAPSHOT_PROP_STRATEGY, strategy)
    return RewriteResult(
        strategy=recorded_strategy,
        rewritten_files=rewritten_total,
        added_files=added_total,
        bytes_rewritten=bytes_rewritten_total,
        bytes_added=bytes_added_total,
        removed_delete_files=removed_deletes_total,
        failed_groups=0,
        commits=len(matches),
        snapshot_ids=[int(s.snapshot_id) for s, _ in matches],
        rewrite_id=rewrite_id,
    )


def _find_batch_snapshot(
    table: PyIcebergTable, rewrite_id: str, batch_label: str, branch: str | None
) -> Snapshot | None:
    """Return the snapshot on the branch that committed ``batch_label`` of this rewrite, if any."""
    for snap in branch_ancestry(table, branch):
        summary = _summary_as_dict(snap.summary)
        if summary.get(SNAPSHOT_PROP_REWRITE_ID) == rewrite_id and summary.get(SNAPSHOT_PROP_BATCH) == batch_label:
            return snap
    return None


def _starting_sequence_number(table: PyIcebergTable, starting_snapshot_id: int | None) -> int | None:
    """Return the sequence number of the plan snapshot, or ``None`` to let the commit assign one."""
    if starting_snapshot_id is None:
        return None
    snapshot = table.metadata.snapshot_by_id(int(starting_snapshot_id))
    if snapshot is None:
        return None
    sequence_number = getattr(snapshot, "sequence_number", None)
    return None if sequence_number is None else int(sequence_number)


_COMPACTION_PRODUCER_CLASS: type | None = None

#: The table format documents ``commit.manifest-merge.enabled`` as true; the catalog library defaults it to false.
_MANIFEST_MERGE_ENABLED_DEFAULT = True


def _compaction_producer_class() -> type:
    """Return the snapshot producer a rewrite commits through, built once."""
    global _COMPACTION_PRODUCER_CLASS
    if _COMPACTION_PRODUCER_CLASS is not None:
        return _COMPACTION_PRODUCER_CLASS

    from pyiceberg.manifest import DataFileContent, ManifestContent, ManifestEntry, ManifestEntryStatus
    from pyiceberg.table.update.snapshot import _ManifestMergeManager, _OverwriteFiles

    class _DeleteManifestMergeManager(_ManifestMergeManager):  # type: ignore[misc]
        """Merge delete manifests into manifests that still declare delete content."""

        def _create_manifest(self, spec_id: int, manifest_bin: list[ManifestFile]) -> ManifestFile:
            producer = self._snapshot_producer
            with producer.new_manifest_writer_for(ManifestContent.DELETES, producer.spec(spec_id)) as writer:
                for manifest in manifest_bin:
                    for entry in producer.fetch_manifest_entry(manifest=manifest, discard_deleted=False):
                        if entry.status == ManifestEntryStatus.DELETED and entry.snapshot_id == producer.snapshot_id:
                            writer.delete(entry)
                        elif entry.status == ManifestEntryStatus.ADDED and entry.snapshot_id == producer.snapshot_id:
                            writer.add(entry)
                        elif entry.status != ManifestEntryStatus.DELETED:
                            writer.existing(entry)
            return writer.to_manifest_file()

    class _CompactionProducer(_OverwriteFiles):  # type: ignore[misc]
        """Commits a rewrite: the same rows, laid out in different files."""

        def __init__(
            self,
            *,
            operation: Operation,
            transaction: Transaction,
            io: FileIO,
            commit_uuid: _uuid.UUID,
            snapshot_properties: dict[str, str],
            branch: str | None,
            starting_sequence_number: int | None,
        ) -> None:
            super().__init__(
                operation=operation,
                transaction=transaction,
                io=io,
                commit_uuid=commit_uuid,
                snapshot_properties=dict(snapshot_properties),
                branch=branch,
            )
            self._starting_sequence_number = starting_sequence_number

        def _added_entry_sequence_number(self) -> int | None:
            """Return the sequence number to stamp on added entries, or ``None`` to assign a new one.

            Rewritten files keep the number they were read at so existing row-level deletes still apply.
            """
            if self._transaction.table_metadata.format_version < 2:
                return None
            return self._starting_sequence_number

        def _added_entry_sequence_number_for(self, data_file: DataFile) -> int | None:
            """Return the sequence number to stamp on one added file; the batch's by default."""
            del data_file
            return self._added_entry_sequence_number()

        def new_manifest_writer_for(self, content: ManifestContent, spec: PartitionSpec) -> ManifestWriter:
            """Return a writer for a manifest of ``content`` under ``spec``."""
            return manifest_writer_for(self, content, spec)

        def _existing_manifests(self) -> list[ManifestFile]:
            """Keep every manifest untouched by this commit; rewrite the rest without the removed files."""
            metadata = self._transaction.table_metadata
            snapshot = metadata.snapshot_by_name(name=self._target_branch)
            if snapshot is None:
                return []
            kept: list[ManifestFile] = []
            for manifest_file in snapshot.manifests(io=self._io):
                entries = manifest_file.fetch_manifest_entry(io=self._io, discard_deleted=True)
                removed = [entry.data_file for entry in entries if entry.data_file in self._deleted_data_files]
                if not removed:
                    kept.append(manifest_file)
                    continue
                survivors = [entry for entry in entries if entry.data_file not in removed]
                if not survivors:
                    continue
                spec = metadata.specs()[manifest_file.partition_spec_id]
                with self.new_manifest_writer_for(manifest_file.content, spec) as writer:
                    for entry in survivors:
                        writer.add_entry(
                            ManifestEntry.from_args(
                                status=ManifestEntryStatus.EXISTING,
                                snapshot_id=entry.snapshot_id,
                                sequence_number=entry.sequence_number,
                                file_sequence_number=entry.file_sequence_number,
                                data_file=entry.data_file,
                            )
                        )
                kept.append(writer.to_manifest_file())
            return kept

        def _process_manifests(self, manifests: list[ManifestFile]) -> list[ManifestFile]:
            """Merge small manifests of each content kind as the table's commit properties ask."""
            from pyiceberg.table import TableProperties
            from pyiceberg.utils.properties import property_as_bool, property_as_int

            properties = self._transaction.table_metadata.properties
            if not property_as_bool(
                properties, TableProperties.MANIFEST_MERGE_ENABLED, _MANIFEST_MERGE_ENABLED_DEFAULT
            ):
                return manifests
            settings = {
                "target_size_bytes": property_as_int(
                    properties,
                    TableProperties.MANIFEST_TARGET_SIZE_BYTES,
                    TableProperties.MANIFEST_TARGET_SIZE_BYTES_DEFAULT,
                ),
                "min_count_to_merge": property_as_int(
                    properties,
                    TableProperties.MANIFEST_MIN_MERGE_COUNT,
                    TableProperties.MANIFEST_MIN_MERGE_COUNT_DEFAULT,
                ),
                "merge_enabled": True,
                "snapshot_producer": self,
            }
            data = [manifest for manifest in manifests if manifest.content == ManifestContent.DATA]
            deletes = [manifest for manifest in manifests if manifest.content == ManifestContent.DELETES]
            return _ManifestMergeManager(**settings).merge_manifests(data) + _DeleteManifestMergeManager(
                **settings
            ).merge_manifests(deletes)

        def _manifests(self) -> list[ManifestFile]:
            from collections import defaultdict

            from pyiceberg.utils.concurrent import ExecutorFactory

            metadata = self._transaction.table_metadata

            def _write_added_manifests() -> list[ManifestFile]:
                if not self._added_data_files:
                    return []
                # A manifest's declared content must match its entries.
                by_kind: dict[tuple[ManifestContent, int], list[DataFile]] = defaultdict(list)
                for data_file in self._added_data_files:
                    is_delete = data_file.content != DataFileContent.DATA
                    content = ManifestContent.DELETES if is_delete else ManifestContent.DATA
                    by_kind[(content, int(data_file.spec_id))].append(data_file)

                written: list[ManifestFile] = []
                for (content, spec_id), data_files in by_kind.items():
                    with self.new_manifest_writer_for(content, metadata.specs()[spec_id]) as writer:
                        for data_file in data_files:
                            writer.add_entry(
                                ManifestEntry.from_args(
                                    status=ManifestEntryStatus.ADDED,
                                    snapshot_id=self._snapshot_id,
                                    sequence_number=self._added_entry_sequence_number_for(data_file),
                                    file_sequence_number=None,
                                    data_file=data_file,
                                )
                            )
                    written.append(writer.to_manifest_file())
                return written

            def _write_deleted_manifests() -> list[ManifestFile]:
                deleted_entries = self._deleted_entries()
                if not deleted_entries:
                    return []
                # A manifest's declared content must match its entries.
                by_kind: dict[tuple[ManifestContent, int], list[ManifestEntry]] = defaultdict(list)
                for entry in deleted_entries:
                    is_delete = entry.data_file.content != DataFileContent.DATA
                    content = ManifestContent.DELETES if is_delete else ManifestContent.DATA
                    by_kind[(content, int(entry.data_file.spec_id))].append(entry)

                written: list[ManifestFile] = []
                for (content, spec_id), entries in by_kind.items():
                    with self.new_manifest_writer_for(content, metadata.specs()[spec_id]) as writer:
                        for entry in entries:
                            writer.add_entry(entry)
                    written.append(writer.to_manifest_file())
                return written

            executor = ExecutorFactory.get_or_create()
            added = executor.submit(_write_added_manifests)
            deleted = executor.submit(_write_deleted_manifests)
            existing = executor.submit(self._existing_manifests)
            return self._process_manifests(added.result() + deleted.result() + existing.result())

        def _summary(self, snapshot_properties: dict[str, str] | None = None) -> Summary:
            from pyiceberg.table import TableProperties
            from pyiceberg.table.snapshots import (
                Operation,
                SnapshotSummaryCollector,
                Summary,
                update_snapshot_summaries,
            )

            properties = dict(snapshot_properties or {})
            metadata = self._transaction.table_metadata
            specs = metadata.specs()
            schema = metadata.schema()

            collector = SnapshotSummaryCollector(
                partition_summary_limit=int(
                    metadata.properties.get(
                        TableProperties.WRITE_PARTITION_SUMMARY_LIMIT,
                        TableProperties.WRITE_PARTITION_SUMMARY_LIMIT_DEFAULT,
                    )
                )
            )
            for data_file in self._added_data_files:
                collector.add_file(
                    data_file=data_file,
                    partition_spec=specs[int(data_file.spec_id)],
                    schema=schema,
                )
            for data_file in self._deleted_data_files:
                collector.remove_file(
                    data_file=data_file,
                    partition_spec=specs[int(data_file.spec_id)],
                    schema=schema,
                )

            previous_snapshot = (
                metadata.snapshot_by_id(self._parent_snapshot_id) if self._parent_snapshot_id is not None else None
            )
            # The totals helper admits only certain operations; the summary is relabelled after.
            totals = update_snapshot_summaries(
                summary=Summary(operation=Operation.OVERWRITE, **collector.build(), **properties),
                previous_summary=previous_snapshot.summary if previous_snapshot is not None else None,
            )
            return Summary(operation=Operation.REPLACE, **totals.additional_properties)

    _COMPACTION_PRODUCER_CLASS = _CompactionProducer
    return _COMPACTION_PRODUCER_CLASS


def _commit_batch(
    *,
    table: PyIcebergTable,
    batch: list[_GroupOutput],
    plan_by_path: dict[str, FileScanTask],
    rewrite_id: str,
    strategy: str,
    batch_label: str | None,
    branch: str | None,
    starting_snapshot_id: int | None,
    conflict_isolation: str,
    use_starting_sequence_number: bool,
) -> tuple[RewriteResult | None, Exception | None]:
    """Commit one batch under retry; return its result, or the error that refused it."""
    all_data_files = [df_ for o in batch for df_ in o.data_files]
    input_paths = sorted({p for o in batch for p in o.input_data_files})
    delete_files_consumed = sorted(
        {p for o in batch for p in (*o.input_positional_delete_files, *o.input_equality_delete_files)}
    )
    total_in = sum(o.bytes_rewritten for o in batch)
    total_out = sum(o.bytes_added for o in batch)
    touched_partitions = {stable_partition_key(plan_by_path[p].file.partition) for p in input_paths}

    snapshot_props: dict[str, str] = {
        SNAPSHOT_PROP_MAINTENANCE_OP: SNAPSHOT_PROP_MAINTENANCE_OP_VALUE,
        SNAPSHOT_PROP_REWRITE_ID: rewrite_id,
        SNAPSHOT_PROP_STRATEGY: strategy,
        SNAPSHOT_PROP_INPUT_FILES: str(len(input_paths)),
        SNAPSHOT_PROP_OUTPUT_FILES: str(len(all_data_files)),
        SNAPSHOT_PROP_CONSUMED_DELETES: str(len(delete_files_consumed)),
    }
    if batch_label is not None:
        snapshot_props[SNAPSHOT_PROP_BATCH] = batch_label

    def _success_result(snapshot_id: int) -> RewriteResult:
        return RewriteResult(
            strategy=strategy,
            rewritten_files=len(input_paths),
            added_files=len(all_data_files),
            bytes_rewritten=total_in,
            bytes_added=total_out,
            removed_delete_files=len(delete_files_consumed),
            failed_groups=0,
            commits=1,
            snapshot_ids=[int(snapshot_id)] if snapshot_id else [],
            rewrite_id=rewrite_id,
        )

    def _check_idempotent_replay(t: PyIcebergTable) -> RewriteResult | None:
        if batch_label is None:
            cached = _lookup_idempotent_result(t, rewrite_id, strategy, branch)
            if cached is not None and cached.commits >= 1:
                return cached
            return None
        existing = _find_batch_snapshot(t, rewrite_id, batch_label, branch)
        if existing is not None:
            return _success_result(int(existing.snapshot_id))
        return None

    def _attempt(_: int) -> RewriteResult:
        table.refresh()
        cached = _check_idempotent_replay(table)
        if cached is not None:
            return cached
        _validate_no_overlap(
            table,
            starting_snapshot_id=starting_snapshot_id,
            input_paths=input_paths,
            touched_partitions=touched_partitions,
            batch=batch,
            rewrite_id=rewrite_id,
            branch=branch,
            isolation=conflict_isolation,
        )
        from pyiceberg.table.refs import MAIN_BRANCH
        from pyiceberg.table.snapshots import Operation

        starting_sequence_number = (
            _starting_sequence_number(table, starting_snapshot_id) if use_starting_sequence_number else None
        )
        obsolete_deletes = _deletes_below_every_live_data_file(
            table,
            branch,
            removed_paths=set(input_paths),
            added_sequence_number=starting_sequence_number,
        )
        tx = table.transaction()
        producer = _compaction_producer_class()(
            operation=Operation.REPLACE,
            transaction=tx,
            io=table.io,
            commit_uuid=_uuid.uuid4(),
            snapshot_properties={**snapshot_props, SNAPSHOT_PROP_DROPPED_DELETES: str(len(obsolete_deletes))},
            branch=branch if branch is not None else MAIN_BRANCH,
            starting_sequence_number=starting_sequence_number,
        )
        for p in input_paths:
            producer.delete_data_file(plan_by_path[p].file)
        for delete_file in obsolete_deletes:
            producer.delete_data_file(delete_file)
        for df_ in all_data_files:
            producer.append_data_file(df_)
        producer.commit()
        tx.commit_transaction()
        table.refresh()
        snap = table.snapshot_by_name(branch) if branch is not None else table.current_snapshot()
        snapshot_id = int(snap.snapshot_id) if snap else 0
        return _success_result(snapshot_id)

    def _on_conflict(t: PyIcebergTable) -> RewriteResult | None:
        return _check_idempotent_replay(t)

    try:
        return (
            commit_with_retry(
                table,
                _attempt,
                op_name="rewrite_data_files",
                on_conflict=_on_conflict,
            ),
            None,
        )
    except CommitRetryExhausted as exc:
        return None, exc
    except RewriteConflict as exc:
        return None, exc


def _validate_no_overlap(
    table: PyIcebergTable,
    *,
    starting_snapshot_id: int | None,
    input_paths: list[str],
    touched_partitions: set[str],
    batch: list[_GroupOutput],
    rewrite_id: str,
    branch: str | None,
    isolation: str = CONFLICT_ISOLATION_SNAPSHOT,
) -> None:
    """Raise :class:`RewriteConflict` if writes since the plan snapshot affect this batch.

    Under ``serializable`` a foreign data file in a touched partition also refuses.
    """
    ancestry = _foreign_snapshots_since(table, starting_snapshot_id, rewrite_id, branch)
    foreign = ancestry.foreign

    if isolation != CONFLICT_ISOLATION_SNAPSHOT:
        for snapshot in foreign:
            for added in _added_files(snapshot, table):
                if int(added.content) != int(_DATA_CONTENT):
                    continue
                partition_key = stable_partition_key(added.partition)
                if partition_key in touched_partitions:
                    orphans = _orphan_output_paths(batch)
                    raise RewriteConflict(
                        f"snapshot {int(snapshot.snapshot_id)} added a data file in "
                        f"partition {partition_key!r} after the rewrite plan was "
                        f"taken; orphan outputs: {orphans!r}"
                    )

    _raise_if_new_deletes_apply(
        foreign,
        table,
        input_paths=input_paths,
        touched_partitions=touched_partitions,
        batch=batch,
    )
    if ancestry.reached_start:
        _raise_if_inputs_removed_since(foreign, table, input_paths=input_paths, batch=batch)
    else:
        _raise_if_inputs_vanished(table, input_paths, batch, branch)


@dataclass(frozen=True)
class _Ancestry:
    """The snapshots others committed since the plan, and whether the plan is still an ancestor."""

    foreign: list[Snapshot]
    reached_start: bool


def _branch_head(table: PyIcebergTable, branch: str | None) -> Snapshot | None:
    """Return the snapshot at the head of the reference being rewritten."""
    return table.snapshot_by_name(branch) if branch is not None else table.current_snapshot()


def _foreign_snapshots_since(
    table: PyIcebergTable,
    starting_snapshot_id: int | None,
    rewrite_id: str,
    branch: str | None,
) -> _Ancestry:
    """Return the snapshots others committed between the plan snapshot and the branch head.

    ``reached_start`` is false after a rollback or branch reset, when the plan snapshot is not met.
    """
    head = _branch_head(table, branch)
    if head is None or starting_snapshot_id is None:
        return _Ancestry(foreign=[], reached_start=starting_snapshot_id is None)
    if int(head.snapshot_id) == int(starting_snapshot_id):
        return _Ancestry(foreign=[], reached_start=True)

    ancestry: list[Snapshot] = []
    snapshot: Snapshot | None = head
    visited: set[int] = set()
    while snapshot is not None and int(snapshot.snapshot_id) != int(starting_snapshot_id):
        snapshot_id = int(snapshot.snapshot_id)
        if snapshot_id in visited:
            break
        visited.add(snapshot_id)
        if _snapshot_rewrite_id(snapshot) != rewrite_id:
            ancestry.append(snapshot)
        parent_id = getattr(snapshot, "parent_snapshot_id", None)
        snapshot = table.metadata.snapshot_by_id(parent_id) if parent_id is not None else None
    reached = snapshot is not None and int(snapshot.snapshot_id) == int(starting_snapshot_id)
    return _Ancestry(foreign=ancestry, reached_start=reached)


def _raise_if_new_deletes_apply(
    foreign: list[Snapshot],
    table: PyIcebergTable,
    *,
    input_paths: list[str],
    touched_partitions: set[str],
    batch: list[_GroupOutput],
) -> None:
    """Refuse the commit when a foreign delete applies to a file being replaced."""
    inputs = set(input_paths)
    for snapshot in foreign:
        for added in _added_files(snapshot, table):
            if int(added.content) == int(_DATA_CONTENT):
                continue
            if not _delete_may_apply(added, table, inputs, touched_partitions):
                continue
            orphans = _orphan_output_paths(batch)
            raise RewriteConflict(
                f"snapshot {int(snapshot.snapshot_id)} added the row-level delete "
                f"{added.file_path!r} covering a file this rewrite replaces; "
                f"committing would restore the rows it removed. Orphan outputs: {orphans!r}"
            )


def _referenced_data_file(delete_file: DataFile) -> str | None:
    """Return the one data file a position delete is scoped to, if it is scoped to one.

    Before format version 3 the fact is carried by the path column's bounds coinciding.
    """
    referenced = getattr(delete_file, "referenced_data_file", None)
    if referenced is not None:
        return str(referenced)
    bounds = _delete_path_bounds(delete_file)
    if bounds is not None and bounds[0] == bounds[1]:
        return bounds[0]
    return None


def _delete_may_apply(
    delete_file: DataFile,
    table: PyIcebergTable,
    input_paths: set[str],
    touched_partitions: set[str],
) -> bool:
    """Return whether a delete file can cover any of the data files being replaced."""
    from pyiceberg.manifest import DataFileContent

    if delete_file.content == DataFileContent.EQUALITY_DELETES:
        if table.specs()[int(delete_file.spec_id)].is_unpartitioned():
            return True
        return stable_partition_key(delete_file.partition) in touched_partitions

    referenced = _referenced_data_file(delete_file)
    if referenced is not None:
        return referenced in input_paths

    bounds = _delete_path_bounds(delete_file)
    if bounds is not None:
        lower, upper = bounds
        return any(lower <= path <= upper for path in input_paths)

    return stable_partition_key(delete_file.partition) in touched_partitions


def _delete_path_bounds(delete_file: DataFile) -> tuple[str, str] | None:
    """Return the path bounds a position delete records, or ``None`` when absent or undecodable."""
    lower_bounds = getattr(delete_file, "lower_bounds", None) or {}
    upper_bounds = getattr(delete_file, "upper_bounds", None) or {}
    lower_raw = lower_bounds.get(_DELETE_FILE_PATH_FIELD_ID)
    upper_raw = upper_bounds.get(_DELETE_FILE_PATH_FIELD_ID)
    if lower_raw is None or upper_raw is None:
        return None
    try:
        return bytes(lower_raw).decode("utf-8"), bytes(upper_raw).decode("utf-8")
    except UnicodeDecodeError:
        return None


def _raise_if_inputs_removed_since(
    foreign: list[Snapshot],
    table: PyIcebergTable,
    *,
    input_paths: list[str],
    batch: list[_GroupOutput],
) -> None:
    """Refuse the commit when a snapshot since the plan removed one of its inputs."""
    from pyiceberg.manifest import ManifestEntryStatus

    inputs = set(input_paths)
    for snapshot in foreign:
        removed = [
            data_file.file_path
            for data_file in _entries_of(snapshot, table, ManifestEntryStatus.DELETED)
            if data_file.file_path in inputs
        ]
        if removed:
            orphans = _orphan_output_paths(batch)
            raise RewriteConflict(
                f"snapshot {int(snapshot.snapshot_id)} removed input files {sorted(removed)!r} "
                f"after the rewrite plan was taken; orphan outputs: {orphans!r}"
            )


def _raise_if_inputs_vanished(
    table: PyIcebergTable,
    input_paths: list[str],
    batch: list[_GroupOutput],
    branch: str | None,
) -> None:
    """Refuse the commit when an input is no longer live on the branch being rewritten."""
    head = _branch_head(table, branch)
    scan = table.scan() if head is None else table.scan(snapshot_id=int(head.snapshot_id))
    live = {t.file.file_path for t in scan.plan_files()}
    missing = [p for p in input_paths if p not in live]
    if missing:
        orphans = _orphan_output_paths(batch)
        raise RewriteConflict(f"input files vanished before commit: {missing!r}; orphan outputs: {orphans!r}")


def _snapshot_rewrite_id(snapshot: Snapshot) -> str | None:
    """Return the rewrite id a snapshot's summary carries, if any."""
    summary = _summary_as_dict(snapshot.summary)
    return summary.get(SNAPSHOT_PROP_REWRITE_ID)


def _added_files(snapshot: Snapshot, table: PyIcebergTable) -> list[DataFile]:
    """Return the data and delete files ``snapshot`` added (status ADDED)."""
    from pyiceberg.manifest import ManifestEntryStatus

    return _entries_of(snapshot, table, ManifestEntryStatus.ADDED)


def _entries_of(snapshot: Snapshot, table: PyIcebergTable, status: ManifestEntryStatus) -> list[DataFile]:
    """Return the files whose manifest entries ``snapshot`` wrote with ``status``."""
    snapshot_id = int(snapshot.snapshot_id)
    try:
        manifests = snapshot.manifests(table.io)
    except OSError as exc:
        raise RewriteConflict(
            f"could not read the manifests of snapshot {snapshot_id} to check for "
            f"conflicting writes, so the rewrite cannot be shown to be safe: {exc}"
        ) from exc

    out: list[DataFile] = []
    for manifest in manifests:
        if int(getattr(manifest, "added_snapshot_id", -1)) != snapshot_id:
            continue
        try:
            entries = manifest.fetch_manifest_entry(table.io, discard_deleted=False)
        except OSError as exc:
            raise RewriteConflict(
                f"could not read manifest {manifest.manifest_path!r} of snapshot "
                f"{snapshot_id} to check for conflicting writes, so the rewrite "
                f"cannot be shown to be safe: {exc}"
            ) from exc
        out.extend(entry.data_file for entry in entries if int(entry.status) == int(status))
    return out


def _remove_dangling_deletes(
    table: PyIcebergTable,
    branch: str | None,
    *,
    rewrite_id: str,
    strategy: str,
) -> tuple[int, int | None]:
    """Drop the delete files no live data file can be covered by, in a replace snapshot of its own."""
    from pyiceberg.table.refs import MAIN_BRANCH
    from pyiceberg.table.snapshots import Operation

    def _replay(t: PyIcebergTable) -> tuple[int, int | None] | None:
        existing = _find_batch_snapshot(t, rewrite_id, DANGLING_BATCH_LABEL, branch)
        if existing is None:
            return None
        summary = _summary_as_dict(existing.summary)
        return int(summary.get(SNAPSHOT_PROP_DANGLING_REMOVED, 0)), int(existing.snapshot_id)

    def _attempt(_: int) -> tuple[int, int | None]:
        table.refresh()
        replayed = _replay(table)
        if replayed is not None:
            return replayed
        to_remove = _dangling_delete_files(table, branch)
        if not to_remove:
            return 0, None
        tx = table.transaction()
        producer = _compaction_producer_class()(
            operation=Operation.REPLACE,
            transaction=tx,
            io=table.io,
            commit_uuid=_uuid.uuid4(),
            snapshot_properties={
                SNAPSHOT_PROP_MAINTENANCE_OP: SNAPSHOT_PROP_MAINTENANCE_OP_VALUE,
                SNAPSHOT_PROP_REWRITE_ID: rewrite_id,
                SNAPSHOT_PROP_STRATEGY: strategy,
                SNAPSHOT_PROP_BATCH: DANGLING_BATCH_LABEL,
                SNAPSHOT_PROP_DANGLING_REMOVED: str(len(to_remove)),
            },
            branch=branch if branch is not None else MAIN_BRANCH,
            starting_sequence_number=None,
        )
        for delete_file in to_remove:
            producer.delete_data_file(delete_file)
        producer.commit()
        tx.commit_transaction()
        table.refresh()
        head = _branch_head(table, branch)
        return len(to_remove), int(head.snapshot_id) if head is not None else None

    return commit_with_retry(
        table,
        _attempt,
        op_name="rewrite_data_files",
        on_conflict=_replay,
    )


def _deletes_below_every_live_data_file(
    table: PyIcebergTable,
    branch: str | None,
    *,
    removed_paths: set[str],
    added_sequence_number: int | None,
) -> list[DataFile]:
    """Return the live delete files older than every data file the commit leaves live.

    A delete applies only to data files at or below its sequence number.
    """
    from pyiceberg.manifest import DataFileContent

    # Only format version 2 and later carry sequence numbers and delete files.
    if table.metadata.format_version < 2:
        return []
    head = _branch_head(table, branch)
    if head is None:
        return []
    from pyiceberg.manifest import ManifestContent

    # Without a delete manifest there is nothing to shed.
    manifests = head.manifests(table.io)
    if all(manifest.content != ManifestContent.DELETES for manifest in manifests):
        return []

    min_live = int(table.metadata.last_sequence_number)
    if added_sequence_number is not None:
        min_live = min(min_live, int(added_sequence_number))
    live_deletes: list[tuple[DataFile, int]] = []
    for manifest in manifests:
        for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=True):
            data_file = entry.data_file
            seq = int(entry.sequence_number) if entry.sequence_number is not None else 0
            if data_file.content == DataFileContent.DATA:
                if data_file.file_path not in removed_paths:
                    min_live = min(min_live, seq)
            else:
                live_deletes.append((data_file, seq))

    return [delete_file for delete_file, seq in live_deletes if 0 < seq < min_live]


def _dangling_delete_files(table: PyIcebergTable, branch: str | None) -> list[DataFile]:
    """Return the live delete files no live data file can be covered by."""
    from pyiceberg.manifest import DataFileContent

    snap = _branch_head(table, branch)
    if snap is None:
        return []

    specs = table.specs()
    min_data_seq: dict[tuple[int, str], int] = {}
    delete_entries: list[tuple[DataFile, int, tuple[int, str]]] = []
    for manifest in snap.manifests(table.io):
        for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=True):
            data_file = entry.data_file
            seq = entry.sequence_number if entry.sequence_number is not None else 0
            key = (int(data_file.spec_id), stable_partition_key(data_file.partition))
            if data_file.content == DataFileContent.DATA:
                cur = min_data_seq.get(key)
                if cur is None or seq < cur:
                    min_data_seq[key] = seq
            else:
                delete_entries.append((data_file, seq, key))
    table_min = min(min_data_seq.values()) if min_data_seq else None

    dangling: list[DataFile] = []
    for delete_file, seq, key in delete_entries:
        is_global = delete_file.content == DataFileContent.EQUALITY_DELETES and specs[key[0]].is_unpartitioned()
        min_seq = table_min if is_global else min_data_seq.get(key)
        if min_seq is None or _delete_is_dangling(delete_file.content, seq, min_seq):
            dangling.append(delete_file)
    return dangling


def _delete_is_dangling(content: DataFileContent, sequence_number: int, min_data_sequence_number: int) -> bool:
    """Return whether a delete at ``sequence_number`` can apply to no live data file.

    A position delete still covers a data file at its own sequence number; an equality delete does not.
    """
    from pyiceberg.manifest import DataFileContent

    if content == DataFileContent.POSITION_DELETES:
        return sequence_number < min_data_sequence_number
    return sequence_number <= min_data_sequence_number


def _orphan_output_paths(batch: list[_GroupOutput]) -> list[str]:
    """Return the paths of every file the batch wrote."""
    out: list[str] = []
    for o in batch:
        for df_ in o.data_files:
            path = getattr(df_, "file_path", None)
            if path:
                out.append(str(path))
    return out
