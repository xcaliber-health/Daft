"""Shared primitives for row-level writes.

Row-level writes replace or delete individual rows rather than whole files. They
all start from the same two things: a stable name for every data file the
operation may touch, and a scan that tells each row which file and position it
came from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from daft.io.iceberg._common import ROW_FILE_COLUMN, ROW_POSITION_COLUMN
from daft.io.iceberg._deletes import read_with_deletes, stable_partition_key
from daft.io.iceberg.iceberg_write import partition_field_to_expr

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from pyiceberg.expressions import BooleanExpression
    from pyiceberg.manifest import DataFile
    from pyiceberg.table import Table as PyIcebergTable
    from pyiceberg.table.snapshots import Operation, Snapshot
    from pyiceberg.typedef import Record

    from daft.daft import IOConfig
    from daft.dataframe import DataFrame
    from daft.io.iceberg._deletes import ScanPlan
    from daft.io.writer import IcebergPositionDeleteWriter


@dataclass(frozen=True)
class FileEntry:
    """One data file a row-level operation reads, and may rewrite or delete from."""

    path: str
    spec_id: int
    partition: Record
    sequence_number: int
    data_file: DataFile


class FileTable:
    """Stable indices for the data files of one row-level operation.

    Rows carry an index rather than a path so provenance stays two narrow columns
    through the join and the write. The path, partition spec and partition values
    are resolved here, at the point where files are written or committed.

    Indices come from the plan, not from the order files happen to be scanned, so
    the same file keeps its index across the passes of one operation and across a
    retry that re-reads a subset.
    """

    def __init__(self, entries: Sequence[FileEntry]) -> None:
        self._entries = tuple(entries)
        self._by_path = {entry.path: index for index, entry in enumerate(self._entries)}

    @classmethod
    def from_plan(cls, plan: ScanPlan) -> FileTable:
        """Build a table over every data file the plan selected.

        Files of one partition sit together, so grouping removals by partition
        never has to look outside a run of neighbouring indices.
        """
        entries = [
            FileEntry(
                path=path,
                spec_id=int(plan.tasks[path].file.spec_id),
                partition=plan.tasks[path].file.partition,
                sequence_number=plan.sequence_by_path[path],
                data_file=plan.tasks[path].file,
            )
            for path in plan.tasks
        ]
        entries.sort(key=lambda entry: (entry.spec_id, stable_partition_key(entry.partition), entry.path))
        return cls(entries)

    def delete_groups(self, granularity: str) -> list[int]:
        """Return the delete file each index belongs to, as a non-decreasing group number.

        Under file granularity every data file gets its own delete file, which is
        what keeps a scan from reading removals for files it is not reading.
        Under partition granularity one delete file covers a whole partition.
        """
        if granularity == DELETE_GRANULARITY_FILE:
            return list(range(len(self._entries)))
        if granularity != DELETE_GRANULARITY_PARTITION:
            raise ValueError(f"unknown delete granularity {granularity!r}")
        groups: list[int] = []
        seen: dict[tuple[int, str], int] = {}
        for entry in self._entries:
            key = (entry.spec_id, stable_partition_key(entry.partition))
            groups.append(seen.setdefault(key, len(seen)))
        return groups

    @property
    def indices(self) -> Mapping[str, int]:
        """Index of each file, by path."""
        return self._by_path

    @property
    def paths(self) -> tuple[str, ...]:
        """Every file path, in index order."""
        return tuple(entry.path for entry in self._entries)

    def entry(self, index: int) -> FileEntry:
        """Return the file at ``index``.

        Raises:
        ------
        IndexError
            If ``index`` names no file, which means a row carried provenance from
            a different plan than the one being committed.
        """
        if index < 0 or index >= len(self._entries):
            raise IndexError(f"file index {index} is outside the {len(self._entries)} planned file(s)")
        return self._entries[index]

    def index_of(self, path: str) -> int:
        """Return the index of ``path``.

        Raises:
        ------
        KeyError
            If the plan did not select ``path``.
        """
        if path not in self._by_path:
            raise KeyError(f"{path} is not one of the {len(self._entries)} planned file(s)")
        return self._by_path[path]

    def __len__(self) -> int:
        return len(self._entries)


def provenance_scan(
    *,
    table: PyIcebergTable,
    plan: ScanPlan,
    file_table: FileTable,
    paths: Sequence[str],
    io_config: IOConfig,
    snapshot_id: int | None = None,
    ignore_corrupt_files: bool = False,
) -> DataFrame:
    """Return a lazy frame over ``paths`` with their deletes applied and provenance attached.

    Every row carries the index of the file it came from and its position in that
    file. Positions count rows as the file stores them, so a row deleted earlier
    shifts nothing: the position still names the row it always named.

    Selecting columns from the returned frame prunes the read, so a caller that
    needs only the join keys pays for only those columns.
    """
    return read_with_deletes(
        table=table,
        plan=plan,
        paths=list(paths),
        snapshot_id=snapshot_id,
        io_config=io_config,
        schema_source="current",
        ignore_corrupt_files=ignore_corrupt_files,
        file_indices=file_table.indices,
    )


def provenance_columns() -> tuple[str, str]:
    """Return the names of the file-index and position columns, in that order."""
    return ROW_FILE_COLUMN, ROW_POSITION_COLUMN


#: How a table writes the rows an operation changes.
COPY_ON_WRITE = "copy-on-write"
MERGE_ON_READ = "merge-on-read"
_WRITE_MODES = (COPY_ON_WRITE, MERGE_ON_READ)

#: What a concurrent commit is allowed to have changed underneath an operation.
SERIALIZABLE = "serializable"
SNAPSHOT_ISOLATION = "snapshot"
_ISOLATION_LEVELS = (SERIALIZABLE, SNAPSHOT_ISOLATION)

#: How many data files one delete file may cover.
DELETE_GRANULARITY_FILE = "file"
DELETE_GRANULARITY_PARTITION = "partition"

#: How rows are spread over the files a write produces.
DISTRIBUTION_NONE = "none"
DISTRIBUTION_HASH = "hash"
DISTRIBUTION_RANGE = "range"
_DISTRIBUTION_MODES = (DISTRIBUTION_NONE, DISTRIBUTION_HASH, DISTRIBUTION_RANGE)

_MODE_KEYS = {"merge": "write.merge.mode", "update": "write.update.mode", "delete": "write.delete.mode"}
_ISOLATION_KEYS = {
    "merge": "write.merge.isolation-level",
    "update": "write.update.isolation-level",
    "delete": "write.delete.isolation-level",
}
_DISTRIBUTION_KEYS = {
    "merge": "write.merge.distribution-mode",
    "update": "write.update.distribution-mode",
    "delete": "write.delete.distribution-mode",
}
_TABLE_DISTRIBUTION_KEY = "write.distribution-mode"
_DELETE_GRANULARITY_KEY = "write.delete.granularity"
_DELETE_TARGET_FILE_SIZE_KEY = "write.delete.target-file-size-bytes"
#: One delete file per data file, which is what the query engines default to even
#: though the property's own default covers a whole partition.
_DEFAULT_DELETE_GRANULARITY = DELETE_GRANULARITY_FILE
_DEFAULT_DELETE_TARGET_FILE_SIZE = 64 * 1024 * 1024

#: Snapshot properties a row-level write records.
SNAPSHOT_PROP_OPERATION = "daft.row-level.op"
SNAPSHOT_PROP_MERGE_ID = "daft.merge-id"
SNAPSHOT_PROP_MODE = "daft.row-level.mode"
MERGE_COUNT_KEYS = {
    "copied": "daft.merge-into.num-target-rows-copied",
    "deleted": "daft.merge-into.num-target-rows-deleted",
    "updated": "daft.merge-into.num-target-rows-updated",
    "inserted": "daft.merge-into.num-target-rows-inserted",
}
UPDATE_COUNT_KEYS = {
    "updated": "daft.update.num-updated-rows",
    "copied": "daft.update.num-copied-rows",
}
DELETE_COUNT_KEYS = {
    "deleted": "daft.delete.num-deleted-rows",
    "copied": "daft.delete.num-copied-rows",
}

#: Columns the merge operator adds to every row it emits.
ACTION_COLUMN = "__daft_merge_action"
TARGET_PRESENT_COLUMN = "__daft_target_present"
SOURCE_PRESENT_COLUMN = "__daft_source_present"

#: What the merge decided for a row, as written to the action column.
ACTION_KEEP = 0
ACTION_UPDATE = 1
ACTION_DELETE = 2
ACTION_INSERT = 3


class RowLevelConflict(RuntimeError):
    """Raised when another writer changed what an operation planned against."""


def resolve_mode(properties: Mapping[str, str], operation: str) -> str:
    """Return whether ``operation`` rewrites whole files or records removed rows.

    Raises:
    ------
    ValueError
        If the table asks for a mode that does not exist.
    """
    mode = str(properties.get(_MODE_KEYS[operation], COPY_ON_WRITE))
    if mode not in _WRITE_MODES:
        raise ValueError(f"{_MODE_KEYS[operation]} must be one of {_WRITE_MODES}, got {mode!r}")
    return mode


def resolve_isolation(properties: Mapping[str, str], operation: str, override: str | None = None) -> str:
    """Return what a concurrent commit may change under ``operation``.

    Raises:
    ------
    ValueError
        If the requested level does not exist.
    """
    level = str(override if override is not None else properties.get(_ISOLATION_KEYS[operation], SERIALIZABLE))
    if level not in _ISOLATION_LEVELS:
        raise ValueError(f"isolation level must be one of {_ISOLATION_LEVELS}, got {level!r}")
    return level


def resolve_delete_granularity(properties: Mapping[str, str]) -> str:
    """Return how many data files one delete file may cover.

    Raises:
    ------
    ValueError
        If the table asks for a granularity that does not exist.
    """
    granularity = str(properties.get(_DELETE_GRANULARITY_KEY, _DEFAULT_DELETE_GRANULARITY))
    if granularity not in (DELETE_GRANULARITY_FILE, DELETE_GRANULARITY_PARTITION):
        raise ValueError(f"{_DELETE_GRANULARITY_KEY} must be 'file' or 'partition', got {granularity!r}")
    return granularity


def resolve_delete_target_size(properties: Mapping[str, str]) -> int:
    """Return the size a delete file is rolled at, in bytes."""
    raw = properties.get(_DELETE_TARGET_FILE_SIZE_KEY)
    if raw is None:
        return _DEFAULT_DELETE_TARGET_FILE_SIZE
    size = int(raw)
    if size <= 0:
        raise ValueError(f"{_DELETE_TARGET_FILE_SIZE_KEY} must be positive, got {size}")
    return size


def resolve_distribution(properties: Mapping[str, str], operation: str, mode: str, partitioned: bool) -> str:
    """Return how rows are spread over the files a write produces.

    A merge that rewrites whole files is laid out like any other write, so an
    unpartitioned table leaves the rows where they are. Everything else groups
    rows by what they are written into, which keeps a writer from holding one
    open file per partition.

    Raises:
    ------
    ValueError
        If the requested distribution does not exist.
    """
    configured = properties.get(_DISTRIBUTION_KEYS[operation])
    if configured is None and operation == "merge" and mode == COPY_ON_WRITE:
        configured = DISTRIBUTION_HASH if partitioned else properties.get(_TABLE_DISTRIBUTION_KEY, DISTRIBUTION_NONE)
    distribution = str(configured if configured is not None else DISTRIBUTION_HASH)
    if distribution not in _DISTRIBUTION_MODES:
        raise ValueError(f"distribution mode must be one of {_DISTRIBUTION_MODES}, got {distribution!r}")
    return distribution


class DeleteWriterOpener:
    """Opens the delete file for one group of removed rows.

    A delete file belongs to the partition of the data files it names, so the
    group is located from the index of one of them.
    """

    def __init__(self, table: PyIcebergTable, file_table: FileTable, io_config: IOConfig) -> None:
        self._table = table
        self._file_table = file_table
        self._io_config = io_config
        self._location = table.properties.get("write.data.path", f"{table.location()}/data")

    def __call__(self, file_idx: int, file_index: int) -> IcebergPositionDeleteWriter:
        """Return a writer for the group containing the file at ``file_index``."""
        from daft.io.writer import IcebergPositionDeleteWriter
        from daft.recordbatch.recordbatch import RecordBatch

        entry = self._file_table.entry(file_index)
        fields = self._table.specs()[entry.spec_id].fields
        partition_values = (
            RecordBatch.from_pydict({field.name: [entry.partition[i]] for i, field in enumerate(fields)})
            if fields
            else None
        )
        return IcebergPositionDeleteWriter(
            root_dir=self._location,
            file_idx=file_idx,
            properties=dict(self._table.properties),
            partition_spec_id=entry.spec_id,
            partition_values=partition_values,
            io_config=self._io_config,
        )


@dataclass(frozen=True)
class WrittenFiles:
    """What a row-level write produced, and how many rows each decision covered."""

    data_files: list[DataFile]
    delete_files: list[DataFile]
    rows_kept: int
    rows_updated: int
    rows_deleted: int
    rows_inserted: int


def collect_written_files(frame: DataFrame) -> WrittenFiles:
    """Run a row-level write and gather what it produced.

    Counts are summed rather than taken from one row, because a distributed run
    reports what each of its tasks decided.
    """
    result = frame.to_pydict()

    def total(column: str) -> int:
        return sum(int(value) for value in result.get(column, []) if value is not None)

    return WrittenFiles(
        data_files=[file for file in result.get("data_file", []) if file is not None],
        delete_files=[file for file in result.get("delete_file", []) if file is not None],
        rows_kept=total("rows_kept"),
        rows_updated=total("rows_updated"),
        rows_deleted=total("rows_deleted"),
        rows_inserted=total("rows_inserted"),
    )


def distribute(frame: DataFrame, table: PyIcebergTable, distribution: str, mode: str) -> DataFrame:
    """Spread rows over the files the write will produce.

    Grouping rows by where they land keeps a writer from holding one open file
    per partition. Ordering them instead also clusters the output, which is what
    a sorted table asks for. On a single machine neither costs a shuffle.
    """
    from daft import runners
    from daft.expressions import col

    if distribution == DISTRIBUTION_NONE:
        return frame
    schema = table.schema()
    # One machine writes one stream, so grouping rows first would only move them.
    if distribution == DISTRIBUTION_HASH and runners.get_or_create_runner().name != "ray":
        return frame
    partition_exprs = [partition_field_to_expr(field, schema) for field in table.spec().fields]
    if distribution == DISTRIBUTION_HASH:
        if partition_exprs:
            return frame.repartition(None, *partition_exprs)
        # Rows removed from one file are written together, which an unpartitioned
        # table cannot express as a partition value.
        return frame.repartition(None, col(ROW_FILE_COLUMN)) if mode == MERGE_ON_READ else frame
    order = table.sort_order()
    if not order.fields:
        raise ValueError("a range distribution needs the table to declare a sort order")
    names = [schema.find_field(field.source_id).name for field in order.fields]
    return frame.sort([col(name) for name in names])


def row_level_producer_class() -> type:
    """Return the snapshot producer a row-level write commits through.

    It inherits the commit machinery a rewrite uses, and differs in two ways:
    the files it adds take the new snapshot's sequence number, so removals
    recorded now apply to the rows they name and older removals do not apply to
    rows written now; and the snapshot keeps the operation it was asked for
    rather than being labelled a rewrite.
    """
    from pyiceberg.table.snapshots import Summary

    from daft.io.iceberg._compact import _compaction_producer_class

    class _RowLevelProducer(_compaction_producer_class()):  # type: ignore[misc]
        """Commits a row-level write: some rows replaced, some removed, some added."""

        def _added_entry_sequence_number_for(self, data_file: DataFile) -> int | None:
            """Return ``None`` so every added file takes this commit's sequence number."""
            del data_file
            return None

        def _summary(self, snapshot_properties: dict[str, str] | None = None) -> Summary:
            summary = super()._summary(snapshot_properties)
            return Summary(operation=self._operation, **summary.additional_properties)

    return _RowLevelProducer


def _live_removed_since(table: PyIcebergTable, foreign: Sequence[Snapshot], touched: set[str]) -> list[str]:
    """Return the touched files a later snapshot removed."""
    from pyiceberg.manifest import DataFileContent, ManifestEntryStatus

    gone: list[str] = []
    for snapshot in foreign:
        for manifest in snapshot.manifests(table.io):
            for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=False):
                if (
                    entry.status == ManifestEntryStatus.DELETED
                    and entry.data_file.content == DataFileContent.DATA
                    and str(entry.data_file.file_path) in touched
                ):
                    gone.append(str(entry.data_file.file_path))
    return gone


def _deletes_added_since(
    table: PyIcebergTable, foreign: Sequence[Snapshot], touched: set[str], partitions: set[str]
) -> bool:
    """Return whether a later snapshot recorded removals covering a touched file."""
    from pyiceberg.manifest import DataFileContent, ManifestContent, ManifestEntryStatus

    from daft.io.iceberg._compact import _delete_may_apply

    for snapshot in foreign:
        for manifest in snapshot.manifests(table.io):
            if manifest.content != ManifestContent.DELETES:
                continue
            for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=True):
                if entry.status == ManifestEntryStatus.DELETED:
                    continue
                if entry.data_file.content == DataFileContent.DATA:
                    continue
                if _delete_may_apply(entry.data_file, table, touched, partitions):
                    return True
    return False


def _data_added_since(table: PyIcebergTable, foreign: Sequence[Snapshot], predicate: BooleanExpression) -> bool:
    """Return whether a later snapshot added rows the operation would have acted on."""
    from pyiceberg.expressions.visitors import _InclusiveMetricsEvaluator
    from pyiceberg.manifest import DataFileContent, ManifestContent, ManifestEntryStatus

    evaluate = _InclusiveMetricsEvaluator(table.schema(), predicate, case_sensitive=True).eval
    for snapshot in foreign:
        for manifest in snapshot.manifests(table.io):
            if manifest.content != ManifestContent.DATA:
                continue
            for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=True):
                if entry.status != ManifestEntryStatus.ADDED:
                    continue
                if entry.data_file.content == DataFileContent.DATA and evaluate(entry.data_file):
                    return True
    return False


def validate_row_level_commit(
    table: PyIcebergTable,
    *,
    starting_snapshot_id: int | None,
    merge_id: str,
    branch: str | None,
    isolation: str,
    checks_deletes: bool,
    touched_paths: Sequence[str],
    touched_partitions: set[str],
    conflict_filter: BooleanExpression,
) -> None:
    """Refuse a commit whose assumptions no longer hold.

    An operation reads a snapshot, decides what to change and commits later. What
    it decided stays true only if the files it read are still there, and, for an
    operation that replaces rows rather than removing whole files, only if no one
    else has recorded removals against those rows in the meantime. Under the
    strictest level it must also be true that no rows were added that the
    operation would have acted on had it seen them.

    Raises:
    ------
    RowLevelConflict
        When another writer invalidated what this operation planned.
    """
    from daft.io.iceberg._compact import _foreign_snapshots_since

    ancestry = _foreign_snapshots_since(table, starting_snapshot_id, merge_id, branch)
    if not ancestry.reached_start:
        raise RowLevelConflict("the snapshot this operation planned against is no longer on the branch")
    if not ancestry.foreign:
        return

    touched = {str(path) for path in touched_paths}
    if touched:
        gone = _live_removed_since(table, ancestry.foreign, touched)
        if gone:
            raise RowLevelConflict(f"{len(gone)} file(s) this operation read were replaced, first: {gone[0]}")
        if checks_deletes and _deletes_added_since(table, ancestry.foreign, touched, touched_partitions):
            raise RowLevelConflict("another writer removed rows from a file this operation is rewriting")
    if isolation == SERIALIZABLE and _data_added_since(table, ancestry.foreign, conflict_filter):
        raise RowLevelConflict(
            "another writer added rows this operation would have covered; "
            "retry, or set the isolation level to 'snapshot' to allow it"
        )


def find_replayed_snapshot(table: PyIcebergTable, merge_id: str, branch: str | None) -> Snapshot | None:
    """Return the snapshot an earlier run of this operation committed, if any."""
    from daft.io.iceberg._common import branch_ancestry

    for snapshot in reversed(branch_ancestry(table, branch)):
        summary = snapshot.summary
        if summary is not None and summary.additional_properties.get(SNAPSHOT_PROP_MERGE_ID) == merge_id:
            return snapshot
    return None


def commit_row_level(
    table: PyIcebergTable,
    *,
    operation: Operation,
    snapshot_properties: Mapping[str, str],
    added_data_files: Sequence[DataFile],
    added_delete_files: Sequence[DataFile],
    removed_files: Sequence[DataFile],
    validate: Callable[[], None],
    replay_key: str,
    branch: str | None,
    op_name: str,
) -> int:
    """Commit what a row-level write produced, retrying while another writer wins the race.

    Returns the snapshot the change landed in. A run that already committed under
    the same key returns that snapshot instead of committing a second time, so a
    caller that retries after losing the answer does not apply the change twice.
    """
    import uuid as _uuid

    from pyiceberg.table.refs import MAIN_BRANCH

    from daft.io.iceberg._common import commit_with_retry

    def _attempt(_: int) -> int:
        table.refresh()
        replayed = find_replayed_snapshot(table, replay_key, branch)
        if replayed is not None:
            return int(replayed.snapshot_id)
        validate()
        transaction = table.transaction()
        producer = row_level_producer_class()(
            operation=operation,
            transaction=transaction,
            io=table.io,
            commit_uuid=_uuid.uuid4(),
            snapshot_properties=dict(snapshot_properties),
            branch=branch if branch is not None else MAIN_BRANCH,
            starting_sequence_number=None,
        )
        for data_file in removed_files:
            producer.delete_data_file(data_file)
        for data_file in (*added_data_files, *added_delete_files):
            producer.append_data_file(data_file)
        producer.commit()
        transaction.commit_transaction()
        table.refresh()
        head = table.snapshot_by_name(branch) if branch is not None else table.current_snapshot()
        if head is None:
            raise RowLevelConflict("the branch has no snapshot after committing")
        return int(head.snapshot_id)

    return commit_with_retry(table, _attempt, op_name=op_name)


def discard_files(table: PyIcebergTable, files: Sequence[DataFile]) -> None:
    """Remove files no snapshot will reference, after an attempt was refused."""
    for data_file in files:
        try:
            table.io.delete(str(data_file.file_path))
        except OSError as exc:
            logger.warning("row-level write: could not remove %s: %s", data_file.file_path, exc)
