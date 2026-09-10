"""Rewrite of position delete files into fewer files at a target size.

Position deletes accumulate one small file per commit. This operation packs
the live position delete files of each partition, drops the rows that name a
data file no longer live, and commits the packed files in place of the old
ones at the sequence number the old ones carried, so exactly the same rows
stay deleted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import uuid as _uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from daft.io.iceberg._common import (
    CommitRetryExhausted,
    MaintenanceOptions,
    branch_ancestry,
    commit_with_retry,
    option_bool,
    option_int,
    scalar_options,
)
from daft.io.iceberg._compact import (
    SNAPSHOT_PROP_BATCH,
    SNAPSHOT_PROP_INPUT_FILES,
    SNAPSHOT_PROP_MAINTENANCE_OP,
    SNAPSHOT_PROP_OUTPUT_FILES,
    SNAPSHOT_PROP_REWRITE_ID,
    SNAPSHOT_PROP_STRATEGY,
    RewriteConflict,
    _compaction_producer_class,
    _io_config_for_table,
    _rewrite_batches,
    _writes_one_file_per_partition,
)
from daft.io.iceberg._deletes import stable_partition_key
from daft.io.iceberg._rewrite_manifests import _summary_as_dict

if TYPE_CHECKING:
    import pyarrow as pa
    from pyiceberg.expressions import BooleanExpression
    from pyiceberg.manifest import DataFile
    from pyiceberg.table import DataScan
    from pyiceberg.table import Table as PyIcebergTable
    from pyiceberg.typedef import Record

    from daft.daft import IOConfig
    from daft.daft._iceberg import CandidateRecord, FileGroupRecord, OptionValue
    from daft.dataframe import DataFrame
    from daft.io.writer import IcebergWriter

logger = logging.getLogger(__name__)

SNAPSHOT_PROP_MAINTENANCE_OP_VALUE = "rewrite-position-delete-files"
SNAPSHOT_PROP_STRATEGY_VALUE = "position-deletes"
SNAPSHOT_PROP_INPUT_BYTES = "daft.rewrite-input-bytes"
SNAPSHOT_PROP_OUTPUT_BYTES = "daft.rewrite-output-bytes"

DELETE_TARGET_FILE_SIZE_KEY = "write.delete.target-file-size-bytes"
_DEFAULT_DELETE_TARGET_FILE_SIZE = 64 * 1024 * 1024
DELETE_GRANULARITY_KEY = "write.delete.granularity"
#: One packed file per data file unless the table asks for one per partition.
_DEFAULT_DELETE_GRANULARITY = "file"
_GRANULARITIES = ("file", "partition")
#: Field ids the format reserves for a position delete's columns.
_DELETE_FILE_PATH_FIELD_ID = 2147483546
_DELETE_POS_FIELD_ID = 2147483545

_PLANNER_OPTIONS = frozenset(
    {
        "target-file-size-bytes",
        "min-file-size-bytes",
        "max-file-size-bytes",
        "min-input-files",
        "rewrite-all",
        "max-file-group-size-bytes",
        "rewrite-job-order",
    }
)
_SUPPORTED_OPTIONS = _PLANNER_OPTIONS | {
    "partial-progress.enabled",
    "partial-progress.max-commits",
    "max-concurrent-file-group-rewrites",
    "rewrite-id",
}


@dataclass(frozen=True)
class RewritePositionDeletesResult:
    """Outcome of one position delete rewrite.

    Attributes:
    ----------
    rewritten_delete_files
        Delete files replaced.
    added_delete_files
        Delete files written.
    bytes_rewritten
        Bytes of the replaced files.
    bytes_added
        Bytes of the written files.
    rewrite_id
        Identity of the run, recorded on every snapshot it commits.
    snapshot_ids
        Snapshots committed, in commit order.
    commits
        Commits that landed.
    failed_groups
        File groups that could not be rewritten or committed.
    """

    rewritten_delete_files: int
    added_delete_files: int
    bytes_rewritten: int
    bytes_added: int
    rewrite_id: str
    snapshot_ids: list[int]
    commits: int
    failed_groups: int


class RewritePositionDeletesFailedException(RuntimeError):
    """A position delete rewrite left no committed result for a group it planned."""


@dataclass(frozen=True)
class _DeleteInput:
    """One live position delete file and the coordinates it applies within."""

    data_file: DataFile
    sequence_number: int
    spec_id: int
    partition_key: str

    @property
    def path(self) -> str:
        return str(self.data_file.file_path)


@dataclass(frozen=True)
class _Plan:
    """The live delete files, the live data files they may reference, and how to group them."""

    deletes: dict[str, _DeleteInput]
    live_data_paths: dict[tuple[int, str], set[str]]
    groups: list[FileGroupRecord]


@dataclass(frozen=True)
class _GroupOutput:
    """The delete files one group replaced and wrote."""

    input_paths: list[str]
    sequence_number: int
    delete_files: list[DataFile]
    bytes_rewritten: int
    bytes_added: int


def run(
    table: PyIcebergTable,
    *,
    where: str | BooleanExpression | None = None,
    branch: str | None = None,
    options: MaintenanceOptions | None = None,
) -> RewritePositionDeletesResult:
    """Pack the live position delete files of each partition into target-sized files.

    Parameters
    ----------
    table
        Table whose delete files are rewritten.
    where
        Row filter whose partition projection selects the partitions to work on.
    branch
        Branch to commit to; defaults to the main branch.
    options
        ``target-file-size-bytes``, ``min-file-size-bytes``,
        ``max-file-size-bytes``, ``min-input-files``, ``rewrite-all``,
        ``max-file-group-size-bytes``, ``rewrite-job-order``,
        ``partial-progress.enabled``, ``partial-progress.max-commits``,
        ``max-concurrent-file-group-rewrites`` and ``rewrite-id``.

    Returns:
    -------
    RewritePositionDeletesResult
        Counts of files and bytes replaced and written, with the snapshots committed.

    Raises:
    ------
    ValueError
        If an option is unknown or out of range.
    RewritePositionDeletesFailedException
        If a group cannot be rewritten or committed and partial progress is off.
    """
    from pyiceberg.expressions import AlwaysTrue

    from daft.daft import _iceberg as _rust_iceberg

    opts: dict[str, OptionValue] = scalar_options(options)
    unknown = sorted(set(opts) - _SUPPORTED_OPTIONS)
    if unknown:
        raise ValueError(f"rewrite_position_delete_files: unsupported option(s) {unknown}")
    planner_options = {key: value for key, value in opts.items() if key in _PLANNER_OPTIONS}
    planner_options.setdefault(
        "target-file-size-bytes",
        int(table.properties.get(DELETE_TARGET_FILE_SIZE_KEY, _DEFAULT_DELETE_TARGET_FILE_SIZE)),
    )
    normalized = _rust_iceberg.validate_options_py(planner_options)
    partial_progress = option_bool(opts, "partial-progress.enabled", False)
    max_commits = option_int(opts, "partial-progress.max-commits", 10)
    if max_commits < 1:
        raise ValueError(f"partial-progress.max-commits must be >= 1, got {max_commits!r}")
    max_concurrent = option_int(opts, "max-concurrent-file-group-rewrites", 5)
    if max_concurrent < 1:
        raise ValueError(f"max-concurrent-file-group-rewrites must be >= 1, got {max_concurrent!r}")
    explicit_rewrite_id = opts.get("rewrite-id")
    granularity = str(table.properties.get(DELETE_GRANULARITY_KEY, _DEFAULT_DELETE_GRANULARITY)).lower()
    if granularity not in _GRANULARITIES:
        raise ValueError(f"{DELETE_GRANULARITY_KEY} must be one of {_GRANULARITIES}, got {granularity!r}")

    snapshot = table.snapshot_by_name(branch) if branch is not None else table.current_snapshot()
    if snapshot is None or table.metadata.format_version < 2:
        return _empty_result(str(explicit_rewrite_id or ""))
    snapshot_id = int(snapshot.snapshot_id)

    row_filter = where if where is not None else AlwaysTrue()
    plan = _plan(table, table.scan(row_filter=row_filter, snapshot_id=snapshot_id), normalized)
    rewrite_id = (
        str(explicit_rewrite_id)
        if explicit_rewrite_id is not None
        else _digest(table, branch, normalized, sorted(plan.deletes))
    )
    cached = _lookup_result(table, rewrite_id, branch)
    if cached is not None:
        logger.info("rewrite_position_delete_files: replay of rewrite_id=%s", rewrite_id)
        return cached
    if not plan.groups:
        return _empty_result(rewrite_id)

    io_config = _io_config_for_table(table)

    def _rewrite_one(group: FileGroupRecord) -> _GroupOutput:
        return _rewrite_group(table, plan, group, io_config, per_file=granularity == "file")

    batches = _batches(plan.groups, max_commits if partial_progress else 1)
    concurrency = 1 if _writes_one_file_per_partition() else max_concurrent
    committed: list[RewritePositionDeletesResult] = []
    failed_groups = 0
    for index, outputs, failures in _rewrite_batches(batches, _rewrite_one, max_concurrent=concurrency):
        failed_groups += len(failures)
        if failures and not partial_progress:
            _abort(table, outputs)
            raise RewritePositionDeletesFailedException(
                "rewrite_position_delete_files: a group could not be rewritten"
            ) from failures[0][1]
        if not outputs:
            continue
        label = f"batch-{index}" if partial_progress else None
        try:
            committed.append(_commit_batch(table, outputs, rewrite_id, label, branch))
        except (CommitRetryExhausted, RewriteConflict) as exc:
            _abort(table, outputs)
            failed_groups += len(outputs)
            if not partial_progress:
                raise RewritePositionDeletesFailedException(
                    "rewrite_position_delete_files: the commit could not land"
                ) from exc
            logger.error("rewrite_position_delete_files: batch %s not committed: %s", index, exc)
    return RewritePositionDeletesResult(
        rewritten_delete_files=sum(r.rewritten_delete_files for r in committed),
        added_delete_files=sum(r.added_delete_files for r in committed),
        bytes_rewritten=sum(r.bytes_rewritten for r in committed),
        bytes_added=sum(r.bytes_added for r in committed),
        rewrite_id=rewrite_id,
        snapshot_ids=[sid for r in committed for sid in r.snapshot_ids],
        commits=len(committed),
        failed_groups=failed_groups,
    )


def _empty_result(rewrite_id: str) -> RewritePositionDeletesResult:
    return RewritePositionDeletesResult(0, 0, 0, 0, rewrite_id, [], 0, 0)


def _plan(table: PyIcebergTable, scan: DataScan, normalized: dict[str, OptionValue]) -> _Plan:
    """Collect the live delete files and data paths the scan reaches and group the deletes by partition."""
    from itertools import chain

    from pyiceberg.manifest import DataFileContent
    from pyiceberg.table.metadata import INITIAL_SEQUENCE_NUMBER

    from daft.daft import _iceberg as _rust_iceberg

    deletes: dict[str, _DeleteInput] = {}
    live_data_paths: dict[tuple[int, str], set[str]] = {}
    for entry in chain.from_iterable(scan.scan_plan_helper()):
        data_file = entry.data_file
        key = (int(data_file.spec_id), stable_partition_key(data_file.partition))
        if data_file.content == DataFileContent.DATA:
            live_data_paths.setdefault(key, set()).add(str(data_file.file_path))
        elif data_file.content == DataFileContent.POSITION_DELETES:
            sequence = int(entry.sequence_number) if entry.sequence_number is not None else INITIAL_SEQUENCE_NUMBER
            deletes[str(data_file.file_path)] = _DeleteInput(data_file, sequence, key[0], key[1])

    by_spec: dict[int, list[CandidateRecord]] = {}
    for delete in deletes.values():
        by_spec.setdefault(delete.spec_id, []).append(
            {
                "path": delete.path,
                "size_bytes": int(delete.data_file.file_size_in_bytes),
                "partition_key": delete.partition_key,
                "partition_spec_id": delete.spec_id,
                "positional_delete_paths": [],
                "equality_delete_paths": [],
                "record_count": int(delete.data_file.record_count or 0),
                "deleted_record_count": 0,
            }
        )
    # A packed delete file stays in its inputs' partition and spec.
    groups: list[FileGroupRecord] = []
    for spec_id, candidates in sorted(by_spec.items()):
        groups.extend(_rust_iceberg.plan_file_groups_py(candidates, normalized, spec_id))
    return _Plan(deletes=deletes, live_data_paths=live_data_paths, groups=groups)


def _rewrite_group(
    table: PyIcebergTable, plan: _Plan, group: FileGroupRecord, io_config: IOConfig, *, per_file: bool
) -> _GroupOutput:
    """Read one group's delete rows, keep those naming a live data file, and write them packed."""
    import daft
    from daft.expressions import col

    paths = [f["path"] for f in group["files"]]
    inputs = [plan.deletes[path] for path in paths]
    spec_id = int(group["output_spec_id"])
    key = (spec_id, str(group["partition_key"]))
    # A delete under an unpartitioned spec may name a data file anywhere.
    if table.specs()[spec_id].is_unpartitioned():
        live = set().union(*plan.live_data_paths.values()) if plan.live_data_paths else set()
    else:
        live = plan.live_data_paths.get(key, set())

    rows = daft.read_parquet(paths, io_config=io_config).select(col("file_path"), col("pos"))
    live_frame = daft.from_pydict({"file_path": sorted(live)})
    rows = rows.join(live_frame, on="file_path", how="semi").sort(["file_path", "pos"])

    writer = _DeleteFileWriter(table, spec_id, inputs[0].data_file.partition, io_config)
    delete_files = writer.write_all(rows, roll_size=int(group["input_split_size"]), per_file=per_file)
    return _GroupOutput(
        input_paths=paths,
        sequence_number=max(delete.sequence_number for delete in inputs),
        delete_files=delete_files,
        bytes_rewritten=sum(int(delete.data_file.file_size_in_bytes) for delete in inputs),
        bytes_added=sum(int(delete_file.file_size_in_bytes) for delete_file in delete_files),
    )


class _DeleteFileWriter:
    """Writes sorted delete rows into files of the format's position delete layout."""

    def __init__(self, table: PyIcebergTable, spec_id: int, partition: Record, io_config: IOConfig) -> None:
        from pyiceberg.schema import Schema
        from pyiceberg.types import LongType, NestedField, StringType

        self._table = table
        self._spec_id = spec_id
        self._partition = partition
        self._io_config = io_config
        self._schema = Schema(
            NestedField(_DELETE_FILE_PATH_FIELD_ID, "file_path", StringType(), required=True),
            NestedField(_DELETE_POS_FIELD_ID, "pos", LongType(), required=True),
        )
        # Full bounds on the path column let a reader skip the file by data file.
        self._properties = {
            **_delete_write_properties(dict(table.properties)),
            "write.metadata.metrics.column.file_path": "full",
        }

    def write_all(self, rows: DataFrame, *, roll_size: int, per_file: bool) -> list[DataFile]:
        """Stream sorted ``rows`` into files, cut at ``roll_size`` bytes and, if asked, at each data file.

        ``rows`` must be sorted by ``file_path`` so a per-file cut sees each data file once.
        """
        import pyarrow as pa

        from daft.recordbatch.micropartition import MicroPartition

        written: list[DataFile] = []
        writer = self._open(len(written))
        current: str | None = None
        for batch in rows.to_arrow_iter():
            for piece in _split_by_file(batch) if per_file else [batch]:
                if piece.num_rows == 0:
                    continue
                first = piece.column("file_path")[0].as_py()
                if per_file and current is not None and first != current and writer.position > 0:
                    written.append(self._close(writer))
                    writer = self._open(len(written))
                current = first
                writer.write(MicroPartition.from_arrow(pa.Table.from_batches([piece])))
                if writer.position >= roll_size:
                    written.append(self._close(writer))
                    writer = self._open(len(written))
        if writer.position > 0:
            written.append(self._close(writer))
        else:
            writer.close()
        return written

    def _open(self, file_idx: int) -> IcebergWriter:
        from daft.io.writer import IcebergWriter
        from daft.recordbatch.recordbatch import RecordBatch

        location = self._table.properties.get("write.data.path", f"{self._table.location()}/data")
        fields = self._table.specs()[self._spec_id].fields
        partition_values = (
            RecordBatch.from_pydict({field.name: [self._partition[i]] for i, field in enumerate(fields)})
            if fields
            else None
        )
        return IcebergWriter(
            root_dir=location,
            file_idx=file_idx,
            schema=self._schema,
            properties=self._properties,
            partition_spec_id=self._spec_id,
            partition_values=partition_values,
            io_config=self._io_config,
        )

    def _close(self, writer: IcebergWriter) -> DataFile:
        from pyiceberg.manifest import DataFile, DataFileContent

        data_file = writer.close().to_pydict()["data_file"][0]
        delete_file = DataFile.from_args(
            _table_format_version=self._table.metadata.format_version,
            content=DataFileContent.POSITION_DELETES,
            file_path=data_file.file_path,
            file_format=data_file.file_format,
            partition=data_file.partition,
            record_count=data_file.record_count,
            file_size_in_bytes=data_file.file_size_in_bytes,
            column_sizes=data_file.column_sizes,
            value_counts=data_file.value_counts,
            null_value_counts=data_file.null_value_counts,
            nan_value_counts=data_file.nan_value_counts,
            lower_bounds=data_file.lower_bounds,
            upper_bounds=data_file.upper_bounds,
            split_offsets=data_file.split_offsets,
            equality_ids=None,
            key_metadata=None,
            sort_order_id=None,
        )
        delete_file.spec_id = self._spec_id
        return delete_file


def _split_by_file(batch: pa.RecordBatch) -> list[pa.RecordBatch]:
    """Cut a batch sorted by ``file_path`` into one piece per data file."""
    import pyarrow.compute as pc

    paths = batch.column("file_path")
    if batch.num_rows == 0 or pc.count_distinct(paths).as_py() <= 1:
        return [batch]
    pieces: list[pa.RecordBatch] = []
    start = 0
    values = paths.to_pylist()
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            pieces.append(batch.slice(start, index - start))
            start = index
    return pieces


_DELETE_PROPERTY_PREFIX = "write.delete."
_DATA_PROPERTY_PREFIX = "write."


def _delete_write_properties(properties: dict[str, str]) -> dict[str, str]:
    """Return the properties a delete file is written under: ``write.delete.*`` over ``write.*``."""
    out = dict(properties)
    for key, value in properties.items():
        if key.startswith(_DELETE_PROPERTY_PREFIX) and not key.endswith(("target-file-size-bytes", "granularity")):
            out[_DATA_PROPERTY_PREFIX + key[len(_DELETE_PROPERTY_PREFIX) :]] = value
    return out


def _commit_batch(
    table: PyIcebergTable,
    outputs: list[_GroupOutput],
    rewrite_id: str,
    batch_label: str | None,
    branch: str | None,
) -> RewritePositionDeletesResult:
    """Commit one batch under retry: the packed files in, the old ones out, sequence numbers kept."""
    from pyiceberg.table.refs import MAIN_BRANCH
    from pyiceberg.table.snapshots import Operation

    input_paths = sorted({path for output in outputs for path in output.input_paths})
    added = [(delete_file, output.sequence_number) for output in outputs for delete_file in output.delete_files]
    bytes_rewritten = sum(output.bytes_rewritten for output in outputs)
    bytes_added = sum(output.bytes_added for output in outputs)
    snapshot_props = {
        SNAPSHOT_PROP_MAINTENANCE_OP: SNAPSHOT_PROP_MAINTENANCE_OP_VALUE,
        SNAPSHOT_PROP_REWRITE_ID: rewrite_id,
        SNAPSHOT_PROP_STRATEGY: SNAPSHOT_PROP_STRATEGY_VALUE,
        SNAPSHOT_PROP_INPUT_FILES: str(len(input_paths)),
        SNAPSHOT_PROP_OUTPUT_FILES: str(len(added)),
        SNAPSHOT_PROP_INPUT_BYTES: str(bytes_rewritten),
        SNAPSHOT_PROP_OUTPUT_BYTES: str(bytes_added),
    }
    if batch_label is not None:
        snapshot_props[SNAPSHOT_PROP_BATCH] = batch_label

    def _result(snapshot_id: int) -> RewritePositionDeletesResult:
        return RewritePositionDeletesResult(
            rewritten_delete_files=len(input_paths),
            added_delete_files=len(added),
            bytes_rewritten=bytes_rewritten,
            bytes_added=bytes_added,
            rewrite_id=rewrite_id,
            snapshot_ids=[snapshot_id],
            commits=1,
            failed_groups=0,
        )

    def _attempt(_: int) -> RewritePositionDeletesResult:
        table.refresh()
        replay = _find_batch_snapshot(table, rewrite_id, batch_label, branch)
        if replay is not None:
            return _result(replay)
        live = _live_delete_files(table, branch)
        missing = [path for path in input_paths if path not in live]
        if missing:
            raise RewriteConflict(
                f"rewrite_position_delete_files: {len(missing)} input delete file(s) are no longer live"
            )
        tx = table.transaction()
        producer = _delete_rewrite_producer_class()(
            operation=Operation.REPLACE,
            transaction=tx,
            io=table.io,
            commit_uuid=_uuid.uuid4(),
            snapshot_properties=snapshot_props,
            branch=branch if branch is not None else MAIN_BRANCH,
            starting_sequence_number=None,
            sequence_by_path={str(delete_file.file_path): sequence for delete_file, sequence in added},
        )
        for path in input_paths:
            producer.delete_data_file(live[path])
        for delete_file, _sequence in added:
            producer.append_data_file(delete_file)
        producer.commit()
        tx.commit_transaction()
        table.refresh()
        head = table.snapshot_by_name(branch) if branch is not None else table.current_snapshot()
        assert head is not None
        return _result(int(head.snapshot_id))

    return commit_with_retry(table, _attempt, op_name="rewrite_position_delete_files")


def _delete_rewrite_producer_class() -> type:
    """Return the commit producer that stamps each added delete file with its own sequence number."""

    class _DeleteRewriteProducer(_compaction_producer_class()):  # type: ignore[misc]
        def __init__(self, *, sequence_by_path: dict[str, int], **kwargs: object) -> None:
            super().__init__(**kwargs)
            self._sequence_by_path = sequence_by_path

        def _added_entry_sequence_number_for(self, data_file: DataFile) -> int | None:
            return self._sequence_by_path.get(
                str(data_file.file_path), super()._added_entry_sequence_number_for(data_file)
            )

    return _DeleteRewriteProducer


def _live_delete_files(table: PyIcebergTable, branch: str | None) -> dict[str, DataFile]:
    """Return the live position delete files on the branch head, by path."""
    from pyiceberg.manifest import DataFileContent, ManifestContent

    head = table.snapshot_by_name(branch) if branch is not None else table.current_snapshot()
    if head is None:
        return {}
    live: dict[str, DataFile] = {}
    for manifest in head.manifests(table.io):
        if manifest.content != ManifestContent.DELETES:
            continue
        for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=True):
            if entry.data_file.content == DataFileContent.POSITION_DELETES:
                live[str(entry.data_file.file_path)] = entry.data_file
    return live


def _abort(table: PyIcebergTable, outputs: list[_GroupOutput]) -> None:
    """Delete the files a batch wrote once no snapshot will reference them."""
    for output in outputs:
        for delete_file in output.delete_files:
            try:
                table.io.delete(str(delete_file.file_path))
            except OSError as exc:
                logger.warning("rewrite_position_delete_files: could not remove %s: %s", delete_file.file_path, exc)


def _batches(groups: list[FileGroupRecord], max_commits: int) -> list[list[FileGroupRecord]]:
    """Cut the groups into at most ``max_commits`` batches of plan order."""
    per_batch = max(1, math.ceil(len(groups) / max_commits))
    return [groups[i : i + per_batch] for i in range(0, len(groups), per_batch)]


def _digest(
    table: PyIcebergTable, branch: str | None, normalized: dict[str, OptionValue], input_paths: list[str]
) -> str:
    payload = {
        "table_uuid": str(table.metadata.table_uuid),
        "branch": branch,
        "options": {key: normalized[key] for key in sorted(normalized)},
        "inputs": input_paths,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _find_batch_snapshot(
    table: PyIcebergTable, rewrite_id: str, batch_label: str | None, branch: str | None
) -> int | None:
    """Return the snapshot a replay of this batch already committed, if any."""
    for snapshot in branch_ancestry(table, branch):
        summary = _summary_as_dict(snapshot.summary)
        if (
            summary.get(SNAPSHOT_PROP_REWRITE_ID) == rewrite_id
            and summary.get(SNAPSHOT_PROP_STRATEGY) == SNAPSHOT_PROP_STRATEGY_VALUE
            and summary.get(SNAPSHOT_PROP_BATCH) == batch_label
        ):
            return int(snapshot.snapshot_id)
    return None


def _lookup_result(table: PyIcebergTable, rewrite_id: str, branch: str | None) -> RewritePositionDeletesResult | None:
    """Return the result an earlier run of this rewrite recorded on the branch, if any."""
    matches = [
        (snapshot, summary)
        for snapshot in branch_ancestry(table, branch)
        for summary in [_summary_as_dict(snapshot.summary)]
        if summary.get(SNAPSHOT_PROP_REWRITE_ID) == rewrite_id
        and summary.get(SNAPSHOT_PROP_STRATEGY) == SNAPSHOT_PROP_STRATEGY_VALUE
    ]
    if not matches:
        return None
    return RewritePositionDeletesResult(
        rewritten_delete_files=sum(int(s.get(SNAPSHOT_PROP_INPUT_FILES, 0)) for _, s in matches),
        added_delete_files=sum(int(s.get(SNAPSHOT_PROP_OUTPUT_FILES, 0)) for _, s in matches),
        bytes_rewritten=sum(int(s.get(SNAPSHOT_PROP_INPUT_BYTES, 0)) for _, s in matches),
        bytes_added=sum(int(s.get(SNAPSHOT_PROP_OUTPUT_BYTES, 0)) for _, s in matches),
        rewrite_id=rewrite_id,
        snapshot_ids=[int(snapshot.snapshot_id) for snapshot, _ in reversed(matches)],
        commits=len(matches),
        failed_groups=0,
    )
