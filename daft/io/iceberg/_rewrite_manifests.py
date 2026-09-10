"""Repack live manifest entries into target-sized manifests.

Only the data manifests of the chosen partition spec are repacked; delete
manifests and other specs' manifests are carried through unchanged. The result
is one REPLACE snapshot whose data and delete files are unchanged.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import uuid as _uuid
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, TypeAlias, TypeVar

from daft.io.iceberg._common import (
    CommitRetryExhausted,
    MaintenanceOptions,
    branch_ancestry,
    commit_with_retry,
    manifest_writer_for,
    option_int,
    option_names,
)

if TYPE_CHECKING:
    from pyiceberg.io import FileIO
    from pyiceberg.manifest import DataFile, ManifestContent, ManifestEntry, ManifestFile, ManifestWriter
    from pyiceberg.partitioning import PartitionSpec
    from pyiceberg.table import Table as PyIcebergTable
    from pyiceberg.table import Transaction
    from pyiceberg.table.snapshots import Operation, Summary
    from pyiceberg.typedef import Record

_T = TypeVar("_T")

#: One partition field's value as stored in a manifest entry: a primitive in its Python form, or null.
PartitionValue: TypeAlias = (
    str | int | float | bool | bytes | Decimal | datetime.date | datetime.datetime | datetime.time | _uuid.UUID | None
)


#: Opens a manifest writer of a given content under a partition spec.
ManifestWriterFactory: TypeAlias = "Callable[[ManifestContent, PartitionSpec], ManifestWriter]"


logger = logging.getLogger(__name__)


MANIFEST_TARGET_SIZE_KEY = "commit.manifest.target-size-bytes"
_DEFAULT_MANIFEST_TARGET_SIZE_BYTES = 8 * 1024 * 1024
_ROLL_FACTOR = 1.2

SNAPSHOT_PROP_REWRITE_ID = "daft.rewrite-id"
SNAPSHOT_PROP_REWRITE_STRATEGY = "daft.rewrite-strategy"
SNAPSHOT_PROP_REWRITE_STRATEGY_VALUE = "manifests"
SNAPSHOT_PROP_MAINTENANCE_OP = "daft.maintenance.op"
SNAPSHOT_PROP_MAINTENANCE_OP_VALUE = "rewrite-manifests"
SNAPSHOT_PROP_SPEC_ID = "daft.spec-id"
SNAPSHOT_PROP_INPUT_MANIFESTS = "daft.input-manifests"
SNAPSHOT_PROP_OUTPUT_MANIFESTS = "daft.output-manifests"
SNAPSHOT_PROP_INPUT_BYTES = "daft.input-manifest-bytes"
SNAPSHOT_PROP_OUTPUT_BYTES = "daft.output-manifest-bytes"

# Summary keys the table format defines for a manifest rewrite.
SUMMARY_MANIFESTS_CREATED = "manifests-created"
SUMMARY_MANIFESTS_KEPT = "manifests-kept"
SUMMARY_MANIFESTS_REPLACED = "manifests-replaced"
SUMMARY_ENTRIES_PROCESSED = "entries-processed"
SUMMARY_CHANGED_PARTITION_COUNT = "changed-partition-count"


@dataclass(frozen=True)
class RewriteManifestsResult:
    """Summary of a rewrite_manifests invocation.

    Parameters
    ----------
    rewritten_manifests_count
        Number of input manifest files whose entries were repacked.
    added_manifests_count
        Number of output manifest files written.
    bytes_rewritten
        Total ``length`` of the input manifests that were repacked.
    bytes_added
        Total ``length`` of the output manifests.
    rewrite_id
        Stable identifier used for idempotent replay.
    snapshot_id
        The REPLACE snapshot's ID, or ``None`` when nothing needed rewriting.
    """

    rewritten_manifests_count: int = 0
    added_manifests_count: int = 0
    bytes_rewritten: int = 0
    bytes_added: int = 0
    rewrite_id: str = ""
    snapshot_id: int | None = None


class RewriteManifestsFailedException(RuntimeError):
    """Raised when rewrite_manifests cannot commit after the retry budget."""


def run(
    table: PyIcebergTable,
    *,
    spec_id: int | None = None,
    branch: str | None = None,
    use_caching: bool = False,
    options: MaintenanceOptions | None = None,
) -> RewriteManifestsResult:
    """Repack the live manifest entries of one partition spec on a branch.

    Data manifests and delete manifests are repacked separately, each into
    as many manifests as its bytes fill at ``manifest-target-size-bytes``. A
    layout already balanced against that size is left alone, and a rewrite
    whose identity is already recorded on the branch returns the recorded
    result instead of committing again.

    Parameters
    ----------
    table
        Table whose manifests are rewritten.
    spec_id
        Partition spec whose manifests are repacked; defaults to the current
        spec.
    branch
        Branch to commit to; defaults to the main branch.
    use_caching
        Accepted for interface compatibility and ignored.
    options
        ``manifest-target-size-bytes``,
        ``manifest-read-concurrency`` and ``sort-by``.

    Returns:
    -------
    RewriteManifestsResult
        Manifest and byte counts, the rewrite identity, and the new snapshot id
        (``None`` when nothing was rewritten).

    Raises:
    ------
    ValueError
        If an option is out of range, ``spec_id`` is unknown, ``branch`` is
        missing or is a tag, or ``sort-by`` names a non-partition field.
    RewriteManifestsFailedException
        If the commit cannot land within the retry budget.
    """
    from pyiceberg.table.snapshots import Operation

    opts = options or {}
    target_size_bytes = option_int(
        opts,
        "manifest-target-size-bytes",
        int(table.properties.get(MANIFEST_TARGET_SIZE_KEY, _DEFAULT_MANIFEST_TARGET_SIZE_BYTES)),
    )
    if target_size_bytes <= 0:
        raise ValueError(f"manifest-target-size-bytes must be > 0, got {target_size_bytes!r}")

    read_concurrency = option_int(opts, "manifest-read-concurrency", 1)
    if read_concurrency < 1:
        raise ValueError(f"manifest-read-concurrency must be >= 1, got {read_concurrency!r}")

    sort_by = option_names(opts, "sort-by")

    # Entries stream into the writers, so there is nothing to cache.
    del use_caching

    resolved_spec_id = _resolve_spec_id(table, spec_id)
    target_branch = _resolve_branch(table, branch)
    _validate_sort_by(table, resolved_spec_id, sort_by)

    plan = _plan(
        table=table,
        spec_id=resolved_spec_id,
        target_branch=target_branch,
        target_size_bytes=target_size_bytes,
    )

    if plan.rewrite_id:
        cached = _lookup_idempotent_result(table, plan.rewrite_id, target_branch)
        if cached is not None:
            return cached

    if not plan.matching_manifests or plan.no_op_reason is not None:
        logger.info(
            "rewrite_manifests: no-op (%s)",
            plan.no_op_reason or "no manifests matched the spec filter",
        )
        return RewriteManifestsResult(
            rewrite_id=plan.rewrite_id,
            snapshot_id=None,
        )

    plan_box = {"plan": plan}

    def _attempt(_: int) -> RewriteManifestsResult:
        return _commit_attempt(
            table=table,
            plan=plan_box["plan"],
            target_branch=target_branch,
            operation=Operation.REPLACE,
            target_size_bytes=target_size_bytes,
            sort_by=sort_by,
            read_concurrency=read_concurrency,
        )

    def _on_conflict(t: PyIcebergTable) -> RewriteManifestsResult | None:
        cached = _lookup_idempotent_result(t, plan_box["plan"].rewrite_id, target_branch)
        if cached is not None:
            return cached
        new_plan = _plan(
            table=t,
            spec_id=resolved_spec_id,
            target_branch=target_branch,
            target_size_bytes=target_size_bytes,
        )
        if not new_plan.matching_manifests or new_plan.no_op_reason is not None:
            return RewriteManifestsResult(
                rewrite_id=new_plan.rewrite_id,
                snapshot_id=None,
            )
        plan_box["plan"] = new_plan
        return None

    try:
        return commit_with_retry(
            table,
            _attempt,
            op_name="rewrite_manifests",
            on_conflict=_on_conflict,
        )
    except CommitRetryExhausted as exc:
        raise RewriteManifestsFailedException(
            "rewrite_manifests: commit could not land within the retry budget"
        ) from exc


@dataclass(frozen=True)
class _ContentGroup:
    """The manifests of one content kind that a rewrite repacks together."""

    content: ManifestContent
    manifests: list[ManifestFile]
    bytes_rewritten: int
    live_entries: int


@dataclass
class _Plan:
    """Manifests one rewrite repacks or carries through, with its replay identity."""

    rewrite_id: str
    groups: list[_ContentGroup]
    untouched_manifests: list[ManifestFile]
    no_op_reason: str | None = None

    @property
    def matching_manifests(self) -> list[ManifestFile]:
        """Every manifest being repacked, data manifests first."""
        return [manifest for group in self.groups for manifest in group.manifests]

    @property
    def bytes_rewritten(self) -> int:
        return sum(group.bytes_rewritten for group in self.groups)

    @property
    def total_live_entries(self) -> int:
        return sum(group.live_entries for group in self.groups)


def _balanced(manifests: list[ManifestFile], bytes_rewritten: int, target_size_bytes: int) -> bool:
    """Return whether a set of manifests is already as many as the target size calls for.

    A manifest without a live entry is never balanced: every scan still opens it.
    """
    roll = int(target_size_bytes * _ROLL_FACTOR)
    expected_count = max(1, (bytes_rewritten + target_size_bytes - 1) // target_size_bytes)
    if len(manifests) != expected_count:
        return False
    for manifest in manifests:
        if int(manifest.manifest_length) > roll:
            return False
        if int(manifest.added_files_count or 0) + int(manifest.existing_files_count or 0) == 0:
            return False
    return True


def _plan(
    *,
    table: PyIcebergTable,
    spec_id: int,
    target_branch: str,
    target_size_bytes: int,
) -> _Plan:
    """Partition the branch head's manifests into repacked and carried-through sets."""
    snapshot = table.metadata.snapshot_by_name(target_branch)
    if snapshot is None:
        return _Plan(
            rewrite_id="",
            groups=[],
            untouched_manifests=[],
            no_op_reason=f"branch {target_branch!r} has no current snapshot",
        )

    from pyiceberg.manifest import ManifestContent

    manifests = list(snapshot.manifests(table.io))
    of_spec = [m for m in manifests if int(m.partition_spec_id) == int(spec_id)]
    # Entries of one content kind must never land in manifests of the other.
    groups: list[_ContentGroup] = []
    balanced_count = 0
    for content in (ManifestContent.DATA, ManifestContent.DELETES):
        matching = [m for m in of_spec if m.content == content]
        if not matching:
            continue
        bytes_rewritten = sum(int(m.manifest_length) for m in matching)
        if _balanced(matching, bytes_rewritten, target_size_bytes):
            balanced_count += 1
            continue
        groups.append(
            _ContentGroup(
                content=content,
                manifests=matching,
                bytes_rewritten=bytes_rewritten,
                live_entries=sum(int(m.added_files_count or 0) + int(m.existing_files_count or 0) for m in matching),
            )
        )
    repacked = {m.manifest_path for group in groups for m in group.manifests}
    untouched = [m for m in manifests if m.manifest_path not in repacked]

    rewrite_id = _resolve_rewrite_id(
        table=table,
        target_branch=target_branch,
        spec_id=spec_id,
        target_size_bytes=target_size_bytes,
        matching_manifest_paths=sorted(repacked),
    )

    no_op_reason: str | None = None
    if not of_spec:
        no_op_reason = "no manifests for spec_id"
    elif not groups:
        no_op_reason = (
            f"manifest layout already balanced ({balanced_count} content kind(s), target={target_size_bytes})"
        )

    return _Plan(rewrite_id=rewrite_id, groups=groups, untouched_manifests=untouched, no_op_reason=no_op_reason)


def _commit_attempt(
    *,
    table: PyIcebergTable,
    plan: _Plan,
    target_branch: str,
    operation: Operation,
    target_size_bytes: int,
    sort_by: list[str] | None,
    read_concurrency: int,
) -> RewriteManifestsResult:
    """Write the new manifests and commit one REPLACE snapshot for ``plan``."""
    snapshot_props = {
        SNAPSHOT_PROP_MAINTENANCE_OP: SNAPSHOT_PROP_MAINTENANCE_OP_VALUE,
        SNAPSHOT_PROP_REWRITE_ID: plan.rewrite_id,
        SNAPSHOT_PROP_REWRITE_STRATEGY: SNAPSHOT_PROP_REWRITE_STRATEGY_VALUE,
        SNAPSHOT_PROP_INPUT_MANIFESTS: str(len(plan.matching_manifests)),
        SNAPSHOT_PROP_INPUT_BYTES: str(plan.bytes_rewritten),
        SNAPSHOT_PROP_SPEC_ID: str(_spec_id_of_plan(plan)),
    }

    producer_cls = _producer_class()
    with table.transaction() as txn:
        producer = producer_cls(
            operation=operation,
            transaction=txn,
            io=table.io,
            branch=target_branch,
            snapshot_properties=snapshot_props,
            commit_uuid=_uuid.uuid4(),
            plan=plan,
            target_size_bytes=target_size_bytes,
            sort_by=sort_by,
            read_concurrency=read_concurrency,
        )
        producer.build_new_manifests()
        producer.snapshot_properties[SNAPSHOT_PROP_OUTPUT_MANIFESTS] = str(len(producer.new_manifests))
        producer.snapshot_properties[SNAPSHOT_PROP_OUTPUT_BYTES] = str(producer.bytes_added)
        producer.snapshot_properties[SUMMARY_MANIFESTS_CREATED] = str(len(producer.new_manifests))
        producer.snapshot_properties[SUMMARY_MANIFESTS_KEPT] = str(len(plan.untouched_manifests))
        producer.snapshot_properties[SUMMARY_MANIFESTS_REPLACED] = str(len(plan.matching_manifests))
        producer.snapshot_properties[SUMMARY_ENTRIES_PROCESSED] = str(producer.entries_processed)
        producer.snapshot_properties[SUMMARY_CHANGED_PARTITION_COUNT] = str(producer.changed_partition_count)
        producer.commit()
        committed_snapshot_id = int(producer.snapshot_id)
        added = len(producer.new_manifests)
        bytes_added = producer.bytes_added

    return RewriteManifestsResult(
        rewritten_manifests_count=len(plan.matching_manifests),
        added_manifests_count=added,
        bytes_rewritten=plan.bytes_rewritten,
        bytes_added=bytes_added,
        rewrite_id=plan.rewrite_id,
        snapshot_id=committed_snapshot_id,
    )


def _spec_id_of_plan(plan: _Plan) -> int:
    """Return the partition spec id shared by the manifests being repacked."""
    return int(plan.matching_manifests[0].partition_spec_id)


_PRODUCER_CLASS: type | None = None


def _producer_class() -> type:
    """Build and cache the snapshot producer that commits a manifest rewrite."""
    global _PRODUCER_CLASS
    if _PRODUCER_CLASS is not None:
        return _PRODUCER_CLASS

    from pyiceberg.table.update.snapshot import _SnapshotProducer

    class _RewriteManifestsProducer(_SnapshotProducer):  # type: ignore[misc]
        """Commit a REPLACE snapshot listing the untouched manifests plus the repacked ones."""

        def __init__(
            self,
            *,
            operation: Operation,
            transaction: Transaction,
            io: FileIO,
            branch: str,
            snapshot_properties: dict[str, str],
            commit_uuid: _uuid.UUID,
            plan: _Plan,
            target_size_bytes: int,
            sort_by: list[str] | None = None,
            read_concurrency: int = 1,
        ) -> None:
            super().__init__(
                operation=operation,
                transaction=transaction,
                io=io,
                commit_uuid=commit_uuid,
                snapshot_properties=dict(snapshot_properties),
                branch=branch,
            )
            self._plan = plan
            self._target_size_bytes = target_size_bytes
            self._sort_by = sort_by
            self._read_concurrency = max(1, int(read_concurrency))
            self._untouched_manifests: list[ManifestFile] = list(plan.untouched_manifests)
            self._new_manifests: list[ManifestFile] = []
            self._bytes_added = 0
            self._entries_processed = 0
            self._changed_partitions: set[tuple[int, tuple[PartitionValue, ...]]] = set()

        @property
        def new_manifests(self) -> list[ManifestFile]:
            """Manifests written by this producer."""
            return self._new_manifests

        @property
        def bytes_added(self) -> int:
            """Total length of the manifests written."""
            return self._bytes_added

        @property
        def entries_processed(self) -> int:
            """Number of live manifest entries read and rewritten."""
            return self._entries_processed

        @property
        def changed_partition_count(self) -> int:
            """Number of distinct partitions whose entries moved to a new manifest."""
            return len(self._changed_partitions)

        def _existing_manifests(self) -> list[ManifestFile]:
            """Return the untouched manifests followed by the new ones."""
            return list(self._untouched_manifests) + list(self._new_manifests)

        def _deleted_entries(self) -> list[ManifestEntry]:
            """Return no entries; a manifest rewrite deletes nothing."""
            return []

        def _summary(self, snapshot_properties: dict[str, str]) -> Summary:
            """Build the snapshot summary, carrying the parent's data totals forward unchanged."""
            from pyiceberg.table.snapshots import (
                TOTAL_DATA_FILES,
                TOTAL_DELETE_FILES,
                TOTAL_EQUALITY_DELETES,
                TOTAL_FILE_SIZE,
                TOTAL_POSITION_DELETES,
                TOTAL_RECORDS,
                Operation,
                Summary,
            )

            previous_snapshot = (
                self._transaction.table_metadata.snapshot_by_id(self._parent_snapshot_id)
                if self._parent_snapshot_id is not None
                else None
            )
            prev_props: dict[str, str] = {}
            if previous_snapshot is not None and previous_snapshot.summary is not None:
                prev_props = dict(getattr(previous_snapshot.summary, "additional_properties", {}) or {})
            carry = {
                k: prev_props[k]
                for k in (
                    TOTAL_DATA_FILES,
                    TOTAL_DELETE_FILES,
                    TOTAL_RECORDS,
                    TOTAL_FILE_SIZE,
                    TOTAL_POSITION_DELETES,
                    TOTAL_EQUALITY_DELETES,
                )
                if k in prev_props
            }
            return Summary(operation=Operation.REPLACE, **carry, **snapshot_properties)

        def new_manifest_writer_for(self, content: ManifestContent, spec: PartitionSpec) -> ManifestWriter:
            """Return a writer for a manifest of ``content`` under ``spec``."""
            return manifest_writer_for(self, content, spec)

        def build_new_manifests(self) -> None:
            """Read live entries of each content kind and write target-sized manifests of that kind.

            Entries are ordered by partition (or by the ``sort-by`` fields) and cut
            into as many contiguous runs as the repacked bytes fill at the target
            size, so a partition-scoped read touches few manifests.
            """
            for group in self._plan.groups:
                self._repack(group)

        def _repack(self, group: _ContentGroup) -> None:
            from pyiceberg.manifest import ManifestEntry, ManifestEntryStatus

            roll_at_bytes = int(self._target_size_bytes * _ROLL_FACTOR)
            # The entry-count budget applies only when the writer cannot report a running byte size.
            avg_bytes = max(1, group.bytes_rewritten // max(1, group.live_entries))
            fallback_roll_at_entries = max(1, int(roll_at_bytes / avg_bytes))
            specs = self._transaction.table_metadata.specs()

            ordered: list[tuple[int, ManifestEntry]] = []
            for manifest, entries in self._read_entries(group.manifests):
                spec_id = manifest.partition_spec_id
                for entry in entries:
                    self._entries_processed += 1
                    self._changed_partitions.add((spec_id, tuple(entry.data_file.partition)))
                    ordered.append((spec_id, entry))
            if ordered and specs[ordered[0][0]].fields:
                ordered.sort(
                    key=lambda item: _ordering_key(_cluster_key(item[1].data_file, specs[item[0]], self._sort_by))
                )

            target_count = max(1, (group.bytes_rewritten + self._target_size_bytes - 1) // self._target_size_bytes)
            for run in _even_runs(ordered, target_count):
                roller = _RollingManifestWriter(
                    open_writer=self.new_manifest_writer_for,
                    content=group.content,
                    spec=specs[run[0][0]],
                    roll_at_bytes=roll_at_bytes,
                    fallback_roll_at_entries=fallback_roll_at_entries,
                )
                for _, entry in run:
                    roller.add(
                        ManifestEntry.from_args(
                            status=ManifestEntryStatus.EXISTING,
                            snapshot_id=entry.snapshot_id,
                            sequence_number=entry.sequence_number,
                            file_sequence_number=entry.file_sequence_number,
                            data_file=entry.data_file,
                        )
                    )
                for mf in roller.finish():
                    self._new_manifests.append(mf)
                    self._bytes_added += int(mf.manifest_length)

        def _read_entries(self, manifests: list[ManifestFile]) -> list[tuple[ManifestFile, list[ManifestEntry]]]:
            """Read live entries from each manifest, preserving input order."""
            if self._read_concurrency == 1 or len(manifests) <= 1:
                return [(m, list(m.fetch_manifest_entry(self._io, discard_deleted=True))) for m in manifests]
            from concurrent.futures import ThreadPoolExecutor

            def _read(m: ManifestFile) -> list[ManifestEntry]:
                return list(m.fetch_manifest_entry(self._io, discard_deleted=True))

            workers = min(self._read_concurrency, len(manifests))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                entry_lists = list(pool.map(_read, manifests))
            return list(zip(manifests, entry_lists))

    _PRODUCER_CLASS = _RewriteManifestsProducer
    return _PRODUCER_CLASS


class _RollingManifestWriter:
    """Roll to a new manifest once the bytes written reach the target size."""

    def __init__(
        self,
        *,
        open_writer: ManifestWriterFactory,
        content: ManifestContent,
        spec: PartitionSpec,
        roll_at_bytes: int,
        fallback_roll_at_entries: int,
    ):
        self._open_writer = open_writer
        self._content = content
        self._spec = spec
        self._roll_at_bytes = max(1, roll_at_bytes)
        self._fallback_roll_at_entries = max(1, fallback_roll_at_entries)
        self._writer: ManifestWriter | None = None
        self._count = 0
        self._finished: list[ManifestFile] = []

    def add(self, entry: ManifestEntry) -> None:
        """Add ``entry``, starting a new manifest first if the current one is full."""
        if self._writer is None or self._should_roll():
            self._close_writer()
            self._writer = self._open_writer(self._content, self._spec).__enter__()
            self._count = 0
        self._writer.add_entry(entry)
        self._count += 1

    def _should_roll(self) -> bool:
        """Return whether the open manifest has reached its size budget."""
        written = _manifest_writer_bytes(self._writer)
        if written is not None:
            return written >= self._roll_at_bytes
        return self._count >= self._fallback_roll_at_entries

    def _close_writer(self) -> None:
        """Close the open manifest, if any, and record its manifest file."""
        if self._writer is None:
            return
        self._writer.__exit__(None, None, None)
        self._finished.append(self._writer.to_manifest_file())
        self._writer = None

    def finish(self) -> list[ManifestFile]:
        """Close the open manifest and return every manifest written."""
        self._close_writer()
        return self._finished


def _manifest_writer_bytes(writer: ManifestWriter | None) -> int | None:
    """Return the bytes written so far by an open manifest writer, or ``None`` when unknown."""
    output_stream = getattr(getattr(writer, "_writer", None), "output_stream", None)
    tell = getattr(output_stream, "tell", None)
    if tell is None:
        return None
    try:
        return int(tell())
    except (OSError, ValueError):
        return None


def _partition_values_in_order(partition: Record, n: int) -> list[PartitionValue]:
    """Return a partition record's values aligned to its spec's field order."""
    try:
        return [partition[i] for i in range(n)]
    except (TypeError, IndexError, KeyError):
        pass
    try:
        return list(partition)[:n]
    except TypeError:
        return list((getattr(partition, "__dict__", None) or {}).values())[:n]


def _cluster_key(
    data_file: DataFile, spec: PartitionSpec, sort_by: list[str] | None
) -> tuple[tuple[str, PartitionValue], ...]:
    """Return the partition values that cluster a data file's entry into a manifest."""
    partition = getattr(data_file, "partition", None)
    if partition is None:
        return ()
    names = [f.name for f in spec.fields]
    values = _partition_values_in_order(partition, len(names))
    pairs = list(zip(names, values))
    if sort_by:
        wanted = set(sort_by)
        pairs = [(n, v) for n, v in pairs if n in wanted]
    return tuple(pairs)


def _ordering_key(cluster_key: tuple[tuple[str, PartitionValue], ...]) -> tuple[tuple[int, PartitionValue], ...]:
    """Return a total order over a clustering key that places a missing value first."""
    return tuple((0, None) if value is None else (1, value) for _, value in cluster_key)


def _even_runs(items: list[_T], count: int) -> list[list[_T]]:
    """Cut ``items`` into up to ``count`` contiguous runs of near-equal length, dropping empty ones."""
    if not items:
        return []
    count = max(1, min(count, len(items)))
    size, remainder = divmod(len(items), count)
    runs: list[list[_T]] = []
    start = 0
    for index in range(count):
        end = start + size + (1 if index < remainder else 0)
        runs.append(items[start:end])
        start = end
    return runs


def _validate_sort_by(table: PyIcebergTable, spec_id: int, sort_by: list[str] | None) -> None:
    """Reject ``sort_by`` columns that are not partition fields of the spec."""
    if not sort_by:
        return
    spec = table.specs()[spec_id]
    names = {f.name for f in spec.fields}
    bad = [c for c in sort_by if c not in names]
    if bad:
        raise ValueError(
            f"sort-by columns {bad!r} are not partition fields of spec {spec_id} (available: {sorted(names)})"
        )


def _resolve_spec_id(table: PyIcebergTable, spec_id: int | None) -> int:
    """Return ``spec_id`` validated against the table's specs, or the current spec id."""
    if spec_id is None:
        return int(table.spec().spec_id)
    if int(spec_id) not in {int(s) for s in table.specs().keys()}:
        raise ValueError(
            f"spec_id={spec_id!r} is not present in table.specs() ({sorted(int(s) for s in table.specs().keys())})"
        )
    return int(spec_id)


def _resolve_branch(table: PyIcebergTable, branch: str | None) -> str:
    """Return the branch to commit to, rejecting unknown names and tags."""
    from pyiceberg.table.refs import MAIN_BRANCH, SnapshotRefType

    if branch is None:
        return MAIN_BRANCH
    ref = table.metadata.refs.get(branch)
    if ref is None:
        raise ValueError(f"branch {branch!r} does not exist on this table")
    if ref.snapshot_ref_type != SnapshotRefType.BRANCH:
        raise ValueError(f"{branch!r} is a tag, not a branch")
    return branch


def _resolve_rewrite_id(
    *,
    table: PyIcebergTable,
    target_branch: str,
    spec_id: int,
    target_size_bytes: int,
    matching_manifest_paths: list[str],
) -> str:
    """Derive a stable rewrite identity from the table, branch, spec, target and inputs."""
    payload = {
        "table_uuid": str(table.metadata.table_uuid),
        "branch": target_branch,
        "spec_id": int(spec_id),
        "target_size_bytes": int(target_size_bytes),
        "manifests": sorted(matching_manifest_paths),
    }
    import json

    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _lookup_idempotent_result(
    table: PyIcebergTable, rewrite_id: str, target_branch: str | None
) -> RewriteManifestsResult | None:
    """Return the result recorded by an earlier run of this rewrite on the branch, if any."""
    if not rewrite_id:
        return None
    for snap in branch_ancestry(table, target_branch):
        summary = _summary_as_dict(snap.summary)
        if (
            summary.get(SNAPSHOT_PROP_REWRITE_ID) == rewrite_id
            and summary.get(SNAPSHOT_PROP_REWRITE_STRATEGY) == SNAPSHOT_PROP_REWRITE_STRATEGY_VALUE
        ):
            return RewriteManifestsResult(
                rewritten_manifests_count=int(summary.get(SNAPSHOT_PROP_INPUT_MANIFESTS, 0)),
                added_manifests_count=int(summary.get(SNAPSHOT_PROP_OUTPUT_MANIFESTS, 0)),
                bytes_rewritten=int(summary.get(SNAPSHOT_PROP_INPUT_BYTES, 0)),
                bytes_added=int(summary.get(SNAPSHOT_PROP_OUTPUT_BYTES, 0)),
                rewrite_id=rewrite_id,
                snapshot_id=int(snap.snapshot_id),
            )
    return None


def _summary_as_dict(summary: Summary | None) -> dict[str, str]:
    """Return a snapshot summary's operation and properties as a flat string mapping."""
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
