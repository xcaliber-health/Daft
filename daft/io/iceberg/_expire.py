"""Expire snapshots and delete the files only they referenced.

Every branch keeps its head and its ancestry up to the minimum count or age
cutoff, a tag keeps its snapshot until the tag ages out, and everything else
expires. The files to delete are those referenced before the metadata commit
and no longer referenced after it.
"""

from __future__ import annotations

import datetime as _dt
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from daft.io.iceberg._common import (
    DEFAULT_DELETE_BACKOFF_BASE_SECONDS,
    DEFAULT_DELETE_NUM_RETRIES,
    DEFAULT_MAX_CONCURRENT_DELETES,
    CommitRetryExhausted,
    MaintenanceOptions,
    commit_with_retry,
    delete_files,
    is_not_found,
    option_float,
    option_int,
    validate_gc_enabled,
)
from daft.io.iceberg._engine import (
    KIND_DATA,
    KIND_EQ_DELETE,
    KIND_MANIFEST,
    KIND_MANIFEST_LIST,
    KIND_METADATA,
    KIND_POS_DELETE,
    KIND_STATS,
    anti_join_paths,
    content_frame,
    engine_delete,
    manifest_frame,
    paths_frame,
    union_paths,
)

if TYPE_CHECKING:
    from pyiceberg.table import Table as PyIcebergTable
    from pyiceberg.table.refs import SnapshotRef

    from daft.dataframe import DataFrame

logger = logging.getLogger(__name__)


MAX_SNAPSHOT_AGE_MS_KEY = "history.expire.max-snapshot-age-ms"
MIN_SNAPSHOTS_TO_KEEP_KEY = "history.expire.min-snapshots-to-keep"
MAX_REF_AGE_MS_KEY = "history.expire.max-ref-age-ms"

_DEFAULT_MAX_SNAPSHOT_AGE_MS = 5 * 24 * 60 * 60 * 1000
_DEFAULT_MIN_SNAPSHOTS_TO_KEEP = 1
#: A reference never ages out unless the table or the reference sets a maximum age.
_DEFAULT_MAX_REF_AGE_MS = 2**63 - 1


@dataclass(frozen=True)
class ExpireResult:
    """Summary of an expire_snapshots invocation.

    Parameters
    ----------
    deleted_data_files_count
        Number of data files removed.
    deleted_position_delete_files_count
        Number of position-delete files removed.
    deleted_equality_delete_files_count
        Number of equality-delete files removed.
    deleted_manifest_files_count
        Number of manifest files removed.
    deleted_manifest_lists_count
        Number of manifest-list files removed.
    deleted_statistics_files_count
        Number of statistics (and partition-statistics) files removed.
    deleted_metadata_files_count
        Number of table-metadata files removed when metadata cleanup is enabled.
    """

    deleted_data_files_count: int = 0
    deleted_position_delete_files_count: int = 0
    deleted_equality_delete_files_count: int = 0
    deleted_manifest_files_count: int = 0
    deleted_manifest_lists_count: int = 0
    deleted_statistics_files_count: int = 0
    deleted_metadata_files_count: int = 0


class ExpireSnapshotsFailedException(RuntimeError):
    """Raised when expire_snapshots cannot commit or make forward progress."""


def run(
    table: PyIcebergTable,
    *,
    older_than: _dt.datetime | int | None = None,
    retain_last: int | None = None,
    snapshot_ids: list[int] | None = None,
    clean_expired_files: bool = True,
    clean_expired_metadata: bool = False,
    stream_results: bool = False,
    options: MaintenanceOptions | None = None,
) -> ExpireResult:
    """Expire snapshots by retention and delete the files only they referenced.

    The retention plan is committed with bounded retry; when file cleanup is
    enabled, the files referenced before the commit and not after it are
    deleted. Table-metadata files are removed only when
    ``clean_expired_metadata`` is set, and the current one is always kept.

    Parameters
    ----------
    table
        Table to expire snapshots from.
    older_than
        Age cutoff as a datetime or epoch milliseconds; defaults to the table's
        maximum snapshot age.
    retain_last
        Minimum number of ancestors every branch keeps; defaults to the table's
        minimum.
    snapshot_ids
        Snapshots to expire regardless of retention.
    clean_expired_files
        Delete the data, delete, manifest, manifest-list and statistics files
        that become unreachable.
    clean_expired_metadata
        Also delete table-metadata files that are no longer referenced.
    stream_results
        Pull deletion candidates one partition at a time.
    options
        ``max-concurrent-deletes``, ``delete-num-retries`` and
        ``delete-backoff-base-seconds``.

    Returns:
    -------
    ExpireResult
        Counts of files removed, by kind.

    Raises:
    ------
    ValueError
        If ``gc.enabled`` is false, ``retain_last`` is below one, or a named
        snapshot is unknown or protected.
    ExpireSnapshotsFailedException
        If the metadata commit cannot land within the retry budget.
    """
    opts = options or {}
    max_concurrent_deletes = option_int(opts, "max-concurrent-deletes", DEFAULT_MAX_CONCURRENT_DELETES)
    delete_num_retries = option_int(opts, "delete-num-retries", DEFAULT_DELETE_NUM_RETRIES)
    delete_backoff_base = option_float(opts, "delete-backoff-base-seconds", DEFAULT_DELETE_BACKOFF_BASE_SECONDS)

    validate_gc_enabled(table)

    if retain_last is not None and retain_last < 1:
        raise ValueError(f"retain_last must be >= 1, got {retain_last!r}")

    plan = plan_expiry(
        table,
        older_than=older_than,
        retain_last=retain_last,
        snapshot_ids=snapshot_ids,
        now_ms=int(time.time() * 1000),
    )
    expired_ids = set(plan.snapshot_ids)

    if not expired_ids and not plan.ref_names:
        return ExpireResult()

    # Only files no retained snapshot reaches may go.
    pre_frame: DataFrame | None = None
    if clean_expired_files:
        try:
            pre_frame = _expire_file_frame(table)
        except Exception as exc:
            if not is_not_found(exc):
                raise
            # An unreadable manifest leaves the candidate set unknown, so nothing is deleted.
            logger.warning(
                "expire_snapshots: cannot enumerate referenced files (%r); expiring snapshots without file cleanup",
                exc,
            )
    pre_metadata: set[str] | None = None
    if clean_expired_metadata:
        pre_metadata = set(_metadata_file_paths(table))

    _commit_expire(table, expired_ids, plan.ref_names)

    if not clean_expired_files and not clean_expired_metadata:
        return ExpireResult()

    table.refresh()

    counts: dict[str, int] = {}
    if clean_expired_files and pre_frame is not None:
        post_frame = _expire_file_frame(table)
        to_delete = anti_join_paths(pre_frame, post_frame, on="path")
        counts, _failed, _sample, _total = engine_delete(
            table,
            to_delete,
            has_kind=True,
            dry_run=False,
            stream=stream_results,
            sample_limit=0,
            max_concurrent_deletes=max_concurrent_deletes,
            num_retries=delete_num_retries,
            backoff_base=delete_backoff_base,
            op_name="expire_snapshots",
        )

    deleted_metadata = 0
    if clean_expired_metadata and pre_metadata is not None:
        deleted_metadata = _clean_metadata(
            table=table,
            pre_metadata=pre_metadata,
            max_concurrent_deletes=max_concurrent_deletes,
            delete_num_retries=delete_num_retries,
            delete_backoff_base=delete_backoff_base,
        )

    return ExpireResult(
        deleted_data_files_count=counts.get(KIND_DATA, 0),
        deleted_position_delete_files_count=counts.get(KIND_POS_DELETE, 0),
        deleted_equality_delete_files_count=counts.get(KIND_EQ_DELETE, 0),
        deleted_manifest_files_count=counts.get(KIND_MANIFEST, 0),
        deleted_manifest_lists_count=counts.get(KIND_MANIFEST_LIST, 0),
        deleted_statistics_files_count=counts.get(KIND_STATS, 0),
        deleted_metadata_files_count=deleted_metadata,
    )


def _expire_file_frame(table: PyIcebergTable) -> DataFrame:
    """Build a ``(path, kind)`` frame of every file reachable from any snapshot."""
    content = content_frame(table.inspect.all_files(), path_col="file_path", content_col="content")
    manifests = manifest_frame(table.inspect.all_manifests(), path_col="path")
    extra: list[tuple[str, str]] = []
    md = table.metadata
    for snap in md.snapshots:
        ml = getattr(snap, "manifest_list", None)
        if ml:
            extra.append((ml, KIND_MANIFEST_LIST))
    for s in getattr(md, "statistics", []) or []:
        extra.append((s.statistics_path, KIND_STATS))
    for s in getattr(md, "partition_statistics", []) or []:
        extra.append((s.statistics_path, KIND_STATS))
    return union_paths(content, manifests, paths_frame(extra))


def _metadata_file_paths(table: PyIcebergTable) -> list[str]:
    """Return the table-metadata files recorded by the table, plus the current one."""
    md = table.metadata
    out: list[str] = []
    for entry in getattr(md, "metadata_log", []) or []:
        if entry.metadata_file:
            out.append(entry.metadata_file)
    current = getattr(table, "metadata_location", None)
    if current:
        out.append(current)
    return out


def _clean_metadata(
    *,
    table: PyIcebergTable,
    pre_metadata: set[str],
    max_concurrent_deletes: int,
    delete_num_retries: int,
    delete_backoff_base: float,
) -> int:
    """Delete metadata files no longer referenced after expiry; keep the current one."""
    survivors = set(_metadata_file_paths(table))
    stale = [p for p in pre_metadata if p not in survivors]
    if not stale:
        return 0
    md_counts, _failed = delete_files(
        table=table,
        to_delete=((p, KIND_METADATA) for p in stale),
        max_concurrent_deletes=max_concurrent_deletes,
        num_retries=delete_num_retries,
        backoff_base=delete_backoff_base,
        op_name="expire_snapshots_metadata",
    )
    return md_counts.get(KIND_METADATA, 0)


@dataclass(frozen=True)
class ExpiryPlan:
    """What one expiry will remove.

    Parameters
    ----------
    snapshot_ids
        Snapshots to expire.
    ref_names
        Tags and branches whose age exceeds their maximum, removed first so the
        snapshots they held can expire.
    protected_ids
        Snapshots that remain the head of a retained tag or branch and can
        never be expired.
    """

    snapshot_ids: frozenset[int]
    ref_names: tuple[str, ...]
    protected_ids: frozenset[int]


def plan_expiry(
    table: PyIcebergTable,
    *,
    older_than: _dt.datetime | int | None,
    retain_last: int | None,
    snapshot_ids: list[int] | None,
    now_ms: int,
) -> ExpiryPlan:
    """Resolve which snapshots and references an expiry removes.

    Each branch keeps its head and its ancestors while fewer than the minimum
    have been kept or the ancestor is at or after the cutoff, stopping at the
    first that is neither; a branch's own settings override ``retain_last`` and
    ``older_than``. A tag or branch older than its maximum reference age is
    removed, and a snapshot no retained reference reaches is kept only while it
    is at or after the cutoff. Snapshots in ``snapshot_ids`` expire regardless
    of retention.

    Raises:
    ------
    ValueError
        If a named snapshot does not exist or heads a retained reference.
    """
    from pyiceberg.table.refs import MAIN_BRANCH, SnapshotRefType
    from pyiceberg.table.snapshots import ancestors_of

    metadata = table.metadata
    properties = table.properties
    default_cutoff = (
        _to_epoch_millis(older_than)
        if older_than is not None
        else now_ms - int(properties.get(MAX_SNAPSHOT_AGE_MS_KEY, _DEFAULT_MAX_SNAPSHOT_AGE_MS))
    )
    default_min_keep = (
        retain_last
        if retain_last is not None
        else int(properties.get(MIN_SNAPSHOTS_TO_KEEP_KEY, _DEFAULT_MIN_SNAPSHOTS_TO_KEEP))
    )
    default_max_ref_age = int(properties.get(MAX_REF_AGE_MS_KEY, _DEFAULT_MAX_REF_AGE_MS))

    retained_refs: dict[str, SnapshotRef] = {}
    expired_refs: list[str] = []
    for name, ref in metadata.refs.items():
        if name == MAIN_BRANCH:
            retained_refs[name] = ref
            continue
        head = metadata.snapshot_by_id(ref.snapshot_id)
        if head is None:
            expired_refs.append(name)
            continue
        max_ref_age = ref.max_ref_age_ms if ref.max_ref_age_ms is not None else default_max_ref_age
        if now_ms - head.timestamp_ms <= max_ref_age:
            retained_refs[name] = ref
        else:
            expired_refs.append(name)

    protected = {ref.snapshot_id for ref in retained_refs.values()}
    explicit = set(snapshot_ids or [])
    _validate_explicit_snapshot_ids(table, sorted(explicit), protected)

    retained = set(protected)
    referenced: set[int] = set()
    for ref in retained_refs.values():
        head = metadata.snapshot_by_id(ref.snapshot_id)
        if head is None:
            continue
        if ref.snapshot_ref_type != SnapshotRefType.BRANCH:
            referenced.add(head.snapshot_id)
            continue
        cutoff = now_ms - ref.max_snapshot_age_ms if ref.max_snapshot_age_ms is not None else default_cutoff
        min_keep = ref.min_snapshots_to_keep if ref.min_snapshots_to_keep is not None else default_min_keep
        kept = 0
        keeping = True
        for ancestor in ancestors_of(head, metadata):
            referenced.add(ancestor.snapshot_id)
            if keeping and (kept < min_keep or ancestor.timestamp_ms >= cutoff):
                retained.add(ancestor.snapshot_id)
                kept += 1
            else:
                keeping = False

    for snapshot in metadata.snapshots:
        if snapshot.snapshot_id not in referenced and snapshot.timestamp_ms >= default_cutoff:
            retained.add(snapshot.snapshot_id)

    expired = {s.snapshot_id for s in metadata.snapshots if s.snapshot_id not in retained} | explicit
    return ExpiryPlan(
        snapshot_ids=frozenset(expired),
        ref_names=tuple(expired_refs),
        protected_ids=frozenset(protected),
    )


def _validate_explicit_snapshot_ids(table: PyIcebergTable, snapshot_ids: list[int], protected_ids: set[int]) -> None:
    """Reject explicit snapshot ids that are unknown or head a retained reference."""
    if not snapshot_ids:
        return
    known = {s.snapshot_id for s in table.metadata.snapshots}
    missing = [sid for sid in snapshot_ids if sid not in known]
    if missing:
        raise ValueError(f"snapshot_ids do not exist: {missing!r}")
    illegal = [sid for sid in snapshot_ids if sid in protected_ids]
    if illegal:
        raise ValueError(f"snapshot_ids are protected by a branch/tag ref and cannot be expired: {illegal!r}")


def _to_epoch_millis(value: _dt.datetime | int) -> int:
    """Return ``value`` as epoch milliseconds, treating a naive datetime as UTC."""
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=_dt.timezone.utc)
        return int(value.timestamp() * 1000)
    return int(value)


def _commit_expire(table: PyIcebergTable, expired_ids: set[int], ref_names: tuple[str, ...]) -> None:
    """Commit the reference removals and the snapshot expiry with bounded OCC retry.

    References go first, because a snapshot a tag or branch points at cannot expire.
    """
    from pyiceberg.table.refs import SnapshotRefType

    pending_ids = sorted(expired_ids)
    pending_refs = list(ref_names)
    sentinel = object()

    def _attempt(_: int) -> object:
        nonlocal pending_ids, pending_refs
        if pending_refs:
            with table.manage_snapshots() as manage:
                for name in pending_refs:
                    ref = table.metadata.refs.get(name)
                    if ref is None:
                        continue
                    if ref.snapshot_ref_type == SnapshotRefType.TAG:
                        manage.remove_tag(name)
                    else:
                        manage.remove_branch(name)
            pending_refs = []
            table.refresh()
        if pending_ids:
            expire = table.maintenance.expire_snapshots()
            # The library re-validates the whole metadata per id; the plan already did.
            expire._snapshot_ids_to_expire.update(pending_ids)
            expire.commit()
        return sentinel

    def _on_conflict(t: PyIcebergTable) -> object | None:
        nonlocal pending_ids, pending_refs
        known = {s.snapshot_id for s in t.metadata.snapshots}
        protected = {ref.snapshot_id for ref in t.metadata.refs.values()}
        pending_refs = [name for name in pending_refs if name in t.metadata.refs]
        pending_ids = [sid for sid in pending_ids if sid in known and sid not in protected]
        if not pending_ids and not pending_refs:
            return sentinel
        return None

    try:
        commit_with_retry(
            table,
            _attempt,
            op_name="expire_snapshots",
            on_conflict=_on_conflict,
        )
    except CommitRetryExhausted as exc:
        raise ExpireSnapshotsFailedException(
            "expire_snapshots: metadata commit could not land within the retry budget"
        ) from exc
