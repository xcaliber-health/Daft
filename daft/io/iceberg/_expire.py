"""Expire snapshots: resolve the kept set, commit the metadata change, delete unreachable files.

The set of files that become unreachable is computed by the execution engine.
The files referenced before the metadata commit (across every snapshot) minus
the files referenced after it (across the survivors) is exactly the set the
expired snapshots alone held, and is deleted. This anti-join distributes on a
cluster and streams on a single host, so it scales to very large tables.
"""

from __future__ import annotations

import datetime as _dt
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from daft.io.iceberg._common import (
    DEFAULT_DELETE_BACKOFF_BASE_SECONDS,
    DEFAULT_DELETE_NUM_RETRIES,
    DEFAULT_MAX_CONCURRENT_DELETES,
    CommitRetryExhausted,
    commit_with_retry,
    delete_files,
    is_not_found,
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
    from daft.dataframe import DataFrame
    from pyiceberg.table import Table as PyIcebergTable

logger = logging.getLogger(__name__)


MAX_SNAPSHOT_AGE_MS_KEY = "history.expire.max-snapshot-age-ms"
MIN_SNAPSHOTS_TO_KEEP_KEY = "history.expire.min-snapshots-to-keep"

_DEFAULT_MAX_SNAPSHOT_AGE_MS = 5 * 24 * 60 * 60 * 1000
_DEFAULT_MIN_SNAPSHOTS_TO_KEEP = 1


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
    options: dict[str, Any] | None = None,
) -> ExpireResult:
    opts = options or {}
    max_concurrent_deletes = int(
        opts.get("max-concurrent-deletes", DEFAULT_MAX_CONCURRENT_DELETES)
    )
    delete_num_retries = int(
        opts.get("delete-num-retries", DEFAULT_DELETE_NUM_RETRIES)
    )
    delete_backoff_base = float(
        opts.get("delete-backoff-base-seconds", DEFAULT_DELETE_BACKOFF_BASE_SECONDS)
    )

    validate_gc_enabled(table)

    if older_than is None and retain_last is None and not snapshot_ids:
        max_age_ms = int(
            table.properties.get(MAX_SNAPSHOT_AGE_MS_KEY, _DEFAULT_MAX_SNAPSHOT_AGE_MS)
        )
        older_than = int(time.time() * 1000) - max_age_ms

    if retain_last is not None and retain_last < 1:
        raise ValueError(f"retain_last must be >= 1, got {retain_last!r}")

    protected_ids = _protected_snapshot_ids(table)

    if snapshot_ids:
        _validate_explicit_snapshot_ids(table, snapshot_ids, protected_ids)

    expired_ids = _resolve_expired_ids(
        table=table,
        older_than=older_than,
        retain_last=retain_last,
        snapshot_ids=snapshot_ids,
        protected_ids=protected_ids,
    )

    if not expired_ids:
        return ExpireResult()

    # Capture what the table references now, while the expired snapshots still
    # exist; the post-commit set is subtracted from this to find what only they
    # held. Frames built from the inspected tables retain the pre-commit data.
    # If a referenced manifest or manifest list is already gone, the candidate
    # set cannot be enumerated; file cleanup is skipped (never deleting data —
    # any leftover is reclaimed by a later run) while the expiry still commits.
    pre_frame: DataFrame | None = None
    if clean_expired_files:
        try:
            pre_frame = _expire_file_frame(table)
        except Exception as exc:
            if not is_not_found(exc):
                raise
            logger.warning(
                "expire_snapshots: cannot enumerate referenced files (%r); "
                "expiring snapshots without file cleanup",
                exc,
            )
    pre_metadata: set[str] | None = None
    if clean_expired_metadata:
        pre_metadata = set(_metadata_file_paths(table))

    _commit_expire(table, expired_ids)

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
    """Build a ``(path, kind)`` frame of every file the table currently references.

    Spans the data and delete files, manifests, manifest lists, and statistics
    files reachable from all snapshots. Table-metadata files are excluded; they
    are handled separately so that retiring an old metadata pointer is not
    mistaken for a data-file deletion.
    """
    content = content_frame(
        table.inspect.all_files(), path_col="file_path", content_col="content"
    )
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
    return union_paths([content, manifests, paths_frame(extra)])


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


def _protected_snapshot_ids(table: PyIcebergTable) -> set[int]:
    from pyiceberg.table.refs import SnapshotRefType

    return {
        ref.snapshot_id
        for ref in table.metadata.refs.values()
        if ref.snapshot_ref_type in (SnapshotRefType.BRANCH, SnapshotRefType.TAG)
    }


def _validate_explicit_snapshot_ids(
    table: PyIcebergTable, snapshot_ids: list[int], protected_ids: set[int]
) -> None:
    known = {s.snapshot_id for s in table.metadata.snapshots}
    missing = [sid for sid in snapshot_ids if sid not in known]
    if missing:
        raise ValueError(f"snapshot_ids do not exist: {missing!r}")
    illegal = [sid for sid in snapshot_ids if sid in protected_ids]
    if illegal:
        raise ValueError(
            f"snapshot_ids are protected by a branch/tag ref and cannot be expired: {illegal!r}"
        )


def _resolve_expired_ids(
    *,
    table: PyIcebergTable,
    older_than: _dt.datetime | int | None,
    retain_last: int | None,
    snapshot_ids: list[int] | None,
    protected_ids: set[int],
) -> set[int]:
    """Combine all three knobs into a final expiry set, honoring table-property floors."""
    candidates: set[int] = set()

    if snapshot_ids:
        candidates.update(snapshot_ids)

    if older_than is not None:
        older_than_ms = _to_epoch_millis(older_than)
        for s in table.metadata.snapshots:
            if s.timestamp_ms < older_than_ms:
                candidates.add(s.snapshot_id)

    if retain_last is not None:
        min_keep = max(
            retain_last,
            int(
                table.properties.get(
                    MIN_SNAPSHOTS_TO_KEEP_KEY, _DEFAULT_MIN_SNAPSHOTS_TO_KEEP
                )
            ),
        )
        kept = _most_recent_main_snapshot_ids(table, min_keep)
        for s in table.metadata.snapshots:
            if s.snapshot_id not in kept:
                candidates.add(s.snapshot_id)
    else:
        min_keep = int(
            table.properties.get(
                MIN_SNAPSHOTS_TO_KEEP_KEY, _DEFAULT_MIN_SNAPSHOTS_TO_KEEP
            )
        )
        if min_keep > 0:
            kept = _most_recent_main_snapshot_ids(table, min_keep)
            candidates -= kept

    candidates -= protected_ids
    return candidates


def _most_recent_main_snapshot_ids(table: PyIcebergTable, n: int) -> set[int]:
    """Return the IDs of the N most-recent snapshots on the table's current ref chain."""
    from pyiceberg.table.snapshots import ancestors_of

    current = table.metadata.current_snapshot()
    if current is None:
        return set()
    out: list[int] = []
    for snap in ancestors_of(current, table.metadata):
        out.append(snap.snapshot_id)
        if len(out) >= n:
            break
    return set(out)


def _to_epoch_millis(value: _dt.datetime | int) -> int:
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=_dt.timezone.utc)
        return int(value.timestamp() * 1000)
    return int(value)


def _commit_expire(table: PyIcebergTable, expired_ids: set[int]) -> None:
    """Commit the snapshot-expiry metadata change with bounded OCC retry."""
    state = {"ids": sorted(expired_ids)}
    sentinel = object()

    def _attempt(_: int) -> object:
        if not state["ids"]:
            return sentinel
        table.maintenance.expire_snapshots().by_ids(state["ids"]).commit()
        return sentinel

    def _on_conflict(t: PyIcebergTable) -> object | None:
        known = {s.snapshot_id for s in t.metadata.snapshots}
        protected = _protected_snapshot_ids(t)
        state["ids"] = [sid for sid in state["ids"] if sid in known and sid not in protected]
        if not state["ids"]:
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

