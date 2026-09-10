"""Delete files under the table location that no snapshot references.

The files listed under the table root minus the files reachable from any
snapshot are deleted. Paths are compared in canonical form so equivalent
spellings of one location never flag a live file as an orphan.
"""

from __future__ import annotations

import datetime as _dt
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from daft.io.iceberg._common import (
    DEFAULT_DELETE_BACKOFF_BASE_SECONDS,
    DEFAULT_DELETE_NUM_RETRIES,
    DEFAULT_MAX_CONCURRENT_DELETES,
    MaintenanceOptions,
    option_bool,
    option_float,
    option_int,
    option_mapping,
    validate_gc_enabled,
)
from daft.io.iceberg._engine import (
    KIND_MANIFEST_LIST,
    KIND_METADATA,
    KIND_STATS,
    CanonSpec,
    build_canon_spec,
    content_frame,
    engine_delete,
    file_list_view_frame,
    find_orphans,
    io_config_for_table,
    listed_files_frame,
    manifest_frame,
    paths_frame,
    union_paths,
    with_uri_parts,
)

if TYPE_CHECKING:
    from pyiceberg.table import Table as PyIcebergTable

    from daft.dataframe import DataFrame

logger = logging.getLogger(__name__)


DEFAULT_OLDER_THAN_MS = 3 * 24 * 60 * 60 * 1000
MIN_AGE_MS = 24 * 60 * 60 * 1000
DEFAULT_SAMPLE_LIMIT = 1000

_VALID_PREFIX_MODES = frozenset({"error", "delete", "ignore"})
_DEFAULT_SPEC = build_canon_spec(None, None)


@dataclass(frozen=True)
class RemoveOrphanResult:
    """Summary of a remove_orphan_files invocation.

    Parameters
    ----------
    orphan_files_count
        Number of files identified as orphans (present in the listing, absent
        from the table's reachable set).
    deleted_files_count
        Number of orphans successfully deleted. Equals ``orphan_files_count``
        unless ``dry_run=True`` or some deletes failed.
    sample_paths
        Up to ``sample-limit`` orphan paths. Useful for operator review when
        running with ``dry_run=True``.
    skipped_prefix_mismatch_count
        Number of listed files dropped from the candidate set because their
        scheme/authority did not match any reachable path and
        ``prefix_mismatch_mode`` was ``"ignore"`` (or rejected under
        ``"error"``).
    failed_deletes
        Orphans whose delete exhausted the per-file retry budget.
    """

    orphan_files_count: int = 0
    deleted_files_count: int = 0
    sample_paths: list[str] = field(default_factory=list)
    skipped_prefix_mismatch_count: int = 0
    failed_deletes: int = 0


class PrefixMismatchError(ValueError):
    """Raised when listed files use a scheme/authority absent from reachable paths."""


def run(
    table: PyIcebergTable,
    *,
    older_than: _dt.datetime | int | None = None,
    location: str | None = None,
    dry_run: bool = False,
    prefix_mismatch_mode: str = "error",
    file_list_view: DataFrame | None = None,
    prefix_listing: bool = False,
    stream_results: bool = False,
    options: MaintenanceOptions | None = None,
) -> RemoveOrphanResult:
    """Delete files under the table location that no snapshot references.

    Files listed under ``location`` and modified before ``older_than`` are
    compared, in canonical form, against every file reachable from any
    snapshot; those absent are deleted unless ``dry_run`` is set.

    Parameters
    ----------
    table
        Table whose location is cleaned.
    older_than
        Modification-time cutoff as a datetime or epoch milliseconds; defaults
        to three days ago and must be at least a day old unless
        ``allow-recent`` is set.
    location
        Subpath of the table location to clean; defaults to the whole location.
    dry_run
        Report orphans without deleting them.
    prefix_mismatch_mode
        ``"error"``, ``"delete"`` or ``"ignore"`` for listed files whose scheme
        or authority matches no reachable path.
    file_list_view
        Inventory with ``file_path`` and ``last_modified`` columns to use
        instead of listing the store.
    prefix_listing
        Accepted for interface compatibility and ignored.
    stream_results
        Pull deletion candidates one partition at a time.
    options
        ``max-concurrent-deletes``, ``delete-num-retries``,
        ``delete-backoff-base-seconds``, ``sample-limit``, ``allow-recent``,
        ``equal-schemes`` and ``equal-authorities``.

    Returns:
    -------
    RemoveOrphanResult
        Orphan, deleted and failed counts with a bounded sample of paths.

    Raises:
    ------
    ValueError
        If ``gc.enabled`` is false, the cutoff is too recent, the mode is
        unknown, or ``location`` lies outside the table location.
    PrefixMismatchError
        Under ``"error"`` mode, when a listed file's prefix matches no
        reachable path.
    """
    if prefix_mismatch_mode not in _VALID_PREFIX_MODES:
        raise ValueError(
            f"prefix_mismatch_mode must be one of {sorted(_VALID_PREFIX_MODES)}, got {prefix_mismatch_mode!r}"
        )

    opts = options or {}
    max_concurrent_deletes = option_int(opts, "max-concurrent-deletes", DEFAULT_MAX_CONCURRENT_DELETES)
    delete_num_retries = option_int(opts, "delete-num-retries", DEFAULT_DELETE_NUM_RETRIES)
    delete_backoff_base = option_float(opts, "delete-backoff-base-seconds", DEFAULT_DELETE_BACKOFF_BASE_SECONDS)
    sample_limit = option_int(opts, "sample-limit", DEFAULT_SAMPLE_LIMIT)
    allow_recent = option_bool(opts, "allow-recent", False)

    validate_gc_enabled(table)

    older_than_ms = _resolve_older_than_ms(older_than, allow_recent=allow_recent)
    base_location = _resolve_location(table, location)
    spec = _build_canonicalizer(opts)

    reachable = with_uri_parts(_reachable_frame(table), spec)
    listed = with_uri_parts(
        _listed_frame(
            table,
            location=base_location,
            older_than_ms=older_than_ms,
            file_list_view=file_list_view,
            prefix_listing=prefix_listing,
        ),
        spec,
    ).distinct("canon_path")

    orphans, mismatched = find_orphans(listed, reachable, mode=prefix_mismatch_mode)

    counts, failed, sample, total = engine_delete(
        table,
        orphans,
        has_kind=False,
        dry_run=dry_run,
        stream=stream_results,
        sample_limit=sample_limit,
        max_concurrent_deletes=max_concurrent_deletes,
        num_retries=delete_num_retries,
        backoff_base=delete_backoff_base,
        op_name="remove_orphan_files",
    )
    deleted = 0 if dry_run else sum(counts.values())
    return RemoveOrphanResult(
        orphan_files_count=total,
        deleted_files_count=deleted,
        sample_paths=sample,
        skipped_prefix_mismatch_count=mismatched,
        failed_deletes=failed,
    )


def _resolve_older_than_ms(older_than: _dt.datetime | int | None, *, allow_recent: bool) -> int:
    """Return the modification-time cutoff in epoch milliseconds, enforcing the one-day floor."""
    now_ms = int(time.time() * 1000)
    if older_than is None:
        return now_ms - DEFAULT_OLDER_THAN_MS

    if isinstance(older_than, _dt.datetime):
        if older_than.tzinfo is None:
            older_than = older_than.replace(tzinfo=_dt.timezone.utc)
        cutoff_ms = int(older_than.timestamp() * 1000)
    else:
        cutoff_ms = int(older_than)

    if not allow_recent and cutoff_ms > now_ms - MIN_AGE_MS:
        raise ValueError(
            "remove_orphan_files: refusing to use a cutoff newer than 24 hours ago "
            "(risk of deleting files written by concurrent jobs). "
            "Pass options={'allow-recent': True} only in tests."
        )
    return cutoff_ms


def _resolve_location(table: PyIcebergTable, location: str | None) -> str:
    """Return the location to list, which must lie within the table location."""
    table_loc = table.location().rstrip("/")
    if location is None:
        return table_loc
    loc = location.rstrip("/")
    table_canon = _DEFAULT_SPEC.canonical(table_loc)
    loc_canon = _DEFAULT_SPEC.canonical(loc)
    if loc_canon != table_canon and not loc_canon.startswith(table_canon + "/"):
        raise ValueError(f"location={location!r} is not a subpath of table.location()={table.location()!r}")
    return loc


def _reachable_frame(table: PyIcebergTable) -> DataFrame:
    """Build a frame of every file path the table references across all snapshots.

    Spans data and delete files, manifests, manifest lists, statistics files,
    every recorded table-metadata file, and the current metadata pointer.
    """
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
    for entry in getattr(md, "metadata_log", []) or []:
        if entry.metadata_file:
            extra.append((entry.metadata_file, KIND_METADATA))
    current_md = getattr(table, "metadata_location", None)
    if current_md:
        extra.append((current_md, KIND_METADATA))
    return union_paths(content, manifests, paths_frame(extra))


def _listed_frame(
    table: PyIcebergTable,
    *,
    location: str,
    older_than_ms: int,
    file_list_view: DataFrame | None,
    prefix_listing: bool,
) -> DataFrame:
    """Build the frame of files present under ``location``.

    The caller-supplied inventory is used when given; otherwise the store is
    listed.
    """
    if file_list_view is not None:
        return file_list_view_frame(file_list_view, location=location, older_than_ms=older_than_ms)
    del prefix_listing  # Listing is already prefix-based, so the flag changes nothing.
    io_config = io_config_for_table(table)
    return listed_files_frame(location, io_config=io_config, older_than_ms=older_than_ms)


def _build_canonicalizer(opts: MaintenanceOptions) -> CanonSpec:
    """Build a path canonicalizer from the scheme/authority equivalence options."""
    return build_canon_spec(option_mapping(opts, "equal-schemes"), option_mapping(opts, "equal-authorities"))
