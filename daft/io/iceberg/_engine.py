"""Execution-engine primitives shared by the maintenance file-cleanup paths.

The expensive stages of snapshot expiry and orphan-file removal — enumerating the
files a table still references, listing the files physically present under its
location, and computing the difference — are expressed here as DataFrame
operations so they distribute on a cluster and stream on a single host with
bounded memory.

Path canonicalization runs inside the join key: two spellings of the same
physical location (an aliased scheme such as ``s3a`` for ``s3``, or an aliased
host) must compare equal, otherwise a live file would be flagged for deletion.
A pure-Python ``canonical`` mirrors the engine expression for testing.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from daft.io.iceberg._common import (
    DEFAULT_DELETE_BACKOFF_BASE_SECONDS,
    DEFAULT_DELETE_NUM_RETRIES,
    DEFAULT_MAX_CONCURRENT_DELETES,
    delete_files,
)

if TYPE_CHECKING:
    import pyarrow as pa
    from pyiceberg.table import Table as PyIcebergTable

    from daft.dataframe import DataFrame
    from daft.io import IOConfig

logger = logging.getLogger(__name__)

# File-kind tags carried on the path frames so per-kind counts survive the join.
KIND_DATA = "data"
KIND_POS_DELETE = "pos_delete"
KIND_EQ_DELETE = "eq_delete"
KIND_MANIFEST = "manifest"
KIND_MANIFEST_LIST = "manifest_list"
KIND_STATS = "stats"
KIND_METADATA = "metadata"
KIND_FILE = "file"

_DEFAULT_SCHEME_ALIASES = {"s3a": "s3", "s3n": "s3"}
_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+\-.]*)://")
# scheme://authority, capturing everything up to the first path separator.
_PREFIX_PATTERN = r"^([a-zA-Z][a-zA-Z0-9+\-.]*://[^/]*)"


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CanonSpec:
    """Scheme and authority equivalences used to canonicalize a path.

    Parameters
    ----------
    scheme_aliases
        Maps an alias scheme to its canonical spelling (lower-cased).
    authority_aliases
        Maps an alias authority (host[:port]) to its canonical spelling.
    """

    scheme_aliases: dict[str, str] = field(default_factory=dict)
    authority_aliases: dict[str, str] = field(default_factory=dict)

    def canonical(self, path: str) -> str:
        """Return ``path`` with its scheme and authority mapped to canonical form."""
        m = _SCHEME_RE.match(path)
        if not m:
            return path
        scheme = m.group(1).lower()
        scheme = self.scheme_aliases.get(scheme, scheme)
        rest = path[m.end() :]
        slash = rest.find("/")
        if slash < 0:
            authority, body = rest, ""
        else:
            authority, body = rest[:slash], rest[slash:]
        authority = self.authority_aliases.get(authority, authority)
        return f"{scheme}://{authority}{body}"

    def prefix(self, canon_path: str) -> str:
        """Return the ``scheme://authority`` prefix of a canonical path."""
        idx = canon_path.find("://")
        if idx < 0:
            return ""
        rest = canon_path[idx + 3 :]
        slash = rest.find("/")
        authority = rest if slash < 0 else rest[:slash]
        return f"{canon_path[:idx]}://{authority}"


def build_canon_spec(
    equal_schemes: dict[str, str] | None,
    equal_authorities: dict[str, str] | None,
) -> CanonSpec:
    """Build a canonicalization spec, extending the built-in scheme equivalences."""
    scheme_aliases = dict(_DEFAULT_SCHEME_ALIASES)
    for key, value in (equal_schemes or {}).items():
        scheme_aliases[str(key).lower()] = str(value).lower()
    authority_aliases = {str(key): str(value) for key, value in (equal_authorities or {}).items()}
    return CanonSpec(scheme_aliases=scheme_aliases, authority_aliases=authority_aliases)


def canon_path_expr(column: str, spec: CanonSpec):
    """Return an expression that canonicalizes ``column`` the way ``CanonSpec`` does.

    Scheme aliases are rewritten at the leading position; authority aliases are
    rewritten between the scheme separator and the next path separator. The
    chain is equivalent to :meth:`CanonSpec.canonical` for flat (non-chained)
    alias maps, which is the only shape produced by the option parsing.
    """
    from daft import col

    expr = col(column)
    for alias, canonical in spec.scheme_aliases.items():
        expr = expr.regexp_replace(f"^{re.escape(alias)}://", f"{canonical}://")
    for alias, canonical in spec.authority_aliases.items():
        a = re.escape(alias)
        expr = expr.regexp_replace(f"://{a}/", f"://{canonical}/")
        expr = expr.regexp_replace(f"://{a}$", f"://{canonical}")
    return expr


def prefix_expr(column: str):
    """Return an expression yielding the ``scheme://authority`` prefix of ``column``."""
    from daft import col

    return col(column).regexp_extract(_PREFIX_PATTERN, 1)


# ---------------------------------------------------------------------------
# Reachable / candidate frames
# ---------------------------------------------------------------------------
def content_frame(
    arrow_table: pa.Table,
    *,
    path_col: str,
    content_col: str,
    status_col: str | None = None,
) -> DataFrame:
    """Build a ``(path, kind)`` frame from a content-files metadata table.

    Rows whose status marks them deleted (status code 2) are dropped when a
    ``status_col`` is supplied, so retired files referenced only by a deleted
    entry are not pinned into the reachable set.
    """
    import daft
    from daft import col, lit
    from daft.functions import when

    df = daft.from_arrow(arrow_table)
    if status_col is not None:
        df = df.where(col(status_col) != lit(2))
    kind = (
        when(col(content_col) == lit(1), lit(KIND_POS_DELETE))
        .when(col(content_col) == lit(2), lit(KIND_EQ_DELETE))
        .otherwise(lit(KIND_DATA))
    )
    return df.select(col(path_col).alias("path"), kind.alias("kind")).distinct()


def manifest_frame(arrow_table: pa.Table, *, path_col: str = "path") -> DataFrame:
    """Build a ``(path, kind)`` frame of manifest files from a manifests table."""
    import daft
    from daft import col, lit

    df = daft.from_arrow(arrow_table)
    return df.select(col(path_col).alias("path"), lit(KIND_MANIFEST).alias("kind")).distinct()


def paths_frame(pairs: Iterable[tuple[str, str]]) -> DataFrame | None:
    """Build a ``(path, kind)`` frame from an in-memory list of pairs.

    Returns ``None`` when the input is empty, so callers can skip the union.
    """
    import daft

    paths: list[str] = []
    kinds: list[str] = []
    for path, kind in pairs:
        if path:
            paths.append(path)
            kinds.append(kind)
    if not paths:
        return None
    return daft.from_pydict({"path": paths, "kind": kinds})


def union_paths(frames: Iterable[DataFrame | None]) -> DataFrame | None:
    """Concatenate path frames, ignoring ``None`` entries."""
    out: DataFrame | None = None
    for f in frames:
        if f is None:
            continue
        out = f if out is None else out.union_all(f)
    return out


def with_uri_parts(df: DataFrame, spec: CanonSpec) -> DataFrame:
    """Add ``canon_path`` and ``prefix`` columns derived from the ``path`` column."""
    return df.with_column("canon_path", canon_path_expr("path", spec)).with_column("prefix", prefix_expr("canon_path"))


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------
def listed_files_frame(
    location: str,
    *,
    io_config: IOConfig | None,
    older_than_ms: int,
) -> DataFrame:
    """List files under ``location`` modified before ``older_than_ms``.

    The listing is performed by the execution engine so it distributes and
    streams. Rows whose modification time is unknown are retained and filtered
    on the driver against the same cutoff during deletion-set assembly.
    """
    import daft
    from daft import col, lit

    df = daft.from_glob_path(f"{location.rstrip('/')}/**", io_config=io_config)
    df = df.select(col("path"), col("mtime"))
    return df.where(col("mtime").is_null() | (col("mtime") < lit(older_than_ms)))


def file_list_view_frame(
    view: DataFrame,
    *,
    location: str,
    older_than_ms: int,
) -> DataFrame:
    """Adapt a caller-supplied inventory to the listing contract.

    The view must carry a string ``file_path`` column and a ``last_modified``
    column (epoch milliseconds or a timestamp). Rows outside ``location`` or
    newer than the cutoff are dropped.
    """
    from daft import col, lit

    names = view.column_names
    if "file_path" not in names or "last_modified" not in names:
        raise ValueError(f"file_list_view must contain 'file_path' and 'last_modified' columns; got {names!r}")
    df = view.select(
        col("file_path").alias("path"),
        col("last_modified").alias("mtime"),
    )
    df = df.where(col("path").startswith(lit(location.rstrip("/"))))
    return df.where(col("mtime").is_null() | (col("mtime") < lit(older_than_ms)))


# ---------------------------------------------------------------------------
# Difference
# ---------------------------------------------------------------------------
def anti_join_paths(left: DataFrame, right: DataFrame, *, on: str = "path") -> DataFrame:
    """Return rows of ``left`` whose ``on`` key has no match in ``right``."""
    return left.join(right.select(on).distinct(), on=on, how="anti")


def find_orphans(
    listed: DataFrame,
    reachable: DataFrame,
    *,
    mode: str,
) -> tuple[DataFrame, int]:
    """Compute orphan paths from canonicalized listed and reachable frames.

    Parameters
    ----------
    listed, reachable
        Frames carrying ``path``, ``canon_path`` and ``prefix`` columns.
    mode
        ``"delete"`` returns every listed path absent from the reachable set.
        ``"error"`` and ``"ignore"`` only return paths whose ``prefix`` is known
        to the reachable set; ``"error"`` additionally raises when a listed path
        shares no prefix with any reachable path.

    Returns:
    -------
    tuple of (orphans, conflicts)
        ``orphans`` carries a ``path`` column. ``conflicts`` counts listed paths
        dropped (or rejected) for prefix mismatch.
    """
    from daft import col

    not_reachable = listed.join(
        reachable.select("canon_path").distinct(),
        on="canon_path",
        how="anti",
    )
    if mode == "delete":
        return not_reachable.select(col("path")), 0

    known_prefixes = reachable.select(col("prefix")).distinct()
    matched_prefix = not_reachable.join(known_prefixes, on="prefix", how="semi")
    mismatched = not_reachable.join(known_prefixes, on="prefix", how="anti")
    conflicts = mismatched.count_rows()
    if mode == "error" and conflicts:
        sample = [r["path"] for r in mismatched.select(col("path")).limit(5).to_pylist()]
        from daft.io.iceberg._remove_orphan import PrefixMismatchError

        raise PrefixMismatchError(
            f"remove_orphan_files: {conflicts} listed file(s) use a scheme/authority "
            f"not present in the table's reachable set. Sample: {sample!r}. Pass "
            f"prefix_mismatch_mode='delete' or 'ignore' to override."
        )
    return matched_prefix.select(col("path")), conflicts


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------
def _iter_path_kind(df: DataFrame, *, has_kind: bool, stream: bool) -> Iterator[tuple[str, str]]:
    """Yield ``(path, kind)`` pairs from a result frame.

    When ``stream`` is set, partitions are pulled one at a time to bound driver
    memory; otherwise the whole result is materialized at once.
    """
    cols = ["path", "kind"] if has_kind else ["path"]
    sel = df.select(*cols)
    if not stream:
        data = sel.to_pydict()
        paths = data.get("path", [])
        kinds = data.get("kind", [KIND_FILE] * len(paths)) if has_kind else [KIND_FILE] * len(paths)
        for p, k in zip(paths, kinds):
            if p:
                yield p, k
        return
    for part in sel.iter_partitions():
        pd = _partition_to_pydict(part)
        paths = pd.get("path", [])
        kinds = pd.get("kind", [KIND_FILE] * len(paths)) if has_kind else [KIND_FILE] * len(paths)
        for p, k in zip(paths, kinds):
            if p:
                yield p, k


def _partition_to_pydict(part: object) -> dict[str, list[str]]:
    """Materialize a partition (local micro-partition or remote handle) to a dict."""
    obj = part
    if not hasattr(obj, "to_pydict"):
        import ray

        obj = ray.get(part)
    return obj.to_pydict()


def engine_delete(
    table: PyIcebergTable,
    df: DataFrame,
    *,
    has_kind: bool,
    dry_run: bool,
    stream: bool,
    sample_limit: int,
    max_concurrent_deletes: int = DEFAULT_MAX_CONCURRENT_DELETES,
    num_retries: int = DEFAULT_DELETE_NUM_RETRIES,
    backoff_base: float = DEFAULT_DELETE_BACKOFF_BASE_SECONDS,
    op_name: str = "iceberg-op",
) -> tuple[dict[str, int], int, list[str], int]:
    """Delete the paths in ``df``, returning counts, failures, a sample, and the total.

    The deletion itself runs through the shared retry-and-backoff pool, which
    treats not-found as success. With ``dry_run`` the paths are counted and
    sampled but not deleted.
    """
    pairs = _iter_path_kind(df, has_kind=has_kind, stream=stream)

    sample: list[str] = []
    total = 0

    def _tracked() -> Iterator[tuple[str, str]]:
        nonlocal total
        for path, kind in pairs:
            total += 1
            if len(sample) < sample_limit:
                sample.append(path)
            yield path, kind

    if dry_run:
        for _ in _tracked():
            pass
        return {}, 0, sample, total

    counts, failed = delete_files(
        table=table,
        to_delete=_tracked(),
        max_concurrent_deletes=max_concurrent_deletes,
        num_retries=num_retries,
        backoff_base=backoff_base,
        op_name=op_name,
    )
    return counts, failed, sample, total


def io_config_for_table(table: PyIcebergTable) -> IOConfig | None:
    """Resolve object-store access configuration recorded on the table, if any."""
    from daft.io.iceberg._iceberg import (
        _convert_iceberg_file_io_properties_to_io_config,
    )

    return _convert_iceberg_file_io_properties_to_io_config(table.io.properties)
