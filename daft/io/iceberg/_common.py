"""Shared primitives for table maintenance.

Option parsing, the ``gc.enabled`` gate, not-found detection, the chunked
parallel delete loop, and the optimistic-concurrency commit retry helper.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, TypeAlias, TypeVar

if TYPE_CHECKING:
    from pyiceberg.manifest import ManifestContent, ManifestWriter
    from pyiceberg.partitioning import PartitionSpec
    from pyiceberg.table import Table as PyIcebergTable
    from pyiceberg.table.snapshots import Snapshot
    from pyiceberg.table.update.snapshot import _SnapshotProducer

logger = logging.getLogger(__name__)


GC_ENABLED_KEY = "gc.enabled"

DEFAULT_MAX_CONCURRENT_DELETES = 4
DEFAULT_MAX_CONCURRENT_MANIFEST_READS = 4
DEFAULT_DELETE_NUM_RETRIES = 3
DEFAULT_DELETE_BACKOFF_BASE_SECONDS = 0.1
DELETE_CHUNK_SIZE = 256

COMMIT_NUM_RETRIES_KEY = "commit.retry.num-retries"
COMMIT_MIN_WAIT_MS_KEY = "commit.retry.min-wait-ms"
COMMIT_MAX_WAIT_MS_KEY = "commit.retry.max-wait-ms"
COMMIT_TOTAL_TIMEOUT_MS_KEY = "commit.retry.total-timeout-ms"

COMMIT_DEFAULT_NUM_RETRIES = 4
COMMIT_DEFAULT_MIN_WAIT_MS = 100
COMMIT_DEFAULT_MAX_WAIT_MS = 60_000
COMMIT_DEFAULT_TOTAL_TIMEOUT_MS = 1_800_000

# Legacy names retained for existing importers.
COMMIT_MAX_ATTEMPTS = COMMIT_DEFAULT_NUM_RETRIES
COMMIT_BACKOFF_BASE_SECONDS = COMMIT_DEFAULT_MIN_WAIT_MS / 1000.0


_NOT_FOUND_EXCEPTION_NAMES = frozenset({"FileNotFoundError", "NoSuchKey", "ObjectNotFound", "BlobNotFound"})
_NOT_FOUND_MESSAGE_SUBSTRINGS = (
    "not found",
    "no such key",
    "nosuchkey",
    "no such file",
    "resource_not_found",
    "404",
)


#: One tuning knob's value: a scalar, a list of names, or a map of aliases.
OptionValue: TypeAlias = str | int | float | bool | Sequence[str] | Mapping[str, str]
#: Tuning knobs for the maintenance operations, by option name.
MaintenanceOptions: TypeAlias = Mapping[str, OptionValue]


def option_int(options: Mapping[str, OptionValue], key: str, default: int) -> int:
    """Return the integer option ``key``, or ``default`` when it is absent.

    Raises:
    ------
    ValueError
        If the value is present but is not a whole number or its text.
    """
    value = options.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{key} must be an integer, got {value!r}")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {value!r}") from exc


def option_float(options: Mapping[str, OptionValue], key: str, default: float) -> float:
    """Return the numeric option ``key``, or ``default`` when it is absent.

    Raises:
    ------
    ValueError
        If the value is present but is not a number or its text.
    """
    value = options.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{key} must be a number, got {value!r}")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number, got {value!r}") from exc


def option_bool(options: Mapping[str, OptionValue], key: str, default: bool) -> bool:
    """Return the boolean option ``key``, or ``default`` when it is absent.

    Text values ``true`` and ``false`` are accepted in any case.

    Raises:
    ------
    ValueError
        If the value is present but is neither a boolean nor such text.
    """
    value = options.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError(f"{key} must be true or false, got {value!r}")


def option_names(options: Mapping[str, OptionValue], key: str) -> list[str] | None:
    """Return the option ``key`` as a list of names, or ``None`` when absent.

    Accepts a sequence of names or one comma-separated string.

    Raises:
    ------
    ValueError
        If the value is present but is neither.
    """
    value = options.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Sequence):
        return [str(part) for part in value]
    raise ValueError(f"{key} must be a list of names, got {value!r}")


def option_mapping(options: Mapping[str, OptionValue], key: str) -> dict[str, str] | None:
    """Return the option ``key`` as a mapping of text to text, or ``None`` when absent.

    Raises:
    ------
    ValueError
        If the value is present but is not a mapping.
    """
    value = options.get(key)
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(k): str(v) for k, v in value.items()}
    raise ValueError(f"{key} must be a mapping, got {value!r}")


def scalar_options(options: Mapping[str, OptionValue] | None) -> dict[str, str | int | float | bool]:
    """Return ``options`` restricted to scalar values, for operations that take no other kind.

    Raises:
    ------
    ValueError
        Naming the first option whose value is a list or a mapping.
    """
    out: dict[str, str | int | float | bool] = {}
    for key, value in (options or {}).items():
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
        else:
            raise ValueError(f"{key} must be a scalar, got {value!r}")
    return out


def branch_ancestry(table: PyIcebergTable, branch: str | None) -> list[Snapshot]:
    """Return the snapshots on ``branch`` from its head back to the root, newest first.

    ``None`` names the table's current reference. A reference that does not
    exist, or a table with no snapshot yet, yields an empty list.
    """
    from pyiceberg.table.snapshots import ancestors_of

    head = table.snapshot_by_name(branch) if branch is not None else table.current_snapshot()
    if head is None:
        return []
    return list(ancestors_of(head, table.metadata))


def is_not_found(exc: BaseException) -> bool:
    """Return True if ``exc`` represents an object-store not-found result.

    Matches by exception class name and by case-insensitive substring of the
    message, which covers the wrapped errors object-store clients raise for a
    missing key.
    """
    if type(exc).__name__ in _NOT_FOUND_EXCEPTION_NAMES:
        return True
    msg = str(exc).lower()
    return any(s in msg for s in _NOT_FOUND_MESSAGE_SUBSTRINGS)


#: Field id and element id of ``equality_ids`` in a manifest's data-file struct.
_EQUALITY_IDS_FIELD_ID = 135
_EQUALITY_IDS_ELEMENT_ID = 136


def declare_equality_ids_as_ints() -> None:
    """Make written manifests type ``equality_ids`` as a list of ints, as the specification requires.

    The catalog library declares the element as a long. A manifest written
    that way is refused by readers that decode the field into an int array,
    so any commit recording an equality delete file, removed or kept, would
    be unreadable by them. Only the struct the library's writers build is
    corrected; its read schema keeps the long, which resolves manifests of
    either encoding.
    """
    from pyiceberg import manifest
    from pyiceberg.types import IntegerType, ListType, NestedField, StructType

    original = manifest.data_file_with_partition
    if getattr(original, "__name__", "") == "_data_file_with_int_equality_ids":
        return

    def _data_file_with_int_equality_ids(partition_type: StructType, format_version: int) -> StructType:
        struct = original(partition_type, format_version)
        fields = []
        for field in struct.fields:
            if field.field_id == _EQUALITY_IDS_FIELD_ID:
                field = NestedField(
                    field_id=field.field_id,
                    name=field.name,
                    field_type=ListType(
                        element_id=_EQUALITY_IDS_ELEMENT_ID, element_type=IntegerType(), element_required=True
                    ),
                    required=field.required,
                    doc=field.doc,
                )
            fields.append(field)
        return StructType(*fields)

    manifest.data_file_with_partition = _data_file_with_int_equality_ids


declare_equality_ids_as_ints()


def manifest_writer_for(producer: _SnapshotProducer, content: ManifestContent, spec: PartitionSpec) -> ManifestWriter:
    """Return a manifest writer of ``content`` for the snapshot ``producer`` is building.

    A manifest declares whether it lists data files or delete files, and
    readers trust the declaration, so a manifest of delete files must be
    written by a writer that declares delete content.
    """
    from pyiceberg.manifest import ManifestContent, ManifestWriterV2, write_manifest

    metadata = producer._transaction.table_metadata
    if content != ManifestContent.DELETES:
        return write_manifest(
            format_version=metadata.format_version,
            spec=spec,
            schema=metadata.schema(),
            output_file=producer.new_manifest_output(),
            snapshot_id=producer._snapshot_id,
            avro_compression=producer._compression,
        )

    class _DeleteManifestWriter(ManifestWriterV2):  # type: ignore[misc]
        """A manifest whose entries are delete files."""

        def content(self) -> ManifestContent:
            return ManifestContent.DELETES

        @property
        def _meta(self) -> dict[str, str]:
            return {**super()._meta, "content": "deletes"}

    return _DeleteManifestWriter(
        spec, metadata.schema(), producer.new_manifest_output(), producer._snapshot_id, producer._compression
    )


def validate_gc_enabled(table: PyIcebergTable) -> None:
    """Refuse to run if the ``gc.enabled`` table property is explicitly false.

    Parameters
    ----------
    table
        Iceberg table whose properties are read.

    Raises:
    ------
    ValueError
        If ``table.properties[gc.enabled]`` resolves to false/0/no.
    """
    raw = table.properties.get(GC_ENABLED_KEY, "true")
    if str(raw).strip().lower() in {"false", "0", "no"}:
        raise ValueError(
            f"refusing to run: table property {GC_ENABLED_KEY}=false. Set it to true to permit physical file deletion."
        )


K = TypeVar("K", bound=Hashable)


def delete_files(
    *,
    table: PyIcebergTable,
    to_delete: Iterable[tuple[str, K]],
    max_concurrent_deletes: int = DEFAULT_MAX_CONCURRENT_DELETES,
    num_retries: int = DEFAULT_DELETE_NUM_RETRIES,
    backoff_base: float = DEFAULT_DELETE_BACKOFF_BASE_SECONDS,
    op_name: str = "iceberg-op",
) -> tuple[dict[K, int], int]:
    """Delete files in chunked parallel batches.

    Parameters
    ----------
    table
        Iceberg table whose ``io.delete`` performs the deletions.
    to_delete
        Iterable of ``(path, kind)`` pairs. ``kind`` is an opaque hashable
        used only as the key in the returned counts.
    max_concurrent_deletes
        Upper bound on the worker pool size.
    num_retries
        Per-file retry budget on transient errors. NotFound results are
        treated as success on the first attempt.
    backoff_base
        Exponential-backoff base in seconds: attempt ``n`` sleeps for
        ``backoff_base * 2**n`` seconds before the next try.
    op_name
        Used only in the warning log emitted on terminal delete failures.

    Returns:
    -------
    tuple of (counts, failed)
        ``counts`` maps each ``kind`` to the number of successful deletes for
        that kind. ``failed`` is the total number of paths that exhausted the
        retry budget.
    """
    counts: dict[K, int] = {}
    failed = 0
    io = table.io

    def _delete_one(item: tuple[str, K]) -> tuple[str, K, bool]:
        path, kind = item
        for attempt in range(num_retries + 1):
            try:
                io.delete(path)
                return path, kind, True
            except FileNotFoundError:
                return path, kind, True
            except Exception as exc:
                if is_not_found(exc):
                    return path, kind, True
                if attempt < num_retries:
                    time.sleep(backoff_base * (2**attempt))
                    continue
                logger.warning("%s: failed to delete %s: %r", op_name, path, exc)
                return path, kind, False
        return path, kind, False

    workers = max(1, max_concurrent_deletes)
    iterator = iter(to_delete)

    if workers == 1:
        for item in iterator:
            _, kind, ok = _delete_one(item)
            if ok:
                counts[kind] = counts.get(kind, 0) + 1
            else:
                failed += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while True:
                chunk: list[tuple[str, K]] = []
                for _ in range(DELETE_CHUNK_SIZE):
                    try:
                        chunk.append(next(iterator))
                    except StopIteration:
                        break
                if not chunk:
                    break
                for _, kind, ok in pool.map(_delete_one, chunk):
                    if ok:
                        counts[kind] = counts.get(kind, 0) + 1
                    else:
                        failed += 1

    return counts, failed


T = TypeVar("T")


class CommitRetryExhausted(RuntimeError):
    """Raised when an Iceberg commit cannot land within the configured retry budget.

    Parameters
    ----------
    op_name
        Identifier of the maintenance operation that gave up.
    attempts
        Number of commit attempts made before exhaustion.
    elapsed_ms
        Wall time in milliseconds spent across attempts and waits.
    """

    def __init__(self, op_name: str, attempts: int, elapsed_ms: int) -> None:
        super().__init__(
            f"{op_name}: commit retry budget exhausted after {attempts} attempt(s), {elapsed_ms}ms elapsed"
        )
        self.op_name = op_name
        self.attempts = attempts
        self.elapsed_ms = elapsed_ms


def _read_retry_policy(table: PyIcebergTable) -> tuple[int, float, float, float]:
    """Read the commit-retry policy from table properties."""
    props = table.properties
    num_retries = max(
        0,
        int(props.get(COMMIT_NUM_RETRIES_KEY, COMMIT_DEFAULT_NUM_RETRIES)),
    )
    min_wait_ms = max(
        0,
        int(props.get(COMMIT_MIN_WAIT_MS_KEY, COMMIT_DEFAULT_MIN_WAIT_MS)),
    )
    max_wait_ms = max(
        min_wait_ms,
        int(props.get(COMMIT_MAX_WAIT_MS_KEY, COMMIT_DEFAULT_MAX_WAIT_MS)),
    )
    total_timeout_ms = max(
        0,
        int(props.get(COMMIT_TOTAL_TIMEOUT_MS_KEY, COMMIT_DEFAULT_TOTAL_TIMEOUT_MS)),
    )
    return (
        num_retries,
        min_wait_ms / 1000.0,
        max_wait_ms / 1000.0,
        total_timeout_ms / 1000.0,
    )


def commit_with_retry(
    table: PyIcebergTable,
    attempt_fn: Callable[[int], T],
    *,
    op_name: str,
    on_conflict: Callable[[PyIcebergTable], T | None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    rng: Callable[[], float] = random.random,
) -> T:
    """Invoke ``attempt_fn`` with optimistic-concurrency retry.

    The loop retries on optimistic-concurrency commit failures using
    exponential backoff with full jitter, bounded by both the attempt count
    and the total wall-time budget read from the table properties
    ``commit.retry.num-retries``, ``commit.retry.min-wait-ms``,
    ``commit.retry.max-wait-ms`` and ``commit.retry.total-timeout-ms``.

    Parameters
    ----------
    table
        The table whose properties supply the retry policy and whose
        ``refresh()`` is called between attempts.
    attempt_fn
        Callable receiving the zero-based attempt index. Returns the result
        of the operation on success.
    op_name
        Operation label used in the exhausted-retry exception.
    on_conflict
        Optional callback invoked after ``table.refresh()`` between attempts.
        May return a non-``None`` value to short-circuit the loop (e.g. when
        the work is now a no-op or a prior attempt's commit already landed).
    sleep, monotonic, rng
        Injection points to make the helper deterministic in tests.

    Returns:
    -------
    T
        The value returned by ``attempt_fn`` on a successful attempt, or the
        value returned by ``on_conflict`` if it short-circuits.

    Raises:
    ------
    CommitRetryExhausted
        When the retry budget is exhausted without a successful commit.
    """
    from pyiceberg.exceptions import CommitFailedException

    num_retries, min_wait_s, max_wait_s, total_timeout_s = _read_retry_policy(table)
    start = monotonic()
    last_err: BaseException | None = None
    attempts_made = 0

    for attempt in range(num_retries + 1):
        attempts_made = attempt + 1
        try:
            return attempt_fn(attempt)
        except CommitFailedException as exc:
            last_err = exc
            elapsed = monotonic() - start
            if attempt >= num_retries or elapsed >= total_timeout_s:
                break
            base = min(max_wait_s, min_wait_s * (2**attempt)) if min_wait_s > 0 else 0.0
            wait = base * (1.0 + rng()) if base > 0 else 0.0
            remaining = max(0.0, total_timeout_s - elapsed)
            if wait > remaining:
                wait = remaining
            if wait > 0:
                sleep(wait)
            table.refresh()
            if on_conflict is not None:
                short_circuit = on_conflict(table)
                if short_circuit is not None:
                    return short_circuit

    elapsed_ms = int((monotonic() - start) * 1000)
    raise CommitRetryExhausted(
        op_name=op_name,
        attempts=attempts_made,
        elapsed_ms=elapsed_ms,
    ) from last_err
