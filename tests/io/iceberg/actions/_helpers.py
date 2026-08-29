"""Shared helpers for Iceberg action tests.

Importable from any test file in this directory. Keep this module pure
Python (no pyiceberg imports at top level beyond what the helpers
themselves need at call time).
"""

from __future__ import annotations

import threading
from typing import Any

import pyarrow as pa

# Field id the spec reserves for the data file path inside a positional delete.
_DELETE_FILE_PATH_FIELD_ID = 2147483546


def strip_scheme(path: str) -> str:
    """Strip a leading ``file://`` from a URI so :mod:`os.path` can read it."""
    return path[len("file://") :] if path.startswith("file://") else path


def scan_file_count(table: Any) -> int:
    """Return the number of live data files reachable from the current snapshot."""
    return len(list(table.scan().plan_files()))


def scan_paths(table: Any) -> set[str]:
    """Return the set of live data file paths reachable from the current snapshot."""
    return {t.file.file_path for t in table.scan().plan_files()}


def read_ids(table: Any) -> list[int]:
    """Return the sorted ``id`` column of the table's live rows."""
    return sorted(int(r["id"]) for r in table.scan().to_arrow().to_pylist())


def snapshot_count(table: Any) -> int:
    """Return the number of snapshots in the table's metadata log."""
    return len(table.metadata.snapshots or [])


def _row_count(table: Any) -> int:
    table.refresh()
    return table.scan().to_arrow().num_rows


class Appender(threading.Thread):
    """Background thread that calls ``table.append`` on a fixed cadence.

    Exposes :attr:`first_commit_event` so tests can wait for the first
    append to land deterministically rather than relying on ``time.sleep``.
    """

    def __init__(
        self,
        table: Any,
        *,
        interval_s: float,
        batch_rows: int = 10,
        max_commits: int | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self._table = table
        self._interval = interval_s
        self._batch = batch_rows
        self._max_commits = max_commits
        self._stop_event = threading.Event()
        self.first_commit_event = threading.Event()
        self.commits = 0
        self.errors: list[BaseException] = []

    def stop(self, timeout: float = 15.0) -> None:
        self._stop_event.set()
        self.join(timeout=timeout)

    def wait_for_first_commit(self, timeout: float = 10.0) -> bool:
        return self.first_commit_event.wait(timeout=timeout)

    def run(self) -> None:
        next_id = 1_000_000
        while not self._stop_event.is_set():
            try:
                self._table.refresh()
                self._table.append(
                    pa.table(
                        {
                            "id": pa.array(
                                list(range(next_id, next_id + self._batch)),
                                type=pa.int64(),
                            ),
                            "label": pa.array(["live"] * self._batch, type=pa.string()),
                        }
                    )
                )
                self.commits += 1
                if not self.first_commit_event.is_set():
                    self.first_commit_event.set()
                next_id += self._batch
            except BaseException as exc:
                self.errors.append(exc)
            if self._max_commits is not None and self.commits >= self._max_commits:
                # Quiesce after a bounded burst so a concurrent maintenance
                # commit has a window to land; the appends already overlapped it.
                break
            self._stop_event.wait(self._interval)


def make_seeded_table(
    catalog: Any,
    name: str,
    *,
    n_files: int = 6,
    rows_per_file: int = 100,
) -> Any:
    """Create an unpartitioned ``(id, label)`` table with ``n_files`` appends."""
    from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC
    from pyiceberg.schema import Schema
    from pyiceberg.types import LongType, NestedField, StringType

    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "label", StringType(), required=False),
    )
    table = catalog.create_table(
        identifier=name,
        schema=schema,
        partition_spec=UNPARTITIONED_PARTITION_SPEC,
    )
    for i in range(n_files):
        start = i * rows_per_file
        table.append(
            pa.table(
                {
                    "id": pa.array(list(range(start, start + rows_per_file)), type=pa.int64()),
                    "label": pa.array(["seed"] * rows_per_file, type=pa.string()),
                }
            )
        )
    return table


def commit_positional_deletes(table: Any, deletes: dict[str, list[int]]) -> str:
    """Commit a merge-on-read delete naming rows by file path and position.

    The delete does not remove the data files it applies to; it adds a delete
    file that names them, which is exactly the conflict a rewrite has to notice.
    Writing one is the only way to reproduce that conflict here, since the
    catalog library reads delete files but does not write them.

    ``deletes`` maps a data file path to the row positions to remove from it.
    The table must be format version 2 or later. Returns the path of the delete
    file that was written.
    """
    import io
    import uuid

    import pyarrow.parquet as pq
    from pyiceberg.manifest import (
        DataFile,
        DataFileContent,
        FileFormat,
        ManifestContent,
        ManifestEntry,
        ManifestEntryStatus,
        ManifestWriterV2,
    )
    from pyiceberg.schema import Schema
    from pyiceberg.table.update.snapshot import Operation, _SnapshotProducer
    from pyiceberg.typedef import Record
    from pyiceberg.types import LongType, NestedField, StringType

    # The spec types a delete's row position as a 64-bit integer. Build the
    # schema here rather than reusing the library's, which declares a 32-bit one
    # and produces files other readers reject.
    delete_schema = Schema(
        NestedField(2147483546, "file_path", StringType(), required=False),
        NestedField(2147483545, "pos", LongType(), required=False),
    )
    delete_arrow_schema = pa.schema(
        [
            pa.field("file_path", pa.large_string(), metadata={b"PARQUET:field_id": b"2147483546"}),
            pa.field("pos", pa.int64(), metadata={b"PARQUET:field_id": b"2147483545"}),
        ]
    )

    table.refresh()
    spec = table.spec()
    rows = [{"file_path": path, "pos": pos} for path, positions in deletes.items() for pos in sorted(positions)]

    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows, schema=delete_arrow_schema), buffer)
    payload = buffer.getvalue()
    delete_path = f"{table.location()}/data/positional-delete-{uuid.uuid4().hex}.parquet"
    with table.io.new_output(delete_path).create(overwrite=True) as handle:
        handle.write(payload)

    delete_file = DataFile.from_args(
        _table_format_version=table.metadata.format_version,
        content=DataFileContent.POSITION_DELETES,
        file_path=delete_path,
        file_format=FileFormat.PARQUET,
        partition=Record(),
        file_size_in_bytes=len(payload),
        sort_order_id=None,
        spec_id=spec.spec_id,
        equality_ids=None,
        key_metadata=None,
        record_count=len(rows),
        # Bounds over the path column say which data files the delete can cover,
        # which is what lets a reader or a rewrite skip it without opening it.
        lower_bounds={_DELETE_FILE_PATH_FIELD_ID: min(deletes).encode("utf-8")},
        upper_bounds={_DELETE_FILE_PATH_FIELD_ID: max(deletes).encode("utf-8")},
    )

    class _DeleteManifestWriter(ManifestWriterV2):
        """A manifest holding delete files rather than data files."""

        def content(self) -> ManifestContent:
            return ManifestContent.DELETES

        @property
        def _meta(self) -> dict[str, str]:
            return {**super()._meta, "content": "deletes"}

    class _AddDeleteFile(_SnapshotProducer):  # type: ignore[misc, valid-type]
        """Adds one delete manifest and keeps every existing manifest as it is."""

        def _existing_manifests(self) -> list[Any]:
            snapshot = self._transaction.table_metadata.current_snapshot()
            return list(snapshot.manifests(self._io)) if snapshot is not None else []

        def _deleted_entries(self) -> list[Any]:
            return []

        def _manifests(self) -> list[Any]:
            with _DeleteManifestWriter(
                spec,
                delete_schema,
                self.new_manifest_output(),
                self._snapshot_id,
                self._compression,
            ) as writer:
                writer.add_entry(
                    ManifestEntry.from_args(
                        status=ManifestEntryStatus.ADDED,
                        snapshot_id=self._snapshot_id,
                        sequence_number=None,
                        file_sequence_number=None,
                        data_file=delete_file,
                    )
                )
            return [writer.to_manifest_file(), *self._existing_manifests()]

    with table.transaction() as transaction:
        _AddDeleteFile(
            operation=Operation.OVERWRITE,
            transaction=transaction,
            io=table.io,
            commit_uuid=uuid.uuid4(),
            snapshot_properties={},
            branch="main",
        ).commit()

    return delete_path


def commit_equality_deletes(table: Any, column: str, values: list[Any]) -> str:
    """Commit an equality delete removing every row whose ``column`` is in ``values``.

    An equality delete names the columns and values it removes rather than the
    rows' positions, so it applies to any data file in the partition. Neither
    the catalog library nor the reference engine's batch writers emit one, so it
    is written here to exercise the path against a real table rather than a
    synthetic planner input.

    Returns the path of the delete file that was written.
    """
    import io
    import uuid

    import pyarrow.parquet as pq
    from pyiceberg.manifest import (
        DataFile,
        DataFileContent,
        FileFormat,
        ManifestContent,
        ManifestEntry,
        ManifestEntryStatus,
        ManifestWriterV2,
    )
    from pyiceberg.schema import Schema
    from pyiceberg.table.update.snapshot import Operation, _SnapshotProducer
    from pyiceberg.typedef import Record

    table.refresh()
    spec = table.spec()
    field = table.schema().find_field(column)
    delete_schema = Schema(field)

    arrow_schema = pa.schema(
        [pa.field(column, pa.int64(), metadata={b"PARQUET:field_id": str(field.field_id).encode()})]
    )
    buffer = io.BytesIO()
    pq.write_table(pa.table({column: pa.array(values, type=pa.int64())}, schema=arrow_schema), buffer)
    payload = buffer.getvalue()
    delete_path = f"{table.location()}/data/equality-delete-{uuid.uuid4().hex}.parquet"
    with table.io.new_output(delete_path).create(overwrite=True) as handle:
        handle.write(payload)

    delete_file = DataFile.from_args(
        _table_format_version=table.metadata.format_version,
        content=DataFileContent.EQUALITY_DELETES,
        file_path=delete_path,
        file_format=FileFormat.PARQUET,
        partition=Record(),
        file_size_in_bytes=len(payload),
        sort_order_id=None,
        spec_id=spec.spec_id,
        equality_ids=[field.field_id],
        key_metadata=None,
        record_count=len(values),
    )

    class _DeleteManifestWriter(ManifestWriterV2):
        """A manifest holding delete files rather than data files."""

        def content(self) -> ManifestContent:
            return ManifestContent.DELETES

        @property
        def _meta(self) -> dict[str, str]:
            return {**super()._meta, "content": "deletes"}

    class _AddEqualityDelete(_SnapshotProducer):  # type: ignore[misc, valid-type]
        """Adds one delete manifest and keeps every existing manifest as it is."""

        def _existing_manifests(self) -> list[Any]:
            snapshot = self._transaction.table_metadata.current_snapshot()
            return list(snapshot.manifests(self._io)) if snapshot is not None else []

        def _deleted_entries(self) -> list[Any]:
            return []

        def _manifests(self) -> list[Any]:
            with _DeleteManifestWriter(
                spec, delete_schema, self.new_manifest_output(), self._snapshot_id, self._compression
            ) as writer:
                writer.add_entry(
                    ManifestEntry.from_args(
                        status=ManifestEntryStatus.ADDED,
                        snapshot_id=self._snapshot_id,
                        sequence_number=None,
                        file_sequence_number=None,
                        data_file=delete_file,
                    )
                )
            return [writer.to_manifest_file(), *self._existing_manifests()]

    with table.transaction() as transaction:
        _AddEqualityDelete(
            operation=Operation.OVERWRITE,
            transaction=transaction,
            io=table.io,
            commit_uuid=uuid.uuid4(),
            snapshot_properties={},
            branch="main",
        ).commit()

    return delete_path
