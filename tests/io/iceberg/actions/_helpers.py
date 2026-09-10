"""Shared helpers for Iceberg action tests.

Importable from any test file in this directory. Keep this module pure
Python (no pyiceberg imports at top level beyond what the helpers
themselves need at call time).
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, ParamSpec, TypeVar, cast

import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest
    from pyiceberg.catalog import Catalog
    from pyiceberg.manifest import DataFile, ManifestEntry, ManifestFile
    from pyiceberg.partitioning import PartitionSpec
    from pyiceberg.table import Table
    from pyiceberg.typedef import Record

_P = ParamSpec("_P")
_R = TypeVar("_R")

# Field id the spec reserves for the data file path inside a positional delete.
_DELETE_FILE_PATH_FIELD_ID = 2147483546


def strip_scheme(path: str) -> str:
    """Strip a leading ``file://`` from a URI so :mod:`os.path` can read it."""
    return path.removeprefix("file://")


def scan_file_count(table: Table) -> int:
    """Return the number of live data files reachable from the current snapshot."""
    return len(list(table.scan().plan_files()))


def scan_paths(table: Table) -> set[str]:
    """Return the set of live data file paths reachable from the current snapshot."""
    return {t.file.file_path for t in table.scan().plan_files()}


def read_ids(table: Table) -> list[int]:
    """Return the sorted ``id`` column of the table's live rows."""
    return sorted(int(r["id"]) for r in table.scan().to_arrow().to_pylist())


def snapshot_count(table: Table) -> int:
    """Return the number of snapshots in the table's metadata log."""
    return len(table.metadata.snapshots or [])


def _row_count(table: Table) -> int:
    table.refresh()
    return table.scan().to_arrow().num_rows


class Appender(threading.Thread):
    """Background thread that calls ``table.append`` on a fixed cadence.

    Exposes :attr:`first_commit_event` so tests can wait for the first
    append to land deterministically rather than relying on ``time.sleep``.
    """

    def __init__(
        self,
        table: Table,
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


def inject_once_around_rewrite(
    monkeypatch: pytest.MonkeyPatch, action: Callable[[Table], None], *, before: bool
) -> dict[str, int]:
    """Run ``action(table)`` exactly once around the first ``_rewrite_group`` call.

    ``action`` receives the live table and performs the foreign operation that
    must land between the rewrite plan and the commit. When ``before`` is true
    it fires ahead of the group read; otherwise it fires after the outputs are
    produced, which is required when the operation removes the group's own input
    files. Returns a counter dict so callers can assert the injection fired.
    """
    from daft.io.iceberg import _compact

    state = {"fired": 0}

    def instrument(real_rewrite_group: Callable[_P, _R]) -> Callable[_P, _R]:
        def instrumented(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            live = cast("Table", kwargs["table"])
            first = state["fired"] == 0
            if first and before:
                state["fired"] = 1
                action(live)
            out = real_rewrite_group(*args, **kwargs)
            if first and not before:
                state["fired"] = 1
                action(live)
            return out

        return instrumented

    monkeypatch.setattr(_compact, "_rewrite_group", instrument(_compact._rewrite_group))
    return state


def make_seeded_table(
    catalog: Catalog,
    name: str,
    *,
    n_files: int = 6,
    rows_per_file: int = 100,
) -> Table:
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


def _write_position_delete_file(
    table: Table, deletes: dict[str, list[int]], partition: Record | None = None
) -> DataFile:
    """Write a position-delete file naming rows by data file path and position.

    ``deletes`` maps a data file path to the row positions to remove from it.
    ``partition`` is the partition record the delete belongs to; it defaults to
    the empty record of an unpartitioned table. Returns the delete file's
    metadata, ready to be committed. Nothing is committed here.
    """
    import io
    import uuid

    import pyarrow.parquet as pq
    from pyiceberg.manifest import DataFile, DataFileContent, FileFormat
    from pyiceberg.typedef import Record

    # The spec types a delete's row position as a 64-bit integer. Build the
    # schema here rather than reusing the library's, which declares a 32-bit one
    # and produces files other readers reject.
    delete_arrow_schema = pa.schema(
        [
            pa.field("file_path", pa.large_string(), metadata={b"PARQUET:field_id": b"2147483546"}),
            pa.field("pos", pa.int64(), metadata={b"PARQUET:field_id": b"2147483545"}),
        ]
    )
    rows = [{"file_path": path, "pos": pos} for path, positions in deletes.items() for pos in sorted(positions)]

    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows, schema=delete_arrow_schema), buffer)
    payload = buffer.getvalue()
    delete_path = f"{table.location()}/data/positional-delete-{uuid.uuid4().hex}.parquet"
    with table.io.new_output(delete_path).create(overwrite=True) as handle:
        handle.write(payload)

    return DataFile.from_args(
        _table_format_version=table.metadata.format_version,
        content=DataFileContent.POSITION_DELETES,
        file_path=delete_path,
        file_format=FileFormat.PARQUET,
        partition=partition if partition is not None else Record(),
        file_size_in_bytes=len(payload),
        sort_order_id=None,
        spec_id=table.spec().spec_id,
        equality_ids=None,
        key_metadata=None,
        record_count=len(rows),
        # Bounds over the path column say which data files the delete can cover,
        # which is what lets a reader or a rewrite skip it without opening it.
        lower_bounds={_DELETE_FILE_PATH_FIELD_ID: min(deletes).encode("utf-8")},
        upper_bounds={_DELETE_FILE_PATH_FIELD_ID: max(deletes).encode("utf-8")},
    )


def _commit_files(table: Table, data_files: list[DataFile], delete_files: list[DataFile]) -> None:
    """Commit one snapshot adding ``data_files`` and ``delete_files`` together.

    Every existing manifest is kept as it is. Adding data and deletes in one
    snapshot gives them one sequence number, which is the shape a row-level
    update produces and the one the dangling-delete rule has to reason about.
    """
    import uuid

    from pyiceberg.manifest import (
        ManifestContent,
        ManifestEntry,
        ManifestEntryStatus,
        ManifestWriterV2,
    )
    from pyiceberg.table.update.snapshot import Operation, _SnapshotProducer

    table.refresh()
    spec = table.spec()
    specs = table.specs()
    # Both manifests are typed by the table schema: the partition summaries a
    # manifest carries are derived from it, whatever the entries hold.
    table_schema = table.schema()
    deletes_by_spec: dict[int, list[DataFile]] = {}
    for delete_file in delete_files:
        deletes_by_spec.setdefault(int(getattr(delete_file, "spec_id", spec.spec_id)), []).append(delete_file)

    class _DeleteManifestWriter(ManifestWriterV2):
        """A manifest holding delete files rather than data files."""

        def content(self) -> ManifestContent:
            return ManifestContent.DELETES

        @property
        def _meta(self) -> dict[str, str]:
            return {**super()._meta, "content": "deletes"}

    class _AddFiles(_SnapshotProducer):  # type: ignore[misc, valid-type]
        """Adds one manifest per file kind and keeps every existing manifest."""

        def _existing_manifests(self) -> list[ManifestFile]:
            snapshot = self._transaction.table_metadata.current_snapshot()
            return list(snapshot.manifests(self._io)) if snapshot is not None else []

        def _deleted_entries(self) -> list[ManifestEntry]:
            return []

        def _manifests(self) -> list[ManifestFile]:
            written: list[ManifestFile] = []
            manifests: list[tuple[type[ManifestWriterV2], PartitionSpec, list[DataFile]]] = [
                (ManifestWriterV2, spec, data_files)
            ]
            manifests.extend(
                (_DeleteManifestWriter, specs[spec_id], files) for spec_id, files in deletes_by_spec.items()
            )
            for writer_cls, manifest_spec, files in manifests:
                if not files:
                    continue
                with writer_cls(
                    manifest_spec, table_schema, self.new_manifest_output(), self._snapshot_id, self._compression
                ) as writer:
                    for file in files:
                        writer.add_entry(
                            ManifestEntry.from_args(
                                status=ManifestEntryStatus.ADDED,
                                snapshot_id=self._snapshot_id,
                                sequence_number=None,
                                file_sequence_number=None,
                                data_file=file,
                            )
                        )
                written.append(writer.to_manifest_file())
            return [*written, *self._existing_manifests()]

    with table.transaction() as transaction:
        _AddFiles(
            operation=Operation.OVERWRITE,
            transaction=transaction,
            io=table.io,
            commit_uuid=uuid.uuid4(),
            snapshot_properties={},
            branch="main",
        ).commit()


def commit_positional_deletes(table: Table, deletes: dict[str, list[int]], partition: Record | None = None) -> str:
    """Commit a merge-on-read delete naming rows by file path and position.

    The delete does not remove the data files it applies to; it adds a delete
    file that names them, which is exactly the conflict a rewrite has to notice.
    Writing one is the only way to reproduce that conflict here, since the
    catalog library reads delete files but does not write them.

    ``deletes`` maps a data file path to the row positions to remove from it;
    ``partition`` is their partition record on a partitioned table. The table
    must be format version 2 or later. Returns the path of the delete file that
    was written.
    """
    table.refresh()
    delete_file = _write_position_delete_file(table, deletes, partition)
    _commit_files(table, [], [delete_file])
    return delete_file.file_path


def commit_row_delta(table: Table, rows: pa.Table, deleted_positions: list[int]) -> tuple[str, str]:
    """Commit new rows and a position delete over some of them in one snapshot.

    A row-level update produces exactly this shape: the data file and the delete
    that applies to it share a sequence number. It is the case the
    dangling-delete rule must keep, because a position delete still covers a
    data file written at its own sequence number. Returns the paths of the data
    file and the delete file.
    """
    from pyiceberg.io.pyarrow import _dataframe_to_data_files

    table.refresh()
    data_file = next(iter(_dataframe_to_data_files(table.metadata, rows, table.io)))
    delete_file = _write_position_delete_file(table, {data_file.file_path: deleted_positions}, data_file.partition)
    _commit_files(table, [data_file], [delete_file])
    return data_file.file_path, delete_file.file_path


def _write_equality_delete_file(
    table: Table,
    rows: pa.Table,
    equality_columns: list[str],
    *,
    partition: Record | None = None,
    spec_id: int | None = None,
) -> DataFile:
    """Write an equality delete file holding ``rows`` and matching on ``equality_columns``.

    Column names are resolved against the table schema to their field ids, which
    is how a reader identifies the delete columns. ``spec_id`` and ``partition``
    place the delete; the unpartitioned spec makes it a global delete. Returns
    the delete file's metadata; nothing is committed here.
    """
    import io
    import uuid

    import pyarrow.parquet as pq
    from pyiceberg.manifest import DataFile, DataFileContent, FileFormat
    from pyiceberg.typedef import Record

    table.refresh()
    schema = table.schema()
    arrow_schema = pa.schema(
        [
            pa.field(
                name,
                rows.schema.field(name).type,
                metadata={b"PARQUET:field_id": str(schema.find_field(name).field_id).encode()},
            )
            for name in rows.column_names
        ]
    )
    buffer = io.BytesIO()
    pq.write_table(rows.cast(arrow_schema), buffer)
    payload = buffer.getvalue()
    delete_path = f"{table.location()}/data/equality-delete-{uuid.uuid4().hex}.parquet"
    with table.io.new_output(delete_path).create(overwrite=True) as handle:
        handle.write(payload)

    delete_file = DataFile.from_args(
        _table_format_version=table.metadata.format_version,
        content=DataFileContent.EQUALITY_DELETES,
        file_path=delete_path,
        file_format=FileFormat.PARQUET,
        partition=partition if partition is not None else Record(),
        file_size_in_bytes=len(payload),
        sort_order_id=None,
        equality_ids=[schema.find_field(name).field_id for name in equality_columns],
        key_metadata=None,
        record_count=rows.num_rows,
    )
    # The spec id is manifest-level metadata, set on the file when a manifest is read.
    delete_file.spec_id = table.spec().spec_id if spec_id is None else spec_id
    return delete_file


def commit_equality_delete_rows(
    table: Table,
    rows: pa.Table,
    equality_columns: list[str],
    *,
    partition: Record | None = None,
    spec_id: int | None = None,
) -> str:
    """Commit an equality delete removing every row equal to a row of ``rows`` on ``equality_columns``.

    Neither the catalog library nor the reference engine's batch writers emit
    one, so it is written here. Returns the path of the delete file.
    """
    delete_file = _write_equality_delete_file(table, rows, equality_columns, partition=partition, spec_id=spec_id)
    _commit_files(table, [], [delete_file])
    return delete_file.file_path


def commit_equality_deletes(table: Table, column: str, values: list[int] | list[str]) -> str:
    """Commit an equality delete removing every row whose ``column`` is in ``values``."""
    return commit_equality_delete_rows(table, pa.table({column: pa.array(values)}), [column])


def commit_equality_upsert(table: Table, rows: pa.Table, equality_columns: list[str]) -> tuple[str, str]:
    """Commit new rows and an equality delete of their keys in one snapshot.

    A streaming upsert produces exactly this shape: the delete and the data
    file share a sequence number, so the delete removes older rows with those
    keys and leaves the rows committed beside it. Returns the paths of the data
    file and the delete file.
    """
    from pyiceberg.io.pyarrow import _dataframe_to_data_files

    table.refresh()
    data_file = next(iter(_dataframe_to_data_files(table.metadata, rows, table.io)))
    delete_file = _write_equality_delete_file(
        table, rows.select(equality_columns), equality_columns, partition=data_file.partition
    )
    _commit_files(table, [data_file], [delete_file])
    return data_file.file_path, delete_file.file_path
