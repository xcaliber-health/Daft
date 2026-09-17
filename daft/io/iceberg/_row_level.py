"""Shared primitives for row-level writes.

Row-level writes replace or delete individual rows rather than whole files. They
all start from the same two things: a stable name for every data file the
operation may touch, and a scan that tells each row which file and position it
came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from daft.io.iceberg._common import ROW_FILE_COLUMN, ROW_POSITION_COLUMN
from daft.io.iceberg._deletes import read_with_deletes

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pyiceberg.manifest import DataFile
    from pyiceberg.table import Table as PyIcebergTable
    from pyiceberg.typedef import Record

    from daft.dataframe import DataFrame
    from daft.io import IOConfig
    from daft.io.iceberg._deletes import ScanPlan


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
        """Build a table over every data file the plan selected."""
        entries = [
            FileEntry(
                path=path,
                spec_id=int(plan.tasks[path].file.spec_id),
                partition=plan.tasks[path].file.partition,
                sequence_number=plan.sequence_by_path[path],
                data_file=plan.tasks[path].file,
            )
            for path in sorted(plan.tasks)
        ]
        return cls(entries)

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
