"""Reads a table's metadata files for planning, one request each, keeping them between reads.

A table writes each manifest list and manifest once, under a path no later
file reuses, so a file read once may be served again by path for as long as
the process keeps it. Files are kept per IO configuration, so a read never
sees a file its own credentials did not fetch, within a bound on their total
bytes, least recently used leaving first.
"""

from __future__ import annotations

import io
import logging
import pickle
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING

from pyiceberg.io import FileIO, InputFile, InputStream, OutputFile

from daft.daft import io_get

if TYPE_CHECKING:
    from daft.io import IOConfig

logger = logging.getLogger(__name__)

# A file larger than this is never kept, as Iceberg's own manifest cache does.
MAX_ENTRY_BYTES = 8 * 1024 * 1024

_CacheKey = tuple[bytes, str]


class _MetadataCache:
    """Metadata file contents by IO configuration and path, within a bound on their bytes."""

    def __init__(self) -> None:
        self._entries: OrderedDict[_CacheKey, bytes] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key: _CacheKey) -> bytes | None:
        with self._lock:
            data = self._entries.get(key)
            if data is not None:
                self._entries.move_to_end(key)
            return data

    def put(self, key: _CacheKey, data: bytes, max_bytes: int) -> None:
        if len(data) > min(max_bytes, MAX_ENTRY_BYTES):
            return
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._bytes -= len(old)
            while self._entries and self._bytes + len(data) > max_bytes:
                _, evicted = self._entries.popitem(last=False)
                self._bytes -= len(evicted)
            self._entries[key] = data
            self._bytes += len(data)


_CACHE = _MetadataCache()


class _MetadataInput(InputFile):
    """A metadata file, read whole on first open."""

    def __init__(self, location: str, file_io: MetadataFileIO) -> None:
        super().__init__(location)
        self._file_io = file_io
        self._data: bytes | None = None

    def _bytes(self) -> bytes:
        if self._data is None:
            self._data = self._file_io.read(self.location)
        return self._data

    def __len__(self) -> int:
        return len(self._bytes())

    def exists(self) -> bool:
        return self._file_io.inner.new_input(self.location).exists()

    def open(self, seekable: bool = True) -> InputStream:
        return io.BytesIO(self._bytes())


class MetadataFileIO(FileIO):
    """A table's file IO whose reads fetch each file whole in one request, through the engine.

    Reads of the table's metadata files go through the engine's IO client with
    ``io_config``, the configuration the read's data files use, and are kept
    by path within ``max_cache_bytes`` (0 keeps none). Writes and deletes pass
    to the table's own file IO.

    Args:
        inner: The table's own file IO.
        io_config: The IO configuration the read uses.
        max_cache_bytes: The most bytes of metadata files the process keeps.
    """

    def __init__(self, inner: FileIO, io_config: IOConfig, max_cache_bytes: int) -> None:
        super().__init__(inner.properties)
        self.inner = inner
        self._io_config = io_config
        self._config_key = pickle.dumps(io_config)
        self._max_cache_bytes = max_cache_bytes

    def read(self, location: str) -> bytes:
        """Return the whole file at ``location``, from the cache when it holds it."""
        key = (self._config_key, location)
        if self._max_cache_bytes > 0:
            cached = _CACHE.get(key)
            if cached is not None:
                return cached
        try:
            data = io_get(location, io_config=self._io_config)
        except Exception as exc:
            # A path the engine cannot read with this configuration is read as
            # the table's own file IO reads it, and not kept.
            logger.debug("reading %s through the table's file IO: %s", location, exc)
            with self.inner.new_input(location).open(seekable=False) as stream:
                return stream.read()
        if self._max_cache_bytes > 0:
            _CACHE.put(key, data, self._max_cache_bytes)
        return data

    def new_input(self, location: str) -> InputFile:
        return _MetadataInput(location, self)

    def new_output(self, location: str) -> OutputFile:
        return self.inner.new_output(location)

    def delete(self, location: str | InputFile | OutputFile) -> None:
        self.inner.delete(location)


def metadata_file_io(inner: FileIO, io_config: IOConfig) -> MetadataFileIO:
    """The metadata file IO for a read with ``io_config``, bounded by the engine's setting."""
    from daft.context import get_context

    max_cache_bytes = get_context().daft_execution_config.iceberg_metadata_cache_bytes
    return MetadataFileIO(inner, io_config, max_cache_bytes)
