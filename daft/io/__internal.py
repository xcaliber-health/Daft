from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, TypeVar

from daft.event_loop import get_or_init_event_loop

if TYPE_CHECKING:
    from collections.abc import Iterator

    from daft.daft import PyRecordBatch
    from daft.io.source import DataSourceTask

_T = TypeVar("_T")

# How far an asynchronous iterator may run ahead of the thread reading it: two
# items, or as many as fit in this many bytes, whichever holds more. Large record
# batches stay two ahead; small ones need not wait for their reader one by one.
_AHEAD_ITEMS = 2
_AHEAD_BYTES = 16 * 1024 * 1024


class _Sentinel:
    """Sentinel value for the queue to indicate the end of the iterator."""


_SENTINEL: _Sentinel = _Sentinel()


class _Ahead:
    """The items, and their bytes, that a producer holds ahead of its reader.

    Changed from two threads, so every change is made under its lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items = 0
        self._bytes = 0
        self.awaiting_room = False

    def try_add(self, size: int) -> bool:
        """Counts an item of ``size`` bytes in, if there is room, and says whether there was.

        When there is none, the producer is marked as awaiting room.
        """
        with self._lock:
            if self._items < _AHEAD_ITEMS or self._bytes < _AHEAD_BYTES:
                self._items += 1
                self._bytes += size
                self.awaiting_room = False
                return True
            self.awaiting_room = True
            return False

    def take(self, size: int) -> bool:
        """Counts an item of ``size`` bytes out, and says whether to wake the producer.

        It is woken once half the room has freed, so that a producer kept waiting
        by a slower reader is woken once for several items, not for each.
        """
        with self._lock:
            self._items -= 1
            self._bytes -= size
            return self.awaiting_room and (self._items < _AHEAD_ITEMS or self._bytes <= _AHEAD_BYTES // 2)


def _drain_async_iter(async_iter: AsyncIterator[_T], size_of: Callable[[_T], int] | None = None) -> Iterator[_T]:
    """Synchronously drain an async iterator, a bounded way ahead of its reader.

    The iterator runs on the shared ``BackgroundEventLoop`` (a persistent daemon
    thread running ``loop.run_forever()``), so this works whether or not an event
    loop is already running in the calling thread (e.g., pytest-asyncio,
    Jupyter). It runs at most ``_AHEAD_ITEMS`` items, or ``_AHEAD_BYTES`` bytes
    as measured by ``size_of``, ahead of the reader, whichever holds more; an
    item with no ``size_of`` counts as all those bytes, so at most
    ``_AHEAD_ITEMS`` are held. When it
    has no room, it awaits some rather than blocking the loop, which every query
    shares. Closing the returned iterator early stops the async one and closes it.
    """
    measure = size_of if size_of is not None else (lambda _: _AHEAD_BYTES)
    loop = get_or_init_event_loop().loop
    results: queue.Queue[_T | BaseException | _Sentinel] = queue.Queue()
    ahead = _Ahead()
    room = asyncio.Event()

    async def _produce() -> None:
        try:
            async for item in async_iter:
                size = measure(item)
                while True:
                    # Cleared before asking, so a reader freeing room after the
                    # answer below wakes this producer rather than an old wake.
                    room.clear()
                    if ahead.try_add(size):
                        break
                    await room.wait()
                results.put_nowait(item)
        except asyncio.CancelledError:
            # Only the reader cancels, once it has stopped reading.
            raise
        except BaseException as ex:
            # Anything else reaches the reader, which would otherwise wait forever.
            results.put_nowait(ex)
            return
        finally:
            close = getattr(async_iter, "aclose", None)
            if close is not None:
                await close()
        results.put_nowait(_SENTINEL)

    producer = asyncio.run_coroutine_threadsafe(_produce(), loop)
    try:
        while True:
            item = results.get()
            if isinstance(item, _Sentinel):
                return
            if isinstance(item, BaseException):
                raise item
            if ahead.take(measure(item)):
                loop.call_soon_threadsafe(room.set)
            yield item
    finally:
        producer.cancel()


def _get_record_batches(task: DataSourceTask) -> Iterator[PyRecordBatch]:
    """Called by the Rust python_factory_func_scan_task at execution time.

    The task instance has been pickled then sent to this stateless method. The
    engine calls it on a thread of its own, so a synchronous ``read`` is iterated
    there directly, and an asynchronous one is drained from the shared event loop.
    """
    batches = task.read()
    if isinstance(batches, AsyncIterator):
        batches = _drain_async_iter(batches, size_of=lambda batch: batch.size_bytes())
    yield from (rb._recordbatch for rb in batches)
