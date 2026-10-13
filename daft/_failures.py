"""What kind of failure a raised exception is, said the same way wherever the query ran.

A query run in-process and the same query served by a remote server report a
failure with the same kind: an engine class by its name, and any other Python
exception as a ``Host`` failure (raised by a server's request host while it
planned) or a ``Python`` failure (raised while the query executed), with the
names of its type. A user-defined function's failure names the function's
original exception.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from daft.daft import PyQueryResult, QueryEndState
    from daft.errors import UDFException

#: Module whose classes are the engine's own.
ENGINE_MODULE = "daft.exceptions"

#: Kind of a Python failure raised by a request host while it planned.
HOST = "Host"
#: Kind of a Python failure raised while a query executed.
PYTHON = "Python"

#: The most named text fields a failure carries to a client.
MAX_FIELDS = 32
#: The most bytes a failure's named text fields take, names and values together.
MAX_FIELDS_BYTES = 8 * 1024


@dataclass(frozen=True)
class FailureKind:
    """The kind of one failure.

    Attributes:
    ----------
    error_class
        The engine class's name, ``"Host"`` or ``"Python"``.
    remote_type, remote_module, remote_types
        For a Python failure: its type's name, that type's module, and the
        names of the type's hierarchy, nearest first. ``None`` and empty for
        an engine failure.
    udf
        Whether a user-defined function raised it.
    fields
        The named text fields the exception carried in ``served_fields``,
        when they keep within the bounds; otherwise empty.
    """

    error_class: str
    remote_type: str | None = None
    remote_module: str | None = None
    remote_types: tuple[str, ...] = ()
    udf: bool = False
    fields: dict[str, str] = field(default_factory=dict)


def failure_kind(exc: BaseException, *, raised_by_host: bool = False) -> FailureKind:
    """Return the kind of ``exc``.

    Parameters
    ----------
    exc
        The exception a query failed with.
    raised_by_host
        Whether a request host raised it while planning, before anything ran.
    """
    from daft.errors import UDFException

    exc = _unwrapped(exc)
    kind = type(exc)
    # The engine raises a missing file as the builtin class.
    if kind.__module__ == ENGINE_MODULE or kind is FileNotFoundError:
        return FailureKind(error_class=kind.__name__)
    udf = isinstance(exc, UDFException)
    original = exc.original_exception if isinstance(exc, UDFException) else None
    named = original if original is not None else exc
    named_type = _raised_type(exc) if isinstance(exc, UDFException) and original is None else type(named)
    return FailureKind(
        error_class=HOST if raised_by_host else PYTHON,
        remote_type=named_type.__name__,
        remote_module=named_type.__module__,
        remote_types=tuple(cls.__name__ for cls in named_type.__mro__ if cls is not object),
        udf=udf,
        fields=served_fields(named),
    )


def _raised_type(exc: UDFException) -> type[BaseException]:
    """Return the type a function's failure raised in another process, from the traceback it sent back."""
    raised = getattr(exc.tb_info, "exc_type", None) if exc.tb_info is not None else None
    if isinstance(raised, type) and issubclass(raised, BaseException):
        return raised
    return type(exc)


def _unwrapped(exc: BaseException) -> BaseException:
    """Return the exception a distributed task raised, rather than the wrapper that carried it back."""
    cause = getattr(exc, "cause", None)
    if type(exc).__module__.startswith("ray.") and isinstance(cause, BaseException):
        return cause
    return exc


def query_result(
    end_state: QueryEndState,
    message: str,
    *,
    rows_returned: int | None = None,
    bytes_returned: int | None = None,
    failed_with: BaseException | None = None,
) -> PyQueryResult:
    """Return the result a query run in-process ended with, naming the kind of failure it failed with.

    The kind is reported as a served query reports it, and without the
    exception's named text fields, which never reach an event.
    """
    from daft.daft import PyQueryResult

    if failed_with is None:
        return PyQueryResult(end_state, message, rows_returned, bytes_returned)
    kind = failure_kind(failed_with)
    return PyQueryResult(
        end_state,
        message,
        rows_returned,
        bytes_returned,
        error_class=kind.error_class,
        remote_type=kind.remote_type,
        remote_module=kind.remote_module,
        remote_types=list(kind.remote_types),
    )


def served_fields(exc: BaseException) -> dict[str, str]:
    """Return the named text fields ``exc`` carries in ``served_fields``, or none when they break the bounds.

    Every name and value must be text, with at most :data:`MAX_FIELDS` fields
    and :data:`MAX_FIELDS_BYTES` bytes in all. Fields that break a bound are
    dropped whole, so the failure still arrives with its type and message.
    """
    raw = getattr(exc, "served_fields", None)
    if not isinstance(raw, Mapping) or len(raw) > MAX_FIELDS:
        return {}
    fields: dict[str, str] = {}
    size = 0
    for name, value in raw.items():
        if not isinstance(name, str) or not isinstance(value, str):
            return {}
        size += len(name.encode()) + len(value.encode())
        if size > MAX_FIELDS_BYTES:
            return {}
        fields[name] = value
    return fields
