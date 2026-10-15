# Do not modify or delete these exceptions before checking where they are used in rust
# src/common/error/src/python.rs
from __future__ import annotations


class DaftCoreException(ValueError):
    """DaftCore Base Exception."""

    pass


class DaftTypeError(DaftCoreException):
    """Type Error that occurred in Daft Core."""

    pass


class DaftFieldNotFoundError(DaftCoreException):
    """A column or field named in a query does not exist."""


class DaftAmbiguousReferenceError(DaftCoreException):
    """A column name refers to more than one column."""


class DaftSchemaMismatchError(DaftCoreException):
    """Two schemas that must agree do not."""


class DaftValueError(DaftCoreException):
    """An argument or value was refused."""


class DaftComputeError(DaftCoreException):
    """A computation could not be carried out on its inputs."""


class DaftCardinalityError(DaftComputeError):
    """A value that must come from exactly one row came from several."""


class DaftNotImplementedError(DaftCoreException):
    """The requested operation is not supported."""


class DaftInvalidArgumentError(DaftCoreException):
    """An argument to an operation was invalid."""


class DaftServeError(DaftCoreException):
    """A query sent to a remote server failed there, with a failure that has no engine class.

    Attributes:
        error_class: Where it was raised. ``"Host"``: by the server's request host while it
            planned, before anything ran. ``"Python"``: by Python code while the query executed,
            so rows may already have been read. Otherwise the server's own failure:
            ``"AtCapacity"``, ``"Timeout"``, ``"Cancelled"``, ``"Unauthenticated"``,
            ``"VersionMismatch"``, ``"PlanPayloadDisabled"``, ``"Draining"``, and the refusals of
            a malformed or oversized request.
        retryable: Whether the same query may succeed if sent again later: true for capacity,
            timeout and draining.
        remote_type: For a ``Host`` or ``Python`` failure, the name of the raised type.
        remote_module: The module of that type.
        remote_types: The names of that type's hierarchy, nearest first.
        fields: Named text fields the raised exception carried in ``served_fields``.
        query_id: The id of the query that failed, when it had one.
        detail: How the query failed, in the words of the server's request host, when the
            query was a request its host planned and the host described the failure; otherwise
            ``None``. Every served failure carries it, an engine class's included.
    """

    def __init__(
        self,
        message: str,
        *,
        error_class: str,
        retryable: bool = False,
        remote_type: str | None = None,
        remote_module: str | None = None,
        remote_types: tuple[str, ...] = (),
        fields: dict[str, str] | None = None,
        query_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.retryable = retryable
        self.remote_type = remote_type
        self.remote_module = remote_module
        self.remote_types = remote_types
        self.fields = dict(fields or {})
        self.query_id = query_id
        self.detail = detail


class DaftTransientError(DaftCoreException):
    """Daft Transient Error.

    This is typically raised when there is a network issue such as timeout or throttling. This can usually be retried.
    """

    pass


class ConnectTimeoutError(DaftTransientError):
    """Daft Connection Timeout Error.

    Daft client was not able to make a connection to the server in the connect timeout time.
    """

    pass


class ReadTimeoutError(DaftTransientError):
    """Daft Read Timeout Error.

    Daft client was not able to read bytes from server under the read timeout time.
    """

    pass


class ByteStreamError(DaftTransientError):
    """Daft Byte Stream Error.

    Daft client had an error while reading bytes in a stream from the server.
    """

    pass


class SocketError(DaftTransientError):
    """Daft Socket Error.

    Daft client had a socket error while reading bytes in a stream from the server.
    """

    pass


class ThrottleError(DaftTransientError):
    """Daft Throttle Error.

    Daft client had a throttle error while making request to server.
    """

    pass


class MiscTransientError(DaftTransientError):
    """Daft Misc Transient Error.

    Daft client had a Misc Transient Error while making request to server.
    """

    pass
