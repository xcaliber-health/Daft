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


class DaftNotImplementedError(DaftCoreException):
    """The requested operation is not supported."""


class DaftInvalidArgumentError(DaftCoreException):
    """An argument to an operation was invalid."""


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
