"""Connect the current process to a remote query server.

After :func:`connect`, every DataFrame in this process executes on the
server: plans are shipped over the wire, optimized and run next to the data,
and results stream back. Textual queries can be routed through the returned
connection as well, resolving against the catalogs attached to the server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from daft.runners import set_runner_remote

if TYPE_CHECKING:
    from daft.daft import DaftServeServerInfo
    from daft.dataframe import DataFrame
    from daft.runners.remote_runner import RemoteRunner

__all__ = ["RemoteConnection", "connect"]


class RemoteConnection:
    """A live connection to a query server.

    DataFrame execution is transparent after connecting; this handle adds
    server-scoped operations: textual queries against server catalogs,
    server-side plan explanation, and query management.
    """

    def __init__(self, runner: RemoteRunner) -> None:
        self._runner = runner

    @property
    def address(self) -> str:
        """Normalized address of the connected server."""
        return self._runner.address

    def server_info(self) -> DaftServeServerInfo:
        """Fetch the server's version, capabilities, and limits.

        Returns:
        -------
        DaftServeServerInfo
            Server description.
        """
        return self._runner.client.server_info()

    def sql(self, query: str, results_buffer_size: int | None = None) -> DataFrame:
        """Run a textual query on the server and materialize the result.

        The query is planned and optimized entirely server-side against the
        catalogs attached to the server session, so it works even when the
        client has no catalog or storage access.

        Parameters
        ----------
        query:
            The query text.
        results_buffer_size:
            Bound on buffered, not-yet-consumed result partitions.

        Returns:
        -------
        DataFrame
            The materialized result.
        """
        import uuid

        from daft.dataframe import DataFrame
        from daft.recordbatch import MicroPartition

        result = self._runner.client.run_sql(query, f"sql-{uuid.uuid4().hex[:12]}", results_buffer_size)
        parts = [MicroPartition._from_pymicropartition(part) for part in result]
        return DataFrame._from_micropartitions(*parts)

    def explain(self, df: DataFrame) -> str:
        """Render the server-side optimized plan for a DataFrame.

        Parameters
        ----------
        df:
            The DataFrame whose plan to explain.

        Returns:
        -------
        str
            Human-readable rendering of the optimized plan.
        """
        return self._runner.client.explain_plan(df._builder._builder)

    def cancel_query(self, query_id: str) -> bool:
        """Cancel a running query by id.

        Parameters
        ----------
        query_id:
            Identifier of the query to cancel.

        Returns:
        -------
        bool
            Whether a running query with that id was found.
        """
        return self._runner.client.cancel_query(query_id)


def connect(address: str, token: str | None = None) -> RemoteConnection:
    """Connect this process to a query server.

    Sets the process-wide runner to remote execution; every DataFrame built
    afterwards executes on the server. The runner can only be set once per
    process.

    Parameters
    ----------
    address:
        Server address: ``daft://host:port``, ``grpc://host:port``, or
        ``host:port`` (port defaults to 9494).
    token:
        Bearer token, required when the server enforces authentication.

    Returns:
    -------
    RemoteConnection
        Handle for server-scoped operations.

    Examples:
    --------
    >>> conn = daft.connect("daft://analytics.internal:9494", token="...")  # doctest: +SKIP
    >>> daft.read_iceberg(tbl).filter(df["x"] > 1).collect()  # runs on the server  # doctest: +SKIP
    """
    from typing import cast

    from daft.runners.remote_runner import RemoteRunner

    runner = cast("RemoteRunner", set_runner_remote(address, token))
    assert isinstance(runner, RemoteRunner)
    return RemoteConnection(runner)
