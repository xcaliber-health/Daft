"""Runner that executes queries on a remote serving process.

The full DataFrame API passes through unchanged: the unoptimized logical
plan (including scan operators and user-defined functions) is serialized and
shipped to the server, which optimizes and executes it next to the data and
streams result partitions back. In-memory partition sets referenced by the
plan are shipped alongside it, subject to the server's size cap.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import daft
from daft.context import get_context
from daft.daft import DaftServeClient, serve_referenced_pset_keys
from daft.execution.metadata import ExecutionMetadata
from daft.naming import generate_query_name
from daft.recordbatch import MicroPartition
from daft.runners.native_runner import NativeRunnerIO
from daft.runners.partitioning import (
    LocalMaterializedResult,
    LocalPartitionSet,
    PartitionCacheEntry,
    PartitionSetCache,
)
from daft.runners.query_id import emit_query_id
from daft.runners.runner import LOCAL_PARTITION_SET_CACHE, Runner

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

    from daft.daft import PyMicroPartition
    from daft.logical.builder import LogicalPlanBuilder

logger = logging.getLogger(__name__)

_URI_SCHEMES = ("daft://", "grpc://", "http://")


def normalize_address(address: str) -> str:
    """Normalize a server address into the transport URI form.

    Accepts ``daft://host:port``, ``grpc://host:port``, ``http://host:port``,
    or a bare ``host:port``; the port defaults to 9494 when omitted.

    Parameters
    ----------
    address:
        Server address in any accepted form.

    Returns:
    -------
    str
        Address in ``grpc://host:port`` form.

    Raises:
    ------
    ValueError
        If the address is empty or uses an unsupported scheme.
    """
    if not address:
        raise ValueError("server address must not be empty")
    if "://" in address:
        scheme = address.split("://", 1)[0] + "://"
        if scheme not in _URI_SCHEMES:
            raise ValueError(f"unsupported scheme `{scheme}` in `{address}`; use daft://, grpc://, or http://")
        rest = address.split("://", 1)[1]
    else:
        rest = address
    if not rest:
        raise ValueError(f"missing host in server address `{address}`")
    if not _has_port(rest):
        rest = f"{rest}:9494"
    return f"grpc://{rest}"


def _has_port(host_and_port: str) -> bool:
    """Whether the address part after the scheme includes an explicit port."""
    if host_and_port.startswith("["):
        # Bracketed IPv6 literal: a port follows the closing bracket.
        return "]:" in host_and_port
    return host_and_port.count(":") == 1


class RemoteRunner(Runner[MicroPartition]):
    """Executes plans on a remote serving process, streaming results back."""

    name = "remote"

    def __init__(self, address: str, token: str | None = None) -> None:
        super().__init__()
        self._address = normalize_address(address)
        self._client = DaftServeClient(self._address, token)
        info = self._client.server_info()
        client_version = daft.get_version()
        if info.version != client_version:
            logger.warning(
                "server version %s differs from client version %s; "
                "plan submission will be rejected until versions match",
                info.version,
                client_version,
            )
        if not info.plan_payload_enabled:
            logger.warning(
                "server at %s rejects serialized plans; only daft.sql queries routed through the connection will work",
                self._address,
            )

    @property
    def address(self) -> str:
        """Normalized address of the connected server."""
        return self._address

    @property
    def client(self) -> DaftServeClient:
        """Underlying protocol client for this connection."""
        return self._client

    def initialize_partition_set_cache(self) -> PartitionSetCache:
        return LOCAL_PARTITION_SET_CACHE

    def runner_io(self) -> NativeRunnerIO:
        return NativeRunnerIO()

    def run_iter(
        self,
        builder: LogicalPlanBuilder,
        results_buffer_size: int | None = None,
    ) -> Generator[LocalMaterializedResult, None, ExecutionMetadata]:
        query_id = generate_query_name()
        emit_query_id(query_id)
        ctx = get_context()

        # Materialize only the cached partition sets the plan references;
        # unrelated cached data would otherwise be converted (and potentially
        # shipped) on every query.
        referenced = set(serve_referenced_pset_keys(builder._builder))
        psets: dict[str, list[PyMicroPartition]] = {
            key: [entry.micropartition()._micropartition for entry in pset.values()]
            for key, pset in self._part_set_cache.get_all_partition_sets().items()
            if key in referenced
        }
        result = self._client.run_plan(
            builder._builder,
            psets,
            query_id,
            exec_config=ctx.daft_execution_config,
            results_buffer_size=results_buffer_size,
        )
        if result.optimized_locally:
            logger.info(
                "plan contains a source that cannot travel; scan planning ran "
                "on the client and the server executed pre-materialized scan tasks"
            )
        for partition in result:
            yield LocalMaterializedResult(MicroPartition._from_pymicropartition(partition))

        stats = result.stats()
        if stats is None:
            # The result stream ended without its terminal statistics
            # trailer: the server stopped mid-query (crash or forced
            # shutdown). Results may be incomplete, so fail loudly instead
            # of returning a silently truncated result.
            raise RuntimeError(
                f"query `{query_id}`: result stream from {self._address} ended "
                "without an execution-statistics trailer; the server may have "
                "died mid-query and the results may be incomplete"
            )
        physical_plan_json, py_stats = stats
        return ExecutionMetadata._from_runner_output(py_stats, query_id, physical_plan_json or "")

    def run_iter_tables(
        self, builder: LogicalPlanBuilder, results_buffer_size: int | None = None
    ) -> Iterator[MicroPartition]:
        for result in self.run_iter(builder, results_buffer_size=results_buffer_size):
            yield result.partition()

    def run(self, builder: LogicalPlanBuilder) -> tuple[PartitionCacheEntry, ExecutionMetadata]:
        results_gen = self.run_iter(builder)
        result_pset = LocalPartitionSet()

        try:
            i = 0
            while True:
                result = next(results_gen)
                result_pset.set_partition(i, result)
                i += 1
        except StopIteration as e:
            metadata = e.value

        pset_entry = self.put_partition_set_into_cache(result_pset)
        return pset_entry, metadata
