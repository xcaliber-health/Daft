"""Server mode: host the local streaming engine behind a network endpoint.

Start a server with ``python -m daft.serve`` or programmatically via
:func:`start_server`. Clients connect with ``daft.connect`` and run the full
DataFrame API against the server; plans are optimized and executed
server-side so scan planning, pruning, and all data access happen next to
the data.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import pathlib
from typing import TYPE_CHECKING

from daft.daft import DaftServeServer
from daft.session import Session

if TYPE_CHECKING:
    from daft.catalog import Catalog

__all__ = [
    "DEFAULT_MAX_CONCURRENT_QUERIES",
    "DEFAULT_MAX_PSET_BYTES",
    "DEFAULT_PORT",
    "DEFAULT_QUERY_TIMEOUT_SECS",
    "DEFAULT_QUEUE_TIMEOUT_SECS",
    "CatalogSpec",
    "ServeSettings",
    "TenantSpec",
    "load_settings",
    "start_server",
]

logger = logging.getLogger(__name__)

DEFAULT_PORT = 9494
DEFAULT_MAX_CONCURRENT_QUERIES = 4
DEFAULT_QUEUE_TIMEOUT_SECS = 60
DEFAULT_MAX_PSET_BYTES = 256 * 1024 * 1024
DEFAULT_QUERY_TIMEOUT_SECS = 0


@dataclasses.dataclass(frozen=True)
class CatalogSpec:
    """Declarative description of one catalog to attach to the server session.

    Parameters
    ----------
    name:
        Alias the catalog is attached under; textual queries reference
        tables through this name.
    properties:
        Keyword arguments passed to the catalog loader, e.g. ``uri``,
        ``warehouse``, and credential properties for a REST catalog.
    """

    name: str
    properties: dict[str, str]

    def load(self) -> Catalog:
        """Load the catalog and wrap it for session attachment.

        Returns:
        -------
        Catalog
            The loaded catalog, ready to attach to a session.
        """
        from pyiceberg.catalog import load_catalog

        from daft.catalog import Catalog

        return Catalog.from_iceberg(load_catalog(self.name, **self.properties))


@dataclasses.dataclass(frozen=True)
class TenantSpec:
    """One named tenant sharing the server, with optional limit overrides.

    Each tenant authenticates with its own bearer token and may override
    the server-wide limits; ``None`` inherits the server default. Tenants
    with their own ``max_concurrent_queries`` get a dedicated execution-slot
    pool, so saturating it never delays other tenants.

    Parameters
    ----------
    name:
        Tenant name; appears in admission errors and logs, never secret.
    token:
        Bearer token identifying this tenant's requests.
    max_concurrent_queries:
        Dedicated execution slots; ``None`` shares the default pool.
    queue_timeout_secs:
        Seconds a query may wait for one of this tenant's slots.
    query_timeout_secs:
        Wall-clock execution limit for this tenant's queries; ``0``
        disables the limit.
    max_pset_bytes:
        Cap on in-memory data shipped with one of this tenant's queries.
    memory_cap_bytes:
        Ceiling on buffered execution memory per query for this tenant;
        queries over it spill to disk rather than grow.
    """

    name: str
    token: str
    max_concurrent_queries: int | None = None
    queue_timeout_secs: int | None = None
    query_timeout_secs: int | None = None
    max_pset_bytes: int | None = None
    memory_cap_bytes: int | None = None


@dataclasses.dataclass(frozen=True)
class ServeSettings:
    """Complete configuration of one serving process.

    Parameters
    ----------
    host:
        Interface to bind, e.g. ``127.0.0.1`` or ``0.0.0.0``.
    port:
        Port to bind; ``0`` picks an ephemeral port.
    token:
        Bearer token every request must present. Required for
        non-loopback binds unless ``allow_insecure_remote`` is set.
    allow_insecure_remote:
        Explicit opt-out permitting a tokenless non-loopback bind.
    max_concurrent_queries:
        Number of queries executing at once; excess queries queue.
    queue_timeout_secs:
        Seconds a queued query waits for a slot before being rejected.
    max_pset_bytes:
        Cap on in-memory data shipped with a single query.
    disable_plan_payload:
        Reject serialized-plan payloads, accepting only textual queries.
    query_timeout_secs:
        Wall-clock seconds one query may execute before being cancelled;
        ``0`` disables the limit.
    query_memory_cap_bytes:
        Ceiling on buffered execution memory per query; ``0`` disables it.
        Queries over the ceiling spill to disk rather than grow.
    tenants:
        Named tenants sharing this server. When non-empty, every request
        must present one of the tenant tokens and ``token`` is ignored.
    catalogs:
        Catalogs to attach to the server session at startup.
    """

    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    token: str | None = None
    allow_insecure_remote: bool = False
    max_concurrent_queries: int = DEFAULT_MAX_CONCURRENT_QUERIES
    queue_timeout_secs: int = DEFAULT_QUEUE_TIMEOUT_SECS
    max_pset_bytes: int = DEFAULT_MAX_PSET_BYTES
    disable_plan_payload: bool = False
    query_timeout_secs: int = DEFAULT_QUERY_TIMEOUT_SECS
    query_memory_cap_bytes: int = 0
    tenants: tuple[TenantSpec, ...] = ()
    catalogs: tuple[CatalogSpec, ...] = ()


def _parse_tenants(raw: dict[str, object]) -> tuple[TenantSpec, ...]:
    """Build tenant specifications from a parsed configuration mapping.

    Tenant tokens never come from the file: each tenant's token is read
    from the environment variable ``DAFT_SERVE_TOKEN_<NAME>`` (name
    upper-cased, dashes mapped to underscores).
    """
    raw_tenants = raw.get("tenants", [])
    if not isinstance(raw_tenants, list):
        raise TypeError("`tenants` must be a list of tenant mappings")
    tenants: list[TenantSpec] = []
    for entry in raw_tenants:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError(f"tenant entry must be a mapping with a `name`: {entry!r}")
        name = str(entry["name"])
        token_env = f"DAFT_SERVE_TOKEN_{name.upper().replace('-', '_')}"
        token = os.environ.get(token_env)
        if not token:
            raise ValueError(f"tenant `{name}`: set its token in the environment variable {token_env}")

        def _opt_int(key: str, entry: dict[str, object] = entry, name: str = name) -> int | None:
            value = entry.get(key)
            if value is None:
                return None
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"tenant `{name}`: `{key}` must be an integer, got {value!r}")
            return value

        tenants.append(
            TenantSpec(
                name=name,
                token=token,
                max_concurrent_queries=_opt_int("max_concurrent_queries"),
                queue_timeout_secs=_opt_int("queue_timeout_secs"),
                query_timeout_secs=_opt_int("query_timeout_secs"),
                max_pset_bytes=_opt_int("max_pset_bytes"),
                memory_cap_bytes=_opt_int("memory_cap_bytes"),
            )
        )
    return tuple(tenants)


def _parse_settings(raw: dict[str, object], token: str | None) -> ServeSettings:
    """Build settings from a parsed configuration mapping."""
    raw_catalogs = raw.get("catalogs", [])
    if not isinstance(raw_catalogs, list):
        raise TypeError("`catalogs` must be a list of {name, properties} mappings")
    catalogs: list[CatalogSpec] = []
    for entry in raw_catalogs:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError(f"catalog entry must be a mapping with a `name`: {entry!r}")
        properties = entry.get("properties", {})
        if not isinstance(properties, dict):
            raise TypeError(f"catalog `properties` must be a mapping: {properties!r}")
        catalogs.append(
            CatalogSpec(
                name=str(entry["name"]),
                properties={str(k): str(v) for k, v in properties.items()},
            )
        )

    def _int(key: str, default: int) -> int:
        value = raw.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"`{key}` must be an integer, got {value!r}")
        return value

    host = raw.get("host", "127.0.0.1")
    if not isinstance(host, str):
        raise TypeError(f"`host` must be a string, got {host!r}")

    return ServeSettings(
        host=host,
        port=_int("port", DEFAULT_PORT),
        token=token,
        allow_insecure_remote=bool(raw.get("allow_insecure_remote", False)),
        max_concurrent_queries=_int("max_concurrent_queries", DEFAULT_MAX_CONCURRENT_QUERIES),
        queue_timeout_secs=_int("queue_timeout_secs", DEFAULT_QUEUE_TIMEOUT_SECS),
        max_pset_bytes=_int("max_pset_bytes", DEFAULT_MAX_PSET_BYTES),
        disable_plan_payload=bool(raw.get("disable_plan_payload", False)),
        query_timeout_secs=_int("query_timeout_secs", DEFAULT_QUERY_TIMEOUT_SECS),
        query_memory_cap_bytes=_int("query_memory_cap_bytes", 0),
        tenants=_parse_tenants(raw),
        catalogs=tuple(catalogs),
    )


def load_settings(config_path: str | os.PathLike[str], token_env: str = "DAFT_SERVE_TOKEN") -> ServeSettings:
    """Load server settings from a configuration file.

    The file may be JSON, or YAML when the ``yaml`` package is installed.
    The token is never read from the file; it comes from the environment
    variable named by ``token_env``.

    Parameters
    ----------
    config_path:
        Path to the configuration file.
    token_env:
        Environment variable holding the bearer token.

    Returns:
    -------
    ServeSettings
        Parsed settings.

    Raises:
    ------
    ValueError
        If the file cannot be parsed or contains invalid values.
    TypeError
        If a configuration field has the wrong type.
    """
    text = pathlib.Path(config_path).read_text()
    raw: object
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as e:
            raise ValueError(
                f"{config_path} is not valid JSON and the `yaml` package is not installed; "
                "install `pyyaml` or provide JSON configuration"
            ) from e
        raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise TypeError(f"configuration root must be a mapping, got {type(raw).__name__}")
    return _parse_settings(raw, os.environ.get(token_env))


def start_server(settings: ServeSettings) -> DaftServeServer:
    """Start a serving process and return its handle.

    Attaches the configured catalogs to a fresh session, binds the
    endpoint, and returns once the listener is ready. The returned handle
    exposes ``address``, ``wait``, and ``shutdown``.

    Parameters
    ----------
    settings:
        Complete server configuration.

    Returns:
    -------
    DaftServeServer
        Handle to the running server.
    """
    session = Session()
    catalog_names: list[str] = []
    for spec in settings.catalogs:
        session.attach_catalog(spec.load(), alias=spec.name)
        catalog_names.append(spec.name)
        logger.info("attached catalog %s", spec.name)

    return DaftServeServer(
        settings.host,
        settings.port,
        token=settings.token,
        allow_insecure_remote=settings.allow_insecure_remote,
        max_concurrent_queries=settings.max_concurrent_queries,
        queue_timeout_secs=settings.queue_timeout_secs,
        max_pset_bytes=settings.max_pset_bytes,
        disable_plan_payload=settings.disable_plan_payload,
        query_timeout_secs=settings.query_timeout_secs,
        query_memory_cap_bytes=settings.query_memory_cap_bytes,
        tenants=[
            (
                tenant.name,
                tenant.token,
                tenant.max_concurrent_queries,
                tenant.queue_timeout_secs,
                tenant.query_timeout_secs,
                tenant.max_pset_bytes,
                tenant.memory_cap_bytes,
            )
            for tenant in settings.tenants
        ],
        session=session,
        catalogs=catalog_names,
    )
