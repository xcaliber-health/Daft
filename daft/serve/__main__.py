"""Command-line entrypoint for the server: ``python -m daft.serve``."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import types

from daft.serve import (
    DEFAULT_MAX_CONCURRENT_QUERIES,
    DEFAULT_MAX_PSET_BYTES,
    DEFAULT_PORT,
    DEFAULT_QUEUE_TIMEOUT_SECS,
    ServeSettings,
    load_settings,
    start_server,
)

logger = logging.getLogger("daft.serve")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m daft.serve",
        description="Serve queries from a single engine process over the network.",
    )
    parser.add_argument("--host", default=None, help="interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help=f"port to bind (default: {DEFAULT_PORT})")
    parser.add_argument(
        "--config",
        default=None,
        help="path to a JSON or YAML configuration file (catalogs, limits)",
    )
    parser.add_argument(
        "--token-env",
        default="DAFT_SERVE_TOKEN",
        help="environment variable holding the bearer token (default: DAFT_SERVE_TOKEN)",
    )
    parser.add_argument(
        "--allow-insecure-remote",
        action="store_true",
        help="permit a tokenless non-loopback bind (NOT recommended)",
    )
    parser.add_argument(
        "--disable-plan-payload",
        action="store_true",
        help="reject serialized-plan payloads; accept only textual queries",
    )
    parser.add_argument(
        "--max-concurrent-queries",
        type=int,
        default=None,
        help=f"queries executing at once (default: {DEFAULT_MAX_CONCURRENT_QUERIES})",
    )
    parser.add_argument(
        "--queue-timeout-secs",
        type=int,
        default=None,
        help=f"seconds a queued query waits for a slot (default: {DEFAULT_QUEUE_TIMEOUT_SECS})",
    )
    parser.add_argument(
        "--max-pset-bytes",
        type=int,
        default=None,
        help=f"cap on in-memory data shipped per query (default: {DEFAULT_MAX_PSET_BYTES})",
    )
    parser.add_argument(
        "--drain-timeout-secs",
        type=int,
        default=120,
        help="seconds to drain in-flight queries on shutdown (default: 120)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the server until interrupted.

    Parameters
    ----------
    argv:
        Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
    -------
    int
        Process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)

    if args.config is not None:
        settings = load_settings(args.config, token_env=args.token_env)
    else:
        import os

        settings = ServeSettings(token=os.environ.get(args.token_env))

    overrides: dict[str, object] = {}
    if args.host is not None:
        overrides["host"] = args.host
    if args.port is not None:
        overrides["port"] = args.port
    if args.allow_insecure_remote:
        overrides["allow_insecure_remote"] = True
    if args.disable_plan_payload:
        overrides["disable_plan_payload"] = True
    if args.max_concurrent_queries is not None:
        overrides["max_concurrent_queries"] = args.max_concurrent_queries
    if args.queue_timeout_secs is not None:
        overrides["queue_timeout_secs"] = args.queue_timeout_secs
    if args.max_pset_bytes is not None:
        overrides["max_pset_bytes"] = args.max_pset_bytes
    if overrides:
        import dataclasses

        settings = dataclasses.replace(settings, **overrides)  # type: ignore[arg-type]

    server = start_server(settings)
    logger.info("serving on %s", server.address())

    def _handle_signal(signum: int, _frame: types.FrameType | None) -> None:
        logger.info("received %s; draining", signal.Signals(signum).name)
        server.shutdown(drain_timeout_secs=args.drain_timeout_secs)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # Block in a helper thread so the main thread stays responsive to
    # signals; a blocking call in the main thread would defer handlers
    # until it returned.
    waiter = threading.Thread(target=server.wait, name="daft-serve-wait", daemon=True)
    waiter.start()
    while waiter.is_alive():
        waiter.join(timeout=0.5)
    logger.info("server stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
