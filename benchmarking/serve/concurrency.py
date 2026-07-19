r"""Concurrency harness: closed-loop clients against the two server lanes.

Each client thread runs the query mix back-to-back for the duration; the
harness reports throughput and latency percentiles per lane and client
count. Point the same harness at the query server and at a quack server for
a like-for-like comparison.

Examples:
--------
::

    python -m benchmarking.serve.concurrency --warehouse /tmp/tpch-sf001 \\
        --clients 1 4 16 --duration-secs 30
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import statistics
import threading
import time

from benchmarking.serve import datagen, lanes

logger = logging.getLogger(__name__)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def run_closed_loop(
    make_lane: object,
    queries: list[str],
    clients: int,
    duration_secs: float,
) -> dict[str, float]:
    """Run `clients` threads in a closed loop; returns throughput/latency stats.

    Each thread owns its own lane instance (one connection per client).
    """
    latencies: list[float] = []
    errors = [0]
    lock = threading.Lock()
    stop_at = time.monotonic() + duration_secs

    def worker() -> None:
        lane = make_lane()  # type: ignore[operator]
        try:
            i = 0
            while time.monotonic() < stop_at:
                query = queries[i % len(queries)]
                i += 1
                start = time.perf_counter()
                try:
                    lane.run(query)
                except Exception:  # noqa: BLE001 - errors are counted, not fatal
                    with lock:
                        errors[0] += 1
                    continue
                elapsed = time.perf_counter() - start
                with lock:
                    latencies.append(elapsed)
        finally:
            lane.close()

    threads = [threading.Thread(target=worker) for _ in range(clients)]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.monotonic() - started

    return {
        "clients": float(clients),
        "completed": float(len(latencies)),
        "errors": float(errors[0]),
        "qps": len(latencies) / wall if wall > 0 else 0.0,
        "p50_s": statistics.median(latencies) if latencies else float("nan"),
        "p95_s": _percentile(latencies, 0.95),
        "p99_s": _percentile(latencies, 0.99),
    }


def main() -> int:
    """Command-line entrypoint; returns process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse", required=True)
    parser.add_argument("--scale-factor", type=float, default=0.01)
    parser.add_argument("--clients", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--duration-secs", type=float, default=30.0)
    parser.add_argument("--queries", nargs="+", default=["q1", "q6"])
    parser.add_argument("--lanes", nargs="+", default=["daft-serve", "duckdb-quack"])
    parser.add_argument("--output", default=None, help="JSON output path")
    args = parser.parse_args()

    warehouse = pathlib.Path(args.warehouse)
    if warehouse.exists():
        locations = datagen.existing_locations(warehouse)
    else:
        locations = datagen.generate_warehouse(warehouse, args.scale_factor)

    # Shared servers, one per lane kind; per-client lanes connect to them.
    factories: dict[str, object] = {}
    cleanups: list[object] = []
    if "daft-serve" in args.lanes:
        import daft.serve

        server = daft.serve.start_server(
            daft.serve.ServeSettings(host="127.0.0.1", port=0, max_concurrent_queries=max(args.clients))
        )
        address = server.address()
        factories["daft-serve"] = lambda: lanes.DaftServeLane(warehouse, address)
        cleanups.append(lambda: server.shutdown(drain_timeout_secs=10))
    if "duckdb-quack" in args.lanes:
        try:
            quack_proc = lanes.start_quack_server(locations, port=9596)
            factories["duckdb-quack"] = lambda: lanes.DuckDBQuackLane(port=9596, existing_server=True)
            cleanups.append(lambda: (quack_proc.kill(), quack_proc.wait(timeout=30)))
        except Exception as e:  # noqa: BLE001
            logger.warning("duckdb-quack unavailable: %s", e)

    results: dict[str, list[dict[str, float]]] = {}
    for lane_name, factory in factories.items():
        lane_stats: list[dict[str, float]] = []
        for clients in args.clients:
            stats = run_closed_loop(factory, args.queries, clients, args.duration_secs)
            logger.info(
                "%-14s clients=%2d qps=%7.2f p50=%6.3fs p95=%6.3fs p99=%6.3fs errors=%d",
                lane_name,
                clients,
                stats["qps"],
                stats["p50_s"],
                stats["p95_s"],
                stats["p99_s"],
                int(stats["errors"]),
            )
            lane_stats.append(stats)
        results[lane_name] = lane_stats

    for cleanup in cleanups:
        cleanup()  # type: ignore[operator]

    if args.output:
        pathlib.Path(args.output).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
