r"""Benchmark driver: timings, correctness gate, and baseline regression.

Runs the TPC-H query subset across the selected lanes, cross-checks result
equality between engines, records timing percentiles, and compares against
committed baselines when present.

Examples:
--------
Smoke run with all local lanes::

    python -m benchmarking.serve.bench --warehouse /tmp/tpch-sf001 \\
        --scale-factor 0.01 --iterations 3

Record a new baseline::

    python -m benchmarking.serve.bench --warehouse /tmp/tpch-sf1 \\
        --scale-factor 1 --iterations 5 --record-baseline
"""

from __future__ import annotations

import argparse
import datetime
import decimal
import json
import logging
import math
import pathlib
import statistics
import time

from benchmarking.serve import datagen, lanes

logger = logging.getLogger(__name__)

BASELINE_DIR = pathlib.Path(__file__).parent / "baselines"
REGRESSION_TOLERANCE = 0.15
FLOAT_TOLERANCE = 1e-3


def _make_lanes(
    selected: list[str],
    warehouse: pathlib.Path,
    locations: dict[str, str],
    serve_address: str | None,
    serve_token: str | None,
) -> tuple[list[lanes.Lane], list[object]]:
    """Builds the selected lanes; returns them plus cleanup callables.

    Server processes started here must outlive the lanes that connect to
    them, so their shutdowns are returned separately and invoked last.
    """
    out: list[lanes.Lane] = []
    cleanups: list[object] = []
    for name in selected:
        try:
            if name == "daft-inprocess":
                out.append(lanes.DaftInProcessLane(warehouse))
            elif name == "daft-serve":
                if serve_address is None:
                    import daft.serve

                    server = daft.serve.start_server(daft.serve.ServeSettings(host="127.0.0.1", port=0))
                    serve_address = server.address()
                    cleanups.append(lambda srv=server: srv.shutdown(drain_timeout_secs=10))
                out.append(lanes.DaftServeLane(warehouse, serve_address, serve_token))
            elif name == "duckdb-inprocess":
                out.append(lanes.DuckDBInProcessLane(locations))
            elif name == "duckdb-quack":
                out.append(lanes.DuckDBQuackLane(locations))
            else:
                raise ValueError(f"unknown lane `{name}`")
        except Exception as e:  # noqa: BLE001 - a lane that cannot start is reported, not fatal
            logger.warning("lane %s unavailable: %s", name, e)
    return out, cleanups


def _values_match(a: object, b: object) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float, decimal.Decimal)) and isinstance(b, (int, float, decimal.Decimal)):
        fa, fb = float(a), float(b)
        if math.isnan(fa) or math.isnan(fb):
            return math.isnan(fa) and math.isnan(fb)
        return math.isclose(fa, fb, rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE)
    if isinstance(a, (datetime.date, datetime.datetime)) or isinstance(b, (datetime.date, datetime.datetime)):
        return str(a) == str(b)
    return a == b


def check_correctness(reference: lanes.Lane, candidate: lanes.Lane, query: str) -> bool:
    """Whether both lanes produce equivalent results for a query."""
    ref_rows = reference.rows(query)
    got_rows = candidate.rows(query)
    if len(ref_rows) != len(got_rows):
        logger.error(
            "%s: row count mismatch %s=%d vs %s=%d",
            query,
            reference.name,
            len(ref_rows),
            candidate.name,
            len(got_rows),
        )
        return False
    for ref, got in zip(ref_rows, got_rows):
        if len(ref) != len(got) or not all(_values_match(a, b) for a, b in zip(ref, got)):
            logger.error("%s: row mismatch\n  %s: %s\n  %s: %s", query, reference.name, ref, candidate.name, got)
            return False
    return True


def run_benchmark(
    active: list[lanes.Lane], queries: list[str], iterations: int
) -> dict[str, dict[str, dict[str, float]]]:
    """Time every lane over every query; returns lane -> query -> stats."""
    results: dict[str, dict[str, dict[str, float]]] = {}
    for lane in active:
        lane_results: dict[str, dict[str, float]] = {}
        for query in queries:
            times: list[float] = []
            rows = 0
            for _ in range(iterations):
                rows, elapsed = lane.run(query)
                times.append(elapsed)
            lane_results[query] = {
                "rows": float(rows),
                "p50_s": statistics.median(times),
                "min_s": min(times),
                "max_s": max(times),
            }
            logger.info(
                "%-18s %-4s p50=%7.3fs min=%7.3fs rows=%d",
                lane.name,
                query,
                lane_results[query]["p50_s"],
                lane_results[query]["min_s"],
                rows,
            )
        results[lane.name] = lane_results
    return results


def compare_to_baseline(results: dict[str, dict[str, dict[str, float]]], scale_factor: float) -> list[str]:
    """Returns regression descriptions exceeding the tolerance, if a baseline exists."""
    path = BASELINE_DIR / f"sf{scale_factor}.json"
    if not path.exists():
        logger.info("no baseline at %s; skipping regression check", path)
        return []
    baseline = json.loads(path.read_text())
    regressions: list[str] = []
    for lane_name, lane_results in results.items():
        for query, stats in lane_results.items():
            base = baseline.get(lane_name, {}).get(query)
            if base is None:
                continue
            if stats["p50_s"] > base["p50_s"] * (1 + REGRESSION_TOLERANCE):
                regressions.append(
                    f"{lane_name}/{query}: p50 {stats['p50_s']:.3f}s vs baseline "
                    f"{base['p50_s']:.3f}s (>{REGRESSION_TOLERANCE:.0%})"
                )
    return regressions


def write_report(
    output: pathlib.Path,
    results: dict[str, dict[str, dict[str, float]]],
    correctness: dict[str, bool],
    scale_factor: float,
    iterations: int,
) -> None:
    """Render a Markdown comparison report."""
    queries = sorted({q for lane in results.values() for q in lane})
    lines = [
        "# Query server benchmark",
        "",
        f"- scale factor: {scale_factor}",
        f"- iterations per query: {iterations} (median reported)",
        f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "## Timings (p50 seconds)",
        "",
        "| query | " + " | ".join(results) + " |",
        "|---" * (len(results) + 1) + "|",
    ]
    for query in queries:
        row = [query]
        for lane_results in results.values():
            stats = lane_results.get(query)
            row.append(f"{stats['p50_s']:.3f}" if stats else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Correctness gate", ""]
    for key, ok in sorted(correctness.items()):
        lines.append(f"- {key}: {'PASS' if ok else 'FAIL'}")
    output.write_text("\n".join(lines) + "\n")
    logger.info("report written to %s", output)


def main() -> int:
    """Command-line entrypoint; returns process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse", required=True)
    parser.add_argument("--scale-factor", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--lanes",
        nargs="+",
        default=["daft-inprocess", "daft-serve", "duckdb-inprocess", "duckdb-quack"],
    )
    parser.add_argument("--queries", nargs="+", default=list(lanes.QUERY_NAMES))
    parser.add_argument("--serve-address", default=None, help="existing server; otherwise one is started")
    parser.add_argument("--serve-token", default=None)
    parser.add_argument("--record-baseline", action="store_true")
    parser.add_argument("--skip-datagen", action="store_true")
    parser.add_argument("--report", default=None, help="report path (default: report.md beside this file)")
    args = parser.parse_args()

    warehouse = pathlib.Path(args.warehouse)
    if args.skip_datagen and warehouse.exists():
        locations = datagen.existing_locations(warehouse)
    else:
        locations = datagen.generate_warehouse(warehouse, args.scale_factor)

    active, cleanups = _make_lanes(args.lanes, warehouse, locations, args.serve_address, args.serve_token)
    if not active:
        logger.error("no lanes available")
        return 2

    results = run_benchmark(active, args.queries, args.iterations)

    # Correctness: every lane must agree with the first available lane.
    correctness: dict[str, bool] = {}
    reference = active[0]
    for lane in active[1:]:
        for query in args.queries:
            correctness[f"{reference.name}-vs-{lane.name}/{query}"] = check_correctness(reference, lane, query)
    for lane in active:
        lane.close()
    for cleanup in cleanups:
        cleanup()  # type: ignore[operator]

    report = pathlib.Path(args.report) if args.report else pathlib.Path(__file__).parent / "report.md"
    write_report(report, results, correctness, args.scale_factor, args.iterations)

    if args.record_baseline:
        BASELINE_DIR.mkdir(exist_ok=True)
        path = BASELINE_DIR / f"sf{args.scale_factor}.json"
        path.write_text(json.dumps(results, indent=2))
        logger.info("baseline recorded at %s", path)
        return 0

    failed = [k for k, ok in correctness.items() if not ok]
    regressions = compare_to_baseline(results, args.scale_factor)
    for regression in regressions:
        logger.error("REGRESSION: %s", regression)
    if failed:
        logger.error("correctness failures: %s", failed)
    return 1 if failed or regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
