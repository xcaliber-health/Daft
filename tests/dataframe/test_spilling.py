"""Grouped aggregation under memory pressure spills instead of failing.

The engine memory budget is process-global and fixed at startup, so the
pressured runs execute in subprocesses with a small budget configured via
the environment; results must match an unpressured run exactly.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import pathlib

_GROUPBY_SCRIPT = """
import json
import sys

import numpy as np
import pyarrow as pa

import daft
from daft import col

rng = np.random.default_rng(11)
n = 2_000_000
df = daft.from_arrow(
    pa.table(
        {
            "k": pa.array(rng.integers(0, 500_000, n)),
            "v": pa.array(rng.integers(0, 1_000, n)),
        }
    )
).collect()

out = df.groupby("k").agg(col("v").sum().alias("s"), col("v").count().alias("c")).to_pydict()
checksum = {
    "groups": len(out["k"]),
    "sum_s": int(sum(out["s"])),
    "sum_c": int(sum(out["c"])),
}
print("RESULT " + json.dumps(checksum))
"""


def _run(
    env_overrides: dict[str, str],
    spill_dir: str | None = None,
    enable_spilling: bool | None = None,
) -> dict[str, int]:
    import json
    import os

    config_args: list[str] = []
    if spill_dir is not None:
        config_args.append(f"spill_dirs=[{spill_dir!r}]")
    if enable_spilling is not None:
        config_args.append(f"enable_spilling={enable_spilling}")
    script = _GROUPBY_SCRIPT
    if config_args:
        script = script.replace(
            "import daft\n",
            "import daft\ndaft.set_execution_config(" + ", ".join(config_args) + ")\n",
        )
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "DAFT_RUNNER": "native", **env_overrides},
        timeout=600,
        check=False,
    )
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    raise AssertionError(f"no result line; stdout={out.stdout!r} stderr={out.stderr[-2000:]!r}")


def test_groupby_under_tiny_memory_budget_matches_unpressured(tmp_path: pathlib.Path) -> None:
    unpressured = _run({})
    # A ~30MB budget against a working set several times larger forces the
    # aggregation to shed buffered partitions to disk (verified by spill
    # events in the engine log at this budget).
    pressured = _run({"DAFT_MEMORY_LIMIT": str(30 * 1024 * 1024)}, spill_dir=str(tmp_path))

    assert pressured == unpressured
    # The scratch directory tree is removed once the query finishes.
    leftovers = list((tmp_path / "daft-spill").glob("*")) if (tmp_path / "daft-spill").exists() else []
    assert leftovers == []


def test_pressure_with_spilling_disabled_still_completes(tmp_path: pathlib.Path) -> None:
    """The escape hatch keeps the pre-spilling behavior available."""
    unpressured = _run({})
    disabled = _run(
        {"DAFT_MEMORY_LIMIT": str(30 * 1024 * 1024)},
        spill_dir=str(tmp_path),
        enable_spilling=False,
    )
    assert disabled == unpressured
    assert not (tmp_path / "daft-spill").exists()


@pytest.mark.parametrize("mem_limit_mb", [30, 60])
def test_varied_budgets_produce_identical_results(tmp_path: pathlib.Path, mem_limit_mb: int) -> None:
    unpressured = _run({})
    pressured = _run(
        {"DAFT_MEMORY_LIMIT": str(mem_limit_mb * 1024 * 1024)},
        spill_dir=str(tmp_path),
    )
    assert pressured == unpressured


_SORT_SCRIPT = """
import json

import numpy as np
import pyarrow as pa

import daft
from daft import col

rng = np.random.default_rng(23)
n = 2_000_000
df = daft.from_arrow(
    pa.table(
        {
            "k": pa.array(rng.integers(0, 10_000_000, n)),
            "v": pa.array(rng.integers(0, 1_000, n)),
        }
    )
).collect()

out = df.sort("k").to_pydict()
ks = out["k"]
checksum = {
    "rows": len(ks),
    "is_sorted": all(a <= b for a, b in zip(ks, ks[1:])),
    "head": ks[:3],
    "tail": ks[-3:],
    "sum_v": int(sum(out["v"])),
}
print("RESULT " + json.dumps(checksum))
"""


def _run_sort(env_overrides: dict[str, str], spill_dir: str | None = None) -> dict[str, object]:
    import json
    import os

    script = _SORT_SCRIPT
    if spill_dir is not None:
        script = script.replace(
            "import daft\n",
            f"import daft\ndaft.set_execution_config(spill_dirs=[{spill_dir!r}])\n",
        )
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "DAFT_RUNNER": "native", **env_overrides},
        timeout=600,
        check=False,
    )
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    raise AssertionError(f"no result line; stdout={out.stdout!r} stderr={out.stderr[-2000:]!r}")


def test_sort_under_tiny_memory_budget_matches_unpressured(tmp_path: pathlib.Path) -> None:
    unpressured = _run_sort({})
    pressured = _run_sort({"DAFT_MEMORY_LIMIT": str(30 * 1024 * 1024)}, spill_dir=str(tmp_path))

    assert pressured == unpressured
    assert pressured["is_sorted"] is True
    leftovers = list((tmp_path / "daft-spill").glob("*")) if (tmp_path / "daft-spill").exists() else []
    assert leftovers == []


_JOIN_AGG_SCRIPT = """
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s:%(message)s")

import numpy as np
import pyarrow as pa

import daft
from daft import col

rng = np.random.default_rng(3)
n = 2_000_000
dim = daft.from_arrow(
    pa.table({"k": pa.array(np.arange(500_000)), "p": pa.array(rng.integers(0, 10, 500_000))})
).collect()
fact = daft.from_arrow(
    pa.table({"k": pa.array(rng.integers(0, 500_000, n)), "v": pa.array(rng.integers(0, 1_000, n))})
).collect()

out = fact.join(dim, on="k").groupby("k").agg(col("v").sum().alias("s")).to_pydict()
checksum = {"groups": len(out["k"]), "sum_s": int(sum(out["s"]))}
print("RESULT " + json.dumps(checksum))
"""


def _run_join_agg(env_overrides: dict[str, str], spill_dir: str | None = None) -> tuple[dict[str, int], str]:
    import json
    import os

    script = _JOIN_AGG_SCRIPT
    if spill_dir is not None:
        script = script.replace(
            "import daft\n",
            f"import daft\ndaft.set_execution_config(spill_dirs=[{spill_dir!r}])\n",
        )
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "DAFT_RUNNER": "native", **env_overrides},
        timeout=600,
        check=False,
    )
    combined = out.stdout + out.stderr
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :]), combined
    raise AssertionError(f"no result line; stdout={out.stdout!r} stderr={out.stderr[-2000:]!r}")


def test_join_feeding_aggregation_under_pressure(tmp_path: pathlib.Path) -> None:
    """The aggregation absorbs the pressure the join's budget share creates.

    Results must stay exact while the shed-able operator spills.
    """
    unpressured, _ = _run_join_agg({})
    pressured, log = _run_join_agg({"DAFT_MEMORY_LIMIT": str(8 * 1024 * 1024)}, spill_dir=str(tmp_path))
    assert pressured == unpressured
    assert "spilled" in log


def test_join_build_over_budget_partitions_and_matches(tmp_path: pathlib.Path) -> None:
    """A build side that alone exceeds the budget switches to a partitioned join.

    Both sides re-partition to disk and replay partition by partition, so
    the query completes within the budget with exact results instead of
    overflowing memory.
    """
    unpressured, _ = _run_join_agg({})
    pressured, log = _run_join_agg({"DAFT_MEMORY_LIMIT": str(2 * 1024 * 1024)}, spill_dir=str(tmp_path))
    assert pressured == unpressured
    assert "switching to a partitioned join" in log
    assert "spilling its input across" in log
    assert "join build side exceeds" not in log


_GRACE_OUTER_SCRIPT = """
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s:%(message)s")

import numpy as np
import pyarrow as pa

import daft
from daft import col

rng = np.random.default_rng(11)
n = 2_000_000
dim = daft.from_arrow(
    pa.table({"k": pa.array(np.arange(500_000)), "p": pa.array(rng.integers(0, 10, 500_000))})
).collect()
fact = daft.from_arrow(
    pa.table({"k": pa.array(rng.integers(0, 600_000, n)), "v": pa.array(rng.integers(0, 1_000, n))})
).collect()

out = (
    fact.join(dim, on="k", how="outer")
    .agg(
        col("k").count().alias("rows"),
        col("v").sum().alias("sv"),
        col("p").sum().alias("sp"),
        col("p").count().alias("matched"),
    )
    .to_pydict()
)
checksum = {key: int(value[0]) for key, value in out.items()}
print("RESULT " + json.dumps(checksum))
"""


def test_outer_join_partitions_with_exact_null_semantics(tmp_path: pathlib.Path) -> None:
    """A partitioned outer join reproduces matched and unmatched rows exactly.

    Unmatched rows from both sides must survive the per-partition replay,
    proving the matched-row tracking works partition by partition.
    """
    import json
    import os

    def run(env_overrides: dict[str, str]) -> tuple[dict[str, int], str]:
        script = _GRACE_OUTER_SCRIPT.replace(
            "import daft\n",
            f"import daft\ndaft.set_execution_config(spill_dirs=[{str(tmp_path)!r}])\n",
        )
        out = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={**os.environ, "DAFT_RUNNER": "native", **env_overrides},
            timeout=600,
            check=False,
        )
        for line in out.stdout.splitlines():
            if line.startswith("RESULT "):
                return json.loads(line[len("RESULT ") :]), out.stdout + out.stderr
        raise AssertionError(f"no result line; stdout={out.stdout!r} stderr={out.stderr[-2000:]!r}")

    unpressured, _ = run({})
    pressured, log = run({"DAFT_MEMORY_LIMIT": str(2 * 1024 * 1024)})
    assert pressured == unpressured
    assert pressured["rows"] > pressured["matched"]
    assert "switching to a partitioned join" in log


_CONCURRENT_SCRIPT = """
import json
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format="%(name)s:%(message)s")

import numpy as np
import pyarrow as pa

import daft
from daft import col
from daft.runners.remote_runner import RemoteRunner
from daft.serve import ServeSettings, start_server

server = start_server(ServeSettings(host="127.0.0.1", port=0))

rng = np.random.default_rng(31)
n = 1_500_000
tables = [
    pa.table(
        {
            "k": pa.array(rng.integers(0, 400_000, n)),
            "v": pa.array(rng.integers(0, 1_000, n)),
        }
    )
    for _ in range(2)
]


def run_query(idx: int) -> dict[str, int]:
    runner = RemoteRunner(server.address())
    df = daft.from_arrow(tables[idx]).collect()
    parts = list(runner.run_iter_tables(df.groupby("k").agg(col("v").sum().alias("s"))._builder))
    groups = 0
    total = 0
    for part in parts:
        d = part.to_pydict()
        groups += len(d["k"])
        total += sum(d["s"])
    return {"groups": groups, "sum_s": int(total)}


with ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(run_query, range(2)))
server.shutdown(drain_timeout_secs=10)
print("RESULT " + json.dumps(results))
"""


def _run_concurrent(env_overrides: dict[str, str], spill_dir: str | None = None) -> tuple[list[dict[str, int]], str]:
    import json
    import os

    script = _CONCURRENT_SCRIPT
    if spill_dir is not None:
        script = script.replace(
            "import daft\n",
            f"import daft\ndaft.set_execution_config(spill_dirs=[{spill_dir!r}])\n",
        )
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "DAFT_RUNNER": "native", **env_overrides},
        timeout=900,
        check=False,
    )
    combined = out.stdout + out.stderr
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :]), combined
    raise AssertionError(f"no result line; stdout={out.stdout!r} stderr={out.stderr[-2000:]!r}")


def test_concurrent_queries_negotiate_the_budget(tmp_path: pathlib.Path) -> None:
    """Two concurrent aggregations share one small budget exactly.

    Contention resolves through negotiated spilling, never failure.
    """
    unpressured, _ = _run_concurrent({})
    pressured, log = _run_concurrent({"DAFT_MEMORY_LIMIT": str(24 * 1024 * 1024)}, spill_dir=str(tmp_path))
    assert pressured == unpressured
    assert "spilled" in log
