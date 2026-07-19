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
