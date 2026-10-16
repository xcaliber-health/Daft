"""A join waiting for its build side reads only a bounded part of its probe side.

The probe side of a join starts with the build side, so its first rows are
ready when the build lands. While the build side is still running, the probe
side holds about one batch per worker and then stops reading, instead of
holding its whole input in memory.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable

import pytest

import daft
from daft import DataFrame, Series, col
from tests.conftest import get_tests_daft_runner_name

pytestmark = pytest.mark.skipif(
    get_tests_daft_runner_name() != "native", reason="probes the native runner's join pipeline"
)

PROBE_ROWS = 1_000_000
BATCH_ROWS = 1_024
BUILD_KEYS = [0, 7, 500_000, 999_999]
# Reading stops once it has not advanced for this long.
SETTLE_SECONDS = 0.5
# Long enough to read the whole probe side many times over.
TIMEOUT_SECONDS = 20.0


class ProbeReads:
    """Counts the probe side's rows as they are read, and what had been read when the build side ended."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows = 0
        self.when_built: int | None = None

    def add(self, rows: int) -> None:
        with self._lock:
            self._rows += rows

    def rows(self) -> int:
        with self._lock:
            return self._rows

    def wait_until_reading_stops(self) -> None:
        deadline = time.monotonic() + TIMEOUT_SECONDS
        seen, still_since = self.rows(), time.monotonic()
        while time.monotonic() < deadline:
            time.sleep(0.05)
            now = self.rows()
            if now != seen:
                seen, still_since = now, time.monotonic()
            elif time.monotonic() - still_since >= SETTLE_SECONDS:
                break
        self.when_built = self.rows()


READS = ProbeReads()


def _reads() -> ProbeReads:
    """The current test's counter, looked up when called.

    Functions are sent to the engine by value, which would copy the counter;
    these run in this process, so they find the live one through this module.
    """
    return sys.modules[__name__].READS


@daft.func.batch(return_dtype=daft.DataType.int64(), use_process=False)
def read_probe(key: Series) -> Series:
    _reads().add(len(key))
    return key


@daft.func.batch(return_dtype=daft.DataType.int64(), use_process=False)
def build_after_probe_settles(key: Series) -> Series:
    _reads().wait_until_reading_stops()
    return key


@pytest.fixture
def reads() -> ProbeReads:
    global READS
    READS = ProbeReads()
    return READS


def _sides() -> tuple[DataFrame, DataFrame]:
    # In memory, so its size is known and the join builds on the other side.
    probe = daft.from_pydict({"id": list(range(PROBE_ROWS))}).select(
        read_probe(col("id")).alias("k"), col("id").alias("v")
    )
    build = daft.from_pydict({"k": BUILD_KEYS, "w": [str(k) for k in BUILD_KEYS]}).select(
        build_after_probe_settles(col("k")).alias("k"), col("w")
    )
    return probe, build


Shape = Callable[[DataFrame, DataFrame], DataFrame]
Row = tuple[int, str | None]

KEYED = {k: str(k) for k in BUILD_KEYS}
EVERY_PROBE_ROW: list[Row] = [(k, KEYED.get(k)) for k in range(PROBE_ROWS)]
MATCHED: list[Row] = [(k, w) for k, w in EVERY_PROBE_ROW if w is not None]
UNMATCHED: list[Row] = [(k, w) for k, w in EVERY_PROBE_ROW if w is None]

SHAPES: dict[str, tuple[Shape, list[Row]]] = {
    "inner join": (lambda probe, build: probe.join(build, on="k"), MATCHED),
    "left join": (lambda probe, build: probe.join(build, on="k", how="left"), EVERY_PROBE_ROW),
    "right join, building on the left": (
        lambda probe, build: build.join(probe, on="k", how="right"),
        EVERY_PROBE_ROW,
    ),
    "semi join": (lambda probe, build: probe.join(build, on="k", how="semi"), MATCHED),
    "anti join": (lambda probe, build: probe.join(build, on="k", how="anti"), UNMATCHED),
}


def _rows(result: dict[str, list[int | str | None]]) -> list[Row]:
    """The result's keys with their build-side values, or the probe side's alone when it has none."""
    keyed = result.get("w", [KEYED.get(k) for k in result["k"]])
    return sorted(zip(result["k"], keyed))


@pytest.mark.parametrize("maintain_order", [False, True], ids=["unordered", "ordered"])
@pytest.mark.parametrize("shape,expected", SHAPES.values(), ids=SHAPES.keys())
def test_a_join_waiting_for_its_build_side_stops_reading_its_probe_side(
    reads: ProbeReads, shape: Shape, expected: list[Row], maintain_order: bool
) -> None:
    probe, build = _sides()

    with daft.execution_config_ctx(
        default_morsel_size=BATCH_ROWS, enable_join_key_filters=False, maintain_order=maintain_order
    ):
        result = shape(probe, build).to_pydict()

    assert reads.when_built is not None
    assert reads.when_built < PROBE_ROWS // 4
    assert _rows(result) == expected


def test_an_ordered_left_join_waiting_for_its_build_side_returns_every_probe_row_in_order(reads: ProbeReads) -> None:
    probe, build = _sides()

    with daft.execution_config_ctx(default_morsel_size=BATCH_ROWS, enable_join_key_filters=False, maintain_order=True):
        result = probe.join(build, on="k", how="left").to_pydict()

    assert reads.when_built is not None
    assert reads.when_built < PROBE_ROWS // 4
    assert result["v"] == list(range(PROBE_ROWS))
    assert {k: w for k, w in zip(result["k"], result["w"]) if w is not None} == {k: str(k) for k in BUILD_KEYS}


ProbeQuery = Callable[[DataFrame, DataFrame], dict[str, list[int | str | None]]]

OTHER_JOINS: dict[str, tuple[ProbeQuery, dict[str, list[int | str | None]]]] = {
    "cross join": (
        lambda probe, build: (
            probe.join(build.select("w", col("k").alias("built")), how="cross")
            .groupby("w")
            .agg(col("k").count().alias("probe_rows"), col("built").count().alias("built_rows"))
            .sort("w")
            .to_pydict()
        ),
        {
            "w": sorted(KEYED.values()),
            "probe_rows": [PROBE_ROWS] * len(BUILD_KEYS),
            "built_rows": [PROBE_ROWS] * len(BUILD_KEYS),
        },
    ),
    "asof join, building on the left": (
        lambda probe, build: (
            build.with_column("t", col("k"))
            .join_asof(probe.with_column("t", col("v")), on="t", by="k")
            .select("k", "w", "v")
            .sort("k")
            .to_pydict()
        ),
        {"k": BUILD_KEYS, "w": [KEYED[k] for k in BUILD_KEYS], "v": BUILD_KEYS},
    ),
}


@pytest.mark.parametrize("query,expected", OTHER_JOINS.values(), ids=OTHER_JOINS.keys())
def test_a_cross_or_asof_join_waiting_for_its_build_side_stops_reading_its_probe_side(
    reads: ProbeReads, query: ProbeQuery, expected: dict[str, list[int | str | None]]
) -> None:
    probe, build = _sides()

    with daft.execution_config_ctx(default_morsel_size=BATCH_ROWS, enable_join_key_filters=False):
        result = query(probe, build)

    assert reads.when_built is not None
    assert reads.when_built < PROBE_ROWS // 4
    assert result == expected
