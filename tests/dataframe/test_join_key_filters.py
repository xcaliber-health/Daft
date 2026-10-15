"""A join whose build side is small filters the scan on its probe side by its keys.

The scan waits for the join's build side, then reads only what those keys can
match: row groups their range and values rule in, and their rows. The answer is
always the join's own; the filter only changes how much is read.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import daft
from daft import DataFrame, col
from daft.subscribers import Subscriber
from daft.subscribers.events import OperatorStarted, Stats
from tests.conftest import get_tests_daft_runner_name

GROUPS = 1_000
ROWS_PER_GROUP = 50
ROW_GROUP_ROWS = 5_000


class ScanReads(Subscriber):
    """Records the rows each Parquet scan of a query emits."""

    def __init__(self) -> None:
        self.names: dict[int, str] = {}
        self.rows: dict[int, int] = {}

    def on_operator_start(self, event: OperatorStarted) -> None:
        self.names[event.node_id] = event.name

    def on_stats(self, event: Stats) -> None:
        for node, stats in event.stats.items():
            if "rows.out" in stats:
                self.rows[node] = int(stats["rows.out"][1])

    @property
    def rows_read(self) -> int:
        """Rows the Parquet scans emitted."""
        return sum(rows for node, rows in self.rows.items() if self.names.get(node) == "Read Parquet")


def _write(path: Path, keys: list[int]) -> str:
    """Writes one row per entry of ``keys``, in that order, in small row groups."""
    path.mkdir()
    table = pa.table({"k": pa.array(keys, pa.int64()), "v": pa.array(range(len(keys)), pa.int64())})
    pq.write_table(table, path / "part.parquet", row_group_size=ROW_GROUP_ROWS)
    return str(path)


@pytest.fixture(params=["sorted", "spread"], ids=["keys sorted", "keys spread"])
def rows(request: pytest.FixtureRequest, tmp_path: Path) -> str:
    if request.param == "sorted":
        keys = [group for group in range(GROUPS) for _ in range(ROWS_PER_GROUP)]
    else:
        keys = [row % GROUPS for row in range(GROUPS * ROWS_PER_GROUP)]
    return _write(tmp_path / "rows", keys)


def _sums(rows: str) -> DataFrame:
    return daft.read_parquet(rows).groupby("k").agg(col("v").sum().alias("total"))


def _collected(
    frame: Callable[[], DataFrame], *, filtered: bool
) -> tuple[list[tuple[object, ...]], ScanReads]:
    """Runs a newly built frame, so no earlier run's result is reused."""
    reads = ScanReads()
    with daft.execution_config_ctx(enable_join_key_filters=filtered):
        with daft.with_subscriber("join-key-filters", reads):
            answer = frame().collect().to_pydict()
    sorted_rows = sorted(zip(*answer.values()), key=lambda row: tuple((value is None, value) for value in row))
    return sorted_rows, reads


Shape = Callable[[str, DataFrame], DataFrame]

SHAPES: dict[str, Shape] = {
    "semi join over a grouping": lambda rows, keys: _sums(rows).join(keys, on="k", how="semi"),
    "inner join over a grouping": lambda rows, keys: _sums(rows).join(keys, on="k", how="inner"),
    "semi join over a scan": lambda rows, keys: daft.read_parquet(rows).join(keys, on="k", how="semi"),
    "semi join over a renaming projection": lambda rows, keys: (
        daft.read_parquet(rows)
        .select(col("k").alias("key"), col("v"))
        .join(keys.select(col("k").alias("key")), on="key", how="semi")
    ),
}


@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
@pytest.mark.parametrize(
    "keys",
    [
        pytest.param([500, 501, 502], id="three keys"),
        pytest.param([7], id="one key"),
        pytest.param([3, 998], id="two keys far apart"),
        pytest.param([500, None], id="a null key"),
        pytest.param([5_000, 6_000], id="keys no row holds"),
        pytest.param(list(range(0, GROUPS, 2)), id="half the keys"),
    ],
)
def test_the_answer_is_the_joins_own(rows: str, shape: Shape, keys: list[int | None]) -> None:
    def frame() -> DataFrame:
        return shape(rows, daft.from_pydict({"k": pa.array(keys, pa.int64())}))

    filtered, _ = _collected(frame, filtered=True)
    unfiltered, _ = _collected(frame, filtered=False)

    assert filtered == unfiltered


@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
def test_an_empty_build_side_answers_nothing(rows: str, shape: Shape) -> None:
    filtered, _ = _collected(
        lambda: shape(rows, daft.from_pydict({"k": pa.array([], pa.int64())})), filtered=True
    )

    assert filtered == []


GROUPED = {name: shape for name, shape in SHAPES.items() if "grouping" in name}
DIRECT = {name: shape for name, shape in SHAPES.items() if "grouping" not in name}


@pytest.mark.skipif(
    get_tests_daft_runner_name() != "native", reason="the filter is planned by the native runner"
)
@pytest.mark.parametrize("shape", GROUPED.values(), ids=GROUPED.keys())
def test_below_a_grouping_the_scan_emits_only_rows_the_keys_match(rows: str, shape: Shape) -> None:
    def frame() -> DataFrame:
        return shape(rows, daft.from_pydict({"k": [500, 501, 502]}))

    _, filtered = _collected(frame, filtered=True)
    _, unfiltered = _collected(frame, filtered=False)

    assert unfiltered.rows_read == GROUPS * ROWS_PER_GROUP
    assert filtered.rows_read == 3 * ROWS_PER_GROUP


@pytest.mark.skipif(
    get_tests_daft_runner_name() != "native", reason="the filter is planned by the native runner"
)
@pytest.mark.parametrize("shape", DIRECT.values(), ids=DIRECT.keys())
def test_directly_below_a_join_the_scan_skips_row_groups_no_key_is_in(tmp_path: Path, shape: Shape) -> None:
    rows = _write(tmp_path / "rows", [group for group in range(GROUPS) for _ in range(ROWS_PER_GROUP)])

    def frame() -> DataFrame:
        return shape(rows, daft.from_pydict({"k": [500, 501, 502]}))

    _, filtered = _collected(frame, filtered=True)

    # The three groups lie in one row group; the join drops its other rows.
    assert filtered.rows_read == ROW_GROUP_ROWS


@pytest.mark.skipif(
    get_tests_daft_runner_name() != "native", reason="the filter is planned by the native runner"
)
def test_more_keys_than_the_listing_limit_still_filter_rows(rows: str) -> None:
    def frame() -> DataFrame:
        return _sums(rows).join(daft.from_pydict({"k": list(range(100, 110))}), on="k", how="semi")

    with daft.execution_config_ctx(join_key_filter_max_values=5):
        listed, reads = _collected(frame, filtered=True)
    unfiltered, _ = _collected(frame, filtered=False)

    assert listed == unfiltered
    assert reads.rows_read == 10 * ROWS_PER_GROUP


@pytest.mark.skipif(
    get_tests_daft_runner_name() != "native", reason="the filter is planned by the native runner"
)
def test_files_whose_partition_no_key_matches_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "partitioned"
    daft.from_pydict({"k": [group for group in range(10) for _ in range(5)], "v": list(range(50))}).write_parquet(
        str(path), partition_cols=["k"]
    )
    def frame() -> DataFrame:
        return daft.read_parquet(f"{path}/**/*.parquet", hive_partitioning=True).join(
            daft.from_pydict({"k": [3, 4]}), on="k", how="semi"
        )

    filtered, reads = _collected(frame, filtered=True)
    unfiltered, _ = _collected(frame, filtered=False)

    assert filtered == unfiltered
    assert reads.rows_read == 10


@pytest.mark.skipif(
    get_tests_daft_runner_name() != "native", reason="the filter is planned by the native runner"
)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(
            lambda rows, keys: daft.read_parquet(rows).join(keys, on="k", how="anti"),
            id="an anti join, which keeps the rows no key matches",
        ),
        pytest.param(
            lambda rows, keys: daft.read_parquet(rows).join(keys, on="k", how="left"),
            id="a left join, which keeps every probe row",
        ),
        pytest.param(
            lambda rows, keys: daft.read_parquet(rows).limit(10_000).join(keys, on="k", how="semi"),
            id="a limit below the join, whose rows the filter would change",
        ),
        pytest.param(
            lambda rows, keys: daft.read_parquet(rows)
            .with_column("k", col("k") + 1)
            .join(keys, on="k", how="semi"),
            id="a key computed above the scan",
        ),
    ],
)
def test_a_join_the_keys_cannot_restrict_reads_every_row(rows: str, shape: Shape) -> None:
    def frame() -> DataFrame:
        return shape(rows, daft.from_pydict({"k": [500, 501, 502]}))

    filtered, reads = _collected(frame, filtered=True)
    unfiltered, _ = _collected(frame, filtered=False)

    assert filtered == unfiltered
    assert reads.rows_read >= 10_000

