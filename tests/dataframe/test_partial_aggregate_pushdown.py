"""An aggregation over a join answers the same when partially aggregated below the join."""

from __future__ import annotations

import datetime
import decimal
import io
from collections.abc import Callable, Iterator

import duckdb
import pyarrow as pa
import pytest

import daft
from daft import DataFrame, Expression, col
from daft.context import execution_config_ctx, planning_config_ctx

# Subjects linked to owners: a self-join on owner repeats each subject once per peer.
LINK = {
    "subject": [1, 1, 2, 2, 3, 3, 4, 5, 5, 6, None, 7],
    "owner": [10, 11, 10, 12, 11, 12, 10, None, 13, 13, 10, 14],
    "weight": [5, None, 7, 1, -3, 4, 2, 9, None, 0, 8, 6],
    "flag": [True, False, True, None, True, True, False, True, None, False, True, True],
}
PEERS = {
    "owner": [10, 10, 11, 12, 12, 12, 13, None, 15],
    "peer": [100, 101, 102, None, 104, 105, 106, 107, 108],
    "score": [1.5, -2.0, None, 3.25, 0.0, 4.5, -1.0, 2.0, 9.0],
}

# The modes the operator may run in, each forced: probing for ever, accumulating, passing
# every row through, and accumulating with every partial sent on at once.
MODES = {
    "probing": {"partial_aggregate_pushdown_probe_rows": 1 << 40},
    "accumulating": {
        "partial_aggregate_pushdown_probe_rows": 1,
        "partial_aggregate_pushdown_skip_ratio": 1.0,
    },
    "passing_through": {
        "partial_aggregate_pushdown_probe_rows": 1,
        "partial_aggregate_pushdown_skip_ratio": 0.0,
    },
    "flushing": {
        "partial_aggregate_pushdown_probe_rows": 1,
        "partial_aggregate_pushdown_skip_ratio": 1.0,
        "partial_aggregate_pushdown_max_buffer_bytes": 1,
    },
}


@pytest.fixture(params=sorted(MODES))
def mode(request: pytest.FixtureRequest) -> Iterator[str]:
    with execution_config_ctx(**MODES[request.param]):
        yield request.param


def frame(data: dict[str, list[object]], partitions: int) -> DataFrame:
    df = daft.from_pydict(data)
    return df.into_partitions(partitions) if partitions > 1 else df


def in_order(columns: dict[str, list[object]], keys: list[str]) -> dict[str, list[object]]:
    """The rows of `columns` ordered by `keys`, nulls first, whatever order they came in."""
    names = list(columns)
    rows = list(zip(*columns.values()))
    positions = [names.index(k) for k in keys]
    rows.sort(key=lambda row: [(row[i] is not None, row[i]) for i in positions])
    return {name: [row[i] for row in rows] for i, name in enumerate(names)}


def answer(build: Callable[[], DataFrame], keys: list[str], pushdown: bool) -> dict[str, list[object]]:
    with planning_config_ctx(disable_partial_aggregate_pushdown=not pushdown):
        return in_order(build().to_pydict(), keys)


def plan(build: Callable[[], DataFrame]) -> str:
    out = io.StringIO()
    build().explain(show_all=True, file=out)
    return out.getvalue().split("Optimized Logical Plan")[1].split("Physical Plan")[0]


def duckdb_answer(sql: str, keys: list[str]) -> dict[str, list[object]]:
    link = pa.table(LINK)  # noqa: F841 - read by DuckDB by name
    peers = pa.table(PEERS)  # noqa: F841 - read by DuckDB by name
    return in_order(duckdb.sql(sql).arrow().read_all().to_pydict(), keys)


AGGREGATIONS: dict[str, tuple[Expression, str]] = {
    "count": (col("peer").count(), "count(peer)"),
    "count_all": (col("peer").count("all"), "count(*)"),
    "count_null": (col("peer").count("null"), "count(*) - count(peer)"),
    "sum": (col("peer").sum(), "sum(peer)"),
    "min": (col("peer").min(), "min(peer)"),
    "max": (col("peer").max(), "max(peer)"),
    "mean": (col("peer").mean(), "avg(peer)"),
    "float_sum": (col("score").sum(), "sum(score)"),
}


@pytest.mark.parametrize("partitions", [1, 3])
@pytest.mark.parametrize("name", sorted(AGGREGATIONS))
def test_an_aggregation_of_the_repeated_side_answers_as_without_pushdown(name: str, partitions: int, mode: str) -> None:
    aggregation, sql = AGGREGATIONS[name]

    def build() -> DataFrame:
        return (
            frame(LINK, partitions)
            .join(frame(PEERS, partitions), on="owner")
            .groupby("subject")
            .agg(aggregation.alias("v"))
        )

    pushed = answer(build, ["subject"], pushdown=True)

    assert "PartialAggregation" in plan(build)
    assert pushed == answer(build, ["subject"], pushdown=False)
    expected = duckdb_answer(
        f"SELECT subject, {sql} AS v FROM link JOIN peers USING (owner) GROUP BY subject", ["subject"]
    )
    if name in ("mean", "float_sum"):
        assert pushed["subject"] == expected["subject"]
        assert pushed["v"] == pytest.approx(expected["v"], nan_ok=True)
    else:
        assert pushed == expected


@pytest.mark.parametrize("partitions", [1, 3])
def test_several_aggregations_and_a_grouping_of_both_sides_answer_as_without_pushdown(
    partitions: int, mode: str
) -> None:
    def build() -> DataFrame:
        return (
            frame(LINK, partitions)
            .join(frame(PEERS, partitions), on="owner")
            .groupby("subject", "peer")
            .agg(
                col("weight").sum().alias("total"),
                col("weight").mean().alias("average"),
                col("flag").bool_and().alias("all_flagged"),
                col("flag").bool_or().alias("any_flagged"),
                col("weight").count().alias("n"),
            )
        )

    keys = ["subject", "peer"]
    assert "PartialAggregation" in plan(build)
    assert answer(build, keys, pushdown=True) == answer(build, keys, pushdown=False)


@pytest.mark.parametrize(
    ("how", "column"),
    [
        pytest.param("left", "weight", id="left_join_of_the_preserved_side"),
        pytest.param("right", "peer", id="right_join_of_the_preserved_side"),
    ],
)
def test_an_outer_side_preserved_by_its_join_is_aggregated_below_it(how: str, column: str, mode: str) -> None:
    def build() -> DataFrame:
        return (
            daft.from_pydict(LINK)
            .join(daft.from_pydict(PEERS), on="owner", how=how)
            .groupby("owner")
            .agg(col(column).sum().alias("s"), col(column).count().alias("n"))
        )

    assert "PartialAggregation" in plan(build)
    assert answer(build, ["owner"], pushdown=True) == answer(build, ["owner"], pushdown=False)


@pytest.mark.parametrize(
    ("how", "column"),
    [
        pytest.param("left", "peer", id="null_supplying_side"),
        pytest.param("outer", "peer", id="outer_join"),
    ],
)
def test_a_side_the_join_may_supply_nulls_for_is_not_aggregated_below_it(how: str, column: str) -> None:
    def build() -> DataFrame:
        return (
            daft.from_pydict(LINK)
            .join(daft.from_pydict(PEERS), on="owner", how=how)
            .groupby("subject")
            .agg(col(column).sum().alias("s"))
        )

    assert "PartialAggregation" not in plan(build)


def test_a_distinct_count_is_not_aggregated_below_the_join() -> None:
    def build() -> DataFrame:
        return (
            daft.from_pydict(LINK)
            .join(daft.from_pydict(PEERS), on="owner")
            .groupby("subject")
            .agg(col("peer").count_distinct().alias("n"))
        )

    assert "PartialAggregation" not in plan(build)


def test_turning_pushdown_off_leaves_the_plan_unchanged() -> None:
    def build() -> DataFrame:
        return (
            daft.from_pydict(LINK).join(daft.from_pydict(PEERS), on="owner").groupby("subject").agg(col("peer").sum())
        )

    with planning_config_ctx(disable_partial_aggregate_pushdown=True):
        assert "PartialAggregation" not in plan(build)


@pytest.mark.parametrize("empty", [False, True], ids=["rows", "no_rows"])
def test_an_ungrouped_count_over_the_join_counts_zero_rows_as_zero(empty: bool, mode: str) -> None:
    def build() -> DataFrame:
        peers = daft.from_pydict(PEERS)
        if empty:
            peers = peers.where(col("owner") < 0)
        return (
            daft.from_pydict(LINK)
            .join(peers, on="owner")
            .agg(col("weight").count().alias("n"), col("weight").sum().alias("s"))
        )

    pushed = build().to_pydict()

    assert pushed == answer(build, ["n"], pushdown=False)
    if empty:
        assert pushed == {"n": [0], "s": [None]}


@pytest.mark.parametrize(
    "keys",
    [
        pytest.param([f"k{i % 4}" for i in range(12)], id="utf8"),
        pytest.param([decimal.Decimal(f"{i % 4}.50") for i in range(12)], id="decimal"),
        pytest.param([datetime.date(2026, 1, 1 + i % 4) for i in range(12)], id="date"),
        pytest.param([i % 4 == 0 for i in range(12)], id="bool"),
    ],
)
def test_join_keys_of_each_type_answer_as_without_pushdown(keys: list[object], mode: str) -> None:
    left = {"g": list(range(12)), "k": keys, "v": [decimal.Decimal(f"{i}.25") for i in range(12)]}
    right = {"k": keys[:6] + keys[:3], "w": list(range(9))}

    def build() -> DataFrame:
        return (
            daft.from_pydict(right)
            .join(daft.from_pydict(left), on="k")
            .groupby("w")
            .agg(col("v").sum().alias("s"), col("v").max().alias("m"))
        )

    assert "PartialAggregation" in plan(build)
    assert answer(build, ["w"], pushdown=True) == answer(build, ["w"], pushdown=False)


def test_a_float_join_key_is_not_grouped_on() -> None:
    left = {"k": [0.0, -0.0, float("nan"), 1.0], "v": [1, 2, 3, 4]}
    right = {"k": [0.0, float("nan"), 1.0], "g": [1, 2, 3]}

    def build() -> DataFrame:
        return daft.from_pydict(right).join(daft.from_pydict(left), on="k").groupby("g").agg(col("v").sum())

    assert "PartialAggregation" not in plan(build)


def test_two_joins_aggregate_below_the_lower_one(mode: str) -> None:
    # Patients seen by providers, providers attributed to groups: count attributions per patient.
    visits = {"patient": [1, 1, 1, 2, 2, 3, 3, 3, 3], "provider": [7, 7, 8, 7, 9, 8, 8, 9, 9]}
    providers = {"provider": [7, 8, 9], "provider_id": [70, 80, 90]}
    attribution = {
        "provider_id": [70, 70, 80, 90, 90, 90],
        "attribution_id": [1, 2, 3, 4, 5, None],
    }

    def build() -> DataFrame:
        return (
            daft.from_pydict(visits)
            .groupby("patient", "provider")
            .agg()
            .join(daft.from_pydict(providers), on="provider")
            .join(daft.from_pydict(attribution), on="provider_id")
            .groupby("patient")
            .agg(col("attribution_id").count().alias("n"))
        )

    assert "PartialAggregation" in plan(build)
    assert answer(build, ["patient"], pushdown=True) == {"patient": [1, 2, 3], "n": [3, 4, 3]}
    assert answer(build, ["patient"], pushdown=True) == answer(build, ["patient"], pushdown=False)


@pytest.mark.parametrize("partitions", [1, 3])
def test_a_parquet_source_answers_as_without_pushdown(tmp_path, partitions: int, mode: str) -> None:
    daft.from_pydict(LINK).into_partitions(partitions).write_parquet(str(tmp_path / "link"))
    daft.from_pydict(PEERS).write_parquet(str(tmp_path / "peers"))

    def build() -> DataFrame:
        return (
            daft.read_parquet(str(tmp_path / "link"))
            .join(daft.read_parquet(str(tmp_path / "peers")), on="owner")
            .groupby("subject")
            .agg(col("weight").sum().alias("s"), col("weight").count().alias("n"))
        )

    assert answer(build, ["subject"], pushdown=True) == answer(build, ["subject"], pushdown=False)


def test_many_rows_answer_as_without_pushdown(mode: str) -> None:
    n = 50_000
    fact = {"subject": [i % 997 for i in range(n)], "owner": [i % 101 for i in range(n)], "v": list(range(n))}
    dim = {"owner": [i % 101 for i in range(303)], "peer": list(range(303))}

    def build() -> DataFrame:
        return (
            daft.from_pydict(dim)
            .join(daft.from_pydict(fact).into_partitions(4), on="owner")
            .groupby("subject")
            .agg(col("v").sum().alias("s"), col("v").count().alias("n"), col("v").min().alias("lo"))
        )

    assert answer(build, ["subject"], pushdown=True) == answer(build, ["subject"], pushdown=False)


def rows_out_of_partial_aggregate(df: DataFrame) -> int:
    metrics = df.metrics.to_pydict()
    return sum(
        int({k: v["value"] for k, v in stats}["rows.out"])
        for name, stats in zip(metrics["name"], metrics["stats"])
        if name == "PartialGroupedAggregate"
    )


@pytest.mark.parametrize("source", ["memory", "parquet"])
def test_rows_a_filtering_join_drops_are_dropped_before_they_are_aggregated(tmp_path, source: str) -> None:
    n = 20_000
    fact = {"k": list(range(n)), "v": [i % 7 for i in range(n)]}
    dim = {"k": list(range(0, n, 100)), "g": [i % 3 for i in range(0, n, 100)]}

    def read(name: str, data: dict[str, list[object]]) -> DataFrame:
        if source == "memory":
            return daft.from_pydict(data)
        daft.from_pydict(data).write_parquet(str(tmp_path / name))
        return daft.read_parquet(str(tmp_path / name))

    facts, dims = read("fact", fact), read("dim", dim)

    def build() -> DataFrame:
        return facts.join(dims, on="k").groupby("g").agg(col("v").sum().alias("s"), col("v").count().alias("n"))

    with planning_config_ctx(disable_partial_aggregate_pushdown=False):
        df = build().collect()

    assert "PartialAggregation" in plan(build)
    assert in_order(df.to_pydict(), ["g"]) == answer(build, ["g"], pushdown=False)
    # Grouped by the unique key, it answers one row per key it kept: only the join's.
    assert rows_out_of_partial_aggregate(df) == len(dim["k"])


def test_a_join_whose_build_side_holds_no_key_aggregates_no_row(mode: str) -> None:
    fact = {"k": list(range(5_000)), "v": list(range(5_000))}

    def build() -> DataFrame:
        dims = daft.from_pydict({"k": [1, 2], "g": [1, 2]}).where(col("g") > 5)
        return daft.from_pydict(fact).join(dims, on="k").groupby("g").agg(col("v").sum().alias("s"))

    assert answer(build, ["g"], pushdown=True) == {"g": [], "s": []}


def test_a_build_side_estimated_large_still_filters_the_partial_aggregation() -> None:
    # The build side joins an aggregation kept by a condition few groups meet, so it is
    # estimated far larger than it is, as TPC-H Q18's orders, totals and customers are.
    n = 20_000
    fact = {"k": [i % n for i in range(5 * n)], "v": [i % 7 for i in range(5 * n)]}
    totals = {"k": [i % n for i in range(4 * n)], "q": [1 if i % n % 1000 else 100 for i in range(4 * n)]}
    labels = {"k": list(range(n)), "label": [i % 3 for i in range(n)]}

    def build() -> DataFrame:
        kept = (
            daft.from_pydict(totals)
            .groupby("k")
            .agg(col("q").sum().alias("total"))
            .where(col("total") > 300)
            .join(daft.from_pydict(labels), on="k")
        )
        return daft.from_pydict(fact).join(kept, on="k").groupby("total").agg(col("v").sum().alias("s"))

    # The build side is estimated past the size a scan waits for, as Q18's is, while its 20
    # keys still fit the budget for hashing them.
    with (
        planning_config_ctx(disable_partial_aggregate_pushdown=False),
        execution_config_ctx(broadcast_join_size_bytes_threshold=1024),
    ):
        df = build().collect()

    assert "PartialAggregation" in plan(build)
    assert in_order(df.to_pydict(), ["total"]) == answer(build, ["total"], pushdown=False)
    assert rows_out_of_partial_aggregate(df) == n // 1000


def test_a_self_join_of_one_source_answers_as_without_pushdown(mode: str) -> None:
    link = daft.from_pydict(LINK)

    def build() -> DataFrame:
        peers = link.select(col("owner"), col("subject").alias("peer"))
        return link.join(peers, on="owner").groupby("subject").agg(col("peer").count().alias("n"))

    assert answer(build, ["subject"], pushdown=True) == answer(build, ["subject"], pushdown=False)
