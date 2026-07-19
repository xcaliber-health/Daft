"""Remote execution must produce exactly what local execution produces.

Each case builds a DataFrame program and executes it twice — through the
serving endpoint and through the in-process engine — asserting identical
results. Programs cover the core relational surface plus user-defined
functions shipped over the wire.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

import daft
from daft import col

from .conftest import assert_remote_matches_native

if TYPE_CHECKING:
    from daft.dataframe import DataFrame
    from daft.runners.native_runner import NativeRunner
    from daft.runners.remote_runner import RemoteRunner


def _base_df() -> DataFrame:
    return daft.from_pydict(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "group": ["a", "b", "a", "b", "c", None],
            "value": [10.0, None, 30.0, 40.0, 50.0, 60.0],
        }
    )


def _filter_select(df: DataFrame) -> DataFrame:
    return df.where(col("id") > 2).select("id", "group")


def _with_columns(df: DataFrame) -> DataFrame:
    return df.with_columns({"doubled": col("id") * 2, "flag": col("value") > 25})


def _groupby_agg(df: DataFrame) -> DataFrame:
    return df.groupby("group").agg(col("value").sum().alias("total"), col("id").count().alias("n"))


def _sort_limit(df: DataFrame) -> DataFrame:
    return df.sort("id", desc=True).limit(3)


def _distinct(df: DataFrame) -> DataFrame:
    return df.select("group").distinct()


def _global_agg(df: DataFrame) -> DataFrame:
    return df.agg(col("value").mean().alias("avg"), col("id").max().alias("mx"))


def _empty_result(df: DataFrame) -> DataFrame:
    return df.where(col("id") > 1000)


def _explode() -> DataFrame:
    return daft.from_pydict({"k": [1, 2], "vals": [[1, 2, 3], [4]]}).explode(col("vals"))


def _concat() -> DataFrame:
    a = daft.from_pydict({"x": [1, 2]})
    b = daft.from_pydict({"x": [3, 4]})
    return a.concat(b)


def _udf_batch() -> DataFrame:
    @daft.func(return_dtype=daft.DataType.int64())
    def add_one(v: int) -> int:
        return v + 1

    return daft.from_pydict({"v": [1, 2, 3]}).select(add_one(col("v")).alias("v1"))


def _window_row_number() -> DataFrame:
    from daft.functions import row_number
    from daft.window import Window

    window = Window().partition_by("group").order_by("id")
    return _base_df().select(col("id"), col("group"), row_number().over(window).alias("rn"))


def _window_rank() -> DataFrame:
    from daft.functions import dense_rank, rank
    from daft.window import Window

    window = Window().partition_by("group").order_by("value")
    return _base_df().select(
        col("id"),
        rank().over(window).alias("rk"),
        dense_rank().over(window).alias("drk"),
    )


def _window_partition_sum() -> DataFrame:
    from daft.window import Window

    window = Window().partition_by("group")
    return _base_df().select(col("id"), col("value").sum().over(window).alias("group_total"))


def _pivot() -> DataFrame:
    df = daft.from_pydict({"g": ["a", "a", "b"], "k": ["x", "y", "x"], "v": [1.0, 2.0, 3.0]})
    return df.pivot(group_by="g", pivot_col="k", value_col="v", agg_fn="sum", names=["x", "y"])


def _unpivot() -> DataFrame:
    df = daft.from_pydict({"k": [1, 2], "a": [10, 20], "b": [30, 40]})
    return df.unpivot(ids="k", values=["a", "b"])


def _struct_get_and_unnest() -> DataFrame:
    from daft.functions import unnest

    df = daft.from_pydict({"st": [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]})
    return df.select(unnest(col("st")))


def _string_namespace() -> DataFrame:
    from daft.functions import contains

    df = daft.from_pydict({"s": ["apple", "banana", None, "cherry"]})
    return df.select(col("s"), contains(col("s"), "an").alias("has_an"))


def _list_namespace() -> DataFrame:
    from daft.functions import list_join, list_sum

    df = daft.from_pydict({"l": [[1, 2, 3], [4], []], "w": [["a", "b"], ["c"], []]})
    return df.select(list_sum(col("l")).alias("total"), list_join(col("w"), "-").alias("joined"))


def _map_namespace() -> DataFrame:
    import pyarrow as pa

    from daft.functions import map_get

    entries = pa.array(
        [[("k1", 1), ("k2", 2)], [("k3", 3)]],
        type=pa.map_(pa.string(), pa.int64()),
    )
    df = daft.from_arrow(pa.table({"m": entries}))
    return df.select(map_get(col("m"), "k1").alias("k1"))


def _temporal_namespace() -> DataFrame:
    import datetime

    from daft.functions import day, month, year

    df = daft.from_pydict({"d": [datetime.date(2020, 1, 2), datetime.date(2021, 6, 30), None]})
    return df.select(year(col("d")).alias("y"), month(col("d")).alias("m"), day(col("d")).alias("dy"))


def _extended_aggs() -> DataFrame:
    from daft.functions import list_agg, string_agg

    return _base_df().agg(
        col("value").stddev().alias("sd"),
        col("id").count_distinct().alias("cd"),
        list_agg(col("id")).alias("ids"),
        string_agg(col("group")).alias("groups"),
        col("value").any_value().alias("any_v"),
    )


def _into_partitions() -> DataFrame:
    return _base_df().into_partitions(3).where(col("id") > 1)


def _repartition() -> DataFrame:
    return _base_df().repartition(2, "group")


CASES: dict[str, tuple[Callable[[], DataFrame], str | None]] = {
    "filter_select": (lambda: _filter_select(_base_df()), None),
    "with_columns": (lambda: _with_columns(_base_df()), None),
    "groupby_agg": (lambda: _groupby_agg(_base_df()), "group"),
    "sort_limit": (lambda: _sort_limit(_base_df()), None),
    "distinct": (lambda: _distinct(_base_df()), "group"),
    "global_agg": (lambda: _global_agg(_base_df()), None),
    "empty_result": (lambda: _empty_result(_base_df()), None),
    "explode": (_explode, None),
    "concat": (_concat, "x"),
    "udf_batch": (_udf_batch, None),
    "window_row_number": (_window_row_number, "id"),
    "window_rank": (_window_rank, "id"),
    "window_partition_sum": (_window_partition_sum, "id"),
    "pivot": (_pivot, "g"),
    "unpivot": (_unpivot, "k"),
    "struct_get_and_unnest": (_struct_get_and_unnest, None),
    "string_namespace": (_string_namespace, None),
    "list_namespace": (_list_namespace, None),
    "map_namespace": (_map_namespace, None),
    "temporal_namespace": (_temporal_namespace, None),
    "extended_aggs": (_extended_aggs, None),
    "into_partitions": (_into_partitions, "id"),
    "repartition": (_repartition, "id"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_remote_matches_native(case: str, remote_runner: RemoteRunner, native_runner: NativeRunner) -> None:
    build, sort_key = CASES[case]
    assert_remote_matches_native(remote_runner, native_runner, build(), sort_key=sort_key)


JOIN_KINDS = ["inner", "left", "right", "outer", "semi", "anti"]


@pytest.mark.parametrize("how", JOIN_KINDS)
def test_joins_match_native(how: str, remote_runner: RemoteRunner, native_runner: NativeRunner) -> None:
    left = daft.from_pydict({"k": [1, 2, 3, 5], "l": ["a", "b", "c", "e"]})
    right = daft.from_pydict({"k": [2, 3, 4], "r": ["x", "y", "z"]})
    df = left.join(right, on="k", how=how)
    assert_remote_matches_native(remote_runner, native_runner, df, sort_key="k")


def test_multiple_in_memory_sources_ship_together(remote_runner: RemoteRunner, native_runner: NativeRunner) -> None:
    a = daft.from_pydict({"k": [1, 2], "a": [10, 20]})
    b = daft.from_pydict({"k": [1, 2], "b": [100, 200]})
    c = daft.from_pydict({"k": [2, 3], "c": [7, 8]})
    df = a.join(b, on="k").join(c, on="k", how="left")
    assert_remote_matches_native(remote_runner, native_runner, df, sort_key="k")


def test_read_parquet_scans_execute_server_side(
    tmp_path_factory: pytest.TempPathFactory,
    remote_runner: RemoteRunner,
    native_runner: NativeRunner,
) -> None:
    path = str(tmp_path_factory.mktemp("scan") / "data.parquet")
    daft.from_pydict({"n": list(range(50)), "s": [f"row{i}" for i in range(50)]}).write_parquet(path)

    df = daft.read_parquet(path).where(col("n") % 2 == 0).select("n")
    assert_remote_matches_native(remote_runner, native_runner, df, sort_key="n")
