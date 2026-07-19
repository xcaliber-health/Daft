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
