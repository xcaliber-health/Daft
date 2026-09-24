from __future__ import annotations

import pathlib

import pyarrow as pa
import pyarrow.parquet as pq

import daft
from daft import col


def test_a_global_any_value_is_not_taken_from_an_empty_partial(tmp_path: pathlib.Path) -> None:
    # The filter leaves every file but one without rows. A partial aggregation over no rows
    # still answers one null row, which must not stand for the value of the one row left.
    for i in range(4):
        rows = {"k": [i] * 100, "v": list(range(i * 100, i * 100 + 100))}
        pq.write_table(pa.table(rows), tmp_path / f"part-{i}.parquet")
    df = daft.read_parquet(str(tmp_path / "*.parquet")).where((col("k") == 3) & (col("v") == 350))

    answers = [df.agg(col("v").any_value()).to_pydict()["v"] for _ in range(5)]

    assert answers == [[350]] * 5


def test_a_global_any_value_of_nulls_answers_null(tmp_path: pathlib.Path) -> None:
    for i in range(2):
        pq.write_table(pa.table({"v": pa.array([None, None], pa.int64())}), tmp_path / f"part-{i}.parquet")

    answer = daft.read_parquet(str(tmp_path / "*.parquet")).agg(col("v").any_value()).to_pydict()

    assert answer == {"v": [None]}
