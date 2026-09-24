from __future__ import annotations

import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import daft
from daft.exceptions import DaftCardinalityError


def _tables() -> dict[str, daft.DataFrame]:
    # In `u`, key 1 holds one row, key 2 two distinct rows, key 5 two nulls, key 6 two equal rows.
    # `t` asks for keys 1, 2 and 3; `t13` asks only for keys 1 and 3.
    return {
        "t": daft.from_pydict({"k": [1, 2, 3], "v": [10, 20, 30]}),
        "t13": daft.from_pydict({"k": [1, 3], "v": [10, 30]}),
        "u": daft.from_pydict({"k": [1, 2, 2, 4, 5, 5, 6, 6], "x": ["a", "b", "c", "z", None, None, "d", "d"]}),
    }


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        pytest.param("SELECT (SELECT x FROM u WHERE k = 1) AS s", {"s": ["a"]}, id="one row"),
        pytest.param("SELECT (SELECT x FROM u WHERE k = 99) AS s", {"s": [None]}, id="no rows"),
        pytest.param(
            "SELECT k, (SELECT max(x) FROM u) AS s FROM t",
            {"k": [1, 2, 3], "s": ["z", "z", "z"]},
            id="an aggregate",
        ),
        pytest.param(
            "SELECT k, (SELECT max(x) FROM u WHERE u.k = t.k) AS s FROM t",
            {"k": [1, 2, 3], "s": ["a", "c", None]},
            id="correlated aggregate",
        ),
        pytest.param(
            "SELECT k, (SELECT x FROM u WHERE u.k = t13.k) AS s FROM t13",
            {"k": [1, 3], "s": ["a", None]},
            id="correlated, each key asked for holds one row or none",
        ),
        pytest.param(
            "SELECT k, (SELECT x FROM u WHERE u.k = t.k) AS s FROM t WHERE k <> 2",
            {"k": [1, 3], "s": ["a", None]},
            id="correlated, the row asking for a key of two rows is filtered out first",
        ),
        pytest.param(
            "SELECT k FROM t13 WHERE (SELECT x FROM u WHERE u.k = t13.k) = 'a'",
            {"k": [1]},
            id="correlated, in a filter, each key asked for holds one row or none",
        ),
        pytest.param(
            "SELECT k, (SELECT x FROM u WHERE k = 2) AS s FROM t WHERE k > 99",
            {"k": [], "s": []},
            id="uncorrelated, no row reaches a subquery of two rows",
        ),
    ],
)
def test_a_scalar_subquery_answers_one_value_per_row(query: str, expected: dict[str, list[object]]) -> None:
    answer = daft.sql(query, **_tables())
    # Sorted here rather than in the query: the order is not what these cases are about.
    if "k" in answer.column_names:
        answer = answer.sort("k")

    assert answer.to_pydict() == expected


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("SELECT (SELECT x FROM u WHERE k = 2) AS s", id="two distinct rows"),
        pytest.param("SELECT (SELECT x FROM u WHERE k = 6) AS s", id="two equal rows"),
        pytest.param("SELECT (SELECT x FROM u WHERE k = 5) AS s", id="two null rows"),
        pytest.param(
            "SELECT k, (SELECT x FROM u WHERE u.k = t.k) AS s FROM t", id="correlated, a key asked for holds two rows"
        ),
        pytest.param(
            "SELECT k FROM t WHERE (SELECT x FROM u WHERE u.k = t.k) = 'a'",
            id="correlated, in a filter, a key asked for holds two rows",
        ),
    ],
)
def test_a_scalar_subquery_of_more_than_one_row_fails_the_query(query: str) -> None:
    with pytest.raises(DaftCardinalityError):
        daft.sql(query, **_tables()).collect()


def test_a_subquery_answers_its_row_when_the_other_files_hold_none(tmp_path: pathlib.Path) -> None:
    # The subquery's filter leaves every file but one without rows, so most of its partial
    # aggregations see nothing; the one row left must still be the answer.
    for i in range(4):
        rows = {"k": [i] * 100, "v": list(range(i * 100, i * 100 + 100))}
        pq.write_table(pa.table(rows), tmp_path / f"part-{i}.parquet")
    tables = {"f": daft.read_parquet(str(tmp_path / "*.parquet")), "t": daft.from_pydict({"v": [350, 351]})}

    answer = daft.sql("SELECT v FROM t WHERE v = (SELECT DISTINCT v FROM f WHERE k = 3 AND v = 350)", **tables)

    assert answer.to_pydict() == {"v": [350]}
