"""A restriction on one side's join key also restricts the other side's read.

When every row one side of ``left_on = right_on`` keeps satisfies a comparison of its key with
literals, the other side is filtered by the same comparison of its own key wherever the join lets
that side's unmatched rows go. The filter moves down only as far as any filter would: through a
window only on its partition keys, through a grouping only on its grouping keys. The answer is
always the join's own; inference only changes what is read.
"""

from __future__ import annotations

import io
import re
from collections.abc import Callable
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import daft
from daft import DataFrame, Window, col, lit
from daft.context import planning_config_ctx
from daft.functions import row_number, upper
from tests.conftest import get_tests_daft_runner_name

pytestmark = pytest.mark.skipif(
    get_tests_daft_runner_name() == "ray", reason="plan shape is checked on the native runner"
)

SUBJECTS = pa.table({"id": ["p1", "p2", "p3", "p4"], "name": ["a", "b", "c", "d"]})
# Record 11's latest version moves it from p1 to p3; p4 has no events; one event has no subject.
EVENTS = pa.table(
    {
        "record": pa.array([10, 10, 11, 11, 12, 13, 14], pa.int64()),
        "version": pa.array([1, 2, 1, 2, 1, 1, 1], pa.int64()),
        "sid": ["p1", "p1", "p1", "p3", "p2", None, "p3"],
        "v": pa.array([5, 6, 7, 8, 9, 10, 11], pa.int64()),
    }
)
CHOSEN = ["p1", "p2"]


def _parquet(path: Path, table: pa.Table) -> str:
    path.mkdir()
    pq.write_table(table, path / "part.parquet")
    return str(path)


Reader = Callable[[], DataFrame]


def _reader(path: str) -> Reader:
    """Reads ``path`` anew on each call, so each read takes the planning config in force."""
    return lambda: daft.read_parquet(path)


@pytest.fixture
def subjects(tmp_path: Path) -> Reader:
    return _reader(_parquet(tmp_path / "subjects", SUBJECTS))


@pytest.fixture
def events(tmp_path: Path) -> Reader:
    return _reader(_parquet(tmp_path / "events", EVENTS))


def _optimized_plan(df: DataFrame) -> str:
    out = io.StringIO()
    df.explain(show_all=True, file=out)
    return out.getvalue().split("== Optimized Logical Plan ==")[1].split("\n== ")[0]


def _scan_filters(df: DataFrame) -> list[str]:
    """The pushdowns each scan of ``df``'s optimized plan reads with, one line per scan."""
    pushdowns = []
    for scan in _optimized_plan(df).split("Num Scan Tasks")[1:]:
        found = re.search(r"Pushdowns: \{(.*?)\}\n[|\s]*Schema:", scan, re.DOTALL)
        pushdowns.append(re.sub(r"\n[|\s]*", " ", found.group(1)) if found else "")
    return pushdowns


def _restricts(df: DataFrame, column: str) -> bool:
    """Whether some scan of ``df`` reads with a comparison of ``column`` against a literal."""
    return any(f"col({column})" in f and "lit(" in f for f in _scan_filters(df))


def _rows(df: DataFrame) -> list[tuple[object, ...]]:
    return sorted((tuple(row.values()) for row in df.to_pylist()), key=repr)


def _rows_without_inference(build: Callable[[], DataFrame]) -> list[tuple[object, ...]]:
    with planning_config_ctx(disable_join_key_inference=True):
        return _rows(build())


def _duckdb_rows(sql: str) -> list[tuple[object, ...]]:
    connection = duckdb.connect()
    connection.register("subjects", SUBJECTS)
    connection.register("events", EVENTS)
    return sorted(connection.execute(sql).fetchall(), key=repr)


@pytest.mark.parametrize(
    ("how", "sql_join"),
    [("inner", "JOIN"), ("left", "LEFT JOIN"), ("semi", "SEMI JOIN"), ("anti", "ANTI JOIN")],
)
@pytest.mark.parametrize("written", ["below the join", "above the join"])
def test_a_restriction_on_the_preserved_key_reaches_the_other_scan(
    subjects: Reader, events: Reader, how: str, sql_join: str, written: str
) -> None:
    # Arrange
    def query() -> DataFrame:
        if written == "below the join":
            return subjects().where(col("id").is_in(CHOSEN)).join(events(), left_on="id", right_on="sid", how=how)
        return subjects().join(events(), left_on="id", right_on="sid", how=how).where(col("id").is_in(CHOSEN))

    columns = "s.*" if how in ("semi", "anti") else "s.*, e.*"
    expected = _duckdb_rows(
        f"SELECT {columns} FROM subjects s {sql_join} events e ON s.id = e.sid WHERE s.id IN ('p1', 'p2')"
    )

    # Act
    df = query()

    # Assert
    assert _restricts(df, "sid")
    assert _rows(df) == expected
    assert _rows(df) == _rows_without_inference(query)


@pytest.mark.parametrize(
    "restriction",
    [
        col("id") == lit("p2"),
        col("id") >= lit("p3"),
        col("id").between(lit("p1"), lit("p2")),
        col("id").is_in(["p1", "p2", "p3", "p5", "p6", "p7"]),
        col("id").is_in([f"p{i}" for i in range(10_000)]),
    ],
    ids=["equality", "range", "between", "six values", "ten thousand values"],
)
def test_each_restriction_shape_reaches_the_other_scan(
    subjects: Reader, events: Reader, restriction: daft.Expression
) -> None:
    def query() -> DataFrame:
        return subjects().where(restriction).join(events(), left_on="id", right_on="sid", how="left")

    df = query()

    assert _restricts(df, "sid")
    assert _rows(df) == _rows_without_inference(query)


def test_a_restriction_crosses_a_grouping_by_the_key(subjects: Reader, events: Reader) -> None:
    def query() -> DataFrame:
        counts = events().groupby("sid").agg(col("v").count().alias("n"))
        return subjects().where(col("id").is_in(CHOSEN)).join(counts, left_on="id", right_on="sid", how="left")

    df = query()

    assert _restricts(df, "sid")
    assert _rows(df) == [("p1", "a", "p1", 3), ("p2", "b", "p2", 1)]
    assert _rows(df) == _rows_without_inference(query)


def test_a_restriction_crosses_a_window_partitioned_by_the_key(subjects: Reader, events: Reader) -> None:
    def query() -> DataFrame:
        latest = events().with_column(
            "rank", row_number().over(Window().partition_by("sid").order_by("version", desc=True))
        ).where(col("rank") == 1)
        return subjects().where(col("id").is_in(CHOSEN)).join(latest, left_on="id", right_on="sid", how="left")

    df = query()

    assert _restricts(df, "sid")
    assert _rows(df) == _rows_without_inference(query)


def test_a_restriction_never_crosses_a_window_partitioned_by_another_key(
    subjects: Reader, events: Reader
) -> None:
    # Arrange: the latest version of each record; record 11's latest version belongs to p3, not p1.
    def query() -> DataFrame:
        latest = events().with_column(
            "rank", row_number().over(Window().partition_by("record").order_by("version", desc=True))
        ).where(col("rank") == 1)
        return subjects().where(col("id").is_in(CHOSEN)).join(latest, left_on="id", right_on="sid", how="left")

    # Act
    df = query()

    # Assert
    assert not _restricts(df, "sid")
    assert [(row[0], row[2]) for row in _rows(df)] == [("p1", 10), ("p2", 12)]
    assert _rows(df) == _rows_without_inference(query)


@pytest.mark.parametrize("how", ["left", "outer"])
def test_a_restriction_on_the_null_supplying_side_stays_there(
    subjects: Reader, events: Reader, how: str
) -> None:
    def query() -> DataFrame:
        return subjects().join(events().where(col("sid").is_in(CHOSEN)), left_on="id", right_on="sid", how=how)

    df = query()

    assert not _restricts(df, "id")
    assert ("p3", "c", None, None, None, None) in _rows(df)
    assert _rows(df) == _rows_without_inference(query)


def test_a_full_outer_join_derives_in_neither_direction(subjects: Reader, events: Reader) -> None:
    def query() -> DataFrame:
        return subjects().where(col("id").is_in(CHOSEN)).join(events(), left_on="id", right_on="sid", how="outer")

    df = query()

    assert not _restricts(df, "sid")
    assert _rows(df) == _rows_without_inference(query)


def test_a_restriction_on_the_right_side_of_a_semi_join_reaches_the_left_scan(
    subjects: Reader, events: Reader
) -> None:
    def query() -> DataFrame:
        return subjects().join(events().where(col("sid") == lit("p2")), left_on="id", right_on="sid", how="semi")

    df = query()

    assert _restricts(df, "id")
    assert _rows(df) == [("p2", "b")]


@pytest.mark.parametrize(
    ("left_keys", "right_keys"),
    [
        (pa.array([1.0, 2.0, -0.0]), pa.array([1.0, 0.0, -0.0])),
        (pa.array([1, 2, 3], pa.int32()), pa.array([1, 2, 3], pa.int64())),
    ],
    ids=["float keys", "keys of different dtypes"],
)
def test_keys_a_filter_may_compare_differently_derive_nothing(
    tmp_path: Path, left_keys: pa.Array, right_keys: pa.Array
) -> None:
    left = _reader(_parquet(tmp_path / "left", pa.table({"a": left_keys})))
    right = _reader(_parquet(tmp_path / "right", pa.table({"b": right_keys, "v": [1, 2, 3]})))

    def query() -> DataFrame:
        return left().where(col("a") <= lit(1)).join(right(), left_on="a", right_on="b")

    df = query()

    assert not _restricts(df, "b")
    assert _rows(df) == _rows_without_inference(query)


def test_an_expression_key_derives_nothing(subjects: Reader, events: Reader) -> None:
    def query() -> DataFrame:
        return subjects().where(col("id").is_in(CHOSEN)).join(
            events(), left_on=upper(col("id")), right_on=upper(col("sid"))
        )

    df = query()

    assert not _restricts(df, "sid")
    assert _rows(df) == _rows_without_inference(query)


def test_a_restriction_follows_a_chain_of_joins(tmp_path: Path, subjects: Reader, events: Reader) -> None:
    details = _reader(
        _parquet(tmp_path / "details", pa.table({"subject": ["p1", "p2", "p3"], "code": ["x", "y", "z"]}))
    )

    def query() -> DataFrame:
        event_details = events().join(details(), left_on="sid", right_on="subject")
        return subjects().where(col("id") == lit("p2")).join(event_details, left_on="id", right_on="sid", how="left")

    df = query()

    assert _restricts(df, "sid")
    assert _restricts(df, "subject")
    assert _rows(df) == _rows_without_inference(query)


def test_a_self_join_restricts_both_reads(subjects: Reader) -> None:
    def query() -> DataFrame:
        return subjects().where(col("id") == lit("p1")).join(subjects(), on="id", how="inner")

    df = query()

    assert all("lit(" in f for f in _scan_filters(df))
    assert _rows(df) == [("p1", "a", "a")]


def test_a_restriction_prunes_a_partitioned_table(tmp_path: Path, subjects: Reader) -> None:
    # Arrange
    root = tmp_path / "events_by_subject"
    pq.write_to_dataset(EVENTS.filter(pa.compute.is_valid(EVENTS["sid"])), root, partition_cols=["sid"])

    def partitioned() -> DataFrame:
        return daft.read_parquet(f"{root}/**/*.parquet", hive_partitioning=True)

    def query() -> DataFrame:
        return subjects().where(col("id").is_in(CHOSEN)).join(partitioned(), left_on="id", right_on="sid", how="left")

    # Act
    df = query()

    # Assert
    assert _restricts(df, "sid")
    assert _rows(df) == _rows_without_inference(query)


def test_a_scan_that_may_not_apply_its_filters_exactly_gets_nothing_derived(
    tmp_path: Path, subjects: Reader
) -> None:
    text = tmp_path / "keys.txt"
    text.write_text("p1\np2\np3\n")

    def keys() -> DataFrame:
        return daft.read_text(str(text))

    def query() -> DataFrame:
        return subjects().where(col("id").is_in(CHOSEN)).join(keys(), left_on="id", right_on="text")

    df = query()

    assert not _restricts(df, "text")
    assert _rows(df) == _rows_without_inference(query)


def test_sql_joins_derive_the_same_restrictions(subjects: Reader, events: Reader) -> None:
    sql = "SELECT s.id, e.v FROM subjects s LEFT JOIN events e ON s.id = e.sid WHERE s.id IN ('p1', 'p2')"

    def query() -> DataFrame:
        return daft.sql(sql, subjects=subjects(), events=events())

    df = query()

    assert _restricts(df, "sid")
    assert _rows(df) == _duckdb_rows(sql)


def test_an_iceberg_table_is_read_only_for_the_chosen_keys(tmp_path: Path) -> None:
    # Arrange
    from pyiceberg.catalog.sql import SqlCatalog

    catalog = SqlCatalog(
        "local", uri=f"sqlite:///{tmp_path / 'catalog.db'}", warehouse=f"file://{tmp_path / 'warehouse'}"
    )
    catalog.create_namespace("hdl")
    for name, table in (("subjects", SUBJECTS), ("events", EVENTS)):
        catalog.create_table(f"hdl.{name}", schema=table.schema).append(table)

    def query() -> DataFrame:
        chosen = daft.read_iceberg(catalog.load_table("hdl.subjects")).where(col("id").is_in(CHOSEN))
        return chosen.join(daft.read_iceberg(catalog.load_table("hdl.events")), left_on="id", right_on="sid", how="left")

    # Act
    df = query()

    # Assert
    assert _restricts(df, "sid")
    assert _rows(df) == _rows_without_inference(query)


def test_inference_can_be_turned_off(subjects: Reader, events: Reader) -> None:
    with planning_config_ctx(disable_join_key_inference=True):
        df = subjects().where(col("id").is_in(CHOSEN)).join(events(), left_on="id", right_on="sid", how="left")

        assert not _restricts(df, "sid")
