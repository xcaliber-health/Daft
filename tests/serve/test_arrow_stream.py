"""A served result reads as an Arrow stream.

A served result offers Arrow's stream interface, as a local frame does: any
reader of it (pyarrow, DuckDB, polars) reads it part by part as it arrives,
schema first, so an empty result still names its columns. Its columns convert
as a local result's do. A failure partway ends the stream with its class as
text, and the result keeps the failure to raise as its class.
"""

from __future__ import annotations

import datetime
import decimal
import time
from collections.abc import AsyncIterator, Collection, Iterator
from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

import daft
from daft import col
from daft.exceptions import DaftServeError
from daft.io.source import DataSource, DataSourceTask
from daft.recordbatch import RecordBatch
from daft.runners.remote_runner import RemoteRunner
from daft.schema import Schema
from daft.serve import ServedRequest, ServeSettings, start_server

if TYPE_CHECKING:
    from daft.daft import DaftServeQueryResult, DaftServeServer
    from daft.io.pushdowns import Pushdowns


class Boom(RuntimeError):
    """Raised by a source after its first batch."""


class FailingTask(DataSourceTask):
    @property
    def schema(self) -> Schema:
        return Schema.from_pydict({"n": daft.DataType.int64()})

    async def read(self) -> AsyncIterator[RecordBatch]:
        yield RecordBatch.from_pydict({"n": [1, 2]})
        raise Boom("the source failed after its first batch")


class FailingSource(DataSource):
    @property
    def name(self) -> str:
        return "failing"

    @property
    def schema(self) -> Schema:
        return Schema.from_pydict({"n": daft.DataType.int64()})

    async def get_tasks(self, pushdowns: Pushdowns) -> AsyncIterator[DataSourceTask]:
        yield FailingTask()


def _typed() -> daft.DataFrame:
    return daft.from_pydict(
        {
            "i": [1, None, 3],
            "f": [1.5, 2.5, None],
            "s": ["a", None, "c"],
            "d": [datetime.date(2026, 10, 13), None, datetime.date(1970, 1, 1)],
            "t": [datetime.datetime(2026, 10, 13, 12), None, datetime.datetime(1970, 1, 1)],
            "m": [decimal.Decimal("1.25"), None, decimal.Decimal("-3.50")],
            "l": [[1, 2], [], None],
            "st": [{"a": 1, "b": "x"}, None, {"a": 3, "b": None}],
            "b": [True, None, False],
        }
    )


def _slow(rows: int) -> daft.DataFrame:
    @daft.func(return_dtype=daft.DataType.int64())
    def crawl(n: int) -> int:
        time.sleep(0.2)
        return n

    return daft.from_pydict({"n": list(range(rows))}).into_batches(1).select(crawl(col("n")).alias("n"))


class Host:
    """Answers each kind of request with its own frame."""

    @property
    def kinds(self) -> Collection[str]:
        return ("rows", "empty", "typed", "slow", "fails", "refuse")

    def plan(self, request: ServedRequest) -> daft.DataFrame:
        if request.kind == "refuse":
            raise PermissionError("the request was refused")
        frames = {
            "rows": lambda: daft.from_pydict({"n": [1, 2, 3]}),
            "empty": lambda: daft.from_pydict({"n": [1, 2, 3]}).where(col("n") > 5),
            "typed": _typed,
            "slow": lambda: _slow(10),
            "fails": lambda: FailingSource().read(),
        }
        return frames[request.kind]()


@pytest.fixture(scope="module")
def server() -> Iterator[DaftServeServer]:
    serving = start_server(ServeSettings(host="127.0.0.1", port=0), request_host=Host())
    yield serving
    serving.shutdown(drain_timeout_secs=10)


def _result(server: DaftServeServer, kind: str, query_id: str) -> DaftServeQueryResult:
    return RemoteRunner(server.address()).client.run_request(kind, b"", query_id, results_buffer_size=1)


@pytest.mark.parametrize(
    ("kind", "rows"),
    [
        pytest.param("rows", 3, id="three rows"),
        pytest.param("empty", 0, id="no rows, still with its schema"),
    ],
)
def test_a_served_result_reads_as_an_arrow_table(server: DaftServeServer, kind: str, rows: int) -> None:
    table = pa.table(_result(server, kind, f"q-{kind}"))

    assert table.num_rows == rows
    assert table.schema == pa.schema([("n", pa.int64())])


def test_its_schema_is_read_before_any_row(server: DaftServeServer) -> None:
    assert pa.schema(_result(server, "empty", "q-schema")) == pa.schema([("n", pa.int64())])


def test_its_columns_convert_as_a_local_results_do(server: DaftServeServer) -> None:
    served = pa.table(_result(server, "typed", "q-typed"))

    assert served.equals(pa.table(_typed()), check_metadata=True)


def test_it_is_read_part_by_part_as_parts_arrive(server: DaftServeServer) -> None:
    reader = pa.RecordBatchReader.from_stream(_result(server, "slow", "q-slow"))

    first = reader.read_next_batch()
    # The server holds one part at a time for a reader, so the query cannot have finished yet.
    running_after_first = server.active_queries()
    rest = list(reader)

    assert running_after_first == 1
    assert len(rest) >= 1
    assert first.num_rows + sum(batch.num_rows for batch in rest) == 10


def test_a_requested_schema_is_honored(server: DaftServeServer) -> None:
    reader = pa.RecordBatchReader.from_stream(_result(server, "rows", "q-cast"), schema=pa.schema([("n", pa.int32())]))

    assert reader.read_all() == pa.table({"n": pa.array([1, 2, 3], type=pa.int32())})


def test_duckdb_reads_it(server: DaftServeServer) -> None:
    duckdb = pytest.importorskip("duckdb")
    relation = duckdb.from_arrow(_result(server, "rows", "q-duckdb"))

    assert relation.aggregate("sum(n)").fetchall() == [(6,)]


def test_polars_reads_it(server: DaftServeServer) -> None:
    pl = pytest.importorskip("polars")

    assert pl.DataFrame(_result(server, "rows", "q-polars"))["n"].to_list() == [1, 2, 3]


def test_a_failure_partway_reaches_the_reader_and_is_kept_as_its_class(server: DaftServeServer) -> None:
    answer = _result(server, "fails", "q-fails")

    with pytest.raises(pa.ArrowException, match=r"\[Python\].*first batch"):
        pa.table(answer)

    failure = answer.failure
    assert isinstance(failure, DaftServeError)
    assert (failure.error_class, failure.remote_type, failure.query_id) == ("Python", "Boom", "q-fails")


def test_a_failure_before_the_schema_is_raised_as_its_class(server: DaftServeServer) -> None:
    answer = _result(server, "refuse", "q-refuse")

    with pytest.raises(DaftServeError) as raised:
        pa.table(answer)

    assert (raised.value.error_class, raised.value.remote_type) == ("Host", "PermissionError")
    assert isinstance(answer.failure, DaftServeError)


def test_a_result_read_whole_has_no_failure(server: DaftServeServer) -> None:
    answer = _result(server, "rows", "q-ok")

    pa.table(answer)

    assert answer.failure is None


def test_the_stream_is_taken_once(server: DaftServeServer) -> None:
    answer = _result(server, "rows", "q-once")
    reader = pa.RecordBatchReader.from_stream(answer)

    with pytest.raises(RuntimeError, match="already being read"):
        pa.RecordBatchReader.from_stream(answer)
    with pytest.raises(RuntimeError, match="being read as an Arrow stream"):
        next(iter(answer))
    assert reader.read_all().num_rows == 3
