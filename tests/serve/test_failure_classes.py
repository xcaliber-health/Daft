"""A served query fails with the class it fails with in-process.

An engine failure arrives as its engine class. A Python exception arrives by
the names of its type, never as a class the client imports: as `DaftServeError`
of class `Host` when the request host raised it while planning, of class
`Python` when it was raised while the query executed, and as `UDFException` when
a user-defined function raised it. The server's own failures arrive as
`DaftServeError` of their class, saying whether a retry may succeed. Every
served failure carries its query id, and a failed query's end event names its
kind, never the exception's named fields.
"""

from __future__ import annotations

import threading
import time
from collections.abc import AsyncIterator, Callable, Collection, Iterator
from typing import TYPE_CHECKING

import pytest

import daft
from daft import col
from daft.daft import QueryEndState
from daft.errors import UDFException
from daft.exceptions import DaftCardinalityError, DaftComputeError, DaftCoreException, DaftServeError
from daft.io.source import DataSource, DataSourceTask
from daft.recordbatch import RecordBatch
from daft.runners.remote_runner import RemoteRunner
from daft.schema import Schema
from daft.serve import ServedRequest, ServeSettings, TenantSpec, start_server
from daft.subscribers import Subscriber
from daft.subscribers.events import Event, QueryFinished

if TYPE_CHECKING:
    import pathlib

    from daft.daft import DaftServeClient, DaftServeServer
    from daft.io.pushdowns import Pushdowns


class Refused(PermissionError):
    """A host's refusal, carrying fields for its client."""

    def __init__(self, message: str, served_fields: object = None) -> None:
        super().__init__(message)
        if served_fields is not None:
            self.served_fields = served_fields


class Boom(RuntimeError):
    """Raised while a query executes."""


class FailingTask(DataSourceTask):
    @property
    def schema(self) -> Schema:
        return Schema.from_pydict({"v": daft.DataType.int64()})

    async def read(self) -> AsyncIterator[RecordBatch]:
        raise Boom("the source could not be read")
        yield RecordBatch.from_pydict({"v": [1]})  # pragma: no cover


class FailingSource(DataSource):
    @property
    def name(self) -> str:
        return "failing"

    @property
    def schema(self) -> Schema:
        return Schema.from_pydict({"v": daft.DataType.int64()})

    async def get_tasks(self, pushdowns: Pushdowns) -> AsyncIterator[DataSourceTask]:
        yield FailingTask()


# A legacy UDF is attached to the session under its own name, so it must not
# shadow a SQL function that tests sharing this process call.
@daft.udf(return_dtype=daft.DataType.int64())
def failing_udf(v: daft.Series) -> daft.Series:
    raise KeyError("no such row")


@daft.func.batch(return_dtype=daft.DataType.int64())
def failing_batch(v: daft.Series) -> daft.Series:
    raise KeyError("no such batch")


@daft.func(return_dtype=daft.DataType.int64())
def failing_rows(v: int) -> int:
    raise KeyError("no such row")


def _two_values_in_one_group() -> daft.DataFrame:
    return daft.from_pydict({"k": [1, 1], "v": [1, 2]}).groupby("k").agg(col("v").single_value())


def _missing_column() -> daft.DataFrame:
    return daft.from_pydict({"v": [1]}).select(col("nope"))


def _a_udf_failure() -> daft.DataFrame:
    return daft.from_pydict({"v": [1]}).select(failing_udf(col("v")))


def _a_batch_failure() -> daft.DataFrame:
    return daft.from_pydict({"v": [1]}).select(failing_batch(col("v")))


def _row_failures() -> daft.DataFrame:
    return daft.from_pydict({"v": [1]}).select(failing_rows(col("v")))


def _a_source_failure() -> daft.DataFrame:
    return FailingSource().read()


class Host:
    """Plans each kind of request into a frame, or refuses it."""

    @property
    def kinds(self) -> Collection[str]:
        return ("cardinality", "missing", "rows", "udf", "batch", "source", "refuse", "fields")

    def plan(self, request: ServedRequest) -> daft.DataFrame:
        if request.kind == "refuse":
            raise Refused("the tenant may not act for this request", {"status": "403", "detail": "acme"})
        if request.kind == "fields":
            raise Refused("refused", _FIELDS[request.body.decode()])
        frames: dict[str, Callable[[], daft.DataFrame]] = {
            "cardinality": _two_values_in_one_group,
            "missing": _missing_column,
            "rows": _row_failures,
            "udf": _a_udf_failure,
            "batch": _a_batch_failure,
            "source": _a_source_failure,
        }
        return frames[request.kind]()


_FIELDS: dict[str, object] = {
    "too many": {f"f{i}": "x" for i in range(33)},
    "too large": {"detail": "x" * (8 * 1024)},
    "not text": {"status": 403},
    "at the bounds": {f"f{i}": "x" for i in range(32)},
}


class Recorder(Subscriber):
    def __init__(self) -> None:
        self.finished: dict[str, QueryFinished] = {}
        self._lock = threading.Lock()

    def on_event(self, event: Event) -> None:
        if isinstance(event, QueryFinished):
            with self._lock:
                self.finished[event.query_id] = event

    def ended(self, query_id: str) -> QueryFinished:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            with self._lock:
                if query_id in self.finished:
                    return self.finished[query_id]
            time.sleep(0.05)
        raise AssertionError(f"query {query_id} never reported its end")


@pytest.fixture(scope="module")
def server() -> Iterator[DaftServeServer]:
    serving = start_server(
        ServeSettings(host="127.0.0.1", port=0, tenants=(TenantSpec(name="svc", token="tok"),)),
        request_host=Host(),
    )
    yield serving
    serving.shutdown(drain_timeout_secs=10)


def _client(server: DaftServeServer, token: str | None = "tok") -> DaftServeClient:
    return RemoteRunner(server.address(), token=token).client


def _served(server: DaftServeServer, kind: str, query_id: str, body: bytes = b"") -> BaseException:
    with pytest.raises(BaseException) as raised:
        for _ in _client(server).run_request(kind, body, query_id):
            pass
    return raised.value


def _in_process(frame: Callable[[], daft.DataFrame]) -> BaseException:
    with pytest.raises(BaseException) as raised:
        frame().collect()
    return raised.value


@pytest.mark.parametrize(
    ("kind", "frame", "expected"),
    [
        pytest.param("cardinality", _two_values_in_one_group, DaftCardinalityError, id="raised while executing"),
        pytest.param("missing", _missing_column, DaftCoreException, id="raised by the host while planning"),
        # A row-wise function's failures are gathered per batch into one engine error.
        pytest.param("rows", _row_failures, DaftComputeError, id="a row-wise function's failures"),
    ],
)
def test_an_engine_failure_arrives_as_its_in_process_class(
    server: DaftServeServer, kind: str, frame: Callable[[], daft.DataFrame], expected: type[BaseException]
) -> None:
    in_process = _in_process(frame)

    served = _served(server, kind, f"q-{kind}")

    assert type(served) is type(in_process)
    assert isinstance(served, expected)
    assert served.query_id == f"q-{kind}"  # type: ignore[attr-defined]


def test_a_host_refusal_arrives_by_its_type_with_its_fields(server: DaftServeServer) -> None:
    served = _served(server, "refuse", "q-refuse")

    assert isinstance(served, DaftServeError)
    assert (served.error_class, served.retryable, served.query_id) == ("Host", False, "q-refuse")
    assert (served.remote_type, served.remote_module) == ("Refused", Refused.__module__)
    assert served.remote_types[:3] == ("Refused", "PermissionError", "OSError")
    assert served.fields == {"status": "403", "detail": "acme"}
    assert "acme" not in str(served)


@pytest.mark.parametrize(
    ("bounds", "kept"),
    [
        pytest.param("at the bounds", True, id="32 fields are kept"),
        pytest.param("too many", False, id="33 fields are dropped"),
        pytest.param("too large", False, id="more than 8 KiB is dropped"),
        pytest.param("not text", False, id="a value that is not text is dropped"),
    ],
)
def test_fields_beyond_their_bounds_are_dropped_and_the_failure_still_arrives(
    server: DaftServeServer, bounds: str, kept: bool
) -> None:
    served = _served(server, "fields", f"q-fields-{len(bounds)}", bounds.encode())

    assert isinstance(served, DaftServeError)
    assert served.remote_type == "Refused"
    assert served.fields == (_FIELDS[bounds] if kept else {})


def test_a_udf_failure_arrives_as_udf_exception_naming_the_original(server: DaftServeServer) -> None:
    in_process = _in_process(_a_udf_failure)

    served = _served(server, "udf", "q-udf")

    assert isinstance(in_process, UDFException)
    assert isinstance(served, UDFException)
    assert (served.error_class, served.remote_type, served.query_id) == ("Python", "KeyError", "q-udf")  # type: ignore[attr-defined]
    assert served.remote_types[:2] == ("KeyError", "LookupError")  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("kind", "frame", "raised"),
    [
        pytest.param("source", _a_source_failure, "Boom", id="a source the plan carries"),
        pytest.param("batch", _a_batch_failure, "KeyError", id="a batch function"),
    ],
)
def test_a_python_failure_while_executing_arrives_by_its_type_as_class_python(
    server: DaftServeServer, kind: str, frame: Callable[[], daft.DataFrame], raised: str
) -> None:
    in_process = _in_process(frame)

    served = _served(server, kind, f"q-{kind}")

    assert type(in_process).__name__ == raised
    assert isinstance(served, DaftServeError)
    assert (served.error_class, served.remote_type, served.retryable) == ("Python", raised, False)
    assert served.remote_module == type(in_process).__module__
    assert str(in_process) in str(served)


@pytest.mark.parametrize(
    ("settings", "token", "error_class", "retryable"),
    [
        pytest.param(
            ServeSettings(host="127.0.0.1", port=0, disable_plan_payload=True),
            None,
            "PlanPayloadDisabled",
            False,
            id="plan payloads disabled",
        ),
        pytest.param(
            ServeSettings(host="127.0.0.1", port=0, query_timeout_secs=1), None, "Timeout", True, id="timed out"
        ),
    ],
)
def test_a_server_failure_arrives_by_its_class_saying_whether_to_retry(
    settings: ServeSettings, token: str | None, error_class: str, retryable: bool
) -> None:
    @daft.func(return_dtype=daft.DataType.int64())
    def crawl(v: int) -> int:
        time.sleep(0.5)
        return v

    df = daft.from_pydict({"v": list(range(8))}).select(crawl(col("v")))
    serving = start_server(settings)
    try:
        client = _client(serving, token)
        with pytest.raises(DaftServeError) as raised:
            for _ in client.run_plan(df._builder._builder, {}, "q-server", results_buffer_size=1):
                pass
    finally:
        serving.shutdown(drain_timeout_secs=10)

    assert (raised.value.error_class, raised.value.retryable, raised.value.query_id) == (
        error_class,
        retryable,
        "q-server",
    )


def test_a_caller_refused_at_connection_is_told_so_by_class() -> None:
    serving = start_server(ServeSettings(host="127.0.0.1", port=0, token="right"))
    try:
        with pytest.raises(DaftServeError) as raised:
            _client(serving, "wrong")
    finally:
        serving.shutdown(drain_timeout_secs=10)

    # No query was sent, so the failure names none.
    assert (raised.value.error_class, raised.value.retryable, raised.value.query_id) == (
        "Unauthenticated",
        False,
        None,
    )


def test_a_failed_query_ends_with_its_kind_served_and_in_process_alike(server: DaftServeServer) -> None:
    recorder = Recorder()
    with daft.with_subscriber("failure-kinds", recorder):
        _in_process(_a_udf_failure)
        in_process_id = next(iter(recorder.finished))
        _served(server, "udf", "q-udf-event")
        _served(server, "refuse", "q-refuse-event")
        in_process = recorder.ended(in_process_id).result
        served = recorder.ended("q-udf-event").result
        refused = recorder.ended("q-refuse-event").result

    def kind(result: object) -> tuple[object, ...]:
        return (
            result.end_state,  # type: ignore[attr-defined]
            result.error_class,  # type: ignore[attr-defined]
            result.remote_type,  # type: ignore[attr-defined]
            tuple(result.remote_types),  # type: ignore[attr-defined]
        )

    assert kind(served) == kind(in_process)
    assert kind(served)[:3] == (QueryEndState.Failed, "Python", "KeyError")
    assert (refused.error_class, refused.remote_type) == ("Host", "Refused")
    assert "acme" not in (refused.error_message or "")


@pytest.mark.parametrize(
    ("cancel", "error_class", "retryable"),
    [
        pytest.param(False, "AtCapacity", True, id="no slot within the queue timeout"),
        pytest.param(True, "Cancelled", False, id="cancelled while queued"),
    ],
)
def test_a_queued_query_fails_by_its_class(
    tmp_path: pathlib.Path, cancel: bool, error_class: str, retryable: bool
) -> None:
    @daft.func(return_dtype=daft.DataType.int64())
    def crawl(v: int) -> int:
        time.sleep(0.3)
        return v

    path = str(tmp_path / "hold.parquet")
    daft.from_pydict({"v": list(range(16))}).write_parquet(path, write_mode="overwrite")

    serving = start_server(
        ServeSettings(
            host="127.0.0.1",
            port=0,
            tenants=(TenantSpec(name="svc", token="tok", max_concurrent_queries=1, queue_timeout_secs=2),),
        )
    )
    client = _client(serving)
    # Holds the only slot for several seconds.
    holder = client.run_plan(
        daft.read_parquet(path).select(crawl(col("v")))._builder._builder,
        {},
        "holder",
        results_buffer_size=1,
    )
    raised: dict[str, BaseException] = {}

    def submit() -> None:
        try:
            client.run_plan(daft.from_pydict({"v": [1]})._builder._builder, {}, "q-queued", results_buffer_size=1)
        except BaseException as failure:
            raised["error"] = failure

    thread = threading.Thread(target=submit, daemon=True)
    thread.start()
    try:
        if cancel:
            time.sleep(0.5)
            client.cancel_query("q-queued")
        thread.join(timeout=10)
    finally:
        del holder
        serving.shutdown(drain_timeout_secs=10)

    error = raised["error"]
    assert isinstance(error, DaftServeError)
    assert (error.error_class, error.retryable, error.query_id) == (error_class, retryable, "q-queued")
