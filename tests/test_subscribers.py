from __future__ import annotations

import signal
import threading
import time
from collections import defaultdict
from typing import Any

import pytest

import daft
from daft.daft import PyQueryMetadata, PyQueryResult, QueryEndState
from daft.subscribers import Subscriber
from daft.subscribers.events import (
    Event,
    ExecutionFinished,
    ExecutionStarted,
    OperatorFinished,
    OperatorStarted,
    OptimizationCompleted,
    QueryFinished,
    QueryStarted,
    Stats,
)
from tests.conftest import get_tests_daft_runner_name


class MockSubscriber(Subscriber):
    query_metadata: dict[str, PyQueryMetadata]
    query_optimized_plan: dict[str, str]
    query_physical_plan: dict[str, str]
    query_node_stats: defaultdict[str, defaultdict[int, dict[str, Any]]]
    end_states: dict[str, QueryEndState]
    end_messages: dict[str, str]
    results: dict[str, PyQueryResult]
    query_ids: list[str]

    def __init__(self):
        self.query_metadata = {}
        self.query_optimized_plan = {}
        self.query_physical_plan = {}
        self.query_node_stats = defaultdict(lambda: defaultdict(dict))
        self.end_states = {}
        self.end_messages = {}
        self.results = {}
        self.query_ids = []

    def on_query_started(self, event: QueryStarted) -> None:
        self.query_ids.append(event.query_id)
        self.query_metadata[event.query_id] = event.metadata

    def on_query_finished(self, event: QueryFinished) -> None:
        self.end_states[event.query_id] = event.result.end_state
        self.end_messages[event.query_id] = event.result.error_message
        self.results[event.query_id] = event.result

    def on_optimization_completed(self, event: OptimizationCompleted) -> None:
        self.query_optimized_plan[event.query_id] = event.optimized_plan

    def on_execution_started(self, event: ExecutionStarted) -> None:
        self.query_physical_plan[event.query_id] = event.physical_plan

    def on_operator_start(self, event: OperatorStarted) -> None:
        pass

    def on_stats(self, event: Stats) -> None:
        for node_id, stats in event.stats.items():
            for stat_name, (_, stat_value) in stats.items():
                self.query_node_stats[event.query_id][node_id][stat_name] = stat_value

    def on_operator_end(self, event: OperatorFinished) -> None:
        pass

    def on_execution_finished(self, event: ExecutionFinished) -> None:
        pass


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_capture_states(monkeypatch):
    subscriber = MockSubscriber()
    daft.attach_subscriber("mock", subscriber)

    def inject_keyboard_interrupt():
        threading.Timer(2.0, lambda: signal.raise_signal(signal.SIGINT)).start()

    @daft.func.batch(return_dtype=daft.DataType.string())
    def failing_udf(_s: daft.Series):
        raise ValueError("This UDF will fail forever")

    @daft.func.batch(return_dtype=daft.DataType.int64())
    def success_udf(s: daft.Series):
        return s

    @daft.func.batch(return_dtype=daft.DataType.int64())
    def long_running_udf(s: daft.Series):
        time.sleep(10)
        return s

    # 1. Finished state

    df = daft.from_pydict({"x": [1, 2, 3]})
    df = df.with_column("y", success_udf(df["x"]))

    df.collect()

    # Get keys from the map
    query_id = subscriber.query_ids[-1]
    assert subscriber.end_states[query_id] == QueryEndState.Finished
    assert subscriber.end_messages[query_id] == "Query finished"
    # contains the Value Error message

    # 2. Failed state

    df = daft.from_pydict({"x": ["1", "2", "3"]})
    df = df.with_column("y", failing_udf(df["x"]))

    with pytest.raises(ValueError):
        df.collect()

    query_id = subscriber.query_ids[-1]
    assert subscriber.end_states[query_id] == QueryEndState.Failed
    # contains the Value Error message
    assert "This UDF will fail forever" in subscriber.end_messages[query_id]

    # 3. Canceled State

    inject_keyboard_interrupt()

    df = daft.from_pydict({"x": ["1", "2", "3"]})
    df = df.with_column("y", long_running_udf(df["x"]))

    with pytest.raises(KeyboardInterrupt):
        df.collect()

    query_id = subscriber.query_ids[-1]
    assert subscriber.end_states[query_id] == QueryEndState.Canceled
    # contains the Value Error message
    assert "Query canceled by the user" in subscriber.end_messages[query_id]


def test_show_fires_query_end():
    """Regression test for DF-1664.

    .show() abandons the run_iter generator early, which must still fire
    on_query_end so the dashboard transitions out of Finalizing.
    """
    subscriber = MockSubscriber()

    with daft.with_subscriber("mock", subscriber):
        # Sanity check: .collect() on a transformed DataFrame fires on_query_end.
        df = daft.from_pydict({"x": list(range(100))})
        df = df.with_column("y", daft.col("x") + 1)
        df.collect()

        query_id = subscriber.query_ids[-1]
        assert query_id in subscriber.end_states, "collect() should fire on_query_end"
        assert subscriber.end_states[query_id] == QueryEndState.Finished

        # The actual bug: .show() on a transformed DataFrame does NOT fire on_query_end,
        # because _construct_show_preview breaks out of the run_iter generator early.
        df = daft.from_pydict({"x": list(range(100))})
        df = df.with_column("y", daft.col("x") + 1)
        df.show()

        query_id = subscriber.query_ids[-1]
        assert query_id in subscriber.end_states, (
            "on_query_end was not called — query would be stuck in Finalizing on the dashboard"
        )
        assert subscriber.end_states[query_id] == QueryEndState.Finished


def test_subscriber_template():
    subscriber = MockSubscriber()

    with daft.with_subscriber("mock", subscriber):
        df = daft.from_pydict({"x": [1, 2, 3]})
        df = df.with_column("y", df["x"] + 1)
        df = df.limit(5)

        output_schema = df.schema()
        unoptimized_plan_json = df._builder.repr_json()
        df = df.collect()

        query_id = subscriber.query_ids[-1]
        assert subscriber.query_metadata[query_id].unoptimized_plan == unoptimized_plan_json
        assert subscriber.query_metadata[query_id].output_schema == output_schema._schema
        # Optimized plan should be different from unoptimized plan
        assert subscriber.query_optimized_plan[query_id] != unoptimized_plan_json

        # Per-row execution stats are only wired through the native runner today.
        if get_tests_daft_runner_name() == "native":
            for _, stats in subscriber.query_node_stats[query_id].items():
                for stat_name, stat_value in stats.items():
                    if stat_name == "rows_in" or stat_name == "rows_out":
                        assert stat_value == 3

    # Verify detach
    df = daft.from_pydict({"x": [1, 2, 3]})
    df = df.with_column("y", df["x"] + 1)
    df.collect()
    # Should only have the previous query
    assert len(subscriber.query_metadata) == 1


def test_an_in_process_query_reports_what_it_returned_and_no_queue():
    subscriber = MockSubscriber()
    df = daft.from_pydict({"x": list(range(10))}).where(daft.col("x") < 4)

    with daft.with_subscriber("mock", subscriber):
        rows = len(df.collect())

    query_id = subscriber.query_ids[-1]
    result = subscriber.results[query_id]
    assert subscriber.query_metadata[query_id].queued_seconds is None
    assert result.rows_returned == rows == 4
    assert result.bytes_returned is not None
    assert result.bytes_returned > 0


@daft.udf(return_dtype=daft.DataType.int64())
def _udf_raising_key_error(v: daft.Series) -> daft.Series:
    raise KeyError("no such row")


@pytest.mark.parametrize(
    ("frame", "error_class", "remote_types"),
    [
        pytest.param(
            lambda: daft.from_pydict({"k": [1, 1], "v": [1, 2]}).groupby("k").agg(daft.col("v").single_value()),
            "DaftCardinalityError",
            [],
            id="an engine failure by its class",
        ),
        pytest.param(
            lambda: daft.from_pydict({"v": [1]}).select(_udf_raising_key_error(daft.col("v"))),
            "Python",
            ["KeyError", "LookupError"],
            id="a function's failure by the type it raised",
        ),
    ],
)
def test_a_failed_query_ends_naming_its_kind(frame, error_class, remote_types):
    subscriber = MockSubscriber()

    with daft.with_subscriber("mock", subscriber):
        with pytest.raises(Exception):
            frame().collect()

    result = subscriber.results[subscriber.query_ids[-1]]
    assert result.end_state == QueryEndState.Failed
    assert result.error_class == error_class
    assert result.remote_types[:2] == remote_types


def test_execution_events_inherit_from_event_base():
    assert isinstance(OperatorStarted(query_id="q", node_id=1, name="scan"), Event)
    assert isinstance(Stats(query_id="q", stats={}), Event)
    assert isinstance(OperatorFinished(query_id="q", node_id=1, name="scan"), Event)


@pytest.mark.skipif(
    get_tests_daft_runner_name() == "ray",
    reason="bytes.read is not yet emitted to Python subscribers on Ray Flotilla",
)
def test_csv_scan_reports_bytes_read(tmp_path):
    subscriber = MockSubscriber()

    with daft.with_subscriber("mock", subscriber):
        csv_path = tmp_path / "input.csv"
        csv_path.write_text("a,b\n1,2\n3,4\n5,6\n")

        daft.read_csv(str(csv_path)).collect()

        query_id = subscriber.query_ids[-1]
        all_node_stats = subscriber.query_node_stats[query_id]
        bytes_read_values = [
            value for stats in all_node_stats.values() for name, value in stats.items() if name == "bytes.read"
        ]
        assert bytes_read_values, "Expected at least one source node to report bytes.read"
        assert any(value > 0 for value in bytes_read_values)
