"""Every column type must survive the wire round trip exactly."""

from __future__ import annotations

import datetime
import decimal
import math
from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

import daft

from .conftest import collect_via

if TYPE_CHECKING:
    from daft.runners.native_runner import NativeRunner
    from daft.runners.remote_runner import RemoteRunner

TYPE_CASES: dict[str, pa.Array] = {
    "int8": pa.array([1, None, -128], type=pa.int8()),
    "int64": pa.array([1, None, 2**62], type=pa.int64()),
    "uint64": pa.array([1, None, 2**63], type=pa.uint64()),
    "float64": pa.array([1.5, None, float("inf"), float("-inf"), float("nan")], type=pa.float64()),
    "bool": pa.array([True, False, None], type=pa.bool_()),
    "utf8": pa.array(["plain", "", None, "êé—🦆", "line\nbreak"], type=pa.large_string()),
    "binary": pa.array([b"\x00\x01", b"", None], type=pa.large_binary()),
    "date32": pa.array([datetime.date(2020, 1, 1), None], type=pa.date32()),
    "timestamp_us": pa.array(
        [datetime.datetime(2024, 5, 1, 12, 30, 15, 123456), None],
        type=pa.timestamp("us"),
    ),
    "timestamp_tz": pa.array(
        [datetime.datetime(2024, 5, 1, 12, 0, tzinfo=datetime.timezone.utc), None],
        type=pa.timestamp("us", tz="UTC"),
    ),
    "duration": pa.array([datetime.timedelta(seconds=90), None], type=pa.duration("us")),
    "decimal128": pa.array([decimal.Decimal("123.45"), None], type=pa.decimal128(10, 2)),
    "list_int": pa.array([[1, 2], None, []], type=pa.large_list(pa.int64())),
    "fixed_size_list": pa.array([[1, 2], None, [3, 4]], type=pa.list_(pa.int64(), 2)),
    "struct": pa.array(
        [{"a": 1, "b": "x"}, None, {"a": 2, "b": None}],
        type=pa.struct([("a", pa.int64()), ("b", pa.large_string())]),
    ),
    "null_only": pa.array([None, None, None], type=pa.null()),
}


@pytest.mark.parametrize("name", sorted(TYPE_CASES))
def test_type_round_trips_exactly(name: str, remote_runner: RemoteRunner, native_runner: NativeRunner) -> None:
    table = pa.table({"c": TYPE_CASES[name]})
    df = daft.from_arrow(table)

    got = collect_via(remote_runner, df)
    expected = collect_via(native_runner, df)

    assert _canonical(got["c"]) == _canonical(expected["c"])


def test_empty_dataframe_round_trips(remote_runner: RemoteRunner, native_runner: NativeRunner) -> None:
    df = daft.from_arrow(pa.table({"a": pa.array([], type=pa.int64())}))

    got = collect_via(remote_runner, df)
    expected = collect_via(native_runner, df)

    assert got == expected == {"a": []}


def _canonical(values: list[object]) -> list[object]:
    """Replace float NaN with a sentinel so equality comparison is stable."""
    out: list[object] = []
    for value in values:
        if isinstance(value, float) and math.isnan(value):
            out.append("__nan__")
        else:
            out.append(value)
    return out
