"""How JSON writes and reads the types it has no literal for.

A binary value is written as base64 text and a duration as ISO 8601 text in
seconds, at any depth, and each reads back as it was written when read with its
type. A column JSON cannot hold is refused before any row is computed, naming
it; a type JSON is never read as is an error, never a crash.
"""

from __future__ import annotations

import base64
import datetime
import json
from pathlib import Path

import pyarrow as pa
import pytest

import daft

_DURATIONS = [datetime.timedelta(seconds=3), datetime.timedelta(milliseconds=1500), -datetime.timedelta(seconds=3)]


def _written_lines(frame: daft.DataFrame, directory: Path) -> list[dict[str, object]]:
    frame.write_json(str(directory))
    return [json.loads(line) for path in sorted(directory.glob("*.json")) for line in path.read_text().splitlines()]


@pytest.mark.parametrize(
    "binary",
    [pa.binary(), pa.large_binary()],
    ids=["binary", "large binary"],
)
def test_bytes_are_written_as_base64(tmp_path: Path, binary: pa.DataType) -> None:
    values = [b"\x00\xff", b"", None]
    frame = daft.from_arrow(pa.table({"b": pa.array(values, binary)}))

    lines = _written_lines(frame, tmp_path)

    assert [line["b"] for line in lines] == ["AP8=", "", None]


@pytest.mark.parametrize(
    ("unit", "texts"),
    [
        ("s", ["PT3S", "PT1S", "-PT3S"]),
        ("ms", ["PT3S", "PT1.5S", "-PT3S"]),
        ("us", ["PT3S", "PT1.5S", "-PT3S"]),
        ("ns", ["PT3S", "PT1.5S", "-PT3S"]),
    ],
)
def test_durations_are_written_as_iso8601_seconds(tmp_path: Path, unit: str, texts: list[str]) -> None:
    frame = daft.from_arrow(pa.table({"d": pa.array(_DURATIONS, pa.duration(unit))}))

    lines = _written_lines(frame, tmp_path)

    assert [line["d"] for line in lines] == texts


def test_bytes_and_durations_are_written_at_any_depth(tmp_path: Path) -> None:
    nested = pa.struct([("b", pa.binary()), ("d", pa.list_(pa.duration("ms")))])
    frame = daft.from_arrow(pa.table({"n": pa.array([{"b": b"\x01", "d": [datetime.timedelta(seconds=2)]}], nested)}))

    lines = _written_lines(frame, tmp_path)

    assert lines == [{"n": {"b": "AQ==", "d": ["PT2S"]}}]


@pytest.mark.parametrize(
    ("column", "values"),
    [
        (pa.binary(), [b"\x00\xff", b"", None]),
        (pa.binary(3), [b"abc", None]),
        (pa.duration("s"), [datetime.timedelta(seconds=3), -datetime.timedelta(seconds=3), None]),
        (pa.duration("ms"), [datetime.timedelta(milliseconds=1500), datetime.timedelta(0), None]),
        (pa.duration("ns"), [datetime.timedelta(microseconds=7), None]),
        (pa.struct([("b", pa.binary())]), [{"b": b"\x02"}, None]),
    ],
    ids=["binary", "fixed binary", "seconds", "milliseconds", "nanoseconds", "nested binary"],
)
def test_what_is_written_reads_back_as_itself(tmp_path: Path, column: pa.DataType, values: list[object]) -> None:
    written = pa.table({"v": pa.array(values, column)})
    daft.from_arrow(written).write_json(str(tmp_path))

    read = daft.read_json(str(tmp_path), schema={"v": daft.DataType.from_arrow_type(column)}).to_arrow()

    assert read.column("v").to_pylist() == written.column("v").to_pylist()


@pytest.mark.parametrize(
    ("column", "text"),
    [
        (pa.binary(), '"not base64!"'),
        (pa.binary(), "5"),
        (pa.binary(2), '"AQID"'),
        (pa.duration("s"), '"PT1.5S"'),
        (pa.duration("s"), '"P1Y"'),
        (pa.duration("s"), '"three seconds"'),
    ],
    ids=[
        "bytes that are not base64",
        "bytes that are a number",
        "bytes of another length",
        "a duration finer than its unit",
        "a duration of years",
        "a duration that is not one",
    ],
)
def test_a_value_that_does_not_decode_reads_as_null(tmp_path: Path, column: pa.DataType, text: str) -> None:
    path = tmp_path / "rows.json"
    path.write_text(f'{{"v": {text}}}\n')

    read = daft.read_json(str(path), schema={"v": daft.DataType.from_arrow_type(column)}).to_pydict()

    assert read == {"v": [None]}


def test_a_duration_reads_from_a_count_of_its_unit_as_before(tmp_path: Path) -> None:
    path = tmp_path / "rows.json"
    path.write_text('{"v": 3500}\n{"v": "P1DT1H"}\n')

    read = daft.read_json(str(path), schema={"v": daft.DataType.duration("ms")}).to_pydict()

    assert read == {"v": [datetime.timedelta(milliseconds=3500), datetime.timedelta(days=1, hours=1)]}


def test_a_column_json_cannot_hold_is_refused_before_any_row_is_computed(tmp_path: Path) -> None:
    computed = [0]

    @daft.func(return_dtype=daft.DataType.python())
    def held(value: int) -> object:
        computed[0] += 1
        return value

    frame = daft.range(1_000).select(held(daft.col("id")).alias("obj"))

    with pytest.raises(Exception, match='JSON cannot hold the column "obj" of type Python'):
        frame.write_json(str(tmp_path))

    assert computed == [0]


def test_a_type_json_is_never_read_as_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "rows.json"
    path.write_text('{"v": 1}\n')

    with pytest.raises(Exception, match='JSON cannot be read as .*the type of the column "v"'):
        daft.read_json(str(path), schema={"v": daft.DataType.decimal128(10, 2)}).collect()


def test_base64_is_the_standard_alphabet_padded() -> None:
    # The form the writer uses, decoded by the standard library.
    assert base64.b64decode("AP8=") == b"\x00\xff"
