from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pyarrow as pa
import pytest

import daft
from daft import DataType, col
from tests.conftest import get_tests_daft_runner_name

Writer = Callable[[daft.DataFrame, str], daft.DataFrame]
Reader = Callable[[str], daft.DataFrame]

FORMATS: dict[str, tuple[Writer, Reader]] = {
    "parquet": (lambda df, path: df.write_parquet(path), daft.read_parquet),
    "csv": (lambda df, path: df.write_csv(path), daft.read_csv),
    "json": (lambda df, path: df.write_json(path), daft.read_json),
}


@pytest.fixture(params=[True, False], ids=["native parquet", "pyarrow parquet"])
def parquet_writer(request: pytest.FixtureRequest) -> Iterator[None]:
    with daft.execution_config_ctx(native_parquet_writer=request.param):
        yield


def _rows(count: int) -> daft.DataFrame:
    return daft.from_pydict({"k": [i % 3 for i in range(count)], "v": list(range(count))})


def _written_rows(result: daft.DataFrame, read: Reader) -> list[tuple[int, int]]:
    """Pair each written file's reported row count with the rows read back from it."""
    files = result.to_pydict()
    return [(reported, read(path).count_rows()) for path, reported in zip(files["path"], files["num_rows"])]


# --- a write reports the rows each file holds ---


@pytest.mark.parametrize("fmt", list(FORMATS))
def test_a_write_reports_the_rows_of_each_file(tmp_path: Path, fmt: str, parquet_writer: None) -> None:
    write, read = FORMATS[fmt]

    result = write(_rows(10).into_partitions(3), str(tmp_path))

    assert result.column_names[:2] == ["path", "num_rows"]
    assert result.schema()["num_rows"].dtype == DataType.int64()
    counted = _written_rows(result, read)
    assert [reported for reported, _ in counted] == [read_back for _, read_back in counted]
    assert sum(reported for reported, _ in counted) == 10


@pytest.mark.parametrize("fmt", list(FORMATS))
def test_a_partitioned_write_reports_the_rows_of_each_file(tmp_path: Path, fmt: str, parquet_writer: None) -> None:
    write, read = FORMATS[fmt]
    partitioned: Writer = {
        "parquet": lambda df, path: df.write_parquet(path, partition_cols=["k"]),
        "csv": lambda df, path: df.write_csv(path, partition_cols=["k"]),
        "json": lambda df, path: df.write_json(path, partition_cols=["k"]),
    }[fmt]

    result = partitioned(_rows(10), str(tmp_path))

    assert result.column_names == ["path", "num_rows", "k"]
    by_key = {k: n for k, n in zip(result.to_pydict()["k"], result.to_pydict()["num_rows"])}
    assert by_key == {0: 4, 1: 3, 2: 3}


def test_a_write_split_across_files_reports_each(tmp_path: Path, parquet_writer: None) -> None:
    with daft.execution_config_ctx(parquet_target_filesize=1024):
        result = _rows(20_000).write_parquet(str(tmp_path))

    counted = _written_rows(result, daft.read_parquet)
    assert len(counted) > 1
    assert all(reported == read_back for reported, read_back in counted)
    assert sum(reported for reported, _ in counted) == 20_000


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
def test_an_empty_write_reports_no_rows(tmp_path: Path, fmt: str) -> None:
    write, _ = FORMATS[fmt]

    result = write(_rows(3).where(col("v") < 0), str(tmp_path))

    assert result.to_pydict()["num_rows"] == [0]


@pytest.mark.parametrize("name", ["path", "num_rows"])
def test_a_write_refuses_a_partition_column_named_like_its_report(tmp_path: Path, name: str) -> None:
    df = daft.from_pydict({name: [1, 2], "v": [3, 4]})

    with pytest.raises(ValueError, match="cannot partition a write by a column named"):
        df.write_parquet(str(tmp_path), partition_cols=[name])


# --- a write that cannot succeed fails before writing, and leaves nothing ---


def _files_in(directory: Path) -> list[Path]:
    return [path for path in directory.rglob("*") if path.is_file()]


@pytest.mark.parametrize(
    "column",
    [
        pytest.param(pa.array([[1, 2], [3]]), id="list"),
        pytest.param(pa.array([{"a": 1}, {"a": 2}]), id="struct"),
        pytest.param(pa.array([[("k", 1)], [("k", 2)]], type=pa.map_(pa.string(), pa.int64())), id="map"),
    ],
)
def test_a_csv_write_of_a_column_csv_cannot_hold_is_refused_before_writing(tmp_path: Path, column: pa.Array) -> None:
    df = daft.from_arrow(pa.table({"id": [1, 2], "nested": column}))

    with pytest.raises(daft.exceptions.DaftCoreException, match='CSV cannot hold the column "nested"'):
        df.write_csv(str(tmp_path / "out"))

    assert _files_in(tmp_path) == []


def test_a_partitioned_csv_write_checks_only_the_columns_it_writes(tmp_path: Path) -> None:
    df = daft.from_pydict({"k": [1, 2], "v": ["a", "b"]})

    result = df.write_csv(str(tmp_path), partition_cols=["k"])

    assert sum(result.to_pydict()["num_rows"]) == 2


_ROWS_BEFORE_FAILURE = 200_000


@daft.func
def _fails_on_the_last_row(v: int) -> int:
    if v == _ROWS_BEFORE_FAILURE - 1:
        raise RuntimeError("the source failed part way")
    return v


@pytest.mark.parametrize("fmt", list(FORMATS))
def test_a_write_that_fails_part_way_leaves_no_file(tmp_path: Path, fmt: str, parquet_writer: None) -> None:
    write, _ = FORMATS[fmt]
    # Small morsels reach the writer, which opens its file, well before the failing row.
    df = daft.from_pydict({"v": list(range(_ROWS_BEFORE_FAILURE))}).with_column("v", _fails_on_the_last_row(col("v")))

    with daft.execution_config_ctx(default_morsel_size=1_000), pytest.raises(Exception, match="failed part way"):
        write(df, str(tmp_path / "out"))

    assert _files_in(tmp_path) == []


# --- csv can be written as one file ---

_NATIVE_ONLY = pytest.mark.skipif(
    get_tests_daft_runner_name() != "native",
    reason="single_file is only supported on the native runner",
)


@_NATIVE_ONLY
def test_a_csv_write_can_be_one_file(tmp_path: Path) -> None:
    target = tmp_path / "panel.csv"

    result = _rows(10).into_partitions(3).write_csv(str(target), single_file=True)

    assert target.is_file()
    assert result.to_pydict() == {"path": [str(target)], "num_rows": [10]}
    assert daft.read_csv(str(target)).sort("v").to_pydict() == _rows(10).sort("v").to_pydict()


@_NATIVE_ONLY
def test_a_one_file_csv_write_replaces_what_is_there_when_overwriting(tmp_path: Path) -> None:
    target = tmp_path / "panel.csv"
    _rows(10).write_csv(str(target), single_file=True)

    _rows(4).write_csv(str(target), write_mode="overwrite", single_file=True)

    assert daft.read_csv(str(target)).count_rows() == 4


@_NATIVE_ONLY
def test_an_empty_one_file_csv_write_still_writes_the_file(tmp_path: Path) -> None:
    target = tmp_path / "panel.csv"

    result = _rows(3).where(col("v") < 0).write_csv(str(target), single_file=True)

    assert target.is_file()
    assert result.to_pydict()["num_rows"] == [0]


@pytest.mark.parametrize(
    ("options", "refusal"),
    [
        pytest.param({"partition_cols": ["k"]}, "partition_cols", id="partitioned"),
        pytest.param(
            {"partition_cols": ["k"], "write_mode": "overwrite-partitions"},
            "overwrite-partitions",
            id="overwriting partitions",
        ),
    ],
)
def test_a_one_file_csv_write_refuses_what_one_file_cannot_honor(
    tmp_path: Path, options: dict[str, str | list[str]], refusal: str
) -> None:
    with pytest.raises(ValueError, match=refusal):
        _rows(3).write_csv(str(tmp_path / "panel.csv"), single_file=True, **options)
