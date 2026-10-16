"""A scan of local files reports the bytes it read from them.

Reading a whole file reports its size, and a read that skips part of a file
reports less than reading all of it. WARC reads report the bytes they parse,
after decompression.
"""

from __future__ import annotations

import gzip
import shutil
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import daft
from daft import DataFrame, col
from tests.conftest import get_tests_daft_runner_name
from tests.test_subscribers import MockSubscriber

pytestmark = pytest.mark.skipif(
    get_tests_daft_runner_name() == "ray",
    reason="bytes.read is not yet emitted to Python subscribers on Ray Flotilla",
)

ROWS = 20_000


def _bytes_read(query: Callable[[], DataFrame]) -> int:
    subscriber = MockSubscriber()
    with daft.with_subscriber("bytes", subscriber):
        query().collect()
    stats = subscriber.query_node_stats[subscriber.query_ids[-1]]
    return sum(value for node in stats.values() for name, value in node.items() if name == "bytes.read")


def _gzipped(path: Path) -> Path:
    zipped = path.with_name(path.name + ".gz")
    with path.open("rb") as src, gzip.open(zipped, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return zipped


@pytest.fixture(scope="module")
def files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("local-bytes")
    csv = root / "rows.csv"
    csv.write_text("a,b\n" + "".join(f"{i},v{i}\n" for i in range(ROWS)))
    jsonl = root / "rows.jsonl"
    jsonl.write_text("".join(f'{{"a": {i}, "b": "v{i}"}}\n' for i in range(ROWS)))
    text = root / "rows.txt"
    text.write_text("".join(f"line {i}\n" for i in range(ROWS)))
    parquet = root / "rows.parquet"
    pq.write_table(
        pa.table({"a": list(range(ROWS)), "b": [f"v{i}" for i in range(ROWS)]}),
        parquet,
        row_group_size=2_000,
        write_page_index=True,
    )
    return {
        "csv": csv,
        "csv.gz": _gzipped(csv),
        "jsonl": jsonl,
        "jsonl.gz": _gzipped(jsonl),
        "text": text,
        "parquet": parquet,
    }


READERS: dict[str, Callable[[str], DataFrame]] = {
    "csv": daft.read_csv,
    "csv.gz": daft.read_csv,
    "jsonl": daft.read_json,
    "jsonl.gz": daft.read_json,
    "text": daft.read_text,
}


@pytest.mark.parametrize("kind", READERS.keys())
def test_reading_a_whole_file_reports_its_size(files: dict[str, Path], kind: str) -> None:
    path = files[kind]

    read = _bytes_read(lambda: READERS[kind](str(path)))

    assert read == path.stat().st_size


def test_reading_a_whole_parquet_file_reports_at_most_its_size(files: dict[str, Path]) -> None:
    path = files["parquet"]
    metadata = pq.ParquetFile(path).metadata
    column_chunks = sum(
        metadata.row_group(g).column(c).total_compressed_size
        for g in range(metadata.num_row_groups)
        for c in range(metadata.num_columns)
    )

    read = _bytes_read(lambda: daft.read_parquet(str(path)))

    assert column_chunks < read <= path.stat().st_size


@pytest.mark.parametrize("page_index", [False, True], ids=["row groups", "pages"])
def test_a_parquet_read_skipping_row_groups_reports_fewer_bytes(files: dict[str, Path], page_index: bool) -> None:
    path = str(files["parquet"])
    whole = _bytes_read(lambda: daft.read_parquet(path))

    with daft.execution_config_ctx(parquet_page_index=page_index):
        filtered = _bytes_read(lambda: daft.read_parquet(path).where(col("a").is_in([5, 6, 7])))

    assert 0 < filtered < whole / 4


def test_a_warc_read_reports_the_bytes_it_parses(tmp_path: Path) -> None:
    record = (
        b"WARC/1.0\r\nWARC-Type: resource\r\nWARC-Record-ID: <urn:uuid:1>\r\n"
        b"WARC-Date: 2024-01-01T00:00:00Z\r\nContent-Length: 5\r\n\r\nhello\r\n\r\n"
    )
    plain = tmp_path / "one.warc"
    plain.write_bytes(record * 3)
    zipped = _gzipped(plain)

    plain_read = _bytes_read(lambda: daft.read_warc(str(plain)))
    zipped_read = _bytes_read(lambda: daft.read_warc(str(zipped)))

    assert plain_read == zipped_read > 0
