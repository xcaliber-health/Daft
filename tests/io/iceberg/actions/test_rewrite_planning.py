"""Which files a rewrite selects, and how much work a cap allows.

Selection happens in two stages that ask different questions. The first asks of
each file whether it is worth touching: is it outside the desired size range, or
carrying enough deletes. The second asks of each packed group whether it is
worth rewriting: enough files, enough content, too much content, or a file with
too many deletes. Applying the file-count question to a whole partition before
packing answers neither, and both discards work and keeps work it should not.
"""

from __future__ import annotations

import random
from typing import Any

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

from daft.catalog import Table

_MIB = 1024 * 1024
_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def _sized_table(catalog, name: str, *, n_files: int, rows_per_file: int, pad: int = 64) -> Any:
    """Seed a table whose files are large enough for size thresholds to bite.

    The padding is random so it does not compress away, which keeps each file's
    on-disk size proportional to its row count.
    """
    rng = random.Random(17)
    table = catalog.create_table(
        identifier=name,
        schema=pa.schema([("id", pa.int64()), ("pad", pa.string())]),
        properties={"format-version": "2"},
    )
    for batch in range(n_files):
        start = batch * rows_per_file
        table.append(
            pa.table(
                {
                    "id": pa.array(range(start, start + rows_per_file), type=pa.int64()),
                    "pad": pa.array(
                        ["".join(rng.choices(_ALPHABET, k=pad)) for _ in range(rows_per_file)],
                        type=pa.string(),
                    ),
                }
            )
        )
    table.refresh()
    return table


def _file_count(table: Any) -> int:
    return len(list(table.scan().plan_files()))


def _rows(table: Any) -> int:
    table.refresh()
    return table.scan().to_arrow().num_rows


def test_a_group_with_enough_content_is_rewritten_below_min_input_files(local_catalog):
    """Too few files is not the only reason to rewrite; too much content is another."""
    table = _sized_table(local_catalog, "default.t_enough_content", n_files=2, rows_per_file=13_000)
    sizes = [task.file.file_size_in_bytes for task in table.scan().plan_files()]
    assert all(size < _MIB for size in sizes), "each file must be under the target"
    assert sum(sizes) > _MIB, "together they must exceed it"

    result = Table.from_iceberg(table).rewrite_data_files(
        "binpack", options={"min-input-files": 5, "target-file-size-bytes": _MIB}
    )

    assert result.rewritten_files == 2
    assert _rows(table) == 26_000


def test_a_group_with_no_reason_to_be_rewritten_is_left_alone(local_catalog):
    """The same files against a target they do not approach are not worth touching."""
    table = _sized_table(local_catalog, "default.t_no_reason", n_files=2, rows_per_file=13_000)
    before = _file_count(table)

    result = Table.from_iceberg(table).rewrite_data_files(
        "binpack", options={"min-input-files": 5, "target-file-size-bytes": 8 * _MIB}
    )

    assert result.rewritten_files == 0
    assert _file_count(table) == before


def test_a_small_packed_group_is_discarded(local_catalog):
    """A residual group too small to be worth rewriting is dropped, not rewritten."""
    table = _sized_table(local_catalog, "default.t_small_group", n_files=2, rows_per_file=20)

    result = Table.from_iceberg(table).rewrite_data_files("binpack", options={"min-input-files": 5})

    assert result.rewritten_files == 0


def test_a_cap_below_the_first_group_still_does_that_much_work(local_catalog):
    """A cap smaller than the first group used to do nothing and report success."""
    table = _sized_table(local_catalog, "default.t_cap_small", n_files=6, rows_per_file=200)

    result = Table.from_iceberg(table).rewrite_data_files(
        "binpack", options={"min-input-files": 2, "max-files-to-rewrite": 2}
    )

    assert result.rewritten_files == 2
    assert result.added_files >= 1
    assert _rows(table) == 1200


@pytest.mark.parametrize("cap", [2, 3, 4, 5])
def test_a_cap_is_filled_rather_than_undershot(local_catalog, cap):
    """The cap bounds how many files are rewritten; it is not a lower bound."""
    table = _sized_table(local_catalog, f"default.t_cap_{cap}", n_files=6, rows_per_file=200)

    result = Table.from_iceberg(table).rewrite_data_files(
        "binpack", options={"min-input-files": 2, "max-files-to-rewrite": cap}
    )

    assert result.rewritten_files == cap
    assert _rows(table) == 1200


def test_the_delete_ratio_threshold_is_accepted_and_validated(local_catalog):
    """The option exists, defaults to the reference's value, and rejects bad input."""
    from daft.daft import _iceberg as rust_iceberg

    assert rust_iceberg.validate_options_py({})["delete-ratio-threshold"] == pytest.approx(0.3)
    normalized = rust_iceberg.validate_options_py({"delete-ratio-threshold": 0.5})
    assert normalized["delete-ratio-threshold"] == pytest.approx(0.5)


@pytest.mark.parametrize("bad", [0.0, -0.2, 1.5])
def test_an_out_of_range_delete_ratio_is_refused(local_catalog, bad):
    """A ratio outside (0, 1] cannot describe a fraction of a file's rows."""
    table = _sized_table(local_catalog, f"default.t_ratio_{abs(hash(bad))}", n_files=2, rows_per_file=20)

    with pytest.raises(ValueError, match="delete-ratio-threshold"):
        Table.from_iceberg(table).rewrite_data_files("binpack", options={"delete-ratio-threshold": bad})


def test_a_rewrite_merges_small_files_to_the_target(local_catalog):
    """Undersized files are merged into ones near the target, on either runner.

    This is what compaction is for. The planner decides how many files a group
    becomes and the writer needs a measured expansion ratio to hit that, since
    it rolls on an estimate of on-disk size taken from rows held in memory.
    """
    table = _sized_table(local_catalog, "default.t_merge", n_files=24, rows_per_file=7_000)
    before = sorted(task.file.file_size_in_bytes for task in table.scan().plan_files())
    assert max(before) < _MIB, "the inputs must be undersized for this to be a merge"

    Table.from_iceberg(table).rewrite_data_files(
        "binpack",
        options={"target-file-size-bytes": 2 * _MIB, "min-input-files": 2, "rewrite-all": True},
    )

    table.refresh()
    sizes = sorted(task.file.file_size_in_bytes for task in table.scan().plan_files())
    assert _rows(table) == 168_000
    assert len(sizes) < len(before), f"a merge must reduce the file count: {len(before)} -> {len(sizes)}"
    # Every file but the remainder lands near the target rather than overshooting.
    full = sizes[1:]
    assert max(full) < 2.5 * _MIB, f"no file should overshoot the target: {sizes}"
    assert min(full) > 1.5 * _MIB, f"no full file should undershoot it: {sizes}"


def test_an_unknown_option_is_refused(local_catalog):
    """A mistyped option fails rather than quietly doing nothing."""
    table = _sized_table(local_catalog, "default.t_unknown_opt", n_files=2, rows_per_file=20)

    with pytest.raises(ValueError, match="unsupported option"):
        Table.from_iceberg(table).rewrite_data_files("binpack", options={"target-file-siz-bytes": 1024})


def test_the_documented_zorder_option_names_are_the_ones_read(local_catalog):
    """The names the documentation gives are the names the parser accepts."""
    from daft.daft import _iceberg as rust_iceberg

    normalized = rust_iceberg.validate_options_py({"max-output-size": 512, "var-length-contribution": 12})

    assert normalized["max-output-size"] == 512
    assert normalized["var-length-contribution"] == 12


def test_min_input_files_of_one_is_accepted(local_catalog):
    """The reference requires only that it be positive."""
    table = _sized_table(local_catalog, "default.t_min_one", n_files=3, rows_per_file=200)

    result = Table.from_iceberg(table).rewrite_data_files(
        "binpack", options={"min-input-files": 1, "rewrite-all": True}
    )

    assert result.rewritten_files == 3
    assert _rows(table) == 600


def test_min_input_files_of_zero_is_refused(local_catalog):
    """Zero would admit a group with nothing in it."""
    table = _sized_table(local_catalog, "default.t_min_zero", n_files=2, rows_per_file=20)

    with pytest.raises(ValueError, match="min-input-files"):
        Table.from_iceberg(table).rewrite_data_files("binpack", options={"min-input-files": 0})


def test_a_group_cap_below_the_target_is_accepted(local_catalog):
    """A small cap bounds how much one group holds; it is not invalid."""
    table = _sized_table(local_catalog, "default.t_small_cap", n_files=6, rows_per_file=200)

    result = Table.from_iceberg(table).rewrite_data_files(
        "binpack",
        options={
            "target-file-size-bytes": 128 * _MIB,
            "max-file-group-size-bytes": 4 * _MIB,
            "min-input-files": 2,
            "rewrite-all": True,
        },
    )

    assert result.rewritten_files == 6
    assert _rows(table) == 1200


def test_a_group_cap_of_zero_is_refused(local_catalog):
    """A cap of zero could never admit a file."""
    table = _sized_table(local_catalog, "default.t_zero_cap", n_files=2, rows_per_file=20)

    with pytest.raises(ValueError, match="max-file-group-size-bytes"):
        Table.from_iceberg(table).rewrite_data_files("binpack", options={"max-file-group-size-bytes": 0})
