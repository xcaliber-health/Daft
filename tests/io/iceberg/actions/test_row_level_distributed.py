"""Row-level writes that only a distributed runner can get wrong.

Which rows pair with which is decided per task there, so a row of the table has
to reach the same task as every source row that could claim it. These assertions
hold on either runner; only the distributed one can fail them.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyiceberg")

import daft
from daft import col
from daft.catalog.__iceberg import IcebergTable
from daft.io.iceberg import MergeCardinalityError, MergeFailedException

from ._helpers import make_seeded_table, read_ids

MODES = ["copy-on-write", "merge-on-read"]


@pytest.fixture(params=MODES)
def mode(request):
    return request.param


def _table(catalog, mode, *, n_files=6, rows_per_file=50):
    table = make_seeded_table(catalog, "default.distributed", n_files=n_files, rows_per_file=rows_per_file)
    with table.transaction() as transaction:
        transaction.set_properties(**{"write.merge.mode": mode, "format-version": "2"})
    return table.refresh()


def _rows(table):
    frame = daft.read_iceberg(table.refresh()).sort("id").to_pydict()
    return list(zip(frame["id"], frame["label"]))


def test_a_merge_spread_over_tasks_changes_every_row_it_matches(local_catalog, mode):
    table = _table(local_catalog, mode)
    changed = list(range(0, 300, 7))
    source = daft.from_pydict({"id": changed, "label": ["merged"] * len(changed)})

    result = (
        IcebergTable.from_iceberg(table)
        .merge_into(source, on=col("target.id") == col("source.id"))
        .when_matched()
        .update({"label": col("source.label")})
        .execute()
    )

    rows = dict(_rows(table))
    assert result.rows_updated == len(changed)
    assert all(rows[row_id] == "merged" for row_id in changed)
    assert sum(1 for label in rows.values() if label == "merged") == len(changed)
    assert len(rows) == 300, "no row was lost or duplicated"


def test_a_row_matched_twice_is_refused_wherever_the_matches_land(local_catalog, mode):
    table = _table(local_catalog, mode)
    # Duplicated keys spread across the source, so the pairs need not be adjacent.
    ids = list(range(0, 300, 11)) * 2
    source = daft.from_pydict({"id": sorted(ids), "label": ["merged"] * len(ids)})

    with pytest.raises((MergeCardinalityError, MergeFailedException, Exception), match="more than one source row"):
        (
            IcebergTable.from_iceberg(table)
            .merge_into(source, on=col("target.id") == col("source.id"))
            .when_matched()
            .update({"label": col("source.label")})
            .execute()
        )

    assert sorted(read_ids(table.refresh())) == list(range(300)), "the table is unchanged"


def test_rows_added_by_a_merge_land_once(local_catalog, mode):
    table = _table(local_catalog, mode)
    added = list(range(1000, 1200))
    source = daft.from_pydict({"id": added, "label": ["added"] * len(added)})

    result = (
        IcebergTable.from_iceberg(table)
        .merge_into(source, on=col("target.id") == col("source.id"))
        .when_matched()
        .update({"label": col("source.label")})
        .when_not_matched()
        .insert_all()
        .execute()
    )

    ids = read_ids(table.refresh())
    assert result.rows_inserted == len(added)
    assert len(ids) == len(set(ids)) == 500
