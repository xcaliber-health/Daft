"""A rewrite over a table that carries equality deletes.

An equality delete names column values rather than row positions, so it
applies to every strictly older data file in its partition, or in every
partition when written under an unpartitioned spec. The rewrite applies each
delete to exactly the files it covers, so no rewritten file carries a row a
delete had removed and no row committed beside a delete is lost.

Neither the catalog library nor the reference engine's batch writers emit
equality deletes, so the helpers here write them the way a streaming writer
would.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("pyiceberg")

from daft.catalog import Table
from daft.io.iceberg import RewriteConflict
from tests.io.iceberg.actions._helpers import (
    commit_equality_delete_rows,
    commit_equality_deletes,
    commit_equality_upsert,
    inject_once_around_rewrite,
    make_seeded_table,
)

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog
    from pyiceberg.table import Table as PyIcebergTable

_REWRITE_ALL = {"rewrite-all": True, "min-input-files": 2}
_REWRITE_ALL_RECLAIMING = {**_REWRITE_ALL, "remove-dangling-deletes": True}


def _live_entries(table: PyIcebergTable) -> list[tuple[int, str, int]]:
    """Return ``(content, path, sequence number)`` of every live file, from the manifests.

    A scan cannot be used: the catalog library refuses to plan a table carrying
    equality deletes.
    """
    table.refresh()
    out: list[tuple[int, str, int]] = []
    for manifest in table.current_snapshot().manifests(table.io):
        for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=True):
            out.append((int(entry.data_file.content), entry.data_file.file_path, int(entry.sequence_number or 0)))
    return out


def _rows(table: PyIcebergTable) -> dict[int, str | None]:
    """Return ``id -> label`` over the live data files, read directly."""
    tables = []
    for content, path, _ in _live_entries(table):
        if content == 0:
            with table.io.new_input(path).open() as handle:
                tables.append(pq.read_table(handle))
    combined = pa.concat_tables(tables).to_pydict()
    id_column = "id" if "id" in combined else next(name for name in combined if name != "label")
    ids = combined[id_column]
    assert len(ids) == len(set(ids)), "a row was duplicated"
    return dict(zip(ids, combined["label"]))


def test_a_table_can_be_given_a_real_equality_delete(local_catalog):
    """The fixture itself is worth asserting: without it nothing below is real."""
    table = make_seeded_table(local_catalog, "default.t_eq_fixture", n_files=4, rows_per_file=25)

    commit_equality_deletes(table, "id", [1, 2, 3])

    assert sum(1 for content, _, _ in _live_entries(table) if content == 2) == 1


def test_rows_an_equality_delete_removes_do_not_survive_the_rewrite(local_catalog):
    # Arrange
    table = make_seeded_table(local_catalog, "default.t_eq_apply", n_files=4, rows_per_file=25)
    commit_equality_deletes(table, "id", [1, 2, 3])

    # Act
    result = Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL)

    # Assert: the rows are gone, and the delete, still live, sits at or below
    # every output's sequence number, so it can no longer apply to anything.
    assert result.rewritten_files == 4
    assert set(_rows(table)) == set(range(100)) - {1, 2, 3}
    entries = _live_entries(table)
    delete_seq = max(seq for content, _, seq in entries if content == 2)
    assert all(seq >= delete_seq for content, _, seq in entries if content == 0)


def test_the_delete_is_reclaimed_once_no_data_file_is_older(local_catalog):
    table = make_seeded_table(local_catalog, "default.t_eq_reclaim", n_files=4, rows_per_file=25)
    commit_equality_deletes(table, "id", [1, 2, 3])

    result = Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL_RECLAIMING)

    assert result.removed_delete_files == 2, "consumed on read, then reclaimed as dangling"
    assert not [path for content, path, _ in _live_entries(table) if content != 0]
    ids = Table.from_iceberg(table).read().to_pydict()["id"]
    assert sorted(ids) == sorted(set(range(100)) - {1, 2, 3})


def test_rows_committed_beside_the_delete_keep_their_new_values(local_catalog):
    # Arrange: a streaming upsert writes the new rows and the delete of their
    # keys in one commit, at one sequence number.
    table = make_seeded_table(local_catalog, "default.t_eq_upsert", n_files=4, rows_per_file=25)
    commit_equality_upsert(
        table,
        pa.table({"id": pa.array([5, 6], type=pa.int64()), "label": pa.array(["new-5", "new-6"])}),
        ["id"],
    )

    # Act
    Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL_RECLAIMING)

    # Assert: every key once, the upserted ones with their new values.
    rows = _rows(table)
    assert set(rows) == set(range(100))
    assert rows[5] == "new-5" and rows[6] == "new-6"
    assert rows[7] == "seed"


def test_a_null_delete_value_matches_null_rows(local_catalog):
    table = make_seeded_table(local_catalog, "default.t_eq_null", n_files=2, rows_per_file=10)
    table.append(pa.table({"id": pa.array([100, 101], type=pa.int64()), "label": pa.array([None, "kept"])}))
    commit_equality_delete_rows(table, pa.table({"label": pa.array([None], type=pa.string())}), ["label"])

    Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL_RECLAIMING)

    rows = _rows(table)
    assert 100 not in rows
    assert rows[101] == "kept" and len(rows) == 21


def test_every_equality_column_must_match(local_catalog):
    table = make_seeded_table(local_catalog, "default.t_eq_multi", n_files=2, rows_per_file=10)
    commit_equality_delete_rows(
        table,
        pa.table({"id": pa.array([1, 2], type=pa.int64()), "label": pa.array(["seed", "other"])}),
        ["id", "label"],
    )

    Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL_RECLAIMING)

    rows = _rows(table)
    assert 1 not in rows
    assert rows[2] == "seed"


def test_a_delete_with_a_null_key_and_a_value_key_matches_both_together(local_catalog):
    # Arrange: rows (1, seed) and (200, null); the delete names (1, null) and
    # (200, null), so only the second is removed.
    table = make_seeded_table(local_catalog, "default.t_eq_mixed_null", n_files=1, rows_per_file=10)
    table.append(pa.table({"id": pa.array([200], type=pa.int64()), "label": pa.array([None], type=pa.string())}))
    commit_equality_delete_rows(
        table,
        pa.table({"id": pa.array([1, 200], type=pa.int64()), "label": pa.array([None, None], type=pa.string())}),
        ["id", "label"],
    )

    # Act
    Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL_RECLAIMING)

    # Assert
    rows = _rows(table)
    assert 200 not in rows
    assert rows[1] == "seed"


def _regional_table(catalog: Catalog, name: str) -> PyIcebergTable:
    """Return a table partitioned by region after an unpartitioned start, with rows under both specs."""
    from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC
    from pyiceberg.schema import Schema
    from pyiceberg.types import LongType, NestedField, StringType

    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "label", StringType(), required=False),
        NestedField(3, "region", StringType(), required=False),
    )
    table = catalog.create_table(identifier=name, schema=schema, partition_spec=UNPARTITIONED_PARTITION_SPEC)
    table.append(_regional_rows(0, 10))
    with table.update_spec() as update:
        update.add_identity("region")
    table.refresh()
    table.append(_regional_rows(10, 20))
    return table


def _regional_rows(start: int, end: int) -> pa.Table:
    return pa.table(
        {
            "id": pa.array(list(range(start, end)), type=pa.int64()),
            "label": pa.array(["seed"] * (end - start)),
            "region": pa.array(["a" if i % 2 == 0 else "b" for i in range(start, end)]),
        }
    )


def test_a_delete_under_the_unpartitioned_spec_covers_every_partition(local_catalog):
    # Arrange: rows under the old unpartitioned spec and the new regional one;
    # a delete written under the unpartitioned spec is global.
    table = _regional_table(local_catalog, "default.t_eq_global")
    commit_equality_delete_rows(table, pa.table({"id": pa.array([2, 3, 12, 13], type=pa.int64())}), ["id"], spec_id=0)

    # Act
    Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL_RECLAIMING)

    # Assert: removed from both regions and both specs, then reclaimed since
    # every surviving file is newer than the delete.
    assert set(_rows(table)) == set(range(20)) - {2, 3, 12, 13}
    assert not [path for content, path, _ in _live_entries(table) if content != 0]


def test_a_partition_scoped_delete_leaves_other_partitions_alone(local_catalog):
    from pyiceberg.typedef import Record

    table = _regional_table(local_catalog, "default.t_eq_partition")
    # Region "a" holds the even ids; a delete over region "a" naming an odd id
    # from region "b" must not reach it.
    commit_equality_delete_rows(
        table,
        pa.table({"id": pa.array([12, 13], type=pa.int64())}),
        ["id"],
        partition=Record("a"),
        spec_id=table.spec().spec_id,
    )

    Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL_RECLAIMING)

    rows = _rows(table)
    assert 12 not in rows
    assert 13 in rows, "the delete is scoped to region a"
    assert 2 in rows, "the unpartitioned file is not in the delete's partition"


def test_a_renamed_column_still_matches_by_field_id(local_catalog):
    table = make_seeded_table(local_catalog, "default.t_eq_rename", n_files=2, rows_per_file=10)
    commit_equality_deletes(table, "id", [4])
    with table.update_schema() as update:
        update.rename_column("id", "ident")

    Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL_RECLAIMING)

    assert 4 not in _rows(table)


def test_a_delete_over_a_dropped_column_is_refused_before_any_work(local_catalog):
    table = make_seeded_table(local_catalog, "default.t_eq_dropped", n_files=2, rows_per_file=10)
    commit_equality_deletes(table, "label", ["seed"])
    with table.update_schema() as update:
        update.delete_column("label")
    snapshots_before = len(table.metadata.snapshots)

    with pytest.raises(ValueError, match="not a top-level column"):
        Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL)

    table.refresh()
    assert len(table.metadata.snapshots) == snapshots_before


def test_an_equality_delete_landing_after_the_plan_refuses_the_commit(local_catalog, monkeypatch):
    # Arrange: a foreign writer commits an equality delete over the inputs while
    # the rewrite is between plan and commit.
    table = make_seeded_table(local_catalog, "default.t_eq_conflict", n_files=4, rows_per_file=25)
    fired = inject_once_around_rewrite(monkeypatch, lambda t: commit_equality_deletes(t, "id", [7]), before=False)

    # Act / Assert: committing the outputs would bring row 7 back.
    with pytest.raises(RewriteConflict):
        Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL)
    assert fired["fired"] == 1


def test_other_maintenance_still_works_alongside_equality_deletes(local_catalog):
    table = make_seeded_table(local_catalog, "default.t_eq_manifests", n_files=6, rows_per_file=25)
    commit_equality_deletes(table, "id", [9])

    result = Table.from_iceberg(table).rewrite_manifests()

    assert result.rewritten_manifests_count >= 1
    assert sum(1 for content, _, _ in _live_entries(table) if content == 2) == 1


def test_written_manifests_declare_equality_ids_as_ints(local_catalog):
    """Readers that decode the field into an int array refuse a manifest typing it as longs."""
    from pyiceberg.avro.file import AvroFile
    from pyiceberg.manifest import MANIFEST_ENTRY_SCHEMAS, ManifestEntry
    from pyiceberg.types import IntegerType

    table = make_seeded_table(local_catalog, "default.t_eq_manifest_type", n_files=2, rows_per_file=10)
    commit_equality_deletes(table, "id", [1])

    Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL_RECLAIMING)

    table.refresh()
    declared = []
    for manifest in table.current_snapshot().manifests(table.io):
        with AvroFile[ManifestEntry](table.io.new_input(manifest.manifest_path), MANIFEST_ENTRY_SCHEMAS[2], {}) as avro:
            data_file = next(field for field in avro.schema.fields if field.name == "data_file")
            equality_ids = next(field for field in data_file.field_type.fields if field.name == "equality_ids")
            declared.append(equality_ids.field_type.element_type)
    assert declared and all(isinstance(t, IntegerType) for t in declared)


def test_manifests_written_by_the_library_alone_still_read(tmp_path):
    """A table written before this process imported the maintenance package types the field as longs."""
    import subprocess
    import sys

    from pyiceberg.catalog.sql import SqlCatalog

    script = f"""
import pyarrow as pa
from pyiceberg.catalog.sql import SqlCatalog
catalog = SqlCatalog("outside", uri="sqlite:///{tmp_path}/outside.db", warehouse="file://{tmp_path}")
catalog.create_namespace("default")
table = catalog.create_table("default.t_outside", schema=pa.schema([("id", pa.int64()), ("label", pa.string())]), properties={{"format-version": "2"}})
for start in (0, 10, 20):
    table.append(pa.table({{"id": pa.array(range(start, start + 10), type=pa.int64()), "label": pa.array(["seed"] * 10)}}))
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)
    table = SqlCatalog("outside", uri=f"sqlite:///{tmp_path}/outside.db", warehouse=f"file://{tmp_path}").load_table(
        "default.t_outside"
    )

    result = Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL)

    assert result.rewritten_files == 3
    assert sorted(_rows(table)) == list(range(30))


def test_many_upsert_generations_rewrite_in_one_pass(local_catalog):
    """Every data file of a long upsert history has its own set of applicable deletes.

    A group then holds one bucket per file; the delete keys are read once and
    each bucket joins against its slice, so the plan stays a scan and a join
    per bucket however long the history.
    """
    table = make_seeded_table(local_catalog, "default.t_eq_generations", n_files=2, rows_per_file=50)
    for generation in range(60):
        key = generation % 50
        commit_equality_upsert(
            table,
            pa.table({"id": pa.array([key], type=pa.int64()), "label": pa.array([f"gen-{generation}"])}),
            ["id"],
        )

    result = Table.from_iceberg(table).rewrite_data_files("binpack", options=_REWRITE_ALL_RECLAIMING)

    rows = _rows(table)
    assert result.rewritten_files == 62
    assert set(rows) == set(range(100))
    assert all(rows[key] == f"gen-{max(g for g in range(60) if g % 50 == key)}" for key in range(50))
    assert all(rows[key] == "seed" for key in range(50, 100))
