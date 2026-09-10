"""Reading and rewriting files that carry no field ids, through the table's name mapping.

A file registered rather than written by the engine has no field ids in its
footer. The table's declared name mapping says which names identify each field
in such a file, and keeps an old name alongside the new one after a rename, so
a file registered under the old name still resolves.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("pyiceberg")

from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC

import daft
from daft.catalog import Table
from daft.daft import PyField
from daft.io.iceberg.schema_field_id_mapping_visitor import (
    NAME_MAPPING_METADATA_KEY,
    NAME_MAPPING_PROPERTY,
    attach_name_mapping,
)

_ROWS = 40
_FILES = 4


def _registered_table(local_catalog, simple_schema, tmp_path, name: str):
    """A table whose files were written outside the engine and registered, so they carry no field ids."""
    table = local_catalog.create_table(
        identifier=name,
        schema=simple_schema,
        partition_spec=UNPARTITIONED_PARTITION_SPEC,
        properties={"format-version": "2"},
    )
    staging = tmp_path / "staged"
    staging.mkdir(exist_ok=True)
    paths = []
    per = _ROWS // _FILES
    for index in range(_FILES):
        batch = pa.table(
            {
                "id": pa.array(range(index * per, (index + 1) * per), type=pa.int64()),
                "label": pa.array([f"r{j}" for j in range(index * per, (index + 1) * per)], type=pa.string()),
            }
        )
        path = staging / f"{name.replace('.', '_')}-{index}.parquet"
        pq.write_table(batch, path)
        paths.append(str(path))
    table.add_files(paths)
    return local_catalog.load_table(name)


def _ids(table, column: str) -> list[int]:
    return sorted(v for v in daft.read_iceberg(table).to_pydict()[column] if v is not None)


def test_a_registered_file_still_reads_after_the_column_is_renamed(local_catalog, simple_schema, tmp_path):
    # Arrange: the files carry "id"; the live schema will carry "identifier".
    table = _registered_table(local_catalog, simple_schema, tmp_path, "default.t_nm_rename")
    assert NAME_MAPPING_PROPERTY in table.properties
    with table.update_schema() as update:
        update.rename_column("id", "identifier")
    table = local_catalog.load_table("default.t_nm_rename")

    # Act / Assert: the old name resolves through the mapping.
    assert _ids(table, "identifier") == list(range(_ROWS))


def test_a_rewrite_after_a_rename_keeps_every_value(local_catalog, simple_schema, tmp_path):
    # Arrange
    table = _registered_table(local_catalog, simple_schema, tmp_path, "default.t_nm_rewrite")
    with table.update_schema() as update:
        update.rename_column("id", "identifier")
    table = local_catalog.load_table("default.t_nm_rewrite")

    # Act
    result = Table.from_iceberg(table).rewrite_data_files(
        "binpack", options={"rewrite-all": True, "min-input-files": 2}
    )

    # Assert: the rewritten files carry field ids and the values the registered ones held.
    assert result.rewritten_files == _FILES
    assert _ids(table.refresh(), "identifier") == list(range(_ROWS))


def test_a_column_the_mapping_does_not_name_is_refused_not_nulled(local_catalog, simple_schema, tmp_path):
    # Arrange: rename the column, then declare a mapping that forgot the old name.
    table = _registered_table(local_catalog, simple_schema, tmp_path, "default.t_nm_unknown")
    with table.update_schema() as update:
        update.rename_column("id", "identifier")
    table = local_catalog.load_table("default.t_nm_unknown")
    with table.transaction() as tx:
        tx.set_properties(
            **{NAME_MAPPING_PROPERTY: '[{"field-id": 1, "names": ["identifier"]}, {"field-id": 2, "names": ["label"]}]'}
        )
    table = local_catalog.load_table("default.t_nm_unknown")

    # Act / Assert: the files carry "id", which nothing now names.
    with pytest.raises(Exception, match="could not be matched by name"):
        daft.read_iceberg(table).to_pydict()


def test_attach_name_mapping_carries_every_declared_name(local_catalog, simple_schema, tmp_path):
    # Arrange: a mapping declaring two names for field 1.
    mapping = {1: PyField.create("identifier", daft.DataType.int64()._dtype)}
    properties = {NAME_MAPPING_PROPERTY: '[{"field-id": 1, "names": ["id", "identifier"]}]'}

    # Act
    attached = attach_name_mapping(mapping, properties)

    # Assert: both names travel with the field; an absent or malformed mapping changes nothing.
    assert attached[1].metadata()[NAME_MAPPING_METADATA_KEY] == "id\nidentifier"
    assert attached[1].name() == "identifier"
    assert attach_name_mapping(mapping, {})[1].metadata() == {}
    assert attach_name_mapping(mapping, {NAME_MAPPING_PROPERTY: "not json"})[1].metadata() == {}
