from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from pyiceberg.io.pyarrow import schema_to_pyarrow
from pyiceberg.schema import SchemaVisitor

from daft import DataType
from daft.daft import PyField

if TYPE_CHECKING:
    from pyiceberg.schema import Schema
    from pyiceberg.types import ListType, MapType, NestedField, PrimitiveType, StructType

FieldIdMapping = dict[int, PyField]

#: Field metadata key carrying a field's declared name-mapping names, one per line.
NAME_MAPPING_METADATA_KEY = "iceberg.name-mapping"
#: Table property holding the declared name mapping.
NAME_MAPPING_PROPERTY = "schema.name-mapping.default"


def _nested_field_to_daft_pyfield(field: NestedField) -> PyField:
    return PyField.create(field.name, DataType.from_arrow_type(schema_to_pyarrow(field.field_type))._dtype)


def attach_name_mapping(mapping: FieldIdMapping, properties: Mapping[str, str]) -> FieldIdMapping:
    """Return ``mapping`` with each field carrying the names its table's name mapping assigns.

    The mapping, declared under ``schema.name-mapping.default``, identifies
    fields in files that carry no field ids and keeps a renamed column's old
    name alongside the new one. Fields it does not name, and a mapping that
    cannot be parsed, leave the input unchanged.
    """
    raw = properties.get(NAME_MAPPING_PROPERTY)
    if not raw:
        return mapping
    from pyiceberg.table.name_mapping import parse_mapping_from_json

    try:
        declared = parse_mapping_from_json(raw)
    except ValueError:
        return mapping
    names_by_id: dict[int, list[str]] = {}
    pending = list(declared.root)
    while pending:
        mapped = pending.pop()
        if mapped.field_id is not None and mapped.names:
            names_by_id[mapped.field_id] = list(mapped.names)
        pending.extend(mapped.fields)
    out = dict(mapping)
    for field_id, names in names_by_id.items():
        field = out.get(field_id)
        if field is None:
            continue
        out[field_id] = PyField.create(field.name(), field.dtype(), {NAME_MAPPING_METADATA_KEY: "\n".join(names)})
    return out


class SchemaFieldIdMappingVisitor(SchemaVisitor[FieldIdMapping]):  # type: ignore[misc]
    """Extracts a mapping of {field_id: PyField} from an Iceberg schema."""

    def schema(self, schema: Schema, struct_result: FieldIdMapping) -> FieldIdMapping:
        """Visit a Schema."""
        return struct_result

    def struct(self, struct: StructType, field_results: list[FieldIdMapping]) -> FieldIdMapping:
        """Visit a StructType."""
        result = {field.field_id: _nested_field_to_daft_pyfield(field) for field in struct.fields}
        for r in field_results:
            result.update(r)
        return result

    def field(self, field: NestedField, field_result: FieldIdMapping) -> FieldIdMapping:
        """Visit a NestedField."""
        field_result[field.field_id] = _nested_field_to_daft_pyfield(field)
        return field_result

    def list(self, list_type: ListType, element_result: FieldIdMapping) -> FieldIdMapping:
        """Visit a ListType."""
        element_result[list_type.element_id] = _nested_field_to_daft_pyfield(list_type.element_field)
        return element_result

    def map(self, map_type: MapType, key_result: FieldIdMapping, value_result: FieldIdMapping) -> FieldIdMapping:
        result = {**key_result, **value_result}
        result[map_type.key_id] = _nested_field_to_daft_pyfield(map_type.key_field)
        result[map_type.value_id] = _nested_field_to_daft_pyfield(map_type.value_field)
        return result

    def primitive(self, primitive: PrimitiveType) -> FieldIdMapping:
        """Visit a PrimitiveType."""
        return {}
