from __future__ import annotations

import datetime
import uuid
import warnings
from typing import TYPE_CHECKING, Any

from daft import Expression, col, lit
from daft.datatype import DataType
from daft.expressions.expressions import ExpressionsProjection
from daft.io.common import _get_schema_from_dict
from daft.recordbatch import MicroPartition
from daft.recordbatch.partitioning import PartitionedTable, partition_strings_to_path

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pyiceberg.manifest import DataFile, DataFileContent
    from pyiceberg.manifest import FileFormat as IcebergFileFormat
    from pyiceberg.partitioning import PartitionField as IcebergPartitionField
    from pyiceberg.schema import Schema as IcebergSchema
    from pyiceberg.table import TableProperties as IcebergTableProperties
    from pyiceberg.typedef import Record as IcebergRecord

    from daft.dependencies import pa, pads, pq


def get_missing_columns(
    data_schema: pa.Schema,
    iceberg_schema: IcebergSchema,
    *,
    require_matching_columns: bool = False,
) -> ExpressionsProjection:
    """Add null values for columns in the table schema that are missing from the data.

    Args:
        data_schema (pa.Schema): Schema of the data about to be written.
        iceberg_schema (IcebergSchema): Schema of the table being written to.
        require_matching_columns (bool): When true, refuse data that both lacks
            table columns and carries extra ones. Off by default, since a write may
            deliberately supply a subset of the table's columns or extra ones the
            table drops.

    Returns:
        ExpressionsProjection: One null literal per table column absent from the
            data, typed to match.

    Raises:
        ValueError: If ``require_matching_columns`` is set and the data both lacks
            columns the table has and carries columns the table does not; padding
            one while dropping the other would silently empty a renamed column.
    """
    from pyiceberg.io.pyarrow import schema_to_pyarrow

    iceberg_pyarrow_schema = schema_to_pyarrow(iceberg_schema)

    existing_columns = set(data_schema.names)
    target_columns = set(iceberg_pyarrow_schema.names)

    missing = [name for name in iceberg_pyarrow_schema.names if name not in existing_columns]
    extra = sorted(existing_columns - target_columns)
    if require_matching_columns and missing and extra:
        raise ValueError(
            "Refusing to write data whose columns do not line up with the table's: "
            f"{sorted(missing)} would be filled with nulls while {extra} would be dropped. "
            "The names most likely refer to the same columns, and writing would empty them."
        )

    to_add = [
        lit(None).alias(name).cast(DataType.from_arrow_type(iceberg_pyarrow_schema.field(name).type))
        for name in missing
    ]

    return ExpressionsProjection(to_add)


def coerce_pyarrow_table_to_schema(pa_table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Coerces a PyArrow table to the supplied schema.

    1. For each field in `pa_table`, cast it to the field in `input_schema` if one with a matching name
        is available
    2. Reorder the fields in the casted table to the supplied schema, dropping any fields in `pa_table`
        that do not exist in the supplied schema
    3. If any fields in the supplied schema are not present, add a null array of the correct type

    This ensures that we populate field_id for iceberg as well as fill in null values where needed
    This might break for nested fields with large_strings
    we should test that behavior

    Args:
        pa_table (pa.Table): Table to coerce
        schema (pa.Schema): Iceberg schema to coerce to

    Returns:
        pa.Table: Table with schema == `schema`
    """
    import pyarrow as pa

    input_schema_names = set(schema.names)

    # Perform casting of types to provided schema's types
    cast_to_schema = [
        (schema.field(inferred_field.name) if inferred_field.name in input_schema_names else inferred_field)
        for inferred_field in pa_table.schema
    ]
    casted_table = pa_table.cast(pa.schema(cast_to_schema))

    # Reorder and pad columns with a null column where necessary
    pa_table_column_names = set(casted_table.column_names)
    columns = []
    for name in schema.names:
        if name in pa_table_column_names:
            columns.append(casted_table[name])
        else:
            columns.append(pa.nulls(len(casted_table), type=schema.field(name).type))
    return pa.table(columns, schema=schema)


def partition_field_to_expr(field: IcebergPartitionField, schema: IcebergSchema) -> Expression:
    from pyiceberg.transforms import (
        BucketTransform,
        DayTransform,
        HourTransform,
        IdentityTransform,
        MonthTransform,
        TruncateTransform,
        YearTransform,
    )

    part_col: Expression = col(schema.find_field(field.source_id).name)

    if isinstance(field.transform, IdentityTransform):
        transform_expr = part_col
    elif isinstance(field.transform, YearTransform):
        transform_expr = part_col.partition_years()
    elif isinstance(field.transform, MonthTransform):
        transform_expr = part_col.partition_months()
    elif isinstance(field.transform, DayTransform):
        transform_expr = part_col.partition_days()
    elif isinstance(field.transform, HourTransform):
        transform_expr = part_col.partition_hours()
    elif isinstance(field.transform, BucketTransform):
        transform_expr = part_col.partition_iceberg_bucket(field.transform.num_buckets)
    elif isinstance(field.transform, TruncateTransform):
        transform_expr = part_col.partition_iceberg_truncate(field.transform.width)
    else:
        warnings.warn(f"{field.transform} not implemented, Please make an issue!")
        transform_expr = part_col

    transform_expr = transform_expr.alias(field.name)

    # currently the partitioning expressions change the name of the column
    # so we need to alias it back to the original column name
    return transform_expr


def to_partition_representation(value: Any) -> Any:
    """Converts a partition value to the format expected by Iceberg metadata.

    Most transforms already do this, but the identity transforms preserve the original value type so we need to convert it.
    """
    if value is None:
        return None

    if isinstance(value, datetime.datetime):
        # Convert to microseconds since epoch
        return (value - datetime.datetime(1970, 1, 1)) // datetime.timedelta(microseconds=1)
    elif isinstance(value, datetime.date):
        # Convert to days since epoch
        return (value - datetime.date(1970, 1, 1)) // datetime.timedelta(days=1)
    elif isinstance(value, datetime.time):
        # Convert to microseconds since midnight
        return (value.hour * 60 * 60 + value.minute * 60 + value.second) * 1_000_000 + value.microsecond
    elif isinstance(value, uuid.UUID):
        return str(value)
    else:
        return value


def nan_countable_fields(schema: IcebergSchema, properties: dict[str, str]) -> set[int]:
    """Return the field ids a NaN count applies to.

    Only floating-point columns can hold a NaN, and only those the metrics
    configuration asks for are counted. A float nested in a struct, list, or map
    is a leaf like any other.

    Args:
        schema (IcebergSchema): Schema of the table being written to.
        properties (dict[str, str]): Table properties, which carry the per-column
            metrics configuration.

    Returns:
        set[int]: Field ids to count, empty when no column qualifies.
    """
    from pyiceberg.io.pyarrow import MetricModeTypes, compute_statistics_plan
    from pyiceberg.schema import index_by_id
    from pyiceberg.types import DoubleType, FloatType

    plan = compute_statistics_plan(schema, properties)
    countable: set[int] = set()
    for field_id, field in index_by_id(schema).items():
        if not isinstance(field.field_type, FloatType | DoubleType):
            continue
        collector = plan.get(field_id)
        if collector is None or collector.mode.type == MetricModeTypes.NONE:
            continue
        countable.add(field_id)
    return countable


def _float_leaves(field: pa.Field, array: pa.Array) -> Iterator[tuple[int, pa.Array]]:
    """Yield each floating-point leaf under ``field`` with its field id.

    Descends structs, lists, and maps, so a float nested at any depth is reached.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    field_type = field.type
    if pa.types.is_struct(field_type):
        for index in range(field_type.num_fields):
            yield from _float_leaves(field_type.field(index), array.field(index))
    elif pa.types.is_list(field_type) or pa.types.is_large_list(field_type):
        yield from _float_leaves(field_type.value_field, pc.list_flatten(array))
    elif pa.types.is_map(field_type):
        yield from _float_leaves(field_type.key_field, array.keys)
        yield from _float_leaves(field_type.item_field, array.items)
    elif pa.types.is_floating(field_type):
        raw = (field.metadata or {}).get(b"PARQUET:field_id")
        if raw is not None:
            yield int(raw), array


def count_nans(table: pa.Table, countable: set[int]) -> dict[int, int]:
    """Count NaN values per field id in one written batch.

    Counting runs as a vectorized pass per leaf rather than row by row, and only
    over the fields named in ``countable``.

    Args:
        table (pa.Table): Batch about to be written, already cast to the table's schema.
        countable (set[int]): Field ids to count, as returned by :func:`nan_countable_fields`.

    Returns:
        dict[int, int]: NaN count by field id, omitting fields that hold none.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    counts: dict[int, int] = {}
    if not countable:
        return counts
    for index, field in enumerate(table.schema):
        column = table.column(index)
        array = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
        for field_id, leaf in _float_leaves(field, array):
            if field_id not in countable:
                continue
            total = pc.sum(pc.is_nan(leaf)).as_py()
            if total:
                counts[field_id] = counts.get(field_id, 0) + int(total)
    return counts


def make_iceberg_data_file(
    file_path: str,
    size: int,
    metadata: pq.FileMetaData,
    partition_record: IcebergRecord,
    spec_id: int,
    schema: IcebergSchema,
    properties: dict[str, str],
    sort_order_id: int = 0,
    nan_value_counts: dict[int, int] | None = None,
) -> DataFile:
    import pyiceberg
    from packaging.version import parse
    from pyiceberg.io.pyarrow import (
        compute_statistics_plan,
        parquet_path_to_id_mapping,
    )
    from pyiceberg.manifest import DataFile, DataFileContent
    from pyiceberg.manifest import FileFormat as IcebergFileFormat

    # Bound to the version 2 layout whatever the table's version is: a manifest
    # writer reads records in that layout and writes them out through its own.
    kwargs: dict[str, DataFileContent | IcebergFileFormat | IcebergRecord | str | int | None] = {
        "content": DataFileContent.DATA,
        "file_path": file_path,
        "file_format": IcebergFileFormat.PARQUET,
        "partition": partition_record,
        "file_size_in_bytes": size,
        # Zero is the format's unsorted order.
        "sort_order_id": sort_order_id,
        "equality_ids": None,
        "key_metadata": None,
    }

    if parse(pyiceberg.__version__) >= parse("0.7.0"):
        from pyiceberg.io.pyarrow import data_file_statistics_from_parquet_metadata

        statistics = data_file_statistics_from_parquet_metadata(
            parquet_metadata=metadata,
            stats_columns=compute_statistics_plan(schema, properties),
            parquet_column_mapping=parquet_path_to_id_mapping(schema),
        )

        # The footer carries no NaN count, so merge in the one counted while writing.
        serialized = statistics.to_serialized_dict()
        if nan_value_counts:
            serialized["nan_value_counts"] = {**(serialized.get("nan_value_counts") or {}), **nan_value_counts}

        if parse(pyiceberg.__version__) >= parse("0.10.0"):
            data_file = DataFile.from_args(**{**kwargs, **serialized})
        else:
            data_file = DataFile(**{**kwargs, **serialized})
    else:
        from pyiceberg.io.pyarrow import fill_parquet_file_metadata

        data_file = DataFile(**kwargs)

        fill_parquet_file_metadata(
            data_file=data_file,
            parquet_metadata=metadata,
            stats_columns=compute_statistics_plan(schema, properties),
            parquet_column_mapping=parquet_path_to_id_mapping(schema),
        )

    # Spec id lives on the manifest holding the file, not in the file's record,
    # so the constructor argument is ignored and it is set as an attribute here.
    data_file.spec_id = spec_id

    return data_file


class IcebergWriteVisitors:
    class FileVisitor:
        def __init__(self, parent: IcebergWriteVisitors, partition_record: IcebergRecord):
            self.parent = parent
            self.partition_record = partition_record

        def __call__(self, written_file: pads.WrittenFile) -> None:
            file_path = f"{self.parent.protocol}://{written_file.path}"
            data_file = make_iceberg_data_file(
                file_path,
                written_file.size,
                written_file.metadata,
                self.partition_record,
                self.parent.spec_id,
                self.parent.schema,
                self.parent.properties,
            )

            self.parent.data_files.append(data_file)

    def __init__(
        self,
        protocol: str,
        spec_id: int,
        schema: IcebergSchema,
        properties: IcebergTableProperties,
    ):
        self.data_files: list[DataFile] = []
        self.protocol = protocol
        self.spec_id = spec_id
        self.schema = schema
        self.properties = properties

    def visitor(self, partition_record: IcebergRecord) -> IcebergWriteVisitors.FileVisitor:
        return self.FileVisitor(self, partition_record)

    def to_metadata(self) -> MicroPartition:
        col_name = "data_file"
        if len(self.data_files) == 0:
            return MicroPartition.empty(_get_schema_from_dict({col_name: DataType.python()}))
        return MicroPartition.from_pydict({col_name: self.data_files})


def make_iceberg_record(partition_values: dict[str, Any] | None) -> IcebergRecord:
    import pyiceberg
    from packaging.version import parse
    from pyiceberg.typedef import Record as IcebergRecord

    if partition_values:
        if parse(pyiceberg.__version__) >= parse("0.10.0"):
            return IcebergRecord(*[to_partition_representation(v) for v in partition_values.values()])

        return IcebergRecord(**{k: to_partition_representation(v) for k, v in partition_values.items()})
    else:
        return IcebergRecord()


def partitioned_table_to_iceberg_iter(
    partitioned: PartitionedTable, root_path: str, schema: pa.Schema
) -> Iterator[tuple[pa.Table, str, IcebergRecord]]:
    partition_values = partitioned.partition_values()

    if partition_values:
        partition_strings = partitioned.partition_values_str()
        assert partition_strings is not None

        for table, part_vals, part_strs in zip(
            partitioned.partitions(),
            partition_values.to_pylist(),
            partition_strings.to_pylist(),
        ):
            part_record = make_iceberg_record(part_vals)
            part_path = partition_strings_to_path(root_path, part_strs, partition_null_fallback="null")

            arrow_table = coerce_pyarrow_table_to_schema(table.to_arrow(), schema)

            yield arrow_table, part_path, part_record
    else:
        arrow_table = coerce_pyarrow_table_to_schema(partitioned.table.to_arrow(), schema)

        yield arrow_table, root_path, make_iceberg_record(None)
