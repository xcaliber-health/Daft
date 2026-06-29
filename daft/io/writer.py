from __future__ import annotations

import math
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from daft.datatype import DataType
from daft.dependencies import pa, pacsv, pafs, pq
from daft.filesystem import (
    _resolve_paths_and_filesystem,
    get_protocol_from_path,
)
from daft.io.common import _get_schema_from_dict
from daft.io.delta_lake.delta_lake_write import (
    make_deltalake_add_action,
    make_deltalake_fs,
    sanitize_table_for_deltalake,
)
from daft.io.iceberg.iceberg_write import (
    coerce_pyarrow_table_to_schema,
    make_iceberg_data_file,
    make_iceberg_record,
)
from daft.recordbatch.partitioning import (
    partition_strings_to_path,
    partition_values_to_str_mapping,
)
from daft.recordbatch.recordbatch import RecordBatch
from daft.series import Series

if TYPE_CHECKING:
    from pyiceberg.schema import Schema as IcebergSchema
    from pyiceberg.table import TableProperties as IcebergTableProperties

    from daft.daft import IOConfig
    from daft.recordbatch.micropartition import MicroPartition


class FileWriterBase(ABC):
    def __init__(
        self,
        root_dir: str,
        file_idx: int,
        file_format: str,
        partition_values: RecordBatch | None = None,
        compression: str | None = None,
        io_config: IOConfig | None = None,
        version: int | None = None,
        default_partition_fallback: str | None = None,
    ):
        self.resolved_path, self.fs = self.resolve_path_and_fs(root_dir, io_config=io_config)
        self.protocol = get_protocol_from_path(root_dir)
        # Determine locality from the resolved filesystem type so that custom
        # schemes aliased to file:// still trigger local directory creation.
        is_local_fs = isinstance(self.fs, pafs.LocalFileSystem)

        self.file_name = (
            f"{uuid.uuid4()}-{file_idx}.{file_format}"
            if version is None
            else f"{version}-{uuid.uuid4()}-{file_idx}.{file_format}"
        )
        self.partition_values = partition_values
        if self.partition_values is not None:
            self.partition_strings = {
                key: next(iter(values))
                for key, values in partition_values_to_str_mapping(self.partition_values).items()
            }
            self.dir_path = partition_strings_to_path(
                self.resolved_path,
                self.partition_strings,
                (
                    default_partition_fallback
                    if default_partition_fallback is not None
                    else "__HIVE_DEFAULT_PARTITION__"
                ),
            )
        else:
            self.partition_strings = {}
            self.dir_path = f"{self.resolved_path}"

        self.full_path = f"{self.dir_path}/{self.file_name}"
        if is_local_fs:
            self.fs.create_dir(self.dir_path, recursive=True)

        # Normalize to a string so downstream code (e.g. _resolve_column_compression)
        # can rely on self.compression always being a valid PyArrow codec name, never None.
        self.compression = compression if compression is not None else "none"
        self.position = 0

    def resolve_path_and_fs(self, root_dir: str, io_config: IOConfig | None = None) -> tuple[str, pafs.FileSystem]:
        [resolved_path], fs = _resolve_paths_and_filesystem(root_dir, io_config=io_config)
        return resolved_path, fs

    @abstractmethod
    def write(self, table: MicroPartition) -> int:
        """Write data to the file using the appropriate writer.

        Args:
            table: MicroPartition containing the data to be written.

        Returns:
            int: The number of bytes written to the file.
        """

    @abstractmethod
    def close(self) -> RecordBatch:
        """Close the writer and return metadata about the written file. Write should not be called after close.

        Returns:
            RecordBatch containing metadata about the written file, including path and partition values.
        """


class ParquetFileWriter(FileWriterBase):
    def __init__(
        self,
        root_dir: str,
        file_idx: int,
        partition_values: RecordBatch | None = None,
        compression: str | None = None,
        io_config: IOConfig | None = None,
        version: int | None = None,
        default_partition_fallback: str | None = None,
        metadata_collector: list[pq.FileMetaData] | None = None,
        column_compression: dict[str, str] | None = None,
    ):
        super().__init__(
            root_dir=root_dir,
            file_idx=file_idx,
            file_format="parquet",
            partition_values=partition_values,
            compression=compression,
            io_config=io_config,
            version=version,
            default_partition_fallback=default_partition_fallback,
        )
        self.is_closed = False
        self.current_writer: pq.ParquetWriter | None = None
        self.metadata_collector: list[pq.FileMetaData] | None = metadata_collector
        self.column_compression = column_compression

    def _create_writer(self, schema: pa.Schema) -> pq.ParquetWriter:
        opts = {}
        if self.metadata_collector is not None:
            opts["metadata_collector"] = self.metadata_collector
        compression: str | dict[str, str]
        if self.column_compression:
            compression = self._resolve_column_compression(schema)
        else:
            compression = self.compression
        return pq.ParquetWriter(
            self.full_path,
            schema,
            compression=compression,
            use_compliant_nested_type=False,
            filesystem=self.fs,
            # When using Arrow 8, it defaults to parquet version 1.
            # This hits a known bug where Arrow cannot correctly write u32 values in Parquet files:
            # https://issues.apache.org/jira/browse/ARROW-12201
            # The fix is to always use at least Parquet version 2.
            version="2.6",
            **opts,
        )

    def write(self, table: MicroPartition) -> int:
        assert not self.is_closed, "Cannot write to a closed ParquetFileWriter"
        if len(table) == 0:
            return 0
        if self.current_writer is None:
            schema = table.schema().to_pyarrow_schema()
            self.current_writer = self._create_writer(schema)
        self.current_writer.write_table(table.to_arrow(), row_group_size=len(table))

        current_position = self.current_writer.file_handle.tell()
        bytes_written = current_position - self.position
        self.position = current_position
        return bytes_written

    def close(self) -> RecordBatch:
        self.is_closed = True
        metadata: dict[str, Series] = {"path": Series.from_pylist([self.full_path])}
        if self.partition_values is not None:
            for column in self.partition_values.columns():
                metadata[column.name()] = column
        if self.current_writer is None:
            return RecordBatch.from_pydict(metadata).slice(0, 0)
        self.current_writer.close()
        return RecordBatch.from_pydict(metadata)

    def _resolve_column_compression(
        self,
        schema: pa.Schema,
    ) -> dict[str, str]:
        """Build a leaf-path -> codec dict for ``pq.ParquetWriter``.

        PyArrow requires every leaf column be listed when a dict is passed.
        Recursion handles top-level struct nesting only; overrides for leaves
        nested inside list/large_list/map types are not supported on the
        PyArrow fallback path (the native writer is unaffected). Any override
        key that does not match a discovered leaf raises ``ValueError`` rather
        than being silently dropped.
        """
        assert self.column_compression is not None
        overrides = self.column_compression
        leaves: list[str] = []

        def collect(field: pa.Field, prefix: str) -> None:
            path = f"{prefix}.{field.name}" if prefix else field.name
            if pa.types.is_struct(field.type):
                for i in range(field.type.num_fields):
                    collect(field.type.field(i), path)
            else:
                leaves.append(path)

        for field in schema:
            collect(field, "")

        leaf_set = set(leaves)
        unmatched = sorted(k for k in overrides if k not in leaf_set)
        if unmatched:
            raise ValueError(
                "column_compression keys do not match any leaf column: "
                f"{unmatched}. The PyArrow writer fallback does not support "
                "overrides for leaves nested inside list/large_list/map types; "
                "apply the override at the top-level column instead."
            )
        return {leaf: overrides.get(leaf, self.compression) for leaf in leaves}


class CSVFileWriter(FileWriterBase):
    def __init__(
        self,
        root_dir: str,
        file_idx: int,
        partition_values: RecordBatch | None = None,
        io_config: IOConfig | None = None,
        delimiter: str | None = None,
        include_header: bool | None = True,
        date_format: str | None = None,
        timestamp_format: str | None = None,
    ):
        super().__init__(
            root_dir=root_dir,
            file_idx=file_idx,
            file_format="csv",
            partition_values=partition_values,
            io_config=io_config,
        )
        self.file_handle = None
        self.current_writer: pacsv.CSVWriter | None = None
        self.is_closed = False
        self.delimiter = delimiter
        self.include_header = True if include_header is None else include_header
        self.date_format = date_format
        self.timestamp_format = timestamp_format

    def _create_writer(self, schema: pa.Schema) -> pacsv.CSVWriter:
        self.file_handle = self.fs.open_output_stream(self.full_path)
        write_options = pacsv.WriteOptions(delimiter=self.delimiter or ",", include_header=self.include_header)
        return pacsv.CSVWriter(self.file_handle, schema, write_options=write_options)

    def _apply_custom_formatting(self, arrow_table: pa.Table) -> pa.Table:
        """Apply custom date/timestamp formatting to the table before writing."""
        if self.date_format is None and self.timestamp_format is None:
            return arrow_table

        import pyarrow.compute as pc

        new_columns = []
        new_fields = []

        for i, field in enumerate(arrow_table.schema):
            column = arrow_table.column(i)

            if pa.types.is_date(field.type) and self.date_format is not None:
                # Convert date to string with custom format
                formatted = pc.strftime(column, format=self.date_format)
                new_columns.append(formatted)
                new_fields.append(pa.field(field.name, pa.large_string(), nullable=field.nullable))
            elif pa.types.is_timestamp(field.type) and self.timestamp_format is not None:
                # Convert timestamp to string with custom format
                formatted = pc.strftime(column, format=self.timestamp_format)
                new_columns.append(formatted)
                new_fields.append(pa.field(field.name, pa.large_string(), nullable=field.nullable))
            else:
                new_columns.append(column)
                new_fields.append(field)

        new_schema = pa.schema(new_fields)
        return pa.table(dict(zip(arrow_table.column_names, new_columns)), schema=new_schema)

    def write(self, table: MicroPartition) -> int:
        assert not self.is_closed, "Cannot write to a closed CSVFileWriter"
        if len(table) == 0:
            return 0
        arrow_table = table.to_arrow()

        # Apply custom date/timestamp formatting if specified
        formatted_table = self._apply_custom_formatting(arrow_table)

        if self.current_writer is None:
            self.current_writer = self._create_writer(formatted_table.schema)
        self.current_writer.write_table(formatted_table)

        assert self.file_handle is not None  # We should have created the file handle in _create_writer
        current_position = self.file_handle.tell()
        bytes_written = current_position - self.position
        self.position = current_position
        return bytes_written

    def close(self) -> RecordBatch:
        self.is_closed = True
        metadata: dict[str, Series] = {"path": Series.from_pylist([self.full_path])}
        if self.partition_values is not None:
            for column in self.partition_values.columns():
                metadata[column.name()] = column
        if self.current_writer is None:
            return RecordBatch.from_pydict(metadata).slice(0, 0)
        self.current_writer.close()
        return RecordBatch.from_pydict(metadata)


_ICEBERG_COMPRESSION_TO_PARQUET = {
    "uncompressed": "none",
    "none": "none",
    "snappy": "snappy",
    "gzip": "gzip",
    "lz4": "lz4",
    "brotli": "brotli",
    "zstd": "zstd",
}

# Bloom-filter table-property contract.
#
# A bloom filter accelerates equality and membership lookups on high-cardinality
# columns (identifiers, UUIDs, session/trace keys) where min/max statistics cannot
# rule a value out. It does not help range predicates, low-cardinality columns
# (a dictionary serves those better), partition-transform columns, or columns kept
# sorted (min/max already prunes those). Each filter adds the byte budget per
# column per row group to the file, so enabling a few well-chosen columns is the
# rule of thumb; size the budget to the column's distinct-value count.
#
# Recognized properties:
#   - per-column enable flag prefix (value ``true`` enables the column)
#   - per-column probability hint prefix
#   - per-column expected distinct-value count prefix
#   - shared byte budget key
# Sizing: when a column's expected distinct-value count is set, the filter is
# sized from that count and the probability, but never larger than the byte
# budget; when it is unset, the byte budget sets the size and the probability is a
# hint only. Filters are written only through the native streaming writer; to add
# them to existing data, rewrite it through compaction.
_BLOOM_ENABLED_PREFIX = "write.parquet.bloom-filter-enabled.column."
_BLOOM_FPP_PREFIX = "write.parquet.bloom-filter-fpp.column."
_BLOOM_NDV_PREFIX = "write.parquet.bloom-filter-ndv.column."
_BLOOM_MAX_BYTES_KEY = "write.parquet.bloom-filter-max-bytes"

# Defaults applied when a column requests a filter without an explicit probability,
# and when no byte budget is configured.
_DEFAULT_BLOOM_FPP = 0.01
_DEFAULT_BLOOM_MAX_BYTES = 1048576


def _distinct_values_for_byte_budget(fpp: float, max_bytes: int) -> int:
    """Return the largest distinct-value count whose filter fits a byte budget.

    Inverts the split-block sizing relation ``m = -8 * n / ln(1 - fpp**(1/8))``
    (``n`` distinct values, ``m`` bits) so that a filter sized for the returned
    count and ``fpp`` occupies at most ``max_bytes`` bytes. This expresses a
    byte-budget configuration as the distinct-value count expected by writers that
    size their filters from a distinct-value count and a probability hint, keeping
    the on-disk filter size equivalent across writers that share the sizing relation.

    Parameters
    ----------
    fpp : float
        Target false-positive probability, in the open interval ``(0, 1)``.
    max_bytes : int
        Maximum bitset size, in bytes.

    Returns:
    -------
    int
        Largest distinct-value count whose filter fits the budget (at least 1).
    """
    ndv = math.floor(-max_bytes * math.log(1.0 - fpp ** (1.0 / 8.0)))
    return max(1, ndv)


def _iceberg_bloom_filter_options(
    properties: dict[str, str],
    valid_columns: set[str] | None = None,
) -> dict[str, dict[str, float | int]] | None:
    """Build per-column bloom-filter parameters from table properties.

    Collects the columns whose per-column enable flag is set, then resolves each
    column's distinct-value count and probability hint. When a column's expected
    distinct-value count is given, it sizes the filter together with the
    probability, but is capped at the count the byte budget allows so the filter
    never exceeds the budget. When it is unset, the count is derived from the byte
    budget, so the budget sets the size and the probability is a hint only.

    Parameters
    ----------
    properties : dict of str to str
        Table properties, including the bloom-filter enable flags, per-column
        probability hints, per-column expected distinct-value counts, and the
        shared byte budget.
    valid_columns : set of str, optional
        Column paths present in the written schema. When given, enable flags for
        any other column are ignored, so a stale or misspelled flag is a no-op
        rather than an error.

    Returns:
    -------
    dict or None
        Mapping from column path to its ``ndv`` and ``fpp`` parameters, or ``None``
        when no column requests a filter.

    Raises:
    ------
    ValueError
        If a column's probability hint is not within the open interval ``(0, 1)``,
        or its expected distinct-value count is not positive.
    """
    enabled = [
        key[len(_BLOOM_ENABLED_PREFIX) :]
        for key, value in properties.items()
        if key.startswith(_BLOOM_ENABLED_PREFIX) and str(value).strip().lower() == "true"
    ]
    enabled = [column for column in enabled if column]
    if valid_columns is not None:
        enabled = [column for column in enabled if column in valid_columns]
    if not enabled:
        return None

    max_bytes = int(properties.get(_BLOOM_MAX_BYTES_KEY, _DEFAULT_BLOOM_MAX_BYTES))
    options: dict[str, dict[str, float | int]] = {}
    for column in enabled:
        fpp = float(properties.get(f"{_BLOOM_FPP_PREFIX}{column}", _DEFAULT_BLOOM_FPP))
        if not 0.0 < fpp < 1.0:
            raise ValueError(
                f"Bloom-filter probability for column {column!r} must be in the open interval (0, 1); got {fpp}."
            )
        budget_ndv = _distinct_values_for_byte_budget(fpp, max_bytes)
        ndv_raw = properties.get(f"{_BLOOM_NDV_PREFIX}{column}")
        if ndv_raw is None:
            ndv = budget_ndv
        else:
            requested_ndv = int(ndv_raw)
            if requested_ndv <= 0:
                raise ValueError(
                    f"Bloom-filter distinct-value count for column {column!r} must be positive; got {requested_ndv}."
                )
            # Honor the requested count, but never size beyond the byte budget.
            ndv = min(requested_ndv, budget_ndv)
        options[column] = {"ndv": ndv, "fpp": fpp}
    return options


def _resolve_iceberg_writer_options(
    properties: dict[str, str] | None,
    valid_columns: set[str] | None = None,
) -> dict[str, Any]:
    """Translate Iceberg ``write.*`` table properties to ParquetWriter kwargs."""
    props = properties or {}
    fmt = (props.get("write.format-default") or "parquet").lower()
    if fmt != "parquet":
        raise ValueError(f"IcebergWriter only supports parquet; got write.format-default={fmt!r}")

    codec_raw = (props.get("write.parquet.compression-codec") or "zstd").lower()
    codec = _ICEBERG_COMPRESSION_TO_PARQUET.get(codec_raw, codec_raw)

    out: dict[str, Any] = {"compression": codec}
    level = props.get("write.parquet.compression-level")
    if level is not None:
        out["compression_level"] = int(level)
    row_group_bytes = props.get("write.parquet.row-group-size-bytes")
    if row_group_bytes is not None:
        out["row_group_byte_size"] = int(row_group_bytes)
    page_size = props.get("write.parquet.page-size-bytes")
    if page_size is not None:
        out["data_page_size"] = int(page_size)
    dict_size = props.get("write.parquet.dict-size-bytes")
    if dict_size is not None:
        out["dictionary_pagesize_limit"] = int(dict_size)
    bloom_options = _iceberg_bloom_filter_options(props, valid_columns)
    if bloom_options is not None:
        out["bloom_filter_options"] = bloom_options
    return out


class IcebergWriter(ParquetFileWriter):
    def __init__(
        self,
        root_dir: str,
        file_idx: int,
        schema: IcebergSchema,
        properties: IcebergTableProperties,
        partition_spec_id: int,
        partition_values: RecordBatch | None = None,
        io_config: IOConfig | None = None,
    ):
        from pyiceberg.io.pyarrow import schema_to_pyarrow

        # Bloom-filter enable flags are honored only for columns that exist in the
        # schema being written, so a stale or misspelled flag is ignored rather
        # than rejected by the writer.
        valid_columns = set(schema.column_names) if schema is not None else None
        self._iceberg_writer_opts = _resolve_iceberg_writer_options(dict(properties or {}), valid_columns)
        super().__init__(
            root_dir=root_dir,
            file_idx=file_idx,
            partition_values=partition_values,
            compression=self._iceberg_writer_opts["compression"],
            io_config=io_config,
            version=None,
            default_partition_fallback="null",
            metadata_collector=[],
        )

        self.part_record = make_iceberg_record(
            partition_values.to_pylist()[0] if partition_values is not None else None
        )
        self.iceberg_schema = schema
        self.file_schema = schema_to_pyarrow(schema)
        self.partition_spec_id = partition_spec_id
        self.properties = properties

    def _create_writer(self, schema: pa.Schema) -> pq.ParquetWriter:
        opts: dict[str, Any] = {}
        if self.metadata_collector is not None:
            opts["metadata_collector"] = self.metadata_collector
        for k in ("compression_level", "data_page_size", "dictionary_pagesize_limit", "bloom_filter_options"):
            v = self._iceberg_writer_opts.get(k)
            if v is not None:
                opts[k] = v
        return pq.ParquetWriter(
            self.full_path,
            schema,
            compression=self.compression,
            use_compliant_nested_type=False,
            filesystem=self.fs,
            version="2.6",
            **opts,
        )

    def write(self, table: MicroPartition) -> int:
        assert not self.is_closed, "Cannot write to a closed IcebergFileWriter"
        if len(table) == 0:
            return 0
        if self.current_writer is None:
            self.current_writer = self._create_writer(self.file_schema)
        casted = coerce_pyarrow_table_to_schema(table.to_arrow(), self.file_schema)
        row_group_byte_cap = self._iceberg_writer_opts.get("row_group_byte_size")
        if row_group_byte_cap is not None and len(table) > 0:
            approx_bytes_per_row = max(1, casted.nbytes // max(1, len(table)))
            row_group_size = max(1, int(row_group_byte_cap // approx_bytes_per_row))
            self.current_writer.write_table(casted, row_group_size=row_group_size)
        else:
            self.current_writer.write_table(casted)

        current_position = self.current_writer.file_handle.tell()
        bytes_written = current_position - self.position
        self.position = current_position
        return bytes_written

    def close(self) -> RecordBatch:
        self.is_closed = True
        if self.current_writer is None:
            return RecordBatch.empty(_get_schema_from_dict({"data_file": DataType.python()}))
        self.current_writer.close()

        assert self.metadata_collector is not None
        metadata = self.metadata_collector[0]
        size = self.fs.get_file_info(self.full_path).size
        path_with_protocol = f"{self.protocol}://{self.full_path}"
        data_file = make_iceberg_data_file(
            path_with_protocol,
            size,
            metadata,
            self.part_record,
            self.partition_spec_id,
            self.iceberg_schema,
            self.properties,
        )
        return RecordBatch.from_pydict({"data_file": [data_file]})


class DeltalakeWriter(ParquetFileWriter):
    def __init__(
        self,
        root_dir: str,
        file_idx: int,
        version: int,
        large_dtypes: bool,
        partition_values: RecordBatch | None = None,
        io_config: IOConfig | None = None,
    ):
        super().__init__(
            root_dir=root_dir,
            file_idx=file_idx,
            partition_values=partition_values,
            compression=None,
            io_config=io_config,
            version=version,
            default_partition_fallback=None,
            metadata_collector=[],
        )

        self.large_dtypes = large_dtypes

    def resolve_path_and_fs(self, root_dir: str, io_config: IOConfig | None = None) -> tuple[str, pafs.PyFileSystem]:
        return "", make_deltalake_fs(root_dir, io_config)

    def write(self, table: MicroPartition) -> int:
        assert not self.is_closed, "Cannot write to a closed DeltalakeFileWriter"
        if len(table) == 0:
            return 0

        converted_arrow_table = sanitize_table_for_deltalake(
            table,
            self.large_dtypes,
            (self.partition_values.schema().column_names() if self.partition_values is not None else None),
        )
        if self.current_writer is None:
            self.current_writer = self._create_writer(converted_arrow_table.schema)
        self.current_writer.write_table(converted_arrow_table)

        current_position = self.current_writer.file_handle.tell()
        bytes_written = current_position - self.position
        self.position = current_position
        return bytes_written

    def close(self) -> RecordBatch:
        self.is_closed = True
        if self.current_writer is None:
            return RecordBatch.empty(_get_schema_from_dict({"add_action": DataType.python()}))
        self.current_writer.close()

        assert self.metadata_collector is not None
        metadata = self.metadata_collector[0]
        size = self.fs.get_file_info(self.full_path).size
        add_action = make_deltalake_add_action(
            path=self.full_path,
            metadata=metadata,
            size=size,
            partition_values=self.partition_strings,
        )

        return RecordBatch.from_pydict({"add_action": [add_action]})
