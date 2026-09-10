use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::Arc,
};

use bytes::{Bytes, BytesMut};
use common_runtime::get_io_runtime;
use daft_core::datatypes::Field;
use daft_io::{GetRange, IOClient, IOStatsRef};
use snafu::ResultExt;

use crate::{Error, ParquetMetadataSnafu, task_err};

const FOOTER_SIZE: usize = 8;

/// Parquet key-value metadata key for the embedded Arrow schema hint written by
/// pyarrow/arrow-cpp. See `parquet-format-thrift` KeyValue.
const ARROW_SCHEMA_KEY: &str = "ARROW:schema";

// ---------------------------------------------------------------------------
// Arrow-rs field ID mapping
// ---------------------------------------------------------------------------

/// Extract the optional field ID from a parquet `BasicTypeInfo`.
fn get_field_id(info: &parquet::schema::types::BasicTypeInfo) -> Option<i32> {
    info.has_id().then(|| info.id())
}

/// Index of column name to field ID for files whose schema carries no field IDs.
///
/// A name is present only when exactly one field claims it, so an ambiguous
/// name is reported as unresolvable rather than bound to the wrong column.
type NameIndex<'a> = HashMap<&'a str, i32>;

/// Field metadata key carrying the names a table's name mapping assigns to a field.
///
/// The names are one per line. A file registered before a column was renamed
/// still carries the old name, and the mapping records that the old name
/// refers to this field.
pub const NAME_MAPPING_METADATA_KEY: &str = "iceberg.name-mapping";

/// Build the unambiguous name index for a field ID mapping.
///
/// Every field is indexed under its current name and under each name the
/// table's declared name mapping assigns to it. A name that two different
/// fields claim is left out.
fn build_name_index(field_id_mapping: &BTreeMap<i32, Field>) -> NameIndex<'_> {
    let mut ambiguous: HashSet<&str> = HashSet::new();
    let mut index: NameIndex<'_> = HashMap::with_capacity(field_id_mapping.len());
    for (field_id, field) in field_id_mapping {
        let aliases = field
            .metadata
            .get(NAME_MAPPING_METADATA_KEY)
            .map(|joined| {
                joined
                    .lines()
                    .filter(|name| !name.is_empty())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let current: &str = &field.name;
        for name in std::iter::once(current).chain(aliases) {
            match index.insert(name, *field_id) {
                Some(previous) if previous != *field_id => {
                    ambiguous.insert(name);
                }
                _ => {}
            }
        }
    }
    for name in ambiguous {
        index.remove(name);
    }
    index
}

/// Report whether any node in the schema declares a field ID.
///
/// A data file registered rather than written through the table carries no
/// field IDs at all, which is the case the name index serves.
fn schema_declares_field_ids(root: &parquet::schema::types::Type) -> bool {
    fn walk(tp: &parquet::schema::types::Type) -> bool {
        if tp.get_basic_info().has_id() {
            return true;
        }
        match tp {
            parquet::schema::types::Type::GroupType { fields, .. } => {
                fields.iter().any(|f| walk(f))
            }
            parquet::schema::types::Type::PrimitiveType { .. } => false,
        }
    }
    match root {
        parquet::schema::types::Type::GroupType { fields, .. } => fields.iter().any(|f| walk(f)),
        parquet::schema::types::Type::PrimitiveType { .. } => false,
    }
}

/// Resolve a node's field ID, falling back to its name when the file declares none.
fn resolve_field_id(
    info: &parquet::schema::types::BasicTypeInfo,
    name_index: Option<&NameIndex<'_>>,
) -> Option<i32> {
    get_field_id(info).or_else(|| name_index.and_then(|index| index.get(info.name()).copied()))
}

/// Rewrite children of a group type per the field_id_mapping.
///
/// When the parent has a logical type (LIST/MAP), intermediate children without
/// a field ID are preserved (with their own children recursed). Otherwise
/// (plain struct), unmapped children are dropped.
fn rewrite_children(
    fields: &[Arc<parquet::schema::types::Type>],
    has_logical_type: bool,
    field_id_mapping: &BTreeMap<i32, Field>,
    name_index: Option<&NameIndex<'_>>,
) -> Vec<Arc<parquet::schema::types::Type>> {
    fields
        .iter()
        .filter_map(|child| {
            if has_logical_type {
                // LIST/MAP intermediate nodes may lack field IDs but still
                // contain mapped descendants (e.g. struct fields inside a list).
                Some(Arc::new(
                    rewrite_arrowrs_type_with_field_ids(child, field_id_mapping, name_index)
                        .unwrap_or_else(|| {
                            recurse_children_only(child, field_id_mapping, name_index)
                        }),
                ))
            } else {
                // Plain struct: drop children not in mapping.
                rewrite_arrowrs_type_with_field_ids(child, field_id_mapping, name_index)
                    .map(Arc::new)
            }
        })
        .collect()
}

/// Recursively rewrite an arrow-rs `Type`, renaming fields per the mapping and
/// dropping unmapped struct children.  Returns `None` if this field should be
/// dropped entirely (no field ID, or ID not in the mapping).
fn rewrite_arrowrs_type_with_field_ids(
    tp: &parquet::schema::types::Type,
    field_id_mapping: &BTreeMap<i32, Field>,
    name_index: Option<&NameIndex<'_>>,
) -> Option<parquet::schema::types::Type> {
    use parquet::schema::types::Type;

    let info = tp.get_basic_info();
    let field_id = resolve_field_id(info, name_index)?;
    let mapped_field = field_id_mapping.get(&field_id)?;

    match tp {
        Type::PrimitiveType {
            physical_type,
            type_length,
            scale,
            precision,
            ..
        } => {
            let new_type = Type::primitive_type_builder(&mapped_field.name, *physical_type)
                .with_repetition(info.repetition())
                .with_converted_type(info.converted_type())
                .with_logical_type(info.logical_type_ref().cloned())
                .with_length(*type_length)
                .with_precision(*precision)
                .with_scale(*scale)
                .with_id(Some(field_id))
                .build()
                .expect("rebuilding primitive type with same attributes should not fail");
            Some(new_type)
        }
        Type::GroupType { fields, .. } => {
            let new_children = rewrite_children(
                fields,
                info.logical_type_ref().is_some(),
                field_id_mapping,
                name_index,
            );
            let new_type = Type::group_type_builder(&mapped_field.name)
                .with_repetition(info.repetition())
                .with_converted_type(info.converted_type())
                .with_logical_type(info.logical_type_ref().cloned())
                .with_fields(new_children)
                .with_id(Some(field_id))
                .build()
                .expect("rebuilding group type with same attributes should not fail");
            Some(new_type)
        }
    }
}

/// Rebuild a group node preserving its original name/attributes, but still
/// recursing into its children to rename/filter them per the field_id_mapping.
/// For primitive nodes, returns an unchanged clone.
fn recurse_children_only(
    tp: &parquet::schema::types::Type,
    field_id_mapping: &BTreeMap<i32, Field>,
    name_index: Option<&NameIndex<'_>>,
) -> parquet::schema::types::Type {
    use parquet::schema::types::Type;
    match tp {
        Type::PrimitiveType { .. } => tp.clone(),
        Type::GroupType { fields, .. } => {
            let info = tp.get_basic_info();
            let new_children = rewrite_children(
                fields,
                info.logical_type_ref().is_some(),
                field_id_mapping,
                name_index,
            );
            Type::group_type_builder(info.name())
                .with_repetition(info.repetition())
                .with_converted_type(info.converted_type())
                .with_logical_type(info.logical_type_ref().cloned())
                .with_fields(new_children)
                .with_id(get_field_id(info))
                .build()
                .expect("rebuilding group type with same attributes should not fail")
        }
    }
}

/// Applies field_ids to an arrow-rs `ParquetMetaData`:
/// 1. Rename columns based on the `field_id_mapping`
/// 2. Drop columns without a field_id or without a corresponding mapping entry
///
/// A file whose schema declares no field IDs at all is resolved by column name
/// instead, using only unambiguous names; a file that cannot be fully resolved
/// that way is rejected rather than read as nulls.
///
/// # Errors
/// Returns an error when the file declares no field IDs and at least one of its
/// top-level columns cannot be matched to the schema by name.
pub(crate) fn apply_field_ids_to_arrowrs_parquet_metadata(
    metadata: Arc<parquet::file::metadata::ParquetMetaData>,
    field_id_mapping: &BTreeMap<i32, Field>,
    path: &str,
) -> crate::Result<Arc<parquet::file::metadata::ParquetMetaData>> {
    use parquet::{
        file::metadata::{ColumnChunkMetaData, ParquetMetaData, RowGroupMetaData},
        schema::types::{SchemaDescriptor, Type},
    };

    let old_schema = metadata.file_metadata().schema_descr();
    let old_root = old_schema.root_schema();

    // Only a file that declares no field IDs at all is resolved by name; one
    // that declares any is resolved by ID alone.
    let by_name =
        (!schema_declares_field_ids(old_root)).then(|| build_name_index(field_id_mapping));
    let name_index = by_name.as_ref();

    // 1. Rewrite the schema type tree: rename + filter by field_id_mapping
    let new_fields: Vec<_> = old_root
        .get_fields()
        .iter()
        .filter_map(|field| {
            rewrite_arrowrs_type_with_field_ids(field, field_id_mapping, name_index).map(Arc::new)
        })
        .collect();

    if name_index.is_some() && new_fields.len() != old_root.get_fields().len() {
        let resolved: HashSet<&str> = new_fields
            .iter()
            .filter_map(|f| resolve_field_id(f.get_basic_info(), name_index))
            .filter_map(|fid| field_id_mapping.get(&fid))
            .map(|f| -> &str { &f.name })
            .collect();
        let unresolved: Vec<&str> = old_root
            .get_fields()
            .iter()
            .map(|f| f.get_basic_info().name())
            .filter(|name| !resolved.contains(name))
            .collect();
        return Err(Error::ReaderInternal {
            path: path.to_string(),
            message: format!(
                "file declares no field ids and these columns could not be matched by name: {}. \
                 Reading it would silently produce nulls",
                unresolved.join(", ")
            ),
        });
    }

    let new_root = Type::group_type_builder(old_root.name())
        .with_fields(new_fields)
        .build()
        .with_context(|_| ParquetMetadataSnafu {
            path: path.to_string(),
        })?;
    let new_schema_descr = Arc::new(SchemaDescriptor::new(Arc::new(new_root)));

    // 2. Build field_id → new ColumnDescriptor mapping
    let field_id_to_col_descr: BTreeMap<i32, _> = new_schema_descr
        .columns()
        .iter()
        .filter_map(|col_descr| {
            let info = col_descr.self_type().get_basic_info();
            get_field_id(info).map(|fid| (fid, col_descr.clone()))
        })
        .collect();

    // 3. Rebuild row groups with filtered/renamed column descriptors
    let new_row_groups: Result<Vec<RowGroupMetaData>, _> = metadata
        .row_groups()
        .iter()
        .map(|rg| {
            let new_columns: Vec<ColumnChunkMetaData> = rg
                .columns()
                .iter()
                .filter_map(|col| {
                    let col_info = col.column_descr().self_type().get_basic_info();
                    let new_descr = resolve_field_id(col_info, name_index)
                        .and_then(|fid| field_id_to_col_descr.get(&fid))?;

                    // Rebuild ColumnChunkMetaData with new descriptor.
                    // No set_column_descr on the builder, so we construct from scratch.
                    let mut builder = ColumnChunkMetaData::builder(new_descr.clone())
                        .set_encodings_mask(*col.encodings_mask())
                        .set_num_values(col.num_values())
                        .set_compression(col.compression())
                        .set_data_page_offset(col.data_page_offset())
                        .set_total_compressed_size(col.compressed_size())
                        .set_total_uncompressed_size(col.uncompressed_size())
                        .set_index_page_offset(col.index_page_offset())
                        .set_dictionary_page_offset(col.dictionary_page_offset())
                        .set_bloom_filter_offset(col.bloom_filter_offset())
                        .set_bloom_filter_length(col.bloom_filter_length())
                        .set_offset_index_offset(col.offset_index_offset())
                        .set_offset_index_length(col.offset_index_length())
                        .set_column_index_offset(col.column_index_offset())
                        .set_column_index_length(col.column_index_length())
                        .set_unencoded_byte_array_data_bytes(col.unencoded_byte_array_data_bytes());
                    if let Some(stats) = col.statistics() {
                        builder = builder.set_statistics(stats.clone());
                    }
                    if let Some(path) = col.file_path() {
                        builder = builder.set_file_path(path.to_string());
                    }
                    Some(
                        builder
                            .build()
                            .expect("column chunk rebuild should not fail"),
                    )
                })
                .collect();

            let total_byte_size: i64 = new_columns.iter().map(|c| c.uncompressed_size()).sum();
            RowGroupMetaData::builder(new_schema_descr.clone())
                .set_num_rows(rg.num_rows())
                .set_total_byte_size(total_byte_size)
                .set_column_metadata(new_columns)
                .build()
                .with_context(|_| ParquetMetadataSnafu {
                    path: path.to_string(),
                })
        })
        .collect();

    // 4. Rebuild FileMetaData and ParquetMetaData. Strip the embedded ARROW:schema
    // hint: it still carries the original (physical) field names and disagrees with
    // the renamed parquet type tree, causing arrow-rs to fail with
    // `incompatible arrow schema, expected field named <physical> got <logical>`.
    // Dropping it forces arrow-rs to infer from the renamed parquet types.
    Ok(Arc::new(ParquetMetaData::new(
        rebuild_file_metadata(metadata.file_metadata(), new_schema_descr, true),
        new_row_groups?,
    )))
}

/// Rebuild a `FileMetaData` with a new schema descriptor (preserving everything else).
///
/// When `strip_arrow_schema` is true the embedded `ARROW:schema` key-value entry is
/// removed; this is required after a field-id rename where the embedded arrow schema
/// would otherwise disagree with the renamed parquet types.
fn rebuild_file_metadata(
    fm: &parquet::file::metadata::FileMetaData,
    new_schema_descr: Arc<parquet::schema::types::SchemaDescriptor>,
    strip_arrow_schema: bool,
) -> parquet::file::metadata::FileMetaData {
    let key_value_metadata = fm.key_value_metadata().map(|kv| {
        kv.iter()
            .filter(|entry| !strip_arrow_schema || entry.key != ARROW_SCHEMA_KEY)
            .cloned()
            .collect::<Vec<_>>()
    });
    parquet::file::metadata::FileMetaData::new(
        fm.version(),
        fm.num_rows(),
        fm.created_by().map(str::to_string),
        key_value_metadata,
        new_schema_descr,
        fm.column_orders().cloned(),
    )
}

/// Strip STRING/UTF8 logical types from BYTE_ARRAY columns in parquet metadata.
///
/// This makes arrow-rs infer `Binary` instead of `Utf8` for string columns,
/// which avoids UTF-8 validation during decode. Used to implement
/// `StringEncoding::Raw` which reads string columns as raw bytes.
pub(crate) fn strip_string_types_from_parquet_metadata(
    metadata: Arc<parquet::file::metadata::ParquetMetaData>,
    path: &str,
) -> crate::Result<Arc<parquet::file::metadata::ParquetMetaData>> {
    use parquet::{
        file::metadata::{ParquetMetaData, RowGroupMetaData},
        schema::types::{SchemaDescriptor, Type},
    };

    let old_schema = metadata.file_metadata().schema_descr();
    let old_root = old_schema.root_schema();

    let new_fields: Vec<_> = old_root
        .get_fields()
        .iter()
        .map(|field| Arc::new(strip_string_type(field)))
        .collect();

    let new_root = Type::group_type_builder(old_root.name())
        .with_fields(new_fields)
        .build()
        .with_context(|_| ParquetMetadataSnafu {
            path: path.to_string(),
        })?;
    let new_schema_descr = Arc::new(SchemaDescriptor::new(Arc::new(new_root)));

    // Rebuild row groups with the new schema descriptor.
    let new_row_groups: Result<Vec<RowGroupMetaData>, _> = metadata
        .row_groups()
        .iter()
        .map(|rg| {
            RowGroupMetaData::builder(new_schema_descr.clone())
                .set_num_rows(rg.num_rows())
                .set_total_byte_size(rg.total_byte_size())
                .set_column_metadata(rg.columns().to_vec())
                .build()
                .with_context(|_| ParquetMetadataSnafu {
                    path: path.to_string(),
                })
        })
        .collect();

    Ok(Arc::new(ParquetMetaData::new(
        rebuild_file_metadata(metadata.file_metadata(), new_schema_descr, false),
        new_row_groups?,
    )))
}

/// Recursively strip STRING/UTF8 annotations from a parquet type.
fn strip_string_type(tp: &parquet::schema::types::Type) -> parquet::schema::types::Type {
    use parquet::{
        basic::{ConvertedType, LogicalType},
        schema::types::Type,
    };

    match tp {
        Type::PrimitiveType {
            physical_type,
            type_length,
            scale,
            precision,
            ..
        } => {
            let info = tp.get_basic_info();
            let is_string = matches!(info.logical_type_ref(), Some(LogicalType::String))
                || info.converted_type() == ConvertedType::UTF8;

            if is_string {
                // Rebuild without String/UTF8 annotations so arrow-rs infers Binary.
                Type::primitive_type_builder(info.name(), *physical_type)
                    .with_repetition(info.repetition())
                    .with_length(*type_length)
                    .with_precision(*precision)
                    .with_scale(*scale)
                    .with_id(get_field_id(info))
                    .build()
                    .expect("rebuilding primitive type should not fail")
            } else {
                tp.clone()
            }
        }
        Type::GroupType { fields, .. } => {
            let info = tp.get_basic_info();
            let new_children: Vec<_> = fields
                .iter()
                .map(|child| Arc::new(strip_string_type(child)))
                .collect();
            Type::group_type_builder(info.name())
                .with_repetition(info.repetition())
                .with_converted_type(info.converted_type())
                .with_logical_type(info.logical_type_ref().cloned())
                .with_fields(new_children)
                .with_id(get_field_id(info))
                .build()
                .expect("rebuilding group type should not fail")
        }
    }
}

fn validate_footer_magic(uri: &str, buffer: &[u8]) -> super::Result<()> {
    const PARQUET_MAGIC: [u8; 4] = [b'P', b'A', b'R', b'1'];

    if buffer.len() < FOOTER_SIZE {
        return Err(Error::FileTooSmall {
            path: uri.into(),
            file_size: buffer.len(),
        });
    }

    if buffer[buffer.len() - 4..] != PARQUET_MAGIC {
        return Err(Error::InvalidParquetFile {
            path: uri.into(),
            footer: buffer[buffer.len() - 4..].into(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared footer I/O
// ---------------------------------------------------------------------------

/// Fetches raw parquet footer bytes from a URI, handling suffix range fallback
/// and two-pass reads for large footers.
///
/// Returns `(footer_bytes, remaining_offset)` where `remaining_offset` is the
/// byte offset within `footer_bytes` where the thrift metadata starts.
async fn fetch_parquet_footer_bytes(
    uri: &str,
    file_size: Option<usize>,
    io_client: Arc<IOClient>,
    io_stats: Option<IOStatsRef>,
    default_footer_read_size: Option<usize>,
) -> super::Result<(Bytes, usize)> {
    async fn fetch_data(
        io_client: Arc<IOClient>,
        uri: &str,
        range: GetRange,
        io_stats: Option<IOStatsRef>,
    ) -> daft_io::Result<Bytes> {
        io_client
            .single_url_get(uri.into(), Some(range), io_stats)
            .await?
            .bytes()
            .await
    }

    let file_size_opt: Option<usize> = if file_size.is_none() && !io_client.support_suffix_range() {
        // suffix range unsupported, get object length for future I/O
        let size = io_client
            .single_url_get_size(uri.into(), io_stats.clone())
            .await?;
        Some(size)
    } else {
        file_size
    };

    // Check the minimum value of file size if provided
    if let Some(size) = file_size_opt
        && size < 12
    {
        return Err(Error::FileTooSmall {
            path: uri.into(),
            file_size: size,
        });
    }

    /// The number of bytes read at the end of the parquet file on first read
    const DEFAULT_FOOTER_READ_SIZE: usize = 128 * 1024;
    let footer_read_size = default_footer_read_size
        .unwrap_or(DEFAULT_FOOTER_READ_SIZE)
        .max(FOOTER_SIZE);
    let range = match file_size_opt {
        None => GetRange::Suffix(footer_read_size),
        Some(size) => {
            let default_end_len = std::cmp::min(footer_read_size, size);
            let start = size - default_end_len;
            (start..size).into()
        }
    };
    let mut data = fetch_data(io_client.clone(), uri, range, io_stats.clone()).await?;
    let buffer = data.as_ref();
    validate_footer_magic(uri, buffer)?;

    let metadata_size = i32::from_le_bytes(
        buffer[buffer.len() - 8..buffer.len() - 4]
            .try_into()
            .unwrap(),
    );
    let footer_len = FOOTER_SIZE + metadata_size as usize;
    if let Some(size) = file_size_opt
        && size < footer_len
    {
        return Err(Error::InvalidParquetFooterSize {
            path: uri.into(),
            footer_size: footer_len,
            file_size: size,
        });
    }

    let remaining = if footer_len <= buffer.len() {
        // the whole metadata is in the bytes we already read
        buffer.len() - footer_len
    } else {
        // the end of file read by default is not long enough, read more bytes of metadata.
        data = match file_size_opt {
            None => fetch_data(io_client, uri, GetRange::Suffix(footer_len), io_stats).await?,
            Some(size) => {
                let range =
                    (size.saturating_sub(footer_len)..size.saturating_sub(buffer.len())).into();
                let new_data = fetch_data(io_client, uri, range, io_stats).await?;

                let mut buffer = BytesMut::with_capacity(new_data.len() + data.len());
                buffer.extend_from_slice(&new_data);
                buffer.extend_from_slice(&data);
                buffer.freeze()
            }
        };
        0
    };

    let buffer = data.as_ref();
    validate_footer_magic(uri, buffer)?;

    Ok((data, remaining))
}

// ---------------------------------------------------------------------------
// Arrow-rs metadata deserialization
// ---------------------------------------------------------------------------

/// Read parquet metadata using arrow-rs deserialization.
///
/// Returns `Arc<parquet::file::metadata::ParquetMetaData>`.
pub(crate) async fn read_parquet_metadata(
    uri: &str,
    file_size: Option<usize>,
    io_client: Arc<IOClient>,
    io_stats: Option<IOStatsRef>,
    field_id_mapping: Option<Arc<BTreeMap<i32, Field>>>,
    default_footer_read_size: Option<usize>,
) -> super::Result<Arc<parquet::file::metadata::ParquetMetaData>> {
    let (data, remaining) = fetch_parquet_footer_bytes(
        uri,
        file_size,
        io_client,
        io_stats,
        default_footer_read_size,
    )
    .await?;

    let metadata = get_io_runtime(true)
        .spawn_blocking(move || {
            let thrift_bytes = &data.as_ref()[remaining..data.len() - FOOTER_SIZE];
            parquet::file::metadata::ParquetMetaDataReader::decode_metadata(thrift_bytes)
        })
        .await
        .map_err(task_err(uri))?
        .map_err(|e| Error::ParquetMetadata {
            path: uri.to_string(),
            source: e,
        })?;

    let metadata = Arc::new(metadata);

    if let Some(field_id_mapping) = field_id_mapping {
        apply_field_ids_to_arrowrs_parquet_metadata(metadata, field_id_mapping.as_ref(), uri)
    } else {
        Ok(metadata)
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Arc};

    use common_error::DaftResult;
    use daft_core::{datatypes::Field, prelude::DataType};
    use daft_io::{IOClient, IOConfig};
    use parquet::{
        basic::{Repetition, Type as PhysicalType},
        schema::types::Type,
    };

    use super::{build_name_index, read_parquet_metadata, schema_declares_field_ids};

    fn leaf(name: &str, field_id: Option<i32>) -> Type {
        Type::primitive_type_builder(name, PhysicalType::INT64)
            .with_repetition(Repetition::OPTIONAL)
            .with_id(field_id)
            .build()
            .unwrap()
    }

    fn root(children: Vec<Type>) -> Type {
        Type::group_type_builder("schema")
            .with_fields(children.into_iter().map(Arc::new).collect())
            .build()
            .unwrap()
    }

    fn field(name: &str) -> Field {
        Field::new(name, DataType::Int64)
    }

    #[test]
    fn name_index_maps_each_unique_name_to_its_field_id() {
        let mapping = BTreeMap::from([(1, field("id")), (2, field("label"))]);
        let index = build_name_index(&mapping);
        assert_eq!(index.get("id"), Some(&1));
        assert_eq!(index.get("label"), Some(&2));
    }

    #[test]
    fn name_index_omits_names_claimed_by_more_than_one_field() {
        // Two structs may each hold a child called `name`; resolving that by
        // name could bind the wrong column, so it must not be resolvable.
        let mapping = BTreeMap::from([(1, field("id")), (4, field("name")), (7, field("name"))]);
        let index = build_name_index(&mapping);
        assert_eq!(index.get("id"), Some(&1));
        assert!(!index.contains_key("name"));
    }

    #[test]
    fn name_index_of_an_empty_mapping_is_empty() {
        assert!(build_name_index(&BTreeMap::new()).is_empty());
    }

    #[test]
    fn schema_without_any_field_id_is_detected() {
        let schema = root(vec![leaf("id", None), leaf("label", None)]);
        assert!(!schema_declares_field_ids(&schema));
    }

    #[test]
    fn schema_with_any_field_id_is_detected() {
        let schema = root(vec![leaf("id", Some(1)), leaf("label", None)]);
        assert!(schema_declares_field_ids(&schema));
    }

    #[test]
    fn field_ids_nested_below_the_root_are_detected() {
        let inner = Type::group_type_builder("who")
            .with_fields(vec![Arc::new(leaf("name", Some(4)))])
            .build()
            .unwrap();
        assert!(schema_declares_field_ids(&root(vec![inner])));
    }

    #[test]
    fn an_empty_schema_declares_no_field_ids() {
        assert!(!schema_declares_field_ids(&root(vec![])));
    }
    use crate::Error;

    #[tokio::test]
    async fn test_parquet_metadata_from_s3() -> DaftResult<()> {
        let file = "s3://daft-public-data/test_fixtures/parquet-dev/mvp.parquet";
        let size = 9882;

        let mut io_config = IOConfig::default();
        io_config.s3.anonymous = true;
        let io_client = Arc::new(IOClient::new(io_config.into())?);

        // Read metadata with actual file size.
        let metadata =
            read_parquet_metadata(file, Some(size), io_client.clone(), None, None, None).await?;
        assert_eq!(metadata.file_metadata().num_rows(), 100);

        // Read metadata without a file size.
        let metadata =
            read_parquet_metadata(file, None, io_client.clone(), None, None, None).await?;
        assert_eq!(metadata.file_metadata().num_rows(), 100);

        // Overwrite the default footer read size which less than footer length but without a file size.
        let metadata =
            read_parquet_metadata(file, None, io_client.clone(), None, None, Some(500)).await?;
        assert_eq!(metadata.file_metadata().num_rows(), 100);

        // Overwrite the default footer read size which less than footer length and a file size.
        let metadata =
            read_parquet_metadata(file, Some(size), io_client.clone(), None, None, Some(500))
                .await?;
        assert_eq!(metadata.file_metadata().num_rows(), 100);

        // Overwrite the default footer read size less than 8 bytes.
        let metadata =
            read_parquet_metadata(file, None, io_client.clone(), None, None, Some(5)).await?;
        assert_eq!(metadata.file_metadata().num_rows(), 100);

        // Test with invalid file size, assume file size is 10 bytes.
        let result =
            read_parquet_metadata(file, Some(10), io_client.clone(), None, None, None).await;
        assert!(matches!(result, Err(Error::FileTooSmall { .. })));

        // Test with invalid footer size, assume file size is 1260 bytes.
        let result = read_parquet_metadata(file, Some(1260), io_client, None, None, None).await;
        assert!(matches!(result, Err(Error::InvalidParquetFile { .. })));

        Ok(())
    }
}
