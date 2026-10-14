use std::{fmt::Write, path::PathBuf, sync::Arc};

use arrow_array::{
    Array, ArrayAccessor, ArrayRef, RecordBatch as ArrowRecordBatch, builder::LargeStringBuilder,
    cast::AsArray, types::Date32Type,
};
use arrow_json::{
    Encoder, EncoderFactory, EncoderOptions, LineDelimitedWriter, WriterBuilder,
    writer::{LineDelimited, NullableEncoder},
};
use arrow_schema::{
    ArrowError, DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema, TimeUnit,
};
use base64::{Engine, engine::general_purpose::STANDARD};
use chrono::{DateTime, NaiveDate, format::StrftimeItems};
use chrono_tz::Tz;
use common_error::{DaftError, DaftResult};
use daft_core::prelude::*;
use daft_io::{IOConfig, SourceType, parse_url, utils::ObjectPath};
use daft_logical_plan::sink_info::JsonFormatOption;
use daft_micropartition::MicroPartition;
use daft_recordbatch::RecordBatch;

use crate::{
    AsyncFileWriter,
    batch_file_writer::BatchFileWriter,
    storage_backend::{FileStorageBackend, ObjectStorageBackend, StorageBackend},
    utils::{build_filename, build_filename_single},
};

type JsonFinishFn<B> =
    Arc<dyn Fn(LineDelimitedWriter<<B as StorageBackend>::Writer>) -> DaftResult<()> + Send + Sync>;

/// Safely format a NaiveDate using strftime
fn safe_date_format(date: &NaiveDate, format: &str) -> Result<String, ArrowError> {
    let items = StrftimeItems::new(format);
    let delayed = date.format_with_items(items);
    let mut result = String::new();
    write!(result, "{}", delayed)
        .map_err(|_| ArrowError::ComputeError(format!("Invalid date format string: {}", format)))?;
    Ok(result)
}

/// Safely format a DateTime using strftime
fn safe_datetime_format<TzType: chrono::TimeZone>(
    dt: &DateTime<TzType>,
    format: &str,
) -> Result<String, ArrowError>
where
    TzType::Offset: std::fmt::Display,
{
    let items = StrftimeItems::new(format);
    let delayed = dt.format_with_items(items);
    let mut result = String::new();
    write!(result, "{}", delayed).map_err(|_| {
        ArrowError::ComputeError(format!("Invalid timestamp format string: {}", format))
    })?;
    Ok(result)
}

/// Whether JSON can hold a column of `field`'s type: every type but an extension
/// type (a tensor, an image, a Python object, …), at any depth.
fn json_can_hold(field: &Arc<arrow_schema::Field>) -> bool {
    if field.extension_type_name().is_some() {
        return false;
    }

    match field.data_type() {
        arrow_schema::DataType::List(inner)
        | arrow_schema::DataType::FixedSizeList(inner, _)
        | arrow_schema::DataType::LargeList(inner) => json_can_hold(inner),
        arrow_schema::DataType::Struct(fields) => fields.iter().all(json_can_hold),
        arrow_schema::DataType::Union(fields, _) => {
            fields.iter().all(|(_, field)| json_can_hold(field))
        }
        arrow_schema::DataType::Map(inner, _) => json_can_hold(inner),
        _ => true,
    }
}

/// The first column of `file_schema` JSON cannot hold, if any.
///
/// # Errors
/// Returns an error if the schema cannot be converted to Arrow's.
pub(crate) fn column_json_cannot_hold(file_schema: &SchemaRef) -> DaftResult<Option<Field>> {
    let arrow_schema = file_schema.to_arrow()?;
    Ok(arrow_schema
        .fields()
        .iter()
        .position(|field| !json_can_hold(field))
        .map(|index| file_schema[index].clone()))
}

/// Writes binary values (`Binary`, `LargeBinary` and `FixedSizeBinary`, at any
/// depth) as base64 text: RFC 4648, standard alphabet, padded. JSON has no bytes,
/// and base64 is the form JSON readers most widely decode as bytes. Every other
/// type is left to the default encoders.
#[derive(Debug)]
struct Base64Binary;

impl EncoderFactory for Base64Binary {
    fn make_default_encoder<'a>(
        &self,
        _field: &'a arrow_schema::FieldRef,
        array: &'a dyn Array,
        _options: &'a EncoderOptions,
    ) -> Result<Option<NullableEncoder<'a>>, ArrowError> {
        let encoder: Box<dyn Encoder + 'a> = match array.data_type() {
            ArrowDataType::Binary => Box::new(Base64Encoder(array.as_binary::<i32>())),
            ArrowDataType::LargeBinary => Box::new(Base64Encoder(array.as_binary::<i64>())),
            ArrowDataType::FixedSizeBinary(_) => {
                Box::new(Base64Encoder(array.as_fixed_size_binary()))
            }
            _ => return Ok(None),
        };
        Ok(Some(NullableEncoder::new(encoder, array.nulls().cloned())))
    }
}

/// Encodes each value of a binary array as a quoted base64 string. Base64's
/// characters need no escaping inside a JSON string.
struct Base64Encoder<A>(A);

impl<'a, A: ArrayAccessor<Item = &'a [u8]>> Encoder for Base64Encoder<A> {
    fn encode(&mut self, idx: usize, out: &mut Vec<u8>) {
        let bytes = self.0.value(idx);
        // `None` only for an encoding longer than memory can hold.
        let encoded = base64::encoded_len(bytes.len(), true).unwrap_or(0);
        out.push(b'"');
        let start = out.len();
        out.resize(start + encoded, 0);
        // The buffer was just sized to the encoding, which is all that can fail.
        let _ = STANDARD.encode_slice(bytes, &mut out[start..]);
        out.push(b'"');
    }
}

/// Creates a writer of newline-delimited JSON under `root_dir`, or, with
/// `single_file`, of one file at exactly `root_dir`, replacing a local file there
/// when `overwrite_single_file_target`.
pub(crate) fn create_native_json_writer(
    root_dir: &str,
    file_idx: usize,
    partition_values: Option<&RecordBatch>,
    io_config: Option<IOConfig>,
    json_option: JsonFormatOption,
    single_file: bool,
    overwrite_single_file_target: bool,
) -> DaftResult<Box<dyn AsyncFileWriter<Input = MicroPartition, Result = Option<RecordBatch>>>> {
    // Parse the root directory and add partition values if present.
    let (source_type, root_dir) = parse_url(root_dir)?;
    let filename = if single_file {
        build_filename_single(&source_type, root_dir.as_ref())?
    } else {
        build_filename(
            &source_type,
            root_dir.as_ref(),
            partition_values,
            file_idx,
            "json",
        )?
    };
    match source_type {
        SourceType::File => {
            let storage_backend = FileStorageBackend::default();
            Ok(Box::new(
                make_json_writer(
                    filename,
                    partition_values.cloned(),
                    storage_backend,
                    json_option,
                )
                .with_overwrite_existing_target(overwrite_single_file_target),
            ))
        }
        source if source.supports_native_writer() => {
            let ObjectPath { scheme, .. } = daft_io::utils::parse_object_url(root_dir.as_ref())?;
            let io_config = io_config.ok_or_else(|| {
                DaftError::InternalError("IO config is required for object writes".to_string())
            })?;
            let storage_backend = ObjectStorageBackend::new(scheme, io_config);
            Ok(Box::new(make_json_writer(
                filename,
                partition_values.cloned(),
                storage_backend,
                json_option,
            )))
        }
        _ => Err(DaftError::ValueError(format!(
            "Unsupported source type: {:?}",
            source_type
        ))),
    }
}

fn make_json_writer<B: StorageBackend + Send + Sync>(
    filename: PathBuf,
    partition_values: Option<RecordBatch>,
    storage_backend: B,
    json_option: JsonFormatOption,
) -> BatchFileWriter<B, LineDelimitedWriter<B::Writer>> {
    let ignore_null_fields = json_option.ignore_null_fields;
    let date_format = json_option.date_format.clone();
    let timestamp_format = json_option.timestamp_format;
    let builder = Arc::new(move |backend: B::Writer| {
        let mut builder = WriterBuilder::new().with_encoder_factory(Arc::new(Base64Binary));

        if let Some(ignore_null_fields) = ignore_null_fields {
            builder = builder.with_explicit_nulls(!ignore_null_fields);
        }

        builder.build::<_, LineDelimited>(backend)
    });
    let write_fn = Arc::new(
        move |writer: &mut LineDelimitedWriter<B::Writer>, batches: &[arrow_array::RecordBatch]| {
            let date_fmt = date_format.clone();
            let timestamp_fmt = timestamp_format.clone();

            fn transform_batch(
                batch: ArrowRecordBatch,
                date_fmt: Option<&String>,
                timestamp_fmt: Option<&String>,
            ) -> Result<ArrowRecordBatch, ArrowError> {
                // If no custom formatting is needed, return the batch as-is
                if date_fmt.is_none() && timestamp_fmt.is_none() {
                    return Ok(batch);
                }

                let schema = batch.schema();
                let mut new_fields: Vec<ArrowField> = Vec::with_capacity(schema.fields().len());
                let mut new_cols: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());

                for (i, field_arc) in schema.fields().iter().enumerate() {
                    let field = field_arc.as_ref();
                    match field.data_type() {
                        ArrowDataType::Date32 => {
                            if let Some(fmt) = date_fmt {
                                let arr = batch.column(i).as_primitive::<Date32Type>();
                                let mut builder = LargeStringBuilder::with_capacity(arr.len(), 0);
                                for idx in 0..arr.len() {
                                    if arr.is_null(idx) {
                                        builder.append_null();
                                    } else {
                                        let days = arr.value(idx);
                                        let date = NaiveDate::from_ymd_opt(1970, 1, 1)
                                            .unwrap()
                                            .checked_add_signed(chrono::Duration::days(days as i64))
                                            .ok_or_else(|| {
                                                ArrowError::ComputeError("Invalid date".to_string())
                                            })?;
                                        builder.append_value(safe_date_format(&date, fmt)?);
                                    }
                                }
                                new_cols.push(Arc::new(builder.finish()) as ArrayRef);
                                new_fields.push(ArrowField::new(
                                    field.name(),
                                    ArrowDataType::LargeUtf8,
                                    field.is_nullable(),
                                ));
                            } else {
                                new_fields.push(field.clone());
                                new_cols.push(batch.column(i).clone());
                            }
                        }
                        // Note: Date64 is not handled here because Daft converts Date64 to
                        // Timestamp[ms] upon ingestion. Any Date64 data will arrive as Timestamp.
                        ArrowDataType::Timestamp(_time_unit, tz_opt) => {
                            if let Some(fmt) = timestamp_fmt {
                                let arr = batch.column(i);
                                let mut builder = LargeStringBuilder::with_capacity(arr.len(), 0);

                                // Parse timezone if present - return error if parsing fails
                                let parsed_tz: Option<Tz> = if let Some(tz_str) = tz_opt.as_ref() {
                                    Some(tz_str.parse::<Tz>().map_err(|_| {
                                        ArrowError::ComputeError(format!(
                                            "Invalid timezone: {}",
                                            tz_str
                                        ))
                                    })?)
                                } else {
                                    None
                                };

                                for idx in 0..arr.len() {
                                    if arr.is_null(idx) {
                                        builder.append_null();
                                    } else {
                                        // Get UTC timestamp
                                        let utc_dt = match arr.data_type() {
                                            ArrowDataType::Timestamp(TimeUnit::Second, _) => {
                                                let ts_arr = arr.as_primitive::<arrow_array::types::TimestampSecondType>();
                                                DateTime::from_timestamp(ts_arr.value(idx), 0)
                                            }
                                            ArrowDataType::Timestamp(TimeUnit::Millisecond, _) => {
                                                let ts_arr = arr.as_primitive::<arrow_array::types::TimestampMillisecondType>();
                                                DateTime::from_timestamp_millis(ts_arr.value(idx))
                                            }
                                            ArrowDataType::Timestamp(TimeUnit::Microsecond, _) => {
                                                let ts_arr = arr.as_primitive::<arrow_array::types::TimestampMicrosecondType>();
                                                DateTime::from_timestamp_micros(ts_arr.value(idx))
                                            }
                                            ArrowDataType::Timestamp(TimeUnit::Nanosecond, _) => {
                                                let ts_arr = arr.as_primitive::<arrow_array::types::TimestampNanosecondType>();
                                                Some(DateTime::from_timestamp_nanos(
                                                    ts_arr.value(idx),
                                                ))
                                            }
                                            _ => unreachable!(),
                                        };

                                        let utc_dt = utc_dt.ok_or_else(|| {
                                            ArrowError::ComputeError(
                                                "Invalid timestamp".to_string(),
                                            )
                                        })?;

                                        // Convert to target timezone if specified
                                        let formatted = if let Some(tz) = parsed_tz {
                                            let tz_dt = utc_dt.with_timezone(&tz);
                                            safe_datetime_format(&tz_dt, fmt)?
                                        } else {
                                            safe_datetime_format(&utc_dt, fmt)?
                                        };

                                        builder.append_value(formatted);
                                    }
                                }
                                new_cols.push(Arc::new(builder.finish()) as ArrayRef);
                                new_fields.push(ArrowField::new(
                                    field.name(),
                                    ArrowDataType::LargeUtf8,
                                    field.is_nullable(),
                                ));
                            } else {
                                new_fields.push(field.clone());
                                new_cols.push(batch.column(i).clone());
                            }
                        }
                        _ => {
                            new_fields.push(field.clone());
                            new_cols.push(batch.column(i).clone());
                        }
                    }
                }
                let new_schema = Arc::new(ArrowSchema::new(new_fields));
                ArrowRecordBatch::try_new(new_schema, new_cols)
            }

            let transformed_batches: Vec<ArrowRecordBatch> = batches
                .iter()
                .map(|rb| {
                    transform_batch(rb.clone(), date_fmt.as_ref(), timestamp_fmt.as_ref())
                        .map_err(|e| DaftError::ComputeError(e.to_string()))
                })
                .collect::<DaftResult<Vec<_>>>()?;
            let refs: Vec<&arrow_array::RecordBatch> = transformed_batches.iter().collect();
            writer.write_batches(&refs)?;
            Ok(())
        },
    );
    let finish_fn: Option<JsonFinishFn<B>> =
        Some(Arc::new(|mut writer: LineDelimitedWriter<B::Writer>| {
            writer.finish()?;
            Ok(())
        }));
    BatchFileWriter::new(
        filename,
        partition_values,
        storage_backend,
        0.5,
        builder,
        write_fn,
        finish_fn,
    )
}
