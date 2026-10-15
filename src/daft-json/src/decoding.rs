use std::{borrow::Borrow, fmt::Write};

use arrow::{
    array::{
        ArrayRef, OffsetSizeTrait,
        builder::{
            ArrayBuilder, BooleanBuilder, FixedSizeBinaryBuilder, FixedSizeListBuilder,
            GenericBinaryBuilder, GenericListBuilder, GenericStringBuilder, NullBuilder,
            PrimitiveBuilder, StructBuilder,
        },
    },
    datatypes::{
        ArrowPrimitiveType, DataType, Date32Type, Date64Type, DurationMicrosecondType,
        DurationMillisecondType, DurationNanosecondType, DurationSecondType, Field, Float16Type,
        Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, IntervalUnit,
        IntervalYearMonthType, Schema, Time32MillisecondType, Time32SecondType,
        Time64MicrosecondType, Time64NanosecondType, TimeUnit, TimestampMicrosecondType,
        TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type,
        UInt16Type, UInt32Type, UInt64Type,
    },
    error::ArrowError,
    temporal_conversions,
};
use chrono::{Datelike, Timelike};
use daft_decoding::deserialize::{
    deserialize_datetime, deserialize_naive_date, deserialize_naive_datetime,
};
use indexmap::IndexMap;
use num_traits::NumCast;
use simd_json::StaticNode;

use crate::deserializer::Value as BorrowedValue;
const JSON_NULL_VALUE: BorrowedValue = BorrowedValue::Static(StaticNode::Null);

/// Deserialize chunk of JSON records into a chunk of Arrow arrays.
pub fn deserialize_records<'a, A: Borrow<BorrowedValue<'a>>>(
    records: &[A],
    schema: &Schema,
) -> Result<Vec<ArrayRef>, ArrowError> {
    // Allocate mutable arrays.
    let fields: Vec<_> = schema.fields().iter().collect();
    let mut builders: IndexMap<&str, Box<dyn ArrayBuilder>> = fields
        .iter()
        .map(|f| Ok((f.name().as_str(), allocate_array(f, records.len())?)))
        .collect::<Result<_, ArrowError>>()?;
    for record in records {
        match record.borrow() {
            BorrowedValue::Object(record) => {
                for (i, (key, arr)) in builders.iter_mut().enumerate() {
                    let dtype = fields[i].data_type();
                    if let Some(value) = record.get(&**key) {
                        deserialize_into(arr, dtype, &[value]);
                    } else {
                        deserialize_into(arr, dtype, &[&JSON_NULL_VALUE]);
                    }
                }
            }
            _ => {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "Each line in a newline-delimited JSON file must be a JSON object, but got: {:?}",
                    record.borrow()
                )));
            }
        }
    }

    Ok(builders
        .into_values()
        .map(|mut builder| builder.finish())
        .collect())
}

/// Allocates a builder for a column of `f`'s type, sized for `length` rows.
///
/// # Errors
/// Returns an error for a type JSON is not read as, so that no value of it is
/// ever decoded.
pub fn allocate_array(f: &Field, length: usize) -> Result<Box<dyn ArrayBuilder>, ArrowError> {
    Ok(match f.data_type() {
        DataType::Null => Box::new(NullBuilder::new()),
        DataType::Int8 => Box::new(PrimitiveBuilder::<Int8Type>::with_capacity(length)),
        DataType::Int16 => Box::new(PrimitiveBuilder::<Int16Type>::with_capacity(length)),
        DataType::Int32 => Box::new(PrimitiveBuilder::<Int32Type>::with_capacity(length)),
        DataType::Date32 => Box::new(PrimitiveBuilder::<Date32Type>::with_capacity(length)),
        DataType::Time32(TimeUnit::Second) => {
            Box::new(PrimitiveBuilder::<Time32SecondType>::with_capacity(length))
        }
        DataType::Time32(TimeUnit::Millisecond) => Box::new(PrimitiveBuilder::<
            Time32MillisecondType,
        >::with_capacity(length)),
        DataType::Interval(IntervalUnit::YearMonth) => Box::new(PrimitiveBuilder::<
            IntervalYearMonthType,
        >::with_capacity(length)),
        DataType::Int64 => Box::new(PrimitiveBuilder::<Int64Type>::with_capacity(length)),
        DataType::Date64 => Box::new(PrimitiveBuilder::<Date64Type>::with_capacity(length)),
        DataType::Time64(TimeUnit::Microsecond) => Box::new(PrimitiveBuilder::<
            Time64MicrosecondType,
        >::with_capacity(length)),
        DataType::Time64(TimeUnit::Nanosecond) => Box::new(
            PrimitiveBuilder::<Time64NanosecondType>::with_capacity(length),
        ),
        DataType::Duration(tu) => match tu {
            TimeUnit::Second => Box::new(PrimitiveBuilder::<DurationSecondType>::with_capacity(
                length,
            )),
            TimeUnit::Millisecond => Box::new(
                PrimitiveBuilder::<DurationMillisecondType>::with_capacity(length),
            ),
            TimeUnit::Microsecond => Box::new(
                PrimitiveBuilder::<DurationMicrosecondType>::with_capacity(length),
            ),
            TimeUnit::Nanosecond => Box::new(
                PrimitiveBuilder::<DurationNanosecondType>::with_capacity(length),
            ),
        },
        dt @ DataType::Timestamp(tu, _) => match tu {
            TimeUnit::Second => Box::new(
                PrimitiveBuilder::<TimestampSecondType>::with_capacity(length)
                    .with_data_type(dt.clone()),
            ),
            TimeUnit::Millisecond => Box::new(
                PrimitiveBuilder::<TimestampMillisecondType>::with_capacity(length)
                    .with_data_type(dt.clone()),
            ),
            TimeUnit::Microsecond => Box::new(
                PrimitiveBuilder::<TimestampMicrosecondType>::with_capacity(length)
                    .with_data_type(dt.clone()),
            ),
            TimeUnit::Nanosecond => Box::new(
                PrimitiveBuilder::<TimestampNanosecondType>::with_capacity(length)
                    .with_data_type(dt.clone()),
            ),
        },
        DataType::UInt8 => Box::new(PrimitiveBuilder::<UInt8Type>::with_capacity(length)),
        DataType::UInt16 => Box::new(PrimitiveBuilder::<UInt16Type>::with_capacity(length)),
        DataType::UInt32 => Box::new(PrimitiveBuilder::<UInt32Type>::with_capacity(length)),
        DataType::UInt64 => Box::new(PrimitiveBuilder::<UInt64Type>::with_capacity(length)),
        DataType::Float16 => Box::new(PrimitiveBuilder::<Float16Type>::with_capacity(length)),
        DataType::Float32 => Box::new(PrimitiveBuilder::<Float32Type>::with_capacity(length)),
        DataType::Float64 => Box::new(PrimitiveBuilder::<Float64Type>::with_capacity(length)),
        DataType::Boolean => Box::new(BooleanBuilder::with_capacity(length)),
        DataType::Utf8 => Box::new(GenericStringBuilder::<i32>::with_capacity(length, 1024)),
        DataType::Binary => Box::new(GenericBinaryBuilder::<i32>::with_capacity(length, 1024)),
        DataType::LargeBinary => Box::new(GenericBinaryBuilder::<i64>::with_capacity(length, 1024)),
        DataType::FixedSizeBinary(size) => {
            Box::new(FixedSizeBinaryBuilder::with_capacity(length, *size))
        }
        DataType::LargeUtf8 => Box::new(GenericStringBuilder::<i64>::with_capacity(length, 1024)),
        DataType::FixedSizeList(inner, size) => {
            let inner_builder = allocate_array(inner.as_ref(), length)?;
            Box::new(FixedSizeListBuilder::with_capacity(
                inner_builder,
                *size,
                length,
            ))
        }
        DataType::List(inner) => {
            let inner_builder = allocate_array(inner.as_ref(), length)?;
            Box::new(
                GenericListBuilder::<i32, _>::with_capacity(inner_builder, length)
                    .with_field(inner.clone()),
            )
        }
        DataType::LargeList(inner) => {
            let inner_builder = allocate_array(inner.as_ref(), length)?;
            Box::new(
                GenericListBuilder::<i64, _>::with_capacity(inner_builder, length)
                    .with_field(inner.clone()),
            )
        }
        DataType::Struct(fields) => {
            // The struct's own builder allocates its fields; check each can be read.
            for field in fields {
                allocate_array(field, 0)?;
            }
            Box::new(StructBuilder::from_fields(fields.clone(), length))
        }
        dt => {
            return Err(ArrowError::NotYetImplemented(format!(
                "JSON cannot be read as {dt}, the type of the column {:?}",
                f.name()
            )));
        }
    })
}

/// Deserialize `rows` by extending them into the given `target`.
pub fn deserialize_into<'a, A: Borrow<BorrowedValue<'a>>>(
    target: &mut Box<dyn ArrayBuilder>,
    dtype: &DataType,
    rows: &[A],
) {
    match dtype {
        DataType::Null => {
            let target = target.as_any_mut().downcast_mut::<NullBuilder>().unwrap();
            for _ in 0..rows.len() {
                target.append_null();
            }
        }
        DataType::Boolean => {
            generic_deserialize_into(target, rows, deserialize_boolean_into);
        }
        DataType::Float32 => deserialize_primitive_into::<_, Float32Type>(target, rows),
        DataType::Float64 => deserialize_primitive_into::<_, Float64Type>(target, rows),
        DataType::Float16 => deserialize_primitive_into::<_, Float16Type>(target, rows),
        DataType::Int8 => deserialize_primitive_into::<_, Int8Type>(target, rows),
        DataType::Int16 => deserialize_primitive_into::<_, Int16Type>(target, rows),
        DataType::Int32 => deserialize_primitive_into::<_, Int32Type>(target, rows),
        DataType::Interval(IntervalUnit::YearMonth) => {
            deserialize_primitive_into::<_, IntervalYearMonthType>(target, rows);
        }
        DataType::Date32 => {
            deserialize_date_into::<_, Date32Type>(target, dtype, rows);
        }
        DataType::Time32(TimeUnit::Second) => {
            deserialize_date_into::<_, Time32SecondType>(target, dtype, rows);
        }
        DataType::Time32(TimeUnit::Millisecond) => {
            deserialize_date_into::<_, Time32MillisecondType>(target, dtype, rows);
        }
        DataType::Int64 => deserialize_primitive_into::<_, Int64Type>(target, rows),
        DataType::Duration(TimeUnit::Second) => {
            deserialize_duration_into::<_, DurationSecondType>(target, 1_000_000_000, rows);
        }
        DataType::Duration(TimeUnit::Millisecond) => {
            deserialize_duration_into::<_, DurationMillisecondType>(target, 1_000_000, rows);
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            deserialize_duration_into::<_, DurationMicrosecondType>(target, 1_000, rows);
        }
        DataType::Duration(TimeUnit::Nanosecond) => {
            deserialize_duration_into::<_, DurationNanosecondType>(target, 1, rows);
        }
        DataType::Binary => generic_deserialize_into::<_, GenericBinaryBuilder<i32>>(
            target,
            rows,
            deserialize_base64_into,
        ),
        DataType::LargeBinary => generic_deserialize_into::<_, GenericBinaryBuilder<i64>>(
            target,
            rows,
            deserialize_base64_into,
        ),
        DataType::FixedSizeBinary(size) => {
            deserialize_fixed_base64_into(target, *size, rows);
        }
        DataType::Timestamp(TimeUnit::Second, _) => {
            deserialize_datetime_into::<_, TimestampSecondType>(target, dtype, rows);
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            deserialize_datetime_into::<_, TimestampMillisecondType>(target, dtype, rows);
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            deserialize_datetime_into::<_, TimestampMicrosecondType>(target, dtype, rows);
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            deserialize_datetime_into::<_, TimestampNanosecondType>(target, dtype, rows);
        }
        DataType::Date64 => {
            deserialize_datetime_into::<_, Date64Type>(target, dtype, rows);
        }
        DataType::Time64(TimeUnit::Microsecond) => {
            deserialize_datetime_into::<_, Time64MicrosecondType>(target, dtype, rows);
        }
        DataType::Time64(TimeUnit::Nanosecond) => {
            deserialize_datetime_into::<_, Time64NanosecondType>(target, dtype, rows);
        }
        DataType::UInt8 => deserialize_primitive_into::<_, UInt8Type>(target, rows),
        DataType::UInt16 => deserialize_primitive_into::<_, UInt16Type>(target, rows),
        DataType::UInt32 => deserialize_primitive_into::<_, UInt32Type>(target, rows),
        DataType::UInt64 => deserialize_primitive_into::<_, UInt64Type>(target, rows),
        DataType::Utf8 => generic_deserialize_into::<_, GenericStringBuilder<i32>>(
            target,
            rows,
            deserialize_utf8_into,
        ),
        DataType::LargeUtf8 => generic_deserialize_into::<_, GenericStringBuilder<i64>>(
            target,
            rows,
            deserialize_utf8_into,
        ),
        DataType::FixedSizeList(inner, size) => {
            deserialize_fixed_size_list_into(target, inner.data_type(), *size, rows);
        }
        DataType::List(inner) => {
            deserialize_list_into::<i32, _>(target, inner.data_type(), rows);
        }
        DataType::LargeList(inner) => {
            deserialize_list_into::<i64, _>(target, inner.data_type(), rows);
        }
        DataType::Struct(fields) => {
            deserialize_struct_into(target, fields, rows);
        }
        // `allocate_array` refuses every other type, so no builder of one exists.
        dt => unreachable!("no JSON builder is allocated for {dt}"),
    }
}

fn deserialize_primitive_into<'a, A: Borrow<BorrowedValue<'a>>, T: ArrowPrimitiveType>(
    target: &mut Box<dyn ArrayBuilder>,
    rows: &[A],
) where
    T::Native: NumCast,
{
    let target = target
        .as_any_mut()
        .downcast_mut::<PrimitiveBuilder<T>>()
        .unwrap();

    for row in rows {
        let value = match row.borrow() {
            BorrowedValue::Static(StaticNode::I64(v)) => NumCast::from(*v),
            BorrowedValue::Static(StaticNode::U64(v)) => NumCast::from(*v),
            BorrowedValue::Static(StaticNode::F64(v)) => NumCast::from(*v),
            BorrowedValue::Static(StaticNode::Bool(v)) => NumCast::from(*v as u8),
            _ => None,
        };
        target.append_option(value);
    }
}

fn generic_deserialize_into<'a, A: Borrow<BorrowedValue<'a>>, M: 'static>(
    target: &mut Box<dyn ArrayBuilder>,
    rows: &[A],
    deserialize_into: fn(&mut M, &[A]),
) {
    deserialize_into(target.as_any_mut().downcast_mut::<M>().unwrap(), rows);
}

fn deserialize_utf8_into<'a, O: OffsetSizeTrait, A: Borrow<BorrowedValue<'a>>>(
    target: &mut GenericStringBuilder<O>,
    rows: &[A],
) {
    let mut scratch = String::new();

    for row in rows {
        match row.borrow() {
            BorrowedValue::String(v) => target.append_value(v.as_ref()),
            BorrowedValue::Static(StaticNode::Bool(v)) => {
                target.append_value(if *v { "true" } else { "false" });
            }
            BorrowedValue::Static(node) if !matches!(node, StaticNode::Null) => {
                write!(scratch, "{node}").unwrap();
                target.append_value(scratch.as_str());
                scratch.clear();
            }
            _ => target.append_null(),
        }
    }
}

/// Appends each row's bytes, decoded from base64 text (RFC 4648, standard
/// alphabet, padded), as the writer writes them. Anything else reads as null.
fn deserialize_base64_into<'a, O: OffsetSizeTrait, A: Borrow<BorrowedValue<'a>>>(
    target: &mut GenericBinaryBuilder<O>,
    rows: &[A],
) {
    for row in rows {
        match decode_base64(row.borrow()) {
            Some(bytes) => target.append_value(bytes),
            None => target.append_null(),
        }
    }
}

/// Appends each row's bytes as [`deserialize_base64_into`] does, into a column of
/// `size` bytes a value; bytes of another length read as null.
fn deserialize_fixed_base64_into<'a, A: Borrow<BorrowedValue<'a>>>(
    target: &mut Box<dyn ArrayBuilder>,
    size: i32,
    rows: &[A],
) {
    let target = target
        .as_any_mut()
        .downcast_mut::<FixedSizeBinaryBuilder>()
        .unwrap();
    for row in rows {
        let fits = decode_base64(row.borrow())
            .filter(|bytes| i32::try_from(bytes.len()).is_ok_and(|len| len == size));
        // Appending refuses only bytes of another length, which `fits` excludes.
        if fits.is_none_or(|bytes| target.append_value(bytes).is_err()) {
            target.append_null();
        }
    }
}

fn decode_base64(value: &BorrowedValue<'_>) -> Option<Vec<u8>> {
    use base64::Engine;

    match value {
        BorrowedValue::String(text) => base64::engine::general_purpose::STANDARD
            .decode(text.as_bytes())
            .ok(),
        _ => None,
    }
}

/// Appends each row's duration, a count of the column's unit or ISO 8601 text
/// (`PT3S`, `-PT1.5S`, `P1DT2H`) as the writer writes it, where `unit_nanos` is
/// the column's unit in nanoseconds. Text not exact in that unit, or not a
/// duration, reads as null.
fn deserialize_duration_into<
    'a,
    A: Borrow<BorrowedValue<'a>>,
    T: ArrowPrimitiveType<Native = i64>,
>(
    target: &mut Box<dyn ArrayBuilder>,
    unit_nanos: i128,
    rows: &[A],
) {
    let target = target
        .as_any_mut()
        .downcast_mut::<PrimitiveBuilder<T>>()
        .unwrap();
    for row in rows {
        let value = match row.borrow() {
            BorrowedValue::String(text) => parse_iso8601_duration(text)
                .filter(|nanos| nanos % unit_nanos == 0)
                .and_then(|nanos| i64::try_from(nanos / unit_nanos).ok()),
            BorrowedValue::Static(StaticNode::I64(v)) => Some(*v),
            BorrowedValue::Static(StaticNode::U64(v)) => i64::try_from(*v).ok(),
            BorrowedValue::Static(StaticNode::F64(v)) => NumCast::from(*v),
            _ => None,
        };
        target.append_option(value);
    }
}

/// Parses an ISO 8601 duration of days, hours, minutes and seconds into
/// nanoseconds: `[-]P[nD][T[nH][nM][n[.f]S]]`. Years, months and weeks name no
/// fixed length, so they are not durations here.
fn parse_iso8601_duration(text: &str) -> Option<i128> {
    let (negative, rest) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text),
    };
    let rest = rest.strip_prefix('P')?;
    let (date, time) = match rest.split_once('T') {
        Some((date, time)) => (date, Some(time)),
        None => (rest, None),
    };
    // A duration names at least one part: zero is `P0D`.
    if date.is_empty() && time.is_none() {
        return None;
    }
    let mut nanos: i128 = if date.is_empty() {
        0
    } else {
        date.strip_suffix('D')?
            .parse::<i128>()
            .ok()?
            .checked_mul(86_400_000_000_000)?
    };
    if let Some(mut time) = time {
        if time.is_empty() {
            return None;
        }
        for (designator, unit) in [('H', 3_600_000_000_000_i128), ('M', 60_000_000_000)] {
            if let Some((count, after)) = time.split_once(designator) {
                nanos = nanos.checked_add(count.parse::<i128>().ok()?.checked_mul(unit)?)?;
                time = after;
            }
        }
        if !time.is_empty() {
            let seconds = time.strip_suffix('S')?;
            let (whole, fraction) = seconds.split_once('.').unwrap_or((seconds, ""));
            if fraction.len() > 9 || !fraction.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            let fraction_nanos: i128 = format!("{fraction:0<9}").parse().ok()?;
            nanos = nanos
                .checked_add(whole.parse::<i128>().ok()?.checked_mul(1_000_000_000)?)?
                .checked_add(fraction_nanos)?;
        }
    }
    Some(if negative { -nanos } else { nanos })
}

fn deserialize_boolean_into<'a, A: Borrow<BorrowedValue<'a>>>(
    target: &mut BooleanBuilder,
    rows: &[A],
) {
    for row in rows {
        match row.borrow() {
            BorrowedValue::Static(StaticNode::Bool(v)) => target.append_value(*v),
            _ => target.append_null(),
        }
    }
}

fn get_factor_from_timeunit(time_unit: &TimeUnit) -> u32 {
    match time_unit {
        TimeUnit::Second => 1,
        TimeUnit::Millisecond => 1_000,
        TimeUnit::Microsecond => 1_000_000,
        TimeUnit::Nanosecond => 1_000_000_000,
    }
}

fn deserialize_date_into<'a, A: Borrow<BorrowedValue<'a>>, T: ArrowPrimitiveType<Native = i32>>(
    target: &mut Box<dyn ArrayBuilder>,
    dtype: &DataType,
    rows: &[A],
) {
    let target = target
        .as_any_mut()
        .downcast_mut::<PrimitiveBuilder<T>>()
        .unwrap();
    let mut last_fmt_idx = 0;

    for row in rows {
        let value = match row.borrow() {
            BorrowedValue::Static(StaticNode::I64(i)) => i32::try_from(*i).ok(),
            BorrowedValue::Static(StaticNode::U64(i)) => i32::try_from(*i).ok(),
            BorrowedValue::String(v) => match dtype {
                DataType::Time32(tu) => {
                    let factor = get_factor_from_timeunit(tu);
                    v.parse::<chrono::NaiveTime>().ok().map(|x| {
                        (x.hour() * 3_600 * factor
                            + x.minute() * 60 * factor
                            + x.second() * factor
                            + x.nanosecond() / (1_000_000_000 / factor))
                            as i32
                    })
                }
                DataType::Date32 => deserialize_naive_date(v, &mut last_fmt_idx)
                    .map(|x| x.num_days_from_ce() - (temporal_conversions::UNIX_EPOCH_DAY as i32)),
                _ => unreachable!(),
            },
            _ => None,
        };
        target.append_option(value);
    }
}

fn deserialize_datetime_into<
    'a,
    A: Borrow<BorrowedValue<'a>>,
    T: ArrowPrimitiveType<Native = i64>,
>(
    target: &mut Box<dyn ArrayBuilder>,
    dtype: &DataType,
    rows: &[A],
) {
    let target = target
        .as_any_mut()
        .downcast_mut::<PrimitiveBuilder<T>>()
        .unwrap();
    let mut last_fmt_idx = 0;

    for row in rows {
        let value = match row.borrow() {
            BorrowedValue::Static(StaticNode::I64(i)) => Some(*i),
            BorrowedValue::Static(StaticNode::U64(i)) => i64::try_from(*i).ok(),
            BorrowedValue::String(v) => match dtype {
                DataType::Time64(tu) => {
                    let factor = get_factor_from_timeunit(tu) as u64;
                    v.parse::<chrono::NaiveTime>().ok().map(|x| {
                        (x.hour() as u64 * 3_600 * factor
                            + x.minute() as u64 * 60 * factor
                            + x.second() as u64 * factor
                            + x.nanosecond() as u64 / (1_000_000_000 / factor))
                            as i64
                    })
                }
                DataType::Date64 => deserialize_naive_datetime(v, &mut last_fmt_idx)
                    .map(|x| x.and_utc().timestamp_millis()),
                DataType::Timestamp(tu, None) => deserialize_naive_datetime(v, &mut last_fmt_idx)
                    .and_then(|dt| match tu {
                        TimeUnit::Second => Some(dt.and_utc().timestamp()),
                        TimeUnit::Millisecond => Some(dt.and_utc().timestamp_millis()),
                        TimeUnit::Microsecond => Some(dt.and_utc().timestamp_micros()),
                        TimeUnit::Nanosecond => dt.and_utc().timestamp_nanos_opt(),
                    }),
                DataType::Timestamp(tu, Some(tz)) => {
                    let tz = if tz.as_ref() == "Z" { "UTC" } else { tz };
                    let tz = daft_schema::time_unit::parse_offset(tz).unwrap();
                    deserialize_datetime(v, &tz, &mut last_fmt_idx).and_then(|dt| match tu {
                        TimeUnit::Second => Some(dt.timestamp()),
                        TimeUnit::Millisecond => Some(dt.timestamp_millis()),
                        TimeUnit::Microsecond => Some(dt.timestamp_micros()),
                        TimeUnit::Nanosecond => dt.timestamp_nanos_opt(),
                    })
                }
                _ => unreachable!(),
            },
            _ => None,
        };
        target.append_option(value);
    }
}

fn deserialize_list_into<'a, O: OffsetSizeTrait, A: Borrow<BorrowedValue<'a>>>(
    target: &mut Box<dyn ArrayBuilder>,
    inner_dtype: &DataType,
    rows: &[A],
) {
    let target = target
        .as_any_mut()
        .downcast_mut::<GenericListBuilder<O, Box<dyn ArrayBuilder>>>()
        .unwrap();

    for row in rows {
        match row.borrow() {
            BorrowedValue::Array(values) => {
                deserialize_into(target.values(), inner_dtype, values);
                target.append(true);
            }
            _ => {
                target.append(false);
            }
        }
    }
}

fn deserialize_fixed_size_list_into<'a, A: Borrow<BorrowedValue<'a>>>(
    target: &mut Box<dyn ArrayBuilder>,
    inner_dtype: &DataType,
    size: i32,
    rows: &[A],
) {
    let target = target
        .as_any_mut()
        .downcast_mut::<FixedSizeListBuilder<Box<dyn ArrayBuilder>>>()
        .unwrap();

    let null_values: Vec<_> = std::iter::repeat_n(&JSON_NULL_VALUE, size as usize).collect();

    for row in rows {
        match row.borrow() {
            BorrowedValue::Array(value) => {
                if value.len() == size as usize {
                    deserialize_into(target.values(), inner_dtype, value);
                    target.append(true);
                } else {
                    // TODO(Clark): Return an error instead of dropping incorrectly sized lists.
                    // Push placeholder nulls to maintain alignment, then mark as null.
                    deserialize_into(target.values(), inner_dtype, &null_values);
                    target.append(false);
                }
            }
            _ => {
                // Push placeholder nulls to maintain alignment, then mark as null.
                deserialize_into(target.values(), inner_dtype, &null_values);
                target.append(false);
            }
        }
    }
}

fn deserialize_struct_into<'a, A: Borrow<BorrowedValue<'a>>>(
    target: &mut Box<dyn ArrayBuilder>,
    fields: &arrow::datatypes::Fields,
    rows: &[A],
) {
    let target = target.as_any_mut().downcast_mut::<StructBuilder>().unwrap();

    // Build a map from struct field name -> accumulated JSON values.
    let mut values: IndexMap<&str, Vec<&BorrowedValue<'a>>> = fields
        .iter()
        .map(|field| (field.name().as_str(), vec![]))
        .collect();

    for row in rows {
        match row.borrow() {
            BorrowedValue::Object(value) => {
                values.iter_mut().for_each(|(s, inner)| {
                    inner.push(value.get(*s).unwrap_or(&JSON_NULL_VALUE));
                });
                target.append(true);
            }
            _ => {
                values
                    .iter_mut()
                    .for_each(|(_, inner)| inner.push(&JSON_NULL_VALUE));
                target.append(false);
            }
        }
    }

    // Then deserialize each field's JSON values buffer to the appropriate Arrow array.
    //
    // Column ordering invariant - this assumes that values and target.field_builders_mut()
    // have aligned columns; we can assume this because:
    // - field_builders_mut() is guaranteed to have the same column ordering as the fields,
    // - values is an ordered map, whose ordering is tied to fields.
    values
        .into_values()
        .zip(fields.iter())
        .zip(target.field_builders_mut().iter_mut())
        .for_each(|((col_values, field), col_builder)| {
            deserialize_into(col_builder, field.data_type(), col_values.as_slice());
        });
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::parse_iso8601_duration;

    const SECOND: i128 = 1_000_000_000;

    #[rstest]
    #[case::seconds("PT3S", Some(3 * SECOND))]
    #[case::fraction("PT1.5S", Some(3 * SECOND / 2))]
    #[case::nanosecond("PT0.000000001S", Some(1))]
    #[case::negative("-PT3S", Some(-3 * SECOND))]
    #[case::zero("P0D", Some(0))]
    #[case::many_seconds("PT129600S", Some(129_600 * SECOND))]
    #[case::days_and_hours("P1DT2H", Some(26 * 3_600 * SECOND))]
    #[case::minutes("PT1M", Some(60 * SECOND))]
    #[case::every_part("P1DT1H1M1.25S", Some(90_061 * SECOND + SECOND / 4))]
    #[case::no_designator("3S", None)]
    #[case::nothing("P", None)]
    #[case::empty_time("PT", None)]
    #[case::years("P1Y", None)]
    #[case::finer_than_nanoseconds("PT1.0000000001S", None)]
    #[case::not_a_number("PTXS", None)]
    fn parses_an_iso8601_duration(#[case] text: &str, #[case] nanos: Option<i128>) {
        assert_eq!(parse_iso8601_duration(text), nanos);
    }
}
