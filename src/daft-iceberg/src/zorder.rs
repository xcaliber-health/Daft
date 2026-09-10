//! Z-order key encoding.
//!
//! Each clustering column is encoded as fixed-width bytes whose lexicographic order
//! matches value order, and the columns' bits are interleaved into one sortable key.

use arrow::{
    array::{
        Array, ArrayData, ArrayRef, BinaryArray, BooleanArray, Date32Array, Decimal128Array,
        Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
        LargeBinaryArray, LargeStringArray, StringArray, TimestampMicrosecondArray,
        TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt8Array,
        UInt16Array, UInt32Array, UInt64Array,
    },
    datatypes::{DataType, TimeUnit},
};

use crate::errors::IcebergRewriteError;

/// Column name of the synthetic interleaved key during a z-order rewrite.
///
/// The column is appended just long enough to sort, then projected away before
/// the write.
pub const ZORDER_KEY_COL: &str = "__daft_zorder_key__";

/// Fixed-width ordered encodings for one column, laid out end to end.
///
/// Every row occupies exactly `width` bytes, so the column is one allocation
/// rather than one per row.
pub struct OrderedColumn {
    bytes: Vec<u8>,
    width: usize,
}

impl OrderedColumn {
    fn new(rows: usize, width: usize) -> Self {
        Self {
            bytes: vec![0u8; rows * width],
            width,
        }
    }

    /// Encoded bytes for one row.
    ///
    /// # Panics
    /// Panics when `row` is not less than the number of rows encoded.
    #[must_use]
    pub fn row(&self, row: usize) -> &[u8] {
        &self.bytes[row * self.width..(row + 1) * self.width]
    }

    /// Width in bytes of every row's encoding.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    fn slot(&mut self, row: usize) -> &mut [u8] {
        let width = self.width;
        &mut self.bytes[row * width..(row + 1) * width]
    }
}

/// Bytes every whole number, floating point value, date and timestamp is encoded into.
///
/// Encoding every number at one width puts their significant bits at the same
/// offset in the interleaved key, so no column dominates the clustering.
const PRIMITIVE_WIDTH: usize = 8;

/// Bytes a 128-bit decimal is encoded into, which is its natural width.
const DECIMAL_WIDTH: usize = 16;

fn downcast<'a, T: 'static>(
    array: &'a dyn Array,
    expected: &str,
) -> Result<&'a T, IcebergRewriteError> {
    array
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| IcebergRewriteError::UnsupportedZOrderType {
            column: expected.to_string(),
            dtype: format!("{:?}", array.data_type()),
        })
}

/// Encode each row of `array` as an ordered byte slice of one fixed width.
///
/// Whole numbers, floating point values, dates and timestamps all encode to
/// [`PRIMITIVE_WIDTH`] bytes, decimals to sixteen, and text and binary to
/// `var_length_contribution` bytes by truncating or padding. Nulls encode to
/// zeroes, which sorts them first.
///
/// # Errors
/// Returns `IcebergRewriteError::UnsupportedZOrderType` when the array's type
/// has no ordered encoding.
pub fn normalize_to_ordered_bytes(
    array: &dyn Array,
    var_length_contribution: u32,
) -> Result<OrderedColumn, IcebergRewriteError> {
    let n = array.len();
    let var_len = var_length_contribution as usize;

    match array.data_type() {
        DataType::Boolean => {
            let a = downcast::<BooleanArray>(array, "boolean")?;
            Ok(whole_numbers(n, |i| i64::from(a.value(i)), array))
        }
        DataType::Int8 => {
            let a = downcast::<Int8Array>(array, "int8")?;
            Ok(whole_numbers(n, |i| i64::from(a.value(i)), array))
        }
        DataType::Int16 => {
            let a = downcast::<Int16Array>(array, "int16")?;
            Ok(whole_numbers(n, |i| i64::from(a.value(i)), array))
        }
        DataType::Int32 => {
            let a = downcast::<Int32Array>(array, "int32")?;
            Ok(whole_numbers(n, |i| i64::from(a.value(i)), array))
        }
        DataType::Int64 => {
            let a = downcast::<Int64Array>(array, "int64")?;
            Ok(whole_numbers(n, |i| a.value(i), array))
        }
        DataType::UInt8 => {
            let a = downcast::<UInt8Array>(array, "uint8")?;
            Ok(unsigned_numbers(n, |i| u64::from(a.value(i)), array))
        }
        DataType::UInt16 => {
            let a = downcast::<UInt16Array>(array, "uint16")?;
            Ok(unsigned_numbers(n, |i| u64::from(a.value(i)), array))
        }
        DataType::UInt32 => {
            let a = downcast::<UInt32Array>(array, "uint32")?;
            Ok(unsigned_numbers(n, |i| u64::from(a.value(i)), array))
        }
        DataType::UInt64 => {
            let a = downcast::<UInt64Array>(array, "uint64")?;
            Ok(unsigned_numbers(n, |i| a.value(i), array))
        }
        DataType::Float32 => {
            let a = downcast::<Float32Array>(array, "float32")?;
            Ok(floats(n, |i| f64::from(a.value(i)), array))
        }
        DataType::Float64 => {
            let a = downcast::<Float64Array>(array, "float64")?;
            Ok(floats(n, |i| a.value(i), array))
        }
        DataType::Date32 => {
            let a = downcast::<Date32Array>(array, "date32")?;
            Ok(whole_numbers(n, |i| i64::from(a.value(i)), array))
        }
        DataType::Timestamp(unit, _) => Ok(match unit {
            TimeUnit::Second => {
                let a = downcast::<TimestampSecondArray>(array, "timestamp")?;
                whole_numbers(n, |i| a.value(i), array)
            }
            TimeUnit::Millisecond => {
                let a = downcast::<TimestampMillisecondArray>(array, "timestamp")?;
                whole_numbers(n, |i| a.value(i), array)
            }
            TimeUnit::Microsecond => {
                let a = downcast::<TimestampMicrosecondArray>(array, "timestamp")?;
                whole_numbers(n, |i| a.value(i), array)
            }
            TimeUnit::Nanosecond => {
                let a = downcast::<TimestampNanosecondArray>(array, "timestamp")?;
                whole_numbers(n, |i| a.value(i), array)
            }
        }),
        DataType::Decimal128(_, _) => {
            let a = downcast::<Decimal128Array>(array, "decimal128")?;
            let mut out = OrderedColumn::new(n, DECIMAL_WIDTH);
            for i in 0..n {
                if array.is_null(i) {
                    continue;
                }
                let mut bytes = a.value(i).to_be_bytes();
                bytes[0] ^= 0x80;
                out.slot(i).copy_from_slice(&bytes);
            }
            Ok(out)
        }
        DataType::Utf8 => {
            let a = downcast::<StringArray>(array, "utf8")?;
            Ok(variable_width(n, var_len, |i| a.value(i).as_bytes(), array))
        }
        DataType::LargeUtf8 => {
            let a = downcast::<LargeStringArray>(array, "utf8")?;
            Ok(variable_width(n, var_len, |i| a.value(i).as_bytes(), array))
        }
        DataType::Binary => {
            let a = downcast::<BinaryArray>(array, "binary")?;
            Ok(variable_width(n, var_len, |i| a.value(i), array))
        }
        DataType::LargeBinary => {
            let a = downcast::<LargeBinaryArray>(array, "binary")?;
            Ok(variable_width(n, var_len, |i| a.value(i), array))
        }
        other => Err(IcebergRewriteError::UnsupportedZOrderType {
            column: String::new(),
            dtype: format!("{other:?}"),
        }),
    }
}

/// Encode signed values so that two's-complement order matches byte order.
fn whole_numbers<F: Fn(usize) -> i64>(n: usize, value: F, nulls: &dyn Array) -> OrderedColumn {
    let mut out = OrderedColumn::new(n, PRIMITIVE_WIDTH);
    for i in 0..n {
        if nulls.is_null(i) {
            continue;
        }
        let bytes = (value(i) ^ i64::MIN).to_be_bytes();
        out.slot(i).copy_from_slice(&bytes);
    }
    out
}

/// Encode unsigned values, whose byte order already matches their value order.
fn unsigned_numbers<F: Fn(usize) -> u64>(n: usize, value: F, nulls: &dyn Array) -> OrderedColumn {
    let mut out = OrderedColumn::new(n, PRIMITIVE_WIDTH);
    for i in 0..n {
        if nulls.is_null(i) {
            continue;
        }
        let bytes = value(i).to_be_bytes();
        out.slot(i).copy_from_slice(&bytes);
    }
    out
}

/// Encode floating point values through the total-order transform.
fn floats<F: Fn(usize) -> f64>(n: usize, value: F, nulls: &dyn Array) -> OrderedColumn {
    let mut out = OrderedColumn::new(n, PRIMITIVE_WIDTH);
    for i in 0..n {
        if nulls.is_null(i) {
            continue;
        }
        out.slot(i).copy_from_slice(&encode_float64(value(i)));
    }
    out
}

/// Take the leading bytes of each value, zero padded to a common width.
fn variable_width<'a, F: Fn(usize) -> &'a [u8]>(
    n: usize,
    width: usize,
    value: F,
    nulls: &dyn Array,
) -> OrderedColumn {
    let mut out = OrderedColumn::new(n, width);
    for i in 0..n {
        if nulls.is_null(i) {
            continue;
        }
        let raw = value(i);
        let take = raw.len().min(width);
        out.slot(i)[..take].copy_from_slice(&raw[..take]);
    }
    out
}

fn encode_float64(v: f64) -> [u8; 8] {
    let bits = v.to_bits();
    // Flip the sign bit for non-negatives and every bit for negatives, so the
    // result orders the way the values do. NaN sorts last.
    let mapped = if bits & 0x8000_0000_0000_0000 == 0 {
        bits ^ 0x8000_0000_0000_0000
    } else {
        !bits
    };
    mapped.to_be_bytes()
}

/// Interleave the columns' bits into one key per row, most significant first.
///
/// A column that has run out of bytes is skipped rather than contributing a
/// zero, so the columns still carrying information keep the whole of the
/// remaining key. Output shorter than the interleave is truncated, and output
/// longer is left zero padded.
///
/// # Panics
/// Panics when the columns do not all hold the same number of rows.
pub fn interleave_bits(columns: &[OrderedColumn], output_size: u64) -> Vec<u8> {
    let output_bytes = output_size as usize;
    if columns.is_empty() || output_bytes == 0 {
        return Vec::new();
    }
    let n_rows = columns[0].bytes.len() / columns[0].width.max(1);
    let output_bits = output_bytes * 8;
    let max_col_bits = columns.iter().map(|c| c.width * 8).max().unwrap_or(0);

    let mut out = vec![0u8; n_rows * output_bytes];
    for row in 0..n_rows {
        let dst = &mut out[row * output_bytes..(row + 1) * output_bytes];
        let mut out_bit = 0usize;
        for src_bit in 0..max_col_bits {
            if out_bit >= output_bits {
                break;
            }
            for col in columns {
                if out_bit >= output_bits {
                    break;
                }
                if src_bit >= col.width * 8 {
                    continue;
                }
                let byte = col.row(row)[src_bit / 8];
                if (byte >> (7 - (src_bit % 8))) & 1 != 0 {
                    dst[out_bit / 8] |= 1 << (7 - (out_bit % 8));
                }
                out_bit += 1;
            }
        }
    }
    out
}

/// Build the z-order key as a `BinaryArray` for a set of input arrays.
///
/// Each row's key is the bit-interleave of the per-column ordered byte
/// encodings, as long as those encodings together or `max_output_size` bytes,
/// whichever is smaller. The returned array has `arrays[0].len()` rows.
///
/// # Errors
/// Returns `IcebergRewriteError::InvalidOption` when `arrays` is empty or the
/// arrays differ in length, and `IcebergRewriteError::UnsupportedZOrderType`
/// when a column has no ordered encoding.
pub fn build_zorder_key_array(
    arrays: &[ArrayRef],
    var_length_contribution: u32,
    max_output_size: u64,
) -> Result<ArrayData, IcebergRewriteError> {
    if arrays.is_empty() {
        return Err(IcebergRewriteError::InvalidOption {
            name: "zorder_by".into(),
            reason: "expected at least one column".into(),
        });
    }
    let n_rows = arrays[0].len();
    for a in arrays {
        if a.len() != n_rows {
            return Err(IcebergRewriteError::InvalidOption {
                name: "zorder_by".into(),
                reason: "all z-order columns must share row count".into(),
            });
        }
    }
    let mut per_column: Vec<OrderedColumn> = Vec::with_capacity(arrays.len());
    for a in arrays {
        per_column.push(normalize_to_ordered_bytes(
            a.as_ref(),
            var_length_contribution,
        )?);
    }
    // Sizing the key to the encodings together, up to the cap, interleaves every
    // bit the columns carry and leaves no padding.
    let total_width: usize = per_column.iter().map(OrderedColumn::width).sum();
    let width = total_width.min(usize::try_from(max_output_size).unwrap_or(usize::MAX));
    let keys = interleave_bits(&per_column, width as u64);
    let arr = BinaryArray::from_iter_values((0..n_rows).map(|r| &keys[r * width..(r + 1) * width]));
    Ok(arr.into_data())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    /// Build a column from explicit rows, all of one width.
    fn column(rows: &[&[u8]]) -> OrderedColumn {
        let width = rows[0].len();
        let mut out = OrderedColumn::new(rows.len(), width);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), width, "rows must share a width");
            out.slot(i).copy_from_slice(row);
        }
        out
    }

    fn assert_lex_order_matches<T, F>(values: Vec<T>, encode: F)
    where
        T: PartialOrd + Copy + std::fmt::Debug,
        F: Fn(T) -> Vec<u8>,
    {
        let mut sorted = values;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let encoded: Vec<(T, Vec<u8>)> = sorted.iter().map(|v| (*v, encode(*v))).collect();
        for w in encoded.windows(2) {
            assert!(
                w[0].1 <= w[1].1,
                "lex order broken: {:?} -> {:?} vs {:?} -> {:?}",
                w[0].0,
                w[0].1,
                w[1].0,
                w[1].1
            );
        }
    }

    #[test]
    fn int32_ordering_matches_lex() {
        let vals = vec![-100i32, -1, 0, 1, 100, i32::MIN, i32::MAX];
        assert_lex_order_matches(vals, |v| {
            let arr = Int32Array::from(vec![v]);
            let bytes = normalize_to_ordered_bytes(&arr, 8).unwrap();
            bytes.row(0).to_vec()
        });
    }

    #[test]
    fn int64_ordering_matches_lex() {
        let vals = vec![
            -1_000_000_000i64,
            -1,
            0,
            1,
            1_000_000_000,
            i64::MIN,
            i64::MAX,
        ];
        assert_lex_order_matches(vals, |v| {
            let arr = Int64Array::from(vec![v]);
            let bytes = normalize_to_ordered_bytes(&arr, 8).unwrap();
            bytes.row(0).to_vec()
        });
    }

    #[test]
    fn float64_ordering_matches_lex() {
        // NaN deliberately excluded: behavior is left to upstream callers.
        let vals = vec![
            -f64::INFINITY,
            -1e10,
            -1.0,
            -0.0,
            0.0,
            1.0,
            1e10,
            f64::INFINITY,
        ];
        assert_lex_order_matches(vals, |v| {
            let arr = Float64Array::from(vec![v]);
            let bytes = normalize_to_ordered_bytes(&arr, 8).unwrap();
            bytes.row(0).to_vec()
        });
    }

    #[test]
    fn null_becomes_zero_padded_bytes() {
        let arr = Int32Array::from(vec![Some(1), None, Some(2)]);
        let bytes = normalize_to_ordered_bytes(&arr, 8).unwrap();
        assert_eq!(
            bytes.width(),
            PRIMITIVE_WIDTH,
            "a narrow int widens like any other"
        );
        assert_eq!(bytes.row(1), vec![0u8; PRIMITIVE_WIDTH]);
    }

    #[test]
    fn narrow_and_wide_whole_numbers_encode_alike() {
        // The point of one width: the same value encodes identically whatever
        // its column's natural size, so neither column dominates the key.
        let narrow = normalize_to_ordered_bytes(&Int32Array::from(vec![12_345i32]), 8).unwrap();
        let wide = normalize_to_ordered_bytes(&Int64Array::from(vec![12_345i64]), 8).unwrap();
        assert_eq!(narrow.row(0), wide.row(0));
    }

    #[test]
    fn narrow_and_wide_floats_encode_alike() {
        let narrow = normalize_to_ordered_bytes(&Float32Array::from(vec![0.5f32]), 8).unwrap();
        let wide = normalize_to_ordered_bytes(&Float64Array::from(vec![0.5f64]), 8).unwrap();
        assert_eq!(narrow.row(0), wide.row(0));
    }

    #[test]
    fn an_exhausted_column_is_skipped_rather_than_padded() {
        // A four-byte column beside an eight-byte one runs out halfway. The
        // longer column should keep the rest of the key to itself.
        let short = column(&[&[0u8; 4]]);
        let long = column(&[&[0xFFu8; 8]]);

        let key = interleave_bits(&[short, long], 16);

        // The first 32 bits of each column interleave into eight bytes, one bit
        // in two coming from the all-zero short column: 0x55 repeated.
        assert_eq!(
            &key[..8],
            &[0x55u8; 8],
            "both columns alternate while both have bytes"
        );
        // The short column is spent after that, so the long column's remaining
        // 32 bits land consecutively rather than every other bit.
        assert_eq!(
            &key[8..12],
            &[0xFFu8; 4],
            "the surviving column keeps the rest"
        );
        assert_eq!(&key[12..], &[0u8; 4], "nothing is left to fill the tail");
    }

    #[test]
    fn utf8_truncates_and_pads() {
        let arr = StringArray::from(vec![Some("a"), Some("abcdefghijklmnop"), None]);
        let bytes = normalize_to_ordered_bytes(&arr, 4).unwrap();
        assert_eq!(bytes.row(0), b"a\x00\x00\x00");
        assert_eq!(bytes.row(1), b"abcd");
        assert_eq!(bytes.row(2), vec![0u8; 4]);
    }

    #[test]
    fn boolean_encoding_matches_canonical() {
        let arr = BooleanArray::from(vec![Some(true), Some(false), None]);
        let bytes = normalize_to_ordered_bytes(&arr, 8).unwrap();
        assert!(bytes.row(0) > bytes.row(1), "true sorts after false");
        assert_eq!(bytes.row(2), vec![0u8; PRIMITIVE_WIDTH], "null sorts first");
    }

    #[test]
    fn decimal128_round_trip_preserves_sign_order() {
        use arrow::array::Decimal128Array;
        let arr = Decimal128Array::from(vec![
            Some(-100i128),
            Some(-1),
            Some(0),
            Some(1),
            Some(100),
            None,
        ])
        .with_precision_and_scale(20, 4)
        .unwrap();
        let bytes = normalize_to_ordered_bytes(&arr, 8).unwrap();
        let rows: Vec<Vec<u8>> = (0..5).map(|i| bytes.row(i).to_vec()).collect();
        for w in rows.windows(2) {
            assert!(
                w[0] <= w[1],
                "decimal ordering broken: {:?} vs {:?}",
                w[0],
                w[1]
            );
        }
        assert_eq!(bytes.row(5), vec![0u8; DECIMAL_WIDTH]);
    }

    #[test]
    fn partial_null_row_still_clusters_by_other_columns() {
        let a = column(&[&[0x80u8, 0, 0, 0], &[0u8; 4]]);
        let b = column(&[&[0x80u8, 0, 0, 0], &[0x80u8, 0, 0, 0]]);
        let out = interleave_bits(&[a, b], 8);
        assert_eq!(out.len(), 16, "two rows of eight bytes");
        assert_ne!(out[..8], out[8..]);
    }

    #[test]
    fn interleave_two_columns_known_fixture() {
        // One byte each: A = 1010_1010 and B = 1111_1111 interleave, A bit first,
        // into 1101_1101 1101_1101, which is 0xDD 0xDD.
        let cols = vec![column(&[&[0xAAu8]]), column(&[&[0xFFu8]])];
        let out = interleave_bits(&cols, 2);
        assert_eq!(out, vec![0xDDu8, 0xDDu8]);
    }

    #[test]
    fn interleave_all_zero_inputs_yield_all_zero_key() {
        let cols = vec![column(&[&[0u8; 4]]), column(&[&[0u8; 4]])];
        let out = interleave_bits(&cols, 4);
        assert_eq!(out, vec![0u8; 4]);
    }

    #[test]
    fn interleave_respects_output_size_truncation() {
        let cols = vec![column(&[&[0xFFu8; 16]]), column(&[&[0xFFu8; 16]])];
        let out = interleave_bits(&cols, 4);
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|b| *b == 0xFFu8));
    }

    #[test]
    fn build_key_array_round_trip() {
        let a = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;
        let b = Arc::new(StringArray::from(vec!["x", "y", "z"])) as ArrayRef;
        let data = build_zorder_key_array(&[a, b], 8, 16).unwrap();
        let arr = BinaryArray::from(data);
        assert_eq!(arr.len(), 3);
        assert_ne!(arr.value(0), arr.value(1));
        assert_ne!(arr.value(1), arr.value(2));
    }
}
