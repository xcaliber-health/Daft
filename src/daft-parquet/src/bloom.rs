//! Bloom-filter-based row-group pruning.
//!
//! Augments statistics-based pruning by probing per-column split-block bloom
//! filters for equality and membership predicates: a row group whose filter
//! proves every candidate value absent is skipped. Probing is conservative —
//! any uncertainty (column absent, no filter present, unsupported value type, or
//! a parse failure) keeps the row group, so results never change, only the
//! amount of data scanned.

use std::collections::{HashMap, HashSet};

use bytes::Bytes;
use common_error::DaftResult;
use daft_core::{lit::Literal, prelude::*};
use daft_dsl::{
    Expr, ExprRef,
    expr::{Column, ResolvedColumn, UnresolvedColumn},
};
use futures::{StreamExt, TryStreamExt};
use parquet::{basic::Type as PhysicalType, bloom_filter::Sbbf, file::metadata::ParquetMetaData};

use crate::reader::chunk_source::ChunkSourceBuilder;

/// Maximum number of bloom-filter bitsets fetched concurrently. Bounds the
/// number of in-flight out-of-band range reads issued before any column data is
/// read, so a wide predicate over many row groups cannot fan out without limit.
const BLOOM_FETCH_CONCURRENCY: usize = 16;

/// A single column probe: the row group survives only if at least one candidate
/// value may be present in the column's bloom filter.
///
/// An equality predicate yields one candidate value; a membership predicate
/// yields the full set, with "any may be present" matching membership semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BloomProbe {
    /// Dotted column path the filter is keyed on.
    pub column: String,
    /// Candidate values; the row group is kept if any may be present.
    pub values: Vec<Literal>,
}

/// Extract bloom-eligible probes from a predicate.
///
/// Walks the top-level conjunction, collecting `column == literal` equalities and
/// `column IN (literals)` memberships. Disjunctions, ranges, and function calls
/// are not bloom-eligible and are ignored here (statistics pruning still applies
/// to them). Each returned probe must hold for a row group to survive.
pub fn extract_bloom_probes(predicate: &ExprRef) -> Vec<BloomProbe> {
    let mut probes = Vec::new();
    collect_conjuncts(predicate, &mut probes);
    probes
}

fn collect_conjuncts(expr: &ExprRef, out: &mut Vec<BloomProbe>) {
    match expr.as_ref() {
        Expr::BinaryOp {
            op: Operator::And,
            left,
            right,
        } => {
            collect_conjuncts(left, out);
            collect_conjuncts(right, out);
        }
        Expr::BinaryOp {
            op: Operator::Eq,
            left,
            right,
        } => {
            if let Some(probe) = equality_probe(left, right) {
                out.push(probe);
            }
        }
        Expr::IsIn(column, items) => {
            if let Some(probe) = membership_probe(column, items) {
                out.push(probe);
            }
        }
        _ => {}
    }
}

/// Build a probe from an equality comparison with a column on one side and a
/// non-null literal on the other.
fn equality_probe(left: &ExprRef, right: &ExprRef) -> Option<BloomProbe> {
    let (column, literal) = match (column_name(left), literal_value(right)) {
        (Some(column), Some(literal)) => (column, literal),
        _ => {
            let column = column_name(right)?;
            let literal = literal_value(left)?;
            (column, literal)
        }
    };
    if matches!(literal, Literal::Null) {
        return None;
    }
    Some(BloomProbe {
        column,
        values: vec![literal.clone()],
    })
}

/// Build a probe from a membership test whose candidate list is entirely
/// non-null literals. A null candidate makes absence unprovable, so the probe is
/// dropped (the row group is kept).
fn membership_probe(column: &ExprRef, items: &[ExprRef]) -> Option<BloomProbe> {
    let column = column_name(column)?;
    let mut values = Vec::with_capacity(items.len());
    for item in items {
        match literal_value(item)? {
            Literal::Null => return None,
            literal => values.push(literal.clone()),
        }
    }
    if values.is_empty() {
        return None;
    }
    Some(BloomProbe { column, values })
}

fn column_name(expr: &ExprRef) -> Option<String> {
    if let Expr::Column(column) = expr.as_ref() {
        match column {
            Column::Unresolved(UnresolvedColumn { name, .. })
            | Column::Resolved(ResolvedColumn::Basic(name)) => Some(name.to_string()),
            _ => None,
        }
    } else {
        None
    }
}

fn literal_value(expr: &ExprRef) -> Option<&Literal> {
    if let Expr::Literal(literal) = expr.as_ref() {
        Some(literal)
    } else {
        None
    }
}

/// Encode a literal as the physical bytes that the writer hashed into the
/// column's bloom filter, or `None` when the value cannot be mapped to the
/// column's physical representation (in which case the row group is kept).
///
/// The physical type — not the logical type — drives the encoding: narrow
/// integers are stored as 32-bit physical values, decimals as their unscaled
/// integer (32/64-bit) or a fixed-width big-endian two's-complement, and
/// fixed-length binary as its raw bytes. `type_length` is the fixed byte width
/// for `FIXED_LEN_BYTE_ARRAY` columns (ignored otherwise); `type_scale` is the
/// column's decimal scale (`-1` when the column is not a decimal).
///
/// Decimal encoding is scale-sensitive: a candidate is encoded only when its
/// scale matches the column's, so the unscaled integer hashed here equals the
/// one the writer hashed. A mismatch returns `None` and keeps the row group,
/// never risking a wrong "absent" verdict.
fn physical_bytes(
    physical: PhysicalType,
    type_length: i32,
    type_scale: i32,
    literal: &Literal,
) -> Option<Vec<u8>> {
    match physical {
        PhysicalType::INT32 => int32_le(type_scale, literal).map(|v| v.to_le_bytes().to_vec()),
        PhysicalType::INT64 => int64_le(type_scale, literal).map(|v| v.to_le_bytes().to_vec()),
        PhysicalType::FLOAT => match literal {
            Literal::Float32(v) => Some(v.to_le_bytes().to_vec()),
            _ => None,
        },
        PhysicalType::DOUBLE => match literal {
            Literal::Float64(v) => Some(v.to_le_bytes().to_vec()),
            _ => None,
        },
        PhysicalType::BYTE_ARRAY => match literal {
            Literal::Utf8(s) => Some(s.as_bytes().to_vec()),
            Literal::Binary(b) => Some(b.clone()),
            _ => None,
        },
        PhysicalType::FIXED_LEN_BYTE_ARRAY => fixed_len_bytes(type_length, type_scale, literal),
        _ => None,
    }
}

fn int32_le(type_scale: i32, literal: &Literal) -> Option<i32> {
    match literal {
        Literal::Int8(v) => Some(i32::from(*v)),
        Literal::Int16(v) => Some(i32::from(*v)),
        Literal::Int32(v) => Some(*v),
        Literal::UInt8(v) => Some(i32::from(*v)),
        Literal::UInt16(v) => Some(i32::from(*v)),
        // Reinterpret the bits: an unsigned column is stored as a 32-bit physical
        // value, and the writer hashes those raw bytes.
        Literal::UInt32(v) => Some(i32::from_le_bytes(v.to_le_bytes())),
        Literal::Date(v) => Some(*v),
        // A low-precision decimal is stored as its unscaled 32-bit integer.
        Literal::Decimal(v, _, scale) if i32::from(*scale) == type_scale => i32::try_from(*v).ok(),
        _ => None,
    }
}

fn int64_le(type_scale: i32, literal: &Literal) -> Option<i64> {
    match literal {
        Literal::Int64(v) => Some(*v),
        Literal::UInt64(v) => Some(i64::from_le_bytes(v.to_le_bytes())),
        Literal::Timestamp(v, _, _) => Some(*v),
        Literal::Time(v, _) => Some(*v),
        Literal::Duration(v, _) => Some(*v),
        // A mid-precision decimal is stored as its unscaled 64-bit integer.
        Literal::Decimal(v, _, scale) if i32::from(*scale) == type_scale => i64::try_from(*v).ok(),
        _ => None,
    }
}

/// Encode a fixed-length-binary candidate as its raw physical bytes: a binary or
/// UUID value passed through when its width equals the column's, or a decimal as
/// the unscaled integer in big-endian two's-complement padded to the column
/// width. Returns `None` (keeping the row group) on any width or scale mismatch.
fn fixed_len_bytes(type_length: i32, type_scale: i32, literal: &Literal) -> Option<Vec<u8>> {
    let len = usize::try_from(type_length).ok().filter(|&l| l > 0)?;
    match literal {
        Literal::Binary(b) if b.len() == len => Some(b.clone()),
        Literal::Uuid(u) if len == 16 => Some(u.as_bytes().to_vec()),
        Literal::Decimal(v, _, scale) if i32::from(*scale) == type_scale => {
            decimal_fixed_be(*v, len)
        }
        _ => None,
    }
}

/// Encode an unscaled decimal as `len` big-endian two's-complement bytes,
/// matching the fixed-width physical layout the writer hashed. Returns `None`
/// when the value does not fit in `len` bytes, so a too-wide candidate keeps the
/// row group rather than hashing a truncated value.
fn decimal_fixed_be(value: i128, len: usize) -> Option<Vec<u8>> {
    if len == 0 || len > 16 {
        return None;
    }
    let full = value.to_be_bytes();
    let (high, low) = full.split_at(16 - len);
    // The dropped high bytes must be pure sign extension, else the value is wider
    // than the column and cannot be represented in `len` bytes.
    let sign_byte = if value < 0 { 0xFF } else { 0x00 };
    if high.iter().any(|&b| b != sign_byte) {
        return None;
    }
    Some(low.to_vec())
}

/// A resolved, conclusive probe against one row group: the column has a filter,
/// and every candidate value encodes to the column's physical bytes. The row
/// group is dropped if none of `encoded` may be present in the fetched filter.
struct PendingProbe {
    /// Row group this probe constrains.
    rg_idx: usize,
    /// Absolute file offset of the column's bloom-filter bitset.
    offset: u64,
    /// Byte length of the bitset.
    length: usize,
    /// Candidate values in physical byte form; the row group survives this probe
    /// if at least one may be present.
    encoded: Vec<Vec<u8>>,
}

/// Prune row groups whose bloom filters prove that an equality or membership
/// probe cannot match.
///
/// `rg_indices` are the candidates that survived statistics pruning, in file
/// order. Returns the subset to read. With no probes, the input is returned
/// unchanged.
///
/// Filter bitsets are read out of band from `source`: every distinct bitset
/// needed across the surviving row groups and probes is fetched concurrently
/// before any probing, rather than one serial read per row group, so the latency
/// of locating a value does not grow with the number of row groups. Probing then
/// runs in memory. A probe drops a row group only when its filter is present,
/// readable, and proves all candidate values absent; any uncertainty keeps the
/// row group.
pub async fn prune_row_groups_by_bloom(
    source: &ChunkSourceBuilder,
    metadata: &ParquetMetaData,
    probes: &[BloomProbe],
    rg_indices: Vec<usize>,
) -> DaftResult<Vec<usize>> {
    if probes.is_empty() {
        return Ok(rg_indices);
    }

    let schema_descr = metadata.file_metadata().schema_descr();
    let mut column_index: HashMap<String, usize> =
        HashMap::with_capacity(schema_descr.num_columns());
    for idx in 0..schema_descr.num_columns() {
        column_index.insert(schema_descr.column(idx).path().string(), idx);
    }

    // Resolve every conclusive (row group, probe) pair up front, collecting the
    // distinct bitset ranges they require.
    let mut pending: Vec<PendingProbe> = Vec::new();
    let mut ranges: HashMap<u64, usize> = HashMap::new();
    for &rg_idx in &rg_indices {
        let row_group = metadata.row_group(rg_idx);
        for probe in probes {
            if let Some(p) = resolve_probe(row_group, rg_idx, probe, &column_index) {
                ranges.insert(p.offset, p.length);
                pending.push(p);
            }
        }
    }
    if pending.is_empty() {
        return Ok(rg_indices);
    }

    // Fetch all distinct bitsets concurrently.
    let fetched: HashMap<u64, Bytes> = futures::stream::iter(ranges)
        .map(|(offset, length)| async move {
            let bytes = source.read_range(offset, length).await?;
            DaftResult::Ok((offset, bytes))
        })
        .buffer_unordered(BLOOM_FETCH_CONCURRENCY)
        .try_collect()
        .await?;

    // Probe in memory: a row group is dropped as soon as one conclusive probe
    // proves all its candidates absent.
    let mut dropped: HashSet<usize> = HashSet::new();
    for p in &pending {
        if dropped.contains(&p.rg_idx) {
            continue;
        }
        let Some(buffer) = fetched.get(&p.offset) else {
            continue;
        };
        let Ok(filter) = Sbbf::from_bytes(buffer) else {
            continue;
        };
        if !p.encoded.iter().any(|bytes| filter.check(bytes)) {
            dropped.insert(p.rg_idx);
        }
    }

    Ok(rg_indices
        .into_iter()
        .filter(|idx| !dropped.contains(idx))
        .collect())
}

/// Resolve a probe against one row group into a [`PendingProbe`], or `None` when
/// the probe is inconclusive for this row group — the column is absent, has no
/// filter, or a candidate cannot be encoded to the column's physical type. An
/// inconclusive probe can never drop the row group.
fn resolve_probe(
    row_group: &parquet::file::metadata::RowGroupMetaData,
    rg_idx: usize,
    probe: &BloomProbe,
    column_index: &HashMap<String, usize>,
) -> Option<PendingProbe> {
    let &col_idx = column_index.get(probe.column.as_str())?;
    let column = row_group.column(col_idx);
    let offset = u64::try_from(column.bloom_filter_offset()?).ok()?;
    let length = usize::try_from(column.bloom_filter_length()?).ok()?;
    if length == 0 {
        return None;
    }

    let descr = column.column_descr();
    let physical = descr.physical_type();
    let type_length = descr.type_length();
    let type_scale = descr.type_scale();
    let mut encoded = Vec::with_capacity(probe.values.len());
    for value in &probe.values {
        encoded.push(physical_bytes(physical, type_length, type_scale, value)?);
    }
    if encoded.is_empty() {
        return None;
    }
    Some(PendingProbe {
        rg_idx,
        offset,
        length,
        encoded,
    })
}

#[cfg(test)]
mod tests {
    use daft_dsl::{lit, resolved_col};

    use super::*;

    #[test]
    fn extracts_equality_probe_both_orientations() {
        let left = resolved_col("id").eq(lit(7i64));
        let right = lit(7i64).eq(resolved_col("id"));

        for predicate in [left, right] {
            let probes = extract_bloom_probes(&predicate);
            assert_eq!(probes.len(), 1);
            assert_eq!(probes[0].column, "id");
            assert_eq!(probes[0].values, vec![Literal::Int64(7)]);
        }
    }

    #[test]
    fn extracts_conjunction_into_multiple_probes() {
        let predicate = resolved_col("id")
            .eq(lit(7i64))
            .and(resolved_col("name").eq(lit("alice")));

        let probes = extract_bloom_probes(&predicate);

        let columns: Vec<&str> = probes.iter().map(|p| p.column.as_str()).collect();
        assert_eq!(columns, vec!["id", "name"]);
    }

    #[test]
    fn extracts_membership_probe() {
        let predicate = resolved_col("id").is_in(vec![lit(1i64), lit(2i64), lit(3i64)]);

        let probes = extract_bloom_probes(&predicate);

        assert_eq!(probes.len(), 1);
        assert_eq!(probes[0].column, "id");
        assert_eq!(probes[0].values.len(), 3);
    }

    #[test]
    fn ignores_range_and_disjunction() {
        let range = resolved_col("id").gt(lit(7i64));
        let disjunction = resolved_col("id")
            .eq(lit(1i64))
            .or(resolved_col("id").eq(lit(2i64)));

        assert!(extract_bloom_probes(&range).is_empty());
        assert!(extract_bloom_probes(&disjunction).is_empty());
    }

    #[test]
    fn ignores_null_equality() {
        let predicate = resolved_col("id").eq(lit(Literal::Null));

        assert!(extract_bloom_probes(&predicate).is_empty());
    }

    #[test]
    fn narrow_integer_uses_physical_width() {
        // Int8 is stored as a 32-bit physical value, so it encodes to four bytes.
        let bytes = physical_bytes(PhysicalType::INT32, 0, -1, &Literal::Int8(5)).unwrap();
        assert_eq!(bytes, 5i32.to_le_bytes().to_vec());

        let wide = physical_bytes(PhysicalType::INT64, 0, -1, &Literal::Int64(5)).unwrap();
        assert_eq!(wide, 5i64.to_le_bytes().to_vec());

        // Temporal values reuse the integer physical width the writer hashed.
        let date = physical_bytes(PhysicalType::INT32, 0, -1, &Literal::Date(19_000)).unwrap();
        assert_eq!(date, 19_000i32.to_le_bytes().to_vec());
        let ts = physical_bytes(
            PhysicalType::INT64,
            0,
            -1,
            &Literal::Timestamp(1_700_000_000_000_000, TimeUnit::Microseconds, None),
        )
        .unwrap();
        assert_eq!(ts, 1_700_000_000_000_000i64.to_le_bytes().to_vec());
    }

    #[test]
    fn string_and_binary_encode_to_raw_bytes() {
        assert_eq!(
            physical_bytes(
                PhysicalType::BYTE_ARRAY,
                0,
                -1,
                &Literal::Utf8("abc".to_string())
            )
            .unwrap(),
            b"abc".to_vec()
        );
        assert_eq!(
            physical_bytes(
                PhysicalType::BYTE_ARRAY,
                0,
                -1,
                &Literal::Binary(vec![1, 2, 3])
            )
            .unwrap(),
            vec![1u8, 2, 3]
        );
    }

    #[test]
    fn mismatched_type_is_unsupported() {
        assert!(
            physical_bytes(PhysicalType::INT64, 0, -1, &Literal::Utf8("x".to_string())).is_none()
        );
        assert!(physical_bytes(PhysicalType::DOUBLE, 0, -1, &Literal::Int64(1)).is_none());
        assert!(physical_bytes(PhysicalType::BOOLEAN, 0, -1, &Literal::Int32(1)).is_none());
    }

    #[test]
    fn decimal_encodes_at_matching_scale_only() {
        // Scale 2 decimal stored in a 32-bit column encodes as the unscaled int.
        let i32_bytes =
            physical_bytes(PhysicalType::INT32, 0, 2, &Literal::Decimal(12345, 9, 2)).unwrap();
        assert_eq!(i32_bytes, 12345i32.to_le_bytes().to_vec());

        // Same value in a 64-bit column.
        let i64_bytes =
            physical_bytes(PhysicalType::INT64, 0, 2, &Literal::Decimal(12345, 18, 2)).unwrap();
        assert_eq!(i64_bytes, 12345i64.to_le_bytes().to_vec());

        // A scale mismatch is inconclusive — never hash a value the writer did not.
        assert!(
            physical_bytes(PhysicalType::INT32, 0, 3, &Literal::Decimal(12345, 9, 2)).is_none()
        );
    }

    #[test]
    fn fixed_len_encodes_binary_uuid_and_decimal() {
        // UUID fills a 16-byte fixed column in big-endian order.
        let uuid = uuid::Uuid::from_bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        assert_eq!(
            physical_bytes(
                PhysicalType::FIXED_LEN_BYTE_ARRAY,
                16,
                -1,
                &Literal::Uuid(uuid)
            )
            .unwrap(),
            (0u8..16).collect::<Vec<u8>>()
        );

        // Fixed binary passes through only at the exact column width.
        assert_eq!(
            physical_bytes(
                PhysicalType::FIXED_LEN_BYTE_ARRAY,
                3,
                -1,
                &Literal::Binary(vec![9, 8, 7])
            )
            .unwrap(),
            vec![9u8, 8, 7]
        );
        assert!(
            physical_bytes(
                PhysicalType::FIXED_LEN_BYTE_ARRAY,
                4,
                -1,
                &Literal::Binary(vec![9, 8, 7])
            )
            .is_none()
        );

        // High-precision decimal: big-endian two's-complement padded to width.
        let pos = physical_bytes(
            PhysicalType::FIXED_LEN_BYTE_ARRAY,
            4,
            0,
            &Literal::Decimal(258, 9, 0),
        )
        .unwrap();
        assert_eq!(pos, vec![0x00, 0x00, 0x01, 0x02]);
        let neg = physical_bytes(
            PhysicalType::FIXED_LEN_BYTE_ARRAY,
            4,
            0,
            &Literal::Decimal(-2, 9, 0),
        )
        .unwrap();
        assert_eq!(neg, vec![0xFF, 0xFF, 0xFF, 0xFE]);

        // A value too wide for the column width does not fit and is inconclusive.
        assert!(decimal_fixed_be(0x01_0000, 2).is_none());
        assert_eq!(decimal_fixed_be(0x0102, 2).unwrap(), vec![0x01, 0x02]);
    }

    /// Round-trip a filter written by the Parquet writer through the same
    /// offset/length read and byte-encoding the pruner uses, proving the probe
    /// agrees with the writer's hashing for present and absent values.
    #[test]
    fn reads_and_probes_writer_produced_filter() {
        use arrow::{
            array::{Int64Array, RecordBatch as ArrowRecordBatch},
            datatypes::{DataType, Field as ArrowField, Schema as ArrowSchema},
        };
        use bytes::Bytes;
        use parquet::{
            arrow::ArrowWriter,
            file::{
                properties::WriterProperties,
                reader::{FileReader, SerializedFileReader},
            },
        };

        let schema = std::sync::Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int64,
            false,
        )]));
        let props = WriterProperties::builder()
            .set_column_bloom_filter_enabled("id".into(), true)
            .set_column_bloom_filter_max_ndv("id".into(), 100)
            .build();

        let mut buffer = Vec::new();
        {
            let mut writer =
                ArrowWriter::try_new(&mut buffer, schema.clone(), Some(props)).unwrap();
            let batch = ArrowRecordBatch::try_new(
                schema,
                vec![std::sync::Arc::new(Int64Array::from(vec![10i64, 20, 30]))],
            )
            .unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }

        let bytes = Bytes::from(buffer);
        let reader = SerializedFileReader::new(bytes.clone()).unwrap();
        let column = reader.metadata().row_group(0).column(0);
        let offset = usize::try_from(column.bloom_filter_offset().unwrap()).unwrap();
        let length = usize::try_from(column.bloom_filter_length().unwrap()).unwrap();

        let filter = Sbbf::from_bytes(&bytes[offset..offset + length]).unwrap();

        let present = physical_bytes(PhysicalType::INT64, 0, -1, &Literal::Int64(20)).unwrap();
        let absent = physical_bytes(PhysicalType::INT64, 0, -1, &Literal::Int64(999)).unwrap();
        assert!(filter.check(&present), "present value must probe true");
        assert!(!filter.check(&absent), "absent value must probe false");
    }

    /// Write a single-column Parquet file with a per-row-group bloom filter to a
    /// temporary path, returning the path. `max_rg` rows per row group lets a
    /// caller place disjoint value ranges in distinct row groups.
    fn write_bloom_parquet(
        field: arrow::datatypes::Field,
        array: arrow::array::ArrayRef,
        max_rg: usize,
    ) -> std::path::PathBuf {
        use arrow::{array::RecordBatch as ArrowRecordBatch, datatypes::Schema as ArrowSchema};
        use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

        let name = field.name().clone();
        let schema = std::sync::Arc::new(ArrowSchema::new(vec![field]));
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(max_rg))
            .set_column_bloom_filter_enabled(name.clone().into(), true)
            .set_column_bloom_filter_max_ndv(name.into(), u64::try_from(max_rg).unwrap())
            .build();

        let mut buffer = Vec::new();
        {
            let mut writer =
                ArrowWriter::try_new(&mut buffer, schema.clone(), Some(props)).unwrap();
            let batch = ArrowRecordBatch::try_new(schema, vec![array]).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }

        let path = std::env::temp_dir().join(format!(
            "daft_bloom_{}_{}.parquet",
            std::process::id(),
            fastrand::u64(..)
        ));
        std::fs::write(&path, &buffer).unwrap();
        path
    }

    fn prune(path: &std::path::Path, probes: Vec<BloomProbe>, rgs: Vec<usize>) -> Vec<usize> {
        let (source, metadata) =
            ChunkSourceBuilder::local_for_test(path.to_str().unwrap()).unwrap();
        common_runtime::get_io_runtime(true)
            .block_on_current_thread(async move {
                prune_row_groups_by_bloom(&source, &metadata, &probes, rgs).await
            })
            .unwrap()
    }

    /// Proof that bloom probing actually drops row groups: with disjoint value
    /// ranges per row group, an equality probe keeps only the row group that may
    /// contain the value, and an absent value drops them all.
    #[test]
    fn prunes_row_groups_via_writer_filter() {
        use arrow::{
            array::Int64Array,
            datatypes::{DataType, Field as ArrowField},
        };

        // 300 rows, 100 per row group => RG0=0..99, RG1=100..199, RG2=200..299.
        let array = std::sync::Arc::new(Int64Array::from((0..300i64).collect::<Vec<_>>()));
        let path = write_bloom_parquet(ArrowField::new("v", DataType::Int64, false), array, 100);

        let present = vec![BloomProbe {
            column: "v".to_string(),
            values: vec![Literal::Int64(150)],
        }];
        assert_eq!(
            prune(&path, present, vec![0, 1, 2]),
            vec![1],
            "only the row group holding 150 survives"
        );

        let absent = vec![BloomProbe {
            column: "v".to_string(),
            values: vec![Literal::Int64(10_000)],
        }];
        assert_eq!(
            prune(&path, absent, vec![0, 1, 2]),
            Vec::<usize>::new(),
            "an absent value prunes every row group"
        );

        let membership = vec![BloomProbe {
            column: "v".to_string(),
            values: vec![Literal::Int64(50), Literal::Int64(250)],
        }];
        assert_eq!(
            prune(&path, membership, vec![0, 1, 2]),
            vec![0, 2],
            "membership keeps every row group that may hold a candidate"
        );

        std::fs::remove_file(&path).ok();
    }

    /// Cross-type encoding parity against a real on-disk filter: for each physical
    /// type, a present value keeps its row group and an absent value prunes it.
    #[test]
    fn cross_type_filter_parity() {
        use arrow::{
            array::{
                ArrayRef, BinaryArray, Date32Array, FixedSizeBinaryArray, Float64Array, Int32Array,
                StringArray,
            },
            datatypes::{DataType, Field as ArrowField},
        };

        fn check(field: ArrowField, array: ArrayRef, present: Literal, absent: Literal) {
            let name = field.name().clone();
            let path = write_bloom_parquet(field, array, 1024);
            let present_probe = vec![BloomProbe {
                column: name.clone(),
                values: vec![present],
            }];
            assert_eq!(
                prune(&path, present_probe, vec![0]),
                vec![0],
                "present value must keep the row group ({name})"
            );
            let absent_probe = vec![BloomProbe {
                column: name.clone(),
                values: vec![absent],
            }];
            assert_eq!(
                prune(&path, absent_probe, vec![0]),
                Vec::<usize>::new(),
                "absent value must prune the row group ({name})"
            );
            std::fs::remove_file(&path).ok();
        }

        check(
            ArrowField::new("v", DataType::Int32, false),
            std::sync::Arc::new(Int32Array::from(vec![1i32, 2, 3])),
            Literal::Int32(2),
            Literal::Int32(99),
        );
        check(
            ArrowField::new("v", DataType::Float64, false),
            std::sync::Arc::new(Float64Array::from(vec![1.5f64, 2.5, 3.5])),
            Literal::Float64(2.5),
            Literal::Float64(9.5),
        );
        check(
            ArrowField::new("v", DataType::Utf8, false),
            std::sync::Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Literal::Utf8("b".to_string()),
            Literal::Utf8("zzz".to_string()),
        );
        check(
            ArrowField::new("v", DataType::Binary, false),
            std::sync::Arc::new(BinaryArray::from(vec![b"\x01".as_ref(), b"\x02", b"\x03"])),
            Literal::Binary(vec![2]),
            Literal::Binary(vec![9]),
        );
        check(
            ArrowField::new("v", DataType::Date32, false),
            std::sync::Arc::new(Date32Array::from(vec![10i32, 20, 30])),
            Literal::Date(20),
            Literal::Date(999),
        );
        // Fixed-length binary exercises the FIXED_LEN_BYTE_ARRAY encoder (the UUID
        // physical layout): 16 raw bytes matched on width.
        let present_fixed = [1u8; 16];
        let absent_fixed = [9u8; 16];
        let fixed = FixedSizeBinaryArray::try_from_iter(
            vec![[0u8; 16], present_fixed, [2u8; 16]].into_iter(),
        )
        .unwrap();
        check(
            ArrowField::new("v", DataType::FixedSizeBinary(16), false),
            std::sync::Arc::new(fixed),
            Literal::Binary(present_fixed.to_vec()),
            Literal::Binary(absent_fixed.to_vec()),
        );
    }
}
