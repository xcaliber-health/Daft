//! Bloom-filter-based row-group pruning.
//!
//! Augments statistics-based pruning by probing per-column split-block bloom
//! filters for equality and membership predicates: a row group whose filter
//! proves every candidate value absent is skipped. Probing is conservative —
//! any uncertainty (column absent, no filter present, unsupported value type, or
//! a parse failure) keeps the row group, so results never change, only the
//! amount of data scanned.

use std::collections::HashMap;

use common_error::DaftResult;
use daft_core::{lit::Literal, prelude::*};
use daft_dsl::{
    Expr, ExprRef,
    expr::{Column, ResolvedColumn, UnresolvedColumn},
};
use parquet::{basic::Type as PhysicalType, bloom_filter::Sbbf, file::metadata::ParquetMetaData};

use crate::reader::chunk_source::ChunkSourceBuilder;

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

/// Encode a literal as the little-endian physical bytes that the writer hashed
/// into the column's bloom filter, or `None` when the value cannot be mapped to
/// the column's physical type (in which case the row group is kept).
///
/// The physical type — not the logical type — drives the width: narrow integers
/// are stored as 32-bit physical values, so they must be hashed as such.
fn physical_bytes(physical: PhysicalType, literal: &Literal) -> Option<Vec<u8>> {
    match physical {
        PhysicalType::INT32 => int32_le(literal).map(|v| v.to_le_bytes().to_vec()),
        PhysicalType::INT64 => int64_le(literal).map(|v| v.to_le_bytes().to_vec()),
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
        _ => None,
    }
}

fn int32_le(literal: &Literal) -> Option<i32> {
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
        _ => None,
    }
}

fn int64_le(literal: &Literal) -> Option<i64> {
    match literal {
        Literal::Int64(v) => Some(*v),
        Literal::UInt64(v) => Some(i64::from_le_bytes(v.to_le_bytes())),
        Literal::Timestamp(v, _, _) => Some(*v),
        Literal::Time(v, _) => Some(*v),
        Literal::Duration(v, _) => Some(*v),
        _ => None,
    }
}

/// Prune row groups whose bloom filters prove that an equality or membership
/// probe cannot match.
///
/// `rg_indices` are the candidates that survived statistics pruning, in file
/// order. Returns the subset to read. With no probes, the input is returned
/// unchanged. Filter bitsets are read on demand from `source`.
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

    let mut kept = Vec::with_capacity(rg_indices.len());
    for rg_idx in rg_indices {
        if !row_group_survives(source, metadata, rg_idx, probes, &column_index).await? {
            continue;
        }
        kept.push(rg_idx);
    }
    Ok(kept)
}

/// Decide whether a single row group survives every probe. A probe drops the row
/// group only when its filter is present, readable, and proves all candidate
/// values absent; otherwise the probe is inconclusive and the row group is kept.
async fn row_group_survives(
    source: &ChunkSourceBuilder,
    metadata: &ParquetMetaData,
    rg_idx: usize,
    probes: &[BloomProbe],
    column_index: &HashMap<String, usize>,
) -> DaftResult<bool> {
    let row_group = metadata.row_group(rg_idx);
    for probe in probes {
        let Some(&col_idx) = column_index.get(&probe.column) else {
            continue;
        };
        let column = row_group.column(col_idx);
        let (Some(offset), Some(length)) =
            (column.bloom_filter_offset(), column.bloom_filter_length())
        else {
            continue;
        };
        let (Ok(offset), Ok(length)) = (u64::try_from(offset), usize::try_from(length)) else {
            continue;
        };
        if length == 0 {
            continue;
        }

        let physical = column.column_descr().physical_type();
        let mut encoded = Vec::with_capacity(probe.values.len());
        let mut unsupported = false;
        for value in &probe.values {
            match physical_bytes(physical, value) {
                Some(bytes) => encoded.push(bytes),
                None => {
                    unsupported = true;
                    break;
                }
            }
        }
        if unsupported || encoded.is_empty() {
            continue;
        }

        let buffer = source.read_range(offset, length).await?;
        let Ok(filter) = Sbbf::from_bytes(&buffer) else {
            continue;
        };
        if !encoded.iter().any(|bytes| filter.check(bytes)) {
            return Ok(false);
        }
    }
    Ok(true)
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
        let bytes = physical_bytes(PhysicalType::INT32, &Literal::Int8(5)).unwrap();
        assert_eq!(bytes, 5i32.to_le_bytes().to_vec());

        let wide = physical_bytes(PhysicalType::INT64, &Literal::Int64(5)).unwrap();
        assert_eq!(wide, 5i64.to_le_bytes().to_vec());
    }

    #[test]
    fn string_and_binary_encode_to_raw_bytes() {
        assert_eq!(
            physical_bytes(PhysicalType::BYTE_ARRAY, &Literal::Utf8("abc".to_string())).unwrap(),
            b"abc".to_vec()
        );
        assert_eq!(
            physical_bytes(PhysicalType::BYTE_ARRAY, &Literal::Binary(vec![1, 2, 3])).unwrap(),
            vec![1u8, 2, 3]
        );
    }

    #[test]
    fn mismatched_type_is_unsupported() {
        assert!(physical_bytes(PhysicalType::INT64, &Literal::Utf8("x".to_string())).is_none());
        assert!(physical_bytes(PhysicalType::DOUBLE, &Literal::Int64(1)).is_none());
        assert!(physical_bytes(PhysicalType::BOOLEAN, &Literal::Int32(1)).is_none());
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
            .set_column_bloom_filter_ndv("id".into(), 100)
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

        let present = physical_bytes(PhysicalType::INT64, &Literal::Int64(20)).unwrap();
        let absent = physical_bytes(PhysicalType::INT64, &Literal::Int64(999)).unwrap();
        assert!(filter.check(&present), "present value must probe true");
        assert!(!filter.check(&absent), "absent value must probe false");
    }
}
