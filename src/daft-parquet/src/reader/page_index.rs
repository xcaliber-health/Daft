//! A Parquet file's page index, read for a filtered scan.
//!
//! The offset index locates each page of a column chunk and the row it starts
//! at; the column index holds each page's minimum, maximum and null count.
//! With them, a filtered read skips the pages of the filter's columns whose
//! statistics rule the filter out, then fetches, of every other column, only
//! the pages holding a kept row. A file without a page index is read whole, as
//! before.

use std::ops::Range;

use common_error::DaftResult;
use daft_core::prelude::Schema;
use daft_dsl::{ExprRef, expr::bound_expr::BoundExpr, optimization::get_required_columns};
use daft_stats::{TableStatistics, TruthValue};
use parquet::{
    arrow::arrow_reader::{RowSelection, RowSelector},
    data_type::{ByteArray, FixedLenByteArray},
    file::{
        metadata::{PageIndexPolicy, ParquetMetaData, ParquetMetaDataReader},
        page_index::column_index::ColumnIndexMetaData,
        reader::ChunkReader,
        statistics::Statistics,
    },
};

use crate::statistics::parquet_statistics_to_column_range_statistics;

/// The smallest row group, in compressed bytes read, of which a filtered read
/// fetches only the pages it needs: the gap below which the remote reader
/// merges requests. Below it, the round trip a page fetch adds costs more than
/// the bytes it saves, and the row group is fetched ahead whole.
pub(super) const PAGE_FETCH_MIN_BYTES: u64 = 1024 * 1024;

/// Whether any row group's `leaves` are large enough to be fetched by page,
/// so that the page index is worth reading at all.
pub(super) fn worth_reading(metadata: &ParquetMetaData, leaves: &[usize]) -> bool {
    metadata.row_groups().iter().any(|row_group| {
        leaves
            .iter()
            .map(|&leaf| u64::try_from(row_group.column(leaf).compressed_size()).unwrap_or(0))
            .sum::<u64>()
            > PAGE_FETCH_MIN_BYTES
    })
}

/// The byte range of the file holding every column chunk's column index and
/// offset index, when the file has a page index.
pub(super) fn index_range(metadata: &ParquetMetaData) -> Option<Range<u64>> {
    let mut range: Option<Range<u64>> = None;
    for row_group in metadata.row_groups() {
        for column in row_group.columns() {
            let indexes = [
                column.offset_index_offset().zip(column.offset_index_length()),
                column.column_index_offset().zip(column.column_index_length()),
            ];
            for (offset, length) in indexes.into_iter().flatten() {
                let (Ok(start), Ok(length)) = (u64::try_from(offset), u64::try_from(length)) else {
                    continue;
                };
                let end = start + length;
                range = Some(match range {
                    Some(range) => range.start.min(start)..range.end.max(end),
                    None => start..end,
                });
            }
        }
    }
    range
}

/// `metadata` with its page index, decoded from `reader`, which must hold the
/// bytes of [`index_range`].
///
/// # Errors
/// Returns an error if the index bytes cannot be decoded.
pub(super) fn with_page_index<R: ChunkReader>(
    metadata: ParquetMetaData,
    reader: &R,
) -> parquet::errors::Result<ParquetMetaData> {
    let mut decoder = ParquetMetaDataReader::new_with_metadata(metadata)
        .with_page_index_policy(PageIndexPolicy::Optional);
    decoder.read_page_indexes(reader)?;
    decoder.finish()
}

/// The rows of row group `rg_idx` that `predicate` might keep, judged by the
/// column index of each column it compares on its own; `None` when no page is
/// ruled out.
///
/// Each conjunct of `predicate` that reads a single column is evaluated against
/// the minimum, maximum and null count of each page of that column; a page it
/// is false for is skipped. Conjuncts are intersected. A conjunct reading
/// several columns, or a column with no column index, rules nothing out.
///
/// # Errors
/// Returns an error if a conjunct cannot be bound to its column.
pub(super) fn rows_not_ruled_out(
    predicate: &ExprRef,
    metadata: &ParquetMetaData,
    rg_idx: usize,
    schema: &Schema,
) -> DaftResult<Option<RowSelection>> {
    let (Some(column_indexes), Some(offset_indexes)) =
        (metadata.column_index(), metadata.offset_index())
    else {
        return Ok(None);
    };
    let row_group = metadata.row_group(rg_idx);
    let rows = usize::try_from(row_group.num_rows()).unwrap_or(0);
    let mut kept: Option<RowSelection> = None;
    for conjunct in daft_algebra::boolean::split_conjunction(predicate) {
        // A column read more than once, as in a chain of equalities, is still
        // one column.
        let mut columns = get_required_columns(&conjunct);
        columns.sort_unstable();
        columns.dedup();
        let [column] = columns.as_slice() else {
            continue;
        };
        let Ok(field) = schema.get_field(column) else {
            continue;
        };
        let Some(leaf) = single_leaf(metadata, column) else {
            continue;
        };
        let pages = &offset_indexes[rg_idx][leaf].page_locations;
        let page_stats = &column_indexes[rg_idx][leaf];
        let one_column = Schema::new(vec![field.clone()]);
        let bound = BoundExpr::try_new(conjunct.clone(), &one_column)?;
        let mut selectors = Vec::with_capacity(pages.len());
        for (page, location) in pages.iter().enumerate() {
            let start = usize::try_from(location.first_row_index).unwrap_or(0);
            let end = pages
                .get(page + 1)
                .and_then(|next| usize::try_from(next.first_row_index).ok())
                .unwrap_or(rows);
            let ruled_out = page_statistics(page_stats, page)
                .and_then(|stats| {
                    parquet_statistics_to_column_range_statistics(
                        &stats,
                        row_group.column(leaf).column_descr(),
                        &field.dtype,
                    )
                    .ok()
                })
                .map(|stats| TableStatistics::new(vec![stats], one_column.clone().into()))
                .and_then(|stats| stats.eval_expression(&bound).ok())
                .is_some_and(|truth| truth.to_truth_value() == TruthValue::False);
            selectors.push(if ruled_out {
                RowSelector::skip(end - start)
            } else {
                RowSelector::select(end - start)
            });
        }
        if selectors.iter().all(|selector| !selector.skip) {
            continue;
        }
        let selection = RowSelection::from(selectors);
        kept = Some(match kept {
            Some(kept) => kept.intersection(&selection),
            None => selection,
        });
    }
    Ok(kept)
}

/// The leaf of a top-level column that is itself a leaf, not a nested type.
fn single_leaf(metadata: &ParquetMetaData, column: &str) -> Option<usize> {
    let schema = metadata.file_metadata().schema_descr();
    let mut leaves = (0..schema.num_columns())
        .filter(|&leaf| schema.column(leaf).path().parts().first().map(String::as_str) == Some(column));
    let leaf = leaves.next()?;
    (leaves.next().is_none() && schema.column(leaf).path().parts().len() == 1).then_some(leaf)
}

/// One page's statistics from the column index, in the form the row-group
/// statistics take.
fn page_statistics(index: &ColumnIndexMetaData, page: usize) -> Option<Statistics> {
    let nulls = index.null_count(page).and_then(|nulls| u64::try_from(nulls).ok());
    Some(match index {
        ColumnIndexMetaData::NONE => return None,
        ColumnIndexMetaData::BOOLEAN(pages) => Statistics::boolean(
            pages.min_value(page).copied(),
            pages.max_value(page).copied(),
            None,
            nulls,
            false,
        ),
        ColumnIndexMetaData::INT32(pages) => Statistics::int32(
            pages.min_value(page).copied(),
            pages.max_value(page).copied(),
            None,
            nulls,
            false,
        ),
        ColumnIndexMetaData::INT64(pages) => Statistics::int64(
            pages.min_value(page).copied(),
            pages.max_value(page).copied(),
            None,
            nulls,
            false,
        ),
        ColumnIndexMetaData::INT96(pages) => Statistics::int96(
            pages.min_value(page).copied(),
            pages.max_value(page).copied(),
            None,
            nulls,
            false,
        ),
        ColumnIndexMetaData::FLOAT(pages) => Statistics::float(
            pages.min_value(page).copied(),
            pages.max_value(page).copied(),
            None,
            nulls,
            false,
        ),
        ColumnIndexMetaData::DOUBLE(pages) => Statistics::double(
            pages.min_value(page).copied(),
            pages.max_value(page).copied(),
            None,
            nulls,
            false,
        ),
        ColumnIndexMetaData::BYTE_ARRAY(pages) => Statistics::byte_array(
            pages.min_value(page).map(|bytes| ByteArray::from(bytes.to_vec())),
            pages.max_value(page).map(|bytes| ByteArray::from(bytes.to_vec())),
            None,
            nulls,
            false,
        ),
        ColumnIndexMetaData::FIXED_LEN_BYTE_ARRAY(pages) => Statistics::fixed_len_byte_array(
            pages
                .min_value(page)
                .map(|bytes| FixedLenByteArray::from(ByteArray::from(bytes.to_vec()))),
            pages
                .max_value(page)
                .map(|bytes| FixedLenByteArray::from(ByteArray::from(bytes.to_vec()))),
            None,
            nulls,
            false,
        ),
    })
}

/// The byte ranges of leaf `leaf` in row group `rg_idx` that decoding
/// `selection` reads: its dictionary page, if any, and every data page holding
/// a selected row. `None` when the file has no offset index.
pub(super) fn page_ranges(
    metadata: &ParquetMetaData,
    rg_idx: usize,
    leaf: usize,
    selection: &RowSelection,
) -> Option<Vec<Range<u64>>> {
    let pages = &metadata.offset_index()?.get(rg_idx)?.get(leaf)?.page_locations;
    let (start, _) = metadata.row_group(rg_idx).column(leaf).byte_range();
    let mut ranges = Vec::new();
    if let Some(first) = pages.first()
        && u64::try_from(first.offset).ok()? != start
    {
        ranges.push(start..u64::try_from(first.offset).ok()?);
    }
    ranges.extend(selection.scan_ranges(pages));
    Some(ranges)
}


#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{ArrayRef, Int64Array, RecordBatch},
        datatypes::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema},
    };
    use bytes::Bytes;
    use daft_core::prelude::{DataType, Field};
    use daft_dsl::{lit, resolved_col};
    use parquet::{
        arrow::ArrowWriter,
        file::{
            metadata::{PageIndexPolicy, ParquetMetaDataReader},
            properties::{EnabledStatistics, WriterProperties},
        },
    };

    use super::*;

    const ROWS: i64 = 10_000;
    const PAGE_ROWS: usize = 1_000;

    /// A file of `ROWS` rows with `k` sorted (`k = row / 100`) and `v = row`,
    /// in pages of `PAGE_ROWS` rows, read back with its page index.
    fn written() -> ParquetMetaData {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k", ArrowDataType::Int64, false),
            ArrowField::new("v", ArrowDataType::Int64, false),
        ]));
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from_iter_values((0..ROWS).map(|row| row / 100))),
            Arc::new(Int64Array::from_iter_values(0..ROWS)),
        ];
        let properties = WriterProperties::builder()
            .set_data_page_row_count_limit(PAGE_ROWS)
            .set_write_batch_size(PAGE_ROWS)
            .set_statistics_enabled(EnabledStatistics::Page)
            .build();
        let mut file = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut file, schema.clone(), Some(properties)).unwrap();
        writer
            .write(&RecordBatch::try_new(schema, columns).unwrap())
            .unwrap();
        writer.close().unwrap();
        ParquetMetaDataReader::new()
            .with_page_index_policy(PageIndexPolicy::Required)
            .parse_and_finish(&Bytes::from(file))
            .unwrap()
    }

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("k", DataType::Int64),
            Field::new("v", DataType::Int64),
        ])
    }

    fn selected(selection: &RowSelection) -> usize {
        selection.row_count()
    }

    #[test]
    fn the_index_range_covers_every_column_and_offset_index() {
        let metadata = written();
        let range = index_range(&metadata).unwrap();
        for column in metadata.row_group(0).columns() {
            let offset = u64::try_from(column.offset_index_offset().unwrap()).unwrap();
            assert!(range.contains(&offset));
        }
    }

    #[test]
    fn a_key_rules_out_every_page_but_its_own() {
        let metadata = written();
        let predicate = resolved_col("k").eq(lit(55i64));

        let kept = rows_not_ruled_out(&predicate, &metadata, 0, &schema())
            .unwrap()
            .unwrap();

        // Key 55 holds rows 5,500..5,600, all in the page of rows 5,000..6,000.
        assert_eq!(selected(&kept), PAGE_ROWS);
    }

    #[test]
    fn conjuncts_on_their_own_columns_intersect() {
        let metadata = written();
        let predicate = resolved_col("k")
            .gt_eq(lit(30i64))
            .and(resolved_col("v").lt(lit(4_000i64)));

        let kept = rows_not_ruled_out(&predicate, &metadata, 0, &schema())
            .unwrap()
            .unwrap();

        assert_eq!(selected(&kept), PAGE_ROWS);
    }

    #[test]
    fn a_disjunction_of_one_columns_values_rules_out_the_pages_holding_none() {
        let metadata = written();
        let predicate = resolved_col("k")
            .eq(lit(55i64))
            .or(resolved_col("k").eq(lit(72i64)));

        let kept = rows_not_ruled_out(&predicate, &metadata, 0, &schema())
            .unwrap()
            .unwrap();

        assert_eq!(selected(&kept), 2 * PAGE_ROWS);
    }

    #[test]
    fn a_conjunct_reading_two_columns_rules_nothing_out() {
        let metadata = written();
        let predicate = resolved_col("k").eq(resolved_col("v"));

        assert!(
            rows_not_ruled_out(&predicate, &metadata, 0, &schema())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn a_selection_reads_only_its_pages_and_the_dictionary() {
        let metadata = written();
        let column = metadata.row_group(0).column(0);
        let (start, _) = column.byte_range();
        let pages = &metadata.offset_index().unwrap()[0][0].page_locations;
        let one_row = RowSelection::from(vec![
            RowSelector::skip(5_500),
            RowSelector::select(1),
            RowSelector::skip(4_499),
        ]);

        let ranges = page_ranges(&metadata, 0, 0, &one_row).unwrap();

        let page = &pages[5];
        let page_range = u64::try_from(page.offset).unwrap()
            ..u64::try_from(page.offset).unwrap()
                + u64::try_from(page.compressed_page_size).unwrap();
        let mut expected = Vec::new();
        if u64::try_from(pages[0].offset).unwrap() != start {
            expected.push(start..u64::try_from(pages[0].offset).unwrap());
        }
        expected.push(page_range);
        assert_eq!(ranges, expected);
    }
}

