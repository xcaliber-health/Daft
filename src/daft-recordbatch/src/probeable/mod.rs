mod int_table;
mod probe_table;
mod probes;

use std::sync::Arc;

use common_error::DaftResult;
use daft_core::prelude::{
    DataType, Int8Type, Int16Type, Int32Type, Int64Type, SchemaRef, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use int_table::ProbeTableBuilder as IntProbeTableBuilder;
use probe_table::ProbeTableBuilder;
use probes::{ProbeContent, ProbeExists, ProbeIndices};

use crate::RecordBatch;

struct ArrowTableEntry(Vec<arrow::array::ArrayRef>);

pub fn make_probeable_builder(
    schema: SchemaRef,
    nulls_equal_aware: Option<&Vec<bool>>,
    track_indices: bool,
) -> DaftResult<Box<dyn ProbeableBuilder>> {
    let physical_type = (schema.len() == 1).then(|| schema[0].dtype.to_physical());

    // TODO: Null handling when nulls can be equal should be specially handled.
    // Ideally via:
    // - Storing Option<T> instead of T in the hash table, but doubles key size
    // - Specially tracking row_idx of nulls in another field (potentially in a nested struct?)
    let nulls_equal = nulls_equal_aware
        .map(|v| v.iter().any(|is_null_equal| *is_null_equal))
        .unwrap_or(false);
    let physical_type = if !nulls_equal { physical_type } else { None };

    if track_indices {
        make_probeable_builder_for_physical_type::<ProbeIndices>(
            physical_type,
            nulls_equal_aware,
            schema,
        )
    } else {
        make_probeable_builder_for_physical_type::<ProbeExists>(
            physical_type,
            nulls_equal_aware,
            schema,
        )
    }
}

fn make_probeable_builder_for_physical_type<T: ProbeContent>(
    physical_type: Option<DataType>,
    nulls_equal_aware: Option<&Vec<bool>>,
    schema: SchemaRef,
) -> DaftResult<Box<dyn ProbeableBuilder>> {
    match physical_type {
        Some(DataType::Int8) => Ok(Box::new(IntProbeTableBuilder::<Int8Type, T>::new()?)),
        Some(DataType::Int16) => Ok(Box::new(IntProbeTableBuilder::<Int16Type, T>::new()?)),
        Some(DataType::Int32) => Ok(Box::new(IntProbeTableBuilder::<Int32Type, T>::new()?)),
        Some(DataType::Int64) => Ok(Box::new(IntProbeTableBuilder::<Int64Type, T>::new()?)),
        Some(DataType::UInt8) => Ok(Box::new(IntProbeTableBuilder::<UInt8Type, T>::new()?)),
        Some(DataType::UInt16) => Ok(Box::new(IntProbeTableBuilder::<UInt16Type, T>::new()?)),
        Some(DataType::UInt32) => Ok(Box::new(IntProbeTableBuilder::<UInt32Type, T>::new()?)),
        Some(DataType::UInt64) => Ok(Box::new(IntProbeTableBuilder::<UInt64Type, T>::new()?)),
        _ => Ok(Box::new(ProbeTableBuilder::<T>::new(
            schema,
            nulls_equal_aware,
        )?)),
    }
}

pub trait ProbeableBuilder: Send + Sync {
    fn add_table(&mut self, table: &RecordBatch) -> DaftResult<()>;
    fn build(self: Box<Self>) -> Arc<dyn Probeable>;
}

pub struct IndicesMapper<'a> {
    table_idx_shift: usize,
    lower_mask: u64,
    idx_iter: Box<dyn Iterator<Item = Option<&'a [u64]>> + 'a>,
}

impl<'a> IndicesMapper<'a> {
    pub fn new(
        idx_iter: Box<dyn Iterator<Item = Option<&'a [u64]>> + 'a>,
        table_idx_shift: usize,
        lower_mask: u64,
    ) -> Self {
        Self {
            table_idx_shift,
            lower_mask,
            idx_iter,
        }
    }

    pub fn make_iter(self) -> impl Iterator<Item = Option<impl Iterator<Item = (u32, u64)> + 'a>> {
        let table_idx_shift = self.table_idx_shift;
        let lower_mask = self.lower_mask;
        self.idx_iter.map(move |indices| match indices {
            Some(indices) => {
                let inner_iter = indices.iter().map(move |idx| {
                    let table_idx = (idx >> table_idx_shift) as u32;
                    let row_idx = idx & lower_mask;
                    (table_idx, row_idx)
                });
                Some(inner_iter)
            }
            None => None,
        })
    }
}

pub trait Probeable: Send + Sync {
    /// Probe_indices returns an iterator of optional iterators. The outer iterator iterates over the rows of the right table.
    /// The inner iterator, if present, iterates over the rows of the left table that match the right row.
    /// Otherwise, if the inner iterator is None, indicates that the right row has no matches.
    /// NOTE: This function only works if track_indices is true.
    fn probe_indices(&'_ self, table: RecordBatch) -> DaftResult<IndicesMapper<'_>>;

    /// Probe_exists returns an iterator of booleans. The iterator iterates over the rows of the right table.
    fn probe_exists<'a>(
        &'a self,
        table: RecordBatch,
    ) -> DaftResult<Box<dyn Iterator<Item = bool> + 'a>>;

    /// Whether some key matches more than one row added, so a probe row can match many.
    fn repeats_keys(&self) -> bool;
}

#[derive(Clone)]
pub struct ProbeState {
    probeable: Arc<dyn Probeable>,
    record_batches: Vec<RecordBatch>,
}

impl ProbeState {
    pub fn new(probeable: Arc<dyn Probeable>, record_batches: Vec<RecordBatch>) -> Self {
        Self {
            probeable,
            record_batches,
        }
    }

    /// Returns an iterator of booleans. The iterator iterates over the rows of the input table.
    /// True if the right row has a match in the left table, false otherwise.
    pub fn probe_exists(
        &self,
        record_batch: RecordBatch,
    ) -> DaftResult<impl Iterator<Item = bool>> {
        self.probeable.probe_exists(record_batch)
    }

    /// Returns an iterator of optional iterators.
    /// The outer iterator iterates over the rows of the input table.
    /// The inner iterator, if present, iterates over the rows of the build table that match the input row.
    pub fn probe_indices(
        &self,
        record_batch: RecordBatch,
    ) -> DaftResult<impl Iterator<Item = Option<impl Iterator<Item = (u32, u64)>>>> {
        self.probeable
            .probe_indices(record_batch)
            .map(|indices_mapper| indices_mapper.make_iter())
    }

    pub fn get_record_batches(&self) -> &[RecordBatch] {
        &self.record_batches
    }

    /// Whether some key matches more than one row, so a probe row can match many.
    pub fn repeats_keys(&self) -> bool {
        self.probeable.repeats_keys()
    }
}

#[cfg(test)]
mod tests {
    use daft_core::prelude::*;
    use rstest::rstest;

    use super::{ProbeState, make_probeable_builder};
    use crate::RecordBatch;

    fn ints(name: &str, values: &[Option<i64>]) -> Series {
        Int64Array::from_iter(Field::new(name, DataType::Int64), values.iter().copied())
            .into_series()
    }

    fn texts(name: &str, values: &[Option<&str>]) -> Series {
        Utf8Array::from_iter(name, values.iter().copied()).into_series()
    }

    /// Whether a table built from `columns`, added in two halves, repeats a key.
    fn repeats(columns: Vec<Series>, nulls_equal: Option<Vec<bool>>) -> bool {
        let batch = RecordBatch::from_nonempty_columns(columns).unwrap();
        let mut builder =
            make_probeable_builder(batch.schema.clone(), nulls_equal.as_ref(), true).unwrap();
        let half = batch.len() / 2;
        builder.add_table(&batch.slice(0, half).unwrap()).unwrap();
        builder
            .add_table(&batch.slice(half, batch.len()).unwrap())
            .unwrap();
        builder.build().repeats_keys()
    }

    #[rstest]
    #[case::unique_ints(vec![ints("k", &[Some(1), Some(2), Some(3), Some(4)])], false)]
    #[case::an_int_repeated_across_adds(vec![ints("k", &[Some(1), Some(2), Some(3), Some(1)])], true)]
    #[case::null_int_keys_match_nothing(vec![ints("k", &[None, Some(2), None, Some(4)])], false)]
    #[case::unique_texts(vec![texts("k", &[Some("a"), Some("b"), Some("c"), Some("d")])], false)]
    #[case::a_text_repeated(vec![texts("k", &[Some("a"), Some("b"), Some("a"), Some("d")])], true)]
    #[case::null_text_keys_match_nothing(vec![texts("k", &[None, Some("b"), None, Some("d")])], false)]
    #[case::unique_pairs_of_repeated_columns(
        vec![ints("a", &[Some(1), Some(1), Some(2), Some(2)]), ints("b", &[Some(1), Some(2), Some(1), Some(2)])],
        false
    )]
    #[case::a_repeated_pair(
        vec![ints("a", &[Some(1), Some(1), Some(2), Some(1)]), ints("b", &[Some(1), Some(2), Some(1), Some(1)])],
        true
    )]
    fn a_built_table_says_whether_some_key_matches_two_rows(
        #[case] columns: Vec<Series>,
        #[case] expected: bool,
    ) {
        assert_eq!(repeats(columns, None), expected);
    }

    #[test]
    fn null_keys_that_compare_equal_repeat() {
        let keys = ints("k", &[None, Some(2), None, Some(4)]);

        assert!(repeats(vec![keys], Some(vec![true])));
    }

    #[test]
    fn a_probe_state_reports_its_table() {
        let batch =
            RecordBatch::from_nonempty_columns(vec![ints("k", &[Some(1), Some(1)])]).unwrap();
        let mut builder = make_probeable_builder(batch.schema.clone(), None, true).unwrap();
        builder.add_table(&batch).unwrap();

        let state = ProbeState::new(builder.build(), vec![batch]);

        assert!(state.repeats_keys());
    }
}
