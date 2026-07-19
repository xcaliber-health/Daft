//! Streaming merge of sorted runs.
//!
//! Combines any number of individually sorted runs — on disk or in memory —
//! into one sorted sequence using pairwise two-cursor merges, so peak memory
//! is bounded by one batch per side regardless of run sizes. Intermediate
//! merged runs stream back to disk; only the final merge materializes
//! output.

use std::collections::VecDeque;

use common_error::{DaftError, DaftResult};
use daft_core::{array::ops::build_multi_array_bicompare, prelude::UInt64Array, series::Series};
use daft_dsl::expr::bound_expr::BoundExpr;
use daft_micropartition::MicroPartition;
use daft_recordbatch::RecordBatch;

use crate::spill::{RunCursor, SpillContext, SpilledRun};

/// Sort ordering shared by every run under merge.
pub(crate) struct MergeOrdering {
    pub sort_by: Vec<BoundExpr>,
    pub descending: Vec<bool>,
    pub nulls_first: Vec<bool>,
}

/// One sorted input to the merge: a spilled run or an in-memory batch list.
pub(crate) enum MergeSource {
    Spilled(RunCursor),
    Memory(VecDeque<RecordBatch>),
}

impl MergeSource {
    async fn next_batch(&mut self) -> DaftResult<Option<RecordBatch>> {
        match self {
            Self::Spilled(cursor) => cursor.next_batch().await,
            Self::Memory(batches) => {
                while let Some(batch) = batches.pop_front() {
                    if !batch.is_empty() {
                        return Ok(Some(batch));
                    }
                }
                Ok(None)
            }
        }
    }
}

/// One side of a two-way merge: the current batch, its evaluated sort-key
/// columns, and the cursor position within it.
struct Side {
    source: MergeSource,
    batch: Option<RecordBatch>,
    keys: Vec<Series>,
    idx: usize,
}

impl Side {
    fn new(source: MergeSource) -> Self {
        Self {
            source,
            batch: None,
            keys: Vec::new(),
            idx: 0,
        }
    }

    /// Advances to the next batch when the current one is exhausted.
    /// Returns whether a batch is available.
    async fn ensure_batch(&mut self, ordering: &MergeOrdering) -> DaftResult<bool> {
        if self
            .batch
            .as_ref()
            .is_some_and(|batch| self.idx < batch.len())
        {
            return Ok(true);
        }
        match self.source.next_batch().await? {
            Some(batch) => {
                let key_table = batch.eval_expression_list(&ordering.sort_by)?;
                self.keys = (0..key_table.num_columns())
                    .map(|i| key_table.get_column(i).clone())
                    .collect();
                self.batch = Some(batch);
                self.idx = 0;
                Ok(true)
            }
            None => {
                self.batch = None;
                Ok(false)
            }
        }
    }

    /// Remaining rows of the current batch as one batch, advancing past
    /// them.
    fn take_rest(&mut self) -> DaftResult<Option<RecordBatch>> {
        let Some(batch) = self.batch.take() else {
            return Ok(None);
        };
        let rest = if self.idx == 0 {
            batch
        } else {
            batch.slice(self.idx, batch.len())?
        };
        self.idx = 0;
        Ok((!rest.is_empty()).then_some(rest))
    }
}

/// Where merged batches go: an intermediate spill run or final collection.
pub(crate) enum MergeOutput<'a> {
    Run(&'a mut crate::spill::RunWriter),
    Collect(&'a mut Vec<MicroPartition>),
}

impl MergeOutput<'_> {
    async fn emit(&mut self, batch: RecordBatch) -> DaftResult<()> {
        if batch.is_empty() {
            return Ok(());
        }
        let part = MicroPartition::new_loaded(batch.schema.clone(), Arc::new(vec![batch]), None);
        match self {
            Self::Run(writer) => writer.push(part).await,
            Self::Collect(out) => {
                out.push(part);
                Ok(())
            }
        }
    }
}

use std::sync::Arc;

/// Merges two sorted sources into `out`, preserving the shared ordering.
/// Ties prefer the first source, keeping the merge stable.
async fn merge_two(
    a: MergeSource,
    b: MergeSource,
    ordering: &MergeOrdering,
    out: &mut MergeOutput<'_>,
) -> DaftResult<()> {
    let mut a = Side::new(a);
    let mut b = Side::new(b);

    loop {
        let a_ready = a.ensure_batch(ordering).await?;
        let b_ready = b.ensure_batch(ordering).await?;
        match (a_ready, b_ready) {
            (false, false) => return Ok(()),
            (true, false) => {
                if let Some(rest) = a.take_rest()? {
                    out.emit(rest).await?;
                }
                continue;
            }
            (false, true) => {
                if let Some(rest) = b.take_rest()? {
                    out.emit(rest).await?;
                }
                continue;
            }
            (true, true) => {}
        }

        // Both sides hold a batch: walk until one side's batch is exhausted,
        // then gather the interleaved picks in one take.
        let cmp = build_multi_array_bicompare(
            &a.keys,
            &b.keys,
            &ordering.descending,
            &ordering.nulls_first,
        )?;
        let a_batch = a
            .batch
            .as_ref()
            .ok_or_else(|| DaftError::InternalError("merge side lost its batch".to_string()))?;
        let b_batch = b
            .batch
            .as_ref()
            .ok_or_else(|| DaftError::InternalError("merge side lost its batch".to_string()))?;
        let (a_len, b_len) = (a_batch.len(), b_batch.len());
        let (a_start, b_start) = (a.idx, b.idx);
        let mut picks: Vec<u64> = Vec::with_capacity((a_len - a_start) + (b_len - b_start));
        let (mut ai, mut bi) = (a_start, b_start);
        while ai < a_len && bi < b_len {
            if cmp(ai, bi) != std::cmp::Ordering::Greater {
                picks.push(ai as u64);
                ai += 1;
            } else {
                picks.push((a_len + bi) as u64);
                bi += 1;
            }
        }

        let pair = [a_batch, b_batch];
        let combined = RecordBatch::concat(pair)?;
        let indices = UInt64Array::from_vec("", picks);
        out.emit(combined.take(&indices)?).await?;
        a.idx = ai;
        b.idx = bi;
    }
}

/// Merges any number of sorted runs into fully sorted output partitions.
///
/// Pairs of runs merge into intermediate on-disk runs until two or fewer
/// remain; the final merge collects output in memory. Peak working memory
/// is bounded by one batch per active side plus the emitted chunk.
///
/// # Errors
/// Returns an error if reading, writing, or comparison fails.
pub(crate) async fn merge_sorted_runs(
    mut runs: Vec<MergeSource>,
    ordering: &MergeOrdering,
    spill: &SpillContext,
) -> DaftResult<Vec<MicroPartition>> {
    while runs.len() > 2 {
        let mut next_round: Vec<MergeSource> = Vec::with_capacity(runs.len().div_ceil(2));
        let mut iter = runs.into_iter();
        while let Some(first) = iter.next() {
            match iter.next() {
                Some(second) => {
                    let mut writer = spill.scratch()?.start_run(spill.compression())?;
                    {
                        let mut out = MergeOutput::Run(&mut writer);
                        merge_two(first, second, ordering, &mut out).await?;
                    }
                    let merged: SpilledRun = writer.finish().await?;
                    next_round.push(MergeSource::Spilled(merged.cursor()));
                }
                None => next_round.push(first),
            }
        }
        runs = next_round;
    }

    let mut out_parts: Vec<MicroPartition> = Vec::new();
    match (runs.pop(), runs.pop()) {
        (Some(only), None) => {
            let mut side = Side::new(only);
            while side.ensure_batch(ordering).await? {
                if let Some(rest) = side.take_rest()? {
                    out_parts.push(MicroPartition::new_loaded(
                        rest.schema.clone(),
                        Arc::new(vec![rest]),
                        None,
                    ));
                }
            }
        }
        (Some(second), Some(first)) => {
            let mut out = MergeOutput::Collect(&mut out_parts);
            merge_two(first, second, ordering, &mut out).await?;
        }
        (None, _) => {}
    }
    Ok(out_parts)
}

#[cfg(test)]
mod tests {
    use daft_core::{
        datatypes::{DataType, Field, Int64Array},
        prelude::Schema,
        series::IntoSeries,
    };
    use daft_dsl::{expr::bound_expr::BoundExpr, resolved_col};

    use super::*;

    fn batch(values: Vec<Option<i64>>) -> RecordBatch {
        let series = Int64Array::from_iter(Field::new("v", DataType::Int64), values).into_series();
        RecordBatch::from_nonempty_columns(vec![series]).unwrap()
    }

    fn ordering(descending: bool, nulls_first: bool) -> MergeOrdering {
        let schema = Schema::new(vec![Field::new("v", DataType::Int64)]);
        MergeOrdering {
            sort_by: vec![BoundExpr::try_new(resolved_col("v"), &schema).unwrap()],
            descending: vec![descending],
            nulls_first: vec![nulls_first],
        }
    }

    fn spill_ctx(dir: &tempfile::TempDir) -> SpillContext {
        let mut cfg = common_daft_config::DaftExecutionConfig::default();
        cfg.spill_dirs = vec![dir.path().to_str().unwrap().to_string()];
        SpillContext::from_config(&cfg).unwrap()
    }

    fn collect_values(parts: &[MicroPartition]) -> Vec<Option<i64>> {
        let mut out = Vec::new();
        for part in parts {
            for rb in part.record_batches() {
                let arr = rb.get_column(0).i64().unwrap();
                for i in 0..arr.len() {
                    out.push(arr.get(i));
                }
            }
        }
        out
    }

    #[tokio::test]
    async fn three_way_merge_matches_full_sort() {
        // Three sorted runs with interleaved ranges and multiple batches;
        // three runs force one intermediate on-disk merge pass.
        let runs = vec![
            MergeSource::Memory(VecDeque::from(vec![
                batch(vec![Some(1), Some(4)]),
                batch(vec![Some(7), Some(10)]),
            ])),
            MergeSource::Memory(VecDeque::from(vec![batch(vec![Some(2), Some(5), Some(8)])])),
            MergeSource::Memory(VecDeque::from(vec![
                batch(vec![Some(3)]),
                batch(vec![Some(6), Some(9)]),
            ])),
        ];
        let dir = tempfile::tempdir().unwrap();
        let merged = merge_sorted_runs(runs, &ordering(false, false), &spill_ctx(&dir))
            .await
            .unwrap();
        assert_eq!(
            collect_values(&merged),
            (1..=10).map(Some).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn descending_merge_with_nulls_first() {
        let runs = vec![
            MergeSource::Memory(VecDeque::from(vec![batch(vec![None, Some(9), Some(3)])])),
            MergeSource::Memory(VecDeque::from(vec![batch(vec![None, Some(6), Some(1)])])),
        ];
        let dir = tempfile::tempdir().unwrap();
        let merged = merge_sorted_runs(runs, &ordering(true, true), &spill_ctx(&dir))
            .await
            .unwrap();
        assert_eq!(
            collect_values(&merged),
            vec![None, None, Some(9), Some(6), Some(3), Some(1)]
        );
    }

    #[tokio::test]
    async fn single_and_empty_run_edge_cases() {
        let dir = tempfile::tempdir().unwrap();
        let merged = merge_sorted_runs(vec![], &ordering(false, false), &spill_ctx(&dir))
            .await
            .unwrap();
        assert!(merged.is_empty());

        let one = vec![MergeSource::Memory(VecDeque::from(vec![batch(vec![
            Some(1),
            Some(2),
        ])]))];
        let merged = merge_sorted_runs(one, &ordering(false, false), &spill_ctx(&dir))
            .await
            .unwrap();
        assert_eq!(collect_values(&merged), vec![Some(1), Some(2)]);
    }
}
