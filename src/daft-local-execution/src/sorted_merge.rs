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

/// Where merged batches go: an intermediate spill run, or a bounded
/// channel consumed while the merge runs.
pub(crate) enum MergeOutput<'a> {
    Run(&'a mut crate::spill::RunWriter),
    Channel(&'a crate::channel::Sender<MicroPartition>),
}

impl MergeOutput<'_> {
    async fn emit(&mut self, batch: RecordBatch) -> DaftResult<()> {
        if batch.is_empty() {
            return Ok(());
        }
        let part = MicroPartition::new_loaded(batch.schema.clone(), Arc::new(vec![batch]), None);
        match self {
            Self::Run(writer) => writer.push(part).await,
            Self::Channel(tx) => tx.send(part).await.map_err(|_| {
                common_error::DaftError::InternalError(
                    "merged output receiver dropped before the merge completed".to_string(),
                )
            }),
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

/// Reduces sorted runs pairwise through intermediate on-disk runs until
/// at most two remain, ready for one final streaming merge.
async fn reduce_to_two(
    mut runs: Vec<MergeSource>,
    ordering: &MergeOrdering,
    spill: &SpillContext,
) -> DaftResult<Vec<MergeSource>> {
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
    Ok(runs)
}

/// Emits the final merge of at most two sorted sources into `out`.
async fn emit_final(
    mut runs: Vec<MergeSource>,
    ordering: &MergeOrdering,
    out: &mut MergeOutput<'_>,
) -> DaftResult<()> {
    match (runs.pop(), runs.pop()) {
        (Some(only), None) => {
            let mut side = Side::new(only);
            while side.ensure_batch(ordering).await? {
                if let Some(rest) = side.take_rest()? {
                    out.emit(rest).await?;
                }
            }
            Ok(())
        }
        (Some(second), Some(first)) => merge_two(first, second, ordering, out).await,
        (None, _) => Ok(()),
    }
}

/// Merges any number of sorted runs, sending each fully sorted output
/// partition to `output` as it is produced instead of collecting the
/// result in memory. Peak working memory is bounded by one batch per
/// active side plus the in-flight chunk held by the channel.
///
/// # Errors
/// Returns an error if reading, writing, or comparison fails, or if the
/// receiving side of `output` is dropped before the merge completes.
pub(crate) async fn merge_sorted_runs_streaming(
    runs: Vec<MergeSource>,
    ordering: &MergeOrdering,
    spill: &SpillContext,
    output: &crate::channel::Sender<MicroPartition>,
) -> DaftResult<()> {
    let runs = reduce_to_two(runs, ordering, spill).await?;
    let mut out = MergeOutput::Channel(output);
    emit_final(runs, ordering, &mut out).await
}

#[cfg(test)]
mod tests {
    use daft_core::{
        datatypes::{DataType, Field, Int64Array, Utf8Array},
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
        let cfg = common_daft_config::DaftExecutionConfig {
            spill_dirs: vec![dir.path().to_str().unwrap().to_string()],
            ..Default::default()
        };
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

    /// Runs the streaming merge and gathers its output, draining the
    /// channel concurrently so the bounded capacity never stalls it.
    async fn merge_and_collect(
        runs: Vec<MergeSource>,
        ordering: &MergeOrdering,
        spill: &SpillContext,
    ) -> Vec<MicroPartition> {
        let (tx, mut rx) = crate::channel::create_channel::<MicroPartition>(1);
        let producer = async {
            let result = merge_sorted_runs_streaming(runs, ordering, spill, &tx).await;
            drop(tx);
            result
        };
        let consumer = async {
            let mut parts = Vec::new();
            while let Some(part) = rx.recv().await {
                parts.push(part);
            }
            parts
        };
        let (result, parts) = tokio::join!(producer, consumer);
        result.unwrap();
        parts
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
        let merged = merge_and_collect(runs, &ordering(false, false), &spill_ctx(&dir)).await;
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
        let merged = merge_and_collect(runs, &ordering(true, true), &spill_ctx(&dir)).await;
        assert_eq!(
            collect_values(&merged),
            vec![None, None, Some(9), Some(6), Some(3), Some(1)]
        );
    }

    #[tokio::test]
    async fn single_and_empty_run_edge_cases() {
        let dir = tempfile::tempdir().unwrap();
        let merged = merge_and_collect(vec![], &ordering(false, false), &spill_ctx(&dir)).await;
        assert!(merged.is_empty());

        let one = vec![MergeSource::Memory(VecDeque::from(vec![batch(vec![
            Some(1),
            Some(2),
        ])]))];
        let merged = merge_and_collect(one, &ordering(false, false), &spill_ctx(&dir)).await;
        assert_eq!(collect_values(&merged), vec![Some(1), Some(2)]);
    }

    #[tokio::test]
    async fn cascade_writes_whole_files_for_the_tail_of_a_batch() {
        // The tail a merge emits is a slice; its size must not be the parent's.
        let rows: Vec<Option<i64>> = (0..20_000).map(Some).collect();
        let strings: Vec<Option<String>> = rows.iter().map(|v| v.map(|v| format!("row-{v:08}"))).collect();
        let make = |values: &[Option<i64>], text: &[Option<String>]| {
            let v = Int64Array::from_iter(Field::new("v", DataType::Int64), values.iter().copied()).into_series();
            let s = Utf8Array::from_iter("s", text.iter().map(|t| t.as_deref())).into_series();
            RecordBatch::from_nonempty_columns(vec![v, s]).unwrap()
        };
        let runs = vec![
            MergeSource::Memory(VecDeque::from(vec![make(&rows[..10], &strings[..10])])),
            MergeSource::Memory(VecDeque::from(vec![make(&rows[10..], &strings[10..])])),
            MergeSource::Memory(VecDeque::from(vec![make(&rows[..1], &strings[..1])])),
        ];
        let dir = tempfile::tempdir().unwrap();
        let spill = spill_ctx(&dir);
        let reduced = reduce_to_two(runs, &ordering(false, false), &spill).await.unwrap();
        assert_eq!(reduced.len(), 2);
        let files = walkdir(dir.path());
        assert_eq!(files, 1, "one merged run of 20000 short rows fits one spill file");
    }

    fn walkdir(path: &std::path::Path) -> usize {
        std::fs::read_dir(path)
            .unwrap()
            .map(|entry| {
                let entry = entry.unwrap();
                if entry.file_type().unwrap().is_dir() {
                    walkdir(&entry.path())
                } else {
                    1
                }
            })
            .sum()
    }

    #[tokio::test]
    async fn streaming_merge_errors_when_receiver_drops() {
        let runs = vec![
            MergeSource::Memory(VecDeque::from(vec![batch(vec![Some(1), Some(2)])])),
            MergeSource::Memory(VecDeque::from(vec![batch(vec![Some(3), Some(4)])])),
        ];
        let dir = tempfile::tempdir().unwrap();
        let (tx, rx) = crate::channel::create_channel::<MicroPartition>(1);
        drop(rx);
        let err = merge_sorted_runs_streaming(runs, &ordering(false, false), &spill_ctx(&dir), &tx)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("receiver dropped"));
    }
}
