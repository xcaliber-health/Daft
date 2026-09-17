use std::sync::Arc;

use arrow::{
    array::{Array, ArrayRef},
    datatypes::Schema as ArrowSchema,
};
use common_error::DaftResult;
use common_runtime::{JoinSet, get_compute_runtime};
use daft_core::prelude::*;
use daft_dsl::expr::bound_expr::BoundExpr;
use daft_recordbatch::RecordBatch;
use futures::{StreamExt, future::try_join_all, stream::BoxStream};
use parquet::{
    arrow::arrow_reader::{RowSelection, RowSelector},
    file::metadata::ParquetMetaData,
};
use tokio::sync::mpsc;

use super::{
    RgTaskCtx,
    chunk_source::ChunkSource,
    field_reader::{decode_one_streaming, leaves_for_top_fields},
    util::{
        eval_predicate_mask, filter_arrays_by_mask, project_to_schema, record_batch_from_arrow,
        schema_from_indices,
    },
};
use crate::helpers::substitute_missing_cols;

type ColRx = mpsc::Receiver<DaftResult<ArrayRef>>;

fn err_stream<T: Send + 'static>(e: common_error::DaftError) -> BoxStream<'static, DaftResult<T>> {
    futures::stream::once(async move { Err(e) }).boxed()
}

/// Recv one chunk from each column receiver in lockstep.
///
/// - `Ok(Some(v))` — every receiver yielded; `v` is the per-col array slice for this chunk.
/// - `Err(e)` — a receiver propagated a decode error.
/// - `Ok(None)` — at least one receiver closed (stream end). Decoders close their
///   channel after the final chunk, so the first `None` we see ends the RG.
pub(super) async fn recv_one_chunk(receivers: &mut [ColRx]) -> DaftResult<Option<Vec<ArrayRef>>> {
    try_join_all(
        receivers
            .iter_mut()
            .map(|r| async { r.recv().await.transpose() }),
    )
    .await
    .map(|chunks| chunks.into_iter().collect())
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn spawn_col_decoders(
    col_indices: &[usize],
    chunk_source: &Arc<ChunkSource>,
    metadata: &Arc<ParquetMetaData>,
    arrow_schema: &Arc<ArrowSchema>,
    selection: Option<&RowSelection>,
    rg_idx: usize,
    chunk_size: usize,
    path: &Arc<str>,
) -> DaftResult<(Vec<ColRx>, JoinSet<DaftResult<()>>)> {
    // The reader picks its own per-RG access pattern (batched pre-fetch for
    // local, lazy per-column for remote). See `ChunkSource::open_rg`.
    let all_leaves: Arc<[usize]> = leaves_for_top_fields(metadata.as_ref(), col_indices).into();
    let rg_reader = chunk_source.clone().open_rg(rg_idx, all_leaves).await?;

    let compute = get_compute_runtime();
    let mut rxs = Vec::with_capacity(col_indices.len());
    let mut joinset: JoinSet<DaftResult<()>> = JoinSet::new();
    for &col_idx in col_indices {
        let (tx, rx) = mpsc::channel::<DaftResult<ArrayRef>>(1);
        let rg_reader = rg_reader.clone();
        let metadata = metadata.clone();
        let arrow_field = arrow_schema.field(col_idx).clone();
        let sel = selection.cloned();
        let path = path.clone();
        let col_leaves: Arc<[usize]> = leaves_for_top_fields(metadata.as_ref(), &[col_idx]).into();
        joinset.spawn_on(
            async move {
                let chunks = match rg_reader.read_col(col_leaves).await {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(Err(e.into())).await;
                        return DaftResult::Ok(());
                    }
                };
                decode_one_streaming(
                    chunks,
                    metadata,
                    rg_idx,
                    col_idx,
                    arrow_field,
                    sel,
                    chunk_size,
                    path,
                    tx,
                )
                .await;
                DaftResult::Ok(())
            },
            &compute,
        );
        rxs.push(rx);
    }
    Ok((rxs, joinset))
}

/// Absolute row positions of the rows a selection keeps, handed out in the same
/// chunk sizes the column decoders emit.
///
/// Positions are file ordinals: they count every row of the file, including rows
/// the selection skips, so they stay valid as row identifiers.
pub(super) struct PositionGenerator {
    selectors: std::vec::IntoIter<RowSelector>,
    next_row: i64,
    remaining_in_run: usize,
}

impl PositionGenerator {
    /// Build a generator over the rows `selection` keeps within one row group.
    ///
    /// `rg_global_start` is the file ordinal of the row group's first row;
    /// `rg_rows` is its row count, used when nothing is selected away.
    pub(super) fn new(
        rg_global_start: usize,
        rg_rows: usize,
        selection: Option<&RowSelection>,
    ) -> Self {
        let selectors: Vec<RowSelector> = match selection {
            Some(selection) => selection.iter().copied().collect(),
            None => vec![RowSelector::select(rg_rows)],
        };
        Self {
            selectors: selectors.into_iter(),
            next_row: rg_global_start as i64,
            remaining_in_run: 0,
        }
    }

    /// Return the next `n` kept positions, or fewer once the selection is exhausted.
    pub(super) fn take(&mut self, n: usize) -> ArrayRef {
        let mut positions: Vec<i64> = Vec::with_capacity(n);
        while positions.len() < n {
            if self.remaining_in_run == 0 {
                let Some(selector) = self.selectors.next() else {
                    break;
                };
                if selector.skip {
                    self.next_row += selector.row_count as i64;
                } else {
                    self.remaining_in_run = selector.row_count;
                }
                continue;
            }
            let take = (n - positions.len()).min(self.remaining_in_run);
            positions.extend(self.next_row..self.next_row + take as i64);
            self.next_row += take as i64;
            self.remaining_in_run -= take;
        }
        Arc::new(arrow::array::Int64Array::from(positions))
    }
}

/// File ordinal of the first row of `rg_idx`.
fn rg_global_start(metadata: &ParquetMetaData, rg_idx: usize) -> usize {
    (0..rg_idx)
        .map(|i| metadata.row_group(i).num_rows() as usize)
        .sum()
}

/// Arrow field for a synthesized position column. Positions always exist for a
/// decoded row, so the field is non-null.
fn position_field(name: &str) -> Arc<arrow::datatypes::Field> {
    Arc::new(arrow::datatypes::Field::new(
        name,
        arrow::datatypes::DataType::Int64,
        false,
    ))
}

struct StreamingState {
    ctx: Arc<RgTaskCtx>,
    col_receivers: Vec<ColRx>,
    offset: usize,
    /// Phase-1 predicate arrays for this RG, already mask-filtered.
    /// Indexed by `ctx.plan.pred_col_indices` position.
    filtered_pred: Vec<ArrayRef>,
    /// Row positions for this row group, when the caller asked for them.
    positions: Option<PositionGenerator>,
}

pub(super) async fn process_rg_with_data_cols(
    ctx: Arc<RgTaskCtx>,
    rg_idx: usize,
    selection: Option<RowSelection>,
    filtered_pred: Vec<ArrayRef>,
) -> BoxStream<'static, DaftResult<RecordBatch>> {
    // Default mode invariant: data_col_indices is non-empty. PredOnly handles
    // the empty case via `process_rg_predicate_only`.
    debug_assert!(!ctx.plan.data_col_indices.is_empty());

    let (col_receivers, mut col_decoders) = match spawn_col_decoders(
        &ctx.plan.data_col_indices,
        &ctx.chunk_source,
        &ctx.metadata,
        &ctx.arrow_schema,
        selection.as_ref(),
        rg_idx,
        ctx.chunk_size,
        &ctx.path,
    )
    .await
    {
        Ok(v) => v,
        Err(e) => return err_stream(e),
    };

    let positions = ctx.plan.position_column.as_ref().map(|_| {
        PositionGenerator::new(
            rg_global_start(&ctx.metadata, rg_idx),
            ctx.metadata.row_group(rg_idx).num_rows() as usize,
            selection.as_ref(),
        )
    });

    let state = StreamingState {
        ctx,
        col_receivers,
        offset: 0,
        filtered_pred,
        positions,
    };

    let stream = futures::stream::unfold(state, |mut state| async move {
        let data_chunks = match recv_one_chunk(&mut state.col_receivers).await {
            Ok(Some(v)) => v,
            Ok(None) => return None,
            Err(e) => return Some((Err(e), state)),
        };
        let chunk_rows = data_chunks[0].len();
        let ctx = &state.ctx;
        let plan = &ctx.plan;

        let mut fields = Vec::with_capacity(plan.read_col_indices.len());
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(plan.read_col_indices.len());
        for &col_idx in &plan.read_col_indices {
            fields.push(Arc::new(ctx.arrow_schema.field(col_idx).clone()));
            if let Some(pp) = plan.pred_col_indices.iter().position(|&i| i == col_idx) {
                arrays.push(state.filtered_pred[pp].slice(state.offset, chunk_rows));
            } else {
                let dp = plan
                    .data_col_indices
                    .iter()
                    .position(|&i| i == col_idx)
                    .expect("col_idx must be in pred or data set");
                arrays.push(data_chunks[dp].clone());
            }
        }

        if let (Some(name), Some(generator)) =
            (plan.position_column.as_deref(), state.positions.as_mut())
        {
            fields.push(position_field(name));
            arrays.push(generator.take(chunk_rows));
        }

        let mut daft_batch = match record_batch_from_arrow(
            Arc::new(ArrowSchema::new(fields)),
            arrays,
            ctx.path.as_ref(),
        ) {
            Ok(b) => b,
            Err(e) => return Some((Err(e), state)),
        };

        if let Some(pred) = ctx.predicate.as_ref()
            && !plan.predicate_pushed
        {
            let eval = substitute_missing_cols(pred, &daft_batch.schema)
                .and_then(|p| BoundExpr::try_new(p, &daft_batch.schema))
                .and_then(|bound| daft_batch.eval_expression(&bound))
                .and_then(|mask| daft_batch.mask_filter(&mask));
            daft_batch = match eval {
                Ok(b) => b,
                Err(e) => return Some((Err(e), state)),
            };
        }

        let projected = match project_to_schema(daft_batch, &plan.return_daft_schema) {
            Ok(b) => b,
            Err(e) => return Some((Err(e), state)),
        };

        state.offset += chunk_rows;
        Some((Ok(projected), state))
    })
    .boxed();

    common_runtime::combine_stream(stream, async move { col_decoders.join_all().await }).boxed()
}

struct PredicateOnlyState {
    ctx: Arc<RgTaskCtx>,
    chunk_arrow_schema: Arc<ArrowSchema>,
    bound_pred: BoundExpr,
    col_receivers: Vec<ColRx>,
    /// Row positions for this row group, when the caller asked for them.
    positions: Option<PositionGenerator>,
}

pub(super) async fn process_rg_predicate_only(
    ctx: Arc<RgTaskCtx>,
    rg_idx: usize,
    selection: Option<RowSelection>,
) -> BoxStream<'static, DaftResult<RecordBatch>> {
    let plan = &ctx.plan;
    let predicate = ctx
        .predicate
        .clone()
        .expect("predicate-only path requires a predicate");

    let (col_receivers, mut col_decoders) = match spawn_col_decoders(
        &plan.pred_col_indices,
        &ctx.chunk_source,
        &ctx.metadata,
        &ctx.arrow_schema,
        selection.as_ref(),
        rg_idx,
        ctx.chunk_size,
        &ctx.path,
    )
    .await
    {
        Ok(v) => v,
        Err(e) => return err_stream(e),
    };

    // Build schemas + bind predicate once per RG (was ~50µs/chunk overhead).
    let chunk_arrow_schema = schema_from_indices(&ctx.arrow_schema, &plan.pred_col_indices);
    let chunk_daft_schema = match Schema::try_from(chunk_arrow_schema.as_ref()) {
        Ok(s) => Arc::new(s),
        Err(e) => return err_stream(e),
    };
    let bound_pred = match substitute_missing_cols(&predicate, &chunk_daft_schema)
        .and_then(|p| BoundExpr::try_new(p, &chunk_daft_schema))
    {
        Ok(b) => b,
        Err(e) => return err_stream(e),
    };
    let positions = ctx.plan.position_column.as_ref().map(|_| {
        PositionGenerator::new(
            rg_global_start(&ctx.metadata, rg_idx),
            ctx.metadata.row_group(rg_idx).num_rows() as usize,
            selection.as_ref(),
        )
    });

    let state = PredicateOnlyState {
        ctx,
        chunk_arrow_schema,
        bound_pred,
        col_receivers,
        positions,
    };

    let stream = futures::stream::unfold(state, |mut state| async move {
        loop {
            let chunks = match recv_one_chunk(&mut state.col_receivers).await {
                Ok(Some(v)) => v,
                Ok(None) => return None,
                Err(e) => return Some((Err(e), state)),
            };

            let daft_batch = match record_batch_from_arrow(
                state.chunk_arrow_schema.clone(),
                chunks.clone(),
                state.ctx.path.as_ref(),
            ) {
                Ok(b) => b,
                Err(e) => return Some((Err(e), state)),
            };

            let arrow_bool = match eval_predicate_mask(&daft_batch, &state.bound_pred) {
                Ok(b) => b,
                Err(e) => return Some((Err(e), state)),
            };

            let chunk_positions = state
                .positions
                .as_mut()
                .map(|generator| generator.take(chunks[0].len()));

            if arrow_bool.true_count() == 0 {
                continue;
            }

            let mut filtered =
                match filter_arrays_by_mask(&chunks, &arrow_bool, state.ctx.path.as_ref()) {
                    Ok(f) => f,
                    Err(e) => return Some((Err(e), state)),
                };

            let mut out_schema = state.chunk_arrow_schema.clone();
            if let (Some(name), Some(chunk_positions)) = (
                state.ctx.plan.position_column.as_deref(),
                chunk_positions.as_ref(),
            ) {
                let kept = match filter_arrays_by_mask(
                    std::slice::from_ref(chunk_positions),
                    &arrow_bool,
                    state.ctx.path.as_ref(),
                ) {
                    Ok(mut arrays) => arrays.remove(0),
                    Err(e) => return Some((Err(e), state)),
                };
                let mut fields = out_schema.fields().to_vec();
                fields.push(position_field(name));
                out_schema = Arc::new(ArrowSchema::new(fields));
                filtered.push(kept);
            }

            // Build a RecordBatch over the predicate columns; project_to_schema
            // picks the subset for `return_daft_schema` (which is read_col_indices,
            // a subset of pred_col_indices in this path).
            let out_res = record_batch_from_arrow(out_schema, filtered, state.ctx.path.as_ref())
                .and_then(|b| project_to_schema(b, &state.ctx.plan.return_daft_schema));
            return Some(match out_res {
                Ok(p) => (Ok(p), state),
                Err(e) => (Err(e), state),
            });
        }
    })
    .boxed();

    common_runtime::combine_stream(stream, async move { col_decoders.join_all().await }).boxed()
}

#[cfg(test)]
mod tests {
    use arrow::array::Int64Array;

    use super::*;

    fn positions_of(array: &ArrayRef) -> Vec<i64> {
        array
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("position arrays are Int64")
            .values()
            .to_vec()
    }

    #[test]
    fn take_counts_every_row_when_nothing_is_selected_away() {
        let mut generator = PositionGenerator::new(0, 5, None);
        assert_eq!(positions_of(&generator.take(5)), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn take_starts_at_the_row_group_offset() {
        let mut generator = PositionGenerator::new(100, 3, None);
        assert_eq!(positions_of(&generator.take(3)), vec![100, 101, 102]);
    }

    #[test]
    fn take_skips_the_rows_the_selection_drops() {
        let selection = RowSelection::from(vec![
            RowSelector::skip(2),
            RowSelector::select(3),
            RowSelector::skip(1),
            RowSelector::select(2),
        ]);
        let mut generator = PositionGenerator::new(0, 8, Some(&selection));
        assert_eq!(positions_of(&generator.take(5)), vec![2, 3, 4, 6, 7]);
    }

    #[test]
    fn take_resumes_across_chunk_boundaries() {
        let selection = RowSelection::from(vec![
            RowSelector::select(2),
            RowSelector::skip(3),
            RowSelector::select(4),
        ]);
        let mut generator = PositionGenerator::new(10, 9, Some(&selection));
        assert_eq!(positions_of(&generator.take(3)), vec![10, 11, 15]);
        assert_eq!(positions_of(&generator.take(3)), vec![16, 17, 18]);
    }

    #[test]
    fn take_returns_fewer_rows_once_the_selection_is_spent() {
        let selection = RowSelection::from(vec![RowSelector::select(2), RowSelector::skip(6)]);
        let mut generator = PositionGenerator::new(0, 8, Some(&selection));
        assert_eq!(positions_of(&generator.take(4)), vec![0, 1]);
        assert!(positions_of(&generator.take(4)).is_empty());
    }

    #[test]
    fn take_handles_a_fully_skipped_selection() {
        let selection = RowSelection::from(vec![RowSelector::skip(4)]);
        let mut generator = PositionGenerator::new(0, 4, Some(&selection));
        assert!(positions_of(&generator.take(4)).is_empty());
    }

    #[test]
    fn global_start_sums_preceding_row_groups() {
        // Row-group starts are the running total of the groups before them.
        let sizes = [4usize, 6, 5];
        let starts: Vec<usize> = (0..sizes.len()).map(|i| sizes[..i].iter().sum()).collect();
        assert_eq!(starts, vec![0, 4, 10]);
    }
}
