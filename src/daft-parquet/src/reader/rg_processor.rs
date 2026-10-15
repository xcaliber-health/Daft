use std::sync::Arc;

use arrow::{
    array::{Array, ArrayRef, AsArray, BooleanArray, new_empty_array},
    compute::{concat, prep_null_mask_filter},
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
use snafu::ResultExt;
use tokio::sync::mpsc;

use super::{
    Prefilter, RgTaskCtx,
    chunk_source::{ChunkSource, RgReader},
    field_reader::{decode_one_streaming, leaves_for_top_fields},
    page_index::{page_ranges, rows_not_ruled_out},
    util::{
        eval_predicate_mask, filter_arrays_by_mask, project_to_schema, record_batch_from_arrow,
        schema_from_indices, truncate_mask_to_n_trues,
    },
};
use crate::{
    ArrowSnafu,
    helpers::{refine_selection, substitute_missing_cols},
};

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
    fetch_pages: bool,
) -> DaftResult<(Vec<ColRx>, JoinSet<DaftResult<()>>)> {
    let all_leaves: Arc<[usize]> = leaves_for_top_fields(metadata.as_ref(), col_indices).into();
    let rg_reader = if fetch_pages {
        // Only the pages `selection` reads, and their dictionaries.
        let wanted = all_leaves
            .iter()
            .map(|&leaf| {
                let ranges = selection
                    .and_then(|selection| page_ranges(metadata, rg_idx, leaf, selection))
                    .unwrap_or_else(|| {
                        let (start, len) = metadata.row_group(rg_idx).column(leaf).byte_range();
                        std::iter::once(start..start + len).collect()
                    });
                (leaf, ranges)
            })
            .collect();
        RgReader::PreFetched(Arc::new(chunk_source.read_ranges(wanted).await?))
    } else {
        // The reader picks its own per-RG access pattern (batched pre-fetch for
        // local, lazy per-column for remote). See `ChunkSource::open_rg`.
        chunk_source.clone().open_rg(rg_idx, all_leaves).await?
    };

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

/// The decoded bytes a row selection must skip, for each run of kept rows it
/// holds, to pay for itself.
///
/// A selection skips decoding the values outside it, but it takes a step for
/// every run, and without a page index it still decompresses every page.
/// Measured over 10M-row files, narrow (8-byte) columns decode faster whole
/// once kept rows fall in more than one run per ~1 KB skipped, while wide
/// string columns (~4 KB a row) gain from the selection at every density.
/// Below this bound the row group is decoded whole, then masked.
const BYTES_SKIPPED_PER_RUN: usize = 1024;

/// How a row group's other columns are decoded once its predicate is known.
struct RowGroupDecode {
    /// The rows to decode.
    selection: Option<RowSelection>,
    /// The predicate's columns, already filtered to the kept rows.
    filtered_pred: Vec<ArrayRef>,
    /// The kept rows among those decoded, when the row group is decoded whole
    /// and masked rather than read through a selection.
    mask: Option<BooleanArray>,
}

/// Decodes a row group's predicate columns, evaluates the predicate, and
/// chooses how its other columns are decoded: through a selection of the
/// kept rows where the values it skips repay its runs, else whole, then
/// masked.
async fn prefilter_row_group(
    ctx: &Arc<RgTaskCtx>,
    prefilter: &Prefilter,
    rg_idx: usize,
    base: Option<RowSelection>,
) -> DaftResult<RowGroupDecode> {
    let plan = &ctx.plan;
    // With a page index, the rows on pages whose statistics rule the
    // predicate out are known not to be kept, and are never fetched.
    let fetch_pages = ctx.page_fetched.contains(&rg_idx);
    let pred_selection = match ctx.predicate.as_ref().filter(|_| fetch_pages) {
        Some(predicate) => {
            match rows_not_ruled_out(predicate, &ctx.metadata, rg_idx, &plan.read_daft_schema)? {
                Some(pages) => Some(match &base {
                    Some(base) => base.intersection(&pages),
                    None => pages,
                }),
                None => base,
            }
        }
        None => base,
    };
    let total = pred_selection.as_ref().map_or_else(
        || ctx.metadata.row_group(rg_idx).num_rows() as usize,
        RowSelection::row_count,
    );
    if total == 0 {
        return Ok(RowGroupDecode {
            selection: pred_selection,
            filtered_pred: plan
                .pred_col_indices
                .iter()
                .map(|&col_idx| new_empty_array(ctx.arrow_schema.field(col_idx).data_type()))
                .collect(),
            mask: None,
        });
    }
    let (mut receivers, _decoders) = spawn_col_decoders(
        &plan.pred_col_indices,
        &ctx.chunk_source,
        &ctx.metadata,
        &ctx.arrow_schema,
        pred_selection.as_ref(),
        rg_idx,
        ctx.chunk_size,
        &ctx.path,
        ctx.page_fetched.contains(&rg_idx),
    )
    .await?;

    let mut remaining = ctx.num_rows.unwrap_or(usize::MAX);
    let mut masks: Vec<BooleanArray> = Vec::new();
    let mut filtered_by_col: Vec<Vec<ArrayRef>> = vec![Vec::new(); plan.pred_col_indices.len()];
    let mut processed = 0usize;
    while remaining > 0 {
        let Some(chunks) = recv_one_chunk(&mut receivers).await? else {
            break;
        };
        let batch =
            record_batch_from_arrow(prefilter.arrow_schema.clone(), chunks.clone(), &ctx.path)?;
        let mut mask = eval_predicate_mask(&batch, &prefilter.bound)?;
        // A null predicate keeps no row; the selection and run count read
        // only the mask's values, so nulls become false.
        if mask.null_count() > 0 {
            mask = prep_null_mask_filter(&mask);
        }
        if mask.true_count() > remaining {
            mask = truncate_mask_to_n_trues(&mask, remaining);
        }
        for (col, filtered) in filter_arrays_by_mask(&chunks, &mask, &ctx.path)?
            .into_iter()
            .enumerate()
        {
            filtered_by_col[col].push(filtered);
        }
        processed += mask.len();
        remaining -= mask.true_count();
        masks.push(mask);
    }
    // Dropping the receivers closes the channels, which ends the decoders.
    drop(receivers);
    if processed < total {
        masks.push(BooleanArray::from(vec![false; total - processed]));
    }
    let mask = concat_booleans(&masks, &ctx.path)?;

    let filtered_pred = plan
        .pred_col_indices
        .iter()
        .zip(&filtered_by_col)
        .map(|(&col_idx, chunks)| {
            let refs: Vec<&dyn Array> = chunks.iter().map(AsRef::as_ref).collect();
            if refs.is_empty() {
                Ok(new_empty_array(ctx.arrow_schema.field(col_idx).data_type()))
            } else {
                concat(&refs)
                    .context(ArrowSnafu {
                        path: ctx.path.to_string(),
                    })
                    .map_err(common_error::DaftError::from)
            }
        })
        .collect::<DaftResult<Vec<_>>>()?;

    let runs = mask.values().set_slices().count();
    let kept = RowSelection::from_filters(std::slice::from_ref(&mask));
    let kept = match &pred_selection {
        Some(read) => refine_selection(read, &kept),
        None => kept,
    };
    // With a page index, a selection that leaves most of the other columns'
    // pages unfetched pays for itself whatever its runs.
    let selection_skips_pages = fetch_pages && {
        let read = pred_selection
            .clone()
            .unwrap_or_else(|| RowSelection::from(vec![RowSelector::select(total)]));
        let data_leaves = leaves_for_top_fields(ctx.metadata.as_ref(), &plan.data_col_indices);
        let fetched = |selection: &RowSelection| -> u64 {
            data_leaves
                .iter()
                .filter_map(|&leaf| page_ranges(&ctx.metadata, rg_idx, leaf, selection))
                .flatten()
                .map(|range| range.end - range.start)
                .sum()
        };
        fetched(&kept).saturating_mul(2) <= fetched(&read)
    };
    if selection_skips_pages
        || selection_pays(
            runs,
            total - mask.true_count(),
            data_bytes_per_row(ctx, rg_idx),
        )
    {
        Ok(RowGroupDecode {
            selection: Some(kept),
            filtered_pred,
            mask: None,
        })
    } else {
        Ok(RowGroupDecode {
            selection: pred_selection,
            filtered_pred,
            mask: Some(mask),
        })
    }
}

/// Whether decoding through a selection of `runs` runs of kept rows, skipping
/// `skipped_rows` rows of `bytes_per_row` bytes each, costs less than decoding
/// the row group whole and masking it.
fn selection_pays(runs: usize, skipped_rows: usize, bytes_per_row: usize) -> bool {
    runs.saturating_mul(BYTES_SKIPPED_PER_RUN) <= skipped_rows.saturating_mul(bytes_per_row)
}

/// The uncompressed bytes a row of the row group's data columns takes, on
/// average, from the row group's own metadata.
fn data_bytes_per_row(ctx: &RgTaskCtx, rg_idx: usize) -> usize {
    let row_group = ctx.metadata.row_group(rg_idx);
    let rows = usize::try_from(row_group.num_rows()).unwrap_or(0).max(1);
    let bytes: i64 = leaves_for_top_fields(ctx.metadata.as_ref(), &ctx.plan.data_col_indices)
        .into_iter()
        .map(|leaf| row_group.column(leaf).uncompressed_size())
        .sum();
    usize::try_from(bytes).unwrap_or(0) / rows
}

/// One mask over the row group from its chunks' masks, which have no nulls.
fn concat_booleans(masks: &[BooleanArray], path: &str) -> DaftResult<BooleanArray> {
    match masks {
        [] => Ok(BooleanArray::from(Vec::<bool>::new())),
        [only] => Ok(only.clone()),
        _ => {
            let refs: Vec<&dyn Array> = masks.iter().map(|mask| mask as &dyn Array).collect();
            Ok(concat(&refs)
                .context(ArrowSnafu {
                    path: path.to_string(),
                })
                .map_err(common_error::DaftError::from)?
                .as_boolean()
                .clone())
        }
    }
}

struct StreamingState {
    ctx: Arc<RgTaskCtx>,
    col_receivers: Vec<ColRx>,
    /// Kept rows emitted so far, which index the filtered predicate columns.
    offset: usize,
    /// Rows decoded so far, which index `mask`.
    row: usize,
    /// The predicate's columns for this row group, already filtered to the
    /// kept rows. Indexed by `ctx.plan.pred_col_indices` position.
    filtered_pred: Vec<ArrayRef>,
    /// The kept rows among those decoded, when the row group is decoded whole.
    mask: Option<BooleanArray>,
}

pub(super) async fn process_rg_with_data_cols(
    ctx: Arc<RgTaskCtx>,
    rg_idx: usize,
    selection: Option<RowSelection>,
) -> BoxStream<'static, DaftResult<RecordBatch>> {
    // Default mode invariant: data_col_indices is non-empty. PredOnly handles
    // the empty case via `process_rg_predicate_only`.
    debug_assert!(!ctx.plan.data_col_indices.is_empty());

    let decode = match &ctx.prefilter {
        Some(prefilter) => match prefilter_row_group(&ctx, prefilter, rg_idx, selection).await {
            Ok(decode) => decode,
            Err(e) => return err_stream(e),
        },
        None => RowGroupDecode {
            selection,
            filtered_pred: Vec::new(),
            mask: None,
        },
    };

    let (col_receivers, mut col_decoders) = match spawn_col_decoders(
        &ctx.plan.data_col_indices,
        &ctx.chunk_source,
        &ctx.metadata,
        &ctx.arrow_schema,
        decode.selection.as_ref(),
        rg_idx,
        ctx.chunk_size,
        &ctx.path,
        ctx.page_fetched.contains(&rg_idx),
    )
    .await
    {
        Ok(v) => v,
        Err(e) => return err_stream(e),
    };

    let state = StreamingState {
        ctx,
        col_receivers,
        offset: 0,
        row: 0,
        filtered_pred: decode.filtered_pred,
        mask: decode.mask,
    };

    let stream = futures::stream::unfold(state, |mut state| async move {
        // Rows emitted from this chunk: all of them, or those its mask keeps.
        let (data_chunks, chunk_rows) = loop {
            let data_chunks = match recv_one_chunk(&mut state.col_receivers).await {
                Ok(Some(v)) => v,
                Ok(None) => return None,
                Err(e) => return Some((Err(e), state)),
            };
            let Some(mask) = state.mask.as_ref() else {
                let rows = data_chunks[0].len();
                break (data_chunks, rows);
            };
            let chunk_mask = mask.slice(state.row, data_chunks[0].len());
            state.row += chunk_mask.len();
            let kept = chunk_mask.true_count();
            if kept == 0 {
                continue;
            }
            match filter_arrays_by_mask(&data_chunks, &chunk_mask, &state.ctx.path) {
                Ok(filtered) => break (filtered, kept),
                Err(e) => return Some((Err(e), state)),
            }
        };
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
        false,
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
    let state = PredicateOnlyState {
        ctx,
        chunk_arrow_schema,
        bound_pred,
        col_receivers,
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

            if arrow_bool.true_count() == 0 {
                continue;
            }

            let filtered =
                match filter_arrays_by_mask(&chunks, &arrow_bool, state.ctx.path.as_ref()) {
                    Ok(f) => f,
                    Err(e) => return Some((Err(e), state)),
                };

            // Build a RecordBatch over the predicate columns; project_to_schema
            // picks the subset for `return_daft_schema` (which is read_col_indices,
            // a subset of pred_col_indices in this path).
            let out_res = record_batch_from_arrow(
                state.chunk_arrow_schema.clone(),
                filtered,
                state.ctx.path.as_ref(),
            )
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
    use super::*;

    #[test]
    fn a_selection_pays_where_the_values_it_skips_repay_its_runs() {
        let cases = [
            // (runs, skipped rows, bytes per row, pays)
            (0, 0, 8, true),
            (1, 99_000, 8, true),
            (300, 9_999_700, 8, true),
            (10, 990, 8, false),
            (500, 500, 8, false),
            (500, 500, 4_000, true),
            (1, 1, 1_024, true),
            (1, 1, 1_023, false),
        ];
        for (runs, skipped, bytes, pays) in cases {
            assert_eq!(
                selection_pays(runs, skipped, bytes),
                pays,
                "{runs} runs, {skipped} rows of {bytes} B"
            );
        }
    }

    #[test]
    fn huge_counts_saturate_rather_than_overflow() {
        assert!(selection_pays(1, usize::MAX, usize::MAX));
        assert!(!selection_pays(usize::MAX, 1, 1));
    }
}
