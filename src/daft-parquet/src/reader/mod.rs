pub(crate) mod chunk_source;
mod field_reader;
mod page_index;
mod rg_processor;
mod util;

use std::{
    collections::{BTreeMap, HashSet},
    sync::Arc,
};

use arrow::datatypes::Schema as ArrowSchema;
use chunk_source::{
    ChunkSource, ChunkSourceBuilder, LocalChunkSource, open_local_file, prepare_remote_chunk_source,
};
use common_error::DaftResult;
use common_runtime::{JoinSet, get_compute_runtime};
use daft_core::prelude::*;
use daft_dsl::{ExprRef, expr::bound_expr::BoundExpr, optimization::get_required_columns};
use daft_recordbatch::RecordBatch;
use futures::{Stream, StreamExt, stream::BoxStream};
use parquet::{
    arrow::arrow_reader::{ArrowReaderMetadata, RowSelection},
    file::metadata::ParquetMetaData,
};
use rg_processor::{process_rg_predicate_only, process_rg_with_data_cols};
use snafu::ResultExt;
use util::{cap_selection_to, project_schema, schema_from_indices};

use crate::{
    ArrowSnafu,
    helpers::{
        build_offset_row_selection, build_single_rg_delete_selection, combine_selections,
        predicate_pushable_cols, prune_row_groups, substitute_missing_cols,
    },
    metadata::{
        apply_field_ids_to_arrowrs_parquet_metadata, strip_string_types_from_parquet_metadata,
    },
    read::{ParquetReadOptions, ParquetSchemaInferenceOptions, StringEncoding},
};

pub enum ParquetSource<'a> {
    Local {
        path: &'a str,
        io_stats: Option<daft_io::IOStatsRef>,
    },
    Url {
        uri: &'a str,
        io_client: Arc<daft_io::IOClient>,
        io_stats: Option<daft_io::IOStatsRef>,
    },
}

impl ParquetSource<'_> {
    fn label(&self) -> &str {
        match self {
            Self::Local { path, .. } => path,
            Self::Url { uri, .. } => uri,
        }
    }
}

const DEFAULT_BATCH_SIZE: usize = 128 * 1024;

async fn open_chunk_source(
    source: &ParquetSource<'_>,
    opts: &ParquetReadOptions,
) -> crate::Result<(ChunkSourceBuilder, ArrowReaderMetadata)> {
    match source {
        ParquetSource::Local { path, io_stats } => {
            let page_index = opts.page_index && opts.predicate.is_some();
            let (file, file_len, arrow_metadata) =
                open_local_file(path, page_index, io_stats.clone()).await?;
            let cs = LocalChunkSource {
                path: Arc::from(*path),
                file,
                file_len,
                metadata: arrow_metadata.metadata().clone(),
                io_stats: io_stats.clone(),
            };
            Ok((ChunkSourceBuilder::Local(cs), arrow_metadata))
        }
        ParquetSource::Url {
            uri,
            io_client,
            io_stats,
        } => {
            Box::pin(prepare_remote_chunk_source(
                uri,
                io_client.clone(),
                io_stats.clone(),
                opts,
            ))
            .await
        }
    }
}

struct PreparedMetadata {
    parquet_metadata: Arc<ParquetMetaData>,
    arrow_schema: Arc<ArrowSchema>,
}

fn prepare_metadata(
    arrow_metadata: ArrowReaderMetadata,
    field_id_mapping: Option<&Arc<BTreeMap<i32, Field>>>,
    opts: ParquetSchemaInferenceOptions,
    path: &str,
) -> crate::Result<PreparedMetadata> {
    let mut parquet_metadata = arrow_metadata.metadata().clone();
    if let Some(mapping) = field_id_mapping {
        parquet_metadata =
            apply_field_ids_to_arrowrs_parquet_metadata(parquet_metadata, mapping, path)?;
    }
    let raw_encoding = opts.string_encoding == StringEncoding::Raw;
    if raw_encoding {
        parquet_metadata = strip_string_types_from_parquet_metadata(parquet_metadata, path)?;
    }
    let arrow_schema = crate::schema_inference::infer_schema_from_parquet_metadata_arrowrs(
        &parquet_metadata,
        Some(opts.coerce_int96_timestamp_unit),
        raw_encoding,
    )
    .with_context(|_| ArrowSnafu {
        path: path.to_string(),
    })?;
    Ok(PreparedMetadata {
        parquet_metadata,
        arrow_schema: Arc::new(arrow_schema),
    })
}

#[derive(Clone)]
pub(super) struct ColumnPlan {
    /// Physical Parquet column indices that must be read, including projection and filter columns.
    pub(super) read_col_indices: Vec<usize>,
    /// Physical Parquet column indices used only for predicate pushdown evaluation.
    pub(super) pred_col_indices: Vec<usize>,
    /// Physical Parquet column indices returned as data after predicate pushdown.
    pub(super) data_col_indices: Vec<usize>,
    /// Whether the predicate can be evaluated while reading Parquet row groups.
    pub(super) predicate_pushed: bool,
    /// Schema visible to callers after applying the requested projection.
    pub(super) return_daft_schema: Arc<Schema>,
    /// Schema for all columns read from Parquet, including predicate helper columns.
    pub(super) read_daft_schema: Schema,
}

fn resolve_column_plan(
    prepared: &PreparedMetadata,
    opts: &ParquetReadOptions,
) -> DaftResult<ColumnPlan> {
    let daft_schema = Schema::try_from(prepared.arrow_schema.as_ref())?;
    let user_cols = opts.columns.as_deref();

    let mut read_col_names: HashSet<String> = match user_cols {
        Some(cols) => cols.iter().cloned().collect(),
        None => daft_schema.field_names().map(str::to_string).collect(),
    };

    let (pred_cols, predicate_pushed) = match opts.predicate.as_ref() {
        Some(pred) => match predicate_pushable_cols(pred, &daft_schema) {
            Some(cols) => (Some(cols), true),
            None => {
                let extra: HashSet<String> = get_required_columns(pred)
                    .into_iter()
                    .filter(|c| daft_schema.get_field(c).is_ok())
                    .collect();
                (Some(extra), false)
            }
        },
        None => (None, false),
    };

    if let Some(filter_cols) = &pred_cols {
        read_col_names.extend(filter_cols.iter().cloned());
    }

    let read_daft_schema = project_schema(&daft_schema, &read_col_names);
    let return_daft_schema = match user_cols {
        Some(cols) => Arc::new(project_schema(
            &daft_schema,
            &cols.iter().cloned().collect(),
        )),
        None => Arc::new(daft_schema),
    };

    let arrow_fields = prepared.arrow_schema.fields();
    let read_col_indices: Vec<usize> = arrow_fields
        .iter()
        .enumerate()
        .filter(|(_, f)| read_col_names.contains(f.name()))
        .map(|(i, _)| i)
        .collect();

    let (pred_col_indices, data_col_indices): (Vec<usize>, Vec<usize>) = if predicate_pushed {
        let pred_set: HashSet<&str> = pred_cols
            .as_ref()
            .unwrap()
            .iter()
            .map(String::as_str)
            .collect();
        read_col_indices
            .iter()
            .copied()
            .partition(|&i| pred_set.contains(arrow_fields[i].name().as_str()))
    } else {
        (Vec::new(), read_col_indices.clone())
    };

    Ok(ColumnPlan {
        read_col_indices,
        pred_col_indices,
        data_col_indices,
        predicate_pushed,
        return_daft_schema,
        read_daft_schema,
    })
}

fn build_global_starts(metadata: &ParquetMetaData) -> Vec<usize> {
    let mut starts = Vec::with_capacity(metadata.num_row_groups());
    let mut acc = 0usize;
    for i in 0..metadata.num_row_groups() {
        starts.push(acc);
        acc += metadata.row_group(i).num_rows() as usize;
    }
    starts
}

fn build_base_selections(
    metadata: &ParquetMetaData,
    rg_indices: &[usize],
    opts: &ParquetReadOptions,
) -> Vec<Option<RowSelection>> {
    // With a predicate, limit is enforced post-filter, not at RG-build time.
    let global_starts = build_global_starts(metadata);
    let start_offset = opts.start_offset.unwrap_or(0);
    let delete_rows = opts.delete_rows.as_deref();
    let mut limit_remaining: Option<usize> =
        opts.predicate.is_none().then_some(opts.num_rows).flatten();
    rg_indices
        .iter()
        .map(|&rg_idx| {
            let rg_rows = metadata.row_group(rg_idx).num_rows() as usize;
            let mut sel = build_offset_selection(start_offset, global_starts[rg_idx], rg_rows);
            if let Some(deletes) = delete_rows
                && !deletes.is_empty()
            {
                sel = combine_selections(
                    sel,
                    Some(build_single_rg_delete_selection(
                        deletes,
                        global_starts[rg_idx],
                        rg_rows,
                    )),
                );
            }
            if let Some(remaining) = limit_remaining.as_mut() {
                sel = apply_limit_to_selection(sel, remaining, rg_rows);
            }
            sel
        })
        .collect()
}

/// Row-level offset selection within the RG straddling `start_offset`. RGs
/// entirely before the offset are already pruned by `prune_row_groups`, so
/// only the straddling case (or no-skip) reaches here.
fn build_offset_selection(
    global_offset: usize,
    rg_global_start: usize,
    rg_rows: usize,
) -> Option<RowSelection> {
    if global_offset > rg_global_start {
        Some(build_offset_row_selection(
            global_offset - rg_global_start,
            rg_rows,
        ))
    } else {
        None
    }
}

/// Cap the selection for the RG straddling the `num_rows` boundary. RGs past
/// the limit are already pruned by `prune_row_groups`, so we only need to
/// cap-or-passthrough — never produce an all-skip selection.
fn apply_limit_to_selection(
    sel: Option<RowSelection>,
    remaining: &mut usize,
    rg_rows: usize,
) -> Option<RowSelection> {
    let contrib = sel.as_ref().map(|s| s.row_count()).unwrap_or(rg_rows);
    if contrib > *remaining {
        let cap = *remaining;
        *remaining = 0;
        Some(cap_selection_to(sel.as_ref(), cap, rg_rows))
    } else {
        *remaining -= contrib;
        sel
    }
}

/// A pushed predicate evaluated within each row group, over its own columns,
/// before the row group's other columns are decoded.
pub(super) struct Prefilter {
    /// The predicate, bound to the schema of its columns.
    pub(super) bound: BoundExpr,
    /// The schema of the predicate's columns, in `plan.pred_col_indices` order.
    pub(super) arrow_schema: Arc<ArrowSchema>,
}

impl Prefilter {
    /// The prefilter for `opts.predicate`, when the reader pushes it and the
    /// read returns columns besides the predicate's.
    fn of(
        opts: &ParquetReadOptions,
        plan: &ColumnPlan,
        arrow_schema: &ArrowSchema,
    ) -> DaftResult<Option<Self>> {
        let Some(predicate) = opts
            .predicate
            .as_ref()
            .filter(|_| plan.predicate_pushed && !plan.data_col_indices.is_empty())
        else {
            return Ok(None);
        };
        let pred_arrow_schema = schema_from_indices(arrow_schema, &plan.pred_col_indices);
        let pred_schema = Schema::try_from(pred_arrow_schema.as_ref())?;
        let bound = BoundExpr::try_new(
            substitute_missing_cols(predicate, &pred_schema)?,
            &pred_schema,
        )?;
        Ok(Some(Self {
            bound,
            arrow_schema: pred_arrow_schema,
        }))
    }
}

/// The compressed bytes of row group `rg_idx`'s columns that a read fetches.
fn read_bytes(metadata: &ParquetMetaData, rg_idx: usize, columns: &[usize]) -> u64 {
    let row_group = metadata.row_group(rg_idx);
    field_reader::leaves_for_top_fields(metadata, columns)
        .into_iter()
        .map(|leaf| u64::try_from(row_group.column(leaf).compressed_size()).unwrap_or(0))
        .sum()
}

/// Shared per-file state for an RG-decoding task. One `Arc<RgTaskCtx>` is
/// cloned per RG task — replaces the previous fistful-of-Arcs cloning ritual.
///
/// Per-RG processor selection happens in `build_rg_stream` based on
/// `plan.data_col_indices.is_empty()`: empty → PredOnly path, non-empty →
/// Default path.
pub(super) struct RgTaskCtx {
    pub(super) path: Arc<str>,
    pub(super) chunk_source: Arc<ChunkSource>,
    pub(super) metadata: Arc<ParquetMetaData>,
    pub(super) arrow_schema: Arc<ArrowSchema>,
    pub(super) plan: ColumnPlan,
    pub(super) predicate: Option<ExprRef>,
    /// The pushed predicate, when each row group filters before decoding its
    /// other columns.
    pub(super) prefilter: Option<Prefilter>,
    /// The row groups of which the read fetches only the pages it needs,
    /// rather than whole column chunks.
    pub(super) page_fetched: HashSet<usize>,
    /// The most rows the read returns, if limited: a row group stops
    /// filtering once it has kept that many.
    pub(super) num_rows: Option<usize>,
    pub(super) chunk_size: usize,
}

/// One bounded channel per RG, drained in RG order. In-file output is always
/// RG-ordered: `maintain_order=false` at the scan layer only reorders BETWEEN
/// scan tasks; downstream code (and tests) expect file-order output within
/// a single file.
fn build_rg_stream(
    ctx: Arc<RgTaskCtx>,
    rg_indices: Vec<usize>,
    base_selections: Vec<Option<RowSelection>>,
) -> BoxStream<'static, DaftResult<RecordBatch>> {
    use tokio_stream::wrappers::ReceiverStream;

    // Capacity 1: each task runs ~one batch ahead — cross-RG overlap, bounded memory.
    let (senders, receivers): (Vec<_>, Vec<_>) = (0..base_selections.len())
        .map(|_| tokio::sync::mpsc::channel::<DaftResult<RecordBatch>>(1))
        .unzip();

    let compute = get_compute_runtime();
    let mut joinset: JoinSet<DaftResult<()>> = JoinSet::new();
    for (rg_pos, (sender, selection)) in senders.into_iter().zip(base_selections).enumerate() {
        let ctx = ctx.clone();
        let rg_idx = rg_indices[rg_pos];
        joinset.spawn_on(
            async move {
                let mut sub_stream = if ctx.plan.data_col_indices.is_empty() {
                    process_rg_predicate_only(ctx.clone(), rg_idx, selection).await
                } else {
                    process_rg_with_data_cols(ctx, rg_idx, selection).await
                };
                while let Some(item) = sub_stream.next().await {
                    if sender.send(item).await.is_err() {
                        break;
                    }
                }
                DaftResult::Ok(())
            },
            &compute,
        );
    }

    let inner_streams = receivers.into_iter().map(ReceiverStream::new);
    let merged: BoxStream<'static, DaftResult<RecordBatch>> =
        Box::pin(futures::stream::iter(inner_streams).flatten());
    common_runtime::combine_stream(merged, async move { joinset.join_all().await }).boxed()
}

fn count_only_stream(
    metadata: &ParquetMetaData,
    rg_indices: &[usize],
    num_rows: Option<usize>,
    return_schema: Arc<Schema>,
) -> DaftResult<(Arc<Schema>, BoxStream<'static, DaftResult<RecordBatch>>)> {
    let total: usize = rg_indices
        .iter()
        .map(|&i| metadata.row_group(i).num_rows() as usize)
        .sum();
    let n = num_rows.map(|n| n.min(total)).unwrap_or(total);
    let batch = RecordBatch::new_with_size(return_schema.clone(), Vec::new(), n)?;
    Ok((
        return_schema,
        futures::stream::once(async move { Ok(batch) }).boxed(),
    ))
}

fn apply_cross_rg_limit(
    stream: impl Stream<Item = DaftResult<RecordBatch>> + Send + 'static,
    limit: Option<usize>,
) -> BoxStream<'static, DaftResult<RecordBatch>> {
    let Some(limit) = limit else {
        return Box::pin(stream);
    };
    // `scan` ends the stream when the closure yields `None`, so once we hit
    // the limit, upstream RG tasks see the receiver close and abort their
    // decoders. `filter_map` would keep polling upstream forever.
    let bounded = stream.scan(limit, |remaining, res| {
        let out = match res {
            Err(e) => Some(Err(e)),
            Ok(_) if *remaining == 0 => None,
            Ok(b) if b.num_rows() <= *remaining => {
                *remaining -= b.num_rows();
                Some(Ok(b))
            }
            Ok(b) => {
                let take = std::mem::replace(remaining, 0);
                Some(b.head(take))
            }
        };
        std::future::ready(out)
    });
    Box::pin(bounded)
}

pub async fn stream_parquet(
    source: ParquetSource<'_>,
    opts: &ParquetReadOptions,
) -> DaftResult<(Arc<Schema>, BoxStream<'static, DaftResult<RecordBatch>>)> {
    let (cs_builder, arrow_metadata) = open_chunk_source(&source, opts).await?;
    let path = cs_builder.path().clone();

    let chunk_size = opts.batch_size.unwrap_or(DEFAULT_BATCH_SIZE).max(1);
    let prepared = prepare_metadata(
        arrow_metadata,
        opts.field_id_mapping.as_ref(),
        opts.schema_infer,
        &path,
    )?;
    let plan = resolve_column_plan(&prepared, opts)?;

    // Row groups are pruned by the predicate and by the prune-only filter
    // together; only the predicate goes on to filter rows.
    let prune_predicate = match (&opts.predicate, &opts.row_group_filter) {
        (Some(predicate), Some(filter)) => Some(predicate.clone().and(filter.clone())),
        (predicate, filter) => predicate.as_ref().or(filter.as_ref()).cloned(),
    };

    // Single RG-level pruning pass: user row_groups + positional (start_offset,
    // num_rows) + predicate stats. Must run before `cs_builder.build`, which is
    // what spawns remote byte fetches.
    let rg_indices = prune_row_groups(
        &prepared.parquet_metadata,
        opts.row_groups.as_deref(),
        opts.start_offset.unwrap_or(0),
        opts.num_rows,
        prune_predicate.as_ref(),
        &plan.read_daft_schema,
        source.label(),
    )?;

    // Bloom-filter pruning: drop row groups whose per-column filters prove an
    // equality/membership predicate cannot match. Additive on top of the stats
    // pruning above; reads filter bitsets via `cs_builder` before fetches spawn.
    let rg_indices = match prune_predicate.as_ref() {
        Some(predicate) => {
            let probes = crate::bloom::extract_bloom_probes(predicate);
            crate::bloom::prune_row_groups_by_bloom(
                &cs_builder,
                &prepared.parquet_metadata,
                &probes,
                rg_indices,
            )
            .await?
        }
        None => rg_indices,
    };

    if rg_indices.is_empty() {
        // Empty stream — NOT a single empty batch. Downstream sinks
        // (e.g. iceberg writer) treat any received batch as "there's
        // something to write" and emit metadata for it; a 0-row batch
        // would still land an empty snapshot.
        return Ok((
            plan.return_daft_schema.clone(),
            futures::stream::empty().boxed(),
        ));
    }
    if plan.read_col_indices.is_empty() {
        return count_only_stream(
            &prepared.parquet_metadata,
            &rg_indices,
            opts.num_rows,
            plan.return_daft_schema,
        );
    }

    // Each row group filters in its own task: the predicate's columns are
    // decoded in parallel across row groups, as the others are.
    let base_selections = build_base_selections(&prepared.parquet_metadata, &rg_indices, opts);
    let prefilter = Prefilter::of(opts, &plan, &prepared.arrow_schema)?;
    // With a page index, a filtered read fetches only the pages it needs of a
    // row group large enough for that to pay, once its kept rows are known;
    // the rest are fetched ahead whole, as without one.
    let page_fetched: HashSet<usize> =
        if prefilter.is_some() && prepared.parquet_metadata.offset_index().is_some() {
            rg_indices
                .iter()
                .copied()
                .filter(|&rg_idx| {
                    read_bytes(&prepared.parquet_metadata, rg_idx, &plan.read_col_indices)
                        > page_index::PAGE_FETCH_MIN_BYTES
                })
                .collect()
        } else {
            HashSet::new()
        };
    let prefetched: Vec<usize> = rg_indices
        .iter()
        .copied()
        .filter(|rg_idx| !page_fetched.contains(rg_idx))
        .collect();

    // Now spawn the byte-range fetches (remote) — pruned set only.
    let chunk_source = Arc::new(cs_builder.build(prepared.parquet_metadata.clone(), &prefetched));

    let return_schema = plan.return_daft_schema.clone();
    let ctx = Arc::new(RgTaskCtx {
        path,
        chunk_source,
        metadata: prepared.parquet_metadata,
        arrow_schema: prepared.arrow_schema,
        plan,
        predicate: opts.predicate.clone(),
        prefilter,
        page_fetched,
        num_rows: opts.num_rows,
        chunk_size,
    });
    let stream = build_rg_stream(ctx, rg_indices, base_selections);

    Ok((return_schema, apply_cross_rg_limit(stream, opts.num_rows)))
}
