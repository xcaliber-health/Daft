//! Writes the rows a row-level merge decided on.
//!
//! One stream carries every decision, so the write reads it once: rows that
//! survive go to the data writer, and rows that leave an existing file are
//! recorded by the file and position they occupy. Those positions are sorted
//! before they are written, as a delete file is read in that order, and they
//! shed to disk under memory pressure rather than growing without bound.

use std::{collections::VecDeque, sync::Arc};

use common_daft_config::DaftExecutionConfig;
use common_error::{DaftError, DaftResult};
use common_metrics::ops::NodeType;
use daft_core::prelude::*;
use daft_dsl::expr::bound_expr::BoundExpr;
use daft_logical_plan::{
    MergeActionKind,
    sink_info::{IcebergDeleteInfo, IcebergRowDeltaInfo},
};
use daft_micropartition::MicroPartition;
use daft_recordbatch::RecordBatch;
use daft_writers::{AsyncFileWriter, WriterFactory};
use tracing::{Span, instrument};

use super::blocking_sink::{
    BlockingSink, BlockingSinkFinalizeResult, BlockingSinkOutput, BlockingSinkSinkResult,
};
use crate::{
    ExecutionTaskSpawner,
    pipeline::{InputId, NodeName},
    resource_manager::{QueryMemoryScope, SpillBudget},
    sorted_merge::{MergeOrdering, MergeSource, merge_sorted_runs_streaming},
    spill::{SpillContext, SpilledRun},
};

/// Column names of the rows a row-level write reports.
const DATA_FILE_COLUMN: &str = "data_file";
const DELETE_FILE_COLUMN: &str = "delete_file";
const COUNT_COLUMNS: [&str; 4] = ["rows_kept", "rows_updated", "rows_deleted", "rows_inserted"];

/// How many rows each decision covered.
#[derive(Default, Clone, Copy)]
struct RowCounts {
    kept: i64,
    updated: i64,
    deleted: i64,
    inserted: i64,
}

impl RowCounts {
    fn add(&mut self, action: u8, rows: i64) {
        match action {
            tag if tag == MergeActionKind::Keep.tag() => self.kept += rows,
            tag if tag == MergeActionKind::Update.tag() => self.updated += rows,
            tag if tag == MergeActionKind::Delete.tag() => self.deleted += rows,
            tag if tag == MergeActionKind::Insert.tag() => self.inserted += rows,
            _ => {}
        }
    }

    fn as_columns(self, rows: usize) -> Vec<Series> {
        [self.kept, self.updated, self.deleted, self.inserted]
            .into_iter()
            .zip(COUNT_COLUMNS)
            .map(|(count, name)| {
                Int64Array::from_iter(
                    Field::new(name, DataType::Int64),
                    std::iter::repeat_n(Some(count), rows),
                )
                .into_series()
            })
            .collect()
    }
}

/// Buffered positions of removed rows, shed to disk under memory pressure.
///
/// Each shed run is sorted on its way out, so the final pass is a streaming
/// merge rather than a sort of everything at once.
#[derive(Default)]
struct DeleteBuffer {
    parts: Vec<MicroPartition>,
    buffered_bytes: u64,
    budget: Option<SpillBudget>,
    spilled: Vec<SpilledRun>,
}

impl DeleteBuffer {
    async fn push(
        &mut self,
        part: MicroPartition,
        ordering: &MergeOrdering,
        spill: Option<&SpillContext>,
        memory_scope: &QueryMemoryScope,
    ) -> DaftResult<()> {
        let added = part.size_bytes() as u64;
        self.parts.push(part);
        self.buffered_bytes += added;

        let Some(spill) = spill else {
            return Ok(());
        };
        let budget = self
            .budget
            .get_or_insert_with(|| SpillBudget::new(memory_scope.clone()));
        let requested_shed = budget.take_shed_request();
        if requested_shed == 0 && budget.try_grow(added) {
            return Ok(());
        }
        let chunk = std::mem::take(&mut self.parts);
        let sorted = sort_positions(MicroPartition::concat(chunk)?, ordering)?;
        self.spilled.push(
            spill
                .scratch()?
                .spill(vec![sorted], spill.compression())
                .await?,
        );
        // Only bytes the budget granted are returned; the morsel that met the
        // refusal was never charged.
        let accounted = self.buffered_bytes - added;
        budget.shrink(accounted);
        self.buffered_bytes = 0;
        Ok(())
    }

    fn take(&mut self) -> (Vec<MicroPartition>, Vec<SpilledRun>) {
        (
            std::mem::take(&mut self.parts),
            std::mem::take(&mut self.spilled),
        )
    }
}

fn sort_positions(part: MicroPartition, ordering: &MergeOrdering) -> DaftResult<MicroPartition> {
    part.sort(
        &ordering.sort_by,
        &ordering.descending,
        &ordering.nulls_first,
    )
}

pub(crate) struct RowDeltaWriteState {
    data: Box<dyn AsyncFileWriter<Input = MicroPartition, Result = Vec<RecordBatch>>>,
    deletes: DeleteBuffer,
    counts: RowCounts,
}

/// Expressions the sink evaluates on every incoming batch.
struct RowDeltaColumns {
    action: BoundExpr,
    data: Vec<BoundExpr>,
    provenance: Vec<BoundExpr>,
}

pub(crate) struct RowDeltaWriteSink {
    data_factory: Arc<dyn WriterFactory<Input = MicroPartition, Result = Vec<RecordBatch>>>,
    deletes: Option<Arc<IcebergDeleteInfo>>,
    columns: Arc<RowDeltaColumns>,
    ordering: Arc<MergeOrdering>,
    spill: Option<Arc<SpillContext>>,
    inflation_factor: f64,
    file_schema: SchemaRef,
}

impl RowDeltaWriteSink {
    /// Build the sink for one row-level write.
    ///
    /// # Errors
    /// If the columns the merge produced do not match what the write expects.
    pub(crate) fn try_new(
        info: &IcebergRowDeltaInfo<BoundExpr>,
        data_factory: Arc<dyn WriterFactory<Input = MicroPartition, Result = Vec<RecordBatch>>>,
        input_schema: &Schema,
        file_schema: SchemaRef,
        cfg: &DaftExecutionConfig,
    ) -> DaftResult<Self> {
        let bind = |name: &str| BoundExpr::try_new(daft_dsl::resolved_col(name), input_schema);
        let provenance = vec![bind(&info.file_index_column)?, bind(&info.position_column)?];
        let reserved = [
            info.action_column.as_str(),
            info.file_index_column.as_str(),
            info.position_column.as_str(),
        ];
        let data_names: Vec<&str> = input_schema
            .field_names()
            .filter(|name| !reserved.contains(name))
            .collect();
        // The data writer's own expressions are bound against this schema, so the
        // columns it writes have to keep their positions: they must be the leading
        // columns, with what the merge added trailing them.
        let trailing: Vec<&str> = input_schema.field_names().skip(data_names.len()).collect();
        if trailing.len() != reserved.len() || trailing.iter().any(|name| !reserved.contains(name))
        {
            return Err(DaftError::ValueError(format!(
                "a row-level write expects the table's columns first and {reserved:?} last, got {:?}",
                input_schema.field_names().collect::<Vec<&str>>()
            )));
        }
        let data = data_names
            .iter()
            .map(|name| bind(name))
            .collect::<DaftResult<Vec<BoundExpr>>>()?;

        let position_schema = Schema::new(vec![
            input_schema.get_field(&info.file_index_column)?.clone(),
            input_schema.get_field(&info.position_column)?.clone(),
        ]);
        let ordering = MergeOrdering {
            sort_by: vec![
                BoundExpr::try_new(
                    daft_dsl::resolved_col(info.file_index_column.as_str()),
                    &position_schema,
                )?,
                BoundExpr::try_new(
                    daft_dsl::resolved_col(info.position_column.as_str()),
                    &position_schema,
                )?,
            ],
            descending: vec![false, false],
            nulls_first: vec![false, false],
        };

        Ok(Self {
            data_factory,
            deletes: info.deletes.clone().map(Arc::new),
            columns: Arc::new(RowDeltaColumns {
                action: bind(&info.action_column)?,
                data,
                provenance,
            }),
            ordering: Arc::new(ordering),
            spill: SpillContext::from_config(cfg).map(Arc::new),
            inflation_factor: cfg.parquet_inflation_factor,
            file_schema,
        })
    }
}

impl BlockingSink for RowDeltaWriteSink {
    type State = RowDeltaWriteState;

    #[instrument(skip_all, name = "RowDeltaWriteSink::sink")]
    fn sink(
        &self,
        input: MicroPartition,
        mut state: Self::State,
        _runtime_stats: Arc<Self::Stats>,
        spawner: &ExecutionTaskSpawner,
    ) -> BlockingSinkSinkResult<Self> {
        let columns = self.columns.clone();
        let writes_deletes = self.deletes.is_some();
        let ordering = self.ordering.clone();
        let spill = self.spill.clone();
        let memory_scope = spawner.memory_scope().clone();
        spawner
            .spawn(
                async move {
                    let Some(batch) = input.concat_or_get()? else {
                        return Ok(state);
                    };
                    if batch.is_empty() {
                        return Ok(state);
                    }
                    let actions = batch.eval_expression(&columns.action)?;
                    let actions = actions.u8()?.clone();
                    for row in 0..batch.len() {
                        if let Some(action) = actions.get(row) {
                            state.counts.add(action, 1);
                        }
                    }

                    // Everything that is not a removal is written as data; a
                    // replacement is both, so it appears on both sides.
                    let survivors =
                        action_mask(&actions, |action| action != MergeActionKind::Delete.tag())?;
                    let data_rows = batch.mask_filter(&survivors)?;
                    if !data_rows.is_empty() {
                        let written = data_rows.eval_expression_list(&columns.data)?;
                        state
                            .data
                            .write(MicroPartition::new_loaded(
                                written.schema.clone(),
                                Arc::new(vec![written]),
                                None,
                            ))
                            .await?;
                    }

                    if writes_deletes {
                        let removed = action_mask(&actions, |action| {
                            action == MergeActionKind::Delete.tag()
                                || action == MergeActionKind::Update.tag()
                        })?;
                        let removed_rows = batch.mask_filter(&removed)?;
                        if !removed_rows.is_empty() {
                            let positions =
                                removed_rows.eval_expression_list(&columns.provenance)?;
                            state
                                .deletes
                                .push(
                                    MicroPartition::new_loaded(
                                        positions.schema.clone(),
                                        Arc::new(vec![positions]),
                                        None,
                                    ),
                                    &ordering,
                                    spill.as_deref(),
                                    &memory_scope,
                                )
                                .await?;
                        }
                    }
                    Ok(state)
                },
                Span::current(),
            )
            .into()
    }

    #[instrument(skip_all, name = "RowDeltaWriteSink::finalize")]
    fn finalize(
        &self,
        states: Vec<Self::State>,
        spawner: &ExecutionTaskSpawner,
    ) -> BlockingSinkFinalizeResult {
        let file_schema = self.file_schema.clone();
        let deletes = self.deletes.clone();
        let ordering = self.ordering.clone();
        let spill = self.spill.clone();
        let inflation_factor = self.inflation_factor;
        let merge_spawner = spawner.clone();
        spawner
            .spawn(
                async move {
                    let mut data_files: Vec<RecordBatch> = Vec::new();
                    let mut counts = RowCounts::default();
                    let mut parts: Vec<MicroPartition> = Vec::new();
                    let mut spilled: Vec<SpilledRun> = Vec::new();
                    for mut state in states {
                        data_files.extend(state.data.close().await?);
                        counts.add(MergeActionKind::Keep.tag(), state.counts.kept);
                        counts.add(MergeActionKind::Update.tag(), state.counts.updated);
                        counts.add(MergeActionKind::Delete.tag(), state.counts.deleted);
                        counts.add(MergeActionKind::Insert.tag(), state.counts.inserted);
                        let (state_parts, state_runs) = state.deletes.take();
                        parts.extend(state_parts);
                        spilled.extend(state_runs);
                    }

                    let delete_files = match deletes {
                        Some(deletes) => {
                            write_delete_files(
                                deletes.as_ref(),
                                parts,
                                spilled,
                                &ordering,
                                spill.clone(),
                                inflation_factor,
                                &merge_spawner,
                            )
                            .await?
                        }
                        None => Vec::new(),
                    };

                    let report = report_batch(&file_schema, data_files, delete_files, counts)?;
                    Ok(BlockingSinkOutput::Partitions(vec![
                        MicroPartition::new_loaded(file_schema, Arc::new(vec![report]), None),
                    ]))
                },
                Span::current(),
            )
            .into()
    }

    fn name(&self) -> NodeName {
        if self.deletes.is_some() {
            "Iceberg Row Delta Write".into()
        } else {
            "Iceberg Row Rewrite Write".into()
        }
    }

    fn op_type(&self) -> NodeType {
        NodeType::Write
    }

    fn make_state(&self, _input_id: InputId) -> DaftResult<Self::State> {
        Ok(RowDeltaWriteState {
            data: self.data_factory.create_writer(0, None)?,
            deletes: DeleteBuffer::default(),
            counts: RowCounts::default(),
        })
    }

    fn multiline_display(&self) -> Vec<String> {
        vec![
            format!("Write: {}", self.name()),
            format!(
                "Removed rows = {}",
                if self.deletes.is_some() {
                    "recorded as delete files"
                } else {
                    "dropped by rewriting their files"
                }
            ),
        ]
    }

    fn max_concurrency(&self) -> usize {
        1
    }
}

/// Mask selecting the rows whose action satisfies `wanted`.
fn action_mask(actions: &UInt8Array, wanted: impl Fn(u8) -> bool) -> DaftResult<Series> {
    let values = (0..actions.len()).map(|row| actions.get(row).map(&wanted));
    Ok(BooleanArray::from_iter("mask", values).into_series())
}

/// Merge the recorded positions and write one delete file per group.
async fn write_delete_files(
    deletes: &IcebergDeleteInfo,
    parts: Vec<MicroPartition>,
    spilled: Vec<SpilledRun>,
    ordering: &MergeOrdering,
    spill_handle: Option<Arc<SpillContext>>,
    inflation_factor: f64,
    spawner: &ExecutionTaskSpawner,
) -> DaftResult<Vec<RecordBatch>> {
    let buffered: usize = parts.iter().map(MicroPartition::len).sum();
    if buffered == 0 && spilled.is_empty() {
        return Ok(Vec::new());
    }

    let mut writer = GroupedDeleteWriter::new(deletes, inflation_factor);
    if spilled.is_empty() {
        let sorted = sort_positions(MicroPartition::concat(parts)?, ordering)?;
        for batch in sorted.record_batches() {
            writer.push(batch).await?;
        }
        return writer.finish().await;
    }

    let Some(spill_handle) = spill_handle else {
        return Err(DaftError::InternalError(
            "positions were shed to disk but spilling is not configured".to_string(),
        ));
    };
    let mut sources: Vec<MergeSource> = spilled
        .into_iter()
        .map(|run| MergeSource::Spilled(run.cursor()))
        .collect();
    if buffered > 0 {
        let sorted = sort_positions(MicroPartition::concat(parts)?, ordering)?;
        sources.push(MergeSource::Memory(VecDeque::from(
            sorted.record_batches().to_vec(),
        )));
    }

    let (tx, mut rx) = crate::channel::create_channel::<MicroPartition>(2);
    let merge_ordering = MergeOrdering {
        sort_by: ordering.sort_by.clone(),
        descending: ordering.descending.clone(),
        nulls_first: ordering.nulls_first.clone(),
    };
    let merge_spill = spill_handle.clone();
    let producer = spawner.spawn(
        async move {
            merge_sorted_runs_streaming(sources, &merge_ordering, merge_spill.as_ref(), &tx).await
        },
        Span::current(),
    );
    while let Some(part) = rx.recv().await {
        for batch in part.record_batches() {
            writer.push(batch).await?;
        }
    }
    producer.await??;
    writer.finish().await
}

/// Writes sorted positions into one delete file per group of data files.
///
/// Positions arrive ordered by file and position, and a file's group never
/// decreases, so a group is closed as soon as a row of the next one arrives.
struct GroupedDeleteWriter<'a> {
    deletes: &'a IcebergDeleteInfo,
    current: Option<OpenDeleteFile>,
    written: Vec<RecordBatch>,
    target_file_size: usize,
    inflation_factor: f64,
}

struct OpenDeleteFile {
    group: i64,
    writer: Box<dyn AsyncFileWriter<Input = MicroPartition, Result = Vec<RecordBatch>>>,
}

impl<'a> GroupedDeleteWriter<'a> {
    fn new(deletes: &'a IcebergDeleteInfo, inflation_factor: f64) -> Self {
        Self {
            deletes,
            current: None,
            written: Vec::new(),
            target_file_size: deletes.target_file_size,
            inflation_factor,
        }
    }

    async fn push(&mut self, batch: &RecordBatch) -> DaftResult<()> {
        if batch.is_empty() {
            return Ok(());
        }
        let indices = batch.get_column(0).i32()?.clone();
        let positions = batch.get_column(1).i64()?.clone();
        let mut start = 0usize;
        while start < batch.len() {
            let file_index = i64::from(indices.get(start).ok_or_else(|| {
                DaftError::ValueError("a removed row carries no file index".to_string())
            })?);
            let group = self.deletes.group_of(file_index)?;
            let mut end = start + 1;
            while end < batch.len() {
                let next = i64::from(indices.get(end).ok_or_else(|| {
                    DaftError::ValueError("a removed row carries no file index".to_string())
                })?);
                if self.deletes.group_of(next)? != group {
                    break;
                }
                end += 1;
            }
            self.open(group, file_index).await?;
            let rows = self.rows_for(&indices, &positions, start, end)?;
            if let Some(open) = self.current.as_mut() {
                open.writer
                    .write(MicroPartition::new_loaded(
                        rows.schema.clone(),
                        Arc::new(vec![rows]),
                        None,
                    ))
                    .await?;
            }
            start = end;
        }
        Ok(())
    }

    /// Rows of one group as the pair a delete file stores: the data file's path
    /// and the position within it.
    fn rows_for(
        &self,
        indices: &Int32Array,
        positions: &Int64Array,
        start: usize,
        end: usize,
    ) -> DaftResult<RecordBatch> {
        let paths = (start..end)
            .map(|row| {
                indices
                    .get(row)
                    .ok_or_else(|| {
                        DaftError::ValueError("a removed row carries no file index".to_string())
                    })
                    .and_then(|index| self.deletes.path_of(i64::from(index)))
                    .map(Some)
            })
            .collect::<DaftResult<Vec<Option<&str>>>>()?;
        let path_column = Utf8Array::from_iter("file_path", paths.into_iter()).into_series();
        let position_column = Int64Array::from_iter(
            Field::new("pos", DataType::Int64),
            (start..end).map(|row| positions.get(row)),
        )
        .into_series();
        RecordBatch::from_nonempty_columns(vec![path_column, position_column])
    }

    async fn open(&mut self, group: i64, file_index: i64) -> DaftResult<()> {
        if self
            .current
            .as_ref()
            .is_some_and(|open| open.group == group)
        {
            return Ok(());
        }
        self.close_current().await?;
        let factory = daft_writers::make_position_delete_writer_factory(
            self.deletes.writer_factory.clone(),
            file_index,
            self.target_file_size,
            self.inflation_factor,
        );
        self.current = Some(OpenDeleteFile {
            group,
            writer: factory.create_writer(0, None)?,
        });
        Ok(())
    }

    async fn close_current(&mut self) -> DaftResult<()> {
        if let Some(mut open) = self.current.take() {
            self.written.extend(open.writer.close().await?);
        }
        Ok(())
    }

    async fn finish(mut self) -> DaftResult<Vec<RecordBatch>> {
        self.close_current().await?;
        Ok(self.written)
    }
}

/// One batch reporting what the write added and how many rows each decision covered.
fn report_batch(
    schema: &SchemaRef,
    data_files: Vec<RecordBatch>,
    delete_files: Vec<RecordBatch>,
    counts: RowCounts,
) -> DaftResult<RecordBatch> {
    let files = |batches: Vec<RecordBatch>| -> DaftResult<Series> {
        let columns: Vec<Series> = batches
            .iter()
            .filter(|batch| !batch.is_empty())
            .map(|batch| batch.get_column(0).clone())
            .collect();
        if columns.is_empty() {
            Ok(Series::empty(DATA_FILE_COLUMN, &DataType::Python))
        } else {
            Series::concat(&columns.iter().collect::<Vec<&Series>>())
        }
    };
    let added_data = files(data_files)?;
    let added_deletes = files(delete_files)?;

    let data_rows = added_data.len();
    let delete_rows = added_deletes.len();
    let total = data_rows + delete_rows + 1;

    let data_column = Series::concat(&[
        &added_data.rename(DATA_FILE_COLUMN),
        &Series::full_null(DATA_FILE_COLUMN, &DataType::Python, delete_rows + 1),
    ])?;
    let delete_column = Series::concat(&[
        &Series::full_null(DELETE_FILE_COLUMN, &DataType::Python, data_rows),
        &added_deletes.rename(DELETE_FILE_COLUMN),
        &Series::full_null(DELETE_FILE_COLUMN, &DataType::Python, 1),
    ])?;

    // Counts sit on their own row, so a reader sums them across tasks without
    // counting a file's row twice.
    let mut columns = vec![data_column, delete_column];
    for (empty, counted) in RowCounts::default()
        .as_columns(total - 1)
        .into_iter()
        .zip(counts.as_columns(1))
    {
        columns.push(Series::concat(&[
            &Series::full_null(empty.name(), &DataType::Int64, total - 1),
            &counted,
        ])?);
    }
    let batch = RecordBatch::from_nonempty_columns(columns)?;
    debug_assert_eq!(batch.schema.len(), schema.len());
    Ok(batch)
}
