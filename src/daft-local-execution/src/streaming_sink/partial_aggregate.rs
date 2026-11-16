//! A grouped partial aggregation that decides as it runs whether aggregating pays.
//!
//! Its output rows, merged per group by each aggregation's merge function, give the whole
//! aggregation of its input; any number of rows may carry the same group. That freedom
//! lets it work in one of two modes, chosen once for all workers from the first morsel,
//! by how many distinct groups a sample of its rows holds, before any row is aggregated
//! (the way the grouped aggregation chooses its strategy):
//!
//! - **Accumulating:** each morsel is aggregated and held; at the end the held partials are
//!   merged per group, in parallel by hash partition, so each group leaves as about one row.
//!   Held partials past a byte budget are merged and sent on early.
//! - **Passing through:** when the probe shows aggregating barely reduces rows, each row
//!   is sent on as its own partial state, at the cost of one projection.
//!
//! Below a join that drops the rows its build side does not match, it waits for that
//! side and takes what it publishes for each input:
//!
//! - **whether its keys repeat.** Aggregating first pays only when the join repeats this
//!   side's rows. When the build side holds each key once, as the one side of a
//!   many-to-one lookup does, the join repeats none, and grouping by a high-cardinality
//!   key below it costs more than grouping above it: every row passes on unchanged but
//!   for its state, and the join does the filtering it does today.
//! - **its key filter**, otherwise: the rows the join would discard are dropped before
//!   they are aggregated or passed on, whatever their source, as a scan would drop them.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex, OnceLock},
};

use common_daft_config::DaftExecutionConfig;
use common_error::{DaftError, DaftResult};
use common_metrics::ops::NodeType;
use common_runtime::get_compute_pool_num_threads;
use daft_core::prelude::SchemaRef;
use daft_dsl::expr::{
    bound_col,
    bound_expr::{BoundAggExpr, BoundExpr},
};
use daft_micropartition::MicroPartition;
use itertools::Itertools;
use tracing::{Span, instrument};

use super::base::{
    StreamingSink, StreamingSinkExecuteResult, StreamingSinkFinalizeOutput,
    StreamingSinkFinalizeResult, StreamingSinkOutput,
};
use crate::{
    ExecutionTaskSpawner,
    join::key_filter::{KeyRestriction, RowKeyFilter, ScanKeyFilter},
    pipeline::{InputId, NodeName},
};

/// Whether aggregating pays, decided once from the first morsel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// Every morsel is aggregated and held.
    Accumulating,
    /// Each row leaves as its own partial state.
    PassingThrough,
}

/// What every worker of the operator shares.
struct PartialAggregateParams {
    /// The partial aggregations, bound to the input.
    aggregations: Vec<BoundAggExpr>,
    /// The grouping, bound to the input.
    group_by: Vec<BoundExpr>,
    /// The grouping, then each row's own partial state, bound to the input.
    single_row_states: Vec<BoundExpr>,
    /// Each partial aggregation's merge, bound to the output.
    merges: Vec<BoundAggExpr>,
    /// The grouping columns, bound to the output.
    output_group_by: Vec<BoundExpr>,
    output_schema: SchemaRef,
    /// Rows of the first morsel sampled to decide the mode.
    probe_rows: usize,
    /// Distinct groups per sampled row above which aggregating does not pay.
    skip_ratio: f64,
    /// Bytes of held partials, per worker, past which they are merged and sent on.
    max_held_bytes: usize,
    /// The mode, set by the first morsel any worker takes.
    mode: OnceLock<Mode>,
}

impl PartialAggregateParams {
    /// The mode in force, deciding it from `input` if no morsel has yet.
    ///
    /// The first worker to ask samples its morsel while any other waits, as the grouped
    /// aggregation decides its strategy: the grouping of at most `probe_rows` rows is
    /// hashed, so no row is aggregated before the decision and one sample serves all.
    /// Either mode answers correctly, so a sample that cannot be hashed accumulates.
    fn mode(&self, input: &MicroPartition) -> Mode {
        *self.mode.get_or_init(|| {
            self.sampled_ratio(input)
                .map_or(Mode::Accumulating, |ratio| {
                    if ratio > self.skip_ratio {
                        Mode::PassingThrough
                    } else {
                        Mode::Accumulating
                    }
                })
        })
    }

    /// Distinct groups per row in a sample of `input`.
    fn sampled_ratio(&self, input: &MicroPartition) -> DaftResult<f64> {
        let sample = input.slice(0, input.len().min(self.probe_rows.max(1)))?;
        let keys = sample.eval_expression_list(&self.group_by)?;
        let mut groups = HashSet::with_capacity(sample.len());
        for batch in keys.record_batches() {
            groups.extend(batch.hash_rows()?.values().iter().copied());
        }
        #[allow(clippy::cast_precision_loss)]
        Ok(groups.len() as f64 / sample.len().max(1) as f64)
    }

    /// Merges partials per group.
    fn merge(&self, partials: Vec<MicroPartition>) -> DaftResult<MicroPartition> {
        MicroPartition::concat_or_empty(partials, self.output_schema.clone())?
            .agg(&self.merges, &self.output_group_by)
    }
}

/// One worker's partials for one input, held until the end or until they pass the byte
/// budget.
pub(crate) struct PartialAggregateState {
    input_id: InputId,
    held: Vec<MicroPartition>,
    held_bytes: usize,
}

impl PartialAggregateState {
    fn new(input_id: InputId) -> Self {
        Self {
            input_id,
            held: Vec::new(),
            held_bytes: 0,
        }
    }
}

/// What a join's key filters keep of one input.
#[derive(Clone)]
enum InputRestriction {
    /// Every row: no join published a filter.
    Everything,
    /// No row: a join's build side holds no key.
    Nothing,
    /// Every row, passed on as its own partial state: the join repeats none of them.
    Unrepeated,
    /// The rows passing this filter.
    Rows(Arc<RowKeyFilter>),
}

/// The key filters of the joins above, taken once per input for all workers.
#[derive(Default)]
struct InputKeyFilters {
    filters: Vec<ScanKeyFilter>,
    by_input: Mutex<HashMap<InputId, Arc<tokio::sync::OnceCell<InputRestriction>>>>,
}

impl InputKeyFilters {
    /// What the filters keep of `input_id`; the first worker to ask waits for each join's
    /// build side, and every other waits for it.
    async fn restriction(&self, input_id: InputId) -> InputRestriction {
        if self.filters.is_empty() {
            return InputRestriction::Everything;
        }
        let cell = self
            .by_input
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(input_id)
            .or_default()
            .clone();
        cell.get_or_init(|| async {
            let mut tests = Vec::with_capacity(self.filters.len());
            let mut unrepeated = false;
            for filter in &self.filters {
                let (restriction, repeats_keys) = filter.restriction_and_repeats(input_id).await;
                unrepeated |= repeats_keys == Some(false);
                match restriction {
                    Some(KeyRestriction::Nothing) => return InputRestriction::Nothing,
                    Some(KeyRestriction::Within { rows, .. }) => tests.push(rows),
                    None => {}
                }
            }
            if unrepeated {
                InputRestriction::Unrepeated
            } else if tests.is_empty() {
                InputRestriction::Everything
            } else {
                InputRestriction::Rows(Arc::new(RowKeyFilter::new(tests)))
            }
        })
        .await
        .clone()
    }

    /// Drops what was kept for `input_id`, once it has ended.
    fn forget(&self, input_id: InputId) {
        self.by_input
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&input_id);
    }
}

/// The operator; see the module documentation.
pub struct PartialAggregateSink {
    params: Arc<PartialAggregateParams>,
    key_filters: Arc<InputKeyFilters>,
}

impl PartialAggregateSink {
    /// A partial aggregation of rows of `input_schema` by `group_by`.
    ///
    /// # Errors
    ///
    /// Returns an error when an aggregation's partials do not combine with an aggregation of
    /// the same kind, when `group_by` is empty, or when an expression cannot be typed.
    pub fn new(
        aggregations: &[BoundAggExpr],
        group_by: &[BoundExpr],
        input_schema: &SchemaRef,
        output_schema: SchemaRef,
        cfg: &DaftExecutionConfig,
    ) -> DaftResult<Self> {
        if group_by.is_empty() {
            return Err(DaftError::InternalError(
                "A partial aggregation must group by at least one expression".to_string(),
            ));
        }
        let key_count = group_by.len();
        let output_group_by = output_schema
            .fields()
            .iter()
            .take(key_count)
            .enumerate()
            .map(|(i, field)| BoundExpr::new_unchecked(bound_col(i, field.clone())))
            .collect();
        let mut single_row_states = group_by.to_vec();
        let mut merges = Vec::with_capacity(aggregations.len());
        for (i, (aggregation, field)) in aggregations
            .iter()
            .zip(output_schema.fields().iter().skip(key_count))
            .enumerate()
        {
            let unsupported = || {
                DaftError::InternalError(format!(
                    "A partial aggregation cannot hold {aggregation}, whose partials do not combine with an aggregation of the same kind"
                ))
            };
            let state = aggregation
                .as_ref()
                .single_row_partial(input_schema)?
                .ok_or_else(unsupported)?;
            single_row_states.push(BoundExpr::new_unchecked(state.alias(field.name.as_ref())));
            let merge = aggregation
                .as_ref()
                .merge_of_partial(bound_col(key_count + i, field.clone()))
                .ok_or_else(unsupported)?;
            merges.push(BoundAggExpr::new_unchecked(merge));
        }
        Ok(Self {
            params: Arc::new(PartialAggregateParams {
                aggregations: aggregations.to_vec(),
                group_by: group_by.to_vec(),
                single_row_states,
                merges,
                output_group_by,
                output_schema,
                probe_rows: cfg.partial_aggregate_pushdown_probe_rows,
                skip_ratio: cfg.partial_aggregate_pushdown_skip_ratio,
                max_held_bytes: cfg.partial_aggregate_pushdown_max_buffer_bytes
                    / get_compute_pool_num_threads().max(1),
                mode: OnceLock::new(),
            }),
            key_filters: Arc::new(InputKeyFilters::default()),
        })
    }

    /// Drops, before aggregating, the rows these join key filters rule out.
    pub(crate) fn with_key_filters(mut self, filters: Vec<ScanKeyFilter>) -> Self {
        self.key_filters = Arc::new(InputKeyFilters {
            filters,
            by_input: Mutex::new(HashMap::new()),
        });
        self
    }
}

/// Aggregates one morsel as the shared mode says, holding what it aggregates in `state`;
/// returns the rows to send on now.
fn partially_aggregate(
    input: &MicroPartition,
    state: &mut PartialAggregateState,
    params: &PartialAggregateParams,
) -> DaftResult<Option<MicroPartition>> {
    if params.mode(input) == Mode::PassingThrough {
        return Ok(Some(input.eval_expression_list(&params.single_row_states)?));
    }
    let partial = input.agg(&params.aggregations, &params.group_by)?;
    state.held_bytes += partial.size_bytes();
    state.held.push(partial);
    if state.held_bytes <= params.max_held_bytes {
        return Ok(None);
    }
    state.held_bytes = 0;
    Ok(Some(params.merge(std::mem::take(&mut state.held))?))
}

impl StreamingSink for PartialAggregateSink {
    type State = PartialAggregateState;
    type BatchingStrategy = crate::dynamic_batching::StaticBatchingStrategy;

    #[instrument(skip_all, name = "PartialAggregateSink::execute")]
    fn execute(
        &self,
        input: MicroPartition,
        mut state: Self::State,
        _runtime_stats: Arc<Self::Stats>,
        spawner: &ExecutionTaskSpawner,
    ) -> StreamingSinkExecuteResult<Self> {
        let params = self.params.clone();
        let key_filters = self.key_filters.clone();
        spawner
            .spawn(
                async move {
                    let input = match key_filters.restriction(state.input_id).await {
                        InputRestriction::Everything => input,
                        InputRestriction::Nothing => {
                            return Ok((state, StreamingSinkOutput::NeedMoreInput(None)));
                        }
                        InputRestriction::Rows(rows) => rows.apply(input)?,
                        InputRestriction::Unrepeated => {
                            let states = (!input.is_empty())
                                .then(|| input.eval_expression_list(&params.single_row_states))
                                .transpose()?;
                            return Ok((state, StreamingSinkOutput::NeedMoreInput(states)));
                        }
                    };
                    if input.is_empty() {
                        return Ok((state, StreamingSinkOutput::NeedMoreInput(None)));
                    }
                    let output = partially_aggregate(&input, &mut state, &params)?;
                    Ok((state, StreamingSinkOutput::NeedMoreInput(output)))
                },
                Span::current(),
            )
            .into()
    }

    #[instrument(skip_all, name = "PartialAggregateSink::finalize")]
    fn finalize(
        &self,
        states: Vec<Self::State>,
        spawner: &ExecutionTaskSpawner,
    ) -> StreamingSinkFinalizeResult<Self> {
        let params = self.params.clone();
        let num_partitions = self.max_concurrency();
        if let Some(state) = states.first() {
            self.key_filters.forget(state.input_id);
        }
        spawner
            .spawn(
                async move {
                    let held = states
                        .into_iter()
                        .flat_map(|state| state.held)
                        .filter(|partial| !partial.is_empty())
                        .collect::<Vec<_>>();
                    if held.is_empty() {
                        // An input ends with a morsel, even an empty one: an operator above
                        // that has seen none of its morsels does not see it end.
                        return Ok(StreamingSinkFinalizeOutput::Finished(Some(
                            MicroPartition::empty(Some(params.output_schema.clone())),
                        )));
                    }
                    let rows: usize = held.iter().map(MicroPartition::len).sum();
                    if num_partitions <= 1 || rows <= params.probe_rows {
                        return Ok(StreamingSinkFinalizeOutput::Finished(Some(
                            params.merge(held)?,
                        )));
                    }
                    // Merge in parallel, each group in exactly one hash partition.
                    let partitions = MicroPartition::concat(held)?
                        .partition_by_hash(&params.output_group_by, num_partitions)?;
                    let mut merges = tokio::task::JoinSet::new();
                    for partition in partitions {
                        let params = params.clone();
                        merges.spawn(async move { params.merge(vec![partition]) });
                    }
                    let merged = merges
                        .join_all()
                        .await
                        .into_iter()
                        .collect::<DaftResult<Vec<_>>>()?;
                    Ok(StreamingSinkFinalizeOutput::Finished(Some(
                        MicroPartition::concat_or_empty(merged, params.output_schema.clone())?,
                    )))
                },
                Span::current(),
            )
            .into()
    }

    fn name(&self) -> NodeName {
        "PartialGroupedAggregate".into()
    }

    fn op_type(&self) -> NodeType {
        NodeType::PartialGroupByAgg
    }

    fn multiline_display(&self) -> Vec<String> {
        vec![
            format!(
                "PartialGroupedAggregate: {}",
                self.params
                    .aggregations
                    .iter()
                    .map(ToString::to_string)
                    .join(", ")
            ),
            format!(
                "Group by: {}",
                self.params
                    .group_by
                    .iter()
                    .map(ToString::to_string)
                    .join(", ")
            ),
        ]
    }

    fn make_state(&self, input_id: InputId) -> DaftResult<Self::State> {
        Ok(PartialAggregateState::new(input_id))
    }

    fn batching_strategy(&self) -> Self::BatchingStrategy {
        crate::dynamic_batching::StaticBatchingStrategy::new(
            self.morsel_size_requirement().unwrap_or_default(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use common_daft_config::DaftExecutionConfig;
    use daft_core::{count_mode::CountMode, prelude::*};
    use daft_dsl::{
        AggExpr, ExprRef,
        expr::{
            bound_col,
            bound_expr::{BoundAggExpr, BoundExpr},
        },
    };
    use daft_micropartition::MicroPartition;
    use daft_recordbatch::RecordBatch;
    use rstest::rstest;

    use super::{PartialAggregateSink, PartialAggregateState, partially_aggregate};

    const KEYS: [Option<i64>; 12] = [
        Some(1),
        Some(1),
        Some(2),
        Some(2),
        None,
        Some(3),
        Some(1),
        Some(2),
        None,
        Some(3),
        Some(4),
        Some(4),
    ];
    const VALUES: [Option<i64>; 12] = [
        Some(3),
        None,
        Some(2),
        Some(7),
        Some(0),
        None,
        Some(5),
        Some(9),
        Some(1),
        Some(4),
        None,
        None,
    ];

    fn key_field() -> Field {
        Field::new("k", DataType::Int64)
    }

    /// Rows of keys and values of `dtype`, cut into morsels of `morsel` rows.
    fn morsels(dtype: &DataType, morsel: usize) -> Vec<MicroPartition> {
        let keys = Int64Array::from_iter(key_field(), KEYS.iter().copied()).into_series();
        let values =
            Int64Array::from_iter(Field::new("v", DataType::Int64), VALUES.iter().copied())
                .into_series()
                .cast(dtype)
                .unwrap();
        let batch = RecordBatch::from_nonempty_columns(vec![keys, values]).unwrap();
        (0..batch.len())
            .step_by(morsel)
            .map(|start| {
                let slice = batch
                    .slice(start, (start + morsel).min(batch.len()))
                    .unwrap();
                MicroPartition::new_loaded(slice.schema.clone(), Arc::new(vec![slice]), None)
            })
            .collect()
    }

    fn value(dtype: &DataType) -> ExprRef {
        bound_col(1, Field::new("v", dtype.clone()))
    }

    fn sorted(partition: &MicroPartition) -> RecordBatch {
        let key = BoundExpr::new_unchecked(bound_col(0, key_field()));
        partition
            .sort(&[key], &[false], &[true])
            .unwrap()
            .concat_or_get()
            .unwrap()
            .unwrap()
    }

    fn config(probe_rows: usize, skip_ratio: f64, max_buffer_bytes: usize) -> DaftExecutionConfig {
        DaftExecutionConfig {
            partial_aggregate_pushdown_probe_rows: probe_rows,
            partial_aggregate_pushdown_skip_ratio: skip_ratio,
            partial_aggregate_pushdown_max_buffer_bytes: max_buffer_bytes,
            ..Default::default()
        }
    }

    #[rstest]
    fn merged_partials_equal_the_whole_aggregation(
        #[values(
            DataType::Int8,
            DataType::UInt16,
            DataType::Int64,
            DataType::Float32,
            DataType::Float64,
            DataType::Decimal128(10, 2)
        )]
        dtype: DataType,
        #[values("count_valid", "count_all", "count_null", "sum", "min", "max")] name: &str,
        #[values(
            (1 << 40, 0.5, usize::MAX),
            (1, 1.0, usize::MAX),
            (1, 0.0, usize::MAX),
            (1, 1.0, 1)
        )]
        mode: (usize, f64, usize),
        #[values(1, 5, 12)] morsel: usize,
    ) {
        let child = value(&dtype);
        let aggregation = match name {
            "count_valid" => AggExpr::Count(child, CountMode::Valid),
            "count_all" => AggExpr::Count(child, CountMode::All),
            "count_null" => AggExpr::Count(child, CountMode::Null),
            "sum" => AggExpr::Sum(child),
            "min" => AggExpr::Min(child),
            _ => AggExpr::Max(child),
        };
        assert_merges_to_whole(&dtype, aggregation, mode, morsel);
    }

    #[rstest]
    fn other_types_merge_to_the_whole_aggregation(
        #[values(
            (DataType::Boolean, "bool_and"),
            (DataType::Boolean, "bool_or"),
            (DataType::Boolean, "min"),
            (DataType::Utf8, "min"),
            (DataType::Utf8, "max"),
            (DataType::Date, "max"),
            (DataType::Utf8, "count_valid")
        )]
        case: (DataType, &str),
        #[values((1 << 40, 0.5, usize::MAX), (1, 0.0, usize::MAX), (1, 1.0, 1))] mode: (
            usize,
            f64,
            usize,
        ),
    ) {
        let (dtype, name) = case;
        let child = value(&dtype);
        let aggregation = match name {
            "bool_and" => AggExpr::BoolAnd(child),
            "bool_or" => AggExpr::BoolOr(child),
            "min" => AggExpr::Min(child),
            "max" => AggExpr::Max(child),
            _ => AggExpr::Count(child, CountMode::Valid),
        };
        assert_merges_to_whole(&dtype, aggregation, mode, 5);
    }

    fn assert_merges_to_whole(
        dtype: &DataType,
        aggregation: AggExpr,
        (probe_rows, skip_ratio, max_buffer_bytes): (usize, f64, usize),
        morsel: usize,
    ) {
        let morsels = morsels(dtype, morsel);
        let input_schema = morsels[0].schema();
        let aggregation = BoundAggExpr::new_unchecked(aggregation);
        let group_by = vec![BoundExpr::new_unchecked(bound_col(0, key_field()))];
        let output_schema = Arc::new(Schema::new(vec![
            key_field(),
            aggregation.as_ref().to_field(&input_schema).unwrap(),
        ]));
        let sink = PartialAggregateSink::new(
            std::slice::from_ref(&aggregation),
            &group_by,
            &input_schema,
            output_schema,
            &config(probe_rows, skip_ratio, max_buffer_bytes),
        )
        .unwrap();
        let params = &sink.params;
        let mut state = PartialAggregateState::new(0);
        let mut partials = morsels
            .iter()
            .filter_map(|input| partially_aggregate(input, &mut state, params).unwrap())
            .collect::<Vec<_>>();
        partials.extend(state.held);

        let merged = params.merge(partials).unwrap();

        let whole = MicroPartition::concat(morsels)
            .unwrap()
            .agg(&[aggregation], &group_by)
            .unwrap();
        let (merged, whole) = (sorted(&merged), sorted(&whole));
        assert_eq!(merged.schema, whole.schema);
        for i in 0..whole.num_columns() {
            assert_eq!(
                merged.get_column(i).to_arrow().unwrap().as_ref(),
                whole.get_column(i).to_arrow().unwrap().as_ref(),
            );
        }
    }

    #[rstest]
    // The first morsel holds one key twice: half as many groups as rows.
    #[case::over_the_ratio_every_row_passes_through(0.4, vec![2, 2, 2, 2, 2, 2], 0)]
    #[case::within_the_ratio_every_morsel_is_held(0.6, vec![0, 0, 0, 0, 0, 0], 6)]
    fn the_first_morsel_decides_for_every_morsel(
        #[case] skip_ratio: f64,
        #[case] sent: Vec<usize>,
        #[case] held: usize,
    ) {
        let dtype = DataType::Int64;
        let morsels = morsels(&dtype, 2);
        let input_schema = morsels[0].schema();
        let aggregation = BoundAggExpr::new_unchecked(AggExpr::Sum(value(&dtype)));
        let output_schema = Arc::new(Schema::new(vec![
            key_field(),
            aggregation.as_ref().to_field(&input_schema).unwrap(),
        ]));
        let sink = PartialAggregateSink::new(
            &[aggregation],
            &[BoundExpr::new_unchecked(bound_col(0, key_field()))],
            &input_schema,
            output_schema,
            &config(1 << 20, skip_ratio, usize::MAX),
        )
        .unwrap();
        let mut state = PartialAggregateState::new(0);

        let rows_sent = morsels
            .iter()
            .map(|input| partially_aggregate(input, &mut state, &sink.params).unwrap())
            .map(|output| output.map_or(0, |partial| partial.len()))
            .collect::<Vec<_>>();

        assert_eq!(rows_sent, sent);
        assert_eq!(state.held.len(), held);
    }
}
