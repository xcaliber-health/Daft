//! A grouped partial aggregation that decides as it runs whether aggregating pays.
//!
//! Its output rows, merged per group by each aggregation's merge function, give the whole
//! aggregation of its input; any number of rows may carry the same group. That freedom
//! lets it work in one of two modes, chosen once for all workers after a probe of the
//! first rows:
//!
//! - **Accumulating:** each morsel is aggregated and held; at the end the held partials are
//!   merged per group, in parallel by hash partition, so each group leaves as about one row.
//!   Held partials past a byte budget are merged and sent on early.
//! - **Passing through:** when the probe shows aggregating barely reduces rows, each row
//!   is sent on as its own partial state, at the cost of one projection.
//!
//! Below a join that drops the rows its build side does not match, it takes that join's
//! key filter and drops those rows before aggregating them or passing them on, whatever
//! their source, as a scan would: rows the join discards cost it no work.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU8, AtomicUsize, Ordering},
    },
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

/// The first rows are aggregated while it is decided whether aggregating pays.
const PROBING: u8 = 0;
/// Aggregating pays: every morsel is aggregated and held.
const ACCUMULATING: u8 = 1;
/// Aggregating does not pay: each row leaves as its own partial state.
const PASSING_THROUGH: u8 = 2;

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
    /// Rows to aggregate before deciding.
    probe_rows: usize,
    /// Aggregated rows per input row above which aggregating does not pay.
    skip_ratio: f64,
    /// Bytes of held partials, per worker, past which they are merged and sent on.
    max_held_bytes: usize,
    /// One of [`PROBING`], [`ACCUMULATING`] or [`PASSING_THROUGH`].
    mode: AtomicU8,
    /// Input rows aggregated while probing.
    probed_rows: AtomicUsize,
    /// Rows the probe's morsels aggregated to.
    probed_groups: AtomicUsize,
}

impl PartialAggregateParams {
    /// Notes a probed morsel of `rows` rows aggregated to `groups`, and decides the mode once
    /// enough rows are probed. Returns the mode in force.
    fn probe(&self, rows: usize, groups: usize) -> u8 {
        let probed_rows = self.probed_rows.fetch_add(rows, Ordering::Relaxed) + rows;
        let probed_groups = self.probed_groups.fetch_add(groups, Ordering::Relaxed) + groups;
        if probed_rows < self.probe_rows {
            return PROBING;
        }
        #[allow(clippy::cast_precision_loss)]
        let ratio = probed_groups as f64 / probed_rows as f64;
        let decided = if ratio > self.skip_ratio {
            PASSING_THROUGH
        } else {
            ACCUMULATING
        };
        match self
            .mode
            .compare_exchange(PROBING, decided, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => decided,
            Err(current) => current,
        }
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
            for filter in &self.filters {
                match filter.restriction(input_id).await {
                    Some(KeyRestriction::Nothing) => return InputRestriction::Nothing,
                    Some(KeyRestriction::Within { rows, .. }) => tests.push(rows),
                    None => {}
                }
            }
            if tests.is_empty() {
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
                mode: AtomicU8::new(PROBING),
                probed_rows: AtomicUsize::new(0),
                probed_groups: AtomicUsize::new(0),
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
    let mode = params.mode.load(Ordering::Acquire);
    if mode == PASSING_THROUGH {
        return Ok(Some(input.eval_expression_list(&params.single_row_states)?));
    }
    let partial = input.agg(&params.aggregations, &params.group_by)?;
    if mode == PROBING && params.probe(input.len(), partial.len()) == PASSING_THROUGH {
        // Partials may leave at any time; this one leaves now.
        return Ok(Some(partial));
    }
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

    #[test]
    fn a_probe_of_rows_that_do_not_repeat_passes_later_rows_through() {
        let dtype = DataType::Int64;
        let morsels = morsels(&dtype, 2);
        let input_schema = morsels[0].schema();
        let aggregation = BoundAggExpr::new_unchecked(AggExpr::Sum(value(&dtype)));
        let output_schema = Arc::new(Schema::new(vec![
            key_field(),
            aggregation.as_ref().to_field(&input_schema).unwrap(),
        ]));
        // Each of the first two morsels holds one key twice: the probe halves its rows.
        let sink = PartialAggregateSink::new(
            &[aggregation],
            &[BoundExpr::new_unchecked(bound_col(0, key_field()))],
            &input_schema,
            output_schema,
            &config(4, 0.4, usize::MAX),
        )
        .unwrap();
        let mut state = PartialAggregateState::new(0);

        let sent = morsels
            .iter()
            .map(|input| partially_aggregate(input, &mut state, &sink.params).unwrap())
            .map(|output| output.map_or(0, |partial| partial.len()))
            .collect::<Vec<_>>();

        // The probe ends after two morsels at 2 groups of 4 rows, over the ratio: the second
        // morsel's partial leaves at once, and every later row passes through.
        assert_eq!(sent, vec![0, 1, 2, 2, 2, 2]);
        assert_eq!(state.held.len(), 1);
    }
}
