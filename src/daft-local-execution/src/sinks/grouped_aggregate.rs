use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

use common_daft_config::DaftExecutionConfig;
use common_error::DaftResult;
use common_metrics::ops::NodeType;
use daft_core::prelude::SchemaRef;
use daft_dsl::expr::{
    bound_col,
    bound_expr::{BoundAggExpr, BoundExpr},
};
use daft_micropartition::MicroPartition;
use itertools::Itertools;
use tracing::{Span, instrument};

use super::blocking_sink::{
    BlockingSink, BlockingSinkFinalizeResult, BlockingSinkOutput, BlockingSinkSinkResult,
};
use crate::{
    ExecutionTaskSpawner,
    pipeline::{InputId, NodeName},
    resource_manager::{MemoryManager, SpillBudget},
    spill::{SpillContext, SpilledRun},
};

#[derive(Clone, Debug)]
pub(crate) enum AggStrategy {
    // TODO: This would probably benefit from doing sharded aggs.
    AggThenPartition,
    PartitionThenAgg(usize),
    PartitionOnly,
}

impl AggStrategy {
    fn execute_strategy(
        &self,
        inner_states: &mut [Option<SinglePartitionAggregateState>],
        input: MicroPartition,
        params: &GroupedAggregateParams,
    ) -> DaftResult<()> {
        match self {
            Self::AggThenPartition => Self::execute_agg_then_partition(inner_states, input, params),
            Self::PartitionThenAgg(threshold) => {
                Self::execute_partition_then_agg(inner_states, input, params, *threshold)
            }
            Self::PartitionOnly => Self::execute_partition_only(inner_states, input, params),
        }
    }

    fn execute_agg_then_partition(
        inner_states: &mut [Option<SinglePartitionAggregateState>],
        input: MicroPartition,
        params: &GroupedAggregateParams,
    ) -> DaftResult<()> {
        let agged = input.agg(
            params.partial_agg_exprs.as_slice(),
            params.group_by.as_slice(),
        )?;
        let partitioned =
            agged.partition_by_hash(params.final_group_by.as_slice(), inner_states.len())?;
        for (p, state) in partitioned.into_iter().zip(inner_states.iter_mut()) {
            state.get_or_insert_default().push_partial(p);
        }
        Ok(())
    }

    fn execute_partition_then_agg(
        inner_states: &mut [Option<SinglePartitionAggregateState>],
        input: MicroPartition,
        params: &GroupedAggregateParams,
        partial_agg_threshold: usize,
    ) -> DaftResult<()> {
        let partitioned =
            input.partition_by_hash(params.group_by.as_slice(), inner_states.len())?;
        for (p, state) in partitioned.into_iter().zip(inner_states.iter_mut()) {
            let state = state.get_or_insert_default();
            if state.unaggregated_size + p.len() >= partial_agg_threshold {
                let mut unaggregated = std::mem::take(&mut state.unaggregated);
                for drained in &unaggregated {
                    state.buffered_bytes = state
                        .buffered_bytes
                        .saturating_sub(drained.size_bytes() as u64);
                }
                state.unaggregated_size = 0;
                unaggregated.push(p);
                let aggregated = MicroPartition::concat(unaggregated)?.agg(
                    params.partial_agg_exprs.as_slice(),
                    params.group_by.as_slice(),
                )?;
                state.push_partial(aggregated);
            } else {
                state.push_raw(p);
            }
        }
        Ok(())
    }

    fn execute_partition_only(
        inner_states: &mut [Option<SinglePartitionAggregateState>],
        input: MicroPartition,
        params: &GroupedAggregateParams,
    ) -> DaftResult<()> {
        let partitioned =
            input.partition_by_hash(params.group_by.as_slice(), inner_states.len())?;
        for (p, state) in partitioned.into_iter().zip(inner_states.iter_mut()) {
            state.get_or_insert_default().push_raw(p);
        }
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct SinglePartitionAggregateState {
    partially_aggregated: Vec<MicroPartition>,
    unaggregated: Vec<MicroPartition>,
    unaggregated_size: usize,
    /// In-memory bytes currently buffered in the two vectors above.
    buffered_bytes: u64,
    /// Runs of partially aggregated state shed to disk under pressure.
    spilled_partial: Vec<SpilledRun>,
    /// Runs of raw input shed to disk under pressure.
    spilled_raw: Vec<SpilledRun>,
}

impl SinglePartitionAggregateState {
    fn push_partial(&mut self, part: MicroPartition) {
        self.buffered_bytes += part.size_bytes() as u64;
        self.partially_aggregated.push(part);
    }

    fn push_raw(&mut self, part: MicroPartition) {
        self.buffered_bytes += part.size_bytes() as u64;
        self.unaggregated_size += part.len();
        self.unaggregated.push(part);
    }

    /// Drains all buffered partitions, returning them split by kind along
    /// with the bytes they occupied.
    fn drain_buffered(&mut self) -> (Vec<MicroPartition>, Vec<MicroPartition>, u64) {
        let drained = self.buffered_bytes;
        self.buffered_bytes = 0;
        self.unaggregated_size = 0;
        (
            std::mem::take(&mut self.partially_aggregated),
            std::mem::take(&mut self.unaggregated),
            drained,
        )
    }
}

pub(crate) enum GroupedAggregateState {
    Accumulating {
        inner_states: Vec<Option<SinglePartitionAggregateState>>,
        strategy: Option<AggStrategy>,
        partial_agg_threshold: usize,
        high_cardinality_threshold_ratio: f64,
        budget: Option<SpillBudget>,
        buffered_total: u64,
    },
    Done,
}

impl GroupedAggregateState {
    fn new(
        num_partitions: usize,
        partial_agg_threshold: usize,
        high_cardinality_threshold_ratio: f64,
    ) -> Self {
        let inner_states = (0..num_partitions).map(|_| None).collect::<Vec<_>>();
        Self::Accumulating {
            inner_states,
            strategy: None,
            partial_agg_threshold,
            high_cardinality_threshold_ratio,
            budget: None,
            buffered_total: 0,
        }
    }

    async fn push(
        &mut self,
        input: MicroPartition,
        params: &GroupedAggregateParams,
        global_strategy_lock: &Arc<Mutex<Option<AggStrategy>>>,
        memory_manager: &Arc<MemoryManager>,
    ) -> DaftResult<()> {
        let Self::Accumulating {
            inner_states,
            strategy,
            partial_agg_threshold,
            high_cardinality_threshold_ratio,
            budget,
            buffered_total,
        } = self
        else {
            panic!("GroupedAggregateSink should be in Accumulating state");
        };

        // If we have determined a strategy, execute it.
        if let Some(strategy) = strategy {
            strategy.execute_strategy(inner_states, input, params)?;
        } else {
            // Otherwise, determine the strategy and execute
            let decided_strategy = Self::determine_agg_strategy(
                &input,
                params,
                *high_cardinality_threshold_ratio,
                *partial_agg_threshold,
                strategy,
                global_strategy_lock,
            )?;
            decided_strategy.execute_strategy(inner_states, input, params)?;
        }

        if let Some(spill) = &params.spill {
            let budget = budget.get_or_insert_with(|| SpillBudget::new(memory_manager.clone()));
            Self::reconcile_and_maybe_spill(inner_states, budget, buffered_total, spill).await?;
        }
        Ok(())
    }

    /// Reconciles this worker's buffered bytes with the shared budget and,
    /// when growth is denied, sheds the largest buffered hash partitions to
    /// disk until the growth fits or nothing buffered remains.
    ///
    /// Invariant on return: `buffered_total` equals the bytes currently
    /// buffered across `inner_states` and mirrors the bytes held from the
    /// shared budget.
    /// Sheds the largest buffered hash partition to disk. Returns the bytes
    /// drained, or `None` when nothing is buffered.
    async fn shed_largest(
        inner_states: &mut [Option<SinglePartitionAggregateState>],
        spill: &SpillContext,
    ) -> DaftResult<Option<u64>> {
        let largest = inner_states
            .iter_mut()
            .flatten()
            .filter(|state| state.buffered_bytes > 0)
            .max_by_key(|state| state.buffered_bytes);
        let Some(state) = largest else {
            return Ok(None);
        };
        let (partial, raw, drained) = state.drain_buffered();
        if !partial.is_empty() {
            let run = spill.scratch()?.spill(partial, spill.compression()).await?;
            state.spilled_partial.push(run);
        }
        if !raw.is_empty() {
            let run = spill.scratch()?.spill(raw, spill.compression()).await?;
            state.spilled_raw.push(run);
        }
        Ok(Some(drained))
    }

    async fn reconcile_and_maybe_spill(
        inner_states: &mut [Option<SinglePartitionAggregateState>],
        budget: &mut SpillBudget,
        buffered_total: &mut u64,
        spill: &SpillContext,
    ) -> DaftResult<()> {
        // Concurrent holders may have asked this one to shed toward its
        // fair share; honor that before growing further.
        let mut requested_shed = budget.take_shed_request();
        while requested_shed > 0 {
            let Some(drained) = Self::shed_largest(inner_states, spill).await? else {
                break;
            };
            let accounted = drained.min(*buffered_total);
            budget.shrink(accounted);
            *buffered_total -= accounted;
            requested_shed = requested_shed.saturating_sub(drained);
        }

        loop {
            let buffered: u64 = inner_states
                .iter()
                .flatten()
                .map(|state| state.buffered_bytes)
                .sum();
            if buffered <= *buffered_total {
                budget.shrink(*buffered_total - buffered);
                *buffered_total = buffered;
                return Ok(());
            }
            let growth = buffered - *buffered_total;
            if budget.try_grow(growth) {
                *buffered_total = buffered;
                return Ok(());
            }

            // Budget denied: shed the largest buffered partition and retry.
            // If nothing is left to shed, record the growth anyway —
            // execution must proceed, and the debt keeps pressure on other
            // operators.
            let largest = inner_states
                .iter_mut()
                .flatten()
                .filter(|state| state.buffered_bytes > 0)
                .max_by_key(|state| state.buffered_bytes);
            let Some(state) = largest else {
                budget.grow_unchecked(growth);
                *buffered_total = buffered;
                return Ok(());
            };
            let (partial, raw, drained) = state.drain_buffered();
            if !partial.is_empty() {
                let run = spill.scratch()?.spill(partial, spill.compression()).await?;
                state.spilled_partial.push(run);
            }
            if !raw.is_empty() {
                let run = spill.scratch()?.spill(raw, spill.compression()).await?;
                state.spilled_raw.push(run);
            }
            // Release only the shed bytes that were previously accounted;
            // bytes newer than the last reconcile never entered the budget.
            let accounted = drained.min(*buffered_total);
            budget.shrink(accounted);
            *buffered_total -= accounted;
        }
    }

    fn determine_agg_strategy(
        input: &MicroPartition,
        params: &GroupedAggregateParams,
        high_cardinality_threshold_ratio: f64,
        partial_agg_threshold: usize,
        local_strategy_cache: &mut Option<AggStrategy>,
        global_strategy_lock: &Arc<Mutex<Option<AggStrategy>>>,
    ) -> DaftResult<AggStrategy> {
        let mut global_strategy = global_strategy_lock.lock().unwrap();
        // If some other worker has determined a strategy, use that.
        if let Some(global_strat) = global_strategy.as_ref() {
            *local_strategy_cache = Some(global_strat.clone());
            return Ok(global_strat.clone());
        }

        // Else determine the strategy.
        let groupby = input.eval_expression_list(params.group_by.as_slice())?;

        let groupkey_hashes = groupby
            .record_batches()
            .iter()
            .map(|t| t.hash_rows())
            .collect::<DaftResult<Vec<_>>>()?;
        let estimated_num_groups = groupkey_hashes
            .iter()
            .flatten()
            .collect::<HashSet<_>>()
            .len();

        let decided_strategy = if estimated_num_groups as f64 / input.len() as f64
            >= high_cardinality_threshold_ratio
        {
            AggStrategy::PartitionThenAgg(partial_agg_threshold)
        } else {
            AggStrategy::AggThenPartition
        };

        *local_strategy_cache = Some(decided_strategy.clone());
        *global_strategy = Some(decided_strategy.clone());
        Ok(decided_strategy)
    }

    fn finalize(&mut self) -> Vec<Option<SinglePartitionAggregateState>> {
        let res = if let Self::Accumulating { inner_states, .. } = self {
            std::mem::take(inner_states)
        } else {
            panic!("GroupedAggregateSink should be in Accumulating state");
        };
        *self = Self::Done;
        res
    }
}

struct GroupedAggregateParams {
    // The original aggregations and group by expressions
    original_aggregations: Vec<BoundAggExpr>,
    group_by: Vec<BoundExpr>,
    // The expressions for to be used for partial aggregation
    partial_agg_exprs: Vec<BoundAggExpr>,
    // The expressions for the final aggregation
    final_agg_exprs: Vec<BoundAggExpr>,
    final_group_by: Vec<BoundExpr>,
    final_projections: Vec<BoundExpr>,
    // Spill configuration; absent when spilling is disabled.
    spill: Option<SpillContext>,
}

pub struct GroupedAggregateSink {
    grouped_aggregate_params: Arc<GroupedAggregateParams>,
    partial_agg_threshold: usize,
    high_cardinality_threshold_ratio: f64,
    global_strategy_lock: Arc<Mutex<Option<AggStrategy>>>,
}

impl GroupedAggregateSink {
    pub fn new(
        aggregations: &[BoundAggExpr],
        group_by: &[BoundExpr],
        input_schema: &SchemaRef,
        cfg: &DaftExecutionConfig,
    ) -> DaftResult<Self> {
        let (partial_agg_exprs, final_agg_exprs, final_projections) =
            daft_local_plan::agg::populate_aggregation_stages_bound(
                aggregations,
                input_schema,
                group_by,
            )?;

        // MapGroups cannot be decomposed into partial/final stages — it must see the full
        // group in one pass, so it always uses PartitionOnly.  AggFn has no single-pass
        // path, so the two cannot coexist in the same aggregation.
        let has_map_groups = aggregations
            .iter()
            .any(|agg| matches!(agg.as_ref(), daft_dsl::AggExpr::MapGroups { .. }));
        let has_agg_fn = aggregations
            .iter()
            .any(|agg| matches!(agg.as_ref(), daft_dsl::AggExpr::AggFn { .. }));
        if has_map_groups && has_agg_fn {
            return Err(common_error::DaftError::ValueError(
                "Cannot mix MapGroups (Python UDFs) and extension aggregations (AggFn) \
                 in the same aggregation; split them into separate operations."
                    .to_string(),
            ));
        }

        let final_group_by = if !partial_agg_exprs.is_empty() {
            group_by
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    let field = e.as_ref().to_field(input_schema)?;
                    Ok(BoundExpr::new_unchecked(bound_col(i, field)))
                })
                .collect::<DaftResult<Vec<_>>>()?
        } else {
            group_by.to_vec()
        };

        let strategy = if has_map_groups {
            // Always use partition-only for MapGroups so that we only hash-partition
            // the data by the group keys and then run the original MapGroups
            // aggregation once per partition in `finalize`.
            Some(AggStrategy::PartitionOnly)
        } else if partial_agg_exprs.is_empty() && !final_agg_exprs.is_empty() {
            Some(AggStrategy::PartitionOnly)
        } else {
            None
        };

        Ok(Self {
            grouped_aggregate_params: Arc::new(GroupedAggregateParams {
                original_aggregations: aggregations.to_vec(),
                group_by: group_by.to_vec(),
                partial_agg_exprs,
                final_agg_exprs,
                final_group_by,
                final_projections,
                spill: SpillContext::from_config(cfg),
            }),
            partial_agg_threshold: cfg.partial_aggregation_threshold,
            high_cardinality_threshold_ratio: cfg.high_cardinality_aggregation_threshold,
            global_strategy_lock: Arc::new(Mutex::new(strategy)),
        })
    }

    fn num_partitions(&self) -> usize {
        self.max_concurrency()
    }
}

impl BlockingSink for GroupedAggregateSink {
    type State = GroupedAggregateState;
    #[instrument(skip_all, name = "GroupedAggregateSink::sink")]
    fn sink(
        &self,
        input: MicroPartition,
        mut state: Self::State,
        _runtime_stats: Arc<Self::Stats>,
        spawner: &ExecutionTaskSpawner,
    ) -> BlockingSinkSinkResult<Self> {
        let params = self.grouped_aggregate_params.clone();
        let strategy_lock = self.global_strategy_lock.clone();
        let memory_manager = spawner.memory_manager().clone();
        spawner
            .spawn(
                async move {
                    state
                        .push(input, &params, &strategy_lock, &memory_manager)
                        .await?;
                    Ok(state)
                },
                Span::current(),
            )
            .into()
    }

    #[instrument(skip_all, name = "GroupedAggregateSink::finalize")]
    fn finalize(
        &self,
        states: Vec<Self::State>,
        spawner: &ExecutionTaskSpawner,
    ) -> BlockingSinkFinalizeResult {
        let params = self.grouped_aggregate_params.clone();
        let num_partitions = self.num_partitions();
        spawner
            .spawn(
                async move {
                    let mut state_iters = states
                        .into_iter()
                        .map(|mut state| state.finalize().into_iter())
                        .collect::<Vec<_>>();

                    let mut per_partition_finalize_tasks = tokio::task::JoinSet::new();
                    for _ in 0..num_partitions {
                        let per_partition_state = state_iters
                            .iter_mut()
                            .map(|state| {
                                state.next().expect(
                                "GroupedAggregateState should have SinglePartitionAggregateState",
                            )
                            })
                            .collect::<Vec<_>>();
                        let params = params.clone();
                        per_partition_finalize_tasks.spawn(async move {
                            let mut unaggregated = vec![];
                            let mut partially_aggregated = vec![];
                            for state in per_partition_state.into_iter().flatten() {
                                unaggregated.extend(state.unaggregated);
                                partially_aggregated.extend(state.partially_aggregated);
                                // Restore any state this partition shed to
                                // disk under memory pressure, in the same
                                // buckets it was shed from.
                                for run in state.spilled_raw {
                                    unaggregated.extend(run.read_back().await?);
                                }
                                for run in state.spilled_partial {
                                    partially_aggregated.extend(run.read_back().await?);
                                }
                            }

                            // If we have no partially aggregated partitions, aggregate the unaggregated partitions using the original aggregations
                            if params.partial_agg_exprs.is_empty() && !unaggregated.is_empty() {
                                let concated = MicroPartition::concat(unaggregated)?;
                                let agged = concated
                                    .agg(&params.original_aggregations, &params.group_by)?;
                                Ok(agged)
                            }
                            // If we have no unaggregated partitions, finalize the partially aggregated partitions
                            else if unaggregated.is_empty() {
                                let concated = MicroPartition::concat(partially_aggregated)?;
                                let agged = concated
                                    .agg(&params.final_agg_exprs, &params.final_group_by)?;
                                let projected =
                                    agged.eval_expression_list(&params.final_projections)?;
                                Ok(projected)
                            }
                            // Otherwise, partially aggregate the unaggregated partitions, concatenate them with the partially aggregated partitions, and finalize the result.
                            else {
                                let leftover_partial_agg = MicroPartition::concat(unaggregated)?
                                    .agg(&params.partial_agg_exprs, &params.group_by)?;
                                partially_aggregated.push(leftover_partial_agg);
                                let concated = MicroPartition::concat(partially_aggregated)?;
                                let agged = concated
                                    .agg(&params.final_agg_exprs, &params.final_group_by)?;
                                let projected =
                                    agged.eval_expression_list(&params.final_projections)?;
                                Ok(projected)
                            }
                        });
                    }
                    let results = per_partition_finalize_tasks
                        .join_all()
                        .await
                        .into_iter()
                        .collect::<DaftResult<Vec<_>>>()?;
                    Ok(BlockingSinkOutput::Partitions(results))
                },
                Span::current(),
            )
            .into()
    }

    fn name(&self) -> NodeName {
        "GroupedAggregate".into()
    }

    fn op_type(&self) -> NodeType {
        NodeType::GroupByAgg
    }

    fn multiline_display(&self) -> Vec<String> {
        let mut display = vec![];
        display.push(format!(
            "GroupedAggregate: {}",
            self.grouped_aggregate_params
                .original_aggregations
                .iter()
                .map(|e| e.to_string())
                .join(", ")
        ));
        display.push(format!(
            "Group by: {}",
            self.grouped_aggregate_params
                .group_by
                .iter()
                .map(|e| e.to_string())
                .join(", ")
        ));
        display
    }

    fn make_state(&self, _input_id: InputId) -> DaftResult<Self::State> {
        Ok(GroupedAggregateState::new(
            self.num_partitions(),
            self.partial_agg_threshold,
            self.high_cardinality_threshold_ratio,
        ))
    }
}
