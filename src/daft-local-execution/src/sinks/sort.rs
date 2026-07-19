use std::{collections::VecDeque, sync::Arc};

use common_daft_config::DaftExecutionConfig;
use common_error::DaftResult;
use common_metrics::ops::NodeType;
use daft_dsl::expr::bound_expr::BoundExpr;
use daft_micropartition::MicroPartition;
use itertools::Itertools;
use tracing::{Span, instrument};

use super::blocking_sink::{
    BlockingSink, BlockingSinkFinalizeResult, BlockingSinkOutput, BlockingSinkSinkResult,
};
use crate::{
    ExecutionTaskSpawner,
    channel::create_channel,
    pipeline::{InputId, NodeName},
    resource_manager::{QueryMemoryScope, SpillBudget},
    sorted_merge::{MergeOrdering, MergeSource, merge_sorted_runs_streaming},
    spill::{SpillContext, SpilledRun},
};

pub(crate) enum SortState {
    Building {
        parts: Vec<MicroPartition>,
        buffered_bytes: u64,
        budget: Option<SpillBudget>,
        spilled_runs: Vec<SpilledRun>,
    },
    Done,
}

impl SortState {
    /// Buffers one morsel; under memory pressure the buffered chunk is
    /// sorted and shed to disk as one sorted run.
    async fn push(
        &mut self,
        part: MicroPartition,
        params: &SortParams,
        memory_scope: &QueryMemoryScope,
    ) -> DaftResult<()> {
        let Self::Building {
            parts,
            buffered_bytes,
            budget,
            spilled_runs,
        } = self
        else {
            panic!("SortSink should be in Building state");
        };
        let added = part.size_bytes() as u64;
        parts.push(part);
        *buffered_bytes += added;

        let Some(spill) = &params.spill else {
            return Ok(());
        };
        let budget = budget.get_or_insert_with(|| SpillBudget::new(memory_scope.clone()));
        // Honor any request from concurrent holders to shed toward the
        // fair share, then account this morsel's growth.
        let requested_shed = budget.take_shed_request();
        if requested_shed == 0 && budget.try_grow(added) {
            return Ok(());
        }
        // Pressure: sort everything buffered — including the morsel that
        // triggered the pressure — and shed it as one sorted run. Sorting
        // the bounded chunk here keeps the final merge a pure streaming
        // pass.
        let chunk = std::mem::take(parts);
        let sorted = MicroPartition::concat(chunk)?.sort(
            &params.sort_by,
            &params.descending,
            &params.nulls_first,
        )?;
        let run = spill
            .scratch()?
            .spill(vec![sorted], spill.compression())
            .await?;
        spilled_runs.push(run);
        // Only previously accounted bytes return to the budget; the new
        // morsel's bytes were never granted.
        let accounted = *buffered_bytes - added;
        budget.shrink(accounted);
        *buffered_bytes = 0;
        Ok(())
    }

    fn finalize(&mut self) -> (Vec<MicroPartition>, Vec<SpilledRun>) {
        let res = if let Self::Building {
            parts,
            spilled_runs,
            ..
        } = self
        {
            (std::mem::take(parts), std::mem::take(spilled_runs))
        } else {
            panic!("SortSink should be in Building state");
        };
        *self = Self::Done;
        res
    }
}

struct SortParams {
    sort_by: Vec<BoundExpr>,
    descending: Vec<bool>,
    nulls_first: Vec<bool>,
    // Spill configuration; absent when spilling is disabled.
    spill: Option<SpillContext>,
}
pub struct SortSink {
    params: Arc<SortParams>,
}

impl SortSink {
    pub fn new(
        sort_by: Vec<BoundExpr>,
        descending: Vec<bool>,
        nulls_first: Vec<bool>,
        cfg: &DaftExecutionConfig,
    ) -> Self {
        Self {
            params: Arc::new(SortParams {
                sort_by,
                descending,
                nulls_first,
                spill: SpillContext::from_config(cfg),
            }),
        }
    }
}

impl BlockingSink for SortSink {
    type State = SortState;

    #[instrument(skip_all, name = "SortSink::sink")]
    fn sink(
        &self,
        input: MicroPartition,
        mut state: Self::State,
        _runtime_stats: Arc<Self::Stats>,
        spawner: &ExecutionTaskSpawner,
    ) -> BlockingSinkSinkResult<Self> {
        let params = self.params.clone();
        let memory_scope = spawner.memory_scope().clone();
        spawner
            .spawn(
                async move {
                    state.push(input, &params, &memory_scope).await?;
                    Ok(state)
                },
                Span::current(),
            )
            .into()
    }

    #[instrument(skip_all, name = "SortSink::finalize")]
    fn finalize(
        &self,
        states: Vec<Self::State>,
        spawner: &ExecutionTaskSpawner,
    ) -> BlockingSinkFinalizeResult {
        let params = self.params.clone();
        let merge_spawner = spawner.clone();
        spawner
            .spawn(
                async move {
                    let mut parts: Vec<MicroPartition> = Vec::new();
                    let mut spilled_runs: Vec<SpilledRun> = Vec::new();
                    for mut state in states {
                        let (state_parts, state_runs) = state.finalize();
                        parts.extend(state_parts);
                        spilled_runs.extend(state_runs);
                    }

                    if spilled_runs.is_empty() {
                        let concated = MicroPartition::concat(parts)?;
                        let sorted = concated.sort(
                            &params.sort_by,
                            &params.descending,
                            &params.nulls_first,
                        )?;
                        return Ok(BlockingSinkOutput::Partitions(vec![sorted]));
                    }

                    // External path: the still-buffered remainder becomes one
                    // final in-memory sorted run, then every run merges in a
                    // dedicated task that streams sorted chunks downstream as
                    // they are produced — the full result never materializes.
                    let (tx, rx) = create_channel::<MicroPartition>(2);
                    let producer = merge_spawner.spawn(
                        async move {
                            let Some(spill) = params.spill.as_ref() else {
                                return Err(common_error::DaftError::InternalError(
                                    "sorted spill runs exist but spilling is not configured"
                                        .to_string(),
                                ));
                            };
                            let mut sources: Vec<MergeSource> = spilled_runs
                                .into_iter()
                                .map(|run| MergeSource::Spilled(run.cursor()))
                                .collect();
                            let total_buffered: usize = parts.iter().map(MicroPartition::len).sum();
                            if total_buffered > 0 {
                                let sorted = MicroPartition::concat(parts)?.sort(
                                    &params.sort_by,
                                    &params.descending,
                                    &params.nulls_first,
                                )?;
                                sources.push(MergeSource::Memory(VecDeque::from(
                                    sorted.record_batches().to_vec(),
                                )));
                            }
                            let ordering = MergeOrdering {
                                sort_by: params.sort_by.clone(),
                                descending: params.descending.clone(),
                                nulls_first: params.nulls_first.clone(),
                            };
                            merge_sorted_runs_streaming(sources, &ordering, spill, &tx).await
                        },
                        Span::current(),
                    );
                    Ok(BlockingSinkOutput::PartitionStream {
                        partitions: rx,
                        producer,
                    })
                },
                Span::current(),
            )
            .into()
    }

    fn name(&self) -> NodeName {
        "Sort".into()
    }

    fn op_type(&self) -> NodeType {
        NodeType::Sort
    }

    fn multiline_display(&self) -> Vec<String> {
        let mut lines = vec![];
        assert!(!self.params.sort_by.is_empty());
        let pairs = self
            .params
            .sort_by
            .iter()
            .zip(self.params.descending.iter())
            .zip(self.params.nulls_first.iter())
            .map(|((sb, d), nf)| {
                format!(
                    "({}, {}, {})",
                    sb,
                    if *d { "descending" } else { "ascending" },
                    if *nf { "nulls first" } else { "nulls last" }
                )
            })
            .join(", ");
        lines.push(format!("Sort: Sort by = {}", pairs));
        lines
    }

    fn make_state(&self, _input_id: InputId) -> DaftResult<Self::State> {
        Ok(SortState::Building {
            parts: Vec::new(),
            buffered_bytes: 0,
            budget: None,
            spilled_runs: Vec::new(),
        })
    }

    fn max_concurrency(&self) -> usize {
        1
    }
}
