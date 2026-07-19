use std::{
    collections::{HashMap, hash_map::Entry},
    sync::Arc,
    time::{Duration, Instant},
};

use common_error::DaftResult;
use common_metrics::{Meter, ops::NodeInfo};
use common_runtime::OrderingAwareJoinSet;
use daft_micropartition::MicroPartition;

use crate::{
    ExecutionTaskSpawner,
    buffer::RowBasedBuffer,
    channel::{Receiver, Sender},
    join::{
        build::{
            BuildStateBridge, FinalizedBuild, FinalizedBuildStateReceiver, partition_into_writers,
        },
        join_operator::{JoinOperator, ProbeOutput},
        stats::JoinStats,
    },
    pipeline::{InputId, PipelineEvent, PipelineMessage, next_event},
    runtime_stats::{RuntimeStats, RuntimeStatsManagerHandle},
    spill::{RunWriter, SpillContext, SpilledRun},
};

enum TaskOutput<Op: JoinOperator> {
    BuildStateReady {
        input_id: InputId,
        finalized: crate::join::build::AccountedBuildState<Op>,
    },
    ProbeComplete {
        input_id: InputId,
        state: Op::ProbeState,
        output: ProbeOutput,
        elapsed: Duration,
    },
    /// A partition-append task returned the run writers for reuse.
    GraceAppend {
        input_id: InputId,
        writers: Vec<RunWriter>,
    },
    Completed,
}

type TaskResult<Op> = DaftResult<TaskOutput<Op>>;

/// Probe-side state of a partitioned join: the build side's spilled
/// partitions, plus the writers re-partitioning the probe input the same
/// way. The scratch configuration is retained so the spilled files
/// outlive the replay.
struct GraceProbe {
    build_partitions: Vec<SpilledRun>,
    /// `None` while an append task owns the writers.
    writers: Option<Vec<RunWriter>>,
    _spill: Arc<SpillContext>,
}

struct PerProbeInput<Op: JoinOperator> {
    states: Vec<Op::ProbeState>,
    buffer: RowBasedBuffer,
    flushed: bool,
    runtime_stats: Arc<JoinStats>,
    max_concurrency: usize,
    /// Keeps the build side's accounted bytes reserved while probing.
    build_budget: Option<crate::resource_manager::SpillBudget>,
    /// Present when the build side switched to partitioned execution.
    grace: Option<GraceProbe>,
}

impl<Op: JoinOperator + 'static> PerProbeInput<Op> {
    fn new(op: &Op, runtime_stats: Arc<JoinStats>, max_concurrency: usize) -> Self {
        let (lower, upper) = op.morsel_size_requirement().unwrap_or_default().values();
        Self {
            states: Vec::new(),
            buffer: RowBasedBuffer::new(lower, upper),
            flushed: false,
            runtime_stats,
            max_concurrency,
            build_budget: None,
            grace: None,
        }
    }

    fn spawn_ready_batches(
        &mut self,
        tasks: &mut OrderingAwareJoinSet<TaskResult<Op>>,
        op: &Arc<Op>,
        spawner: &ExecutionTaskSpawner,
        input_id: InputId,
    ) -> DaftResult<()> {
        if let Some(grace) = &mut self.grace {
            // Partitioned mode: instead of probing, each ready batch is
            // split by join-key hash and appended to the partition runs.
            // One append task at a time keeps writer ownership simple.
            if let Some(writers) = grace.writers.take() {
                if let Some(batch) = self.buffer.next_batch_if_ready()? {
                    let Some((_, probe_exprs)) = op.grace_partition_exprs() else {
                        return Err(common_error::DaftError::InternalError(
                            "partitioned join active without partitioning support".to_string(),
                        ));
                    };
                    let exprs = probe_exprs.to_vec();
                    tasks.spawn(async move {
                        partition_into_writers(&batch, &exprs, &writers).await?;
                        Ok(TaskOutput::GraceAppend { input_id, writers })
                    });
                } else {
                    grace.writers = Some(writers);
                }
            }
            return Ok(());
        }
        while let Some(state) = self.states.pop() {
            if let Some(batch) = self.buffer.next_batch_if_ready()? {
                let op = op.clone();
                let spawner = spawner.clone();
                tasks.spawn(async move {
                    let now = Instant::now();
                    let (state, output) = op.probe(batch, state, &spawner).await??;
                    Ok(TaskOutput::ProbeComplete {
                        input_id,
                        state,
                        output,
                        elapsed: now.elapsed(),
                    })
                });
            } else {
                self.states.push(state);
                break;
            }
        }
        Ok(())
    }

    fn all_states_idle(&self) -> bool {
        match &self.grace {
            Some(grace) => grace.writers.is_some(),
            None => self.states.len() == self.max_concurrency,
        }
    }

    fn ready_to_complete(&self) -> bool {
        self.flushed && self.all_states_idle()
    }
}

pub(crate) struct ProbeExecutionContext<Op: JoinOperator> {
    op: Arc<Op>,
    task_spawner: ExecutionTaskSpawner,
    finalize_spawner: ExecutionTaskSpawner,
    output_sender: Sender<PipelineMessage>,
    build_state_bridge: Arc<BuildStateBridge<Op>>,
    maintain_order: bool,
    stats_manager: RuntimeStatsManagerHandle,
    node_id: usize,
    meter: Meter,
    node_info: Arc<NodeInfo>,
}

impl<Op: JoinOperator + 'static> ProbeExecutionContext<Op> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        op: Arc<Op>,
        task_spawner: ExecutionTaskSpawner,
        finalize_spawner: ExecutionTaskSpawner,
        output_sender: Sender<PipelineMessage>,
        build_state_bridge: Arc<BuildStateBridge<Op>>,
        maintain_order: bool,
        stats_manager: RuntimeStatsManagerHandle,
        node_id: usize,
        meter: Meter,
        node_info: Arc<NodeInfo>,
    ) -> Self {
        Self {
            op,
            task_spawner,
            finalize_spawner,
            output_sender,
            build_state_bridge,
            maintain_order,
            stats_manager,
            node_id,
            meter,
            node_info,
        }
    }

    /// Probes `partition` (and any follow-up output rounds) against
    /// `state`, sending every produced chunk downstream. Returns the state
    /// once the operator asks for more input.
    async fn probe_to_completion(
        op: &Op,
        mut partition: MicroPartition,
        mut state: Op::ProbeState,
        input_id: InputId,
        task_spawner: &ExecutionTaskSpawner,
        output_tx: &Sender<PipelineMessage>,
        runtime_stats: &JoinStats,
    ) -> DaftResult<Op::ProbeState> {
        loop {
            let now = Instant::now();
            let (new_state, result) = op.probe(partition, state, task_spawner).await??;
            runtime_stats.add_duration_us(now.elapsed().as_micros() as u64);

            let output_mp = match &result {
                ProbeOutput::NeedMoreInput(mp) => mp.as_ref(),
                ProbeOutput::HasMoreOutput { output, .. } => Some(output),
            };
            if let Some(mp) = output_mp {
                runtime_stats.add_probe_rows_out(mp.len() as u64);
                runtime_stats.add_probe_bytes_out(mp.size_bytes() as u64);
                let _ = output_tx
                    .send(PipelineMessage::Morsel {
                        input_id,
                        partition: mp.clone(),
                    })
                    .await;
            }

            match result {
                ProbeOutput::NeedMoreInput(_) => return Ok(new_state),
                ProbeOutput::HasMoreOutput { input, .. } => {
                    partition = input;
                    state = new_state;
                }
            }
        }
    }

    /// Replays a partitioned join: the remaining probe input joins the
    /// partition runs, then each partition is rebuilt from its spilled
    /// build input and probed with its spilled probe input, one partition
    /// at a time, so peak memory is one partition's build side.
    #[allow(clippy::too_many_arguments)]
    async fn replay_grace_partitions(
        op: &Op,
        grace: GraceProbe,
        mut buffer: RowBasedBuffer,
        mut build_budget: Option<crate::resource_manager::SpillBudget>,
        input_id: InputId,
        task_spawner: &ExecutionTaskSpawner,
        finalize_spawner: &ExecutionTaskSpawner,
        output_tx: &Sender<PipelineMessage>,
        runtime_stats: &JoinStats,
    ) -> DaftResult<()> {
        let Some((_, probe_exprs)) = op.grace_partition_exprs() else {
            return Err(common_error::DaftError::InternalError(
                "partitioned join active without partitioning support".to_string(),
            ));
        };
        let Some(writers) = grace.writers else {
            return Err(common_error::DaftError::InternalError(
                "partitioned join completing while an append task is still running".to_string(),
            ));
        };
        if let Some(partition) = buffer.pop_all()? {
            partition_into_writers(&partition, probe_exprs, &writers).await?;
        }
        let mut probe_runs = Vec::with_capacity(writers.len());
        for writer in writers {
            probe_runs.push(writer.finish().await?);
        }

        let mut over_budget_logged = false;
        for (build_run, probe_run) in grace.build_partitions.into_iter().zip(probe_runs) {
            if build_run.num_rows() == 0 && probe_run.num_rows() == 0 {
                continue;
            }
            // Account the resident partition for the duration of its
            // replay. The on-disk size understates the decoded size, but
            // keeps concurrent holders shedding while partitions load.
            let partition_bytes = build_run.size_bytes() as u64;
            if let Some(budget) = &mut build_budget
                && !budget.try_grow(partition_bytes)
            {
                budget.grow_unchecked(partition_bytes);
                if !over_budget_logged {
                    over_budget_logged = true;
                    log::warn!(
                        "a partitioned join partition exceeds the available memory budget; \
                         the process may grow beyond the configured limit during its replay"
                    );
                }
            }

            let mut build_state = op.make_build_state()?;
            for part in build_run.read_back().await? {
                build_state = op.build(part, build_state, task_spawner).await??;
            }
            let finalized = op.finalize_build(build_state)?;
            let mut state = op.make_probe_state(finalized);

            let mut cursor = probe_run.cursor();
            while let Some(batch) = cursor.next_batch().await? {
                let partition =
                    MicroPartition::new_loaded(batch.schema.clone(), Arc::new(vec![batch]), None);
                state = Self::probe_to_completion(
                    op,
                    partition,
                    state,
                    input_id,
                    task_spawner,
                    output_tx,
                    runtime_stats,
                )
                .await?;
            }

            if op.needs_probe_finalization()
                && let Some(mp) = op.finalize_probe(vec![state], finalize_spawner).await??
            {
                runtime_stats.add_probe_rows_out(mp.len() as u64);
                runtime_stats.add_probe_bytes_out(mp.size_bytes() as u64);
                let _ = output_tx
                    .send(PipelineMessage::Morsel {
                        input_id,
                        partition: mp,
                    })
                    .await;
            }

            if let Some(budget) = &mut build_budget {
                budget.shrink(partition_bytes);
            }
        }
        Ok(())
    }

    /// Drain remaining buffer, finalize, and send output downstream.
    fn spawn_complete_input(
        op: Arc<Op>,
        per_input: PerProbeInput<Op>,
        input_id: InputId,
        task_spawner: ExecutionTaskSpawner,
        finalize_spawner: ExecutionTaskSpawner,
        output_tx: Sender<PipelineMessage>,
        tasks: &mut OrderingAwareJoinSet<TaskResult<Op>>,
    ) {
        tasks.spawn(async move {
            let mut states = per_input.states;
            let mut buffer = per_input.buffer;
            let runtime_stats = per_input.runtime_stats;

            if let Some(grace) = per_input.grace {
                Self::replay_grace_partitions(
                    &op,
                    grace,
                    buffer,
                    per_input.build_budget,
                    input_id,
                    &task_spawner,
                    &finalize_spawner,
                    &output_tx,
                    &runtime_stats,
                )
                .await?;
                let _ = output_tx.send(PipelineMessage::Flush(input_id)).await;
                return Ok(TaskOutput::Completed);
            }
            // Holds the build side's accounting until probing finishes.
            let _build_budget = per_input.build_budget;

            if let Some(partition) = buffer.pop_all()? {
                let Some(state) = states.pop() else {
                    return Err(common_error::DaftError::InternalError(
                        "probe completion requires an idle probe state".to_string(),
                    ));
                };
                let state = Self::probe_to_completion(
                    &op,
                    partition,
                    state,
                    input_id,
                    &task_spawner,
                    &output_tx,
                    &runtime_stats,
                )
                .await?;
                states.push(state);
            }

            if op.needs_probe_finalization()
                && let Some(mp) = op.finalize_probe(states, &finalize_spawner).await??
            {
                runtime_stats.add_probe_rows_out(mp.len() as u64);
                runtime_stats.add_probe_bytes_out(mp.size_bytes() as u64);
                let _ = output_tx
                    .send(PipelineMessage::Morsel {
                        input_id,
                        partition: mp,
                    })
                    .await;
            }

            let _ = output_tx.send(PipelineMessage::Flush(input_id)).await;
            Ok(TaskOutput::Completed)
        });
    }

    pub(crate) async fn process_probe_input(
        &self,
        receiver: Receiver<PipelineMessage>,
    ) -> DaftResult<()> {
        let mut receiver = receiver;
        let max_concurrency = self.op.max_probe_concurrency();
        let mut inputs: HashMap<InputId, PerProbeInput<Op>> = HashMap::new();
        let mut tasks: OrderingAwareJoinSet<TaskResult<Op>> =
            OrderingAwareJoinSet::new(self.maintain_order);
        let mut child_closed = false;

        while let Some(event) = next_event(
            &mut tasks,
            max_concurrency,
            &mut receiver,
            &mut child_closed,
        )
        .await?
        {
            match event {
                PipelineEvent::TaskCompleted(TaskOutput::Completed) => {}
                PipelineEvent::TaskCompleted(TaskOutput::BuildStateReady {
                    input_id,
                    finalized,
                }) => {
                    let per_input = inputs.get_mut(&input_id).unwrap();
                    match finalized.build {
                        FinalizedBuild::Resident(state) => {
                            per_input.states = (0..max_concurrency)
                                .map(|_| self.op.make_probe_state(state.clone()))
                                .collect();
                        }
                        FinalizedBuild::Grace(grace_build) => {
                            // Partition the probe input the same way the
                            // build side was partitioned; probing happens
                            // per partition once this input's stream ends.
                            let scratch = grace_build.spill.scratch()?;
                            let writers = (0..grace_build.partitions.len())
                                .map(|_| scratch.start_run(grace_build.spill.compression()))
                                .collect::<DaftResult<Vec<_>>>()?;
                            log::info!(
                                "probe side of a partitioned join is spilling its input across \
                                 {} partitions",
                                grace_build.partitions.len()
                            );
                            per_input.grace = Some(GraceProbe {
                                build_partitions: grace_build.partitions,
                                writers: Some(writers),
                                _spill: grace_build.spill,
                            });
                        }
                    }
                    // Hold the build side's memory accounting for as long as
                    // this input probes against it.
                    per_input.build_budget = finalized.budget;
                    per_input.spawn_ready_batches(
                        &mut tasks,
                        &self.op,
                        &self.task_spawner,
                        input_id,
                    )?;

                    if inputs.get(&input_id).is_some_and(|p| p.ready_to_complete()) {
                        Self::spawn_complete_input(
                            self.op.clone(),
                            inputs.remove(&input_id).unwrap(),
                            input_id,
                            self.task_spawner.clone(),
                            self.finalize_spawner.clone(),
                            self.output_sender.clone(),
                            &mut tasks,
                        );
                    }
                }
                PipelineEvent::TaskCompleted(TaskOutput::GraceAppend { input_id, writers }) => {
                    let per_input = inputs.get_mut(&input_id).unwrap();
                    let Some(grace) = &mut per_input.grace else {
                        return Err(common_error::DaftError::InternalError(
                            "partition-append task completed without partitioned state".to_string(),
                        )
                        .into());
                    };
                    grace.writers = Some(writers);
                    per_input.spawn_ready_batches(
                        &mut tasks,
                        &self.op,
                        &self.task_spawner,
                        input_id,
                    )?;

                    if inputs.get(&input_id).is_some_and(|p| p.ready_to_complete()) {
                        Self::spawn_complete_input(
                            self.op.clone(),
                            inputs.remove(&input_id).unwrap(),
                            input_id,
                            self.task_spawner.clone(),
                            self.finalize_spawner.clone(),
                            self.output_sender.clone(),
                            &mut tasks,
                        );
                    }
                }
                PipelineEvent::TaskCompleted(TaskOutput::ProbeComplete {
                    input_id,
                    state,
                    output,
                    elapsed,
                }) => {
                    let per_input = inputs.get_mut(&input_id).unwrap();
                    per_input
                        .runtime_stats
                        .add_duration_us(elapsed.as_micros() as u64);

                    let output_mp = match &output {
                        ProbeOutput::NeedMoreInput(mp) => mp.as_ref(),
                        ProbeOutput::HasMoreOutput { output, .. } => Some(output),
                    };
                    if let Some(mp) = output_mp {
                        per_input.runtime_stats.add_probe_rows_out(mp.len() as u64);
                        per_input
                            .runtime_stats
                            .add_probe_bytes_out(mp.size_bytes() as u64);
                        let _ = self
                            .output_sender
                            .send(PipelineMessage::Morsel {
                                input_id,
                                partition: mp.clone(),
                            })
                            .await;
                    }

                    match output {
                        ProbeOutput::NeedMoreInput(_) => {
                            per_input.states.push(state);
                            per_input.spawn_ready_batches(
                                &mut tasks,
                                &self.op,
                                &self.task_spawner,
                                input_id,
                            )?;

                            if inputs.get(&input_id).is_some_and(|p| p.ready_to_complete()) {
                                Self::spawn_complete_input(
                                    self.op.clone(),
                                    inputs.remove(&input_id).unwrap(),
                                    input_id,
                                    self.task_spawner.clone(),
                                    self.finalize_spawner.clone(),
                                    self.output_sender.clone(),
                                    &mut tasks,
                                );
                            }
                        }
                        ProbeOutput::HasMoreOutput { input, .. } => {
                            let op = self.op.clone();
                            let spawner = self.task_spawner.clone();
                            tasks.spawn(async move {
                                let now = Instant::now();
                                let (state, output) = op.probe(input, state, &spawner).await??;
                                Ok(TaskOutput::ProbeComplete {
                                    input_id,
                                    state,
                                    output,
                                    elapsed: now.elapsed(),
                                })
                            });
                        }
                    }
                }
                PipelineEvent::Morsel {
                    input_id,
                    partition,
                } => {
                    let per_input = match inputs.entry(input_id) {
                        Entry::Occupied(e) => e.into_mut(),
                        Entry::Vacant(e) => {
                            let runtime_stats =
                                Arc::new(JoinStats::new(&self.meter, &self.node_info));
                            self.stats_manager.register_runtime_stats(
                                self.node_id,
                                input_id,
                                runtime_stats.clone(),
                            );

                            let bridge = self.build_state_bridge.clone();
                            tasks.spawn(async move {
                                let finalized = match bridge.subscribe(input_id) {
                                    FinalizedBuildStateReceiver::Receiver(rx) => {
                                        rx.await.map_err(|e| {
                                            common_error::DaftError::ValueError(format!(
                                                "Failed to receive finalized build state: {e}"
                                            ))
                                        })?
                                    }
                                    FinalizedBuildStateReceiver::Ready(v) => v,
                                };
                                Ok(TaskOutput::BuildStateReady {
                                    input_id,
                                    finalized,
                                })
                            });

                            e.insert(PerProbeInput::new(&self.op, runtime_stats, max_concurrency))
                        }
                    };
                    per_input
                        .runtime_stats
                        .add_probe_rows_in(partition.len() as u64);
                    per_input
                        .runtime_stats
                        .add_probe_bytes_in(partition.size_bytes() as u64);
                    per_input.buffer.push(partition);
                    per_input.spawn_ready_batches(
                        &mut tasks,
                        &self.op,
                        &self.task_spawner,
                        input_id,
                    )?;
                }
                PipelineEvent::Flush(input_id) => {
                    if let Some(p) = inputs.get_mut(&input_id) {
                        p.flushed = true;
                    }
                    if inputs.get(&input_id).is_some_and(|p| p.ready_to_complete()) {
                        Self::spawn_complete_input(
                            self.op.clone(),
                            inputs.remove(&input_id).unwrap(),
                            input_id,
                            self.task_spawner.clone(),
                            self.finalize_spawner.clone(),
                            self.output_sender.clone(),
                            &mut tasks,
                        );
                    }
                }
                PipelineEvent::FlightPartitionRef => {
                    unreachable!("Probe join should not receive flight partition refs from child")
                }
                PipelineEvent::InputClosed => {
                    for p in inputs.values_mut() {
                        p.flushed = true;
                    }
                    let ready_ids: Vec<_> = inputs
                        .iter()
                        .filter(|(_, p)| p.ready_to_complete())
                        .map(|(id, _)| *id)
                        .collect();
                    for input_id in ready_ids {
                        Self::spawn_complete_input(
                            self.op.clone(),
                            inputs.remove(&input_id).unwrap(),
                            input_id,
                            self.task_spawner.clone(),
                            self.finalize_spawner.clone(),
                            self.output_sender.clone(),
                            &mut tasks,
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
