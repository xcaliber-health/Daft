use std::{
    collections::{HashMap, VecDeque, hash_map::Entry},
    sync::{Arc, Mutex},
};

use common_error::DaftResult;
use common_metrics::{Meter, ops::NodeInfo};
use common_runtime::{OrderingAwareJoinSet, get_compute_pool_num_threads};
use daft_dsl::expr::bound_expr::BoundExpr;
use daft_micropartition::MicroPartition;
use tokio::sync::oneshot;

use crate::{
    ExecutionTaskSpawner,
    channel::Receiver,
    join::{join_operator::JoinOperator, key_filter::KeyFilterPublisher, stats::JoinStats},
    pipeline::{InputId, PipelineEvent, PipelineMessage, next_event},
    resource_manager::SpillBudget,
    runtime_stats::{RuntimeStats, RuntimeStatsManagerHandle},
    spill::{RunWriter, SpillContext, SpilledRun},
};

/// Number of hash partitions used when a build side exceeds its memory
/// budget and switches to partitioned execution. Each partition is
/// replayed independently, so peak memory is roughly the largest
/// partition's build input rather than the whole build side.
pub(crate) const GRACE_PARTITION_COUNT: usize = 16;

enum BuildStateSlot<T> {
    Sender(oneshot::Sender<T>),
    Ready(T),
}

/// The build side of a partitioned join: input re-partitioned by join-key
/// hash and spilled, one run per partition, plus the shared scratch the
/// probe side must use to partition its own input symmetrically.
pub(crate) struct GraceBuild {
    pub(crate) partitions: Vec<SpilledRun>,
    pub(crate) spill: Arc<SpillContext>,
}

/// How the finalized build side is delivered to the probe phase.
pub(crate) enum FinalizedBuild<Op: JoinOperator> {
    /// The build result is resident in memory; probing streams against it.
    Resident(Op::FinalizedBuildState),
    /// The build input was re-partitioned to disk under memory pressure;
    /// the probe side partitions symmetrically and replays per partition.
    Grace(GraceBuild),
}

/// A finalized build result together with the memory accounting held on its
/// behalf. The accounted bytes return to the shared budget when this is
/// dropped, i.e. when the probe phase that owns the build result finishes.
pub(crate) struct AccountedBuildState<Op: JoinOperator> {
    pub(crate) build: FinalizedBuild<Op>,
    pub(crate) budget: Option<SpillBudget>,
}

/// What a probe input gets when it subscribes to its build state: the state
/// itself if it was already sent, or a receiver that yields it, or ends
/// without it if the build side stops first.
pub(crate) enum Subscription<T> {
    Receiver(oneshot::Receiver<T>),
    Ready(T),
}

pub(crate) type FinalizedBuildStateReceiver<Op> = Subscription<AccountedBuildState<Op>>;

/// Hands each input's finalized build side to the probe side waiting for it.
///
/// Once the build side stops, finished or failed, the bridge is closed: a
/// probe input still waiting, or subscribing later, for a build state that was
/// never sent sees its receiver end at once instead of waiting for a build
/// that will not come.
pub(crate) struct BuildStateBridge<Op: JoinOperator> {
    slots: Mutex<BridgeSlots<AccountedBuildState<Op>>>,
}

impl<Op: JoinOperator> BuildStateBridge<Op> {
    pub(crate) fn new() -> Self {
        Self {
            slots: Mutex::new(BridgeSlots::new()),
        }
    }

    fn slots(&self) -> std::sync::MutexGuard<'_, BridgeSlots<AccountedBuildState<Op>>> {
        self.slots
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Hands `input_id`'s finalized build state to its probe input, or
    /// returns the error finalizing it failed with, so the query reports
    /// that error instead of the probe input waiting for a state.
    pub(crate) fn send_finalized_build_state(
        &self,
        input_id: InputId,
        finalized: DaftResult<AccountedBuildState<Op>>,
    ) -> DaftResult<()> {
        self.slots().send(input_id, finalized)
    }

    pub(crate) fn subscribe(&self, input_id: InputId) -> FinalizedBuildStateReceiver<Op> {
        self.slots().subscribe(input_id)
    }

    /// Marks the build side stopped; see [`BridgeSlots::close`].
    pub(crate) fn close(&self) {
        self.slots().close();
    }
}

/// The build states of a join's inputs, each sent once by the build side and
/// taken once by the probe side, in either order.
struct BridgeSlots<T> {
    by_input: HashMap<InputId, BuildStateSlot<T>>,
    closed: bool,
}

impl<T> BridgeSlots<T> {
    fn new() -> Self {
        Self {
            by_input: HashMap::new(),
            closed: false,
        }
    }

    /// Hands `input_id`'s state to its probe input, waiting or not yet
    /// subscribed. A failed state is not handed over: its error is returned.
    fn send(&mut self, input_id: InputId, state: DaftResult<T>) -> DaftResult<()> {
        let state = state?;
        match self.by_input.remove(&input_id) {
            Some(BuildStateSlot::Sender(tx)) => {
                // A probe input that stopped waiting has no use for it.
                let _ = tx.send(state);
            }
            Some(BuildStateSlot::Ready(_)) | None => {
                self.by_input.insert(input_id, BuildStateSlot::Ready(state));
            }
        }
        Ok(())
    }

    fn subscribe(&mut self, input_id: InputId) -> Subscription<T> {
        let (tx, rx) = oneshot::channel();
        match self.by_input.remove(&input_id) {
            Some(BuildStateSlot::Ready(state)) => Subscription::Ready(state),
            // A stopped build side sends nothing more: `tx` drops here, so
            // `rx` ends at once.
            None if self.closed => Subscription::Receiver(rx),
            Some(BuildStateSlot::Sender(_)) | None => {
                self.by_input.insert(input_id, BuildStateSlot::Sender(tx));
                Subscription::Receiver(rx)
            }
        }
    }

    /// Marks the build side stopped. Every probe input still waiting for a
    /// state is released, its receiver ending without one; states already
    /// sent stay available to their probe inputs.
    fn close(&mut self) {
        self.closed = true;
        self.by_input
            .retain(|_, slot| matches!(slot, BuildStateSlot::Ready(_)));
    }
}

/// Closes a build state bridge when dropped, however the build side stops.
struct CloseOnDrop<'a, Op: JoinOperator>(&'a BuildStateBridge<Op>);

impl<Op: JoinOperator> Drop for CloseOnDrop<'_, Op> {
    fn drop(&mut self) {
        self.0.close();
    }
}

/// Splits `part` by the hash of `exprs` and appends each non-empty piece
/// to the writer of the matching partition.
pub(crate) async fn partition_into_writers(
    part: &MicroPartition,
    exprs: &[BoundExpr],
    writers: &[RunWriter],
) -> DaftResult<()> {
    let pieces = part.partition_by_hash(exprs, writers.len())?;
    for (piece, writer) in pieces.into_iter().zip(writers) {
        if !piece.is_empty() {
            writer.push(piece).await?;
        }
    }
    Ok(())
}

/// Build-side accumulation: a resident in-memory build state, or the
/// per-partition run writers of a build side that switched to disk.
enum BuildAccum<Op: JoinOperator> {
    Resident(Op::BuildState),
    Grace(Vec<RunWriter>),
}

enum BuildTaskOutput<Op: JoinOperator> {
    /// A build or partitioning task returned the accumulation for reuse.
    Accum(InputId, BuildAccum<Op>),
    /// A finalize task delivered the build result to the probe side.
    Finalized,
}

type BuildTaskResult<Op> = DaftResult<BuildTaskOutput<Op>>;

#[derive(PartialEq, Eq, Clone, Copy)]
enum GraceStatus {
    /// Building resident; accounting normally.
    Off,
    /// Budget denied growth; switch to partitioned execution at the next
    /// flush.
    Requested,
    /// Partitioned execution engaged; incoming morsels stream to disk.
    Active,
}

struct PerBuildInput<Op: JoinOperator> {
    accum: Option<BuildAccum<Op>>,
    pending: VecDeque<MicroPartition>,
    flushed: bool,
    runtime_stats: Arc<JoinStats>,
    /// Accounted share of the memory budget for this input's build state.
    /// A partition-capable build sheds to disk when growth is denied;
    /// otherwise the result must stay resident through the probe phase and
    /// overflow is recorded as unfunded growth — keeping system-wide
    /// pressure honest so shed-able operators spill sooner.
    budget: Option<SpillBudget>,
    /// Bytes currently charged to the budget for resident build input.
    accounted_bytes: u64,
    grace: GraceStatus,
    over_budget_logged: bool,
}

impl<Op: JoinOperator + 'static> PerBuildInput<Op> {
    fn new(state: Op::BuildState, runtime_stats: Arc<JoinStats>) -> Self {
        Self {
            accum: Some(BuildAccum::Resident(state)),
            pending: VecDeque::new(),
            flushed: false,
            runtime_stats,
            budget: None,
            accounted_bytes: 0,
            grace: GraceStatus::Off,
            over_budget_logged: false,
        }
    }

    /// Charges `bytes` of incoming build data to the shared budget. Once
    /// partitioned execution is requested or active, incoming data streams
    /// to disk and is no longer charged.
    fn account(&mut self, bytes: u64, spawner: &ExecutionTaskSpawner, grace_available: bool) {
        if self.grace != GraceStatus::Off {
            return;
        }
        let budget = self.budget.get_or_insert_with(|| {
            SpillBudget::with_shed_capability(spawner.memory_scope().clone(), grace_available)
        });
        let shed_requested = grace_available && budget.take_shed_request() > 0;
        if !shed_requested && budget.try_grow(bytes) {
            self.accounted_bytes += bytes;
            return;
        }
        if grace_available {
            self.grace = GraceStatus::Requested;
            return;
        }
        budget.grow_unchecked(bytes);
        self.accounted_bytes += bytes;
        if !self.over_budget_logged {
            self.over_budget_logged = true;
            log::warn!(
                "join build side exceeds the available memory budget; the process may \
                 grow beyond the configured limit while this join runs"
            );
        }
    }

    /// If the accumulation is idle, spawns the next unit of work: the
    /// switch to partitioned execution when requested, a build task over
    /// the pending input when resident, or a partition-and-spill task when
    /// partitioned.
    fn flush_pending(
        &mut self,
        tasks: &mut OrderingAwareJoinSet<BuildTaskResult<Op>>,
        op: &Arc<Op>,
        spawner: &ExecutionTaskSpawner,
        spill: Option<&Arc<SpillContext>>,
        input_id: InputId,
    ) -> DaftResult<()> {
        let Some(accum) = self.accum.take() else {
            return Ok(());
        };
        match accum {
            BuildAccum::Resident(state) if self.grace == GraceStatus::Requested => {
                match op.take_raw_build_input(state) {
                    Err(state) => {
                        // Raw input is not recoverable; stay resident and
                        // fall back to unfunded growth.
                        self.grace = GraceStatus::Off;
                        self.accum = Some(BuildAccum::Resident(state));
                        if !self.over_budget_logged {
                            self.over_budget_logged = true;
                            log::warn!(
                                "join build side exceeds the available memory budget; the process \
                                 may grow beyond the configured limit while this join runs"
                            );
                        }
                        self.flush_pending(tasks, op, spawner, spill, input_id)
                    }
                    Ok(raw_batches) => {
                        let (Some(spill), Some((build_exprs, _))) =
                            (spill, op.grace_partition_exprs())
                        else {
                            return Err(common_error::DaftError::InternalError(
                                "partitioned join requested without partitioning support"
                                    .to_string(),
                            ));
                        };
                        let spill = spill.clone();
                        let exprs = build_exprs.to_vec();
                        let pending: Vec<MicroPartition> = self.pending.drain(..).collect();
                        tasks.spawn(async move {
                            let scratch = spill.scratch()?;
                            let mut writers = Vec::with_capacity(GRACE_PARTITION_COUNT);
                            for _ in 0..GRACE_PARTITION_COUNT {
                                writers.push(scratch.start_run(spill.compression())?);
                            }
                            for batch in raw_batches {
                                let part = MicroPartition::new_loaded(
                                    batch.schema.clone(),
                                    Arc::new(vec![batch]),
                                    None,
                                );
                                partition_into_writers(&part, &exprs, &writers).await?;
                            }
                            for part in pending {
                                partition_into_writers(&part, &exprs, &writers).await?;
                            }
                            Ok(BuildTaskOutput::Accum(input_id, BuildAccum::Grace(writers)))
                        });
                        self.grace = GraceStatus::Active;
                        // The buffered bytes are on their way to disk;
                        // return them to the budget so other holders can
                        // proceed. The hand-off window is bounded by the
                        // partitioning task's in-flight chunks.
                        if let Some(budget) = &mut self.budget {
                            budget.shrink(self.accounted_bytes);
                        }
                        self.accounted_bytes = 0;
                        log::info!(
                            "join build side exceeded its memory budget; switching to a \
                             partitioned join with {GRACE_PARTITION_COUNT} partitions"
                        );
                        Ok(())
                    }
                }
            }
            BuildAccum::Resident(state) => {
                if self.pending.is_empty() {
                    self.accum = Some(BuildAccum::Resident(state));
                    return Ok(());
                }
                let partition = if self.pending.len() == 1 {
                    self.pending.pop_front().unwrap()
                } else {
                    MicroPartition::concat(self.pending.drain(..).collect::<Vec<_>>())?
                };
                let op = op.clone();
                let spawner = spawner.clone();
                tasks.spawn(async move {
                    let state = op.build(partition, state, &spawner).await??;
                    Ok(BuildTaskOutput::Accum(
                        input_id,
                        BuildAccum::Resident(state),
                    ))
                });
                Ok(())
            }
            BuildAccum::Grace(writers) => {
                if self.pending.is_empty() {
                    self.accum = Some(BuildAccum::Grace(writers));
                    return Ok(());
                }
                let Some((build_exprs, _)) = op.grace_partition_exprs() else {
                    return Err(common_error::DaftError::InternalError(
                        "partitioned join active without partitioning support".to_string(),
                    ));
                };
                let exprs = build_exprs.to_vec();
                let parts: Vec<MicroPartition> = self.pending.drain(..).collect();
                tasks.spawn(async move {
                    for part in parts {
                        partition_into_writers(&part, &exprs, &writers).await?;
                    }
                    Ok(BuildTaskOutput::Accum(input_id, BuildAccum::Grace(writers)))
                });
                Ok(())
            }
        }
    }

    fn is_idle(&self) -> bool {
        self.accum.is_some()
    }

    fn ready_to_finalize(&self) -> bool {
        self.flushed && self.is_idle() && self.grace != GraceStatus::Requested
    }
}

pub(crate) struct BuildExecutionContext<Op: JoinOperator> {
    op: Arc<Op>,
    task_spawner: ExecutionTaskSpawner,
    build_state_bridge: Arc<BuildStateBridge<Op>>,
    stats_manager: RuntimeStatsManagerHandle,
    node_id: usize,
    meter: Meter,
    node_info: Arc<NodeInfo>,
    /// Scratch configuration for partitioned execution; absent when
    /// spilling is disabled.
    spill: Option<Arc<SpillContext>>,
    /// Whether partitioned execution is permitted for this join. Replaying
    /// partitions reorders output, so it is only allowed when downstream
    /// does not require input order.
    allow_grace: bool,
    /// Publishes each input's keys to the probe-side scans they filter, when
    /// any do. Dropped, and so closed, when the build side stops.
    key_filter: Option<KeyFilterPublisher>,
}

impl<Op: JoinOperator + 'static> BuildExecutionContext<Op> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        op: Arc<Op>,
        task_spawner: ExecutionTaskSpawner,
        build_state_bridge: Arc<BuildStateBridge<Op>>,
        stats_manager: RuntimeStatsManagerHandle,
        node_id: usize,
        meter: Meter,
        node_info: Arc<NodeInfo>,
        spill: Option<Arc<SpillContext>>,
        allow_grace: bool,
    ) -> Self {
        Self {
            op,
            task_spawner,
            build_state_bridge,
            stats_manager,
            node_id,
            meter,
            node_info,
            spill,
            allow_grace,
            key_filter: None,
        }
    }

    /// Publishes each finished input's keys through `publisher`.
    pub(crate) fn with_key_filter(mut self, publisher: Option<KeyFilterPublisher>) -> Self {
        self.key_filter = publisher;
        self
    }

    /// Publishes an input's key filter, if this join's keys filter a scan.
    fn publish_key_filter(&self, input_id: InputId, finalized: Option<&Op::FinalizedBuildState>) {
        if let Some(publisher) = &self.key_filter {
            publisher.publish(
                input_id,
                finalized.and_then(|state| self.op.build_key_columns(state)),
            );
        }
    }

    /// Whether this build side can switch to partitioned execution under
    /// memory pressure.
    fn grace_available(&self) -> bool {
        self.allow_grace && self.spill.is_some() && self.op.grace_partition_exprs().is_some()
    }

    fn try_finalize(
        &self,
        per_input: PerBuildInput<Op>,
        input_id: InputId,
        tasks: &mut OrderingAwareJoinSet<BuildTaskResult<Op>>,
    ) -> DaftResult<()> {
        let Some(accum) = per_input.accum else {
            return Err(common_error::DaftError::InternalError(
                "join build side must be idle when finalizing".to_string(),
            ));
        };
        match accum {
            BuildAccum::Resident(state) => {
                let finalized = self.op.finalize_build(state);
                self.publish_key_filter(input_id, finalized.as_ref().ok());
                self.build_state_bridge.send_finalized_build_state(
                    input_id,
                    finalized.map(|finalized| AccountedBuildState {
                        build: FinalizedBuild::Resident(finalized),
                        budget: per_input.budget,
                    }),
                )?;
            }
            BuildAccum::Grace(writers) => {
                // A build side too large for memory is too large to filter by.
                self.publish_key_filter(input_id, None);
                let Some(spill) = self.spill.clone() else {
                    return Err(common_error::DaftError::InternalError(
                        "partitioned join active but spilling is not configured".to_string(),
                    ));
                };
                let bridge = self.build_state_bridge.clone();
                let budget = per_input.budget;
                tasks.spawn(async move {
                    let mut partitions = Vec::with_capacity(writers.len());
                    for writer in writers {
                        partitions.push(writer.finish().await?);
                    }
                    bridge.send_finalized_build_state(
                        input_id,
                        Ok(AccountedBuildState {
                            build: FinalizedBuild::Grace(GraceBuild { partitions, spill }),
                            budget,
                        }),
                    )?;
                    Ok(BuildTaskOutput::Finalized)
                });
            }
        }
        Ok(())
    }

    pub(crate) async fn process_build_input(
        &self,
        receiver: Receiver<PipelineMessage>,
    ) -> DaftResult<()> {
        // Finished, failed or aborted, the build side sends nothing more:
        // release any probe input still waiting for its build state.
        let _closes_bridge = CloseOnDrop(&self.build_state_bridge);
        let result = self.build_inputs(receiver).await;
        // Likewise any scan still waiting for its keys.
        if let Some(publisher) = &self.key_filter {
            publisher.close();
        }
        result
    }

    async fn build_inputs(&self, receiver: Receiver<PipelineMessage>) -> DaftResult<()> {
        let mut receiver = receiver;
        let mut inputs: HashMap<InputId, PerBuildInput<Op>> = HashMap::new();
        let mut tasks: OrderingAwareJoinSet<BuildTaskResult<Op>> = OrderingAwareJoinSet::new(false);
        let mut node_initialized = false;
        let mut child_closed = false;

        while let Some(event) = next_event(
            &mut tasks,
            get_compute_pool_num_threads(),
            &mut receiver,
            &mut child_closed,
        )
        .await?
        {
            match event {
                PipelineEvent::TaskCompleted(BuildTaskOutput::Finalized) => {}
                PipelineEvent::TaskCompleted(BuildTaskOutput::Accum(input_id, accum)) => {
                    let per_input = inputs.get_mut(&input_id).unwrap();
                    per_input.accum = Some(accum);
                    per_input.flush_pending(
                        &mut tasks,
                        &self.op,
                        &self.task_spawner,
                        self.spill.as_ref(),
                        input_id,
                    )?;

                    if inputs.get(&input_id).is_some_and(|p| p.ready_to_finalize()) {
                        self.try_finalize(inputs.remove(&input_id).unwrap(), input_id, &mut tasks)?;
                    }
                }
                PipelineEvent::Morsel {
                    input_id,
                    partition,
                } => {
                    if !node_initialized {
                        self.stats_manager.activate_node(self.node_id);
                        node_initialized = true;
                    }

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
                            let state = self.op.make_build_state()?;
                            e.insert(PerBuildInput::new(state, runtime_stats))
                        }
                    };
                    per_input
                        .runtime_stats
                        .add_build_rows_inserted(partition.len() as u64);
                    per_input
                        .runtime_stats
                        .add_build_bytes_inserted(partition.size_bytes() as u64);
                    per_input.account(
                        partition.size_bytes() as u64,
                        &self.task_spawner,
                        self.grace_available(),
                    );
                    per_input.pending.push_back(partition);
                    per_input.flush_pending(
                        &mut tasks,
                        &self.op,
                        &self.task_spawner,
                        self.spill.as_ref(),
                        input_id,
                    )?;
                }
                PipelineEvent::Flush(input_id) => {
                    if let Some(p) = inputs.get_mut(&input_id) {
                        p.flushed = true;
                        p.flush_pending(
                            &mut tasks,
                            &self.op,
                            &self.task_spawner,
                            self.spill.as_ref(),
                            input_id,
                        )?;
                    }
                    if inputs.get(&input_id).is_some_and(|p| p.ready_to_finalize()) {
                        self.try_finalize(inputs.remove(&input_id).unwrap(), input_id, &mut tasks)?;
                    }
                }
                PipelineEvent::FlightPartitionRef => {
                    unreachable!(
                        "BuildExecutionContext should not receive flight partition refs from child"
                    )
                }
                PipelineEvent::InputClosed => {
                    let ids: Vec<_> = inputs.keys().copied().collect();
                    for input_id in ids {
                        let p = inputs.get_mut(&input_id).unwrap();
                        p.flushed = true;
                        p.flush_pending(
                            &mut tasks,
                            &self.op,
                            &self.task_spawner,
                            self.spill.as_ref(),
                            input_id,
                        )?;
                        if inputs
                            .get(&input_id)
                            .is_some_and(PerBuildInput::ready_to_finalize)
                        {
                            self.try_finalize(
                                inputs.remove(&input_id).unwrap(),
                                input_id,
                                &mut tasks,
                            )?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use common_error::DaftError;

    use super::{BridgeSlots, Subscription};

    fn received(subscription: Subscription<u32>) -> tokio::sync::oneshot::Receiver<u32> {
        match subscription {
            Subscription::Receiver(rx) => rx,
            Subscription::Ready(state) => panic!("expected to wait, got {state}"),
        }
    }

    #[test]
    fn a_state_sent_before_its_probe_subscribes_is_ready() {
        let mut slots = BridgeSlots::new();

        slots.send(1, Ok(10)).unwrap();

        assert!(matches!(slots.subscribe(1), Subscription::Ready(10)));
    }

    #[tokio::test]
    async fn a_state_sent_after_its_probe_subscribes_is_received() {
        let mut slots = BridgeSlots::new();
        let rx = received(slots.subscribe(1));

        slots.send(1, Ok(10)).unwrap();

        assert_eq!(rx.await.unwrap(), 10);
    }

    #[tokio::test]
    async fn closing_releases_a_probe_waiting_for_a_state_never_sent() {
        let mut slots = BridgeSlots::<u32>::new();
        let rx = received(slots.subscribe(1));

        slots.close();

        assert!(rx.await.is_err());
    }

    #[tokio::test]
    async fn a_probe_subscribing_after_the_close_is_released_at_once() {
        let mut slots = BridgeSlots::<u32>::new();
        slots.close();

        assert!(received(slots.subscribe(1)).await.is_err());
    }

    #[test]
    fn a_state_sent_before_the_close_stays_available() {
        let mut slots = BridgeSlots::new();
        slots.send(1, Ok(10)).unwrap();

        slots.close();

        assert!(matches!(slots.subscribe(1), Subscription::Ready(10)));
    }

    #[tokio::test]
    async fn a_failed_state_returns_its_error_and_hands_nothing_over() {
        let mut slots = BridgeSlots::<u32>::new();
        let rx = received(slots.subscribe(1));

        let sent = slots.send(
            1,
            Err(DaftError::ComputeError("finalizing failed".to_string())),
        );
        slots.close();

        assert!(
            matches!(sent, Err(DaftError::ComputeError(message)) if message == "finalizing failed")
        );
        assert!(rx.await.is_err());
    }
}
