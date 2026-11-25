//! A part of a plan computed once and handed to each of its readers.
//!
//! The part runs as one pipeline. Every message it sends is appended once to a
//! log, and each reader takes the log from its own cursor at its own pace, so
//! one reader waiting on another (a join's probe side waiting for its build
//! side) never holds the others back.
//!
//! The part is read while some reader has taken everything logged so far; once
//! every reader lags, it waits, so it never runs ahead of all of them. What
//! lagging readers have yet to take is held in memory up to a fixed cap and the
//! query's memory budget, and spills to disk past either, to be replayed in
//! order. An entry is freed once every reader has passed it, and the part stops
//! once every reader has stopped.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

use common_daft_config::DaftExecutionConfig;
use common_display::tree::TreeDisplay;
use common_error::{DaftError, DaftResult};
use common_metrics::{
    Meter,
    ops::{NodeCategory, NodeInfo, NodeType},
};
use daft_local_plan::{InputId, LocalNodeContext};
use daft_logical_plan::stats::StatsState;
use daft_micropartition::MicroPartition;
use snafu::ResultExt;
use tokio::sync::Notify;

use crate::{
    ExecutionRuntimeContext, PipelineCreationSnafu,
    channel::{Receiver, Sender, create_channel},
    pipeline::{BuilderContext, MorselSizeRequirement, PipelineMessage, PipelineNode},
    resource_manager::SpillBudget,
    runtime_stats::{DefaultRuntimeStats, RuntimeStats, RuntimeStatsManagerHandle},
    spill::{SpillContext, SpilledRun},
};

/// The part's pipeline and its readers' outputs, gathered while the readers are
/// built and handed to the fan-out once the last of them starts.
pub(crate) struct SharedHub {
    id: usize,
    setup: Mutex<HubSetup>,
}

struct HubSetup {
    /// The part, held by its first reader for display and left here when that
    /// reader starts.
    body: Option<Box<dyn PipelineNode>>,
    /// One output per reader built, in the order they were built, with that
    /// reader's statistics once it has started.
    outputs: Vec<(Sender<PipelineMessage>, Option<ReaderStats>)>,
    started: usize,
    maintain_order: bool,
    cfg: Arc<DaftExecutionConfig>,
}

impl SharedHub {
    /// A hub for the part `id`, which spills and caps its log as `cfg` says.
    pub(crate) fn new(id: usize, cfg: Arc<DaftExecutionConfig>) -> Arc<Self> {
        Arc::new(Self {
            id,
            setup: Mutex::new(HubSetup {
                body: None,
                outputs: Vec::new(),
                started: 0,
                maintain_order: false,
                cfg,
            }),
        })
    }

    fn setup(&self) -> MutexGuard<'_, HubSetup> {
        self.setup.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Adds a reader; returns its place among the readers and the end it receives
    /// the part's messages on.
    fn add_reader(&self) -> (usize, Receiver<PipelineMessage>) {
        let (sender, receiver) = create_channel(1);
        let mut setup = self.setup();
        setup.outputs.push((sender, None));
        (setup.outputs.len() - 1, receiver)
    }

    /// Records one reader's start; the last reader to start starts the part.
    ///
    /// Every pipeline node starts its children as it starts, so every reader
    /// has been built, and starts, before the query runs.
    fn start_reader(
        &self,
        reader: usize,
        stats: ReaderStats,
        body: Option<Box<dyn PipelineNode>>,
        maintain_order: bool,
        runtime_handle: &mut ExecutionRuntimeContext,
    ) -> crate::Result<()> {
        let mut setup = self.setup();
        if body.is_some() {
            setup.body = body;
        }
        if let Some((_, slot)) = setup.outputs.get_mut(reader) {
            *slot = Some(stats);
        }
        setup.maintain_order |= maintain_order;
        setup.started += 1;
        if setup.started < setup.outputs.len() {
            return Ok(());
        }
        let name = format!("Shared #{}", self.id);
        let body = setup
            .body
            .take()
            .ok_or_else(|| DaftError::InternalError(format!("{name} started without its body")))
            .context(PipelineCreationSnafu { plan_name: &name })?;
        let outputs = std::mem::take(&mut setup.outputs);
        let maintain_order = setup.maintain_order;
        let cfg = setup.cfg.clone();
        drop(setup);

        let receiver = body.start(maintain_order, runtime_handle)?;
        let fanout = Arc::new(Fanout::new(
            outputs.len(),
            SpillBudget::new(runtime_handle.memory_manager()),
            SpillContext::from_config(&cfg),
            cfg.shared_part_buffer_bytes as u64,
        ));
        for (reader, (output, stats)) in outputs.into_iter().enumerate() {
            let stats = stats
                .ok_or_else(|| {
                    DaftError::InternalError(format!(
                        "reader {reader} of {name} started without its statistics"
                    ))
                })
                .context(PipelineCreationSnafu { plan_name: &name })?;
            let fanout = fanout.clone();
            runtime_handle.spawn(
                async move {
                    let mut active = false;
                    let served = fanout
                        .serve(reader, output, |rows| {
                            if !active {
                                stats.manager.activate_node(stats.node_id);
                                active = true;
                            }
                            stats.runtime_stats.add_rows_out(rows);
                        })
                        .await;
                    stats.manager.finalize_node(stats.node_id);
                    served
                },
                &name,
            );
        }
        runtime_handle.spawn(async move { fanout.read(receiver).await }, &name);
        Ok(())
    }
}

/// Where one reader reports what it passes on.
struct ReaderStats {
    node_id: usize,
    runtime_stats: Arc<dyn RuntimeStats>,
    manager: RuntimeStatsManagerHandle,
}

/// One entry of the log.
enum Entry {
    /// A morsel held in memory, with its size as charged to the budget.
    Morsel {
        input_id: InputId,
        partition: MicroPartition,
        bytes: u64,
    },
    /// Consecutive morsels of one input spilled to disk as one run; it stands
    /// for `span` positions of the log, the rest of which are `Covered`.
    Spilled {
        input_id: InputId,
        run: Arc<SpilledRun>,
        span: u64,
    },
    /// A position a spilled run stands for; no reader stops on one.
    Covered,
    /// Any other message, passed on as it is.
    Message(PipelineMessage),
}

/// What a reader takes from the log at its cursor.
enum Taken {
    Message(PipelineMessage),
    Spilled {
        input_id: InputId,
        run: Arc<SpilledRun>,
        span: u64,
    },
}

/// The log and the cursors of the readers on it.
struct Log {
    /// `entries[i]` is position `base + i` of the log.
    entries: VecDeque<Entry>,
    base: u64,
    /// Each reader's next position, or `None` once it has stopped.
    cursors: Vec<Option<u64>>,
    /// Whether the part has sent its last message.
    finished: bool,
    /// Bytes of the morsels held in memory.
    resident: u64,
    budget: SpillBudget,
}

impl Log {
    fn end(&self) -> u64 {
        self.base + self.entries.len() as u64
    }

    fn live(&self) -> impl Iterator<Item = u64> + '_ {
        self.cursors.iter().filter_map(|cursor| *cursor)
    }

    /// Whether some reader has taken everything logged so far, or none is left.
    fn wants_more(&self) -> bool {
        let end = self.end();
        let mut live = self.live().peekable();
        live.peek().is_none() || self.live().any(|cursor| cursor == end)
    }

    fn entry(&self, position: u64) -> Option<&Entry> {
        position
            .checked_sub(self.base)
            .and_then(|index| self.entries.get(usize::try_from(index).ok()?))
    }

    /// Frees the entries every reader has passed.
    fn free_passed(&mut self) {
        let low = self.live().min().unwrap_or_else(|| self.end());
        while self.base < low {
            if let Some(Entry::Morsel { bytes, .. }) = self.entries.pop_front() {
                self.resident -= bytes;
                self.budget.shrink(bytes);
            }
            self.base += 1;
        }
    }

    /// The longest stretch of consecutive morsels of one input held in memory
    /// that no reader is in the middle of: from just past a reader's position to
    /// the next reader's position or the end.
    fn spillable(&self) -> Option<(u64, u64)> {
        let mut bounds: Vec<u64> = self.live().collect();
        bounds.sort_unstable();
        bounds.dedup();
        let end = self.end();
        let mut best: Option<(u64, u64)> = None;
        for (i, cursor) in bounds.iter().enumerate() {
            let limit = bounds.get(i + 1).copied().unwrap_or(end);
            // The reader at `cursor` may already hold that entry; spill past it.
            let mut start = cursor + 1;
            while start < limit {
                let Some(Entry::Morsel { input_id, .. }) = self.entry(start) else {
                    start += 1;
                    continue;
                };
                let mut stop = start + 1;
                while stop < limit
                    && matches!(self.entry(stop), Some(Entry::Morsel { input_id: next, .. }) if next == input_id)
                {
                    stop += 1;
                }
                if best.is_none_or(|(s, e)| stop - start > e - s) {
                    best = Some((start, stop));
                }
                start = stop;
            }
        }
        best
    }
}

/// The fan-out of one shared part: its log and the signals between the task
/// reading the part and the tasks serving its readers.
struct Fanout {
    log: Mutex<Log>,
    /// Signalled when the log grows or the part finishes.
    appended: Notify,
    /// Signalled when a reader catches up or stops.
    advanced: Notify,
    spill: Option<SpillContext>,
    /// Bytes of morsels held in memory past which they spill, whatever the budget.
    cap: u64,
}

impl Fanout {
    fn new(readers: usize, budget: SpillBudget, spill: Option<SpillContext>, cap: u64) -> Self {
        Self {
            log: Mutex::new(Log {
                entries: VecDeque::new(),
                base: 0,
                cursors: vec![Some(0); readers],
                finished: false,
                resident: 0,
                budget,
            }),
            appended: Notify::new(),
            advanced: Notify::new(),
            spill,
            cap,
        }
    }

    fn log(&self) -> MutexGuard<'_, Log> {
        self.log.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Reads the part into the log while some reader wants more, until the part
    /// ends or every reader has stopped.
    async fn read(&self, mut part: Receiver<PipelineMessage>) -> DaftResult<()> {
        loop {
            if !self.wait_for_taker().await {
                break;
            }
            let Some(message) = part.recv().await else {
                break;
            };
            self.append(message).await?;
        }
        self.log().finished = true;
        self.appended.notify_waiters();
        Ok(())
    }

    /// Waits until some reader has taken everything logged; returns whether any
    /// reader is left.
    async fn wait_for_taker(&self) -> bool {
        loop {
            let advanced = self.advanced.notified();
            tokio::pin!(advanced);
            advanced.as_mut().enable();
            {
                let log = self.log();
                if log.live().next().is_none() {
                    return false;
                }
                if log.wants_more() {
                    return true;
                }
            }
            advanced.await;
        }
    }

    async fn append(&self, message: PipelineMessage) -> DaftResult<()> {
        let mut under_pressure = false;
        let pressure = {
            let mut log = self.log();
            let entry = match message {
                PipelineMessage::Morsel {
                    input_id,
                    partition,
                } => {
                    let bytes = partition.size_bytes() as u64;
                    log.resident += bytes;
                    // Honor any request from other holders to shed, then account the morsel.
                    let fits = log.budget.take_shed_request() == 0 && log.budget.try_grow(bytes);
                    if !fits {
                        // Held all the same until spilled, so it is counted.
                        log.budget.grow_unchecked(bytes);
                    }
                    under_pressure = !fits;
                    Entry::Morsel {
                        input_id,
                        partition,
                        bytes,
                    }
                }
                other => Entry::Message(other),
            };
            log.entries.push_back(entry);
            under_pressure || log.resident > self.cap
        };
        self.appended.notify_waiters();
        if pressure {
            self.shed().await?;
        }
        Ok(())
    }

    /// Spills morsels lagging readers have yet to take until those held in
    /// memory fit the cap again, or nothing more can be spilled.
    async fn shed(&self) -> DaftResult<()> {
        let Some(spill) = &self.spill else {
            return Ok(());
        };
        loop {
            let (range, input_id, parts) = {
                let log = self.log();
                if log.resident <= self.cap / 2 {
                    return Ok(());
                }
                let Some((start, stop)) = log.spillable() else {
                    return Ok(());
                };
                let mut parts = Vec::new();
                let mut input = None;
                for position in start..stop {
                    if let Some(Entry::Morsel {
                        input_id,
                        partition,
                        ..
                    }) = log.entry(position)
                    {
                        input = Some(*input_id);
                        parts.push(partition.clone());
                    }
                }
                let Some(input_id) = input else {
                    return Ok(());
                };
                ((start, stop), input_id, parts)
            };
            let run = spill.scratch()?.spill(parts, spill.compression()).await?;
            let mut log = self.log();
            let (start, stop) = range;
            // A reader that moved into the stretch while it was written keeps the
            // morsels in memory; the run is dropped with the scratch directory.
            if start < log.base || log.live().any(|cursor| start <= cursor && cursor < stop) {
                return Ok(());
            }
            let mut freed = 0;
            let mut run = Some(Arc::new(run));
            for position in start..stop {
                let index = usize::try_from(position - log.base)
                    .map_err(|e| DaftError::InternalError(e.to_string()))?;
                let replacement = match run.take() {
                    Some(run) => Entry::Spilled {
                        input_id,
                        run,
                        span: stop - start,
                    },
                    None => Entry::Covered,
                };
                if let Entry::Morsel { bytes, .. } =
                    std::mem::replace(&mut log.entries[index], replacement)
                {
                    freed += bytes;
                }
            }
            log.resident -= freed;
            log.budget.shrink(freed);
        }
    }

    /// Sends the log to one reader from its cursor until the part has ended, or
    /// the reader stops taking.
    ///
    /// `delivered_rows` is called with the rows of each morsel as it is passed on.
    async fn serve(
        &self,
        reader: usize,
        output: Sender<PipelineMessage>,
        mut delivered_rows: impl FnMut(u64),
    ) -> DaftResult<()> {
        while let Some(taken) = self.next_for(reader).await? {
            let delivered = match taken {
                Taken::Message(message) => {
                    if let PipelineMessage::Morsel { partition, .. } = &message {
                        delivered_rows(partition.len() as u64);
                    }
                    let delivered = output.send(message).await.is_ok();
                    delivered.then_some(1)
                }
                Taken::Spilled {
                    input_id,
                    run,
                    span,
                } => {
                    let mut cursor = run.reader();
                    let mut delivered = Some(span);
                    while let Some(batch) = cursor.next_batch().await? {
                        let partition = MicroPartition::new_loaded(
                            batch.schema.clone(),
                            Arc::new(vec![batch]),
                            None,
                        );
                        delivered_rows(partition.len() as u64);
                        let message = PipelineMessage::Morsel {
                            input_id,
                            partition,
                        };
                        if output.send(message).await.is_err() {
                            delivered = None;
                            break;
                        }
                    }
                    delivered
                }
            };
            let mut log = self.log();
            match delivered {
                Some(span) => {
                    let cursor = log.cursors[reader].map(|position| position + span);
                    log.cursors[reader] = cursor;
                }
                None => log.cursors[reader] = None,
            }
            log.free_passed();
            let caught_up = log.cursors[reader].is_none_or(|cursor| cursor == log.end());
            drop(log);
            if caught_up {
                self.advanced.notify_waiters();
            }
            if delivered.is_none() {
                break;
            }
        }
        Ok(())
    }

    /// The next entry for `reader`, waiting for one; `None` once the part has
    /// ended and the reader has taken everything, or the reader has stopped.
    async fn next_for(&self, reader: usize) -> DaftResult<Option<Taken>> {
        loop {
            let appended = self.appended.notified();
            tokio::pin!(appended);
            appended.as_mut().enable();
            {
                let log = self.log();
                let Some(cursor) = log.cursors[reader] else {
                    return Ok(None);
                };
                match log.entry(cursor) {
                    Some(Entry::Morsel {
                        input_id,
                        partition,
                        ..
                    }) => {
                        return Ok(Some(Taken::Message(PipelineMessage::Morsel {
                            input_id: *input_id,
                            partition: partition.clone(),
                        })));
                    }
                    Some(Entry::Spilled {
                        input_id,
                        run,
                        span,
                    }) => {
                        return Ok(Some(Taken::Spilled {
                            input_id: *input_id,
                            run: run.clone(),
                            span: *span,
                        }));
                    }
                    Some(Entry::Message(message)) => {
                        return Ok(Some(Taken::Message(copy_message(message))));
                    }
                    // Readers move past a spilled run as a whole, and a run is only
                    // written where no reader stands.
                    Some(Entry::Covered) => {
                        return Err(DaftError::InternalError(format!(
                            "a reader stopped inside a spilled run at position {cursor}"
                        )));
                    }
                    None if log.finished => return Ok(None),
                    None => {}
                }
            }
            appended.await;
        }
    }
}

/// A copy of a message that holds no morsel.
fn copy_message(message: &PipelineMessage) -> PipelineMessage {
    match message {
        PipelineMessage::Morsel {
            input_id,
            partition,
        } => PipelineMessage::Morsel {
            input_id: *input_id,
            partition: partition.clone(),
        },
        PipelineMessage::FlightPartitionRef {
            input_id,
            partition_ref,
        } => PipelineMessage::FlightPartitionRef {
            input_id: *input_id,
            partition_ref: partition_ref.clone(),
        },
        PipelineMessage::Flush(input_id) => PipelineMessage::Flush(*input_id),
    }
}

/// One reader of a shared part.
///
/// The first reader built holds the part's pipeline, so it is shown, and sized,
/// once.
pub(crate) struct SharedReaderNode {
    hub: Arc<SharedHub>,
    reader: usize,
    body: Option<Box<dyn PipelineNode>>,
    receiver: Receiver<PipelineMessage>,
    meter: Meter,
    plan_stats: StatsState,
    node_info: Arc<NodeInfo>,
}

impl SharedReaderNode {
    /// The first reader of the part `hub` stands for, holding the part's pipeline.
    pub(crate) fn first(
        hub: Arc<SharedHub>,
        body: Box<dyn PipelineNode>,
        plan_stats: StatsState,
        ctx: &BuilderContext,
        context: &LocalNodeContext,
    ) -> Self {
        Self::new(hub, Some(body), plan_stats, ctx, context)
    }

    /// A further reader of the part `hub` stands for.
    pub(crate) fn another(
        hub: Arc<SharedHub>,
        plan_stats: StatsState,
        ctx: &BuilderContext,
        context: &LocalNodeContext,
    ) -> Self {
        Self::new(hub, None, plan_stats, ctx, context)
    }

    fn new(
        hub: Arc<SharedHub>,
        body: Option<Box<dyn PipelineNode>>,
        plan_stats: StatsState,
        ctx: &BuilderContext,
        context: &LocalNodeContext,
    ) -> Self {
        let name: Arc<str> = format!("Shared #{}", hub.id).into();
        let node_info =
            ctx.next_node_info(name, NodeType::Shared, NodeCategory::Intermediate, context);
        let (reader, receiver) = hub.add_reader();
        Self {
            hub,
            reader,
            body,
            receiver,
            meter: ctx.meter.clone(),
            plan_stats,
            node_info: Arc::new(node_info),
        }
    }

    /// The reader as a pipeline node.
    pub(crate) fn boxed(self) -> Box<dyn PipelineNode> {
        Box::new(self)
    }
}

impl TreeDisplay for SharedReaderNode {
    fn id(&self) -> String {
        self.node_id().to_string()
    }

    fn display_as(&self, level: common_display::DisplayLevel) -> String {
        use std::fmt::Write;
        let mut display = String::new();
        writeln!(display, "{}", self.name()).unwrap_or_default();
        if !matches!(level, common_display::DisplayLevel::Compact)
            && let StatsState::Materialized(stats) = &self.plan_stats
        {
            writeln!(display, "Stats = {stats}").unwrap_or_default();
        }
        display
    }

    #[recursive::recursive]
    fn repr_json(&self) -> serde_json::Value {
        let children: Vec<serde_json::Value> = self
            .get_children()
            .iter()
            .map(|child| child.repr_json())
            .collect();
        let mut json = serde_json::json!({
            "id": self.node_id(),
            "category": "Intermediate",
            "type": NodeType::Shared.to_string(),
            "name": self.name(),
        });
        json["children"] = serde_json::Value::Array(children);
        if let StatsState::Materialized(stats) = &self.plan_stats {
            json["approx_stats"] = serde_json::json!(stats);
        }
        json
    }

    fn get_children(&self) -> Vec<&dyn TreeDisplay> {
        self.children()
            .iter()
            .map(|child| child.as_tree_display())
            .collect()
    }
}

impl PipelineNode for SharedReaderNode {
    fn children(&self) -> Vec<&dyn PipelineNode> {
        self.body.iter().map(AsRef::as_ref).collect()
    }

    fn boxed_children(&self) -> Vec<&Box<dyn PipelineNode>> {
        self.body.iter().collect()
    }

    fn name(&self) -> Arc<str> {
        self.node_info.name.clone()
    }

    fn propagate_morsel_size_requirement(
        &mut self,
        _downstream_requirement: MorselSizeRequirement,
        default_requirement: MorselSizeRequirement,
    ) {
        // The part serves every reader alike, so it takes no one reader's size.
        if let Some(body) = &mut self.body {
            body.propagate_morsel_size_requirement(default_requirement, default_requirement);
        }
    }

    fn start(
        self: Box<Self>,
        maintain_order: bool,
        runtime_handle: &mut ExecutionRuntimeContext,
    ) -> crate::Result<Receiver<PipelineMessage>> {
        let Self {
            hub,
            reader,
            body,
            receiver,
            meter,
            node_info,
            ..
        } = *self;
        let manager = runtime_handle.stats_manager();
        let runtime_stats: Arc<dyn RuntimeStats> =
            Arc::new(DefaultRuntimeStats::new(&meter, &node_info));
        manager.register_runtime_stats(node_info.id, 0, runtime_stats.clone());
        let stats = ReaderStats {
            node_id: node_info.id,
            runtime_stats,
            manager,
        };
        hub.start_reader(reader, stats, body, maintain_order, runtime_handle)?;
        Ok(receiver)
    }

    fn as_tree_display(&self) -> &dyn TreeDisplay {
        self
    }

    fn node_id(&self) -> usize {
        self.node_info.id
    }

    fn node_info(&self) -> Arc<NodeInfo> {
        self.node_info.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use common_daft_config::DaftExecutionConfig;
    use daft_core::{datatypes::Int64Array, series::IntoSeries};
    use daft_micropartition::MicroPartition;
    use daft_recordbatch::RecordBatch;

    use super::{Entry, Fanout};
    use crate::{
        channel::{Receiver, create_channel},
        pipeline::PipelineMessage,
        resource_manager::{MemoryManager, QueryMemoryScope, SpillBudget},
        spill::SpillContext,
    };

    fn morsel(values: Vec<i64>) -> PipelineMessage {
        let series = Int64Array::from_vec("v", values).into_series();
        let batch = RecordBatch::from_nonempty_columns(vec![series]).unwrap();
        PipelineMessage::Morsel {
            input_id: 0,
            partition: MicroPartition::new_loaded(
                batch.schema.clone(),
                Arc::new(vec![batch]),
                None,
            ),
        }
    }

    /// A fan-out to `readers` readers that spills past `cap` bytes into `dir`.
    fn fanout(readers: usize, cap: u64, dir: &std::path::Path) -> Arc<Fanout> {
        let cfg = DaftExecutionConfig {
            spill_dirs: vec![dir.to_str().unwrap().to_string()],
            ..Default::default()
        };
        let scope = QueryMemoryScope::new(Arc::new(MemoryManager::new()), None);
        Arc::new(Fanout::new(
            readers,
            SpillBudget::new(scope),
            SpillContext::from_config(&cfg),
            cap,
        ))
    }

    /// Sends `count` morsels of two rows, then the end of input 0, through a part.
    fn part(count: i64) -> Receiver<PipelineMessage> {
        let (sender, receiver) = create_channel(1);
        tokio::spawn(async move {
            for i in 0..count {
                if sender.send(morsel(vec![2 * i, 2 * i + 1])).await.is_err() {
                    return;
                }
            }
            let _ = sender.send(PipelineMessage::Flush(0)).await;
        });
        receiver
    }

    /// Everything a reader receives: its values in order, and whether input 0 ended.
    async fn drain(mut receiver: Receiver<PipelineMessage>) -> (Vec<i64>, bool) {
        let mut values = vec![];
        let mut flushed = false;
        while let Some(message) = receiver.recv().await {
            match message {
                PipelineMessage::Morsel { partition, .. } => {
                    for batch in partition.record_batches() {
                        values.extend(batch.get_column(0).i64().unwrap().values().iter().copied());
                    }
                }
                PipelineMessage::Flush(0) => flushed = true,
                other => panic!("unexpected message {other:?}"),
            }
        }
        (values, flushed)
    }

    /// Starts the fan-out and its readers; returns each reader's receiving end.
    fn start(
        fanout: &Arc<Fanout>,
        part: Receiver<PipelineMessage>,
        readers: usize,
    ) -> Vec<Receiver<PipelineMessage>> {
        let mut receivers = vec![];
        for reader in 0..readers {
            let (sender, receiver) = create_channel(1);
            let fanout = fanout.clone();
            tokio::spawn(async move { fanout.serve(reader, sender, |_| {}).await });
            receivers.push(receiver);
        }
        let reading = fanout.clone();
        tokio::spawn(async move { reading.read(part).await });
        receivers
    }

    fn all_values(count: i64) -> Vec<i64> {
        (0..2 * count).collect()
    }

    #[tokio::test]
    async fn every_reader_receives_every_message_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let fanout = fanout(3, 1 << 20, dir.path());
        let receivers = start(&fanout, part(50), 3);
        for receiver in receivers {
            assert_eq!(drain(receiver).await, (all_values(50), true));
        }
    }

    #[tokio::test]
    async fn a_reader_that_waits_gets_every_row_spilled_and_in_order() {
        let dir = tempfile::tempdir().unwrap();
        // A cap of one byte spills everything the waiting reader has not taken.
        let fanout = fanout(2, 1, dir.path());
        let mut receivers = start(&fanout, part(40), 2);
        let waiting = receivers.pop().unwrap();
        let eager = receivers.pop().unwrap();
        // The eager reader takes the whole part while the other takes nothing.
        assert_eq!(drain(eager).await, (all_values(40), true));
        {
            let log = fanout.log();
            assert!(log.finished);
            assert!(
                log.entries
                    .iter()
                    .any(|entry| matches!(entry, Entry::Spilled { .. }))
            );
            assert!(log.resident <= 1 + 64, "resident {} bytes", log.resident);
        }
        assert_eq!(drain(waiting).await, (all_values(40), true));
        // Every reader has passed every entry, so nothing is held.
        let log = fanout.log();
        assert!(log.entries.is_empty());
        assert_eq!(log.resident, 0);
    }

    #[tokio::test]
    async fn a_reader_that_stops_leaves_the_others_their_rows() {
        let dir = tempfile::tempdir().unwrap();
        let fanout = fanout(2, 1 << 20, dir.path());
        let mut receivers = start(&fanout, part(30), 2);
        drop(receivers.pop());
        assert_eq!(
            drain(receivers.pop().unwrap()).await,
            (all_values(30), true)
        );
    }

    #[tokio::test]
    async fn the_part_stops_once_every_reader_has_stopped() {
        let dir = tempfile::tempdir().unwrap();
        let fanout = fanout(2, 1 << 20, dir.path());
        let (sender, receiver) = create_channel(1);
        let receivers = start(&fanout, receiver, 2);
        drop(receivers);
        // The part's output closes: it is no longer read once no reader is left.
        let mut sent = 0;
        while sender.send(morsel(vec![sent])).await.is_ok() {
            sent += 1;
            assert!(sent < 1000, "the part kept being read with no reader left");
        }
    }
}
