use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use common_daft_config::DaftExecutionConfig;
use common_display::{DisplayLevel, mermaid::MermaidDisplayOptions};
use common_error::DaftResult;
use common_metrics::{QueryEndState, QueryID};
use common_runtime::RuntimeTask;
use common_tracing::flush_opentelemetry_providers;
use daft_context::{
    DaftContext, Subscriber,
    subscribers::{
        Event, event_header,
        events::{TaskInfo, TaskStartEvent},
    },
};
use daft_local_plan::{ExecutionStats, Input, InputId, LocalPhysicalPlanRef, SourceId, translate};
use daft_logical_plan::LogicalPlanBuilder;
use daft_micropartition::MicroPartition;
use daft_partition_refs::FlightPartitionRef;
use daft_shuffles::server::flight_server::{
    FlightServerConnectionHandle, ShuffleFlightServer, start_server_loop,
};
use futures::{FutureExt, future::BoxFuture};
use tokio::runtime::Handle;
use tokio_util::sync::CancellationToken;
#[cfg(feature = "python")]
use {
    common_daft_config::PyDaftExecutionConfig,
    daft_context::python::PyDaftContext,
    daft_local_plan::python::PyExecutionStats,
    daft_logical_plan::PyLogicalPlanBuilder,
    daft_micropartition::python::PyMicroPartition,
    daft_partition_refs::PyFlightPartitionRef,
    pyo3::{
        Bound, IntoPyObject, PyAny, PyRef, PyResult, Python, pyclass, pymethods, sync::MutexExt,
    },
};

use crate::{
    ExecutionRuntimeContext,
    channel::{Sender, UnboundedSender, create_channel, create_unbounded_channel},
    pipeline::{
        BuilderContext, PipelineMessage, translate_physical_plan_to_pipeline, viz_pipeline_ascii,
        viz_pipeline_mermaid,
    },
    resource_manager::get_or_init_memory_manager,
    runtime_stats::{RuntimeStatsManager, RuntimeStatsManagerHandle},
};

enum ExecutionEngineResultItem {
    Partition(MicroPartition),
    FlightPartitionRef(FlightPartitionRef),
}

/// Global tokio runtime shared by all NativeExecutor instances
static GLOBAL_RUNTIME: OnceLock<Handle> = OnceLock::new();

/// Get or initialize the global tokio runtime
#[cfg(feature = "python")]
fn get_global_runtime() -> &'static Handle {
    GLOBAL_RUNTIME.get_or_init(|| {
        let mut builder = tokio::runtime::Builder::new_current_thread();
        builder.enable_all();
        pyo3_async_runtimes::tokio::init(builder);
        std::thread::spawn(move || {
            pyo3_async_runtimes::tokio::get_runtime().block_on(futures::future::pending::<()>());
        });
        pyo3_async_runtimes::tokio::get_runtime().handle().clone()
    })
}

#[cfg(not(feature = "python"))]
fn get_global_runtime() -> &'static Handle {
    GLOBAL_RUNTIME.get_or_init(|| {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("build global tokio runtime for NativeExecutor");
        let handle = rt.handle().clone();
        // Keep the runtime alive for the duration of the process.
        std::thread::spawn(move || {
            rt.block_on(futures::future::pending::<()>());
        });
        handle
    })
}

/// Message sent to the execution task to enqueue inputs
pub(crate) struct EnqueueInputMessage {
    /// The input_id for this enqueue operation
    input_id: InputId,
    /// Plan inputs grouped by source_id
    inputs: HashMap<SourceId, Input>,
    /// Sender for results of this input_id
    result_sender: ResultSender,
}

/// Where one input's results go.
///
/// A bounded channel holds the pipeline back while its reader falls behind; an
/// unbounded one, for a caller that takes everything, never waits and costs no
/// more than a plain send.
#[derive(Clone)]
enum ResultSender {
    Bounded(Sender<ExecutionEngineResultItem>),
    Unbounded(UnboundedSender<ExecutionEngineResultItem>),
}

/// The reading end of a [`ResultSender`].
enum ResultReceiver {
    Bounded(crate::channel::Receiver<ExecutionEngineResultItem>),
    Unbounded(crate::channel::UnboundedReceiver<ExecutionEngineResultItem>),
}

impl ResultReceiver {
    async fn recv(&mut self) -> Option<ExecutionEngineResultItem> {
        match self {
            Self::Bounded(receiver) => receiver.recv().await,
            Self::Unbounded(receiver) => receiver.recv().await,
        }
    }
}

/// A channel for one input's results, bounded to `buffer_size` results when one is given.
fn result_channel(buffer_size: Option<usize>) -> (ResultSender, ResultReceiver) {
    match buffer_size {
        Some(size) => {
            let (sender, receiver) = create_channel(size);
            (
                ResultSender::Bounded(sender),
                ResultReceiver::Bounded(receiver),
            )
        }
        None => {
            let (sender, receiver) = create_unbounded_channel();
            (
                ResultSender::Unbounded(sender),
                ResultReceiver::Unbounded(receiver),
            )
        }
    }
}

/// A result the loop is waiting to hand to its consumer.
type PendingDelivery = BoxFuture<'static, ()>;

/// Routes pipeline messages to per-input_id channels.
struct MessageRouter {
    output_senders: HashMap<InputId, ResultSender>,
    /// Wall-clock start instant when each `input_id` was enqueued to the pipeline.
    input_start_times: HashMap<InputId, Instant>,
}

impl MessageRouter {
    fn new() -> Self {
        Self {
            output_senders: HashMap::new(),
            input_start_times: HashMap::new(),
        }
    }

    /// Routes a message to the channel of its input_id.
    ///
    /// Returns the delivery of a result, which completes once its consumer's
    /// buffer has room; the caller awaits it before routing the next message,
    /// so a full buffer holds the pipeline back and results keep their order.
    /// A consumer that has gone away fails the delivery at once, dropping it.
    fn route_message(&mut self, msg: PipelineMessage) -> Option<PendingDelivery> {
        let (input_id, item) = match msg {
            PipelineMessage::Flush(input_id) => {
                self.input_start_times.remove(&input_id);
                self.output_senders.remove(&input_id);
                return None;
            }
            PipelineMessage::Morsel {
                input_id,
                partition,
            } => (input_id, ExecutionEngineResultItem::Partition(partition)),
            PipelineMessage::FlightPartitionRef {
                input_id,
                partition_ref,
            } => (
                input_id,
                ExecutionEngineResultItem::FlightPartitionRef(partition_ref),
            ),
        };
        match self.output_senders.get(&input_id)? {
            ResultSender::Unbounded(sender) => {
                let _ = sender.send(item);
                None
            }
            ResultSender::Bounded(sender) => {
                let sender = sender.clone();
                Some(
                    async move {
                        let _ = sender.send(item).await;
                    }
                    .boxed(),
                )
            }
        }
    }

    fn insert_output_sender(&mut self, input_id: InputId, sender: ResultSender) {
        self.input_start_times.insert(input_id, Instant::now());
        self.output_senders.insert(input_id, sender);
    }
}

impl Drop for MessageRouter {
    fn drop(&mut self) {
        for (input_id, started) in self.input_start_times.drain() {
            log::debug!(
                "NativeExecutor: input_id={input_id} ended without Flush after {:?} (cancel/shutdown?)",
                started.elapsed()
            );
        }
    }
}

/// Per-plan execution state
struct PlanState {
    task_handle: RuntimeTask<DaftResult<()>>,
    enqueue_input_sender: Sender<EnqueueInputMessage>,
    stats_handle: RuntimeStatsManagerHandle,
    active_input_ids: HashSet<InputId>,
    skipped_corrupt_files: Arc<std::sync::Mutex<Vec<(String, String, bool)>>>,
}

#[cfg_attr(
    feature = "python",
    pyclass(module = "daft.daft", name = "NativeExecutor", frozen)
)]
pub struct PyNativeExecutor {
    executor: Arc<Mutex<NativeExecutor>>,
    address: Option<String>,
}

#[cfg(feature = "python")]
impl Default for PyNativeExecutor {
    fn default() -> Self {
        Self::new(false, "")
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PyNativeExecutor {
    #[new]
    pub fn new(is_flotilla_worker: bool, ip: &str) -> Self {
        let executor = NativeExecutor::new(is_flotilla_worker, ip);
        let address = executor.shuffle_address();
        Self {
            executor: Arc::new(Mutex::new(executor)),
            address,
        }
    }

    pub fn shuffle_address(&self) -> Option<String> {
        self.address.clone()
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (local_physical_plan, daft_ctx, input_id, inputs, context=None, maintain_order=true))]
    pub fn run<'py>(
        &self,
        py: Python<'py>,
        local_physical_plan: &daft_local_plan::PyLocalPhysicalPlan,
        daft_ctx: &PyDaftContext,
        input_id: InputId,
        inputs: HashMap<SourceId, Input>,
        context: Option<HashMap<String, String>>,
        maintain_order: bool,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let daft_ctx: &DaftContext = daft_ctx.into();
        let plan = local_physical_plan.plan.clone();
        let exec_cfg = daft_ctx.execution_config();
        let subscribers = daft_ctx.subscribers();
        let (fingerprint, enqueue_future) = {
            self.executor.lock_py_attached(py).unwrap().run(
                &plan,
                exec_cfg,
                subscribers,
                context,
                inputs,
                input_id,
                maintain_order,
            )?
        };

        let executor = self.executor.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = enqueue_future.await?;
            Ok(PyResultReceiver {
                result: Arc::new(tokio::sync::Mutex::new(Some(result))),
                fingerprint,
                input_id,
                executor,
            })
        })
    }

    pub fn active_plan_count(&self, py: Python<'_>) -> usize {
        self.executor.lock_py_attached(py).unwrap().plans.len()
    }

    pub fn cancel_plan(&self, py: Python<'_>, fingerprint: u64) -> PyResult<()> {
        self.executor
            .lock_py_attached(py)
            .unwrap()
            .cancel_plan(fingerprint);
        Ok(())
    }

    #[staticmethod]
    pub fn repr_ascii(
        logical_plan_builder: &PyLogicalPlanBuilder,
        cfg: PyDaftExecutionConfig,
        simple: bool,
    ) -> PyResult<String> {
        Ok(NativeExecutor::repr_ascii(
            &logical_plan_builder.builder,
            cfg.config,
            simple,
        ))
    }

    #[staticmethod]
    pub fn repr_mermaid(
        logical_plan_builder: &PyLogicalPlanBuilder,
        cfg: PyDaftExecutionConfig,
        options: MermaidDisplayOptions,
    ) -> PyResult<String> {
        Ok(NativeExecutor::repr_mermaid(
            &logical_plan_builder.builder,
            cfg.config,
            options,
        ))
    }
}

/// Returns a fingerprint that is unique for each call when the caller does not
/// supply one. Using a fixed value (e.g. 0) caused `NativeExecutor::run` to
/// reuse the cached pipeline from a prior execution when the new plan requires
/// a different `InputSender` variant, which reached the `unreachable!` branch
/// in `InputSender::send` (see GitHub issue #7087).
fn next_auto_fingerprint() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

fn parse_context(
    ctx: Option<&HashMap<String, String>>,
) -> (QueryID, u64, Option<u32>, Option<u64>, Option<usize>) {
    let query_id = ctx
        .as_ref()
        .and_then(|c| c.get("query_id"))
        .map(|s| QueryID::from(s.as_str()))
        .unwrap_or_else(|| QueryID::from(""));
    let fingerprint = ctx
        .as_ref()
        .and_then(|c| c.get("plan_fingerprint"))
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or_else(next_auto_fingerprint);
    let task_id = ctx
        .as_ref()
        .and_then(|c| c.get("task_id"))
        .and_then(|s| s.parse::<u32>().ok());
    // Ceiling on this query's combined buffered bytes; set by an embedding
    // server, never by the client-shipped configuration.
    let memory_cap = ctx
        .as_ref()
        .and_then(|c| c.get("memory_cap_bytes"))
        .and_then(|s| s.parse::<u64>().ok());
    // How many results may wait for the caller before the pipeline is held back.
    // Absent, results never wait: a distributed worker shares one pipeline across
    // tasks, and holding it for one task's reader could stall the others.
    let result_buffer_size = ctx
        .as_ref()
        .and_then(|c| c.get("result_buffer_size"))
        .and_then(|s| s.parse::<usize>().ok())
        .map(|size| size.max(1));

    (
        query_id,
        fingerprint,
        task_id,
        memory_cap,
        result_buffer_size,
    )
}

// TODO: fix configuration for events
// This is copied from task_lifecycle.rs to avoid the daft-distributed dependency
pub fn task_events_enabled() -> bool {
    if let Ok(val) = std::env::var("DAFT_TASK_EVENTS_ENABLED") {
        matches!(val.trim().to_lowercase().as_str(), "1" | "true")
    } else {
        false // Disabled by default; enable with DAFT_TASK_EVENTS_ENABLED=true
    }
}

/// How long the remaining tasks of a failed pipeline may take to end on their own
/// before they are aborted.
const FAILURE_WIND_DOWN_GRACE: std::time::Duration = std::time::Duration::from_secs(10);

/// The core execution loop that drives a pipeline to completion.
/// Receives inputs via `enqueue_input_rx`, routes pipeline outputs to
/// per-input_id channels, and runs until the pipeline finishes, errors,
/// or is cancelled.
async fn run_execution_loop(
    cancel: CancellationToken,
    stats_manager: RuntimeStatsManager,
    mut enqueue_input_rx: crate::channel::Receiver<EnqueueInputMessage>,
    input_senders: Arc<HashMap<SourceId, crate::input_sender::InputSender>>,
    pipeline: Box<dyn crate::pipeline::PipelineNode>,
    maintain_order: bool,
    memory_cap: Option<u64>,
) -> DaftResult<()> {
    let stats_manager_handle = stats_manager.handle();
    let memory_manager = get_or_init_memory_manager();
    let memory_scope =
        crate::resource_manager::QueryMemoryScope::new(memory_manager.clone(), memory_cap);
    let mut runtime_handle = ExecutionRuntimeContext::new(memory_scope, stats_manager_handle);
    let mut output_receiver = pipeline.start(maintain_order, &mut runtime_handle)?;

    let mut message_router = MessageRouter::new();
    let mut input_senders = Some(input_senders);
    let mut input_exhausted = false;
    let mut pending_delivery: Option<PendingDelivery> = None;

    let (result, finish_status) = loop {
        tokio::select! {
            biased;
            () = cancel.cancelled() => {
                println!("Execution engine cancelled");
                break (Ok(()), QueryEndState::Canceled);
            }
            _ = tokio::signal::ctrl_c() => {
                println!("Received Ctrl-C, shutting down execution engine");
                break (Ok(()), QueryEndState::Canceled);
            }
            Some(join_result) = runtime_handle.join_next() => {
                if let Err(e) = join_result {
                    if matches!(&e, common_error::DaftError::JoinError(source) if source.is_cancelled()) {
                        break (Ok(()), QueryEndState::Canceled);
                    }
                    // Close the pipeline's inputs and stop taking its output, so the remaining
                    // tasks end on their own and release what they hold (an unfinished output
                    // file among it) before the failure is reported.
                    input_senders.take();
                    pending_delivery.take();
                    output_receiver.close();
                    runtime_handle.wind_down(FAILURE_WIND_DOWN_GRACE).await;
                    break (Err(e), QueryEndState::Failed);
                }
            }
            enqueue_msg = enqueue_input_rx.recv(), if !input_exhausted => {
                if let Some(EnqueueInputMessage { input_id, inputs, result_sender }) = enqueue_msg {
                    message_router.insert_output_sender(input_id, result_sender);
                    let senders = input_senders.as_ref().unwrap();
                    for (key, plan_input) in inputs {
                        if let Some(sender) = senders.get(&key) {
                            let _ = sender.send(input_id, plan_input);
                        }
                    }
                } else {
                    // All senders dropped — drop input channels so
                    // pipeline sources see EOF.
                    input_senders.take();
                    input_exhausted = true;
                }
            }
            () = async { pending_delivery.as_mut().expect("guarded by is_some").await }, if pending_delivery.is_some() => {
                pending_delivery = None;
            }
            // Nothing more is taken from the pipeline while a result waits for room.
            msg = output_receiver.recv(), if pending_delivery.is_none() => {
                match msg {
                    Some(msg) => {
                        pending_delivery = message_router.route_message(msg);
                    }
                    None => {
                        // Pipeline finished. Close result channels so waiters
                        // unblock, then drain runtime tasks.
                        drop(message_router);
                        let res = runtime_handle.shutdown().await;
                        let status = if res.is_ok() { QueryEndState::Finished } else { QueryEndState::Failed };
                        break (res, status);
                    }
                }
            }
        }
    };

    stats_manager.finish(finish_status).await;
    flush_opentelemetry_providers();
    result
}

pub struct NativeExecutor {
    cancel: CancellationToken,
    is_flotilla_worker: bool,
    shuffle_server: Option<Arc<ShuffleFlightServer>>,
    shuffle_server_connection: Option<FlightServerConnectionHandle>,
    plans: HashMap<u64, PlanState>,
}

impl NativeExecutor {
    pub fn new(is_flotilla_worker: bool, ip: &str) -> Self {
        // Determine if we are running in a flotilla worker.
        if is_flotilla_worker {
            let shuffle_server = Arc::new(ShuffleFlightServer::new());
            let shuffle_server_connection = Some(start_server_loop(ip, shuffle_server.clone()));

            Self {
                cancel: CancellationToken::new(),
                is_flotilla_worker: true,
                shuffle_server: Some(shuffle_server),
                shuffle_server_connection,
                plans: HashMap::new(),
            }
        } else {
            Self {
                cancel: CancellationToken::new(),
                is_flotilla_worker: false,
                shuffle_server: None,
                shuffle_server_connection: None,
                plans: HashMap::new(),
            }
        }
    }

    pub fn shuffle_address(&self) -> Option<String> {
        self.shuffle_server_connection
            .as_ref()
            .map(|conn| conn.shuffle_address())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        local_physical_plan: &LocalPhysicalPlanRef,
        exec_cfg: Arc<DaftExecutionConfig>,
        subscribers: Vec<Arc<dyn Subscriber>>,
        additional_context: Option<HashMap<String, String>>,
        inputs: HashMap<SourceId, Input>,
        input_id: InputId,
        maintain_order: bool,
    ) -> DaftResult<(u64, BoxFuture<'static, DaftResult<ExecutionEngineResult>>)> {
        let (query_id, fingerprint, task_id, memory_cap, result_buffer_size) =
            parse_context(additional_context.as_ref());

        if self.is_flotilla_worker {
            debug_assert_eq!(
                task_id,
                Some(input_id),
                "Flotilla invariant violated: task_id must match input_id"
            );
        }

        let task_start_dispatch = if self.is_flotilla_worker
            && task_events_enabled()
            && let Some(task_id) = task_id
        {
            Some((
                Event::TaskStart(TaskStartEvent {
                    header: event_header(query_id.clone()),
                    task: Arc::new(TaskInfo {
                        id: task_id,
                        last_node_id: 0,  // TODO: propagate last_node_id
                        node_ids: vec![], // TODO: propagate node_ids
                        plan_fingerprint: fingerprint as u32,
                        name: None,
                    }),
                    worker_id: None, // TODO: propagate worker id
                }),
                subscribers.clone(),
            ))
        } else {
            None
        };

        if !self.plans.contains_key(&fingerprint) {
            let cancel = self.cancel.clone();
            let additional_context = additional_context.unwrap_or_default();
            let shuffle_address = self.shuffle_address();
            let ctx = BuilderContext::new_with_context(
                query_id.clone(),
                additional_context,
                self.shuffle_server
                    .as_ref()
                    .map(|server| (server.clone(), shuffle_address.unwrap())),
            );
            let (pipeline, input_senders) =
                translate_physical_plan_to_pipeline(local_physical_plan, &exec_cfg, &ctx)?;

            let handle = get_global_runtime();
            let stats_manager = RuntimeStatsManager::try_new(
                handle,
                &pipeline,
                subscribers,
                query_id,
                self.is_flotilla_worker,
            )?;
            let stats_handle = stats_manager.handle();

            let (enqueue_input_tx, enqueue_input_rx) = create_channel::<EnqueueInputMessage>(1);

            let input_senders = Arc::new(input_senders);
            let task = run_execution_loop(
                cancel,
                stats_manager,
                enqueue_input_rx,
                input_senders,
                pipeline,
                maintain_order,
                memory_cap,
            );

            let task_handle = RuntimeTask::new(handle, task);
            self.plans.insert(
                fingerprint,
                PlanState {
                    task_handle,
                    enqueue_input_sender: enqueue_input_tx,
                    stats_handle,
                    active_input_ids: HashSet::new(),
                    skipped_corrupt_files: ctx.skipped_corrupt_files.clone(),
                },
            );
        }

        let plan_state = self.plans.get_mut(&fingerprint).unwrap();
        let enqueue_input_sender = plan_state.enqueue_input_sender.clone();
        plan_state.active_input_ids.insert(input_id);

        Ok((
            fingerprint,
            async move {
                let (result_tx, result_rx) = result_channel(result_buffer_size);
                let enqueue_msg = EnqueueInputMessage {
                    input_id,
                    inputs,
                    result_sender: result_tx,
                };
                if enqueue_input_sender.send(enqueue_msg).await.is_err() {
                    return Err(common_error::DaftError::InternalError(
                        "Plan execution task has died; cannot enqueue new input".to_string(),
                    ));
                }

                // Send the event after the task has been enqueued for execution
                if let Some((event, subscribers)) = task_start_dispatch {
                    dispatch_task_start_event(&subscribers, &event);
                }

                Ok(ExecutionEngineResult {
                    receiver: result_rx,
                })
            }
            .boxed(),
        ))
    }

    /// Finish tracking an input_id. If no active input_ids remain (or the
    /// enqueue channel is closed), removes the plan and awaits the exec task.
    pub fn try_finish(
        &mut self,
        fingerprint: u64,
        input_id: InputId,
    ) -> DaftResult<BoxFuture<'static, DaftResult<ExecutionStats>>> {
        let Some(plan_state) = self.plans.get_mut(&fingerprint) else {
            // Plan already removed (pipeline died and another input_id cleaned it up).
            // Return empty stats; the actual error was already surfaced by the first caller.
            let query_id = QueryID::from("");
            return Ok(async move { Ok(ExecutionStats::new(query_id, vec![])) }.boxed());
        };

        plan_state.active_input_ids.remove(&input_id);
        let pipeline_dead = plan_state.enqueue_input_sender.is_closed();
        let should_remove = plan_state.active_input_ids.is_empty() || pipeline_dead;

        if should_remove {
            let plan_state = self.plans.remove(&fingerprint).unwrap();
            Ok(async move {
                // Try to get stats for this input_id. If the pipeline already died,
                // the stats manager may be finished so this can fail — that's OK.
                let stats = plan_state.stats_handle.take_input_snapshot(input_id).await;
                drop(plan_state.enqueue_input_sender);
                plan_state.task_handle.await??;
                let skipped = plan_state
                    .skipped_corrupt_files
                    .lock()
                    .map(|v| v.clone())
                    .unwrap_or_default();
                // If the snapshot failed (e.g. pipeline died), return empty stats.
                Ok(stats
                    .unwrap_or_else(|_| ExecutionStats::new(QueryID::from(""), vec![]))
                    .with_skipped_corrupt_files(skipped))
            }
            .boxed())
        } else {
            let stats_handle = plan_state.stats_handle.clone();
            let skipped_corrupt_files = plan_state.skipped_corrupt_files.clone();
            Ok(async move {
                let skipped = skipped_corrupt_files
                    .lock()
                    .map(|v| v.clone())
                    .unwrap_or_default();
                Ok(stats_handle
                    .take_input_snapshot(input_id)
                    .await
                    .unwrap_or_else(|_| ExecutionStats::new(QueryID::from(""), vec![]))
                    .with_skipped_corrupt_files(skipped))
            }
            .boxed())
        }
    }

    pub fn cancel_plan(&mut self, fingerprint: u64) {
        // RuntimeTask drop cancels the spawned task
        self.plans.remove(&fingerprint);
    }

    fn repr_ascii(
        logical_plan_builder: &LogicalPlanBuilder,
        cfg: Arc<DaftExecutionConfig>,
        simple: bool,
    ) -> String {
        let logical_plan = logical_plan_builder.build();
        let (physical_plan, _) = translate(&logical_plan, &HashMap::new()).unwrap();
        let ctx = BuilderContext::new();
        let (pipeline_node, _) =
            translate_physical_plan_to_pipeline(&physical_plan, &cfg, &ctx).unwrap();

        viz_pipeline_ascii(pipeline_node.as_ref(), simple)
    }

    fn repr_mermaid(
        logical_plan_builder: &LogicalPlanBuilder,
        cfg: Arc<DaftExecutionConfig>,
        options: MermaidDisplayOptions,
    ) -> String {
        let logical_plan = logical_plan_builder.build();
        let (physical_plan, _) = translate(&logical_plan, &HashMap::new()).unwrap();
        let ctx = BuilderContext::new();
        let (pipeline_node, _) =
            translate_physical_plan_to_pipeline(&physical_plan, &cfg, &ctx).unwrap();

        let display_type = if options.simple {
            DisplayLevel::Compact
        } else {
            DisplayLevel::Default
        };
        viz_pipeline_mermaid(
            pipeline_node.as_ref(),
            display_type,
            options.bottom_up,
            options.subgraph_options,
        )
    }
}

impl Drop for NativeExecutor {
    fn drop(&mut self) {
        self.cancel.cancel();
        if let Some(conn) = &mut self.shuffle_server_connection {
            let _ = conn.shutdown();
        }
    }
}

pub struct ExecutionEngineResult {
    receiver: ResultReceiver,
}

impl ExecutionEngineResult {
    async fn next(&mut self) -> Option<ExecutionEngineResultItem> {
        self.receiver.recv().await
    }

    /// Receives the next completed output partition, or `None` at end of
    /// stream. Shuffle partition references are skipped; they are only
    /// produced when running as a distributed worker. Failures are surfaced
    /// by `try_finish` after the stream ends.
    pub async fn next_partition(&mut self) -> Option<MicroPartition> {
        while let Some(item) = self.receiver.recv().await {
            if let ExecutionEngineResultItem::Partition(p) = item {
                return Some(p);
            }
        }
        None
    }

    /// Consume all pipeline output for this input_id until EOF, returning any
    /// emitted `MicroPartition`s. `FlightPartitionRef` items are skipped (they
    /// are only relevant when shuffles are enabled). Intended for tests that
    /// exercise `NativeExecutor` end-to-end and need the pipeline to finish
    /// producing output before `try_finish` is called — mirroring what the
    /// production Python `__anext__` loop does.
    pub async fn collect_partitions_for_testing(mut self) -> Vec<MicroPartition> {
        let mut out = Vec::new();
        while let Some(item) = self.receiver.recv().await {
            if let ExecutionEngineResultItem::Partition(p) = item {
                out.push(p);
            }
        }
        out
    }
}

#[cfg_attr(
    feature = "python",
    pyclass(module = "daft.daft", name = "PyResultReceiver", frozen)
)]
pub struct PyResultReceiver {
    result: Arc<tokio::sync::Mutex<Option<ExecutionEngineResult>>>,
    fingerprint: u64,
    input_id: InputId,
    executor: Arc<Mutex<NativeExecutor>>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyResultReceiver {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, pyo3::PyAny>> {
        let result = self.result.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut result = result.lock().await;
            let part = result
                .as_mut()
                .expect("PyResultReceiver.__anext__() should not be called after try_finish().")
                .next()
                .await;
            Python::attach(|py| {
                Ok(match part {
                    None => py.None(),
                    Some(ExecutionEngineResultItem::Partition(partition)) => {
                        PyMicroPartition::from(partition)
                            .into_pyobject(py)?
                            .unbind()
                            .into_any()
                    }
                    Some(ExecutionEngineResultItem::FlightPartitionRef(partition_ref)) => {
                        PyFlightPartitionRef::from(partition_ref)
                            .into_pyobject(py)?
                            .unbind()
                            .into_any()
                    }
                })
            })
        })
    }

    fn try_finish<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let result = self.result.clone();
        let executor = self.executor.clone();
        let fingerprint = self.fingerprint;
        let input_id = self.input_id;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // Take the result to drop the receiver
            let mut result = result.lock().await;
            let _ = result
                .take()
                .expect("PyResultReceiver.try_finish() should not be called more than once.");
            drop(result);

            // Delegate to NativeExecutor::try_finish
            let finish_future = executor.lock().unwrap().try_finish(fingerprint, input_id)?;
            let stats = finish_future.await?;
            Ok(PyExecutionStats::from(stats))
        })
    }
}

fn dispatch_task_start_event(subscribers: &[Arc<dyn Subscriber>], event: &Event) {
    for subscriber in subscribers {
        if let Err(e) = subscriber.on_event(event.clone()) {
            log::debug!("Failed to dispatch task start event: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use daft_micropartition::MicroPartition;
    use futures::poll;

    use super::{ExecutionEngineResultItem, MessageRouter, ResultSender};
    use crate::{
        channel::{create_channel, create_unbounded_channel},
        pipeline::PipelineMessage,
    };

    fn morsel() -> PipelineMessage {
        PipelineMessage::Morsel {
            input_id: 0,
            partition: MicroPartition::empty(None),
        }
    }

    #[tokio::test]
    async fn a_result_waits_for_room_in_its_consumers_buffer() {
        let (tx, mut rx) = create_channel::<ExecutionEngineResultItem>(1);
        let mut router = MessageRouter::new();
        router.insert_output_sender(0, ResultSender::Bounded(tx));

        router.route_message(morsel()).expect("a delivery").await;
        let mut second = router.route_message(morsel()).expect("a delivery");

        assert!(poll!(&mut second).is_pending());
        assert!(rx.recv().await.is_some());
        second.await;
        assert!(rx.recv().await.is_some());
    }

    #[tokio::test]
    async fn a_result_for_a_departed_consumer_is_dropped() {
        let (tx, rx) = create_channel::<ExecutionEngineResultItem>(1);
        let mut router = MessageRouter::new();
        router.insert_output_sender(0, ResultSender::Bounded(tx));
        drop(rx);

        router.route_message(morsel()).expect("a delivery").await;
        router.route_message(morsel()).expect("a delivery").await;
    }

    #[tokio::test]
    async fn nothing_is_delivered_for_an_input_after_its_flush() {
        let (tx, _rx) = create_channel::<ExecutionEngineResultItem>(1);
        let mut router = MessageRouter::new();
        router.insert_output_sender(0, ResultSender::Bounded(tx));

        assert!(router.route_message(PipelineMessage::Flush(0)).is_none());
        assert!(router.route_message(morsel()).is_none());
    }

    #[tokio::test]
    async fn an_unbounded_result_is_delivered_at_once() {
        let (tx, mut rx) = create_unbounded_channel::<ExecutionEngineResultItem>();
        let mut router = MessageRouter::new();
        router.insert_output_sender(0, ResultSender::Unbounded(tx));

        assert!(router.route_message(morsel()).is_none());
        assert!(router.route_message(morsel()).is_none());
        assert!(rx.recv().await.is_some());
        assert!(rx.recv().await.is_some());
    }
}
