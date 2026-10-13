//! Server-side query execution: request body to streaming result.
//!
//! A request is turned into a logical plan (deserialized directly, or planned
//! from text against the server session), optimized *on the server* so scan
//! planning and pruning run next to the data, translated to a local physical
//! plan, and driven through the streaming execution engine. Results are
//! encoded into wire messages as they are produced; a metadata-only terminal
//! message carries execution statistics.
//!
//! Cancellation is cooperative: the per-query token (tripped by an explicit
//! cancel action, the time limit, or the server's drain) and the receiver side
//! of the output channel (dropped when the client disconnects) both abort
//! execution promptly, dropping the engine instance which cancels its running
//! pipeline. A query waiting for its client to read is still cancelled at
//! once, and a query that ends unfinished discards the rows its client has
//! not read, so the client's next read raises why it ended. A schema not yet
//! read is kept: a planned query hands over its schema before any failure.

use std::{
    collections::HashMap,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};

use arrow_flight::FlightData;
use common_daft_config::DaftExecutionConfig;
use daft_local_execution::NativeExecutor;
use daft_local_plan::translate;
use daft_logical_plan::{LogicalPlan, LogicalPlanBuilder, PyLogicalPlanBuilder};
use daft_micropartition::MicroPartitionRef;
use daft_schema::schema::Schema;
use futures::{FutureExt, Stream, channel::oneshot};
use pyo3::{intern, prelude::*};
use tokio_util::sync::CancellationToken;
use tonic::Status;

use crate::{
    admission::AdmissionPermit,
    codec::{self, BatchEncoder},
    error::{ServeError, ServeResult},
    lifecycle::{PlanningEvents, QueryLifecycle},
    registry::QueryGuard,
    wire::{ExplainResult, NamedPartitionSet, QueryPayload, QueryRequest, QueryStatsWire},
};

/// What turns non-plan payloads into plans on this server: the session that
/// plans textual queries, and the host that plans requests of its kinds.
///
/// Either may be absent, which closes its lane.
#[derive(Debug, Clone, Default)]
pub struct Planners {
    /// Session planning textual queries against the server's catalogs.
    pub session: Option<Arc<Py<PyAny>>>,
    /// Registered host planning requests of the kinds it declares.
    pub host: Option<Arc<Py<PyAny>>>,
}

/// Where an admitted query's encoded results go, for its client to read at
/// its own pace.
///
/// Holds the sending half of the bounded results channel, a weak handle on
/// its receiving half, which never keeps the channel open by itself, and the
/// slot for why the query ended, which its client reads after the results.
pub struct ResultSink {
    tx: async_channel::Sender<Result<FlightData, Status>>,
    unread: async_channel::WeakReceiver<Result<FlightData, Status>>,
    ending: oneshot::Sender<Status>,
}

/// What a query's client reads: its results, then why it ended, if it ended
/// unfinished.
pub struct ResultStream {
    results: Pin<Box<async_channel::Receiver<Result<FlightData, Status>>>>,
    ending: Option<oneshot::Receiver<Status>>,
}

impl ResultSink {
    /// Opens a sink holding up to `buffer` results its client has not read,
    /// and the stream its client reads them from.
    #[must_use]
    pub fn open(buffer: usize) -> (Self, ResultStream) {
        let (tx, results) = async_channel::bounded(buffer);
        let (ending, ended) = oneshot::channel();
        let sink = Self {
            tx,
            unread: results.downgrade(),
            ending,
        };
        let stream = ResultStream {
            results: Box::pin(results),
            ending: Some(ended),
        };
        (sink, stream)
    }

    /// Hands `message` to the client, waiting for room while it is not reading.
    ///
    /// Returns `false` without sending once `cancel` trips or the client is
    /// gone, so a cancelled query never waits on its client.
    async fn deliver(&self, message: FlightData, cancel: &CancellationToken) -> bool {
        tokio::select! {
            biased;
            () = cancel.cancelled() => false,
            sent = self.tx.send(Ok(message)) => sent.is_ok(),
        }
    }

    /// Ends the client's stream with `status`, discarding the results it has
    /// not read, so its next read raises why the query ended.
    ///
    /// The schema, when the client has not read it yet, is kept: a query that
    /// was planned hands over its schema before any failure.
    fn end(self, status: Status) {
        if let Some(unread) = self.unread.upgrade() {
            let schema = unread.try_recv().ok().filter(is_schema);
            while unread.try_recv().is_ok() {}
            if let Some(schema) = schema {
                // Only the query sends, and the channel was just emptied, so
                // there is room unless the client is gone.
                let _ = self.tx.try_send(schema);
            }
        }
        // Read once the results are, whatever room the channel has; a client
        // that is gone has no one to tell.
        let _ = self.ending.send(status);
    }
}

/// Whether `message` is a result's schema rather than a part of its rows.
fn is_schema(message: &Result<FlightData, Status>) -> bool {
    message.as_ref().is_ok_and(|data| {
        arrow_ipc::root_as_message(&data.data_header)
            .is_ok_and(|header| header.header_type() == arrow_ipc::MessageHeader::Schema)
    })
}

impl Stream for ResultStream {
    type Item = Result<FlightData, Status>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if let Some(result) = ready!(this.results.as_mut().poll_next(cx)) {
            return Poll::Ready(Some(result));
        }
        // Every result is read and the query is over: say why it ended, if
        // it ended unfinished.
        let Some(ending) = this.ending.as_mut() else {
            return Poll::Ready(None);
        };
        let ended = ready!(ending.poll_unpin(cx));
        this.ending = None;
        Poll::Ready(ended.ok().map(Err))
    }
}

/// Who sent a request, and under which id: handed to the host with it.
#[derive(Debug, Clone)]
pub struct RequestOrigin {
    /// Client-chosen query id.
    pub query_id: String,
    /// Authenticated caller: the tenant name, or `None` when the server does
    /// not identify callers.
    pub caller: Option<String>,
}

/// Extracts the logical plan builder from a frame returned to the server.
fn frame_builder(frame: &Bound<'_, PyAny>) -> ServeResult<LogicalPlanBuilder> {
    let py = frame.py();
    frame
        .getattr(intern!(py, "_builder"))
        .and_then(|b| b.getattr(intern!(py, "_builder")))
        .and_then(|b| b.extract::<PyLogicalPlanBuilder>().map_err(Into::into))
        .map(|builder| builder.builder)
        .map_err(|e: PyErr| ServeError::Execution(e.into()))
}

/// Reconstructs the logical plan builder from a request payload.
///
/// Plan payloads decode directly; textual payloads are planned against the
/// server session (which owns catalog attachments), and host requests by the
/// registered host, both under the interpreter lock. The host is handed the
/// request together with its query id and authenticated caller.
///
/// # Errors
/// Returns a malformed-payload error if a plan blob does not decode or the
/// payload's lane is closed on this server, or an execution error if
/// planning fails.
pub fn payload_to_builder(
    payload: &QueryPayload,
    planners: &Planners,
    origin: &RequestOrigin,
) -> ServeResult<LogicalPlanBuilder> {
    match payload {
        QueryPayload::Plan(bytes) => {
            let plan: Arc<LogicalPlan> = crate::wire::decode(bytes)?;
            Ok(LogicalPlanBuilder::new(plan, None))
        }
        QueryPayload::Sql(query) => Python::attach(|py| {
            let session = planners.session.as_ref().ok_or_else(|| {
                ServeError::MalformedPayload(
                    "this server has no session configured for textual queries".to_string(),
                )
            })?;
            let dataframe = session
                .bind(py)
                .call_method1(intern!(py, "sql"), (query.as_str(),))
                .map_err(|e| ServeError::Execution(e.into()))?;
            frame_builder(&dataframe)
        }),
        QueryPayload::Request { kind, body } => Python::attach(|py| {
            let host = planners.host.as_ref().ok_or_else(|| {
                ServeError::MalformedPayload(format!(
                    "this server has no host for request kind `{kind}`"
                ))
            })?;
            let served = py
                .import(intern!(py, "daft.serve"))
                .and_then(|module| module.getattr(intern!(py, "ServedRequest")))
                .and_then(|served_request| {
                    served_request.call1((
                        kind.as_str(),
                        pyo3::types::PyBytes::new(py, body),
                        origin.query_id.as_str(),
                        origin.caller.as_deref(),
                    ))
                })
                .map_err(|e| ServeError::Execution(e.into()))?;
            let dataframe = host
                .bind(py)
                .call_method1(intern!(py, "plan"), (served,))
                .map_err(|e| crate::failure::of_py_err(py, &e, crate::failure::Raised::ByHost))?;
            frame_builder(&dataframe)
        }),
    }
}

/// Plans and optimizes a request off the async runtime threads.
///
/// Planning may call into the interpreter and, for textual or host requests,
/// read catalogs over the network, so it runs on a blocking thread. It holds
/// `slot` until it returns: planning cannot be interrupted, so a query that
/// ends while it plans, by its time limit, keeps its slot taken until the
/// planning it started is done. Planning that returns after `cancel` trips
/// stops there, without optimizing or reporting.
///
/// `planning` reports the query's start and optimization as they happen; an
/// explanation passes `None`, since it starts no query.
async fn plan_off_runtime(
    payload: QueryPayload,
    planners: Planners,
    origin: RequestOrigin,
    exec_config: Arc<DaftExecutionConfig>,
    planning: Option<PlanningEvents>,
    slot: AdmissionPermit,
    cancel: CancellationToken,
) -> ServeResult<LogicalPlanBuilder> {
    tokio::task::spawn_blocking(move || -> ServeResult<_> {
        let _slot = slot;
        let builder = payload_to_builder(&payload, &planners, &origin)?;
        if cancel.is_cancelled() {
            return Err(ServeError::Cancelled(origin.query_id));
        }
        if let Some(events) = &planning {
            events.started(&builder);
        }
        let optimized = builder.optimize(exec_config)?;
        if let Some(events) = &planning {
            events.optimized(&optimized);
        }
        Ok(optimized)
    })
    .await
    .map_err(|e| {
        ServeError::Execution(common_error::DaftError::InternalError(format!(
            "query planning task failed: {e}"
        )))
    })?
}

/// Renders the server-side optimized plan for an admitted request without
/// executing it.
///
/// Planning runs off the async runtime threads, bounded by `timeout_secs`
/// (`0` leaves it unbounded) and abandoned if `cancel` trips.
///
/// # Errors
/// Returns the planning failure, [`ServeError::Timeout`] if planning exceeds
/// its limit, or [`ServeError::Cancelled`] if the request is cancelled.
pub async fn explain(
    request: QueryRequest,
    planners: Planners,
    origin: RequestOrigin,
    cancel: &CancellationToken,
    timeout_secs: u64,
    slot: AdmissionPermit,
) -> ServeResult<ExplainResult> {
    let exec_config = resolve_exec_config(&request)?;
    let query_id = origin.query_id.clone();
    let planning = plan_off_runtime(
        request.payload,
        planners,
        origin,
        exec_config,
        None,
        slot,
        cancel.clone(),
    );
    let bounded = async {
        if timeout_secs > 0 {
            tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), planning)
                .await
                .map_err(|_| {
                    // Planning still running stops when it returns; it keeps the slot until then.
                    cancel.cancel();
                    ServeError::Timeout {
                        query_id: query_id.clone(),
                        limit_secs: timeout_secs,
                    }
                })?
        } else {
            planning.await
        }
    };
    let optimized = tokio::select! {
        biased;
        () = cancel.cancelled() => return Err(ServeError::Cancelled(query_id)),
        optimized = bounded => optimized.map_err(crate::failure::classify)?,
    };
    Ok(ExplainResult {
        optimized_plan: optimized.build().repr_ascii(false),
    })
}

/// Decodes shipped in-memory partition sets into engine partitions keyed by
/// cache key.
///
/// # Errors
/// Returns a malformed-payload error if a schema or partition blob does not
/// decode.
pub fn decode_psets(
    psets: &[NamedPartitionSet],
) -> ServeResult<HashMap<String, Vec<MicroPartitionRef>>> {
    let mut out = HashMap::with_capacity(psets.len());
    for pset in psets {
        if pset.partitions.is_empty() {
            out.insert(pset.key.clone(), Vec::new());
            continue;
        }
        let schema: Arc<Schema> = crate::wire::decode(&pset.schema)?;
        let mut partitions = Vec::with_capacity(pset.partitions.len());
        for blob in &pset.partitions {
            partitions.push(Arc::new(codec::ipc_to_micropartition(&schema, blob)?));
        }
        out.insert(pset.key.clone(), partitions);
    }
    Ok(out)
}

/// Collects the requested partition sets from this process's cache.
///
/// Textual queries planned server-side register literal values and other
/// in-memory data in the process-global cache rather than shipping them, so
/// execution must see those sets in addition to any shipped with the
/// request. Only the named keys are read and converted: fetching the whole
/// cache would leak unrelated (possibly other clients') data into the query
/// and cost time proportional to everything cached rather than everything
/// referenced. Keys absent from the cache are skipped; the caller decides
/// whether a missing set is fatal.
///
/// # Errors
/// Returns an execution error if the cache cannot be read.
fn server_local_psets_for(keys: &[String]) -> ServeResult<HashMap<String, Vec<MicroPartitionRef>>> {
    use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods};

    if keys.is_empty() {
        return Ok(HashMap::new());
    }
    Python::attach(
        |py| -> pyo3::PyResult<HashMap<String, Vec<MicroPartitionRef>>> {
            let module = py.import(intern!(py, "daft.runners.runner"))?;
            let cache = module.getattr(intern!(py, "LOCAL_PARTITION_SET_CACHE"))?;
            let all = cache.call_method0(intern!(py, "get_all_partition_sets"))?;
            let dict = all.cast_into::<PyDict>()?;
            let mut out = HashMap::with_capacity(keys.len());
            for key in keys {
                let Some(pset) = dict.get_item(key.as_str())? else {
                    continue;
                };
                let values = pset.call_method0(intern!(py, "values"))?;
                let mut parts = Vec::new();
                for item in values.try_iter()? {
                    let part = item?
                        .call_method0(intern!(py, "micropartition"))?
                        .getattr(intern!(py, "_micropartition"))?
                        .extract::<daft_micropartition::python::PyMicroPartition>()?;
                    parts.push(part.inner);
                }
                out.insert(key.clone(), parts);
            }
            Ok(out)
        },
    )
    .map_err(|e| ServeError::Execution(e.into()))
}

/// Resolves the effective execution configuration for a request: the
/// client-shipped configuration when present, otherwise the server context
/// default.
///
/// # Errors
/// Returns a malformed-payload error if a shipped configuration does not
/// decode.
pub fn resolve_exec_config(request: &QueryRequest) -> ServeResult<Arc<DaftExecutionConfig>> {
    match &request.exec_config {
        Some(bytes) => Ok(Arc::new(crate::wire::decode(bytes)?)),
        None => Ok(daft_context::get_context().execution_config()),
    }
}

/// Bounds one query's execution: wall-clock time and memory.
///
/// A zero `timeout_secs` leaves execution unbounded in time; `None` for the
/// memory cap leaves it to the shared pool.
#[derive(Debug, Clone, Copy)]
pub struct QueryBounds {
    pub timeout_secs: u64,
    pub memory_cap_bytes: Option<u64>,
    /// Results the engine may hold for the response stream before it pauses.
    pub result_buffer_size: usize,
}

/// What an admitted query holds until it finishes: its execution slot, its
/// registry entry, and its lifecycle reporting.
#[derive(Debug)]
pub struct QueryHold {
    /// The execution slot; dropping it frees the slot.
    pub permit: AdmissionPermit,
    /// The registry entry; dropping it deregisters the query.
    pub guard: QueryGuard,
    /// The query's lifecycle events; its end is reported when the query ends.
    pub lifecycle: QueryLifecycle,
}

/// Runs one admitted query, sending encoded messages into `sink`.
///
/// The query's hold (its execution slot and registry entry) is kept for the
/// duration of the send loop and released on return, whether the query completes, fails,
/// times out, or is cancelled. A nonzero timeout in `bounds` limits execution
/// wall-clock time: on expiry the execution future is dropped, which cancels
/// the running pipeline and frees the admission slot.
pub async fn run_query(
    request: QueryRequest,
    planners: Planners,
    caller: Option<String>,
    cancel: CancellationToken,
    hold: QueryHold,
    sink: ResultSink,
    bounds: QueryBounds,
) {
    let QueryHold {
        permit,
        guard,
        mut lifecycle,
    } = hold;
    let query_id = request.query_id.clone();
    let inner = run_query_inner(
        request,
        planners,
        caller,
        bounds,
        RunningQuery {
            cancel: &cancel,
            sink: &sink,
            lifecycle: &mut lifecycle,
            slot: &permit,
        },
    );
    let result = if bounds.timeout_secs > 0 {
        match tokio::time::timeout(std::time::Duration::from_secs(bounds.timeout_secs), inner).await
        {
            Ok(result) => result,
            Err(_) => {
                // Work still running for the query stops at its next chance; work that cannot
                // be interrupted keeps its share of the slot until it returns.
                cancel.cancel();
                Err(ServeError::Timeout {
                    query_id: query_id.clone(),
                    limit_secs: bounds.timeout_secs,
                })
            }
        }
    } else {
        inner.await
    };
    drop(guard);
    drop(permit);
    let result = result.map_err(|err| {
        if cancel.is_cancelled() && !matches!(err, ServeError::Timeout { .. }) {
            ServeError::Cancelled(query_id)
        } else {
            crate::failure::classify(err)
        }
    });
    lifecycle.finish(&result);
    if let Err(err) = result {
        sink.end(err.into());
    }
}

/// What an admitted query's work shares while it runs: its cancellation, the
/// sink of its results, its lifecycle reporting, and its slot, a share of which
/// each piece of its work holds until that work returns.
struct RunningQuery<'a> {
    cancel: &'a CancellationToken,
    sink: &'a ResultSink,
    lifecycle: &'a mut QueryLifecycle,
    slot: &'a AdmissionPermit,
}

async fn run_query_inner(
    request: QueryRequest,
    planners: Planners,
    caller: Option<String>,
    bounds: QueryBounds,
    running: RunningQuery<'_>,
) -> ServeResult<()> {
    let RunningQuery {
        cancel,
        sink,
        lifecycle,
        slot,
    } = running;
    let exec_config = resolve_exec_config(&request)?;
    let maintain_order = exec_config.maintain_order;
    let QueryRequest {
        query_id,
        payload,
        psets,
        ..
    } = request;

    log::debug!("serve query {query_id}: setup starting");
    let origin = RequestOrigin {
        query_id: query_id.clone(),
        caller,
    };
    let optimized = plan_off_runtime(
        payload,
        planners,
        origin,
        exec_config.clone(),
        lifecycle.planning(),
        slot.clone(),
        cancel.clone(),
    )
    .await?;
    // Translation reads in-memory data and may call into the interpreter;
    // keep it off the async runtime threads too.
    let setup = {
        let slot = slot.clone();
        tokio::task::spawn_blocking(move || -> ServeResult<_> {
            let _slot = slot;
            let schema = optimized.schema();
            let optimized_plan = optimized.build();
            // Shipped sets are decoded first (pure engine work, no
            // interpreter); only referenced keys they do not cover are then
            // read from the process cache, so unrelated cached data is never
            // touched and the interpreter is not entered at all in the common
            // file-scan case.
            let mut all_psets = decode_psets(&psets)?;
            let referenced = crate::client::referenced_pset_keys(&optimized_plan);
            let missing: Vec<String> = referenced
                .iter()
                .filter(|key| !all_psets.contains_key(key.as_str()))
                .cloned()
                .collect();
            all_psets.extend(server_local_psets_for(&missing)?);
            // A referenced set missing from both the request and the local
            // cache would leave the pipeline waiting for input forever;
            // fail fast instead.
            for key in &referenced {
                if !all_psets.contains_key(key) {
                    return Err(ServeError::MalformedPayload(format!(
                        "in-memory data `{key}` referenced by the plan was not \
                         shipped with the request and is not present on the server"
                    )));
                }
            }
            let (local_plan, inputs) = translate(&optimized_plan, &all_psets)?;
            Ok((schema, local_plan, inputs))
        })
        .await
        .map_err(|e| {
            ServeError::Execution(common_error::DaftError::InternalError(format!(
                "query setup task failed: {e}"
            )))
        })??
    };
    let (schema, local_plan, inputs) = setup;
    log::debug!("serve query {query_id}: setup complete");

    if cancel.is_cancelled() {
        return Err(ServeError::Cancelled(query_id));
    }

    let ctx = daft_context::get_context();
    let subscribers = ctx.subscribers();
    let mut executor = NativeExecutor::new(false, "");
    let mut context = HashMap::from([
        ("query_id".to_string(), query_id.clone()),
        // The engine pauses while this many results wait on a slow client, rather
        // than holding everything it produces until the client catches up.
        (
            "result_buffer_size".to_string(),
            bounds.result_buffer_size.to_string(),
        ),
    ]);
    if let Some(cap) = bounds.memory_cap_bytes {
        // Server-resolved ceiling; deliberately not part of the
        // client-shipped configuration so callers cannot lift their own
        // limits.
        context.insert("memory_cap_bytes".to_string(), cap.to_string());
    }
    let input_id = 0;
    let (fingerprint, enqueue_future) = executor.run(
        &local_plan,
        exec_config,
        subscribers,
        Some(context),
        inputs,
        input_id,
        maintain_order,
    )?;

    log::debug!("serve query {query_id}: execution started");
    let mut results = enqueue_future.await?;
    log::debug!("serve query {query_id}: inputs enqueued");

    // The stream's schema comes from the first produced partition rather
    // than the logical plan: operators with runtime-derived output columns
    // can order or type columns differently than planning predicted, and the
    // wire schema must describe the batches actually sent. The plan schema
    // is the fallback for empty results.
    let first = tokio::select! {
        () = cancel.cancelled() => {
            executor.cancel_plan(fingerprint);
            return Err(ServeError::Cancelled(query_id));
        }
        partition = results.next_partition() => partition,
    };
    let stream_schema = first
        .as_ref()
        .map_or(schema, |partition| partition.schema());
    let mut encoder = BatchEncoder::try_new(&stream_schema)?;
    let schema_message = encoder.schema_message(&stream_schema)?;
    if !sink.deliver(schema_message, cancel).await {
        // Cancelled, or the client disconnected, before the stream started.
        executor.cancel_plan(fingerprint);
        return Err(ServeError::Cancelled(query_id));
    }

    let mut rows: u64 = 0;
    let mut bytes: u64 = 0;
    let mut pending = first;
    loop {
        let partition = match pending.take() {
            Some(partition) => Some(partition),
            None => tokio::select! {
                () = cancel.cancelled() => {
                    executor.cancel_plan(fingerprint);
                    return Err(ServeError::Cancelled(query_id));
                }
                partition = results.next_partition() => partition,
            },
        };
        let Some(partition) = partition else {
            break;
        };
        rows += partition.len() as u64;
        for batch in partition.record_batches() {
            for message in encoder.encode(batch)? {
                bytes += message.data_body.len() as u64;
                if !sink.deliver(message, cancel).await {
                    // Cancelled, or the client disconnected, mid-stream.
                    executor.cancel_plan(fingerprint);
                    return Err(ServeError::Cancelled(query_id));
                }
            }
        }
        lifecycle.returned(partition.len() as u64, partition.size_bytes() as u64);
    }

    let stats = executor.try_finish(fingerprint, input_id)?.await?;
    let physical_plan_json = stats
        .query_plan
        .as_ref()
        .map(std::string::ToString::to_string);
    let stats_wire = QueryStatsWire {
        query_id,
        rows,
        bytes,
        physical_plan_json,
        stats: stats.encode(),
    };
    if !sink
        .deliver(encoder.stats_message(&stats_wire)?, cancel)
        .await
    {
        return Err(ServeError::Cancelled(stats_wire.query_id));
    }
    Ok(())
}
