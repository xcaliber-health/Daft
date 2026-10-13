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
//! cancel action) and the receiver side of the output channel (dropped when
//! the client disconnects) both abort execution promptly, dropping the
//! engine instance which cancels its running pipeline.

use std::{collections::HashMap, sync::Arc};

use arrow_flight::FlightData;
use common_daft_config::DaftExecutionConfig;
use daft_local_execution::NativeExecutor;
use daft_local_plan::translate;
use daft_logical_plan::{LogicalPlan, LogicalPlanBuilder, PyLogicalPlanBuilder};
use daft_micropartition::MicroPartitionRef;
use daft_schema::schema::Schema;
use pyo3::{intern, prelude::*};
use tokio_util::sync::CancellationToken;
use tonic::Status;

use crate::{
    admission::AdmissionPermit,
    codec::{self, BatchEncoder},
    error::{ServeError, ServeResult},
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
                .map_err(|e| ServeError::Execution(e.into()))?;
            frame_builder(&dataframe)
        }),
    }
}

/// Plans and optimizes a request off the async runtime threads.
///
/// Planning may call into the interpreter and, for textual or host requests,
/// read catalogs over the network, so it runs on a blocking thread.
async fn plan_off_runtime(
    payload: QueryPayload,
    planners: Planners,
    origin: RequestOrigin,
    exec_config: Arc<DaftExecutionConfig>,
) -> ServeResult<LogicalPlanBuilder> {
    tokio::task::spawn_blocking(move || -> ServeResult<_> {
        let builder = payload_to_builder(&payload, &planners, &origin)?;
        Ok(builder.optimize(exec_config)?)
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
) -> ServeResult<ExplainResult> {
    let exec_config = resolve_exec_config(&request)?;
    let query_id = origin.query_id.clone();
    let planning = plan_off_runtime(request.payload, planners, origin, exec_config);
    let bounded = async {
        if timeout_secs > 0 {
            tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), planning)
                .await
                .map_err(|_| ServeError::Timeout {
                    query_id: query_id.clone(),
                    limit_secs: timeout_secs,
                })?
        } else {
            planning.await
        }
    };
    let optimized = tokio::select! {
        biased;
        () = cancel.cancelled() => return Err(ServeError::Cancelled(query_id)),
        optimized = bounded => optimized?,
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

/// What an admitted query holds until it finishes: its execution slot and its
/// registry entry.
#[derive(Debug)]
pub struct QueryHold {
    /// The execution slot; dropping it frees the slot.
    pub permit: AdmissionPermit,
    /// The registry entry; dropping it deregisters the query.
    pub guard: QueryGuard,
}

/// Runs one admitted query, sending encoded messages into `tx`.
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
    tx: async_channel::Sender<Result<FlightData, Status>>,
    bounds: QueryBounds,
) {
    let query_id = request.query_id.clone();
    let inner = run_query_inner(request, planners, caller, &cancel, &tx, bounds);
    let result = if bounds.timeout_secs > 0 {
        match tokio::time::timeout(std::time::Duration::from_secs(bounds.timeout_secs), inner).await
        {
            Ok(result) => result,
            Err(_) => Err(ServeError::Timeout {
                query_id: query_id.clone(),
                limit_secs: bounds.timeout_secs,
            }),
        }
    } else {
        inner.await
    };
    drop(hold);
    if let Err(err) = result {
        let err = if cancel.is_cancelled() && !matches!(err, ServeError::Timeout { .. }) {
            ServeError::Cancelled(query_id)
        } else {
            err
        };
        // A send failure means the client is gone; nothing left to notify.
        let _ = tx.send(Err(err.into())).await;
    }
}

async fn run_query_inner(
    request: QueryRequest,
    planners: Planners,
    caller: Option<String>,
    cancel: &CancellationToken,
    tx: &async_channel::Sender<Result<FlightData, Status>>,
    bounds: QueryBounds,
) -> ServeResult<()> {
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
    let optimized = plan_off_runtime(payload, planners, origin, exec_config.clone()).await?;
    // Translation reads in-memory data and may call into the interpreter;
    // keep it off the async runtime threads too.
    let setup = {
        tokio::task::spawn_blocking(move || -> ServeResult<_> {
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
    if tx.send(Ok(schema_message)).await.is_err() {
        // Client disconnected before the stream started.
        executor.cancel_plan(fingerprint);
        return Ok(());
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
                if tx.send(Ok(message)).await.is_err() {
                    // Client disconnected mid-stream; abort execution.
                    executor.cancel_plan(fingerprint);
                    return Ok(());
                }
            }
        }
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
    let _ = tx.send(Ok(encoder.stats_message(&stats_wire)?)).await;
    Ok(())
}
