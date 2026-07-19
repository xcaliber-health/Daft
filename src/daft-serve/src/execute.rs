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
    wire::{NamedPartitionSet, QueryPayload, QueryRequest, QueryStatsWire},
};

/// Reconstructs the logical plan builder from a request payload.
///
/// Plan payloads decode directly; textual payloads are planned against the
/// server session (which owns catalog attachments) under the interpreter
/// lock.
///
/// # Errors
/// Returns a malformed-payload error if a plan blob does not decode, or an
/// execution error if textual planning fails.
pub fn payload_to_builder(
    payload: &QueryPayload,
    sql_session: Option<&Arc<Py<PyAny>>>,
) -> ServeResult<LogicalPlanBuilder> {
    match payload {
        QueryPayload::Plan(bytes) => {
            let plan: Arc<LogicalPlan> = crate::wire::decode(bytes)?;
            Ok(LogicalPlanBuilder::new(plan, None))
        }
        QueryPayload::Sql(query) => Python::attach(|py| {
            let session = sql_session.ok_or_else(|| {
                ServeError::MalformedPayload(
                    "this server has no session configured for textual queries".to_string(),
                )
            })?;
            let dataframe = session
                .bind(py)
                .call_method1(intern!(py, "sql"), (query.as_str(),))
                .map_err(|e| ServeError::Execution(e.into()))?;
            let builder = dataframe
                .getattr(intern!(py, "_builder"))
                .and_then(|b| b.getattr(intern!(py, "_builder")))
                .and_then(|b| b.extract::<PyLogicalPlanBuilder>().map_err(Into::into))
                .map_err(|e: pyo3::PyErr| ServeError::Execution(e.into()))?;
            Ok(builder.builder)
        }),
    }
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

/// Collects the partition sets registered in this process's cache.
///
/// Textual queries planned server-side register literal values and other
/// in-memory data in the process-global cache rather than shipping them, so
/// execution must see those sets in addition to any shipped with the
/// request.
///
/// # Errors
/// Returns an execution error if the cache cannot be read.
fn server_local_psets() -> ServeResult<HashMap<String, Vec<MicroPartitionRef>>> {
    use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods};

    Python::attach(
        |py| -> pyo3::PyResult<HashMap<String, Vec<MicroPartitionRef>>> {
            let module = py.import(intern!(py, "daft.runners.runner"))?;
            let cache = module.getattr(intern!(py, "LOCAL_PARTITION_SET_CACHE"))?;
            let all = cache.call_method0(intern!(py, "get_all_partition_sets"))?;
            let dict = all.cast_into::<PyDict>()?;
            let mut out = HashMap::with_capacity(dict.len());
            for (key, pset) in &dict {
                let key: String = key.extract()?;
                let values = pset.call_method0(intern!(py, "values"))?;
                let mut parts = Vec::new();
                for item in values.try_iter()? {
                    let part = item?
                        .call_method0(intern!(py, "micropartition"))?
                        .getattr(intern!(py, "_micropartition"))?
                        .extract::<daft_micropartition::python::PyMicroPartition>()?;
                    parts.push(part.inner);
                }
                out.insert(key, parts);
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

/// Runs one admitted query, sending encoded messages into `tx`.
///
/// The admission permit and registry guard are held for the duration of the
/// send loop and released on return, whether the query completes, fails, or
/// is cancelled.
pub async fn run_query(
    request: QueryRequest,
    sql_session: Option<Arc<Py<PyAny>>>,
    cancel: CancellationToken,
    permit: AdmissionPermit,
    guard: QueryGuard,
    tx: async_channel::Sender<Result<FlightData, Status>>,
) {
    let query_id = request.query_id.clone();
    let result = run_query_inner(request, sql_session, &cancel, &tx).await;
    drop(guard);
    drop(permit);
    if let Err(err) = result {
        let err = if cancel.is_cancelled() {
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
    sql_session: Option<Arc<Py<PyAny>>>,
    cancel: &CancellationToken,
    tx: &async_channel::Sender<Result<FlightData, Status>>,
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
    // Plan building, optimization (including scan planning against remote
    // catalogs), and translation are blocking and may call into the
    // interpreter; keep them off the async runtime threads.
    let setup = {
        let exec_config = exec_config.clone();
        tokio::task::spawn_blocking(move || -> ServeResult<_> {
            let builder = payload_to_builder(&payload, sql_session.as_ref())?;
            let optimized = builder.optimize(exec_config)?;
            let schema = optimized.schema();
            let mut all_psets = server_local_psets()?;
            all_psets.extend(decode_psets(&psets)?);
            let optimized_plan = optimized.build();
            // A referenced set missing from both the request and the local
            // cache would leave the pipeline waiting for input forever;
            // fail fast instead.
            for key in crate::client::referenced_pset_keys(&optimized_plan) {
                if !all_psets.contains_key(&key) {
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

    let mut encoder = BatchEncoder::try_new(&schema)?;
    let schema_message = encoder.schema_message(&schema)?;
    if tx.send(Ok(schema_message)).await.is_err() {
        // Client disconnected before the stream started.
        return Ok(());
    }

    let ctx = daft_context::get_context();
    let subscribers = ctx.subscribers();
    let mut executor = NativeExecutor::new(false, "");
    let context = HashMap::from([("query_id".to_string(), query_id.clone())]);
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
    let mut rows: u64 = 0;
    let mut bytes: u64 = 0;
    loop {
        let partition = tokio::select! {
            () = cancel.cancelled() => {
                executor.cancel_plan(fingerprint);
                return Err(ServeError::Cancelled(query_id));
            }
            partition = results.next_partition() => partition,
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
