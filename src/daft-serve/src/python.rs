//! Python bindings for the serving endpoint and its client.
//!
//! The server binding starts the transport on the shared IO runtime and
//! exposes lifecycle control (wait, graceful shutdown). The client binding
//! bridges the async protocol client into synchronous iteration, releasing
//! the interpreter lock around every blocking call so result decoding and
//! network waits never stall unrelated interpreter threads.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
    time::Duration,
};

use common_daft_config::PyDaftExecutionConfig;
use daft_local_plan::{ExecutionStats, python::PyExecutionStats};
use daft_logical_plan::PyLogicalPlanBuilder;
use daft_micropartition::python::PyMicroPartition;
use pyo3::{prelude::*, types::PyDict};

use crate::{
    auth::AuthPolicy,
    client::{QueryResultStream, ServeClient, referenced_pset_keys},
    codec,
    error::{Failure, ServeError},
    server::{DaftServeService, ServeConfig, ServeShutdownHandle, TenantConfig, start_server},
    wire::{self, NamedPartitionSet, QueryPayload, QueryRequest},
};

fn to_py_err(err: ServeError) -> PyErr {
    served_error(err, None)
}

/// Builds the exception a failure is raised as on this side.
///
/// An engine failure is raised as the engine class it names, a user-defined
/// function's failure as `UDFException`, and any other failure as
/// `DaftServeError`, each carrying the failure's fields and `query_id`. A
/// failure of this side's own engine is raised as it is in-process. No class
/// is ever imported by a name the server sent.
fn served_error(err: ServeError, query_id: Option<&str>) -> PyErr {
    Python::attach(|py| {
        let built = match err {
            ServeError::Execution(local) => Ok(PyErr::from(local)),
            ServeError::Failed { failure, message } => failure_error(py, &failure, message),
            other => failure_error(py, &other.failure(), other.to_string()),
        };
        let raised = built.unwrap_or_else(|building| building);
        if let Some(query_id) = query_id
            && let Err(setting) = raised.value(py).setattr("query_id", query_id)
        {
            return setting;
        }
        raised
    })
}

fn failure_error(py: Python<'_>, failure: &Failure, message: String) -> PyResult<PyErr> {
    if failure.engine {
        return Ok(common_error::python::engine_exception(
            &failure.error_class,
            message,
        ));
    }
    let attrs = PyDict::new(py);
    attrs.set_item("error_class", &failure.error_class)?;
    attrs.set_item("retryable", failure.retryable)?;
    if let Some(python) = &failure.python {
        attrs.set_item("remote_type", &python.remote_type)?;
        attrs.set_item("remote_module", &python.remote_module)?;
        attrs.set_item(
            "remote_types",
            pyo3::types::PyTuple::new(py, &python.remote_types)?,
        )?;
        let fields = PyDict::new(py);
        for (name, value) in &python.fields {
            fields.set_item(name, value)?;
        }
        attrs.set_item("fields", fields)?;
        if python.udf {
            let raised = py
                .import(pyo3::intern!(py, "daft.errors"))?
                .getattr(pyo3::intern!(py, "UDFException"))?
                .call1((message,))?;
            for (name, value) in attrs.iter() {
                raised.setattr(name.extract::<String>()?.as_str(), value)?;
            }
            return Ok(PyErr::from_value(raised));
        }
    }
    let raised = py
        .import(pyo3::intern!(py, "daft.exceptions"))?
        .getattr(pyo3::intern!(py, "DaftServeError"))?
        .call((message,), Some(&attrs))?;
    Ok(PyErr::from_value(raised))
}

fn block_on<F, T>(py: Python<'_>, future: F) -> PyResult<T>
where
    F: std::future::Future<Output = Result<T, ServeError>> + Send,
    T: Send,
{
    py.detach(|| common_runtime::get_io_runtime(true).block_on_current_thread(future))
        .map_err(to_py_err)
}

/// Runs a call for the query `query_id`, raising its failure with that id.
fn block_on_query<F, T>(py: Python<'_>, query_id: &str, future: F) -> PyResult<T>
where
    F: std::future::Future<Output = Result<T, ServeError>> + Send,
    T: Send,
{
    py.detach(|| common_runtime::get_io_runtime(true).block_on_current_thread(future))
        .map_err(|err| served_error(err, Some(query_id)))
}

/// A serving process bound to a host and port.
/// One tenant as configured from the client: name, token, and its optional
/// per-tenant limits (concurrent queries, queue timeout, query timeout, plan
/// payload bytes, memory cap bytes, request body bytes), each `None` where
/// the default applies.
type TenantSpec = (
    String,
    String,
    Option<usize>,
    Option<u64>,
    Option<u64>,
    Option<usize>,
    Option<usize>,
    Option<usize>,
);

/// Reads the request kinds a host declares, once, when the server starts.
fn declared_kinds(host: &Bound<'_, PyAny>) -> PyResult<HashSet<String>> {
    host.getattr(pyo3::intern!(host.py(), "kinds"))?
        .try_iter()?
        .map(|kind| kind?.extract::<String>())
        .collect()
}

#[pyclass(module = "daft.daft", name = "DaftServeServer", frozen)]
pub struct PyDaftServer {
    port: u16,
    address: String,
    shutdown: ServeShutdownHandle,
    task: Mutex<Option<common_runtime::RuntimeTask<common_error::DaftResult<()>>>>,
}

#[pymethods]
impl PyDaftServer {
    /// Starts a server; returns once the listener is bound.
    #[new]
    #[pyo3(signature = (
        host,
        port,
        token=None,
        allow_insecure_remote=false,
        max_concurrent_queries=4,
        queue_timeout_secs=60,
        max_pset_bytes=256 * 1024 * 1024,
        disable_plan_payload=false,
        query_timeout_secs=0,
        query_memory_cap_bytes=0,
        tenants=Vec::new(),
        session=None,
        catalogs=Vec::new(),
        request_host=None,
        max_request_body_bytes=1024 * 1024,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        py: Python<'_>,
        host: String,
        port: u16,
        token: Option<String>,
        allow_insecure_remote: bool,
        max_concurrent_queries: usize,
        queue_timeout_secs: u64,
        max_pset_bytes: usize,
        disable_plan_payload: bool,
        query_timeout_secs: u64,
        query_memory_cap_bytes: usize,
        tenants: Vec<TenantSpec>,
        session: Option<Py<PyAny>>,
        catalogs: Vec<String>,
        request_host: Option<Py<PyAny>>,
        max_request_body_bytes: usize,
    ) -> PyResult<Self> {
        let host_kinds = request_host
            .as_ref()
            .map(|request_host| declared_kinds(request_host.bind(py)))
            .transpose()?
            .unwrap_or_default();
        let tenants: Vec<TenantConfig> = tenants
            .into_iter()
            .map(
                |(
                    name,
                    tenant_token,
                    max_concurrent,
                    queue_timeout,
                    query_timeout,
                    pset_cap,
                    memory_cap,
                    request_body_cap,
                )| {
                    TenantConfig {
                        name,
                        token: tenant_token,
                        max_concurrent_queries: max_concurrent,
                        queue_timeout_secs: queue_timeout,
                        query_timeout_secs: query_timeout,
                        max_pset_bytes: pset_cap,
                        memory_cap_bytes: memory_cap,
                        max_request_body_bytes: request_body_cap,
                    }
                },
            )
            .collect();
        // Tenant credentials take precedence: when tenants are configured,
        // every caller must present one of the tenant tokens.
        let auth = if tenants.is_empty() {
            token.map_or(AuthPolicy::Insecure, AuthPolicy::Token)
        } else {
            AuthPolicy::Tenants(
                tenants
                    .iter()
                    .map(|tenant| crate::auth::TenantAuth {
                        name: tenant.name.clone(),
                        token: tenant.token.clone(),
                    })
                    .collect(),
            )
        };
        let config = ServeConfig {
            host,
            port,
            auth,
            allow_insecure_remote,
            max_concurrent_queries,
            queue_timeout_secs,
            max_pset_bytes,
            disable_plan_payload,
            query_timeout_secs,
            query_memory_cap_bytes,
            max_request_body_bytes,
            tenants,
        };
        let planners = crate::execute::Planners {
            session: session.map(Arc::new),
            host: request_host.map(Arc::new),
        };
        let service = DaftServeService::new(
            config,
            common_version::VERSION.to_string(),
            planners,
            host_kinds,
            catalogs,
        );
        let mut handle = py.detach(|| start_server(service)).map_err(to_py_err)?;
        let address = format!("grpc://{}", handle.addr());
        let port = handle.addr().port();
        let shutdown = handle.shutdown_handle();
        let task = handle.take_task();
        Ok(Self {
            port,
            address,
            shutdown,
            task: Mutex::new(task),
        })
    }

    /// Address clients should connect to.
    pub fn address(&self) -> String {
        self.address.clone()
    }

    /// Bound port.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Number of queries accepted and not yet finished, whether waiting for an
    /// execution slot or executing.
    pub fn active_queries(&self) -> usize {
        self.shutdown.active_queries()
    }

    /// Blocks until the transport loop exits (after `shutdown` is called
    /// from another thread, or the process is torn down).
    pub fn wait(&self, py: Python<'_>) -> PyResult<()> {
        let task = {
            let mut guard = self
                .task
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            guard.take()
        };
        if let Some(task) = task {
            py.detach(|| common_runtime::get_io_runtime(true).block_on_current_thread(task)?)
                .map_err(PyErr::from)?;
        }
        Ok(())
    }

    /// Begins a graceful shutdown: refuse new queries, drain in-flight
    /// streams up to `drain_timeout_secs`, then cancel whatever remains.
    /// Returns immediately; pair with `wait` to block until drained.
    #[pyo3(signature = (drain_timeout_secs=120))]
    pub fn shutdown(&self, drain_timeout_secs: u64) {
        self.shutdown
            .begin_shutdown(Duration::from_secs(drain_timeout_secs));
    }
}

/// Server description exposed to clients.
#[pyclass(module = "daft.daft", name = "DaftServeServerInfo", frozen)]
pub struct PyServeServerInfo {
    inner: wire::ServerInfo,
}

#[pymethods]
impl PyServeServerInfo {
    /// Engine version of the serving process.
    #[getter]
    pub fn version(&self) -> String {
        self.inner.version.clone()
    }

    /// Whether serialized-plan payloads are accepted.
    #[getter]
    pub fn plan_payload_enabled(&self) -> bool {
        self.inner.plan_payload_enabled
    }

    /// Maximum concurrently executing queries.
    #[getter]
    pub fn max_concurrent_queries(&self) -> usize {
        self.inner.max_concurrent_queries
    }

    /// Cap on in-memory partition bytes shipped with one query.
    #[getter]
    pub fn max_pset_bytes(&self) -> usize {
        self.inner.max_pset_bytes
    }

    /// Names of catalogs attached to the server session.
    #[getter]
    pub fn catalogs(&self) -> Vec<String> {
        self.inner.catalogs.clone()
    }
}

/// One query's result stream, iterated synchronously.
#[pyclass(module = "daft.daft", name = "DaftServeQueryResult", frozen)]
pub struct PyServeQueryResult {
    stream: tokio::sync::Mutex<QueryResultStream>,
    optimized_locally: bool,
    query_id: String,
}

#[pymethods]
impl PyServeQueryResult {
    /// Whether scan planning ran on the client because a source could not
    /// travel; the server then executed pre-materialized scan tasks.
    #[getter]
    pub fn optimized_locally(&self) -> bool {
        self.optimized_locally
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<PyMicroPartition>> {
        let partition = py.detach(|| {
            common_runtime::get_io_runtime(true)
                .block_on_current_thread(async { self.stream.lock().await.next_partition().await })
        });
        match partition {
            None => Ok(None),
            Some(Ok(partition)) => Ok(Some(PyMicroPartition::from(partition))),
            Some(Err(err)) => Err(served_error(err, Some(&self.query_id))),
        }
    }

    /// Execution statistics, available after the stream is fully drained:
    /// a `(physical_plan_json, stats)` pair.
    pub fn stats(&self, py: Python<'_>) -> PyResult<Option<(Option<String>, PyExecutionStats)>> {
        let stream = py.detach(|| {
            common_runtime::get_io_runtime(true)
                .block_on_current_thread(async { self.stream.lock().await })
        });
        Ok(stream.stats().map(|wire_stats| {
            let stats = ExecutionStats::decode(&wire_stats.stats);
            (
                wire_stats.physical_plan_json.clone(),
                PyExecutionStats::from(stats),
            )
        }))
    }
}

/// Connection to a remote serving endpoint.
///
/// Safe to share across threads: each call works on its own clone of the
/// connection handle, so a query waiting for an execution slot never blocks a
/// concurrent query, cancel or probe on the same client.
#[pyclass(module = "daft.daft", name = "DaftServeClient", frozen)]
pub struct PyDaftServeClient {
    inner: ServeClient,
}

impl PyDaftServeClient {
    fn build_plan_request(
        py: Python<'_>,
        builder: &PyLogicalPlanBuilder,
        psets: HashMap<String, Vec<PyMicroPartition>>,
        exec_config: Option<&PyDaftExecutionConfig>,
        query_id: String,
        results_buffer_size: Option<usize>,
    ) -> PyResult<(QueryRequest, bool)> {
        let plan = builder.builder.plan.clone();
        let exec_config_bytes = exec_config
            .map(|cfg| wire::encode(cfg.config.as_ref()))
            .transpose()
            .map_err(to_py_err)?;
        let effective_config = exec_config.map_or_else(
            || daft_context::get_context().execution_config(),
            |cfg| cfg.config.clone(),
        );
        let request = py
            .detach(|| -> Result<(QueryRequest, bool), ServeError> {
                let (plan, optimized_locally) =
                    crate::client::prepare_plan_for_shipping(plan, &effective_config)
                        .map_err(ServeError::Execution)?;
                let plan_bytes = wire::encode(&plan)?;
                let referenced = referenced_pset_keys(&plan);
                let mut named = Vec::with_capacity(referenced.len());
                for key in referenced {
                    let Some(partitions) = psets.get(&key) else {
                        continue;
                    };
                    let schema = partitions
                        .first()
                        .map(|p| wire::encode(&p.inner.schema()))
                        .transpose()?
                        .unwrap_or_default();
                    let blobs = partitions
                        .iter()
                        .map(|p| codec::micropartition_to_ipc(&p.inner))
                        .collect::<Result<Vec<_>, _>>()?;
                    named.push(NamedPartitionSet {
                        key,
                        schema,
                        partitions: blobs,
                    });
                }
                Ok((
                    QueryRequest {
                        query_id,
                        payload: QueryPayload::Plan(plan_bytes),
                        exec_config: exec_config_bytes,
                        psets: named,
                        results_buffer_size,
                    },
                    optimized_locally,
                ))
            })
            .map_err(to_py_err)?;
        Ok(request)
    }
}

#[pymethods]
impl PyDaftServeClient {
    /// Connects to a serving endpoint, e.g. `grpc://host:9494`.
    #[new]
    #[pyo3(signature = (address, token=None))]
    pub fn new(py: Python<'_>, address: String, token: Option<String>) -> PyResult<Self> {
        let inner = block_on(py, ServeClient::connect(&address, token))?;
        Ok(Self { inner })
    }

    /// Fetches the server description.
    pub fn server_info(&self, py: Python<'_>) -> PyResult<PyServeServerInfo> {
        let mut client = self.inner.clone();
        let info = block_on(py, async move { client.server_info().await })?;
        Ok(PyServeServerInfo { inner: info })
    }

    /// Probes server liveness.
    pub fn health(&self, py: Python<'_>) -> PyResult<()> {
        let mut client = self.inner.clone();
        block_on(py, async move { client.health().await })
    }

    /// Cancels a running query by id; returns whether it was found.
    pub fn cancel_query(&self, py: Python<'_>, query_id: String) -> PyResult<bool> {
        let mut client = self.inner.clone();
        block_on(py, async move { client.cancel_query(&query_id).await })
    }

    /// Submits a serialized plan with its referenced in-memory partitions.
    #[pyo3(signature = (builder, psets, query_id, exec_config=None, results_buffer_size=None))]
    pub fn run_plan(
        &self,
        py: Python<'_>,
        builder: &PyLogicalPlanBuilder,
        psets: HashMap<String, Vec<PyMicroPartition>>,
        query_id: String,
        exec_config: Option<PyDaftExecutionConfig>,
        results_buffer_size: Option<usize>,
    ) -> PyResult<PyServeQueryResult> {
        let (request, optimized_locally) = Self::build_plan_request(
            py,
            builder,
            psets,
            exec_config.as_ref(),
            query_id,
            results_buffer_size,
        )?;
        let query_id = request.query_id.clone();
        let mut client = self.inner.clone();
        let stream = block_on_query(py, &query_id, async move { client.run(&request).await })?;
        Ok(PyServeQueryResult {
            stream: tokio::sync::Mutex::new(stream),
            optimized_locally,
            query_id,
        })
    }

    /// Submits a textual query resolved against server-side catalogs.
    #[pyo3(signature = (sql, query_id, results_buffer_size=None))]
    pub fn run_sql(
        &self,
        py: Python<'_>,
        sql: String,
        query_id: String,
        results_buffer_size: Option<usize>,
    ) -> PyResult<PyServeQueryResult> {
        let request = QueryRequest {
            query_id,
            payload: QueryPayload::Sql(sql),
            exec_config: None,
            psets: vec![],
            results_buffer_size,
        };
        let query_id = request.query_id.clone();
        let mut client = self.inner.clone();
        let stream = block_on_query(py, &query_id, async move { client.run(&request).await })?;
        Ok(PyServeQueryResult {
            stream: tokio::sync::Mutex::new(stream),
            optimized_locally: false,
            query_id,
        })
    }

    /// Submits a request of a kind the server's host plans.
    ///
    /// The server routes it by `kind` and hands the host the body unread,
    /// together with the query id and the authenticated caller.
    #[pyo3(signature = (kind, body, query_id, results_buffer_size=None))]
    pub fn run_request(
        &self,
        py: Python<'_>,
        kind: String,
        body: Vec<u8>,
        query_id: String,
        results_buffer_size: Option<usize>,
    ) -> PyResult<PyServeQueryResult> {
        let request = QueryRequest {
            query_id,
            payload: QueryPayload::Request { kind, body },
            exec_config: None,
            psets: vec![],
            results_buffer_size,
        };
        let query_id = request.query_id.clone();
        let mut client = self.inner.clone();
        let stream = block_on_query(py, &query_id, async move { client.run(&request).await })?;
        Ok(PyServeQueryResult {
            stream: tokio::sync::Mutex::new(stream),
            optimized_locally: false,
            query_id,
        })
    }

    /// Renders the plan the server's host would run for a request, without
    /// executing it.
    #[pyo3(signature = (kind, body, query_id=None))]
    pub fn explain_request(
        &self,
        py: Python<'_>,
        kind: String,
        body: Vec<u8>,
        query_id: Option<String>,
    ) -> PyResult<String> {
        let request = QueryRequest {
            query_id: query_id.unwrap_or_else(explain_query_id),
            payload: QueryPayload::Request { kind, body },
            exec_config: None,
            psets: vec![],
            results_buffer_size: None,
        };
        let query_id = request.query_id.clone();
        let mut client = self.inner.clone();
        let explained =
            block_on_query(py, &query_id, async move { client.explain(&request).await })?;
        Ok(explained.optimized_plan)
    }

    /// Renders the server-side optimized plan without executing.
    #[pyo3(signature = (builder, query_id=None))]
    pub fn explain_plan(
        &self,
        py: Python<'_>,
        builder: &PyLogicalPlanBuilder,
        query_id: Option<String>,
    ) -> PyResult<String> {
        let (request, optimized_locally) = Self::build_plan_request(
            py,
            builder,
            HashMap::new(),
            None,
            query_id.unwrap_or_else(explain_query_id),
            None,
        )?;
        let query_id = request.query_id.clone();
        let mut client = self.inner.clone();
        let explained =
            block_on_query(py, &query_id, async move { client.explain(&request).await })?;
        if optimized_locally {
            Ok(format!(
                "Note: a source in this plan cannot travel, so scan planning \
                 ran on the client; the server received pre-materialized scan \
                 tasks.\n{}",
                explained.optimized_plan
            ))
        } else {
            Ok(explained.optimized_plan)
        }
    }
}

/// A fresh id for an explanation the caller did not name.
///
/// An explanation is registered like a query, so concurrent explanations
/// need distinct ids.
fn explain_query_id() -> String {
    format!("explain-{}", uuid::Uuid::new_v4().simple())
}

/// In-memory cache keys referenced by a plan's sources.
///
/// Lets a client materialize and ship only the cached partition sets a query
/// actually reads instead of everything in its cache.
#[pyfunction]
#[must_use]
pub fn serve_referenced_pset_keys(builder: &PyLogicalPlanBuilder) -> Vec<String> {
    referenced_pset_keys(&builder.builder.plan)
}

/// Registers the serving classes on the extension module.
///
/// # Errors
/// Returns an error if a class cannot be added to the module.
pub fn register_modules(parent: &Bound<PyModule>) -> PyResult<()> {
    parent.add_class::<PyDaftServer>()?;
    parent.add_class::<PyDaftServeClient>()?;
    parent.add_class::<PyServeQueryResult>()?;
    parent.add_class::<PyServeServerInfo>()?;
    parent.add_function(wrap_pyfunction!(serve_referenced_pset_keys, parent)?)?;
    Ok(())
}
