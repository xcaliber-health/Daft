//! The serving endpoint: a streaming RPC service executing queries against
//! the local engine.
//!
//! One `DoGet` call carries a complete query (single round trip) and returns
//! one result stream. Auxiliary operations (health, server info, explain,
//! cross-connection cancel) are exposed as actions. A standard health
//! service is mounted alongside for orchestrator probes.

use std::{
    net::{IpAddr, SocketAddr},
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use arrow_flight::{
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PollInfo, PutResult, SchemaResult, Ticket,
    flight_service_server::{FlightService, FlightServiceServer},
};
use common_error::{DaftError, DaftResult};
use common_runtime::RuntimeTask;
use futures::Stream;
use tonic::{Request, Response, Status, transport::Server};

use crate::{
    admission::Admission,
    auth::{AUTHORIZATION_KEY, AuthPolicy},
    error::{ServeError, ServeResult},
    execute,
    registry::QueryRegistry,
    wire::{self, ExplainResult, QueryRequest, ServerInfo, actions},
};

/// Runtime configuration of one serving process.
#[derive(Debug, Clone)]
pub struct ServeConfig {
    /// Interface to bind, e.g. `127.0.0.1` or `0.0.0.0`.
    pub host: String,
    /// Port to bind; `0` picks an ephemeral port.
    pub port: u16,
    /// Authentication policy for every request.
    pub auth: AuthPolicy,
    /// Whether an insecure policy is explicitly permitted on a non-loopback
    /// bind.
    pub allow_insecure_remote: bool,
    /// Maximum concurrently executing queries.
    pub max_concurrent_queries: usize,
    /// Seconds a query may wait for an execution slot.
    pub queue_timeout_secs: u64,
    /// Cap on total in-memory partition bytes shipped with one query.
    pub max_pset_bytes: usize,
    /// Whether serialized-plan payloads are rejected (text-only server).
    pub disable_plan_payload: bool,
    /// Wall-clock seconds one query may execute before being cancelled;
    /// `0` disables the limit.
    pub query_timeout_secs: u64,
}

impl ServeConfig {
    /// Validates the configuration before binding.
    ///
    /// # Errors
    /// Returns an error if no token is configured for a non-loopback bind
    /// without the explicit insecure opt-out, or if limits are zero.
    pub fn validate(&self) -> ServeResult<()> {
        let loopback = self
            .host
            .parse::<IpAddr>()
            .map(|ip| ip.is_loopback())
            .unwrap_or(self.host == "localhost");
        if matches!(self.auth, AuthPolicy::Insecure) && !loopback && !self.allow_insecure_remote {
            return Err(ServeError::Unauthenticated(format!(
                "refusing to bind {} without a token; configure one or opt out explicitly",
                self.host
            )));
        }
        if self.max_concurrent_queries == 0 {
            return Err(ServeError::InvalidEnvelope(
                "max_concurrent_queries must be at least 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Shared state of the serving process.
pub struct DaftServeService {
    config: ServeConfig,
    admission: Admission,
    registry: QueryRegistry,
    server_version: String,
    draining: Arc<AtomicBool>,
    sql_session: Option<Arc<pyo3::Py<pyo3::PyAny>>>,
    catalogs: Vec<String>,
}

impl DaftServeService {
    /// Creates the service from validated configuration.
    #[must_use]
    pub fn new(
        config: ServeConfig,
        server_version: String,
        sql_session: Option<Arc<pyo3::Py<pyo3::PyAny>>>,
        catalogs: Vec<String>,
    ) -> Self {
        let admission = Admission::new(
            config.max_concurrent_queries,
            Duration::from_secs(config.queue_timeout_secs),
        );
        Self {
            config,
            admission,
            registry: QueryRegistry::new(),
            server_version,
            draining: Arc::new(AtomicBool::new(false)),
            sql_session,
            catalogs,
        }
    }

    /// Registry of running queries; exposed for drain handling.
    #[must_use]
    pub fn registry(&self) -> &QueryRegistry {
        &self.registry
    }

    /// Flag shared with the transport loop; set to refuse new queries.
    #[must_use]
    pub fn draining_flag(&self) -> Arc<AtomicBool> {
        self.draining.clone()
    }

    fn check_auth<T>(&self, request: &Request<T>) -> ServeResult<()> {
        let header = request
            .metadata()
            .get(AUTHORIZATION_KEY)
            .and_then(|v| v.to_str().ok());
        self.config.auth.check(header)
    }

    /// Parses and policy-checks a request envelope into a query request.
    fn admit_request(&self, ticket_bytes: &[u8]) -> ServeResult<QueryRequest> {
        if self.draining.load(Ordering::Acquire) {
            return Err(ServeError::Draining);
        }
        let envelope = wire::parse_envelope(ticket_bytes)?;
        let request: QueryRequest = wire::decode(envelope.body)?;
        wire::check_request_policy(
            &request,
            &envelope.sender_version,
            &self.server_version,
            self.config.disable_plan_payload,
            self.config.max_pset_bytes,
        )?;
        Ok(request)
    }

    fn server_info(&self) -> ServerInfo {
        ServerInfo {
            version: self.server_version.clone(),
            wire_version: wire::WIRE_VERSION,
            plan_payload_enabled: !self.config.disable_plan_payload,
            max_concurrent_queries: self.admission.max_concurrent(),
            max_pset_bytes: self.config.max_pset_bytes,
            catalogs: self.catalogs.clone(),
        }
    }

    /// Renders the server-side optimized plan for a request without
    /// executing it.
    fn explain(&self, body: &[u8]) -> ServeResult<ExplainResult> {
        let request = self.admit_request(body)?;
        let exec_config = execute::resolve_exec_config(&request)?;
        let builder = execute::payload_to_builder(&request.payload, self.sql_session.as_ref())?;
        let optimized = builder.optimize(exec_config)?;
        Ok(ExplainResult {
            optimized_plan: optimized.build().repr_ascii(false),
        })
    }
}

type BoxedStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

/// Response stream that owns its execution task, so dropping the stream
/// aborts the query.
struct QueryResponseStream {
    rx: Pin<Box<async_channel::Receiver<Result<FlightData, Status>>>>,
    _task: RuntimeTask<()>,
}

impl Stream for QueryResponseStream {
    type Item = Result<FlightData, Status>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.get_mut().rx.as_mut().poll_next(cx)
    }
}

#[tonic::async_trait]
impl FlightService for DaftServeService {
    type HandshakeStream = BoxedStream<HandshakeResponse>;
    type ListFlightsStream = BoxedStream<FlightInfo>;
    type DoGetStream = BoxedStream<FlightData>;
    type DoPutStream = BoxedStream<PutResult>;
    type DoExchangeStream = BoxedStream<FlightData>;
    type DoActionStream = BoxedStream<arrow_flight::Result>;
    type ListActionsStream = BoxedStream<ActionType>;

    async fn handshake(
        &self,
        request: Request<tonic::Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        self.check_auth(&request).map_err(Status::from)?;
        let response = HandshakeResponse {
            protocol_version: u64::from(wire::WIRE_VERSION),
            payload: self.server_version.clone().into_bytes().into(),
        };
        let stream = futures::stream::once(async move { Ok(response) });
        Ok(Response::new(Box::pin(stream)))
    }

    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        Err(Status::unimplemented("listing flights is not supported"))
    }

    async fn get_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        Err(Status::unimplemented(
            "flight info is not supported; submit queries via do_get",
        ))
    }

    async fn poll_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<PollInfo>, Status> {
        Err(Status::unimplemented("polling is not supported"))
    }

    async fn get_schema(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        Err(Status::unimplemented(
            "schema lookup is not supported; the result stream carries its schema",
        ))
    }

    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        self.check_auth(&request).map_err(Status::from)?;
        let ticket = request.into_inner();
        let query = self.admit_request(&ticket.ticket).map_err(Status::from)?;

        let permit = self.admission.acquire().await.map_err(Status::from)?;
        let guard = self.registry.register(&query.query_id);
        let cancel = guard.token();

        // Buffer a handful of encoded messages so encoding overlaps with
        // network sends without unbounded memory growth.
        let buffer = query.results_buffer_size.unwrap_or(8).clamp(1, 64);
        let (tx, rx) = async_channel::bounded(buffer);
        let sql_session = self.sql_session.clone();
        let query_timeout_secs = self.config.query_timeout_secs;
        let task = common_runtime::get_io_runtime(true).spawn(async move {
            execute::run_query(
                query,
                sql_session,
                cancel,
                permit,
                guard,
                tx,
                query_timeout_secs,
            )
            .await;
        });

        // The response stream owns the execution task: dropping the stream
        // (client disconnect or completion) aborts the task, which drops the
        // engine instance and cancels its running pipeline.
        Ok(Response::new(Box::pin(QueryResponseStream {
            rx: Box::pin(rx),
            _task: task,
        })))
    }

    async fn do_put(
        &self,
        _request: Request<tonic::Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        Err(Status::unimplemented(
            "uploads are not supported; ship in-memory data with the query",
        ))
    }

    async fn do_exchange(
        &self,
        _request: Request<tonic::Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("exchange is not supported"))
    }

    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        self.check_auth(&request).map_err(Status::from)?;
        let action = request.into_inner();
        let body: Vec<u8> = match action.r#type.as_str() {
            actions::HEALTH => vec![],
            actions::SERVER_INFO => wire::encode(&self.server_info()).map_err(Status::from)?,
            actions::CANCEL_QUERY => {
                let query_id = String::from_utf8(action.body.to_vec())
                    .map_err(|e| Status::invalid_argument(format!("query id not UTF-8: {e}")))?;
                let found = self.registry.cancel(&query_id);
                vec![u8::from(found)]
            }
            actions::EXPLAIN => {
                let explained = self.explain(&action.body).map_err(Status::from)?;
                wire::encode(&explained).map_err(Status::from)?
            }
            other => {
                return Err(Status::unimplemented(format!("unknown action `{other}`")));
            }
        };
        let stream =
            futures::stream::once(async move { Ok(arrow_flight::Result { body: body.into() }) });
        Ok(Response::new(Box::pin(stream)))
    }

    async fn list_actions(
        &self,
        request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        self.check_auth(&request).map_err(Status::from)?;
        let all = [
            (actions::HEALTH, "liveness probe"),
            (actions::SERVER_INFO, "server version, capabilities, limits"),
            (actions::CANCEL_QUERY, "cancel a running query by id"),
            (actions::EXPLAIN, "render the optimized plan for a request"),
        ];
        let stream = futures::stream::iter(all.map(|(name, description)| {
            Ok(ActionType {
                r#type: name.to_string(),
                description: description.to_string(),
            })
        }));
        Ok(Response::new(Box::pin(stream)))
    }
}

/// Cloneable signal half of a running server: initiates a graceful
/// shutdown from any thread while another thread waits on the join half.
#[derive(Clone)]
pub struct ServeShutdownHandle {
    registry: QueryRegistry,
    draining: Arc<AtomicBool>,
    shutdown_signal: Arc<std::sync::Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
}

impl ServeShutdownHandle {
    /// Begins a graceful shutdown: refuse new queries, stop accepting
    /// connections, and after `drain_timeout` cancel any still-running
    /// queries so the transport can finish. Idempotent; returns immediately.
    pub fn begin_shutdown(&self, drain_timeout: Duration) {
        self.draining.store(true, Ordering::Release);
        let signal = self
            .shutdown_signal
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(signal) = signal {
            let _ = signal.send(());
            let registry = self.registry.clone();
            drop(common_runtime::get_io_runtime(true).spawn(async move {
                tokio::time::sleep(drain_timeout).await;
                registry.cancel_all();
            }));
        }
    }

    /// Number of queries currently executing.
    #[must_use]
    pub fn active_queries(&self) -> usize {
        self.registry.len()
    }
}

/// Handle to a running server: address, shutdown signalling, and join.
pub struct ServeHandle {
    addr: SocketAddr,
    shutdown: ServeShutdownHandle,
    server_task: Option<RuntimeTask<DaftResult<()>>>,
}

impl ServeHandle {
    /// Bound socket address.
    #[must_use]
    pub const fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Signal half, cloneable across threads.
    #[must_use]
    pub fn shutdown_handle(&self) -> ServeShutdownHandle {
        self.shutdown.clone()
    }

    /// Takes the join half; subsequent calls return `None`.
    #[must_use]
    pub fn take_task(&mut self) -> Option<RuntimeTask<DaftResult<()>>> {
        self.server_task.take()
    }

    /// Blocks until the transport loop exits.
    ///
    /// # Errors
    /// Returns an error if the transport loop failed.
    pub fn wait(&mut self) -> DaftResult<()> {
        if let Some(task) = self.server_task.take() {
            common_runtime::get_io_runtime(true).block_on_current_thread(task)??;
        }
        Ok(())
    }

    /// Gracefully shuts down and waits for the transport to exit.
    ///
    /// # Errors
    /// Returns an error if the transport loop failed.
    pub fn shutdown(&mut self, drain_timeout: Duration) -> DaftResult<()> {
        self.shutdown.begin_shutdown(drain_timeout);
        self.wait()
    }
}

/// Binds and starts the serving transport, returning a handle once the
/// listener is ready.
///
/// # Errors
/// Returns an error if configuration is invalid or the address cannot be
/// bound.
pub fn start_server(service: DaftServeService) -> ServeResult<ServeHandle> {
    service.config.validate()?;
    let registry = service.registry().clone();
    let draining = service.draining_flag();
    let bind_addr = format!("{}:{}", service.config.host, service.config.port);

    let io_runtime = common_runtime::get_io_runtime(true);
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let (addr_tx, addr_rx) = tokio::sync::oneshot::channel::<Result<SocketAddr, String>>();

    let server_task = io_runtime.spawn(async move {
        let listener = match tokio::net::TcpListener::bind(&bind_addr).await {
            Ok(listener) => listener,
            Err(e) => {
                let _ = addr_tx.send(Err(format!("failed to bind {bind_addr}: {e}")));
                return Ok(());
            }
        };
        let addr = match listener.local_addr() {
            Ok(addr) => addr,
            Err(e) => {
                let _ = addr_tx.send(Err(format!("failed to read bound address: {e}")));
                return Ok(());
            }
        };
        let _ = addr_tx.send(Ok(addr));

        let incoming = tonic::transport::server::TcpIncoming::from(listener)
            .with_nodelay(Some(true))
            .with_keepalive(None);

        let (health_reporter, health_service) = tonic_health::server::health_reporter();
        health_reporter
            .set_serving::<FlightServiceServer<DaftServeService>>()
            .await;

        // Inbound requests carry the whole query (plan plus any shipped
        // partition data) in one message, so the transport's decode limit
        // must track the configured payload cap — otherwise requests between
        // the transport default and `max_pset_bytes` die with an opaque
        // transport error before the friendly size check runs. Headroom
        // covers the plan and envelope alongside the partition bytes.
        let max_inbound_bytes = service
            .config
            .max_pset_bytes
            .saturating_add(64 * 1024 * 1024);
        Server::builder()
            .add_service(health_service)
            .add_service(
                FlightServiceServer::new(service)
                    .max_decoding_message_size(max_inbound_bytes)
                    .max_encoding_message_size(usize::MAX),
            )
            .serve_with_incoming_shutdown(incoming, async move {
                let _ = shutdown_rx.await;
            })
            .await
            .map_err(|e| DaftError::InternalError(format!("serving transport failed: {e}")))?;
        Ok(())
    });

    let addr = addr_rx
        .blocking_recv()
        .map_err(|_| {
            ServeError::Execution(DaftError::InternalError(
                "server task exited before reporting its address".to_string(),
            ))
        })?
        .map_err(|msg| ServeError::Execution(DaftError::InternalError(msg)))?;

    Ok(ServeHandle {
        addr,
        shutdown: ServeShutdownHandle {
            registry,
            draining,
            shutdown_signal: Arc::new(std::sync::Mutex::new(Some(shutdown_tx))),
        },
        server_task: Some(server_task),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(host: &str, auth: AuthPolicy, allow_insecure_remote: bool) -> ServeConfig {
        ServeConfig {
            host: host.to_string(),
            port: 0,
            auth,
            allow_insecure_remote,
            max_concurrent_queries: 4,
            queue_timeout_secs: 30,
            max_pset_bytes: 64 * 1024 * 1024,
            disable_plan_payload: false,
            query_timeout_secs: 0,
        }
    }

    #[test]
    fn loopback_without_token_is_allowed() {
        for host in ["127.0.0.1", "::1", "localhost"] {
            assert!(config(host, AuthPolicy::Insecure, false).validate().is_ok());
        }
    }

    #[test]
    fn remote_bind_without_token_is_refused() {
        let err = config("0.0.0.0", AuthPolicy::Insecure, false)
            .validate()
            .unwrap_err();
        assert!(matches!(err, ServeError::Unauthenticated(_)));
    }

    #[test]
    fn remote_bind_with_token_is_allowed() {
        assert!(
            config("0.0.0.0", AuthPolicy::Token("t".to_string()), false)
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn remote_bind_with_explicit_insecure_optout_is_allowed() {
        assert!(
            config("0.0.0.0", AuthPolicy::Insecure, true)
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn zero_concurrency_is_refused() {
        let mut cfg = config("127.0.0.1", AuthPolicy::Insecure, false);
        cfg.max_concurrent_queries = 0;
        assert!(cfg.validate().is_err());
    }
}
