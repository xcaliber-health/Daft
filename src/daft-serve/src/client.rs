//! Client for the serving endpoint: submits queries and decodes streaming
//! results back into engine partitions.
//!
//! The client speaks the same single-round-trip protocol the server exposes:
//! the full request rides in the retrieval ticket, and the response stream is
//! decoded incrementally so partitions are available before the query
//! finishes. The terminal metadata-only message is captured as execution
//! statistics.

use std::sync::Arc;

use arrow_flight::{
    Action, Ticket,
    decode::{DecodedPayload, FlightDataDecoder},
    flight_service_client::FlightServiceClient,
};
use common_error::DaftError;
use daft_core::prelude::SchemaRef;
use daft_logical_plan::{LogicalPlan, source_info::SourceInfo};
use daft_micropartition::MicroPartition;
use daft_schema::field::FieldRef;
use futures::StreamExt;
use tonic::transport::{Channel, Endpoint};

use crate::{
    codec,
    error::{ERROR_CLASS_METADATA_KEY, ServeError, ServeResult},
    wire::{self, ExplainResult, QueryRequest, QueryStatsWire, ServerInfo, actions},
};

/// Prepares a plan for serialization.
///
/// Operators that can travel — interpreter-backed sources such as catalog
/// tables — are left lazy so their scan planning runs on the server, next
/// to the data. If any source's operator cannot travel (native file scans,
/// or catalog handles holding process-local state), the plan is instead
/// optimized on this side: optimization materializes every scan into
/// concrete tasks *with* projection and predicate pushdowns applied, so the
/// server executes exactly what local execution would. Native file scans
/// already performed schema inference on this side at build time, so
/// planning them here adds no new access requirement.
///
/// Returns the plan to ship plus whether it was optimized on this side, so
/// callers can surface where scan planning ran.
///
/// # Errors
/// Returns an error if client-side optimization fails.
pub fn prepare_plan_for_shipping(
    plan: Arc<LogicalPlan>,
    exec_config: &Arc<common_daft_config::DaftExecutionConfig>,
) -> Result<(Arc<LogicalPlan>, bool), common_error::DaftError> {
    use common_treenode::{TreeNode, TreeNodeRecursion};
    use daft_logical_plan::source_info::SourceInfo;
    use daft_scan::scan_state::ScanState;

    // An operator is shippable only if it exposes an interpreter object AND
    // that object actually serializes — some catalog implementations hold
    // process-local state (connection pools, weak references) that cannot
    // travel even though the operator type generally can.
    let operator_ships = |op: &daft_scan::ScanOperatorRef| -> bool {
        op.0.shippable_py_object().is_some_and(|obj| {
            pyo3::Python::attach(|py| common_py_serde::pickle_dumps(py, &obj).is_ok())
        })
    };

    let mut all_ship = true;
    plan.apply(|node| {
        if let LogicalPlan::Source(source) = node.as_ref()
            && let SourceInfo::Physical(info) = source.source_info.as_ref()
            && let ScanState::Operator(op) = &info.scan_state
            && !operator_ships(op)
        {
            all_ship = false;
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;

    if all_ship {
        return Ok((plan, false));
    }
    // Materializing scans without running the optimizer would bake in empty
    // pushdowns and force the server to scan full-width tables; optimize
    // here instead so tasks carry their projections and predicates.
    let optimized =
        daft_logical_plan::LogicalPlanBuilder::new(plan, None).optimize(exec_config.clone())?;
    Ok((optimized.build(), true))
}

/// Collects the in-memory cache keys referenced by a plan, so a client ships
/// only the partition sets the query actually reads.
#[must_use]
pub fn referenced_pset_keys(plan: &LogicalPlan) -> Vec<String> {
    let mut keys = Vec::new();
    let mut stack = vec![plan];
    while let Some(node) = stack.pop() {
        if let LogicalPlan::Source(source) = node
            && let SourceInfo::InMemory(info) = source.source_info.as_ref()
        {
            keys.push(info.cache_key.clone());
        }
        stack.extend(node.children());
    }
    keys.sort_unstable();
    keys.dedup();
    keys
}

/// Maps a transport status into a client-side error, preserving the server's
/// error class for faithful re-raising.
fn status_to_error(status: &tonic::Status) -> ServeError {
    let class = status
        .metadata()
        .get(ERROR_CLASS_METADATA_KEY)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("Transport");
    ServeError::Execution(DaftError::External(
        format!("[{class}] {}", status.message()).into(),
    ))
}

/// Connection to one serving endpoint.
pub struct ServeClient {
    inner: FlightServiceClient<Channel>,
    token: Option<String>,
    client_version: String,
}

impl ServeClient {
    /// Establishes a connection.
    ///
    /// # Errors
    /// Returns an error if the address is invalid or unreachable.
    pub async fn connect(address: &str, token: Option<String>) -> ServeResult<Self> {
        let endpoint = Endpoint::from_shared(address.to_string())
            .map_err(|e| ServeError::InvalidEnvelope(format!("invalid address: {e}")))?;
        let channel = endpoint.connect().await.map_err(|e| {
            ServeError::Execution(DaftError::External(
                format!("failed to connect to {address}: {e}").into(),
            ))
        })?;
        let inner = FlightServiceClient::new(channel)
            .max_decoding_message_size(usize::MAX)
            .max_encoding_message_size(usize::MAX);
        // The override exists solely so integration tests can exercise the
        // server's version-mismatch rejection end to end without building a
        // second engine version.
        let client_version = std::env::var("DAFT_SERVE_CLIENT_VERSION_OVERRIDE")
            .unwrap_or_else(|_| common_version::VERSION.to_string());
        Ok(Self {
            inner,
            token,
            client_version,
        })
    }

    fn authorize<T>(&self, message: T) -> ServeResult<tonic::Request<T>> {
        let mut request = tonic::Request::new(message);
        if let Some(token) = &self.token {
            let value = format!("{}{token}", crate::auth::BEARER_PREFIX)
                .parse()
                .map_err(|_| {
                    ServeError::Unauthenticated("token contains invalid characters".to_string())
                })?;
            request
                .metadata_mut()
                .insert(crate::auth::AUTHORIZATION_KEY, value);
        }
        Ok(request)
    }

    /// Submits a query and returns the decoding result stream.
    ///
    /// # Errors
    /// Returns an error if the request cannot be encoded or the call is
    /// rejected.
    pub async fn run(&mut self, request: &QueryRequest) -> ServeResult<QueryResultStream> {
        let ticket = Ticket {
            ticket: wire::encode_envelope(&self.client_version, request)?.into(),
        };
        let call = self.authorize(ticket)?;
        let response = self
            .inner
            .do_get(call)
            .await
            .map_err(|s| status_to_error(&s))?;
        let stream = response
            .into_inner()
            .map(|r| r.map_err(|s| arrow_flight::error::FlightError::Tonic(Box::new(s))));
        Ok(QueryResultStream {
            decoder: FlightDataDecoder::new(stream),
            schema: None,
            stats: None,
        })
    }

    /// Executes an auxiliary action and returns its raw response body.
    ///
    /// # Errors
    /// Returns an error if the action is rejected or the connection fails.
    pub async fn action(&mut self, name: &str, body: Vec<u8>) -> ServeResult<Vec<u8>> {
        let action = Action {
            r#type: name.to_string(),
            body: body.into(),
        };
        let call = self.authorize(action)?;
        let mut stream = self
            .inner
            .do_action(call)
            .await
            .map_err(|s| status_to_error(&s))?
            .into_inner();
        let first = stream
            .next()
            .await
            .transpose()
            .map_err(|s| status_to_error(&s))?;
        Ok(first.map(|r| r.body.to_vec()).unwrap_or_default())
    }

    /// Fetches the server description.
    ///
    /// # Errors
    /// Returns an error if the call fails or the response does not decode.
    pub async fn server_info(&mut self) -> ServeResult<ServerInfo> {
        let body = self.action(actions::SERVER_INFO, vec![]).await?;
        wire::decode(&body)
    }

    /// Probes server liveness.
    ///
    /// # Errors
    /// Returns an error if the server is unreachable or refuses the call.
    pub async fn health(&mut self) -> ServeResult<()> {
        self.action(actions::HEALTH, vec![]).await.map(|_| ())
    }

    /// Cancels a running query by id; returns whether it was found.
    ///
    /// # Errors
    /// Returns an error if the call fails.
    pub async fn cancel_query(&mut self, query_id: &str) -> ServeResult<bool> {
        let body = self
            .action(actions::CANCEL_QUERY, query_id.as_bytes().to_vec())
            .await?;
        Ok(body.first().copied() == Some(1))
    }

    /// Requests the server-side optimized plan rendering for a query.
    ///
    /// # Errors
    /// Returns an error if the call fails or the response does not decode.
    pub async fn explain(&mut self, request: &QueryRequest) -> ServeResult<ExplainResult> {
        let envelope = wire::encode_envelope(&self.client_version, request)?;
        let body = self.action(actions::EXPLAIN, envelope).await?;
        wire::decode(&body)
    }
}

/// Incrementally decodes one query's result stream.
pub struct QueryResultStream {
    decoder: FlightDataDecoder,
    schema: Option<(SchemaRef, Vec<FieldRef>)>,
    stats: Option<QueryStatsWire>,
}

impl QueryResultStream {
    /// Receives the next result partition, or `None` at end of stream.
    ///
    /// # Errors
    /// Yields an error item if the stream fails or a message cannot be
    /// decoded into engine types.
    pub async fn next_partition(&mut self) -> Option<ServeResult<MicroPartition>> {
        loop {
            let message = match self.decoder.next().await? {
                Ok(message) => message,
                Err(arrow_flight::error::FlightError::Tonic(status)) => {
                    return Some(Err(status_to_error(status.as_ref())));
                }
                Err(e) => {
                    return Some(Err(ServeError::Execution(DaftError::External(
                        format!("result stream failed: {e}").into(),
                    ))));
                }
            };
            match &message.payload {
                DecodedPayload::Schema(_) => {
                    let app_metadata = &message.inner.app_metadata;
                    match wire::decode::<SchemaRef>(app_metadata) {
                        Ok(schema) => {
                            let fields = schema
                                .fields()
                                .iter()
                                .map(|f| Arc::new(f.clone()))
                                .collect();
                            self.schema = Some((schema, fields));
                        }
                        Err(e) => return Some(Err(e)),
                    }
                }
                DecodedPayload::RecordBatch(batch) => {
                    let app_metadata = &message.inner.app_metadata;
                    if !app_metadata.is_empty() {
                        // The trailer: an empty batch carrying execution
                        // statistics; not surfaced as data.
                        match wire::decode::<QueryStatsWire>(app_metadata) {
                            Ok(stats) => self.stats = Some(stats),
                            Err(e) => return Some(Err(e)),
                        }
                        continue;
                    }
                    let Some((schema, fields)) = &self.schema else {
                        return Some(Err(ServeError::MalformedPayload(
                            "result batch arrived before the stream schema".to_string(),
                        )));
                    };
                    return Some(
                        codec::arrow_batch_to_engine_batch(schema, fields, batch).map(
                            |engine_batch| {
                                MicroPartition::new_loaded(
                                    schema.clone(),
                                    Arc::new(vec![engine_batch]),
                                    None,
                                )
                            },
                        ),
                    );
                }
                DecodedPayload::None => {}
            }
        }
    }

    /// Schema of the result stream, available after the first message.
    #[must_use]
    pub fn schema(&self) -> Option<&SchemaRef> {
        self.schema.as_ref().map(|(schema, _)| schema)
    }

    /// Execution statistics, available once the stream has fully drained.
    #[must_use]
    pub fn stats(&self) -> Option<&QueryStatsWire> {
        self.stats.as_ref()
    }
}
