//! Wire format for the query-serving protocol.
//!
//! A request travels as an opaque byte payload inside a transport-level
//! ticket. The layout is a small hand-parsed header followed by a
//! binary-serialized body:
//!
//! ```text
//! magic:        4 bytes, b"DFS1"
//! wire version: u16 little-endian
//! version len:  u16 little-endian
//! version:      UTF-8 engine version string of the sender
//! body:         binary-serialized [`QueryRequest`]
//! ```
//!
//! The header is parsed with fixed-layout reads and never changes across
//! releases, so a server can always identify the sender's version and reject
//! incompatible requests with an actionable error *before* attempting to
//! decode the body. The body encoding is only guaranteed compatible between
//! identical engine versions; the header carries everything needed to enforce
//! that.

use serde::{Deserialize, Serialize};

use crate::error::{ServeError, ServeResult};

/// Magic bytes identifying a serve-protocol request.
pub const WIRE_MAGIC: [u8; 4] = *b"DFS1";

/// Version of the envelope layout itself. Bumped only if the header layout or
/// body encoding scheme changes.
pub const WIRE_VERSION: u16 = 1;

/// Default port the server listens on.
pub const DEFAULT_PORT: u16 = 9494;

/// Upper bound on the encoded version string, to bound header parsing.
const MAX_VERSION_LEN: usize = 256;

/// A single named set of in-memory partitions shipped with a query.
///
/// Partitions are encoded as self-contained streaming-IPC byte blobs plus the
/// engine-native schema, so the receiver can reconstruct exact column types
/// (including extension types) without lossy inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedPartitionSet {
    /// Cache key this partition set is registered under in the plan.
    pub key: String,
    /// Engine-native schema of the partitions, binary-serialized.
    pub schema: Vec<u8>,
    /// One streaming-IPC blob per partition (schema message + batches).
    pub partitions: Vec<Vec<u8>>,
}

/// The query body: either a serialized unoptimized logical plan or a textual
/// query resolved against server-side catalogs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryPayload {
    /// Binary-serialized unoptimized logical plan. Requires identical engine
    /// versions on client and server.
    Plan(Vec<u8>),
    /// Textual query planned and optimized entirely server-side.
    Sql(String),
}

/// A complete query request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    /// Client-generated identifier; used for cross-connection cancellation
    /// and idempotent resubmission.
    pub query_id: String,
    /// The query itself.
    pub payload: QueryPayload,
    /// Binary-serialized execution configuration of the client context, if
    /// the client wants to override the server defaults.
    pub exec_config: Option<Vec<u8>>,
    /// In-memory partition sets referenced by the plan.
    pub psets: Vec<NamedPartitionSet>,
    /// Client hint bounding buffered, not-yet-consumed results.
    pub results_buffer_size: Option<usize>,
}

/// Execution summary appended as metadata on the final message of a result
/// stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryStatsWire {
    /// Identifier of the completed query.
    pub query_id: String,
    /// Total rows streamed back.
    pub rows: u64,
    /// Total serialized bytes streamed back.
    pub bytes: u64,
    /// JSON rendering of the optimized physical plan, when available.
    pub physical_plan_json: Option<String>,
    /// Binary-serialized execution statistics.
    pub stats: Vec<u8>,
}

/// Server description returned by the `server_info` action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    /// Engine version of the serving process.
    pub version: String,
    /// Envelope version the server speaks.
    pub wire_version: u16,
    /// Whether serialized-plan payloads are accepted.
    pub plan_payload_enabled: bool,
    /// Maximum concurrently executing queries.
    pub max_concurrent_queries: usize,
    /// Cap on total in-memory partition bytes shipped with one query.
    pub max_pset_bytes: usize,
    /// Names of catalogs attached to the server session.
    pub catalogs: Vec<String>,
}

/// Explain output returned by the `explain` action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplainResult {
    /// Human-readable rendering of the optimized plan.
    pub optimized_plan: String,
}

/// Action verbs accepted by the server outside of query execution.
pub mod actions {
    /// Liveness probe; returns an empty body.
    pub const HEALTH: &str = "health";
    /// Returns a serialized [`super::ServerInfo`].
    pub const SERVER_INFO: &str = "server_info";
    /// Cancels a running query; body is the query id bytes.
    pub const CANCEL_QUERY: &str = "cancel_query";
    /// Returns a serialized [`super::ExplainResult`] for a request body.
    pub const EXPLAIN: &str = "explain";
}

/// Serializes a value with the engine's canonical binary encoding.
///
/// # Errors
/// Returns [`ServeError::MalformedPayload`] if the value cannot be encoded.
pub fn encode<T: Serialize>(value: &T) -> ServeResult<Vec<u8>> {
    common_py_serde::bincode::serde::encode_to_vec(
        value,
        common_py_serde::bincode::config::legacy(),
    )
    .map_err(|e| ServeError::MalformedPayload(format!("encode failed: {e}")))
}

/// Deserializes a value produced by [`encode`].
///
/// # Errors
/// Returns [`ServeError::MalformedPayload`] if the bytes do not decode.
pub fn decode<T: for<'de> Deserialize<'de>>(bytes: &[u8]) -> ServeResult<T> {
    common_py_serde::bincode::serde::decode_from_slice(
        bytes,
        common_py_serde::bincode::config::legacy(),
    )
    .map(|(value, _)| value)
    .map_err(|e| ServeError::MalformedPayload(format!("decode failed: {e}")))
}

/// Builds a complete request envelope: header plus encoded body.
///
/// # Errors
/// Returns [`ServeError::MalformedPayload`] if the body cannot be encoded, or
/// [`ServeError::InvalidEnvelope`] if `sender_version` exceeds the header
/// length bound.
pub fn encode_envelope(sender_version: &str, request: &QueryRequest) -> ServeResult<Vec<u8>> {
    let version_bytes = sender_version.as_bytes();
    if version_bytes.len() > MAX_VERSION_LEN {
        return Err(ServeError::InvalidEnvelope(format!(
            "version string too long: {} bytes",
            version_bytes.len()
        )));
    }
    let version_len = u16::try_from(version_bytes.len()).map_err(|_| {
        ServeError::InvalidEnvelope(format!(
            "version string too long: {} bytes",
            version_bytes.len()
        ))
    })?;
    let body = encode(request)?;
    let mut out = Vec::with_capacity(4 + 2 + 2 + version_bytes.len() + body.len());
    out.extend_from_slice(&WIRE_MAGIC);
    out.extend_from_slice(&WIRE_VERSION.to_le_bytes());
    out.extend_from_slice(&version_len.to_le_bytes());
    out.extend_from_slice(version_bytes);
    out.extend_from_slice(&body);
    Ok(out)
}

/// A parsed envelope header plus the undecoded body.
#[derive(Debug)]
pub struct Envelope<'a> {
    /// Envelope layout version declared by the sender.
    pub wire_version: u16,
    /// Engine version string of the sender.
    pub sender_version: String,
    /// Undecoded request body.
    pub body: &'a [u8],
}

/// Parses the fixed envelope header, returning the sender version and body.
///
/// This function only performs fixed-layout reads and is stable across
/// releases; it never attempts to decode the body.
///
/// # Errors
/// Returns [`ServeError::InvalidEnvelope`] on bad magic, truncation, an
/// unsupported wire version, or a non-UTF-8 version string.
pub fn parse_envelope(bytes: &[u8]) -> ServeResult<Envelope<'_>> {
    if bytes.len() < 8 {
        return Err(ServeError::InvalidEnvelope(format!(
            "request too short: {} bytes",
            bytes.len()
        )));
    }
    if bytes[0..4] != WIRE_MAGIC {
        return Err(ServeError::InvalidEnvelope(
            "bad magic; not a serve-protocol request".to_string(),
        ));
    }
    let wire_version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if wire_version != WIRE_VERSION {
        return Err(ServeError::InvalidEnvelope(format!(
            "unsupported wire version {wire_version}; this server speaks {WIRE_VERSION}"
        )));
    }
    let version_len = usize::from(u16::from_le_bytes([bytes[6], bytes[7]]));
    if version_len > MAX_VERSION_LEN {
        return Err(ServeError::InvalidEnvelope(format!(
            "version length {version_len} exceeds bound {MAX_VERSION_LEN}"
        )));
    }
    let body_start = 8 + version_len;
    if bytes.len() < body_start {
        return Err(ServeError::InvalidEnvelope(
            "truncated header: version string incomplete".to_string(),
        ));
    }
    let sender_version = std::str::from_utf8(&bytes[8..body_start])
        .map_err(|e| ServeError::InvalidEnvelope(format!("version not UTF-8: {e}")))?
        .to_string();
    Ok(Envelope {
        wire_version,
        sender_version,
        body: &bytes[body_start..],
    })
}

/// Computes the total serialized size of all shipped partitions.
#[must_use]
pub fn total_pset_bytes(psets: &[NamedPartitionSet]) -> usize {
    psets
        .iter()
        .map(|p| p.partitions.iter().map(Vec::len).sum::<usize>())
        .sum()
}

/// Enforces per-request policy after envelope parsing.
///
/// Checks plan-payload availability, exact version match for plan payloads,
/// and the shipped in-memory data cap. Textual payloads tolerate version
/// differences.
///
/// # Errors
/// Returns the specific policy violation so clients receive an actionable
/// error class.
pub fn check_request_policy(
    request: &QueryRequest,
    sender_version: &str,
    server_version: &str,
    disable_plan_payload: bool,
    max_pset_bytes: usize,
) -> ServeResult<()> {
    if matches!(request.payload, QueryPayload::Plan(_)) {
        if disable_plan_payload {
            return Err(ServeError::PlanPayloadDisabled);
        }
        if sender_version != server_version {
            return Err(ServeError::VersionMismatch {
                client: sender_version.to_string(),
                server: server_version.to_string(),
            });
        }
    }
    let pset_bytes = total_pset_bytes(&request.psets);
    if pset_bytes > max_pset_bytes {
        return Err(ServeError::PsetTooLarge {
            actual_bytes: pset_bytes,
            max_bytes: max_pset_bytes,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> QueryRequest {
        QueryRequest {
            query_id: "q-1".to_string(),
            payload: QueryPayload::Sql("select 1".to_string()),
            exec_config: None,
            psets: vec![],
            results_buffer_size: Some(4),
        }
    }

    #[test]
    fn envelope_round_trips() {
        let encoded = encode_envelope("1.2.3", &request()).unwrap();
        let envelope = parse_envelope(&encoded).unwrap();
        assert_eq!(envelope.wire_version, WIRE_VERSION);
        assert_eq!(envelope.sender_version, "1.2.3");
        let decoded: QueryRequest = decode(envelope.body).unwrap();
        assert_eq!(decoded.query_id, "q-1");
        match decoded.payload {
            QueryPayload::Sql(sql) => assert_eq!(sql, "select 1"),
            QueryPayload::Plan(_) => panic!("expected sql payload"),
        }
        assert_eq!(decoded.results_buffer_size, Some(4));
    }

    #[test]
    fn empty_request_is_rejected() {
        let err = parse_envelope(&[]).unwrap_err();
        assert!(matches!(err, ServeError::InvalidEnvelope(_)));
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut encoded = encode_envelope("1.2.3", &request()).unwrap();
        encoded[0] = b'X';
        let err = parse_envelope(&encoded).unwrap_err();
        assert!(matches!(err, ServeError::InvalidEnvelope(_)));
    }

    #[test]
    fn unsupported_wire_version_is_rejected() {
        let mut encoded = encode_envelope("1.2.3", &request()).unwrap();
        encoded[4] = 0xFF;
        encoded[5] = 0xFF;
        let err = parse_envelope(&encoded).unwrap_err();
        assert!(matches!(err, ServeError::InvalidEnvelope(_)));
    }

    #[test]
    fn truncated_version_string_is_rejected() {
        let encoded = encode_envelope("1.2.3", &request()).unwrap();
        let err = parse_envelope(&encoded[..9]).unwrap_err();
        assert!(matches!(err, ServeError::InvalidEnvelope(_)));
    }

    #[test]
    fn truncated_body_fails_decode_not_parse() {
        let encoded = encode_envelope("1.2.3", &request()).unwrap();
        let cut = encoded.len() - 3;
        let envelope = parse_envelope(&encoded[..cut]).unwrap();
        assert_eq!(envelope.sender_version, "1.2.3");
        let err = decode::<QueryRequest>(envelope.body).unwrap_err();
        assert!(matches!(err, ServeError::MalformedPayload(_)));
    }

    #[test]
    fn oversized_version_string_is_rejected_at_encode() {
        let long = "v".repeat(MAX_VERSION_LEN + 1);
        let err = encode_envelope(&long, &request()).unwrap_err();
        assert!(matches!(err, ServeError::InvalidEnvelope(_)));
    }

    #[test]
    fn pset_byte_accounting_sums_all_partitions() {
        let psets = vec![
            NamedPartitionSet {
                key: "a".to_string(),
                schema: vec![0; 10],
                partitions: vec![vec![0; 100], vec![0; 50]],
            },
            NamedPartitionSet {
                key: "b".to_string(),
                schema: vec![],
                partitions: vec![vec![0; 25]],
            },
        ];
        assert_eq!(total_pset_bytes(&psets), 175);
    }

    fn plan_request(pset_bytes: usize) -> QueryRequest {
        QueryRequest {
            query_id: "q".to_string(),
            payload: QueryPayload::Plan(vec![1, 2, 3]),
            exec_config: None,
            psets: vec![NamedPartitionSet {
                key: "k".to_string(),
                schema: vec![],
                partitions: vec![vec![0; pset_bytes]],
            }],
            results_buffer_size: None,
        }
    }

    #[test]
    fn plan_payload_with_matching_version_is_admitted() {
        assert!(check_request_policy(&plan_request(1), "1.0.0", "1.0.0", false, 1024).is_ok());
    }

    #[test]
    fn plan_payload_with_version_mismatch_is_rejected() {
        let err =
            check_request_policy(&plan_request(1), "1.0.0", "1.0.1", false, 1024).unwrap_err();
        assert!(matches!(err, ServeError::VersionMismatch { .. }));
    }

    #[test]
    fn sql_payload_tolerates_version_mismatch() {
        assert!(check_request_policy(&request(), "1.0.0", "9.9.9", false, 1024).is_ok());
    }

    #[test]
    fn plan_payload_is_rejected_when_disabled() {
        let err = check_request_policy(&plan_request(1), "1.0.0", "1.0.0", true, 1024).unwrap_err();
        assert!(matches!(err, ServeError::PlanPayloadDisabled));
    }

    #[test]
    fn sql_payload_is_admitted_when_plans_disabled() {
        assert!(check_request_policy(&request(), "1.0.0", "9.9.9", true, 1024).is_ok());
    }

    #[test]
    fn oversized_pset_is_rejected() {
        let err =
            check_request_policy(&plan_request(2048), "1.0.0", "1.0.0", false, 1024).unwrap_err();
        assert!(matches!(
            err,
            ServeError::PsetTooLarge {
                actual_bytes: 2048,
                max_bytes: 1024,
            }
        ));
    }

    #[test]
    fn plan_payload_round_trips() {
        let req = QueryRequest {
            query_id: "q-2".to_string(),
            payload: QueryPayload::Plan(vec![1, 2, 3]),
            exec_config: Some(vec![9, 9]),
            psets: vec![NamedPartitionSet {
                key: "k".to_string(),
                schema: vec![7],
                partitions: vec![vec![8]],
            }],
            results_buffer_size: None,
        };
        let encoded = encode_envelope("0.0.0+local", &req).unwrap();
        let envelope = parse_envelope(&encoded).unwrap();
        let decoded: QueryRequest = decode(envelope.body).unwrap();
        match decoded.payload {
            QueryPayload::Plan(bytes) => assert_eq!(bytes, vec![1, 2, 3]),
            QueryPayload::Sql(_) => panic!("expected plan payload"),
        }
        assert_eq!(decoded.exec_config.as_deref(), Some(&[9u8, 9][..]));
        assert_eq!(decoded.psets.len(), 1);
    }
}
