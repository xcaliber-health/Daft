//! Error types for the query-serving protocol.
//!
//! Errors are split into a typed enum so that the transport layer can map each
//! variant onto a distinct wire status code, and clients can re-raise the
//! matching exception class.

use common_error::DaftError;

/// Metadata key carrying the error class of a failed query so clients can
/// re-raise a faithful exception type.
pub const ERROR_CLASS_METADATA_KEY: &str = "x-daft-error-class";

/// Errors produced while parsing, admitting, or executing a remote query.
#[derive(Debug, thiserror::Error)]
pub enum ServeError {
    /// The request envelope could not be parsed (bad magic, truncation, or an
    /// unsupported wire version).
    #[error("invalid request envelope: {0}")]
    InvalidEnvelope(String),

    /// The client and server versions do not match, which is required for
    /// serialized-plan payloads.
    #[error(
        "version mismatch: client is `{client}`, server is `{server}`. \
         Serialized plans require identical versions on both ends; \
         upgrade the client or server so the versions match"
    )]
    VersionMismatch {
        /// Version string reported by the client.
        client: String,
        /// Version string of the serving process.
        server: String,
    },

    /// The request payload failed to deserialize after envelope validation.
    #[error("malformed request payload: {0}")]
    MalformedPayload(String),

    /// The request was rejected by the authentication layer.
    #[error("unauthenticated: {0}")]
    Unauthenticated(String),

    /// Plan payloads are disabled by server policy; only textual queries are
    /// accepted.
    #[error(
        "serialized plan payloads are disabled on this server; \
         submit the query as text instead"
    )]
    PlanPayloadDisabled,

    /// The embedded in-memory data exceeded the server's configured cap.
    #[error(
        "in-memory data of {actual_bytes} bytes exceeds the server cap of \
         {max_bytes} bytes; reduce the data shipped with the query or raise \
         the server's `max_pset_bytes` limit"
    )]
    PsetTooLarge {
        /// Total serialized size of the shipped partitions.
        actual_bytes: usize,
        /// Server-side cap on shipped partition bytes.
        max_bytes: usize,
    },

    /// The server is at its concurrent-query capacity and the request timed
    /// out waiting for a slot.
    #[error(
        "server is at capacity ({max_concurrent} concurrent queries); \
         timed out after {waited_secs}s waiting for a slot"
    )]
    AtCapacity {
        /// Configured maximum number of concurrently executing queries.
        max_concurrent: usize,
        /// Seconds the request waited before giving up.
        waited_secs: u64,
    },

    /// The query was cancelled, either by an explicit cancel action or by the
    /// client disconnecting.
    #[error("query `{0}` was cancelled")]
    Cancelled(String),

    /// The query exceeded the server's per-query execution time limit and was
    /// cancelled.
    #[error(
        "query `{query_id}` exceeded the server's execution time limit of \
         {limit_secs}s and was cancelled"
    )]
    Timeout {
        /// Identifier of the timed-out query.
        query_id: String,
        /// Configured per-query execution limit in seconds.
        limit_secs: u64,
    },

    /// The server is draining ahead of shutdown and not accepting new queries.
    #[error("server is shutting down and not accepting new queries")]
    Draining,

    /// Execution of the query failed after admission.
    #[error(transparent)]
    Execution(#[from] DaftError),
}

impl ServeError {
    /// Stable, machine-readable class name for this error, transmitted to
    /// clients alongside the message.
    #[must_use]
    pub fn class(&self) -> &'static str {
        match self {
            Self::InvalidEnvelope(_) => "InvalidEnvelope",
            Self::VersionMismatch { .. } => "VersionMismatch",
            Self::MalformedPayload(_) => "MalformedPayload",
            Self::Unauthenticated(_) => "Unauthenticated",
            Self::PlanPayloadDisabled => "PlanPayloadDisabled",
            Self::PsetTooLarge { .. } => "PsetTooLarge",
            Self::AtCapacity { .. } => "AtCapacity",
            Self::Cancelled(_) => "Cancelled",
            Self::Timeout { .. } => "Timeout",
            Self::Draining => "Draining",
            Self::Execution(_) => "Execution",
        }
    }
}

impl From<ServeError> for tonic::Status {
    fn from(err: ServeError) -> Self {
        let class = err.class();
        let mut status = match &err {
            ServeError::InvalidEnvelope(_) | ServeError::MalformedPayload(_) => {
                Self::invalid_argument(err.to_string())
            }
            ServeError::VersionMismatch { .. } => Self::failed_precondition(err.to_string()),
            ServeError::Unauthenticated(_) => Self::unauthenticated(err.to_string()),
            ServeError::PlanPayloadDisabled => Self::permission_denied(err.to_string()),
            ServeError::PsetTooLarge { .. } => Self::resource_exhausted(err.to_string()),
            ServeError::AtCapacity { .. } => Self::resource_exhausted(err.to_string()),
            ServeError::Cancelled(_) => Self::cancelled(err.to_string()),
            ServeError::Timeout { .. } => Self::deadline_exceeded(err.to_string()),
            ServeError::Draining => Self::unavailable(err.to_string()),
            ServeError::Execution(_) => Self::internal(err.to_string()),
        };
        if let Ok(value) = class.parse() {
            status
                .metadata_mut()
                .insert(ERROR_CLASS_METADATA_KEY, value);
        }
        status
    }
}

impl From<ServeError> for DaftError {
    fn from(err: ServeError) -> Self {
        match err {
            ServeError::Execution(inner) => inner,
            other => Self::External(format!("[{}] {other}", other.class()).into()),
        }
    }
}

/// Convenience alias for results in this crate.
pub type ServeResult<T> = Result<T, ServeError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_classes_are_stable() {
        let cases: Vec<(ServeError, &str)> = vec![
            (ServeError::InvalidEnvelope("x".into()), "InvalidEnvelope"),
            (
                ServeError::VersionMismatch {
                    client: "a".into(),
                    server: "b".into(),
                },
                "VersionMismatch",
            ),
            (ServeError::MalformedPayload("x".into()), "MalformedPayload"),
            (ServeError::Unauthenticated("x".into()), "Unauthenticated"),
            (ServeError::PlanPayloadDisabled, "PlanPayloadDisabled"),
            (
                ServeError::PsetTooLarge {
                    actual_bytes: 2,
                    max_bytes: 1,
                },
                "PsetTooLarge",
            ),
            (
                ServeError::AtCapacity {
                    max_concurrent: 4,
                    waited_secs: 5,
                },
                "AtCapacity",
            ),
            (ServeError::Cancelled("q".into()), "Cancelled"),
            (
                ServeError::Timeout {
                    query_id: "q".into(),
                    limit_secs: 1,
                },
                "Timeout",
            ),
            (ServeError::Draining, "Draining"),
        ];
        for (err, expected) in cases {
            assert_eq!(err.class(), expected);
        }
    }

    #[test]
    fn status_carries_error_class_metadata() {
        let status: tonic::Status = ServeError::Unauthenticated("bad token".into()).into();
        assert_eq!(status.code(), tonic::Code::Unauthenticated);
        assert_eq!(
            status
                .metadata()
                .get(ERROR_CLASS_METADATA_KEY)
                .and_then(|v| v.to_str().ok()),
            Some("Unauthenticated")
        );
    }

    #[test]
    fn status_codes_map_by_variant() {
        let cases: Vec<(ServeError, tonic::Code)> = vec![
            (
                ServeError::InvalidEnvelope("x".into()),
                tonic::Code::InvalidArgument,
            ),
            (
                ServeError::VersionMismatch {
                    client: "a".into(),
                    server: "b".into(),
                },
                tonic::Code::FailedPrecondition,
            ),
            (
                ServeError::PlanPayloadDisabled,
                tonic::Code::PermissionDenied,
            ),
            (
                ServeError::AtCapacity {
                    max_concurrent: 1,
                    waited_secs: 1,
                },
                tonic::Code::ResourceExhausted,
            ),
            (ServeError::Cancelled("q".into()), tonic::Code::Cancelled),
            (
                ServeError::Timeout {
                    query_id: "q".into(),
                    limit_secs: 1,
                },
                tonic::Code::DeadlineExceeded,
            ),
            (ServeError::Draining, tonic::Code::Unavailable),
        ];
        for (err, code) in cases {
            let status: tonic::Status = err.into();
            assert_eq!(status.code(), code);
        }
    }
}
