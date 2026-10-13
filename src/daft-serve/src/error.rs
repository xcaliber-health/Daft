//! Error types for the query-serving protocol.
//!
//! Errors are split into a typed enum so that the transport layer can map each
//! variant onto a distinct wire status code, and clients can re-raise the
//! matching exception class. Every failure also travels as a [`Failure`]: where
//! it was raised, whether it may be retried, and, for a Python failure, the
//! names of the raised type.

use common_error::DaftError;
use serde::{Deserialize, Serialize};

/// Metadata key carrying the error class of a failed query so clients can
/// re-raise a faithful exception type.
pub const ERROR_CLASS_METADATA_KEY: &str = "x-daft-error-class";

/// Binary metadata key carrying a failure's [`Failure`] record.
pub const FAILURE_METADATA_KEY: &str = "x-daft-failure-bin";

/// What a client needs to raise a failure as the class it is raised as where
/// the query ran; its message travels as the status message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Failure {
    /// The engine class's name, `Host`, `Python`, or the server's own class.
    pub error_class: String,
    /// Whether the engine raised it, so `error_class` names one of its classes.
    pub engine: bool,
    /// Whether the same query may succeed if sent again later.
    pub retryable: bool,
    /// The raised Python type, for a `Host` or `Python` failure.
    pub python: Option<PythonFailure>,
}

/// The raised type of a Python failure, by name only: a client never imports it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PythonFailure {
    /// Name of the raised type.
    pub remote_type: String,
    /// Module of the raised type.
    pub remote_module: String,
    /// Names of the type's hierarchy, nearest first.
    pub remote_types: Vec<String>,
    /// Whether a user-defined function raised it.
    pub udf: bool,
    /// Named text fields the exception carried, within their bounds. Sent to
    /// the client only: never logged, and never put in an event.
    pub fields: Vec<(String, String)>,
}

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

    /// A host request's body exceeded the server's configured cap.
    #[error(
        "request body of {actual_bytes} bytes exceeds the server cap of \
         {max_bytes} bytes; send a smaller request or raise the server's \
         `max_request_body_bytes` limit"
    )]
    RequestTooLarge {
        /// Size of the request body.
        actual_bytes: usize,
        /// Server-side cap on request body bytes.
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

    /// A failure classified for its client: an engine class, a Python failure
    /// raised by the host or during execution, or one a server reported.
    #[error("{message}")]
    Failed {
        /// Where it was raised and what it was.
        failure: Box<Failure>,
        /// The message it is raised with.
        message: String,
    },
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
            Self::RequestTooLarge { .. } => "RequestTooLarge",
            Self::AtCapacity { .. } => "AtCapacity",
            Self::Cancelled(_) => "Cancelled",
            Self::Timeout { .. } => "Timeout",
            Self::Draining => "Draining",
            Self::Execution(_) => "Execution",
            Self::Failed { .. } => "Failed",
        }
    }

    /// Whether the same query may succeed if sent again later: at capacity,
    /// timed out, or turned away while the server drains.
    #[must_use]
    pub fn retryable(&self) -> bool {
        match self {
            Self::AtCapacity { .. } | Self::Timeout { .. } | Self::Draining => true,
            Self::Failed { failure, .. } => failure.retryable,
            _ => false,
        }
    }

    /// The failure as a client raises it. An unclassified engine error names
    /// no class, and is raised as the base class.
    #[must_use]
    pub fn failure(&self) -> Failure {
        match self {
            Self::Failed { failure, .. } => failure.as_ref().clone(),
            Self::Execution(_) => Failure {
                error_class: "DaftCoreException".to_string(),
                engine: true,
                retryable: false,
                python: None,
            },
            other => Failure {
                error_class: other.class().to_string(),
                engine: false,
                retryable: other.retryable(),
                python: None,
            },
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
            ServeError::PsetTooLarge { .. } | ServeError::RequestTooLarge { .. } => {
                Self::resource_exhausted(err.to_string())
            }
            ServeError::AtCapacity { .. } => Self::resource_exhausted(err.to_string()),
            ServeError::Cancelled(_) => Self::cancelled(err.to_string()),
            ServeError::Timeout { .. } => Self::deadline_exceeded(err.to_string()),
            ServeError::Draining => Self::unavailable(err.to_string()),
            ServeError::Execution(_) | ServeError::Failed { .. } => Self::internal(err.to_string()),
        };
        let failure = err.failure();
        let class = if matches!(err, ServeError::Failed { .. }) {
            failure.error_class.as_str()
        } else {
            class
        };
        if let Ok(value) = class.parse() {
            status
                .metadata_mut()
                .insert(ERROR_CLASS_METADATA_KEY, value);
        }
        if let Ok(bytes) = crate::wire::encode(&failure) {
            status.metadata_mut().insert_bin(
                FAILURE_METADATA_KEY,
                tonic::metadata::MetadataValue::from_bytes(&bytes),
            );
        }
        status
    }
}

/// Rebuilds the failure a status reports.
///
/// A status from a server that sends no [`Failure`] record, or from the
/// transport itself, is a failure of the class its metadata names, or
/// `Transport`, raised as the base class.
#[must_use]
pub fn failure_of_status(status: &tonic::Status) -> ServeError {
    let failure = status
        .metadata()
        .get_bin(FAILURE_METADATA_KEY)
        .and_then(|value| value.to_bytes().ok())
        .and_then(|bytes| crate::wire::decode::<Failure>(&bytes).ok())
        .unwrap_or_else(|| Failure {
            error_class: status
                .metadata()
                .get(ERROR_CLASS_METADATA_KEY)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("Transport")
                .to_string(),
            engine: false,
            retryable: false,
            python: None,
        });
    ServeError::Failed {
        failure: Box::new(failure),
        message: status.message().to_string(),
    }
}

impl From<ServeError> for DaftError {
    fn from(err: ServeError) -> Self {
        match err {
            ServeError::Execution(inner) => inner,
            ServeError::Failed { failure, message } => {
                Self::External(format!("[{}] {message}", failure.error_class).into())
            }
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
                ServeError::RequestTooLarge {
                    actual_bytes: 2,
                    max_bytes: 1,
                },
                "RequestTooLarge",
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
    fn retryable_failures_are_capacity_timeout_and_draining() {
        let retryable: Vec<bool> = [
            ServeError::AtCapacity {
                max_concurrent: 1,
                waited_secs: 1,
            },
            ServeError::Timeout {
                query_id: "q".into(),
                limit_secs: 1,
            },
            ServeError::Draining,
            ServeError::Cancelled("q".into()),
            ServeError::Unauthenticated("x".into()),
            ServeError::PlanPayloadDisabled,
        ]
        .iter()
        .map(ServeError::retryable)
        .collect();

        assert_eq!(retryable, [true, true, true, false, false, false]);
    }

    #[test]
    fn a_failure_crosses_a_status_whole() {
        let sent = Failure {
            error_class: "Host".into(),
            engine: false,
            retryable: false,
            python: Some(PythonFailure {
                remote_type: "Refused".into(),
                remote_module: "host".into(),
                remote_types: vec!["Refused".into(), "Exception".into()],
                udf: false,
                fields: vec![("status".into(), "403".into())],
            }),
        };
        let status: tonic::Status = ServeError::Failed {
            failure: Box::new(sent.clone()),
            message: "Refused: no".into(),
        }
        .into();

        let ServeError::Failed { failure, message } = failure_of_status(&status) else {
            panic!("a status rebuilds as a failure");
        };

        assert_eq!((*failure, message.as_str()), (sent, "Refused: no"));
    }

    #[test]
    fn a_server_failure_crosses_a_status_by_its_class() {
        let status: tonic::Status = ServeError::Draining.into();

        let ServeError::Failed { failure, .. } = failure_of_status(&status) else {
            panic!("a status rebuilds as a failure");
        };

        assert_eq!(
            (failure.error_class.as_str(), failure.retryable),
            ("Draining", true)
        );
    }

    #[test]
    fn a_status_without_a_failure_record_names_its_class_or_the_transport() {
        let bare = tonic::Status::unavailable("connection refused");

        let ServeError::Failed { failure, .. } = failure_of_status(&bare) else {
            panic!("a status rebuilds as a failure");
        };

        assert_eq!(
            (failure.error_class.as_str(), failure.engine),
            ("Transport", false)
        );
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
            (
                ServeError::RequestTooLarge {
                    actual_bytes: 2,
                    max_bytes: 1,
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
