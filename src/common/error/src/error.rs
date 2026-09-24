use thiserror::Error;

pub type DaftResult<T> = std::result::Result<T, DaftError>;
pub type GenericError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, Error)]
pub enum DaftError {
    #[error("DaftError::AmbiguousReference {0}")]
    AmbiguousReference(String),
    #[error("DaftError::FieldNotFound {0}")]
    FieldNotFound(String),
    #[error("DaftError::SchemaMismatch {0}")]
    SchemaMismatch(String),
    #[error("DaftError::TypeError {0}")]
    TypeError(String),
    #[error("DaftError::ComputeError {0}")]
    ComputeError(String),
    /// A value that must come from exactly one row came from several, such as a
    /// group's single value.
    #[error("DaftError::CardinalityViolation {0}")]
    CardinalityViolation(String),
    #[error("DaftError::ArrowRsError {0}")]
    ArrowRsError(#[from] arrow_schema::ArrowError),
    // TODO(desmond): We can't currently implement this as a From<parquet::errors::ParquetError>
    // because this results in infinite nesting of types in `fixed_size_binary_op` in arithmetic.rs.
    #[error("DaftError::ParquetError {0}")]
    ParquetError(String),
    /// Raised when a file is identified as corrupt or unreadable due to format/integrity
    /// failures (e.g. bad magic bytes, truncated footer, bad encoding, wrong field counts).
    /// Used by `is_parquet_corrupt` and `is_csv_corrupt` to identify files that should be
    /// skipped when `ignore_corrupt_files` is enabled.
    /// General operation errors (write failures, schema mismatches, etc.) are NOT routed
    /// here — they use format-specific variants or `External`.
    #[error("DaftError::CorruptFile {0}")]
    CorruptFile(String),
    #[error("DaftError::ValueError {0}")]
    ValueError(String),
    #[cfg(feature = "python")]
    #[error("DaftError::PyO3Error {0}")]
    PyO3Error(#[from] pyo3::PyErr),
    #[error("DaftError::IoError {0}")]
    IoError(#[from] std::io::Error),
    #[error("DaftError::FileNotFound {path} not found: {source}")]
    FileNotFound { path: String, source: GenericError },
    #[error("DaftError::InternalError {0}")]
    InternalError(String),
    #[error("ConnectTimeout {0}")]
    ConnectTimeout(#[source] GenericError),
    #[error("ReadTimeout {0}")]
    ReadTimeout(#[source] GenericError),
    #[error("ByteStreamError {0}")]
    ByteStreamError(#[source] GenericError),
    #[error("SocketError {0}")]
    SocketError(#[source] GenericError),
    #[error("ThrottledIo {0}")]
    ThrottledIo(#[source] GenericError),
    #[error("MiscTransient {0}")]
    MiscTransient(#[source] GenericError),
    #[error("DaftError::External {0}")]
    External(#[source] GenericError),
    #[error("DaftError::SerdeJsonError {0}")]
    SerdeJsonError(#[from] serde_json::Error),
    #[error("DaftError::FmtError {0}")]
    FmtError(#[from] std::fmt::Error),
    #[error("DaftError::RegexError {0}")]
    RegexError(#[from] regex::Error),
    #[error("DaftError::FromUtf8Error {0}")]
    FromUtf8Error(#[from] std::string::FromUtf8Error),
    #[error("Not Yet Implemented: {0}")]
    NotImplemented(String),
    #[error("DaftError::CatalogError {0}")]
    CatalogError(String),
    #[error("DaftError::JoinError {0}")]
    JoinError(#[from] tokio::task::JoinError),
    #[error("DaftError::InvalidArgumentError {0}")]
    InvalidArgumentError(String),
}

impl DaftError {
    /// Returns the innermost of Daft's own errors this one carries, or itself.
    ///
    /// Errors raised while planning or running arrive wrapped in the context they
    /// passed through; the kind of the failure is that of the error at the bottom.
    #[must_use]
    pub fn innermost(&self) -> &Self {
        let mut innermost = self;
        let mut next = std::error::Error::source(self);
        while let Some(cause) = next {
            if let Some(daft_error) = cause.downcast_ref::<Self>() {
                innermost = daft_error;
            }
            next = cause.source();
        }
        innermost
    }

    pub fn not_implemented<T: std::fmt::Display>(msg: T) -> Self {
        Self::NotImplemented(msg.to_string())
    }
    pub fn type_error<T: std::fmt::Display>(msg: T) -> Self {
        Self::TypeError(msg.to_string())
    }
}

#[macro_export]
macro_rules! ensure {
    ($cond:expr, $msg:expr) => {
        if !$cond {
            return Err($crate::DaftError::ComputeError($msg.to_string()));
        }
    };
    ($cond:expr, $variant:ident: $($msg:tt)*) => {
        if !$cond {
            return Err($crate::DaftError::$variant(format!($($msg)*)));
        }
    };
}

#[macro_export]
macro_rules! value_err {
    ($($arg:tt)*) => {
        return Err(common_error::DaftError::ValueError(format!($($arg)*)))
    };
}

#[cfg(feature = "python")]
impl<'py> From<pyo3::pyclass::PyClassGuardError<'_, 'py>> for DaftError {
    fn from(error: pyo3::pyclass::PyClassGuardError<'_, 'py>) -> Self {
        Self::PyO3Error(error.into())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[derive(Debug, Error)]
    #[error("could not build the plan")]
    struct Context {
        #[source]
        source: DaftError,
    }

    fn wrapped(kind: DaftError) -> DaftError {
        DaftError::External(Box::new(Context { source: kind }))
    }

    #[rstest]
    #[case::bare(DaftError::FieldNotFound("a".into()), "FieldNotFound")]
    #[case::wrapped(wrapped(DaftError::FieldNotFound("a".into())), "FieldNotFound")]
    #[case::wrapped_twice(wrapped(wrapped(DaftError::TypeError("t".into()))), "TypeError")]
    #[case::wrapping_nothing_of_ours(DaftError::External("io".into()), "External")]
    fn the_kind_is_that_of_the_innermost_error(#[case] err: DaftError, #[case] kind: &str) {
        assert!(format!("{:?}", err.innermost()).starts_with(kind));
    }
}
