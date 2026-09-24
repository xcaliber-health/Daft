use pyo3::{exceptions::PyFileNotFoundError, import_exception};

use crate::{DaftError, format::format_error_for_user};

import_exception!(daft.exceptions, DaftCoreException);
import_exception!(daft.exceptions, DaftTypeError);
import_exception!(daft.exceptions, DaftFieldNotFoundError);
import_exception!(daft.exceptions, DaftAmbiguousReferenceError);
import_exception!(daft.exceptions, DaftSchemaMismatchError);
import_exception!(daft.exceptions, DaftValueError);
import_exception!(daft.exceptions, DaftComputeError);
import_exception!(daft.exceptions, DaftCardinalityError);
import_exception!(daft.exceptions, DaftNotImplementedError);
import_exception!(daft.exceptions, DaftInvalidArgumentError);
import_exception!(daft.exceptions, ConnectTimeoutError);
import_exception!(daft.exceptions, ReadTimeoutError);
import_exception!(daft.exceptions, ByteStreamError);
import_exception!(daft.exceptions, SocketError);
import_exception!(daft.exceptions, ThrottleError);
import_exception!(daft.exceptions, MiscTransientError);

impl std::convert::From<DaftError> for pyo3::PyErr {
    fn from(err: DaftError) -> Self {
        match err {
            DaftError::PyO3Error(pyerr) => pyerr,
            DaftError::TypeError(msg) => DaftTypeError::new_err(msg),
            other => {
                let formatted = format_error_for_user(&other);
                match other {
                    DaftError::FileNotFound { .. } => PyFileNotFoundError::new_err(formatted),
                    DaftError::ConnectTimeout(_) => ConnectTimeoutError::new_err(formatted),
                    DaftError::ReadTimeout(_) => ReadTimeoutError::new_err(formatted),
                    DaftError::ByteStreamError(_) => ByteStreamError::new_err(formatted),
                    DaftError::SocketError(_) => SocketError::new_err(formatted),
                    DaftError::ThrottledIo(_) => ThrottleError::new_err(formatted),
                    DaftError::MiscTransient(_) => MiscTransientError::new_err(formatted),
                    _ => error_of_kind(other.innermost(), formatted),
                }
            }
        }
    }
}

/// Builds the Python exception whose class names the kind of `kind`, carrying `message`.
///
/// Every class is a `DaftCoreException`, so a caller catching that still catches all.
fn error_of_kind(kind: &DaftError, message: String) -> pyo3::PyErr {
    match kind {
        DaftError::FieldNotFound(_) => DaftFieldNotFoundError::new_err(message),
        DaftError::AmbiguousReference(_) => DaftAmbiguousReferenceError::new_err(message),
        DaftError::SchemaMismatch(_) => DaftSchemaMismatchError::new_err(message),
        DaftError::TypeError(_) => DaftTypeError::new_err(message),
        DaftError::ValueError(_) => DaftValueError::new_err(message),
        DaftError::ComputeError(_) => DaftComputeError::new_err(message),
        DaftError::CardinalityViolation(_) => DaftCardinalityError::new_err(message),
        DaftError::NotImplemented(_) => DaftNotImplementedError::new_err(message),
        DaftError::InvalidArgumentError(_) => DaftInvalidArgumentError::new_err(message),
        _ => DaftCoreException::new_err(message),
    }
}
