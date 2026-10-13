//! The Python exception each engine error is raised as.
//!
//! One table names the class an error is raised as, and one builds an
//! exception of a named class, so an error raised in another process can be
//! rebuilt as the class it is raised as here.

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

/// Name of the class raised for an error that names no narrower kind.
pub const BASE_EXCEPTION_CLASS: &str = "DaftCoreException";

impl std::convert::From<DaftError> for pyo3::PyErr {
    fn from(err: DaftError) -> Self {
        match err {
            DaftError::PyO3Error(pyerr) => pyerr,
            other => {
                let class = exception_class_name(&other);
                engine_exception(class, user_message(&other))
            }
        }
    }
}

/// Returns the name of the class `err` is raised as in Python.
///
/// Every name is that of a class in `daft.exceptions`, except
/// `FileNotFoundError`. An error carrying a Python exception has no name
/// here: it is raised as that exception, and this answers the base class.
#[must_use]
pub fn exception_class_name(err: &DaftError) -> &'static str {
    match err {
        DaftError::TypeError(_) => "DaftTypeError",
        DaftError::FileNotFound { .. } => "FileNotFoundError",
        DaftError::ConnectTimeout(_) => "ConnectTimeoutError",
        DaftError::ReadTimeout(_) => "ReadTimeoutError",
        DaftError::ByteStreamError(_) => "ByteStreamError",
        DaftError::SocketError(_) => "SocketError",
        DaftError::ThrottledIo(_) => "ThrottleError",
        DaftError::MiscTransient(_) => "MiscTransientError",
        other => match other.innermost() {
            DaftError::FieldNotFound(_) => "DaftFieldNotFoundError",
            DaftError::AmbiguousReference(_) => "DaftAmbiguousReferenceError",
            DaftError::SchemaMismatch(_) => "DaftSchemaMismatchError",
            DaftError::TypeError(_) => "DaftTypeError",
            DaftError::ValueError(_) => "DaftValueError",
            DaftError::ComputeError(_) => "DaftComputeError",
            DaftError::CardinalityViolation(_) => "DaftCardinalityError",
            DaftError::NotImplemented(_) => "DaftNotImplementedError",
            DaftError::InvalidArgumentError(_) => "DaftInvalidArgumentError",
            _ => BASE_EXCEPTION_CLASS,
        },
    }
}

/// Returns the message `err` is raised with in Python: its whole chain of
/// causes, without repeated text. A type error is raised with its own text.
#[must_use]
pub fn user_message(err: &DaftError) -> String {
    match err {
        DaftError::TypeError(msg) => msg.clone(),
        other => format_error_for_user(other),
    }
}

/// Builds an exception of the engine class named `class`, carrying `message`.
///
/// `class` is one of the names [`exception_class_name`] answers; any other
/// name builds the base class, so no name can make this construct a class
/// outside that set. Every class but `FileNotFoundError` is a
/// `DaftCoreException`, so a caller catching that still catches all.
#[must_use]
pub fn engine_exception(class: &str, message: String) -> pyo3::PyErr {
    match class {
        "DaftTypeError" => DaftTypeError::new_err(message),
        "FileNotFoundError" => PyFileNotFoundError::new_err(message),
        "ConnectTimeoutError" => ConnectTimeoutError::new_err(message),
        "ReadTimeoutError" => ReadTimeoutError::new_err(message),
        "ByteStreamError" => ByteStreamError::new_err(message),
        "SocketError" => SocketError::new_err(message),
        "ThrottleError" => ThrottleError::new_err(message),
        "MiscTransientError" => MiscTransientError::new_err(message),
        "DaftFieldNotFoundError" => DaftFieldNotFoundError::new_err(message),
        "DaftAmbiguousReferenceError" => DaftAmbiguousReferenceError::new_err(message),
        "DaftSchemaMismatchError" => DaftSchemaMismatchError::new_err(message),
        "DaftValueError" => DaftValueError::new_err(message),
        "DaftComputeError" => DaftComputeError::new_err(message),
        "DaftCardinalityError" => DaftCardinalityError::new_err(message),
        "DaftNotImplementedError" => DaftNotImplementedError::new_err(message),
        "DaftInvalidArgumentError" => DaftInvalidArgumentError::new_err(message),
        _ => DaftCoreException::new_err(message),
    }
}
