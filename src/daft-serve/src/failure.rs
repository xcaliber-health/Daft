//! Classifying a served query's failure as the class it is raised as in-process.
//!
//! An engine error is named by the class it is raised as in Python. A Python
//! exception is classified by `daft._failures.failure_kind`, the same rule the
//! in-process runners report with: an engine class by its name, and any other
//! exception as a `Host` failure (raised by the host's `plan`) or a `Python`
//! failure (raised while the query executed), with the names of its type.

use common_error::{
    DaftError,
    python::{exception_class_name, user_message},
};
use pyo3::{prelude::*, types::PyDict};

use crate::error::{Failure, PythonFailure, ServeError};

/// When a Python exception was raised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Raised {
    /// By the request host's `plan`, before anything ran.
    ByHost,
    /// While the query was planned from its payload or executed.
    Running,
}

/// Classifies an unclassified engine error; any other error is returned as it is.
#[must_use]
pub fn classify(err: ServeError) -> ServeError {
    match err {
        ServeError::Execution(err) => of_engine_error(&err),
        other => other,
    }
}

/// Classifies an engine error. One carrying a Python exception, at its
/// surface or at the bottom of its causes, is classified by that exception.
#[must_use]
pub fn of_engine_error(err: &DaftError) -> ServeError {
    let raised = match err {
        DaftError::PyO3Error(raised) => Some(raised),
        other => match other.innermost() {
            DaftError::PyO3Error(raised) => Some(raised),
            _ => None,
        },
    };
    if let Some(raised) = raised {
        return Python::attach(|py| of_py_err(py, raised, Raised::Running));
    }
    ServeError::Failed {
        failure: Box::new(Failure {
            error_class: exception_class_name(err).to_string(),
            engine: true,
            retryable: false,
            python: None,
        }),
        message: user_message(err),
    }
}

/// Classifies a Python exception raised as `raised` says.
///
/// Should the classification itself fail, the exception is still reported,
/// as a Python failure of no named type.
#[must_use]
pub fn of_py_err(py: Python<'_>, err: &PyErr, raised: Raised) -> ServeError {
    let value = err.value(py);
    let text = value
        .str()
        .map_or_else(|_| String::new(), |text| text.to_string());
    let failure = kind_of(py, err, raised).unwrap_or_else(|classifying| {
        log::warn!("could not classify a served failure: {classifying}");
        Failure {
            error_class: python_class(raised).to_string(),
            engine: false,
            retryable: false,
            python: None,
        }
    });
    let message = match &failure.python {
        Some(python) if !python.udf => format!("{}: {text}", python.remote_type),
        _ => text,
    };
    ServeError::Failed {
        failure: Box::new(failure),
        message,
    }
}

fn python_class(raised: Raised) -> &'static str {
    match raised {
        Raised::ByHost => "Host",
        Raised::Running => "Python",
    }
}

fn kind_of(py: Python<'_>, err: &PyErr, raised: Raised) -> PyResult<Failure> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("raised_by_host", raised == Raised::ByHost)?;
    let kind = py
        .import(pyo3::intern!(py, "daft._failures"))?
        .getattr(pyo3::intern!(py, "failure_kind"))?
        .call((err.value(py),), Some(&kwargs))?;
    let error_class: String = kind.getattr(pyo3::intern!(py, "error_class"))?.extract()?;
    let remote_type: Option<String> = kind.getattr(pyo3::intern!(py, "remote_type"))?.extract()?;
    let Some(remote_type) = remote_type else {
        return Ok(Failure {
            error_class,
            engine: true,
            retryable: false,
            python: None,
        });
    };
    let mut fields: Vec<(String, String)> = kind
        .getattr(pyo3::intern!(py, "fields"))?
        .extract::<std::collections::HashMap<String, String>>()?
        .into_iter()
        .collect();
    fields.sort_unstable();
    Ok(Failure {
        error_class,
        engine: false,
        retryable: false,
        python: Some(PythonFailure {
            remote_type,
            remote_module: kind
                .getattr(pyo3::intern!(py, "remote_module"))?
                .extract::<Option<String>>()?
                .unwrap_or_default(),
            remote_types: kind.getattr(pyo3::intern!(py, "remote_types"))?.extract()?,
            udf: kind.getattr(pyo3::intern!(py, "udf"))?.extract()?,
            fields,
        }),
    })
}
