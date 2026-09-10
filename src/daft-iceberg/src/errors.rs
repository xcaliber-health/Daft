//! Error type shared by the rewrite planner and the z-order encoder.

use common_error::DaftError;

/// Error raised while planning or encoding a data-file rewrite.
#[derive(Debug, thiserror::Error)]
pub enum IcebergRewriteError {
    /// An option value is outside the range the planner accepts.
    #[error("invalid option `{name}`: {reason}")]
    InvalidOption { name: String, reason: String },

    /// A z-order column has a type with no ordered byte encoding.
    #[error("zorder column `{column}` has unsupported type `{dtype}`")]
    UnsupportedZOrderType { column: String, dtype: String },

    /// The requested output partition spec is not declared by the table.
    #[error("output_spec_id {output} not present in table specs")]
    UnknownOutputSpec { output: i32 },

    /// One or more option keys are not recognised by the planner.
    #[error("unsupported option(s) {names:?}; supported options are {supported:?}")]
    UnknownOptions {
        names: Vec<String>,
        supported: Vec<String>,
    },
}

impl From<IcebergRewriteError> for DaftError {
    fn from(e: IcebergRewriteError) -> Self {
        Self::ValueError(e.to_string())
    }
}
