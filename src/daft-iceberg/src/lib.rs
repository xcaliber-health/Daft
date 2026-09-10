//! Table-maintenance planning primitives.
//!
//! Provides file-group planning for `rewrite_data_files`, the option and strategy types
//! that configure it, and the z-order key encoding used by the clustering strategies.

pub mod errors;
pub mod options;
pub mod planner;
pub mod zorder;

#[cfg(feature = "python")]
pub mod python;

pub use errors::IcebergRewriteError;
pub use options::{
    JobOrder, NullOrder, RewriteOptions, SortColumn, SortDirection, Strategy, ZOrderKey,
};
pub use planner::{CandidateFile, FileGroup, plan_file_groups};
pub use zorder::{
    ZORDER_KEY_COL, build_zorder_key_array, interleave_bits, normalize_to_ordered_bytes,
};

/// Register this crate's Python submodule under `parent`.
///
/// # Errors
/// Returns an error when the submodule or any of its functions cannot be created.
#[cfg(feature = "python")]
pub fn register_modules(parent: &pyo3::Bound<pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    python::register_modules(parent)
}
