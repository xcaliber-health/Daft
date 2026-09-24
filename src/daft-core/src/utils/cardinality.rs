//! The failure of a value that must come from exactly one row.

use common_error::DaftError;

/// The error for a group that holds two or more rows where one was required.
///
/// Every single-value aggregation over such a group fails alike, whichever column
/// it reads, so the message names the aggregation rather than a column.
#[must_use]
pub fn several_rows_in_a_group() -> DaftError {
    DaftError::CardinalityViolation(
        "single_value found a group holding more than one row".to_string(),
    )
}
