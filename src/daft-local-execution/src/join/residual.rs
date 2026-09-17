//! Applying a join predicate that key equality alone cannot express.
//!
//! A hash join finds candidate pairs by key. When the predicate says more than
//! that, the rest of it is checked on the pairs themselves, and a probe row whose
//! every candidate fails is treated as having matched nothing at all. Deciding
//! that after the check, rather than before, is what keeps an outer join correct.

use arrow::array::Array;
use common_error::DaftResult;
use daft_core::{array::ops::as_arrow::AsArrow, prelude::UInt64Array};
use daft_dsl::expr::bound_expr::BoundExpr;
use daft_recordbatch::{GrowableRecordBatch, ProbeState, RecordBatch};

/// Candidate pairs of one probe batch, and the probe rows that had none.
pub(crate) struct Candidates {
    /// Build-side table and row of each candidate, in emission order.
    pub(crate) build: Vec<(u32, u64)>,
    /// Probe row of each candidate.
    pub(crate) probe: Vec<u64>,
    /// Probe rows no candidate covers, in ascending order.
    pub(crate) unmatched: Vec<u64>,
}

impl Candidates {
    /// Find the pairs key equality allows, for one probe batch.
    ///
    /// # Errors
    /// If the join keys cannot be evaluated or probed.
    pub(crate) fn collect(
        input_table: &RecordBatch,
        probe_state: &ProbeState,
        probe_on: &[BoundExpr],
    ) -> DaftResult<Self> {
        let join_keys = input_table.eval_expression_list(probe_on)?;
        let mut build = Vec::new();
        let mut probe = Vec::new();
        let mut unmatched = Vec::new();
        for (probe_row, matches) in probe_state.probe_indices(join_keys)?.enumerate() {
            match matches {
                Some(matches) => {
                    let before = build.len();
                    for (build_table, build_row) in matches {
                        build.push((build_table, build_row));
                        probe.push(probe_row as u64);
                    }
                    if build.len() == before {
                        unmatched.push(probe_row as u64);
                    }
                }
                None => unmatched.push(probe_row as u64),
            }
        }
        Ok(Self {
            build,
            probe,
            unmatched,
        })
    }

    /// Whether any pair survived.
    pub(crate) fn is_empty(&self) -> bool {
        self.build.is_empty()
    }

    /// Drop the pairs `predicate` rejects, and count a probe row left without one
    /// as having matched nothing.
    ///
    /// The predicate reads both sides of a pair side by side, in join order, so
    /// it does not depend on which side was built on or on what the join itself
    /// returns.
    ///
    /// # Errors
    /// If the pairs cannot be gathered or the predicate cannot be evaluated.
    pub(crate) fn retain(
        self,
        predicate: &BoundExpr,
        input_table: &RecordBatch,
        build_side_tables: &[&RecordBatch],
        build_on_left: bool,
    ) -> DaftResult<Self> {
        if self.is_empty() {
            return Ok(self);
        }
        let build_rows = take_build_rows(build_side_tables, &self.build)?;
        let probe_rows = take_probe_rows(input_table, &self.probe)?;
        let candidate = if build_on_left {
            build_rows.union(&probe_rows)?
        } else {
            probe_rows.union(&build_rows)?
        };
        let kept = candidate.eval_expression(predicate)?;
        let kept = kept.bool()?.as_arrow()?.clone();

        let mut build = Vec::with_capacity(self.build.len());
        let mut probe = Vec::with_capacity(self.probe.len());
        let mut unmatched = self.unmatched;
        let mut row = 0usize;
        while row < self.build.len() {
            let probe_row = self.probe[row];
            let mut survived = false;
            while row < self.build.len() && self.probe[row] == probe_row {
                if kept.is_valid(row) && kept.value(row) {
                    build.push(self.build[row]);
                    probe.push(probe_row);
                    survived = true;
                }
                row += 1;
            }
            if !survived {
                unmatched.push(probe_row);
            }
        }
        unmatched.sort_unstable();
        Ok(Self {
            build,
            probe,
            unmatched,
        })
    }
}

/// Build-side rows of the given pairs, in pair order.
///
/// # Errors
/// If the rows cannot be gathered.
pub(crate) fn take_build_rows(
    build_side_tables: &[&RecordBatch],
    pairs: &[(u32, u64)],
) -> DaftResult<RecordBatch> {
    let mut growable = GrowableRecordBatch::new(build_side_tables, false, pairs.len().max(1))?;
    for (build_table, build_row) in pairs {
        growable.extend(*build_table as usize, *build_row as usize, 1);
    }
    growable.build()
}

/// Probe-side rows of the given pairs, in pair order.
///
/// # Errors
/// If the rows cannot be gathered.
pub(crate) fn take_probe_rows(input_table: &RecordBatch, rows: &[u64]) -> DaftResult<RecordBatch> {
    input_table.take(&UInt64Array::from_vec("", rows.to_vec()))
}
