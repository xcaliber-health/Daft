//! Description of a row-level merge: what to do with each row of a joined stream.
//!
//! A merge pairs rows of a target with rows of a source and decides, per pair,
//! whether to keep, replace, remove or add a row. The rules are grouped by which
//! side of the pair exists, and within a group the first rule whose condition
//! holds decides the row. A row no rule claims produces no output, so a caller
//! that wants unclaimed rows carried through adds an unconditional keep rule.

use std::sync::Arc;

use common_error::{DaftError, DaftResult};
use daft_core::prelude::*;
use daft_dsl::{ExprRef, expr::bound_expr::BoundExpr};
use serde::{Deserialize, Serialize};

/// What a rule does with the row it claims.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MergeActionKind {
    /// Emit the target row unchanged.
    Keep,
    /// Emit a replacement for the target row.
    Update,
    /// Emit the target row so it can be removed.
    Delete,
    /// Emit a row that has no counterpart in the target.
    Insert,
}

impl MergeActionKind {
    /// Value written to the action column for this kind.
    ///
    /// The tags are part of the operator's output, so consumers depend on them.
    #[must_use]
    pub const fn tag(self) -> u8 {
        match self {
            Self::Keep => 0,
            Self::Update => 1,
            Self::Delete => 2,
            Self::Insert => 3,
        }
    }

    /// Name of the kind, for plan displays.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Keep => "keep",
            Self::Update => "update",
            Self::Delete => "delete",
            Self::Insert => "insert",
        }
    }
}

/// One rule of a merge: an optional condition and the row it produces.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MergeClause<E = ExprRef> {
    /// Condition the pair must satisfy; absent means the rule always applies.
    pub condition: Option<E>,
    /// What the rule does with the row it claims.
    pub action: MergeActionKind,
    /// One expression per output column, in output order.
    pub outputs: Vec<E>,
}

impl MergeClause {
    /// Bind the rule's expressions against the schema of the joined stream.
    ///
    /// # Errors
    /// If a condition or output expression does not resolve against `schema`.
    pub fn bind(&self, schema: &Schema) -> DaftResult<MergeClause<BoundExpr>> {
        Ok(MergeClause {
            condition: self
                .condition
                .as_ref()
                .map(|condition| BoundExpr::try_new(condition.clone(), schema))
                .transpose()?,
            action: self.action,
            outputs: BoundExpr::bind_all(&self.outputs, schema)?,
        })
    }
}

/// The rules of one merge, grouped by which side of a pair exists.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MergeRowsConfig<E = ExprRef> {
    /// Rules for pairs where both sides exist.
    pub matched: Vec<MergeClause<E>>,
    /// Rules for rows that exist only on the source side.
    pub not_matched: Vec<MergeClause<E>>,
    /// Rules for rows that exist only on the target side.
    pub not_matched_by_source: Vec<MergeClause<E>>,
    /// Predicate telling whether the target side of a pair exists.
    pub target_present: E,
    /// Predicate telling whether the source side of a pair exists.
    pub source_present: E,
    /// Columns identifying a target row; empty skips the one-match check.
    pub row_id: Vec<E>,
    /// Name of the column carrying each output row's action tag.
    pub action_column: String,
}

impl<E: std::fmt::Display> MergeRowsConfig<E> {
    /// Every rule, in the order the groups are evaluated.
    pub fn clauses(&self) -> impl Iterator<Item = &MergeClause<E>> {
        self.matched
            .iter()
            .chain(&self.not_matched)
            .chain(&self.not_matched_by_source)
    }

    /// Whether a target row matched by more than one source row is an error.
    #[must_use]
    pub fn checks_cardinality(&self) -> bool {
        !self.row_id.is_empty()
    }

    /// Lines describing the merge for a plan display.
    #[must_use]
    pub fn multiline_display(&self) -> Vec<String> {
        let describe = |group: &str, clauses: &[MergeClause<E>]| -> Option<String> {
            (!clauses.is_empty()).then(|| {
                let rules = clauses
                    .iter()
                    .map(|clause| match &clause.condition {
                        Some(condition) => format!("{} if {condition}", clause.action.name()),
                        None => clause.action.name().to_string(),
                    })
                    .collect::<Vec<String>>()
                    .join(", ");
                format!("{group} = [{rules}]")
            })
        };
        let mut lines: Vec<String> = [
            describe("Matched", &self.matched),
            describe("Not matched", &self.not_matched),
            describe("Not matched by source", &self.not_matched_by_source),
        ]
        .into_iter()
        .flatten()
        .collect();
        if self.checks_cardinality() {
            lines.push("One match per target row = enforced".to_string());
        }
        lines
    }
}

impl MergeRowsConfig {
    /// Bind every expression against the schema of the joined stream.
    ///
    /// # Errors
    /// If any expression does not resolve against `schema`.
    pub fn bind(&self, schema: &Schema) -> DaftResult<MergeRowsConfig<BoundExpr>> {
        let bind_group = |clauses: &[MergeClause]| -> DaftResult<Vec<MergeClause<BoundExpr>>> {
            clauses.iter().map(|clause| clause.bind(schema)).collect()
        };
        Ok(MergeRowsConfig {
            matched: bind_group(&self.matched)?,
            not_matched: bind_group(&self.not_matched)?,
            not_matched_by_source: bind_group(&self.not_matched_by_source)?,
            target_present: BoundExpr::try_new(self.target_present.clone(), schema)?,
            source_present: BoundExpr::try_new(self.source_present.clone(), schema)?,
            row_id: BoundExpr::bind_all(&self.row_id, schema)?,
            action_column: self.action_column.clone(),
        })
    }

    /// Schema of the rows this merge emits: the columns its rules produce, then the
    /// action column.
    ///
    /// # Errors
    /// If there are no rules, if the rules disagree on the columns they produce, if
    /// a condition is not a boolean, or if an expression does not resolve.
    pub fn output_schema(&self, input: &Schema) -> DaftResult<Arc<Schema>> {
        self.validate(input)?;
        let first = self
            .clauses()
            .next()
            .ok_or_else(|| DaftError::ValueError("a merge needs at least one rule".to_string()))?;
        let mut fields = first
            .outputs
            .iter()
            .map(|output| output.to_field(input))
            .collect::<DaftResult<Vec<Field>>>()?;
        fields.push(Field::new(self.action_column.clone(), DataType::UInt8));
        Ok(Arc::new(Schema::new(fields)))
    }

    /// Check that the rules agree with each other and with `input`.
    ///
    /// # Errors
    /// See [`Self::output_schema`].
    fn validate(&self, input: &Schema) -> DaftResult<()> {
        Self::require_boolean(&self.target_present, input, "target-present predicate")?;
        Self::require_boolean(&self.source_present, input, "source-present predicate")?;

        let mut expected: Option<Vec<Field>> = None;
        for clause in self.clauses() {
            if let Some(condition) = &clause.condition {
                Self::require_boolean(condition, input, "rule condition")?;
            }
            let fields = clause
                .outputs
                .iter()
                .map(|output| output.to_field(input))
                .collect::<DaftResult<Vec<Field>>>()?;
            match &expected {
                None => expected = Some(fields),
                // Rows of different rules end up in one stream, so they must line up.
                Some(expected) if *expected != fields => {
                    return Err(DaftError::ValueError(format!(
                        "merge rules must produce the same columns: expected {expected:?}, got {fields:?}"
                    )));
                }
                Some(_) => {}
            }
        }
        if expected.is_none() {
            return Err(DaftError::ValueError(
                "a merge needs at least one rule".to_string(),
            ));
        }
        for column in &self.row_id {
            column.to_field(input)?;
        }
        Ok(())
    }

    fn require_boolean(expr: &ExprRef, input: &Schema, what: &str) -> DaftResult<()> {
        let field = expr.to_field(input)?;
        if field.dtype == DataType::Boolean {
            Ok(())
        } else {
            Err(DaftError::ValueError(format!(
                "{what} must be a boolean, got {}",
                field.dtype
            )))
        }
    }
}

#[cfg(feature = "python")]
mod python {
    use daft_dsl::python::PyExpr;
    use pyo3::prelude::*;

    use super::{MergeActionKind, MergeClause, MergeRowsConfig};

    /// One rule of a merge, as built from Python.
    #[pyclass(module = "daft.daft", name = "MergeClause", frozen, from_py_object)]
    #[derive(Clone)]
    pub struct PyMergeClause {
        pub(super) clause: MergeClause,
    }

    #[pymethods]
    impl PyMergeClause {
        /// Build a rule.
        ///
        /// `action` is one of `keep`, `update`, `delete` or `insert`. `outputs`
        /// holds one expression per output column, in output order. `condition`
        /// restricts the rows the rule claims.
        #[new]
        #[pyo3(signature = (action, outputs, condition=None))]
        fn new(action: &str, outputs: Vec<PyExpr>, condition: Option<PyExpr>) -> PyResult<Self> {
            let action = match action {
                "keep" => MergeActionKind::Keep,
                "update" => MergeActionKind::Update,
                "delete" => MergeActionKind::Delete,
                "insert" => MergeActionKind::Insert,
                other => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "unknown merge action {other}"
                    )));
                }
            };
            Ok(Self {
                clause: MergeClause {
                    condition: condition.map(|condition| condition.expr),
                    action,
                    outputs: outputs.into_iter().map(|output| output.expr).collect(),
                },
            })
        }
    }

    /// The rules of one merge, as built from Python.
    #[pyclass(module = "daft.daft", name = "MergeRowsConfig", frozen, from_py_object)]
    #[derive(Clone)]
    pub struct PyMergeRowsConfig {
        pub(crate) config: MergeRowsConfig,
    }

    #[pymethods]
    impl PyMergeRowsConfig {
        /// Build the rules of one merge.
        ///
        /// `row_id` names the columns identifying a target row; leaving it empty
        /// allows a target row to be claimed by more than one source row.
        #[new]
        #[pyo3(signature = (
            matched,
            not_matched,
            not_matched_by_source,
            target_present,
            source_present,
            row_id,
            action_column,
        ))]
        fn new(
            matched: Vec<PyMergeClause>,
            not_matched: Vec<PyMergeClause>,
            not_matched_by_source: Vec<PyMergeClause>,
            target_present: PyExpr,
            source_present: PyExpr,
            row_id: Vec<PyExpr>,
            action_column: String,
        ) -> Self {
            let clauses = |group: Vec<PyMergeClause>| group.into_iter().map(|c| c.clause).collect();
            Self {
                config: MergeRowsConfig {
                    matched: clauses(matched),
                    not_matched: clauses(not_matched),
                    not_matched_by_source: clauses(not_matched_by_source),
                    target_present: target_present.expr,
                    source_present: source_present.expr,
                    row_id: row_id.into_iter().map(|column| column.expr).collect(),
                    action_column,
                },
            }
        }
    }
}

#[cfg(feature = "python")]
pub use python::{PyMergeClause, PyMergeRowsConfig};

#[cfg(test)]
mod tests {
    use daft_dsl::{lit, resolved_col};

    use super::*;

    fn input_schema() -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::Int64),
            Field::new("present", DataType::Boolean),
            Field::new("source_present", DataType::Boolean),
        ])
    }

    fn clause(action: MergeActionKind, outputs: Vec<ExprRef>) -> MergeClause {
        MergeClause {
            condition: None,
            action,
            outputs,
        }
    }

    fn config(matched: Vec<MergeClause>) -> MergeRowsConfig {
        MergeRowsConfig {
            matched,
            not_matched: vec![],
            not_matched_by_source: vec![],
            target_present: resolved_col("present"),
            source_present: resolved_col("source_present"),
            row_id: vec![],
            action_column: "__merge_action".to_string(),
        }
    }

    #[test]
    fn output_schema_is_the_rules_columns_plus_the_action() {
        let config = config(vec![clause(
            MergeActionKind::Update,
            vec![resolved_col("id")],
        )]);

        let schema = config.output_schema(&input_schema()).unwrap();

        assert_eq!(
            schema.fields(),
            &[
                Field::new("id", DataType::Int64),
                Field::new("__merge_action", DataType::UInt8),
            ]
        );
    }

    #[test]
    fn rules_producing_different_columns_are_refused() {
        let config = config(vec![
            clause(MergeActionKind::Update, vec![resolved_col("id")]),
            clause(MergeActionKind::Keep, vec![lit("x").alias("id")]),
        ]);

        let error = config.output_schema(&input_schema()).unwrap_err();

        assert!(
            error.to_string().contains("same columns"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn a_non_boolean_condition_is_refused() {
        let mut config = config(vec![clause(
            MergeActionKind::Update,
            vec![resolved_col("id")],
        )]);
        config.matched[0].condition = Some(resolved_col("id"));

        let error = config.output_schema(&input_schema()).unwrap_err();

        assert!(
            error.to_string().contains("must be a boolean"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn a_merge_without_rules_is_refused() {
        let config = config(vec![]);

        let error = config.output_schema(&input_schema()).unwrap_err();

        assert!(
            error.to_string().contains("at least one rule"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn cardinality_is_checked_only_when_a_row_identity_is_given() {
        let mut config = config(vec![clause(
            MergeActionKind::Update,
            vec![resolved_col("id")],
        )]);
        assert!(!config.checks_cardinality());

        config.row_id = vec![resolved_col("id")];

        assert!(config.checks_cardinality());
    }
}
