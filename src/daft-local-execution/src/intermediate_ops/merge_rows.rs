//! Applies row-level merge rules to a joined stream of target and source rows.
//!
//! Each pair of rows is claimed by the first rule of its group whose condition
//! holds; a pair no rule claims produces nothing. Rows are grouped by rule before
//! their output expressions run, so a rule's expressions only ever see the rows it
//! claimed.

use std::sync::Arc;

use arrow::{
    array::{Array, BooleanArray as ArrowBooleanArray},
    compute::{
        kernels::boolean::{and, not},
        prep_null_mask_filter,
    },
};
use common_error::{DaftError, DaftResult};
use common_metrics::ops::NodeType;
use daft_core::prelude::*;
use daft_dsl::expr::bound_expr::BoundExpr;
use daft_logical_plan::{MergeClause, MergeRowsConfig};
use daft_micropartition::MicroPartition;
use daft_recordbatch::RecordBatch;
use tracing::{Span, instrument};

use super::intermediate_op::{IntermediateOpExecuteResult, IntermediateOperator};
use crate::{
    ExecutionTaskSpawner,
    pipeline::{InputId, NodeName},
};

/// Identity of the last target row seen, so a second claim on it is caught even
/// when the two claims arrive in different morsels.
#[derive(Default)]
pub struct MergeRowsState {
    last_row_id: Option<RecordBatch>,
}

pub struct MergeRowsOperator {
    config: Arc<MergeRowsConfig<BoundExpr>>,
    schema: SchemaRef,
}

impl MergeRowsOperator {
    pub fn new(config: MergeRowsConfig<BoundExpr>, schema: SchemaRef) -> Self {
        Self {
            config: Arc::new(config),
            schema,
        }
    }
}

impl IntermediateOperator for MergeRowsOperator {
    type State = MergeRowsState;
    type BatchingStrategy = crate::dynamic_batching::StaticBatchingStrategy;

    #[instrument(skip_all, name = "MergeRowsOperator::execute")]
    fn execute(
        &self,
        input: MicroPartition,
        mut state: Self::State,
        _runtime_stats: Arc<Self::Stats>,
        task_spawner: &ExecutionTaskSpawner,
        _input_id: InputId,
    ) -> IntermediateOpExecuteResult<Self> {
        let config = self.config.clone();
        let schema = self.schema.clone();
        task_spawner
            .spawn(
                async move {
                    let mut outputs = Vec::new();
                    if let Some(batch) = input.concat_or_get()? {
                        outputs.extend(merge_batch(&batch, &config, &mut state)?);
                    }
                    let out = MicroPartition::new_loaded(schema, Arc::new(outputs), None);
                    Ok((state, out))
                },
                Span::current(),
            )
            .into()
    }

    fn multiline_display(&self) -> Vec<String> {
        let mut lines = vec!["MergeRows".to_string()];
        lines.extend(self.config.multiline_display());
        lines
    }

    fn name(&self) -> NodeName {
        "MergeRows".into()
    }

    fn op_type(&self) -> NodeType {
        NodeType::MergeRows
    }

    fn make_state(&self) -> Self::State {
        MergeRowsState::default()
    }

    fn batching_strategy(&self) -> DaftResult<Self::BatchingStrategy> {
        Ok(crate::dynamic_batching::StaticBatchingStrategy::new(
            self.morsel_size_requirement().unwrap_or_default(),
        ))
    }
}

/// Apply the rules to one batch, returning one batch per rule that claimed rows.
fn merge_batch(
    batch: &RecordBatch,
    config: &MergeRowsConfig<BoundExpr>,
    state: &mut MergeRowsState,
) -> DaftResult<Vec<RecordBatch>> {
    if batch.is_empty() {
        return Ok(Vec::new());
    }
    let target = predicate_mask(batch, &config.target_present)?;
    let source = predicate_mask(batch, &config.source_present)?;
    let matched = and(&target, &source)?;

    if config.checks_cardinality() {
        check_one_match_per_target_row(batch, config, &matched, state)?;
    }

    let groups = [
        (&config.matched, matched),
        (&config.not_matched, and(&source, &not(&target)?)?),
        (&config.not_matched_by_source, and(&target, &not(&source)?)?),
    ];

    let mut outputs = Vec::new();
    for (clauses, group) in groups {
        let mut unclaimed = group;
        for clause in clauses {
            if unclaimed.true_count() == 0 {
                break;
            }
            let claimed = match &clause.condition {
                Some(condition) => and(&unclaimed, &predicate_mask(batch, condition)?)?,
                None => unclaimed.clone(),
            };
            if claimed.true_count() == 0 {
                continue;
            }
            outputs.push(apply_clause(batch, clause, &claimed)?);
            unclaimed = and(&unclaimed, &not(&claimed)?)?;
        }
    }
    Ok(outputs)
}

/// Rows claimed by `clause`, projected to the merge's output columns.
fn apply_clause(
    batch: &RecordBatch,
    clause: &MergeClause<BoundExpr>,
    claimed: &ArrowBooleanArray,
) -> DaftResult<RecordBatch> {
    let rows = batch.mask_filter(&mask_series(claimed)?)?;
    let projected = rows.eval_expression_list(&clause.outputs)?;
    let action = UInt8Array::from_iter(
        Field::new(ACTION_COLUMN_PLACEHOLDER, DataType::UInt8),
        std::iter::repeat_n(Some(clause.action.tag()), projected.len()),
    )
    .into_series();
    let mut columns: Vec<Series> = (0..projected.schema.len())
        .map(|index| projected.get_column(index).clone())
        .collect();
    columns.push(action);
    RecordBatch::from_nonempty_columns(columns)
}

/// Placeholder name for the action column; the real name is applied by the caller's schema.
const ACTION_COLUMN_PLACEHOLDER: &str = "__merge_action";

/// Evaluate a predicate, reading a null result as "does not hold".
fn predicate_mask(batch: &RecordBatch, expr: &BoundExpr) -> DaftResult<ArrowBooleanArray> {
    let evaluated = batch.eval_expression(expr)?;
    let boolean = evaluated.bool()?.as_arrow()?.clone();
    Ok(if boolean.null_count() == 0 {
        boolean
    } else {
        prep_null_mask_filter(&boolean)
    })
}

/// Wrap a mask so a record batch can be filtered by it.
fn mask_series(mask: &ArrowBooleanArray) -> DaftResult<Series> {
    Ok(BooleanArray::from_arrow(
        Field::new("mask", DataType::Boolean),
        Arc::new(mask.clone()),
    )?
    .into_series())
}

/// Refuse a target row claimed by more than one source row.
///
/// All pairs of one target row arrive together, so a repeat shows up as two
/// neighbouring matched rows with the same identity. The last identity of a
/// morsel is carried over, which covers a target row whose pairs straddle two
/// morsels.
fn check_one_match_per_target_row(
    batch: &RecordBatch,
    config: &MergeRowsConfig<BoundExpr>,
    matched: &ArrowBooleanArray,
    state: &mut MergeRowsState,
) -> DaftResult<()> {
    if matched.true_count() == 0 {
        return Ok(());
    }
    let rows = batch.mask_filter(&mask_series(matched)?)?;
    let identities = rows.eval_expression_list(&config.row_id)?;
    let rows_seen = identities.len();

    if let Some(previous) = state.last_row_id.take()
        && repeats_any(&previous, &identities.slice(0, 1)?)?
    {
        return Err(cardinality_error());
    }
    if rows_seen > 1
        && repeats_any(
            &identities.slice(0, rows_seen - 1)?,
            &identities.slice(1, rows_seen)?,
        )?
    {
        return Err(cardinality_error());
    }

    state.last_row_id = Some(identities.slice(rows_seen - 1, rows_seen)?);
    Ok(())
}

/// Whether any row of `left` has the same identity as the row facing it in `right`.
///
/// A null identity never counts as a repeat: it cannot be shown to name the same
/// target row, and refusing the write on it would reject correct merges.
fn repeats_any(left: &RecordBatch, right: &RecordBatch) -> DaftResult<bool> {
    let mut same: Option<ArrowBooleanArray> = None;
    for column in 0..left.schema.len() {
        let equal = arrow::compute::kernels::cmp::eq(
            &left.get_column(column).to_arrow()?,
            &right.get_column(column).to_arrow()?,
        )?;
        let equal = if equal.null_count() == 0 {
            equal
        } else {
            prep_null_mask_filter(&equal)
        };
        same = Some(match same {
            None => equal,
            Some(previous) => and(&previous, &equal)?,
        });
    }
    Ok(same.is_some_and(|same| same.true_count() > 0))
}

fn cardinality_error() -> DaftError {
    DaftError::ValueError(
        "a target row was matched by more than one source row; remove the duplicate keys \
         from the source or narrow the match condition"
            .to_string(),
    )
}

#[cfg(test)]
mod tests {
    use daft_dsl::{ExprRef, lit, resolved_col};
    use daft_logical_plan::MergeActionKind;

    use super::*;

    /// Target rows 0..n paired with a source row where `has_source` says so.
    fn joined_batch(ids: &[i64], has_source: &[bool], source_values: &[i64]) -> RecordBatch {
        let target = Int64Array::from_iter(
            Field::new("target.id", DataType::Int64),
            ids.iter().copied().map(Some),
        )
        .into_series();
        let file = Int32Array::from_iter(
            Field::new("_file_idx", DataType::Int32),
            std::iter::repeat_n(Some(0i32), ids.len()),
        )
        .into_series();
        let position = Int64Array::from_iter(
            Field::new("_pos", DataType::Int64),
            (0..ids.len() as i64).map(Some),
        )
        .into_series();
        let source = Int64Array::from_iter(
            Field::new("source.id", DataType::Int64),
            has_source
                .iter()
                .zip(source_values)
                .map(|(present, value)| present.then_some(*value)),
        )
        .into_series();
        RecordBatch::from_nonempty_columns(vec![target, file, position, source]).unwrap()
    }

    fn outputs() -> Vec<ExprRef> {
        vec![
            resolved_col("target.id").alias("id"),
            resolved_col("_file_idx"),
            resolved_col("_pos"),
        ]
    }

    fn insert_outputs() -> Vec<ExprRef> {
        vec![
            resolved_col("source.id").alias("id"),
            lit(Option::<i32>::None).alias("_file_idx"),
            lit(Option::<i64>::None).alias("_pos"),
        ]
    }

    fn clause(
        action: MergeActionKind,
        condition: Option<ExprRef>,
        outputs: Vec<ExprRef>,
    ) -> MergeClause {
        MergeClause {
            condition,
            action,
            outputs,
        }
    }

    fn config(
        matched: Vec<MergeClause>,
        not_matched: Vec<MergeClause>,
        not_matched_by_source: Vec<MergeClause>,
        check_cardinality: bool,
    ) -> MergeRowsConfig {
        MergeRowsConfig {
            matched,
            not_matched,
            not_matched_by_source,
            target_present: resolved_col("target.id").not_null(),
            source_present: resolved_col("source.id").not_null(),
            row_id: if check_cardinality {
                vec![resolved_col("_file_idx"), resolved_col("_pos")]
            } else {
                vec![]
            },
            action_column: "__merge_action".to_string(),
        }
    }

    /// Run the merge over one batch and return (action tag, id) per output row.
    fn run(batch: &RecordBatch, config: &MergeRowsConfig) -> DaftResult<Vec<(u8, Option<i64>)>> {
        let bound = config.bind(&batch.schema)?;
        let mut state = MergeRowsState::default();
        let batches = merge_batch(batch, &bound, &mut state)?;
        let mut rows = Vec::new();
        for output in batches {
            let ids = output.get_column(0).i64()?.clone();
            let actions = output.get_column(3).u8()?.clone();
            for row in 0..output.len() {
                rows.push((actions.get(row).unwrap(), ids.get(row)));
            }
        }
        Ok(rows)
    }

    #[test]
    fn matched_rows_take_the_first_rule_whose_condition_holds() -> DaftResult<()> {
        let batch = joined_batch(&[1, 2, 3], &[true, true, true], &[1, 2, 3]);
        let config = config(
            vec![
                clause(
                    MergeActionKind::Delete,
                    Some(resolved_col("target.id").eq(lit(2i64))),
                    outputs(),
                ),
                clause(MergeActionKind::Update, None, outputs()),
            ],
            vec![],
            vec![],
            false,
        );

        let rows = run(&batch, &config)?;

        assert_eq!(
            rows,
            vec![
                (MergeActionKind::Delete.tag(), Some(2)),
                (MergeActionKind::Update.tag(), Some(1)),
                (MergeActionKind::Update.tag(), Some(3)),
            ]
        );
        Ok(())
    }

    #[test]
    fn rows_no_rule_claims_produce_nothing() -> DaftResult<()> {
        let batch = joined_batch(&[1, 2], &[true, false], &[1, 0]);
        let config = config(
            vec![clause(
                MergeActionKind::Update,
                Some(resolved_col("target.id").eq(lit(99i64))),
                outputs(),
            )],
            vec![],
            vec![],
            false,
        );

        assert!(run(&batch, &config)?.is_empty());
        Ok(())
    }

    #[test]
    fn unclaimed_rows_are_carried_through_by_an_unconditional_keep() -> DaftResult<()> {
        let batch = joined_batch(&[1, 2], &[true, false], &[1, 0]);
        let config = config(
            vec![clause(
                MergeActionKind::Update,
                Some(resolved_col("target.id").eq(lit(99i64))),
                outputs(),
            )],
            vec![],
            vec![clause(MergeActionKind::Keep, None, outputs())],
            false,
        );

        let rows = run(&batch, &config)?;

        assert_eq!(rows, vec![(MergeActionKind::Keep.tag(), Some(2))]);
        Ok(())
    }

    #[test]
    fn source_only_rows_are_inserted() -> DaftResult<()> {
        let batch = joined_batch(&[1], &[true], &[1]);
        let source_only = {
            let ids = Int64Array::from_iter(
                Field::new("target.id", DataType::Int64),
                std::iter::once(None),
            )
            .into_series();
            let file = Int32Array::from_iter(
                Field::new("_file_idx", DataType::Int32),
                std::iter::once(None),
            )
            .into_series();
            let position =
                Int64Array::from_iter(Field::new("_pos", DataType::Int64), std::iter::once(None))
                    .into_series();
            let source = Int64Array::from_iter(
                Field::new("source.id", DataType::Int64),
                std::iter::once(Some(7i64)),
            )
            .into_series();
            RecordBatch::from_nonempty_columns(vec![ids, file, position, source]).unwrap()
        };
        let batch = RecordBatch::concat(&[batch, source_only])?;
        let config = config(
            vec![clause(MergeActionKind::Update, None, outputs())],
            vec![clause(MergeActionKind::Insert, None, insert_outputs())],
            vec![],
            false,
        );

        let rows = run(&batch, &config)?;

        assert_eq!(
            rows,
            vec![
                (MergeActionKind::Update.tag(), Some(1)),
                (MergeActionKind::Insert.tag(), Some(7)),
            ]
        );
        Ok(())
    }

    #[test]
    fn a_target_row_matched_twice_is_refused() {
        let mut batch = joined_batch(&[1, 1], &[true, true], &[1, 1]);
        // Both pairs name the same target row, as a duplicated source key would.
        batch = batch
            .eval_expression_list(
                &BoundExpr::bind_all(
                    &[
                        resolved_col("target.id"),
                        resolved_col("_file_idx"),
                        lit(0i64).alias("_pos"),
                        resolved_col("source.id"),
                    ],
                    &batch.schema,
                )
                .unwrap(),
            )
            .unwrap();
        let config = config(
            vec![clause(MergeActionKind::Update, None, outputs())],
            vec![],
            vec![],
            true,
        );

        let error = run(&batch, &config).unwrap_err();

        assert!(
            error.to_string().contains("more than one source row"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn a_repeat_split_across_morsels_is_refused() -> DaftResult<()> {
        let first = joined_batch(&[1], &[true], &[1]);
        let second = joined_batch(&[1], &[true], &[1]);
        let config = config(
            vec![clause(MergeActionKind::Update, None, outputs())],
            vec![],
            vec![],
            true,
        );
        let bound = config.bind(&first.schema)?;
        let mut state = MergeRowsState::default();

        merge_batch(&first, &bound, &mut state)?;
        let error = merge_batch(&second, &bound, &mut state).unwrap_err();

        assert!(
            error.to_string().contains("more than one source row"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn distinct_target_rows_are_not_a_repeat() -> DaftResult<()> {
        let batch = joined_batch(&[1, 2, 3], &[true, true, true], &[1, 2, 3]);
        let config = config(
            vec![clause(MergeActionKind::Update, None, outputs())],
            vec![],
            vec![],
            true,
        );

        assert_eq!(run(&batch, &config)?.len(), 3);
        Ok(())
    }

    #[test]
    fn an_empty_batch_produces_nothing() -> DaftResult<()> {
        let batch = joined_batch(&[1], &[true], &[1]);
        let empty = batch.slice(0, 0)?;
        let config = config(
            vec![clause(MergeActionKind::Update, None, outputs())],
            vec![],
            vec![],
            true,
        );

        assert!(run(&empty, &config)?.is_empty());
        Ok(())
    }
}
