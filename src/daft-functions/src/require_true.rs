//! Failing a query when a condition does not hold on every row it is evaluated on.

use common_error::{DaftError, DaftResult};
use daft_core::prelude::*;
use daft_dsl::functions::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct RequireTrue;

#[derive(FunctionArgs)]
struct Args<T> {
    value: T,
    condition: T,
    message: String,
}

/// How many of `rows` rows `condition` does not hold on: false or null.
///
/// A single condition stands for every row, as a literal does.
fn rows_not_true(condition: &BooleanArray, rows: usize) -> usize {
    let not_true = condition.len() - condition.true_count();
    if condition.len() == 1 && not_true == 1 {
        rows
    } else {
        not_true
    }
}

#[typetag::serde]
impl ScalarUDF for RequireTrue {
    fn name(&self) -> &'static str {
        "require_true"
    }

    fn call(
        &self,
        inputs: daft_dsl::functions::FunctionArgs<Series>,
        _ctx: &daft_dsl::functions::scalar::EvalContext,
    ) -> DaftResult<Series> {
        let Args {
            value,
            condition,
            message,
        } = inputs.try_into()?;
        let failing = rows_not_true(condition.bool()?, value.len());
        if failing > 0 {
            return Err(DaftError::ValueError(format!(
                "{message}: {failing} row(s) do not satisfy it"
            )));
        }
        Ok(value)
    }

    fn get_return_field(
        &self,
        inputs: FunctionArgs<ExprRef>,
        schema: &Schema,
    ) -> DaftResult<Field> {
        let Args {
            value, condition, ..
        } = inputs.try_into()?;
        let condition_field = condition.to_field(schema)?;
        if condition_field.dtype != DataType::Boolean {
            return Err(DaftError::TypeError(format!(
                "Expected the condition to be type Boolean, got {}",
                condition_field.dtype
            )));
        }
        value.to_field(schema)
    }

    fn docstring(&self) -> &'static str {
        "Answers `value` when `condition` is true on every row; otherwise fails with `message`, \
         counting the rows where it is false or null."
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::rows_not_true;
    use crate::require_true::BooleanArray;

    #[rstest]
    #[case::all_true(&[Some(true), Some(true)], 2, 0)]
    #[case::a_false(&[Some(true), Some(false)], 2, 1)]
    #[case::a_null(&[Some(true), None], 2, 1)]
    #[case::no_rows(&[], 0, 0)]
    #[case::a_true_literal(&[Some(true)], 3, 0)]
    #[case::a_false_literal_over_rows(&[Some(false)], 3, 3)]
    #[case::a_null_literal_over_rows(&[None], 3, 3)]
    #[case::a_false_literal_over_no_rows(&[Some(false)], 0, 0)]
    fn the_rows_a_condition_does_not_hold_on_are_counted(
        #[case] condition: &[Option<bool>],
        #[case] rows: usize,
        #[case] expected: usize,
    ) {
        let condition = BooleanArray::from_iter("condition", condition.iter().copied());

        assert_eq!(rows_not_true(&condition, rows), expected);
    }
}
