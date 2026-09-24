//! Completing a single value taken in parts: per partial aggregation, or per key before
//! being joined back to the rows that ask for it.

use common_error::{DaftError, DaftResult};
use daft_core::{prelude::*, utils::cardinality::several_rows_in_a_group};
use serde::{Deserialize, Serialize};

use crate::{
    ExprRef,
    functions::{FunctionArgs, ScalarUDF, scalar::ScalarFn},
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub(super) struct SingleRowValueFunction;

#[derive(FunctionArgs)]
struct Args<T> {
    value: T,
    rows: T,
}

/// Whether any present count in `rows` exceeds one.
fn any_holds_several(rows: &UInt64Array) -> bool {
    if rows.null_count() == 0 {
        rows.values().iter().any(|&n| n > 1)
    } else {
        rows.into_iter().flatten().any(|n| n > 1)
    }
}

#[typetag::serde]
impl ScalarUDF for SingleRowValueFunction {
    fn name(&self) -> &'static str {
        "single_row_value"
    }

    fn call(
        &self,
        inputs: FunctionArgs<Series>,
        _ctx: &crate::functions::scalar::EvalContext,
    ) -> DaftResult<Series> {
        let Args { value, rows } = inputs.try_into()?;
        if any_holds_several(rows.u64()?) {
            return Err(several_rows_in_a_group());
        }
        Ok(value)
    }

    fn get_return_field(
        &self,
        inputs: FunctionArgs<ExprRef>,
        schema: &Schema,
    ) -> DaftResult<Field> {
        let Args { value, rows } = inputs.try_into()?;
        let rows_field = rows.to_field(schema)?;
        if rows_field.dtype != DataType::UInt64 {
            return Err(DaftError::SchemaMismatch(format!(
                "Expected the row count to be type UInt64, got {}",
                rows_field.dtype
            )));
        }
        value.to_field(schema)
    }
}

/// Answers `value` on each row whose count of source rows, `rows`, is at most one.
///
/// It completes a single value taken in parts, where `rows` counts the rows the value was
/// taken from: the partial aggregations of a group, or the rows of a key joined back to
/// the rows that ask for it. Only the rows it is evaluated on are checked. A null count
/// (a row with no key) answers `value`, which is null there.
///
/// Evaluating it on a row whose count exceeds one fails the query with a cardinality
/// violation.
#[must_use]
pub fn single_row_value(value: ExprRef, rows: ExprRef) -> ExprRef {
    ScalarFn::builtin(SingleRowValueFunction {}, vec![value, rows]).into()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::any_holds_several;
    use crate::functions::agg::single_row::UInt64Array;

    #[rstest]
    #[case::each_key_one_row(&[Some(1), Some(1)], false)]
    #[case::rows_without_a_key(&[Some(1), None], false)]
    #[case::a_key_of_two_rows(&[Some(1), Some(2)], true)]
    #[case::a_key_of_two_rows_among_rows_without_a_key(&[None, Some(2)], true)]
    #[case::no_rows(&[], false)]
    fn a_count_over_one_is_found(#[case] rows: &[Option<u64>], #[case] expected: bool) {
        let rows = UInt64Array::from_iter(
            daft_core::prelude::Field::new("rows", daft_core::prelude::DataType::UInt64),
            rows.iter().copied(),
        );

        assert_eq!(any_holds_several(&rows), expected);
    }
}
