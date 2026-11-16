//! SQL's truncating `%` (`MOD`) and `//`.
//!
//! The DataFrame operators `%` and `//` floor, as Python's do. SQL's truncate toward
//! zero, so their remainder takes the dividend's sign, as the SQL standard's `MOD`
//! does. The SQL planner lowers its operators to these functions.

use std::ops::Rem;

use common_error::DaftResult;
use daft_core::{
    datatypes::InferDataType,
    prelude::{Field, Schema},
    series::Series,
};
use daft_dsl::{
    ExprRef,
    functions::{FunctionArgs, ScalarUDF, scalar::EvalContext},
};
use serde::{Deserialize, Serialize};

#[derive(FunctionArgs)]
struct DivisionArgs<T> {
    dividend: T,
    divisor: T,
}

/// The type and name of a division's result: the dividend's name, with `infer`'s type.
fn division_field(
    inputs: FunctionArgs<ExprRef>,
    schema: &Schema,
    infer: impl Fn(InferDataType, InferDataType) -> DaftResult<daft_core::prelude::DataType>,
) -> DaftResult<Field> {
    let DivisionArgs { dividend, divisor } = inputs.try_into()?;
    let dividend = dividend.to_field(schema)?;
    let divisor = divisor.to_field(schema)?;
    let dtype = infer(
        InferDataType::from(&dividend.dtype),
        InferDataType::from(&divisor.dtype),
    )?;
    Ok(Field::new(dividend.name, dtype))
}

/// SQL `MOD(a, b)` and `a % b`: the remainder with the dividend's sign.
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Mod;

#[typetag::serde]
impl ScalarUDF for Mod {
    fn call(&self, inputs: FunctionArgs<Series>, _ctx: &EvalContext) -> DaftResult<Series> {
        let DivisionArgs { dividend, divisor } = inputs.try_into()?;
        dividend.sql_remainder(&divisor)
    }

    fn name(&self) -> &'static str {
        "mod"
    }

    fn get_return_field(
        &self,
        inputs: FunctionArgs<ExprRef>,
        schema: &Schema,
    ) -> DaftResult<Field> {
        division_field(inputs, schema, |dividend, divisor| dividend.rem(divisor))
    }

    fn docstring(&self) -> &'static str {
        "Returns the remainder of a divided by b, with the sign of a (truncated toward zero), \
         as SQL's MOD does. Returns NULL if b is an integer 0."
    }
}

/// SQL's `a // b`: integer division truncated toward zero.
///
/// Float operands are divided and floored, as the operator has always done in SQL. It
/// has no public name: the SQL planner builds it for `//` only.
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SqlIntegerDivide;

#[typetag::serde]
impl ScalarUDF for SqlIntegerDivide {
    fn call(&self, inputs: FunctionArgs<Series>, _ctx: &EvalContext) -> DaftResult<Series> {
        let DivisionArgs { dividend, divisor } = inputs.try_into()?;
        dividend.sql_integer_divide(&divisor)
    }

    fn name(&self) -> &'static str {
        "sql_integer_divide"
    }

    fn get_return_field(
        &self,
        inputs: FunctionArgs<ExprRef>,
        schema: &Schema,
    ) -> DaftResult<Field> {
        division_field(inputs, schema, |dividend, divisor| {
            dividend.floor_div(&divisor)
        })
    }

    fn docstring(&self) -> &'static str {
        "Returns a divided by b, truncated toward zero for integers and floored for floats, \
         as SQL's // does. Returns NULL if b is an integer 0."
    }
}
