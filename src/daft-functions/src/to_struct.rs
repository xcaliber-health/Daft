use common_error::{DaftError, DaftResult};
use daft_core::prelude::*;
use daft_dsl::functions::{prelude::*, scalar::ScalarFn};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub(super) struct ToStructFunction;

#[typetag::serde]
impl ScalarUDF for ToStructFunction {
    fn name(&self) -> &'static str {
        "struct"
    }
    fn call(
        &self,
        inputs: daft_dsl::functions::FunctionArgs<Series>,
        ctx: &daft_dsl::functions::scalar::EvalContext,
    ) -> DaftResult<Series> {
        let inputs = inputs.into_inner();
        if inputs.is_empty() {
            return Err(DaftError::ValueError(
                "Cannot call struct with no inputs".to_string(),
            ));
        }
        let target_len = ctx.row_count;
        let inputs = inputs
            .into_iter()
            .map(|s| {
                if s.len() == 1 && target_len > 1 {
                    s.broadcast(target_len)
                } else {
                    Ok(s)
                }
            })
            .collect::<DaftResult<Vec<_>>>()?;
        let child_fields: Vec<Field> = inputs.iter().map(|s| s.field().clone()).collect();
        let field = Field::new("struct", DataType::Struct(child_fields));

        Ok(StructArray::new(field, inputs, None).into_series())
    }
    fn get_return_field(
        &self,
        inputs: FunctionArgs<ExprRef>,
        schema: &Schema,
    ) -> DaftResult<Field> {
        let inputs = inputs.into_inner();
        if inputs.is_empty() {
            return Err(DaftError::ValueError(
                "Cannot call struct with no inputs".to_string(),
            ));
        }
        let child_fields = inputs
            .iter()
            .map(|e| e.to_field(schema))
            .collect::<DaftResult<Vec<_>>>()?;
        ensure_distinct_names(&child_fields)?;
        Ok(Field::new("struct", DataType::Struct(child_fields)))
    }
}

/// Refuses two fields of one name: a struct's fields are read back by name, so
/// only one of them could ever be reached and the other would be lost silently.
fn ensure_distinct_names(fields: &[Field]) -> DaftResult<()> {
    let mut seen = std::collections::HashSet::with_capacity(fields.len());
    match fields.iter().find(|field| !seen.insert(&*field.name)) {
        Some(repeated) => Err(DaftError::ValueError(format!(
            "struct() received two fields named {:?}; give each field its own name with .alias()",
            repeated.name
        ))),
        None => Ok(()),
    }
}

#[must_use]
pub fn to_struct(inputs: Vec<ExprRef>) -> ExprRef {
    ScalarFn::builtin(ToStructFunction, inputs).into()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn fields(names: &[&str]) -> Vec<Field> {
        names
            .iter()
            .map(|name| Field::new(*name, DataType::Int64))
            .collect()
    }

    #[rstest]
    #[case::distinct(&["x", "y"], true)]
    #[case::repeated(&["x", "x"], false)]
    #[case::repeated_apart(&["x", "y", "x"], false)]
    fn a_struct_takes_each_name_once(#[case] names: &[&str], #[case] accepted: bool) {
        assert_eq!(ensure_distinct_names(&fields(names)).is_ok(), accepted);
    }
}
