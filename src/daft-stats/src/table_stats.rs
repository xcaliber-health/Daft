use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    hash::Hash,
    ops::{BitAnd, BitOr, Index, Not},
    sync::Arc,
};

use common_error::{DaftError, DaftResult};
use daft_core::prelude::*;
use daft_dsl::{
    Column, Expr, ExprRef,
    expr::{BoundColumn, bound_expr::BoundExpr},
    null_lit, resolved_col,
};
use daft_recordbatch::RecordBatch;
use snafu::ResultExt;

use crate::{DaftCoreComputeSnafu, column_stats::ColumnRangeStatistics};

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize, Hash)]
pub struct TableStatistics {
    columns: Vec<ColumnRangeStatistics>,
    schema: SchemaRef,
}

/// The longest value list a membership test is judged by value against
/// statistics; a longer one is judged by the range its values span.
const MAX_LISTED_EQUALITIES: usize = 64;

impl TableStatistics {
    pub fn new(columns: Vec<ColumnRangeStatistics>, schema: SchemaRef) -> Self {
        Self { columns, schema }
    }

    pub fn from_stats_table(table: &RecordBatch) -> DaftResult<Self> {
        // Assumed format is each column having 2 rows:
        // - row 0: Minimum value for the column.
        // - row 1: Maximum value for the column.
        if table.len() != 2 {
            return Err(DaftError::ValueError(format!(
                "Expected stats table to have 2 rows, with min and max values for each column, but got {} rows: {}",
                table.len(),
                table
            )));
        }
        let columns = table
            .columns()
            .iter()
            .map(|col| {
                Ok(ColumnRangeStatistics::new(
                    Some(col.slice(0, 1)?.take_materialized_series()),
                    Some(col.slice(1, 2)?.take_materialized_series()),
                )?)
            })
            .collect::<DaftResult<_>>()?;

        Ok(Self {
            columns,
            schema: table.schema.clone(),
        })
    }

    #[must_use]
    pub fn from_table(table: &RecordBatch) -> Self {
        let columns = table
            .columns()
            .iter()
            .map(|col| ColumnRangeStatistics::from_series(col.as_materialized_series()))
            .collect();
        Self {
            columns,
            schema: table.schema.clone(),
        }
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    pub fn union(&self, other: &Self) -> crate::Result<Self> {
        if self.schema != other.schema {
            return Err(crate::Error::DaftCoreCompute {
                source: DaftError::SchemaMismatch(format!(
                    "TableStatistics::union requires schemas to match, found: {} vs {}",
                    self.schema, other.schema
                )),
            });
        }

        let columns = self
            .columns
            .iter()
            .zip(other.columns.iter())
            .map(|(l, r)| l.union(r))
            .collect::<crate::Result<_>>()?;

        Ok(Self {
            columns,
            schema: self.schema.clone(),
        })
    }

    pub fn eval_expression_list(&self, exprs: &[BoundExpr]) -> crate::Result<Self> {
        let columns = exprs
            .iter()
            .map(|e| self.eval_expression(e))
            .collect::<crate::Result<Vec<_>>>()?;

        let schema = Arc::new(Schema::new(
            exprs
                .iter()
                .map(|e| e.inner().to_field(&self.schema))
                .collect::<DaftResult<Vec<_>>>()
                .context(DaftCoreComputeSnafu)?,
        ));

        Ok(Self { columns, schema })
    }

    pub fn estimate_row_size(&self) -> super::Result<f64> {
        self.columns
            .iter()
            .filter_map(|col| col.element_size().transpose())
            .sum()
    }

    #[recursive::recursive]
    pub fn eval_expression(&self, expr: &BoundExpr) -> crate::Result<ColumnRangeStatistics> {
        match expr.as_ref() {
            Expr::Alias(col, _) => self.eval_expression(&BoundExpr::new_unchecked(col.clone())),
            Expr::Column(Column::Bound(BoundColumn { index, .. })) => {
                Ok(self.columns[*index].clone())
            }
            Expr::Literal(lit_value) => lit_value.clone().try_into(),
            Expr::Not(col) => self
                .eval_expression(&BoundExpr::new_unchecked(col.clone()))?
                .not(),
            Expr::BinaryOp { op, left, right } => {
                let lhs = self.eval_expression(&BoundExpr::new_unchecked(left.clone()))?;
                let rhs = self.eval_expression(&BoundExpr::new_unchecked(right.clone()))?;
                use daft_core::prelude::Operator::*;
                match op {
                    Lt => lhs.lt(&rhs),
                    LtEq => lhs.lte(&rhs),
                    Eq => lhs.equal(&rhs),
                    NotEq => lhs.not_equal(&rhs),
                    GtEq => lhs.gte(&rhs),
                    Gt => lhs.gt(&rhs),
                    Plus => &lhs + &rhs,
                    Minus => &lhs - &rhs,
                    And => lhs.bitand(&rhs),
                    Or => lhs.bitor(&rhs),
                    _ => Ok(ColumnRangeStatistics::Missing),
                }
            }
            Expr::IsIn(col, items) => self.eval_is_in(col, items),
            Expr::Cast(col, dtype, try_cast) => {
                let stats = self.eval_expression(&BoundExpr::new_unchecked(col.clone()))?;
                if *try_cast {
                    Ok(stats.cast(dtype).unwrap_or(ColumnRangeStatistics::Missing))
                } else {
                    stats.cast(dtype)
                }
            }
            _ => Ok(ColumnRangeStatistics::Missing),
        }
    }

    /// Whether `col` is one of `items`, judged by statistics.
    ///
    /// A short list is the disjunction of its equalities, which can rule out a
    /// range its values straddle. A longer one is judged by the range its
    /// values span, which costs one comparison whatever its length. Either way
    /// it is false only where no listed value can lie in the column's range.
    /// A list holding anything but literals is not judged.
    fn eval_is_in(&self, col: &ExprRef, items: &[ExprRef]) -> crate::Result<ColumnRangeStatistics> {
        if !items.iter().all(|item| matches!(item.as_ref(), Expr::Literal(_))) {
            return Ok(ColumnRangeStatistics::Missing);
        }
        let column = self.eval_expression(&BoundExpr::new_unchecked(col.clone()))?;
        let mut values = items
            .iter()
            .map(|item| self.eval_expression(&BoundExpr::new_unchecked(item.clone())));
        let Some(first) = values.next().transpose()? else {
            return ColumnRangeStatistics::try_from(Literal::Boolean(false));
        };
        if items.len() <= MAX_LISTED_EQUALITIES {
            values.try_fold(column.equal(&first)?, |any, value| {
                any.bitor(&column.equal(&value?)?)
            })
        } else {
            let span = values.try_fold(first, |span, value| span.union(&value?))?;
            column.equal(&span)
        }
    }

    #[deprecated(note = "name-referenced columns")]
    /// Casts a `TableStatistics` to a schema.
    ///
    /// Note: this method is deprecated because it maps fields by name, which will not work for schemas with duplicate field names.
    /// It should only be used for scans, and once we support reading files with duplicate column names, we should remove this function.
    pub fn cast_to_schema(&self, schema: &Schema) -> crate::Result<Self> {
        #[allow(deprecated)]
        self.cast_to_schema_with_fill(schema, None)
    }

    #[deprecated(note = "name-referenced columns")]
    /// Casts a `TableStatistics` to a schema, using `fill_map` to specify the default expression for a column that doesn't exist.
    ///
    /// Note: this method is deprecated because it maps fields by name, which will not work for schemas with duplicate field names.
    /// It should only be used for scans, and once we support reading files with duplicate column names, we should remove this function.
    pub fn cast_to_schema_with_fill(
        &self,
        schema: &Schema,
        fill_map: Option<&HashMap<&str, ExprRef>>,
    ) -> crate::Result<Self> {
        let current_col_names = HashSet::<_>::from_iter(self.schema.field_names());
        let null_lit = null_lit();
        let exprs: Vec<_> = schema
            .into_iter()
            .map(|field| {
                if current_col_names.contains(field.name.as_ref()) {
                    // For any fields already in the table, perform a cast
                    resolved_col(field.name.clone()).cast(&field.dtype)
                } else {
                    // For any fields in schema that are not in self.schema, use fill map to fill with an expression.
                    // If no entry for column name, fall back to null literal (i.e.s create a null array for that column).
                    fill_map
                        .as_ref()
                        .and_then(|m| m.get(field.name.as_ref()))
                        .unwrap_or(&null_lit)
                        .clone()
                        .alias(field.name.clone())
                        .cast(&field.dtype)
                }
            })
            .map(|expr| BoundExpr::try_new(expr, &self.schema))
            .collect::<DaftResult<_>>()
            .context(DaftCoreComputeSnafu)?;
        self.eval_expression_list(&exprs)
    }
}

impl Display for TableStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let columns = self
            .columns
            .iter()
            .zip(self.schema.as_ref())
            .map(|(c, s)| c.combined_series().unwrap().rename(&s.name))
            .collect::<Vec<_>>();
        let tbl_schema = Schema::new(columns.iter().map(|s| s.field().clone()));
        let tab = RecordBatch::new_with_size(tbl_schema, columns, 2).unwrap();
        write!(f, "{tab}")
    }
}

impl Index<usize> for TableStatistics {
    type Output = ColumnRangeStatistics;

    fn index(&self, index: usize) -> &Self::Output {
        &self.columns[index]
    }
}

impl<'a> IntoIterator for &'a TableStatistics {
    type Item = &'a ColumnRangeStatistics;
    type IntoIter = std::slice::Iter<'a, ColumnRangeStatistics>;

    fn into_iter(self) -> Self::IntoIter {
        self.columns.iter()
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use daft_core::prelude::*;
    use daft_dsl::{Expr, expr::bound_expr::BoundExpr, lit, null_lit, resolved_col};
    use daft_recordbatch::RecordBatch;
    use snafu::ResultExt;

    use super::TableStatistics;
    use crate::{DaftCoreComputeSnafu, column_stats::TruthValue};

    /// The truth of `column a IN values` over statistics of `a` spanning 10..=20.
    fn is_in(values: &[i64]) -> TruthValue {
        let table = RecordBatch::from_nonempty_columns(vec![
            Int64Array::from_slice("a", &[10, 20]).into_series(),
        ])
        .unwrap();
        let expr = BoundExpr::try_new(
            resolved_col("a").is_in(values.iter().map(|value| lit(*value)).collect()),
            &table.schema,
        )
        .unwrap();
        TableStatistics::from_table(&table)
            .eval_expression(&expr)
            .unwrap()
            .to_truth_value()
    }

    #[test]
    fn a_value_list_is_false_only_where_no_value_can_lie_in_the_range() {
        let long_straddling: Vec<i64> = (0..100).map(|v| v * 3).collect();
        let long_outside: Vec<i64> = (30..200).collect();
        let cases: [(&[i64], TruthValue); 6] = [
            (&[1, 2], TruthValue::False),
            (&[15], TruthValue::Maybe),
            (&[5, 25], TruthValue::False),
            (&[5, 12], TruthValue::Maybe),
            (&long_straddling, TruthValue::Maybe),
            (&long_outside, TruthValue::False),
        ];
        for (values, truth) in cases {
            assert_eq!(is_in(values), truth, "{values:?}");
        }
    }

    #[test]
    fn an_empty_value_list_is_false() {
        assert_eq!(is_in(&[]), TruthValue::False);
    }

    #[test]
    fn test_equal() -> crate::Result<()> {
        let table = RecordBatch::from_nonempty_columns(vec![
            Int64Array::from_slice("a", &[1, 2, 3, 4]).into_series(),
        ])
        .unwrap();
        let table_stats = TableStatistics::from_table(&table);

        // False case
        let expr = BoundExpr::try_new(resolved_col("a").eq(lit(0)), &table.schema)
            .context(DaftCoreComputeSnafu)?;
        let result = table_stats.eval_expression(&expr)?;
        assert_eq!(result.to_truth_value(), TruthValue::False);

        // Maybe case
        let expr = BoundExpr::try_new(resolved_col("a").eq(lit(3)), &table.schema)
            .context(DaftCoreComputeSnafu)?;
        let result = table_stats.eval_expression(&expr)?;
        assert_eq!(result.to_truth_value(), TruthValue::Maybe);

        // True case
        let table = RecordBatch::from_nonempty_columns(vec![
            Int64Array::from_slice("a", &[0, 0, 0]).into_series(),
        ])
        .unwrap();
        let table_stats = TableStatistics::from_table(&table);

        let expr = BoundExpr::try_new(resolved_col("a").eq(lit(0)), &table.schema)
            .context(DaftCoreComputeSnafu)?;
        let result = table_stats.eval_expression(&expr)?;
        assert_eq!(result.to_truth_value(), TruthValue::True);

        Ok(())
    }

    #[test]
    fn test_not_null_literal_is_maybe() -> crate::Result<()> {
        // NOT null_lit() simulates a predicate on a column absent from a Parquet file
        // (schema evolution). The stats evaluator must conservatively return Maybe so
        // the row group is never incorrectly pruned.
        let table = RecordBatch::from_nonempty_columns(vec![
            Int64Array::from_slice("a", &[1, 2, 3]).into_series(),
        ])
        .unwrap();
        let table_stats = TableStatistics::from_table(&table);

        let not_null = Arc::new(Expr::Not(null_lit()));
        let bound = BoundExpr::try_new(not_null, &table.schema).context(DaftCoreComputeSnafu)?;
        let result = table_stats.eval_expression(&bound)?;
        assert_eq!(result.to_truth_value(), TruthValue::Maybe);

        Ok(())
    }

    #[test]
    fn test_missing_string_col_eq_is_maybe() -> crate::Result<()> {
        // WHERE missing_str_col = 'foo' — non-boolean column absent from file.
        // null_lit() = lit("foo") must be Maybe (not an error, not False).
        let table = RecordBatch::from_nonempty_columns(vec![
            Int64Array::from_slice("a", &[1, 2, 3]).into_series(),
        ])
        .unwrap();
        let table_stats = TableStatistics::from_table(&table);

        let pred = Arc::new(Expr::BinaryOp {
            op: daft_core::prelude::Operator::Eq,
            left: null_lit(),
            right: lit("foo"),
        });
        let bound = BoundExpr::try_new(pred, &table.schema).context(DaftCoreComputeSnafu)?;
        let result = table_stats.eval_expression(&bound)?;
        assert_eq!(result.to_truth_value(), TruthValue::Maybe);

        Ok(())
    }

    #[test]
    fn test_missing_int_col_eq_is_maybe() -> crate::Result<()> {
        // WHERE missing_int_col = 42 — numeric column absent from file.
        let table = RecordBatch::from_nonempty_columns(vec![
            Utf8Array::from_slice("name", &["x"]).into_series(),
        ])
        .unwrap();
        let table_stats = TableStatistics::from_table(&table);

        let pred = Arc::new(Expr::BinaryOp {
            op: daft_core::prelude::Operator::Eq,
            left: null_lit(),
            right: lit(42i64),
        });
        let bound = BoundExpr::try_new(pred, &table.schema).context(DaftCoreComputeSnafu)?;
        let result = table_stats.eval_expression(&bound)?;
        assert_eq!(result.to_truth_value(), TruthValue::Maybe);

        Ok(())
    }
}
