use common_error::{DaftError, DaftResult};
use daft_core::{count_mode::CountMode, prelude::Schema};

use crate::{AggExpr, ApproxPercentileParams, Expr, ExprRef, lit};

#[recursive::recursive]
pub fn extract_agg_expr(expr: &ExprRef) -> DaftResult<AggExpr> {
    match expr.as_ref() {
        Expr::Agg(agg_expr) => Ok(agg_expr.clone()),
        Expr::Alias(e, name) => extract_agg_expr(e).map(|agg_expr| {
            // reorder expressions so that alias goes before agg
            match agg_expr {
                AggExpr::Count(e, count_mode) => {
                    AggExpr::Count(Expr::Alias(e, name.clone()).into(), count_mode)
                }
                AggExpr::CountDistinct(e) => {
                    AggExpr::CountDistinct(Expr::Alias(e, name.clone()).into())
                }
                AggExpr::Sum(e) => AggExpr::Sum(Expr::Alias(e, name.clone()).into()),
                AggExpr::Product(e) => AggExpr::Product(Expr::Alias(e, name.clone()).into()),
                AggExpr::ApproxPercentile(ApproxPercentileParams {
                    child: e,
                    percentiles,
                    force_list_output,
                }) => AggExpr::ApproxPercentile(ApproxPercentileParams {
                    child: Expr::Alias(e, name.clone()).into(),
                    percentiles,
                    force_list_output,
                }),
                AggExpr::ApproxCountDistinct(e) => {
                    AggExpr::ApproxCountDistinct(Expr::Alias(e, name.clone()).into())
                }
                AggExpr::ApproxSketch(e, sketch_type) => {
                    AggExpr::ApproxSketch(Expr::Alias(e, name.clone()).into(), sketch_type)
                }
                AggExpr::MergeSketch(e, sketch_type) => {
                    AggExpr::MergeSketch(Expr::Alias(e, name.clone()).into(), sketch_type)
                }
                AggExpr::Mean(e) => AggExpr::Mean(Expr::Alias(e, name.clone()).into()),
                AggExpr::Percentile(e, percentile) => {
                    AggExpr::Percentile(Expr::Alias(e, name.clone()).into(), percentile)
                }
                AggExpr::Stddev(e, ddof) => {
                    AggExpr::Stddev(Expr::Alias(e, name.clone()).into(), ddof)
                }
                AggExpr::Var(e, ddof) => AggExpr::Var(Expr::Alias(e, name.clone()).into(), ddof),
                AggExpr::Min(e) => AggExpr::Min(Expr::Alias(e, name.clone()).into()),
                AggExpr::Max(e) => AggExpr::Max(Expr::Alias(e, name.clone()).into()),
                AggExpr::BoolAnd(e) => AggExpr::BoolAnd(Expr::Alias(e, name.clone()).into()),
                AggExpr::BoolOr(e) => AggExpr::BoolOr(Expr::Alias(e, name.clone()).into()),
                AggExpr::AnyValue(e, ignore_nulls) => {
                    AggExpr::AnyValue(Expr::Alias(e, name.clone()).into(), ignore_nulls)
                }
                AggExpr::SingleValue(e) => {
                    AggExpr::SingleValue(Expr::Alias(e, name.clone()).into())
                }
                AggExpr::List(e) => AggExpr::List(Expr::Alias(e, name.clone()).into()),
                AggExpr::Set(e) => AggExpr::Set(Expr::Alias(e, name.clone()).into()),
                AggExpr::Concat(e, delimiter) => {
                    AggExpr::Concat(Expr::Alias(e, name.clone()).into(), delimiter)
                }
                AggExpr::Median(e) => AggExpr::Median(Expr::Alias(e, name.clone()).into()),
                AggExpr::Skew(e) => AggExpr::Skew(Expr::Alias(e, name.clone()).into()),
                AggExpr::MapGroups { func, inputs } => AggExpr::MapGroups {
                    func,
                    inputs: inputs
                        .into_iter()
                        .map(|input| input.alias(name.clone()))
                        .collect(),
                },
                AggExpr::AggFn { handle, inputs } => AggExpr::AggFn {
                    handle,
                    inputs: inputs
                        .into_iter()
                        .map(|input| input.alias(name.clone()))
                        .collect(),
                },
                AggExpr::AggFnMap { .. }
                | AggExpr::AggFnCombine { .. }
                | AggExpr::AggFnReduce { .. } => {
                    unreachable!(
                        "AggFnMap / AggFnCombine / AggFnReduce are planner-internal expressions \
                         and must not appear in a user-level alias context"
                    )
                }
            }
        }),
        _ => Err(DaftError::InternalError(format!(
            "Expected non-agg expressions in aggregation to be factored out before plan translation. Got: {:?}",
            expr
        ))),
    }
}

impl AggExpr {
    /// The aggregation that combines partial results of `self` into its whole result,
    /// when that is an aggregation of the same kind over the partials.
    ///
    /// Counts combine by summing, and sums, minima, maxima and boolean and/or by
    /// themselves; any value combines by taking any non-null partial. `None` for an
    /// aggregation whose partials need a different shape (a mean, a distinct count, a
    /// list, a sketch). `partial` is the column holding the partial results.
    #[must_use]
    pub fn merge_of_partial(&self, partial: ExprRef) -> Option<Self> {
        match self {
            Self::Count(..) | Self::Sum(_) => Some(Self::Sum(partial)),
            Self::Min(_) => Some(Self::Min(partial)),
            Self::Max(_) => Some(Self::Max(partial)),
            Self::BoolAnd(_) => Some(Self::BoolAnd(partial)),
            Self::BoolOr(_) => Some(Self::BoolOr(partial)),
            Self::AnyValue(..) => Some(Self::AnyValue(partial, true)),
            _ => None,
        }
    }

    /// The partial result of `self` over a group of one row, as an expression over that
    /// row, of the type `self` answers; `None` when [`Self::merge_of_partial`] is.
    ///
    /// Combining these per-row results with [`Self::merge_of_partial`] gives the same
    /// answer as aggregating the rows, so a partial aggregation may pass rows on this way
    /// instead of grouping them.
    ///
    /// # Errors
    ///
    /// Returns an error when `self` cannot be typed against `schema`.
    pub fn single_row_partial(&self, schema: &Schema) -> DaftResult<Option<ExprRef>> {
        let state = match self {
            Self::Count(_, CountMode::All) => lit(1u64),
            Self::Count(child, CountMode::Valid) => child.clone().not_null(),
            Self::Count(child, CountMode::Null) => child.clone().is_null(),
            Self::Sum(child)
            | Self::Min(child)
            | Self::Max(child)
            | Self::BoolAnd(child)
            | Self::BoolOr(child)
            | Self::AnyValue(child, _) => child.clone(),
            _ => return Ok(None),
        };
        // A state already of the partial's type is the argument itself, not a copy of it.
        let dtype = self.to_field(schema)?.dtype;
        Ok(Some(if state.to_field(schema)?.dtype == dtype {
            state
        } else {
            state.cast(&dtype)
        }))
    }
}
