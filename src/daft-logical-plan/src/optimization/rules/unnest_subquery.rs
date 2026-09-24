use std::{collections::HashSet, sync::Arc};

use common_error::{DaftError, DaftResult};
use common_treenode::{DynTreeNode, Transformed, TreeNode};
use daft_algebra::boolean::{combine_conjunction, split_conjunction};
use daft_core::{
    count_mode::CountMode,
    join::JoinType,
    prelude::{Operator, SchemaRef},
};
use daft_dsl::{
    Column, Expr, ExprRef, ResolvedColumn, Subquery, functions::agg::single_row_value, resolved_col,
};
use itertools::multiunzip;
use uuid::Uuid;

use super::OptimizerRule;
use crate::{
    LogicalPlan, LogicalPlanRef,
    logical_plan::downcast_subquery,
    ops::{Aggregate, Filter, Join, Limit, Project, Sort, join::JoinPredicate},
};

/// Rewriter rule to convert scalar subqueries into joins.
///
/// ## Examples
/// ### Example 1 - Uncorrelated subquery
/// Before:
/// ```sql
/// SELECT val
/// FROM tbl1
/// WHERE key = (SELECT max(key) FROM tbl2)
/// ```
/// After:
/// ```sql
/// SELECT val
/// FROM tbl1
/// CROSS JOIN (SELECT max(key) FROM tbl2) AS subquery
/// WHERE key = subquery.key  -- this can be then pushed into join in a future rule
/// ```
///
/// ### Example 2 - Correlated subquery
/// Before:
/// ```sql
/// SELECT val
/// FROM tbl1
/// WHERE outer_key =
///     (
///         SELECT max(outer_key)
///         FROM tbl2
///         WHERE inner_key = tbl1.inner_key
///     )
/// ```
/// After:
/// ```sql
/// SELECT val
/// FROM tbl1
/// LEFT JOIN
///     (
///         SELECT inner_key, max(outer_key)
///         FROM tbl2
///         GROUP BY inner_key
///     ) AS subquery
/// ON inner_key
/// WHERE outer_key = subquery.outer_key
/// ```
///
/// ### Cardinality
/// A scalar subquery answers one value per outer row: null when it yields no row, and a
/// failure when it yields more than one. A join alone gives neither.
///
/// A subquery not already known to yield at most one row per correlation key (exactly one
/// row, when uncorrelated) keeps one value and a row count per key, and the value is
/// checked on each outer row that reaches it, as if evaluated once per row: a key that no
/// outer row asks for never fails the query.
#[derive(Debug)]
pub struct UnnestScalarSubquery {}

impl UnnestScalarSubquery {
    pub fn new() -> Self {
        Self {}
    }
}

impl UnnestScalarSubquery {
    fn unnest_subqueries(
        input: LogicalPlanRef,
        exprs: &[ExprRef],
    ) -> DaftResult<Transformed<(LogicalPlanRef, Vec<ExprRef>)>> {
        let mut subqueries = HashSet::new();

        let new_exprs = exprs
            .iter()
            .map(|expr| {
                expr.clone()
                    .transform_down(|e| {
                        if let Expr::Subquery(subquery) = e.as_ref() {
                            subqueries.insert(subquery.clone());

                            Ok(Transformed::yes(resolved_col(subquery.semantic_id().id)))
                        } else {
                            Ok(Transformed::no(e))
                        }
                    })
                    .unwrap()
                    .data
            })
            .collect();

        if subqueries.is_empty() {
            return Ok(Transformed::no((input, new_exprs)));
        }

        // The value and row-count columns of each subquery whose rows must be checked
        // where they are joined back.
        let mut checked = Vec::new();

        let new_input = subqueries
            .into_iter()
            .try_fold(input, |curr_input, subquery| {
                let subquery_alias = subquery.semantic_id().id;
                let subquery_plan = downcast_subquery(&subquery);

                let subquery_col_names = subquery_plan.schema().names();
                let [output_col] = subquery_col_names.as_slice() else {
                    return Err(DaftError::ValueError(format!(
                        "Expected scalar subquery to have one output column, received: {}",
                        subquery_col_names.len()
                    )));
                };

                // alias output column
                let subquery_plan = Arc::new(LogicalPlan::Project(Project::try_new(
                    subquery_plan,
                    vec![resolved_col(output_col.as_str()).alias(subquery_alias.clone())],
                )?));

                let (decorrelated_subquery, subquery_on, input_on) =
                    pull_up_correlated_cols(subquery_plan)?;
                let (decorrelated_subquery, rows_col) =
                    one_row_per_key(decorrelated_subquery, &subquery_alias, &subquery_on)?;
                if let Some(rows_col) = rows_col {
                    checked.push((subquery_alias, rows_col));
                }

                let on_expr =
                    combine_conjunction(input_on.into_iter().zip(subquery_on.into_iter()).map(
                        |(i, s)| {
                            let i_left = i
                                .to_left_cols(curr_input.schema())
                                .expect("input columns to be in curr_input");
                            let s_right = s
                                .to_right_cols(decorrelated_subquery.schema())
                                .expect("subquery columns to be in decorrelated_subquery");

                            i_left.eq(s_right)
                        },
                    ));

                // use inner join when uncorrelated so that filter can be pushed into join and other optimizations;
                // an uncorrelated subquery now yields exactly one row, so no outer row is lost or repeated
                let join_type = if on_expr.is_none() {
                    JoinType::Inner
                } else {
                    JoinType::Left
                };

                let (curr_input, decorrelated_subquery, on_expr) = Join::deduplicate_join_columns(
                    curr_input,
                    decorrelated_subquery,
                    on_expr,
                    &[],
                    join_type,
                    Default::default(),
                )?;

                let on = JoinPredicate::try_new(on_expr)?;

                Ok(Arc::new(LogicalPlan::Join(Join::try_new(
                    curr_input,
                    decorrelated_subquery,
                    on,
                    join_type,
                    None,
                )?)))
            })?;

        let new_input = check_joined_rows(new_input, &checked)?;

        Ok(Transformed::yes((new_input, new_exprs)))
    }
}

impl OptimizerRule for UnnestScalarSubquery {
    fn try_optimize(&self, plan: Arc<LogicalPlan>) -> DaftResult<Transformed<Arc<LogicalPlan>>> {
        plan.transform_down(|node| match node.as_ref() {
            LogicalPlan::Filter(Filter {
                input, predicate, ..
            }) => {
                let unnest_result =
                    Self::unnest_subqueries(input.clone(), &split_conjunction(predicate))?;

                if !unnest_result.transformed {
                    return Ok(Transformed::no(node));
                }

                let (new_input, new_predicates) = unnest_result.data;

                let new_predicate = combine_conjunction(new_predicates)
                    .expect("predicates are guaranteed to exist at this point, so 'conjunct' should never return 'None'");

                let new_filter = Arc::new(LogicalPlan::Filter(Filter::try_new(
                    new_input,
                    new_predicate,
                )?));

                // preserve original schema
                let new_plan = Arc::new(LogicalPlan::Project(Project::new_from_schema(
                    new_filter,
                    input.schema(),
                )?));

                Ok(Transformed::yes(new_plan))
            }
            LogicalPlan::Project(Project {
                input, projection, ..
            }) => {
                let unnest_result =
                    Self::unnest_subqueries(input.clone(), projection)?;

                if !unnest_result.transformed {
                    return Ok(Transformed::no(node));
                }

                let (new_input, new_projection) = unnest_result.data;

                // preserve original schema
                let new_plan = Arc::new(LogicalPlan::Project(Project::try_new(
                    new_input,
                    new_projection,
                )?));

                Ok(Transformed::yes(new_plan))
            }
            _ => Ok(Transformed::no(node)),
        })
    }
}

/// Rewriter rule to convert IN and EXISTS subqueries into joins.
///
/// ## Examples
/// ### Example 1 - Uncorrelated `IN` Query
/// Before:
/// ```sql
/// SELECT val
/// FROM tbl1
/// WHERE key IN (SELECT key FROM tbl2)
/// ```
/// After:
/// ```sql
/// SELECT val
/// FROM tbl1
/// SEMI JOIN (SELECT key FROM tbl2) AS subquery
/// ON key = subquery.key
/// ```
///
/// ### Example 2 - Correlated `NOT EXISTS` Query
/// Before:
/// ```sql
/// SELECT val
/// FROM tbl1
/// WHERE NOT EXISTS
///     (
///         SELECT *
///         FROM tbl2
///         WHERE key = tbl1.key
///     )
/// ```
///
/// After:
/// ```sql
/// SELECT val
/// FROM tbl1
/// ANTI JOIN (SELECT * FROM tbl2) AS subquery
/// ON key = subquery.key
/// ```
#[derive(Debug)]
pub struct UnnestPredicateSubquery {}

impl UnnestPredicateSubquery {
    pub fn new() -> Self {
        Self {}
    }
}

#[derive(Eq, Hash, PartialEq)]
struct PredicateSubquery {
    pub subquery: Subquery,
    pub in_expr: Option<ExprRef>,
    pub join_type: JoinType,
}

impl OptimizerRule for UnnestPredicateSubquery {
    fn try_optimize(&self, plan: Arc<LogicalPlan>) -> DaftResult<Transformed<Arc<LogicalPlan>>> {
        plan.transform_down(|node| match node.as_ref() {
            LogicalPlan::Filter(Filter {
                input, predicate, ..
            }) => {
                let mut subqueries = HashSet::new();

                let new_predicates = split_conjunction(predicate)
                    .into_iter()
                    .filter(|expr| {
                        match expr.as_ref() {
                            Expr::InSubquery(in_expr, subquery) => {
                                subqueries.insert(PredicateSubquery { subquery: subquery.clone(), in_expr: Some(in_expr.clone()), join_type: JoinType::Semi });
                                false
                            }
                            Expr::Exists(subquery) => {
                                subqueries.insert(PredicateSubquery { subquery: subquery.clone(), in_expr: None, join_type: JoinType::Semi });
                                false
                            }
                            Expr::Not(e) => {
                                match e.as_ref() {
                                    Expr::InSubquery(in_expr, subquery) => {
                                        subqueries.insert(PredicateSubquery { subquery: subquery.clone(), in_expr: Some(in_expr.clone()), join_type: JoinType::Anti });
                                        false
                                    }
                                    Expr::Exists(subquery) => {
                                        subqueries.insert(PredicateSubquery { subquery: subquery.clone(), in_expr: None, join_type: JoinType::Anti });
                                        false
                                    }
                                    _ => true
                                }
                            }
                            _ => true
                        }
                    })
                    .collect::<Vec<_>>();

                if subqueries.is_empty() {
                    return Ok(Transformed::no(node));
                }

                let new_input = subqueries.into_iter().try_fold(input.clone(), |curr_input, PredicateSubquery { subquery, in_expr, join_type }| {
                    let subquery_plan = downcast_subquery(&subquery);
                    let subquery_schema = subquery_plan.schema();

                    let (decorrelated_subquery, mut subquery_on, mut input_on) =
                        pull_up_correlated_cols(subquery_plan)?;

                    if let Some(in_expr) = in_expr {
                        let subquery_col_names = subquery_schema.names();
                        let [output_col] = subquery_col_names.as_slice() else {
                            return Err(DaftError::ValueError(format!("Expected IN subquery to have one output column, received: {}", subquery_col_names.len())));
                        };

                        input_on.push(in_expr);
                        subquery_on.push(resolved_col(output_col.as_str()));
                    }

                    if subquery_on.is_empty() {
                        return Err(DaftError::ValueError("Expected IN/EXISTS subquery to be correlated, found uncorrelated subquery.".to_string()));
                    }

                    let on_expr = combine_conjunction(input_on.into_iter().zip(subquery_on.into_iter()).map(
                        |(i, s)| {
                            let i_left = i
                                .to_left_cols(curr_input.schema())
                                .expect("input columns to be in curr_input");
                            let s_right = s
                                .to_right_cols(decorrelated_subquery.schema())
                                .expect("subquery columns to be in decorrelated_subquery");

                            i_left.eq(s_right)
                        },
                    ));

                    let on = JoinPredicate::try_new(on_expr)?;

                    Ok(Arc::new(LogicalPlan::Join(Join::try_new(
                        curr_input,
                        decorrelated_subquery,
                        on,
                        join_type,
                        None,
                    )?)))
                })?;

                let new_plan = if let Some(new_predicate) = combine_conjunction(new_predicates) {
                    // add filter back if there are non-subquery predicates
                    Arc::new(LogicalPlan::Filter(Filter::try_new(
                        new_input,
                        new_predicate,
                    )?))
                } else {
                    new_input
                };

                Ok(Transformed::yes(new_plan))
            }
            _ => Ok(Transformed::no(node)),
        })
    }
}

fn pull_up_correlated_cols(
    plan: LogicalPlanRef,
) -> DaftResult<(LogicalPlanRef, Vec<ExprRef>, Vec<ExprRef>)> {
    let (new_inputs, subquery_on, outer_on): (Vec<_>, Vec<_>, Vec<_>) = multiunzip(
        plan.arc_children()
            .into_iter()
            .map(pull_up_correlated_cols)
            .collect::<DaftResult<Vec<_>>>()?,
    );

    let plan = if new_inputs.is_empty() {
        plan
    } else {
        Arc::new(plan.with_new_children(&new_inputs))
    };

    let mut subquery_on = subquery_on.into_iter().flatten().collect::<Vec<_>>();
    let mut outer_on = outer_on.into_iter().flatten().collect::<Vec<_>>();

    match plan.as_ref() {
        LogicalPlan::Filter(Filter {
            input, predicate, ..
        }) => {
            let mut found_correlated_col = false;

            let preds = split_conjunction(predicate)
                .into_iter()
                .filter(|expr| {
                    if let Expr::BinaryOp {
                        op: Operator::Eq,
                        left,
                        right,
                    } = expr.as_ref()
                    {
                        match (left.as_ref(), right.as_ref()) {
                            (
                                Expr::Column(Column::Resolved(ResolvedColumn::Basic(
                                    subquery_col_name,
                                ))),
                                Expr::Column(Column::Resolved(ResolvedColumn::OuterRef(
                                    outer_field,
                                    _,
                                ))),
                            )
                            | (
                                Expr::Column(Column::Resolved(ResolvedColumn::OuterRef(
                                    outer_field,
                                    _,
                                ))),
                                Expr::Column(Column::Resolved(ResolvedColumn::Basic(
                                    subquery_col_name,
                                ))),
                            ) => {
                                // remove correlated col from filter, use in join instead
                                subquery_on.push(resolved_col(subquery_col_name.clone()));
                                outer_on.push(resolved_col(outer_field.name.as_ref()));

                                found_correlated_col = true;
                                return false;
                            }
                            _ => {}
                        }
                    }

                    true
                })
                .collect::<Vec<_>>();

            // no new correlated cols found
            if !found_correlated_col {
                return Ok((plan.clone(), subquery_on, outer_on));
            }

            if let Some(new_predicate) = combine_conjunction(preds) {
                let new_plan = Arc::new(LogicalPlan::Filter(Filter::try_new(
                    input.clone(),
                    new_predicate,
                )?));

                Ok((new_plan, subquery_on, outer_on))
            } else {
                // all predicates are correlated so filter can be completely removed
                Ok((input.clone(), subquery_on, outer_on))
            }
        }
        LogicalPlan::Project(Project {
            input,
            projection,
            projected_schema,
            ..
        }) => {
            // ensure all columns that need to be pulled up are in the projection

            let (new_subquery_on, missing_exprs) =
                get_missing_exprs(subquery_on, projection, projected_schema);

            if missing_exprs.is_empty() {
                // project already contains all necessary columns
                Ok((plan.clone(), new_subquery_on, outer_on))
            } else {
                let new_projection = [projection.clone(), missing_exprs].concat();

                let new_plan = Arc::new(LogicalPlan::Project(Project::try_new(
                    input.clone(),
                    new_projection,
                )?));

                Ok((new_plan, new_subquery_on, outer_on))
            }
        }
        LogicalPlan::Aggregate(Aggregate {
            input,
            aggregations,
            groupby,
            output_schema,
            ..
        }) => {
            // put columns that need to be pulled up into the groupby

            let (new_subquery_on, missing_groupbys) =
                get_missing_exprs(subquery_on, groupby, output_schema);

            if missing_groupbys.is_empty() {
                // agg already contains all necessary columns
                Ok((plan.clone(), new_subquery_on, outer_on))
            } else {
                let new_groupby = [groupby.clone(), missing_groupbys].concat();

                let new_plan = Arc::new(LogicalPlan::Aggregate(Aggregate::try_new(
                    input.clone(),
                    aggregations.clone(),
                    new_groupby,
                )?));

                Ok((new_plan, new_subquery_on, outer_on))
            }
        }

        // ops that can trivially pull up correlated cols
        LogicalPlan::Distinct(..)
        | LogicalPlan::MonotonicallyIncreasingId(..)
        | LogicalPlan::Repartition(..)
        | LogicalPlan::IntoPartitions(..)
        | LogicalPlan::IntoBatches(..)
        | LogicalPlan::Union(..)
        | LogicalPlan::Intersect(..)
        | LogicalPlan::Sort(..)
        | LogicalPlan::Shuffle(..)
        | LogicalPlan::SubqueryAlias(..) => Ok((plan.clone(), subquery_on, outer_on)),

        // ops that cannot pull up correlated columns
        LogicalPlan::UDFProject(..)
        | LogicalPlan::Limit(..)
        | LogicalPlan::Offset(..)
        | LogicalPlan::Shard(..)
        | LogicalPlan::TopN(..)
        | LogicalPlan::Sample(..)
        | LogicalPlan::Source(..)
        | LogicalPlan::Explode(..)
        | LogicalPlan::Unpivot(..)
        | LogicalPlan::Pivot(..)
        | LogicalPlan::Concat(..)
        | LogicalPlan::Join(..)
        | LogicalPlan::AsofJoin(..)
        | LogicalPlan::Sink(..)
        | LogicalPlan::Window(..)
        | LogicalPlan::VLLMProject(..)
        | LogicalPlan::StageCheckpointKeys(..) => {
            if subquery_on.is_empty() {
                Ok((plan.clone(), vec![], vec![]))
            } else {
                Err(DaftError::NotImplemented(format!(
                    "Pulling up correlated columns not supported for: {}",
                    plan.name()
                )))
            }
        }
    }
}

/// Makes `subquery` yield at most one row for each value of `keys`, or exactly one row
/// when there are no keys, and says which rows must still be checked once joined.
///
/// A subquery that already does is returned unchanged. Any other keeps one value per key
/// (one value in all, without keys) along with the number of rows it held, and the name of
/// that count is returned for [`check_joined_rows`]. The check is left to the rows that
/// reach the value because only they ask for it: a key that no outer row asks for, or a
/// subquery that no row reaches, never fails the query.
fn one_row_per_key(
    subquery: LogicalPlanRef,
    value: &Arc<str>,
    keys: &[ExprRef],
) -> DaftResult<(LogicalPlanRef, Option<Arc<str>>)> {
    let already_one = if keys.is_empty() {
        yields_exactly_one_row(&subquery)
    } else {
        let key_names = keys.iter().map(|key| key.name()).collect::<Vec<_>>();
        yields_at_most_one_row_per(&subquery, &key_names)
    };
    if already_one {
        return Ok((subquery, None));
    }
    // Without keys this is a global aggregation, which yields exactly one row: a null
    // value and a count of zero when the subquery yields none.
    let rows: Arc<str> = format!("{value}.rows").into();
    let per_key = Aggregate::try_new(
        subquery,
        vec![
            // Nulls are ignored so the value never comes from an empty partial; a key
            // whose rows all hold null still answers null.
            resolved_col(value.clone()).any_value(true),
            resolved_col(value.clone())
                .count(CountMode::All)
                .alias(rows.clone()),
        ],
        keys.to_vec(),
    )?;
    Ok((Arc::new(LogicalPlan::Aggregate(per_key)), Some(rows)))
}

/// Replaces each checked subquery value in `joined` with that value checked row by row,
/// and drops the row counts that checked it.
///
/// The check is computed in its own projection above the joins. A filter that reads the
/// value cannot move below a projection that computes it, so it is only ever evaluated on
/// rows that reached the join, never on the keys of the subquery alone.
fn check_joined_rows(
    joined: LogicalPlanRef,
    checked: &[(Arc<str>, Arc<str>)],
) -> DaftResult<LogicalPlanRef> {
    if checked.is_empty() {
        return Ok(joined);
    }
    let projection = joined
        .schema()
        .field_names()
        .filter(|name| !checked.iter().any(|(_, rows)| rows.as_ref() == *name))
        .map(
            |name| match checked.iter().find(|(value, _)| value.as_ref() == name) {
                Some((value, rows)) => {
                    single_row_value(resolved_col(value.clone()), resolved_col(rows.clone()))
                        .alias(value.clone())
                }
                None => resolved_col(name),
            },
        )
        .collect();
    Ok(Arc::new(LogicalPlan::Project(Project::try_new(
        joined, projection,
    )?)))
}

/// Whether `plan` yields exactly one row whatever its input holds: a global aggregation,
/// seen through projections.
fn yields_exactly_one_row(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::Project(Project { input, .. }) => yields_exactly_one_row(input),
        LogicalPlan::Aggregate(Aggregate { groupby, .. }) => groupby.is_empty(),
        _ => false,
    }
}

/// Whether `plan` yields at most one row for each value of the columns named `keys`.
///
/// Proven only for an aggregation grouped by nothing but keys, seen through projections
/// that pass the keys through and through operators that only drop or reorder rows.
/// Anything else answers `false`, which costs an aggregation but never a wrong answer.
fn yields_at_most_one_row_per(plan: &LogicalPlan, keys: &[&str]) -> bool {
    match plan {
        LogicalPlan::Project(Project {
            input, projection, ..
        }) => {
            let input_keys = keys
                .iter()
                .map(|key| {
                    projection
                        .iter()
                        .find(|expr| expr.name() == *key)
                        .and_then(|expr| passed_through_column(expr))
                })
                .collect::<Option<Vec<_>>>();
            input_keys.is_some_and(|input_keys| yields_at_most_one_row_per(input, &input_keys))
        }
        LogicalPlan::Filter(Filter { input, .. })
        | LogicalPlan::Sort(Sort { input, .. })
        | LogicalPlan::Limit(Limit { input, .. }) => yields_at_most_one_row_per(input, keys),
        LogicalPlan::Aggregate(Aggregate { groupby, .. }) => {
            groupby.iter().all(|group| keys.contains(&group.name()))
        }
        _ => false,
    }
}

/// The name of the input column `expr` passes through unchanged, if it does.
fn passed_through_column(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Alias(inner, _) => passed_through_column(inner),
        Expr::Column(Column::Resolved(ResolvedColumn::Basic(name))) => Some(name),
        _ => None,
    }
}

fn get_missing_exprs(
    subquery_on: Vec<ExprRef>,
    existing_exprs: &[ExprRef],
    schema: &SchemaRef,
) -> (Vec<ExprRef>, Vec<ExprRef>) {
    let mut new_subquery_on = Vec::new();
    let mut missing_exprs = Vec::new();

    for expr in subquery_on {
        if existing_exprs.contains(&expr) {
            // column already exists in schema
            new_subquery_on.push(expr);
        } else if schema.has_field(expr.name()) {
            // another expression takes pull up column name, we rename the pull up column.
            let new_name = format!("{}-{}", expr.name(), Uuid::new_v4());

            new_subquery_on.push(resolved_col(new_name.clone()));
            missing_exprs.push(expr.alias(new_name));
        } else {
            // missing from schema, can keep original name

            new_subquery_on.push(expr.clone());
            missing_exprs.push(expr);
        }
    }

    (new_subquery_on, missing_exprs)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use common_error::DaftResult;
    use daft_core::{count_mode::CountMode, join::JoinType};
    use daft_dsl::{
        Column, Expr, PlanRef, ResolvedColumn, Subquery, functions::agg::single_row_value,
        unresolved_col,
    };
    use daft_schema::{dtype::DataType, field::Field};

    use super::{UnnestPredicateSubquery, UnnestScalarSubquery};
    use crate::{
        LogicalPlanRef,
        optimization::{
            optimizer::{RuleBatch, RuleExecutionStrategy},
            test::assert_optimized_plan_with_rules_eq,
        },
        test::{dummy_scan_node, dummy_scan_operator},
    };

    fn assert_scalar_optimized_plan_eq(
        plan: LogicalPlanRef,
        expected: LogicalPlanRef,
    ) -> DaftResult<()> {
        assert_optimized_plan_with_rules_eq(
            plan,
            expected,
            vec![RuleBatch::new(
                vec![Box::new(UnnestScalarSubquery::new())],
                RuleExecutionStrategy::Once,
            )],
        )
    }

    fn assert_predicate_optimized_plan_eq(
        plan: LogicalPlanRef,
        expected: LogicalPlanRef,
    ) -> DaftResult<()> {
        assert_optimized_plan_with_rules_eq(
            plan,
            expected,
            vec![RuleBatch::new(
                vec![Box::new(UnnestPredicateSubquery::new())],
                RuleExecutionStrategy::Once,
            )],
        )
    }

    #[test]
    fn uncorrelated_scalar_subquery() -> DaftResult<()> {
        let tbl1 = dummy_scan_node(dummy_scan_operator(vec![
            Field::new("key", DataType::Int64),
            Field::new("val", DataType::Int64),
        ]));

        let tbl2 = dummy_scan_node(dummy_scan_operator(vec![Field::new(
            "key",
            DataType::Int64,
        )]));

        let subquery = tbl2.aggregate(vec![unresolved_col("key").max()], vec![])?;
        let subquery_expr = Arc::new(Expr::Subquery(Subquery {
            plan: subquery.build(),
        }));
        let subquery_alias = subquery_expr.semantic_id(&subquery.schema()).id;

        let plan = tbl1
            .filter(unresolved_col("key").eq(subquery_expr))?
            .select(vec![unresolved_col("val")])?
            .build();

        let expected = tbl1
            .join(
                subquery.select(vec![unresolved_col("key").alias(subquery_alias.clone())])?,
                None,
                vec![],
                JoinType::Inner,
                None,
                Default::default(),
            )?
            .filter(unresolved_col("key").eq(unresolved_col(subquery_alias)))?
            .select(vec![unresolved_col("key"), unresolved_col("val")])?
            .select(vec![unresolved_col("val")])?
            .build();

        assert_scalar_optimized_plan_eq(plan, expected)?;
        Ok(())
    }

    #[test]
    fn correlated_scalar_subquery() -> DaftResult<()> {
        let tbl1 = dummy_scan_node(dummy_scan_operator(vec![
            Field::new("outer_key", DataType::Int64),
            Field::new("inner_key", DataType::Int64),
            Field::new("val", DataType::Int64),
        ]));

        let tbl2 = dummy_scan_node(dummy_scan_operator(vec![
            Field::new("outer_key", DataType::Int64),
            Field::new("inner_key2", DataType::Int64),
        ]));

        let subquery =
            tbl2.filter(unresolved_col("inner_key2").eq(Arc::new(Expr::Column(
                Column::Resolved(ResolvedColumn::OuterRef(
                    Field::new("inner_key", DataType::Int64),
                    PlanRef::Unqualified,
                )),
            ))))?
            .aggregate(vec![unresolved_col("outer_key").max()], vec![])?;
        let subquery_expr = Arc::new(Expr::Subquery(Subquery {
            plan: subquery.build(),
        }));
        let subquery_alias = subquery_expr.semantic_id(&subquery.schema()).id;

        let plan = tbl1
            .filter(unresolved_col("outer_key").eq(subquery_expr))?
            .select(vec![unresolved_col("val")])?
            .build();

        let expected = tbl1
            .join(
                tbl2.aggregate(
                    vec![unresolved_col("outer_key").max()],
                    vec![unresolved_col("inner_key2")],
                )?
                .select(vec![
                    unresolved_col("outer_key").alias(subquery_alias.clone()),
                    unresolved_col("inner_key2"),
                ])?,
                unresolved_col("inner_key")
                    .eq(unresolved_col("inner_key2"))
                    .into(),
                vec![],
                JoinType::Left,
                None,
                Default::default(),
            )?
            .filter(unresolved_col("outer_key").eq(unresolved_col(subquery_alias)))?
            .select(vec![
                unresolved_col("outer_key"),
                unresolved_col("inner_key"),
                unresolved_col("val"),
            ])?
            .select(vec![unresolved_col("val")])?
            .build();

        assert_scalar_optimized_plan_eq(plan, expected)?;
        Ok(())
    }

    #[test]
    fn uncorrelated_scalar_subquery_of_rows_is_checked_on_the_rows_that_reach_it() -> DaftResult<()>
    {
        let tbl1 = dummy_scan_node(dummy_scan_operator(vec![
            Field::new("key", DataType::Int64),
            Field::new("val", DataType::Int64),
        ]));

        let tbl2 = dummy_scan_node(dummy_scan_operator(vec![Field::new(
            "key",
            DataType::Int64,
        )]));

        let subquery = tbl2.select(vec![unresolved_col("key")])?;
        let subquery_expr = Arc::new(Expr::Subquery(Subquery {
            plan: subquery.build(),
        }));
        let subquery_alias = subquery_expr.semantic_id(&subquery.schema()).id;
        let rows = format!("{subquery_alias}.rows");

        let plan = tbl1
            .filter(unresolved_col("key").eq(subquery_expr))?
            .select(vec![unresolved_col("val")])?
            .build();

        let expected = tbl1
            .join(
                subquery
                    .select(vec![unresolved_col("key").alias(subquery_alias.clone())])?
                    .aggregate(
                        vec![
                            unresolved_col(subquery_alias.clone()).any_value(true),
                            unresolved_col(subquery_alias.clone())
                                .count(CountMode::All)
                                .alias(rows.clone()),
                        ],
                        vec![],
                    )?,
                None,
                vec![],
                JoinType::Inner,
                None,
                Default::default(),
            )?
            .select(vec![
                unresolved_col("key"),
                unresolved_col("val"),
                single_row_value(unresolved_col(subquery_alias.clone()), unresolved_col(rows))
                    .alias(subquery_alias.clone()),
            ])?
            .filter(unresolved_col("key").eq(unresolved_col(subquery_alias)))?
            .select(vec![unresolved_col("key"), unresolved_col("val")])?
            .select(vec![unresolved_col("val")])?
            .build();

        assert_scalar_optimized_plan_eq(plan, expected)?;
        Ok(())
    }

    fn correlated_on_inner_key(
        input: crate::LogicalPlanBuilder,
    ) -> DaftResult<crate::LogicalPlanBuilder> {
        input.filter(
            unresolved_col("inner_key2").eq(Arc::new(Expr::Column(Column::Resolved(
                ResolvedColumn::OuterRef(
                    Field::new("inner_key", DataType::Int64),
                    PlanRef::Unqualified,
                ),
            )))),
        )
    }

    #[test]
    fn correlated_scalar_subquery_of_rows_is_checked_on_the_rows_that_reach_it() -> DaftResult<()> {
        let tbl1 = dummy_scan_node(dummy_scan_operator(vec![
            Field::new("inner_key", DataType::Int64),
            Field::new("val", DataType::Int64),
        ]));

        let tbl2 = dummy_scan_node(dummy_scan_operator(vec![
            Field::new("x", DataType::Int64),
            Field::new("inner_key2", DataType::Int64),
        ]));

        let subquery = correlated_on_inner_key(tbl2.clone())?.select(vec![unresolved_col("x")])?;
        let subquery_expr = Arc::new(Expr::Subquery(Subquery {
            plan: subquery.build(),
        }));
        let subquery_alias = subquery_expr.semantic_id(&subquery.schema()).id;
        let rows = format!("{subquery_alias}.rows");

        let plan = tbl1
            .select(vec![unresolved_col("val"), subquery_expr.alias("s")])?
            .build();

        let expected = tbl1
            .join(
                tbl2.select(vec![unresolved_col("x"), unresolved_col("inner_key2")])?
                    .select(vec![
                        unresolved_col("x").alias(subquery_alias.clone()),
                        unresolved_col("inner_key2"),
                    ])?
                    .aggregate(
                        vec![
                            unresolved_col(subquery_alias.clone()).any_value(true),
                            unresolved_col(subquery_alias.clone())
                                .count(CountMode::All)
                                .alias(rows.clone()),
                        ],
                        vec![unresolved_col("inner_key2")],
                    )?,
                unresolved_col("inner_key")
                    .eq(unresolved_col("inner_key2"))
                    .into(),
                vec![],
                JoinType::Left,
                None,
                Default::default(),
            )?
            .select(vec![
                unresolved_col("inner_key"),
                unresolved_col("val"),
                unresolved_col("inner_key2"),
                single_row_value(unresolved_col(subquery_alias.clone()), unresolved_col(rows))
                    .alias(subquery_alias.clone()),
            ])?
            .select(vec![
                unresolved_col("val"),
                unresolved_col(subquery_alias).alias("s"),
            ])?
            .build();

        assert_scalar_optimized_plan_eq(plan, expected)?;
        Ok(())
    }

    #[test]
    fn correlated_scalar_subquery_grouped_by_more_than_its_key_is_checked() -> DaftResult<()> {
        let tbl1 = dummy_scan_node(dummy_scan_operator(vec![
            Field::new("inner_key", DataType::Int64),
            Field::new("val", DataType::Int64),
        ]));

        let tbl2 = dummy_scan_node(dummy_scan_operator(vec![
            Field::new("x", DataType::Int64),
            Field::new("y", DataType::Int64),
            Field::new("inner_key2", DataType::Int64),
        ]));

        // One row per `y` for each key: several rows per key.
        let subquery = correlated_on_inner_key(tbl2)?
            .aggregate(vec![unresolved_col("x").max()], vec![unresolved_col("y")])?
            .select(vec![unresolved_col("x")])?;
        let subquery_expr = Arc::new(Expr::Subquery(Subquery {
            plan: subquery.build(),
        }));
        let subquery_alias = subquery_expr.semantic_id(&subquery.schema()).id;

        let plan = tbl1
            .select(vec![unresolved_col("val"), subquery_expr.alias("s")])?
            .build();

        let optimized = crate::optimization::optimizer::Optimizer::with_rule_batches(
            vec![RuleBatch::new(
                vec![Box::new(UnnestScalarSubquery::new())],
                RuleExecutionStrategy::Once,
            )],
            Default::default(),
        )
        .optimize(plan, |_, _, _, _, _| {})?;

        let checked = optimized
            .repr_indent()
            .contains(&format!("single_row_value(col({subquery_alias})"));
        assert!(checked, "{}", optimized.repr_indent());
        Ok(())
    }

    #[test]
    fn uncorrelated_predicate_subquery() -> DaftResult<()> {
        let tbl1 = dummy_scan_node(dummy_scan_operator(vec![
            Field::new("key", DataType::Int64),
            Field::new("val", DataType::Int64),
        ]));

        let tbl2 = dummy_scan_node(dummy_scan_operator(vec![Field::new(
            "key2",
            DataType::Int64,
        )]));

        let plan = tbl1
            .filter(Arc::new(Expr::InSubquery(
                unresolved_col("key"),
                Subquery { plan: tbl2.build() },
            )))?
            .select(vec![unresolved_col("val")])?
            .build();

        let expected = tbl1
            .join(
                tbl2,
                unresolved_col("key").eq(unresolved_col("key2")).into(),
                vec![],
                JoinType::Semi,
                None,
                Default::default(),
            )?
            .select(vec![unresolved_col("val")])?
            .build();

        assert_predicate_optimized_plan_eq(plan, expected)?;
        Ok(())
    }

    #[test]
    fn correlated_predicate_subquery() -> DaftResult<()> {
        let tbl1 = dummy_scan_node(dummy_scan_operator(vec![
            Field::new("key", DataType::Int64),
            Field::new("val", DataType::Int64),
        ]));

        let tbl2 = dummy_scan_node(dummy_scan_operator(vec![Field::new(
            "key2",
            DataType::Int64,
        )]));

        let subquery = tbl2
            .filter(
                unresolved_col("key2").eq(Arc::new(Expr::Column(Column::Resolved(
                    ResolvedColumn::OuterRef(
                        Field::new("key", DataType::Int64),
                        PlanRef::Unqualified,
                    ),
                )))),
            )?
            .build();

        let plan = tbl1
            .filter(Arc::new(Expr::Exists(Subquery { plan: subquery })).not())?
            .select(vec![unresolved_col("val")])?
            .build();

        let expected = tbl1
            .join(
                tbl2,
                unresolved_col("key").eq(unresolved_col("key2")).into(),
                vec![],
                JoinType::Anti,
                None,
                Default::default(),
            )?
            .select(vec![unresolved_col("val")])?
            .build();

        assert_predicate_optimized_plan_eq(plan, expected)?;
        Ok(())
    }
}
