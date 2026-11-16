//! Pushes a partial aggregation below an equi-join whose aggregations read one side.
//!
//! An aggregation over a join that repeats rows aggregates every repeated row. When its
//! aggregations read only one side, that side can be aggregated first, by every column
//! the join and the grouping read from it: rows equal on those columns meet the same rows
//! of the other side, so each group of them stands for all its rows at once. The join then
//! repeats one row per group, and an aggregation above combines the partial results.
//!
//! The partial aggregation is a [`PartialAggregate`], which execution is free to cut short
//! or pass rows through; the answer never changes. How much it reduces rows decides
//! whether it pays, and that is only known as it runs, so it is placed wherever it is
//! correct and decides at run time.

use std::{collections::HashMap, sync::Arc};

use common_error::DaftResult;
use common_treenode::{Transformed, TreeNode, TreeNodeRecursion};
use daft_core::{
    count_mode::CountMode,
    join::{JoinSide, JoinType},
    prelude::*,
};
use daft_dsl::{
    AggExpr, Column, Expr, ExprRef, ResolvedColumn,
    expr::agg::extract_agg_expr,
    functions::agg::merge_mean,
    lit,
    optimization::{get_required_columns, replace_columns_with_expressions},
    resolved_col,
};
use indexmap::{IndexMap, IndexSet};

use super::OptimizerRule;
use crate::{
    LogicalPlan,
    ops::{Aggregate, Distinct, Filter, Join, Limit, PartialAggregate, Project, Sort},
};

/// Pushes a [`PartialAggregate`] below an equi-join whose aggregations read one side;
/// see the module documentation.
#[derive(Default, Debug)]
pub struct PushPartialAggregateThroughJoin {}

impl PushPartialAggregateThroughJoin {
    pub fn new() -> Self {
        Self {}
    }
}

impl OptimizerRule for PushPartialAggregateThroughJoin {
    fn try_optimize(&self, plan: Arc<LogicalPlan>) -> DaftResult<Transformed<Arc<LogicalPlan>>> {
        plan.transform_down(|node| {
            let LogicalPlan::Aggregate(aggregate) = node.as_ref() else {
                return Ok(Transformed::no(node));
            };
            Ok(match push_below_join(aggregate)? {
                Some(rewritten) => Transformed::yes(rewritten),
                None => Transformed::no(node),
            })
        })
    }
}

/// Which side of the join the aggregations read.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    Left,
    Right,
}

/// One original aggregation, as the partial aggregations it needs below the join.
enum Split {
    /// Combines by an aggregation of the same kind: the partial at this index.
    Direct { partial: usize, counts: bool },
    /// A mean: the partial sum and the partial count at these indexes.
    Mean { sum: usize, count: usize },
}

impl From<Side> for JoinSide {
    fn from(side: Side) -> Self {
        match side {
            Side::Left => Self::Left,
            Side::Right => Self::Right,
        }
    }
}

/// The columns of the join's `side` that a join predicate reads.
fn columns_of_side(predicate: &ExprRef, side: JoinSide) -> DaftResult<Vec<String>> {
    let mut columns = Vec::new();
    predicate.apply(|expr| {
        if let Expr::Column(Column::Resolved(ResolvedColumn::JoinSide(field, of))) = expr.as_ref()
            && *of == side
        {
            columns.push(field.name.to_string());
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(columns)
}

/// Key and grouping column types for which grouping and joining agree on equality.
///
/// Floats are left out: grouping and joining treat NaN and negative zero differently.
fn groups_like_joins(dtype: &DataType) -> bool {
    dtype.is_integer()
        || matches!(
            dtype,
            DataType::Boolean
                | DataType::Utf8
                | DataType::Binary
                | DataType::FixedSizeBinary(_)
                | DataType::Date
                | DataType::Time(_)
                | DataType::Timestamp(..)
                | DataType::Duration(_)
                | DataType::Decimal128(..)
        )
}

/// The aggregate's input columns, renamed through the plain renaming projections between it
/// and a join, as the join's output names; and the join. `None` when anything else lies
/// between them.
fn join_below(
    mut plan: &Arc<LogicalPlan>,
    columns: impl IntoIterator<Item = String>,
) -> Option<(HashMap<String, String>, &Join)> {
    let mut names: HashMap<String, String> = columns.into_iter().map(|c| (c.clone(), c)).collect();
    loop {
        match plan.as_ref() {
            LogicalPlan::Join(join) => return Some((names, join)),
            LogicalPlan::Project(Project {
                input, projection, ..
            }) => {
                let renames: HashMap<String, String> = projection
                    .iter()
                    .filter_map(|expr| Some((expr.name().to_string(), expr.input_mapping()?)))
                    .collect();
                for name in names.values_mut() {
                    *name = renames.get(name.as_str())?.clone();
                }
                plan = input;
            }
            _ => return None,
        }
    }
}

/// Whether `plan` provably holds at most one row for each value of `columns`: it is an
/// aggregation or a distinct keyed within them, seen through filters, sorts, limits and
/// renaming projections.
fn unique_by(plan: &LogicalPlan, columns: &IndexSet<String>) -> bool {
    let within = |keys: &[ExprRef]| keys.iter().all(|key| columns.contains(key.name()));
    match plan {
        LogicalPlan::Aggregate(Aggregate { groupby, .. }) => within(groupby),
        LogicalPlan::Distinct(Distinct {
            columns: Some(keys),
            ..
        }) => within(keys),
        LogicalPlan::Distinct(Distinct { columns: None, .. }) => plan
            .schema()
            .field_names()
            .all(|name| columns.contains(name)),
        LogicalPlan::Filter(Filter { input, .. })
        | LogicalPlan::Sort(Sort { input, .. })
        | LogicalPlan::Limit(Limit { input, .. }) => unique_by(input, columns),
        // Unique on the columns it passes through is unique on all of `columns`.
        LogicalPlan::Project(Project {
            input, projection, ..
        }) => {
            let below = projection
                .iter()
                .filter(|expr| columns.contains(expr.name()))
                .filter_map(|expr| expr.input_mapping())
                .collect();
            unique_by(input, &below)
        }
        _ => false,
    }
}

/// A partial-aggregation column name used by neither join input nor the aggregate.
fn fresh_name(taken: &IndexSet<String>, index: usize) -> String {
    let mut name = format!("__daft_pagg_{index}");
    while taken.contains(&name) {
        name.push('_');
    }
    name
}

#[allow(clippy::too_many_lines)]
fn push_below_join(aggregate: &Aggregate) -> DaftResult<Option<Arc<LogicalPlan>>> {
    if aggregate.aggregations.is_empty() {
        return Ok(None);
    }
    let aggregations = aggregate
        .aggregations
        .iter()
        .map(|expr| Ok((expr.name().to_string(), extract_agg_expr(expr)?)))
        .collect::<DaftResult<Vec<_>>>()?;

    // Every column the aggregate reads, followed down to the join's output names.
    let read: IndexSet<String> = aggregations
        .iter()
        .flat_map(|(_, agg)| agg.children())
        .chain(aggregate.groupby.iter().cloned())
        .flat_map(|expr| get_required_columns(&expr))
        .collect();
    let Some((names, join)) = join_below(&aggregate.input, read) else {
        return Ok(None);
    };
    if join.key_filtering_config.is_some() {
        return Ok(None);
    }
    let (remaining, left_keys, right_keys, _) = join.on.split_eq_preds();
    if left_keys.is_empty()
        || !left_keys
            .iter()
            .chain(&right_keys)
            .chain(remaining.inner())
            .all(Expr::is_deterministic)
    {
        return Ok(None);
    }

    let left_schema = join.left.schema();
    let right_schema = join.right.schema();
    let left_names: IndexSet<String> = left_schema.field_names().map(ToString::to_string).collect();
    let right_names: IndexSet<String> = right_schema
        .field_names()
        .map(ToString::to_string)
        .collect();
    // A column both inputs name is merged by the join; that is sound only for a key the
    // two sides equate.
    let merged_keys: IndexSet<&String> = left_names.intersection(&right_names).collect();
    if merged_keys.iter().any(|name| {
        !left_keys.iter().zip(&right_keys).any(|(l, r)| {
            l.input_mapping().as_deref() == Some(name.as_str())
                && r.input_mapping().as_deref() == Some(name.as_str())
        })
    }) {
        return Ok(None);
    }

    // The side every aggregation reads; a merged key is on both.
    let replacements: HashMap<String, ExprRef> = names
        .iter()
        .map(|(top, below)| (top.clone(), resolved_col(below.clone())))
        .collect();
    let to_join = |expr: &ExprRef| replace_columns_with_expressions(expr.clone(), &replacements);
    let argument_columns: IndexSet<String> = aggregations
        .iter()
        .flat_map(|(_, agg)| agg.children())
        .map(|expr| to_join(&expr))
        .flat_map(|expr| get_required_columns(&expr))
        .collect();
    let reads = |side: &IndexSet<String>| argument_columns.iter().all(|c| side.contains(c));
    let side = match (join.join_type, reads(&left_names), reads(&right_names)) {
        (JoinType::Inner | JoinType::Left, true, _) => Side::Left,
        (JoinType::Inner | JoinType::Right, _, true) => Side::Right,
        _ => return Ok(None),
    };
    let (side_plan, side_names, side_schema, side_keys, other_plan, other_keys) = match side {
        Side::Left => (
            &join.left,
            &left_names,
            &left_schema,
            &left_keys,
            &join.right,
            &right_keys,
        ),
        Side::Right => (
            &join.right,
            &right_names,
            &right_schema,
            &right_keys,
            &join.left,
            &left_keys,
        ),
    };
    if matches!(side_plan.as_ref(), LogicalPlan::PartialAggregate(_)) {
        return Ok(None);
    }
    // When the other side holds at most one row per key, the join repeats no row of this
    // side: aggregating it first saves no repeated work, only moves the aggregation below
    // a join that may drop its rows.
    let other_key_columns: IndexSet<String> = other_keys
        .iter()
        .filter_map(|key| key.input_mapping())
        .collect();
    if unique_by(other_plan, &other_key_columns) {
        return Ok(None);
    }

    // Group the side by every column of it that is read above it outside the aggregations:
    // its keys, the rest of the join predicate, and the grouping.
    let remaining_reads = match remaining.inner() {
        Some(predicate) => columns_of_side(predicate, side.into())?,
        None => Vec::new(),
    };
    let joined_by: IndexSet<String> = side_keys
        .iter()
        .flat_map(get_required_columns)
        .filter(|c| side_names.contains(c))
        .chain(remaining_reads)
        .collect();
    if joined_by.is_empty()
        || joined_by.iter().any(|c| {
            side_schema
                .get_field(c)
                .map_or(true, |field| !groups_like_joins(&field.dtype))
        })
    {
        return Ok(None);
    }
    let groupby: Vec<ExprRef> = aggregate.groupby.iter().map(&to_join).collect();
    let partial_groupby: IndexSet<String> = joined_by
        .into_iter()
        .chain(
            groupby
                .iter()
                .flat_map(get_required_columns)
                .filter(|c| side_names.contains(c)),
        )
        .collect();
    if unique_by(side_plan, &partial_groupby) {
        return Ok(None);
    }

    // The partial aggregations, each once.
    let mut partials: IndexMap<AggExpr, usize> = IndexMap::new();
    let mut add = |agg: AggExpr| -> usize {
        let next = partials.len();
        *partials.entry(agg).or_insert(next)
    };
    let mut splits = Vec::with_capacity(aggregations.len());
    for (_, agg) in &aggregations {
        let agg = agg.with_new_children(agg.children().iter().map(&to_join).collect());
        let split = match agg {
            AggExpr::Mean(child) => Split::Mean {
                sum: add(AggExpr::Sum(child.clone())),
                count: add(AggExpr::Count(child, CountMode::Valid)),
            },
            agg if agg.merge_of_partial(lit(0)).is_some() => Split::Direct {
                counts: matches!(agg, AggExpr::Count(..)),
                partial: add(agg),
            },
            _ => return Ok(None),
        };
        splits.push(split);
    }

    let taken: IndexSet<String> = left_names
        .iter()
        .chain(&right_names)
        .cloned()
        .chain(
            aggregate
                .output_schema
                .field_names()
                .map(ToString::to_string),
        )
        .collect();
    let partial_names: Vec<String> = (0..partials.len()).map(|i| fresh_name(&taken, i)).collect();
    let partial_aggregations: Vec<ExprRef> = partials
        .keys()
        .zip(&partial_names)
        .map(|(agg, name)| Arc::new(Expr::Agg(agg.clone())).alias(name.as_str()))
        .collect();

    let partial = Arc::new(LogicalPlan::PartialAggregate(PartialAggregate::try_new(
        side_plan.clone(),
        partial_aggregations,
        partial_groupby.into_iter().map(resolved_col).collect(),
    )?));
    let (left, right) = match side {
        Side::Left => (partial, join.right.clone()),
        Side::Right => (join.left.clone(), partial),
    };
    let new_join = Arc::new(LogicalPlan::Join(
        Join::try_new(
            left,
            right,
            join.on.clone(),
            join.join_type,
            join.join_strategy,
        )?
        .with_key_filtering_config(join.key_filtering_config.clone()),
    ));

    // Combine the partials, grouped as the aggregate was.
    let mut merged = Vec::with_capacity(partials.len());
    for (agg, name) in partials.keys().zip(&partial_names) {
        let Some(merge) = agg.merge_of_partial(resolved_col(name.as_str())) else {
            return Ok(None);
        };
        merged.push(Arc::new(Expr::Agg(merge)).alias(name.as_str()));
    }
    let group_names: Vec<String> = aggregate
        .groupby
        .iter()
        .map(|expr| expr.name().to_string())
        .collect();
    let upper_groupby: Vec<ExprRef> = groupby
        .iter()
        .zip(&group_names)
        .map(|(expr, name)| {
            if expr.name() == name {
                expr.clone()
            } else {
                expr.alias(name.as_str())
            }
        })
        .collect();
    let upper = Arc::new(LogicalPlan::Aggregate(Aggregate::try_new(
        new_join,
        merged,
        upper_groupby,
    )?));

    // Finish each aggregation from its combined partials, in the original order and types.
    let output_schema = aggregate.output_schema.clone();
    let ungrouped = aggregate.groupby.is_empty();
    let finished = group_names
        .iter()
        .map(|name| resolved_col(name.as_str()))
        .chain(splits.iter().map(|split| match split {
            Split::Direct { partial, counts } => {
                let column = resolved_col(partial_names[*partial].as_str());
                // A sum over no rows is null where a count over none is zero.
                if *counts && ungrouped {
                    column.fill_null(lit(0u64))
                } else {
                    column
                }
            }
            Split::Mean { sum, count } => merge_mean(
                resolved_col(partial_names[*sum].as_str()),
                resolved_col(partial_names[*count].as_str()),
            ),
        }))
        .zip(output_schema.fields())
        .map(|(expr, field)| expr.cast(&field.dtype).alias(field.name.as_ref()))
        .collect::<Vec<_>>();
    let rewritten = Arc::new(LogicalPlan::Project(Project::try_new(upper, finished)?));

    // The rewrite must answer exactly the columns the aggregate did.
    if rewritten.schema() != output_schema {
        return Ok(None);
    }
    Ok(Some(rewritten))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use common_error::DaftResult;
    use common_treenode::{TreeNode, TreeNodeRecursion};
    use daft_core::{join::JoinType, prelude::*};
    use daft_dsl::{ExprRef, left_col, lit, right_col, unresolved_col};
    use rstest::rstest;

    use super::PushPartialAggregateThroughJoin;
    use crate::{
        LogicalPlan, LogicalPlanBuilder,
        optimization::{
            optimizer::{RuleBatch, RuleExecutionStrategy},
            test::optimize_with_rules,
        },
        test::{dummy_scan_node, dummy_scan_operator},
    };

    /// A link of subjects to owners, and a second copy of it keyed by owner.
    fn link() -> LogicalPlanBuilder {
        dummy_scan_node(dummy_scan_operator(vec![
            Field::new("subject", DataType::Int64),
            Field::new("owner", DataType::Int64),
            Field::new("weight", DataType::Int64),
        ]))
    }

    fn peers() -> LogicalPlanBuilder {
        dummy_scan_node(dummy_scan_operator(vec![
            Field::new("owner", DataType::Int64),
            Field::new("peer", DataType::Int64),
            Field::new("score", DataType::Float64),
            Field::new("flag", DataType::Boolean),
        ]))
    }

    fn joined(join_type: JoinType) -> DaftResult<LogicalPlanBuilder> {
        link().join(
            peers(),
            None,
            vec!["owner".to_string()],
            join_type,
            None,
            Default::default(),
        )
    }

    fn optimized(plan: Arc<LogicalPlan>) -> DaftResult<Arc<LogicalPlan>> {
        optimize_with_rules(
            plan,
            vec![RuleBatch::new(
                vec![Box::new(PushPartialAggregateThroughJoin::new())],
                RuleExecutionStrategy::Once,
            )],
        )
    }

    /// Each partial aggregation in `plan`, as its group-by and aggregations, written out.
    fn partials(plan: &Arc<LogicalPlan>) -> Vec<(Vec<String>, Vec<String>)> {
        let mut found = vec![];
        plan.apply(|node| {
            if let LogicalPlan::PartialAggregate(partial) = node.as_ref() {
                found.push((
                    partial.groupby.iter().map(ToString::to_string).collect(),
                    partial
                        .aggregations
                        .iter()
                        .map(ToString::to_string)
                        .collect(),
                ));
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        found
    }

    fn aggregate(
        builder: LogicalPlanBuilder,
        aggregations: Vec<ExprRef>,
        groupby: Vec<ExprRef>,
    ) -> DaftResult<Arc<LogicalPlan>> {
        Ok(builder.aggregate(aggregations, groupby)?.build())
    }

    #[rstest]
    #[case::count(unresolved_col("peer").count(CountMode::Valid), "count(col(peer)")]
    #[case::sum(unresolved_col("peer").sum(), "sum(col(peer)")]
    #[case::min(unresolved_col("peer").min(), "min(col(peer)")]
    #[case::max(unresolved_col("peer").max(), "max(col(peer)")]
    #[case::bool_and(unresolved_col("flag").bool_and(), "bool_and(col(flag)")]
    fn an_aggregation_of_the_repeated_side_is_partially_aggregated_below_the_join(
        #[case] aggregation: ExprRef,
        #[case] partial: &str,
    ) -> DaftResult<()> {
        let plan = aggregate(
            joined(JoinType::Inner)?,
            vec![aggregation.alias("n")],
            vec![unresolved_col("subject")],
        )?;

        let optimized = optimized(plan.clone())?;

        let found = partials(&optimized);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0, vec!["col(owner)".to_string()]);
        assert!(found[0].1[0].starts_with(partial), "{:?}", found[0].1);
        assert_eq!(optimized.schema(), plan.schema());
        Ok(())
    }

    #[test]
    fn a_mean_is_partially_summed_and_counted() -> DaftResult<()> {
        let plan = aggregate(
            joined(JoinType::Inner)?,
            vec![unresolved_col("peer").mean().alias("m")],
            vec![unresolved_col("subject")],
        )?;

        let optimized = optimized(plan.clone())?;

        let found = partials(&optimized);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].1.len(), 2);
        assert_eq!(optimized.schema(), plan.schema());
        Ok(())
    }

    #[test]
    fn grouping_columns_of_the_aggregated_side_join_the_partial_grouping() -> DaftResult<()> {
        let plan = aggregate(
            joined(JoinType::Inner)?,
            vec![unresolved_col("peer").sum().alias("s")],
            vec![unresolved_col("subject"), unresolved_col("flag")],
        )?;

        let optimized = optimized(plan)?;

        assert_eq!(
            partials(&optimized)[0].0,
            vec!["col(owner)".to_string(), "col(flag)".to_string()]
        );
        Ok(())
    }

    #[rstest]
    // The aggregated side is the left, preserved by a left join.
    #[case::left_join_preserving_it(JoinType::Left, "weight", 1)]
    // A left join supplies nulls for the right side: its rows are not each matched.
    #[case::left_join_of_its_null_supplying_side(JoinType::Left, "peer", 0)]
    #[case::right_join_preserving_it(JoinType::Right, "peer", 1)]
    #[case::outer_join(JoinType::Outer, "peer", 0)]
    #[case::semi_join(JoinType::Semi, "weight", 0)]
    fn only_a_side_every_row_of_which_the_join_keeps_is_aggregated(
        #[case] join_type: JoinType,
        #[case] column: &str,
        #[case] pushed: usize,
    ) -> DaftResult<()> {
        let plan = aggregate(
            joined(join_type)?,
            vec![unresolved_col(column).sum().alias("s")],
            vec![unresolved_col("owner")],
        )?;

        let optimized = optimized(plan)?;

        assert_eq!(partials(&optimized).len(), pushed);
        Ok(())
    }

    #[rstest]
    #[case::count_distinct(vec![unresolved_col("peer").count_distinct().alias("n")])]
    #[case::list(vec![unresolved_col("peer").agg_list().alias("l")])]
    #[case::both_sides(vec![unresolved_col("peer").sum().alias("a"), unresolved_col("weight").sum().alias("b")])]
    fn aggregations_that_do_not_split_or_read_both_sides_stay_above(
        #[case] aggregations: Vec<ExprRef>,
    ) -> DaftResult<()> {
        let plan = aggregate(
            joined(JoinType::Inner)?,
            aggregations,
            vec![unresolved_col("subject")],
        )?;

        let optimized = optimized(plan.clone())?;

        assert!(partials(&optimized).is_empty());
        assert_eq!(optimized, plan);
        Ok(())
    }

    #[test]
    fn a_float_join_key_is_not_grouped_on() -> DaftResult<()> {
        let left = dummy_scan_node(dummy_scan_operator(vec![
            Field::new("k", DataType::Float64),
            Field::new("g", DataType::Int64),
        ]));
        let right = dummy_scan_node(dummy_scan_operator(vec![
            Field::new("k", DataType::Float64),
            Field::new("v", DataType::Int64),
        ]));
        let plan = aggregate(
            left.join(
                right,
                None,
                vec!["k".to_string()],
                JoinType::Inner,
                None,
                Default::default(),
            )?,
            vec![unresolved_col("v").sum().alias("s")],
            vec![unresolved_col("g")],
        )?;

        assert!(partials(&optimized(plan)?).is_empty());
        Ok(())
    }

    #[test]
    fn columns_a_remaining_join_condition_reads_join_the_partial_grouping() -> DaftResult<()> {
        let int = |name: &str| Field::new(name, DataType::Int64);
        let on = left_col(int("owner"))
            .eq(right_col(int("owner")))
            .and(left_col(int("weight")).lt(right_col(int("peer"))));
        let plan = aggregate(
            link().join(
                peers(),
                Some(on),
                vec![],
                JoinType::Inner,
                None,
                Default::default(),
            )?,
            vec![unresolved_col("peer").sum().alias("s")],
            vec![unresolved_col("subject")],
        )?;

        let optimized = optimized(plan)?;

        assert_eq!(
            partials(&optimized)[0].0,
            vec!["col(right.owner)".to_string(), "col(peer)".to_string()]
        );
        Ok(())
    }

    #[test]
    fn a_side_already_aggregated_by_the_join_key_is_not_aggregated_again() -> DaftResult<()> {
        let per_owner = peers().aggregate(
            vec![unresolved_col("peer").sum().alias("peer")],
            vec![unresolved_col("owner")],
        )?;
        let plan = aggregate(
            link().join(
                per_owner,
                None,
                vec!["owner".to_string()],
                JoinType::Inner,
                None,
                Default::default(),
            )?,
            vec![unresolved_col("peer").sum().alias("s")],
            vec![unresolved_col("subject")],
        )?;

        assert!(partials(&optimized(plan)?).is_empty());
        Ok(())
    }

    #[test]
    fn a_join_whose_other_side_is_unique_by_its_key_is_not_pushed_through() -> DaftResult<()> {
        // Each link row meets at most one row of a side aggregated by owner, seen through a
        // filter and a renaming projection.
        let per_owner = link()
            .aggregate(
                vec![unresolved_col("weight").sum().alias("total")],
                vec![unresolved_col("owner")],
            )?
            .filter(unresolved_col("total").gt(lit(0)))?
            .select(vec![
                unresolved_col("owner").alias("o"),
                unresolved_col("total"),
            ])?;
        let plan = aggregate(
            peers().join(
                per_owner,
                Some(unresolved_col("owner").eq(unresolved_col("o"))),
                vec![],
                JoinType::Inner,
                None,
                Default::default(),
            )?,
            vec![unresolved_col("peer").sum().alias("s")],
            vec![unresolved_col("total")],
        )?;

        assert!(partials(&optimized(plan)?).is_empty());
        Ok(())
    }

    #[test]
    fn pushing_twice_adds_no_second_partial_aggregation() -> DaftResult<()> {
        let plan = aggregate(
            joined(JoinType::Inner)?,
            vec![unresolved_col("peer").sum().alias("s")],
            vec![unresolved_col("subject")],
        )?;

        let once = optimized(plan)?;
        let twice = optimized(once)?;

        assert_eq!(partials(&twice).len(), 1);
        Ok(())
    }
}
