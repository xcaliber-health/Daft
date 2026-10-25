use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use common_error::DaftResult;
use common_treenode::{DynTreeNode, Transformed, TreeNode};
use daft_algebra::boolean::{combine_conjunction, split_conjunction, to_cnf};
use daft_core::{join::JoinType, prelude::Schema};
use daft_dsl::{
    ExprRef, estimated_selectivity,
    optimization::{get_required_columns, replace_columns_with_expressions},
    resolved_col,
};
use daft_scan::{PredicateGroups, ScanState, rewrite_predicate_for_partitioning};

use super::OptimizerRule;
use crate::{
    LogicalPlan,
    ops::{Concat, Filter, Join},
    source_info::SourceInfo,
};

/// Optimization rules for pushing Filters further into the logical plan.
#[derive(Default, Debug)]
pub struct PushDownFilter {
    strict_pushdown: bool,
}

impl PushDownFilter {
    pub fn new(strict_pushdown: bool) -> Self {
        Self { strict_pushdown }
    }
}

impl OptimizerRule for PushDownFilter {
    fn try_optimize(&self, plan: Arc<LogicalPlan>) -> DaftResult<Transformed<Arc<LogicalPlan>>> {
        plan.transform_down(|node| self.try_optimize_node(node))
    }
}

impl PushDownFilter {
    #[allow(clippy::only_used_in_recursion)]
    #[recursive::recursive]
    fn try_optimize_node(
        &self,
        plan: Arc<LogicalPlan>,
    ) -> DaftResult<Transformed<Arc<LogicalPlan>>> {
        let filter = match plan.as_ref() {
            LogicalPlan::Filter(filter) => filter,
            _ => return Ok(Transformed::no(plan)),
        };
        let child_plan = filter.input.as_ref();
        let new_plan = match child_plan {
            LogicalPlan::Filter(child_filter) => {
                // Combine filters.
                //
                // Filter-Filter --> Filter

                // Split predicate expression on conjunctions (ANDs).
                let parent_predicates = split_conjunction(&filter.predicate);
                let predicate_set: HashSet<&ExprRef> = parent_predicates.iter().collect();
                // Add child predicate expressions to parent predicate expressions, eliminating duplicates.
                let new_predicates: Vec<ExprRef> = parent_predicates
                    .iter()
                    .chain(
                        split_conjunction(&child_filter.predicate)
                            .iter()
                            .filter(|e| !predicate_set.contains(*e)),
                    )
                    .map(|e| (*e).clone())
                    .collect::<Vec<_>>();
                // Reconjunct predicate expressions.
                let new_predicate = combine_conjunction(new_predicates).unwrap();
                let new_filter: Arc<LogicalPlan> =
                    LogicalPlan::from(Filter::try_new(child_filter.input.clone(), new_predicate)?)
                        .into();
                self.try_optimize_node(new_filter.clone())?
                    .or(Transformed::yes(new_filter))
                    .data
            }
            LogicalPlan::Source(source) => {
                // If the Source has a checkpoint, keep the Filter as a separate op
                // above the Source. Letting the filter sink into the scan's pushdowns
                // would cause Parquet to drop rows at decode time before SCKO can
                // stage their keys — breaking the "checkpoint all source keys" guarantee
                // and forcing every re-run to re-read and re-evaluate those rows.
                if source.checkpoint.is_some() {
                    return Ok(Transformed::no(plan));
                }
                match source.source_info.as_ref() {
                    // Filter pushdown is not supported for in-memory sources.
                    SourceInfo::InMemory(_) => return Ok(Transformed::no(plan)),
                    // Filter pushdown is not supported for glob scan sources.
                    SourceInfo::GlobScan(_) => return Ok(Transformed::no(plan)),
                    // Do not pushdown if Source node already has a limit
                    SourceInfo::Physical(external_info)
                        if let Some(_) = external_info.pushdowns.limit =>
                    {
                        return Ok(Transformed::no(plan));
                    }

                    // Pushdown filter into the Source node
                    SourceInfo::Physical(external_info) => {
                        // If the scan is materialized, we don't pushdown the filter.
                        if matches!(external_info.scan_state, ScanState::Tasks(_)) {
                            return Ok(Transformed::no(plan));
                        }
                        let predicate = &filter.predicate;
                        let new_predicate = external_info
                            .pushdowns
                            .filters
                            .as_ref()
                            .map(|f| predicate.clone().and(f.clone()))
                            .unwrap_or_else(|| predicate.clone());
                        // We split the predicate into three groups:
                        // 1. All partition-only filters, which can be applied directly to partition values and can be
                        //    dropped from the data-level filter.
                        // 2. Predicates that only reference data columns (no partition column references) or only
                        //    reference partition columns but involve non-identity transformations; these need to be
                        //    applied to the data, but don't require a separate filter op (i.e. they can be pushed into
                        //    the scan).
                        // 3. Filters needing their own dedicated filter op (unable to be pushed into scan); this
                        //    includes predicates involving both partition and data columns, and predicates containing
                        //    UDFs.
                        let PredicateGroups {
                            partition_only_filter,
                            data_only_filter,
                            needing_filter_op,
                        } = rewrite_predicate_for_partitioning(
                            &new_predicate,
                            external_info.scan_state.get_scan_op().0.partitioning_keys(),
                        )?;
                        assert!(
                            partition_only_filter.len()
                                + data_only_filter.len()
                                + needing_filter_op.len()
                                > 0
                        );

                        if !needing_filter_op.is_empty()
                            && partition_only_filter.is_empty()
                            && data_only_filter.is_empty()
                        {
                            // If the filter predicate consists of only expressions that rely on both a partition
                            // column and a data column (or contain a UDF), then no pushdown into the scan is possible,
                            // so we short-circuit.
                            // TODO(Clark): Support pushing predicates referencing both partition and data columns into the scan.
                            return Ok(Transformed::no(plan));
                        }

                        //deduplicate
                        let data_only_filter = {
                            let mut seen = HashSet::new();
                            data_only_filter
                                .into_iter()
                                .filter(|e| seen.insert(e.clone()))
                                .collect::<Vec<_>>()
                        };

                        let data_filter = combine_conjunction(data_only_filter.clone());
                        let partition_filter = combine_conjunction(partition_only_filter);
                        assert!(data_filter.is_some() || partition_filter.is_some());

                        let new_pushdowns = if let Some(data_filter) = data_filter {
                            external_info.pushdowns.with_filters(Some(data_filter))
                        } else {
                            external_info.pushdowns.clone()
                        };
                        let new_pushdowns = if let Some(partition_filter) = partition_filter {
                            // Merge new partition filters with any existing ones via AND,
                            // rather than overwriting. Partition filters are monotonic under
                            // AND — adding predicates only makes pruning stricter, never
                            // looser — so merging is always safe. This prevents previously
                            // derived partition filters from being lost when the optimizer
                            // re-runs this rule on a rebuilt Filter node.
                            let merged_partition_filter =
                                if let Some(existing) = &new_pushdowns.partition_filters {
                                    existing.clone().and(partition_filter)
                                } else {
                                    partition_filter
                                };
                            new_pushdowns.with_partition_filters(Some(merged_partition_filter))
                        } else {
                            new_pushdowns
                        };

                        let scan_op = external_info.scan_state.get_scan_op().0.clone();
                        let remaining_filters = if let Some(supports_pushdown) =
                            scan_op.as_pushdown_filter()
                            && self.strict_pushdown
                        {
                            let filters_to_push = new_pushdowns.filters.as_slice();

                            let (pushed_filters, post_filters) =
                                supports_pushdown.push_filters(filters_to_push);
                            let _ = new_pushdowns.with_pushed_filters(Some(pushed_filters.clone()));
                            if !post_filters.is_empty()
                                && post_filters.len() == filters_to_push.len()
                            {
                                return Ok(Transformed::no(plan));
                            }

                            let mut seen = HashSet::new();
                            let pushed_filters_set = if pushed_filters.len() == 1 {
                                split_conjunction(&pushed_filters[0])
                                    .into_iter()
                                    .collect::<HashSet<_>>()
                            } else {
                                pushed_filters
                                    .iter()
                                    .flat_map(split_conjunction)
                                    .collect::<HashSet<_>>()
                            };
                            post_filters
                                .into_iter()
                                .chain(
                                    data_only_filter
                                        .iter()
                                        .filter(|f| !pushed_filters_set.contains(*f))
                                        .cloned(),
                                )
                                .filter(|e| seen.insert(e.clone()))
                                .collect::<Vec<_>>()
                        } else {
                            // Return an empty Vec if strict pushdown is not supported
                            Vec::new()
                        };

                        let needing_filter_op = {
                            let mut seen = HashSet::new();
                            remaining_filters
                                .into_iter()
                                .chain(needing_filter_op)
                                .filter(|e| seen.insert(e.clone()))
                                .collect::<Vec<_>>()
                        };

                        let new_external_info = external_info.with_pushdowns(new_pushdowns);
                        let new_source: LogicalPlan = source
                            .clone()
                            .with_source_info(SourceInfo::Physical(new_external_info).into())
                            .into();
                        if !needing_filter_op.is_empty() {
                            // We need to apply any filter predicates that reference both partition and data columns after the scan.
                            // TODO(Clark): Support pushing predicates referencing both partition and data columns into the scan.
                            let filter_op: LogicalPlan = Filter::try_new(
                                new_source.into(),
                                combine_conjunction(needing_filter_op).unwrap(),
                            )?
                            .into();
                            return Ok(Transformed::yes(filter_op.into()));
                        } else {
                            return Ok(Transformed::yes(new_source.into()));
                        }
                    }
                    SourceInfo::PlaceHolder(..) => {
                        panic!("PlaceHolderInfo should not exist for optimization!");
                    }
                }
            }
            LogicalPlan::Project(child_project) => {
                // Commute filter with projection if predicate only depends on projection columns that
                // don't involve compute.
                //
                // Filter-Projection --> {Filter-}Projection-Filter
                //
                // TODO(Clark): Push Filters depending on Projection columns involving compute if those expressions are
                // (1) deterministic && (pure || idempotent),
                // (2) inexpensive to recompute.
                // This can be done by rewriting the Filter predicate expression to contain the relevant Projection expression.
                let passed_through = passed_through_columns(&child_project.projection);
                // A projection keeps every row, so even a non-deterministic conjunct
                // reads the same rows on either side of it.
                let (Some(below), above) =
                    split_conjuncts(&filter.predicate, &passed_through, None)
                else {
                    return Ok(Transformed::no(plan));
                };
                place_around(child_plan, &child_project.input, below, above)?
            }
            // A restriction on the columns rows are grouped by keeps or drops each
            // group whole, so it may drop the group's rows before they are grouped:
            // fewer rows are grouped, and the restriction can reach the scan.
            //
            // Not through a global aggregate, which answers one row even over no
            // rows, nor a window with no partitions, nor for a non-deterministic
            // conjunct, which would run once per row instead of once per group.
            LogicalPlan::Aggregate(aggregate) if !aggregate.groupby.is_empty() => {
                let passed_through = passed_through_columns(&aggregate.groupby);
                let (Some(below), above) = split_conjuncts(
                    &filter.predicate,
                    &passed_through,
                    Some(&child_plan.schema()),
                ) else {
                    return Ok(Transformed::no(plan));
                };
                place_around(child_plan, &aggregate.input, below, above)?
            }
            LogicalPlan::Window(window) if !window.window_spec.partition_by.is_empty() => {
                let passed_through = passed_through_columns(&window.window_spec.partition_by);
                let (Some(below), above) = split_conjuncts(
                    &filter.predicate,
                    &passed_through,
                    Some(&child_plan.schema()),
                ) else {
                    return Ok(Transformed::no(plan));
                };
                place_around(child_plan, &window.input, below, above)?
            }
            LogicalPlan::Distinct(distinct) => {
                let passed_through = match &distinct.columns {
                    Some(columns) => passed_through_columns(columns),
                    // Distinct over whole rows groups by every column.
                    None => distinct
                        .input
                        .schema()
                        .field_names()
                        .map(|name| (name.to_string(), resolved_col(name)))
                        .collect(),
                };
                let (Some(below), above) = split_conjuncts(
                    &filter.predicate,
                    &passed_through,
                    Some(&child_plan.schema()),
                ) else {
                    return Ok(Transformed::no(plan));
                };
                place_around(child_plan, &distinct.input, below, above)?
            }
            LogicalPlan::Sort(_)
            | LogicalPlan::Shuffle(_)
            | LogicalPlan::Repartition(_)
            | LogicalPlan::IntoBatches(_)
            | LogicalPlan::IntoPartitions(_) => {
                // Naive commuting with unary ops.
                let new_filter = plan
                    .with_new_children(&[child_plan.arc_children()[0].clone()])
                    .into();
                child_plan.with_new_children(&[new_filter]).into()
            }
            LogicalPlan::Concat(Concat { input, other, .. }) => {
                // Push filter into each side of the concat.
                let new_input: LogicalPlan =
                    Filter::try_new(input.clone(), filter.predicate.clone())?.into();
                let new_other: LogicalPlan =
                    Filter::try_new(other.clone(), filter.predicate.clone())?.into();
                let new_concat: LogicalPlan =
                    Concat::try_new(new_input.into(), new_other.into())?.into();
                new_concat.into()
            }
            LogicalPlan::Join(Join {
                left,
                right,
                join_type,
                on,
                join_strategy,
                key_filtering_config,
                ..
            }) => {
                // TODO(Kevin): add more filter pushdowns for joins
                // Example:
                //      For `foo JOIN bar ON foo.a == (bar.b + 2) WHERE a > 0`, the filter `a > 0` is pushed down to the left side, but can also be pushed down to the right side as `(b + 2) > 0`

                let mut left_pushdowns = vec![];
                let mut right_pushdowns = vec![];
                let mut kept_predicates = vec![];

                let left_cols = HashSet::<_>::from_iter(left.schema().names());
                let right_cols = HashSet::<_>::from_iter(right.schema().names());

                // TODO: simplify predicates, since they may be expanded with `to_cnf`
                for predicate in split_conjunction(&to_cnf(filter.predicate.clone())) {
                    let pred_cols = HashSet::<_>::from_iter(get_required_columns(&predicate));

                    // Extract join key column names for anti/semi join handling
                    let join_key_cols: HashSet<String> =
                        if matches!(join_type, JoinType::Anti | JoinType::Semi) {
                            let (_, left_keys, right_keys, _) = on.split_eq_preds();
                            left_keys
                                .iter()
                                .chain(right_keys.iter())
                                .filter_map(|e| e.input_mapping())
                                .collect()
                        } else {
                            HashSet::new()
                        };

                    match (
                        pred_cols.is_subset(&left_cols),
                        pred_cols.is_subset(&right_cols),
                    ) {
                        (true, true) => {
                            // predicate columns exist in both left and right input schemas
                            if matches!(join_type, JoinType::Anti | JoinType::Semi)
                                && !pred_cols.iter().all(|c| join_key_cols.contains(c))
                            {
                                // For anti/semi joins, if the predicate column is NOT a join key,
                                // only push to the left side. The output only contains left columns,
                                // so filtering the right side on a non-join-key column would
                                // incorrectly change the join semantics.
                                // (Issue #6086)
                                left_pushdowns.push(predicate);
                            } else {
                                // For join key columns or other join types, push to both sides
                                left_pushdowns.push(predicate.clone());
                                right_pushdowns.push(predicate);
                            }
                        }
                        (false, false) => {
                            // predicate depends on unique columns on both left and right sides, so we can't push it down
                            kept_predicates.push(predicate);
                        }
                        (true, false) => {
                            if join_type.left_produces_nulls() {
                                kept_predicates.push(predicate);
                            } else {
                                left_pushdowns.push(predicate);
                            }
                        }
                        (false, true) => {
                            if join_type.right_produces_nulls() {
                                kept_predicates.push(predicate);
                            } else {
                                right_pushdowns.push(predicate);
                            }
                        }
                    }
                }

                let left_pushdowns = combine_conjunction(left_pushdowns);
                let right_pushdowns = combine_conjunction(right_pushdowns);

                if left_pushdowns.is_some() || right_pushdowns.is_some() {
                    let kept_predicates = combine_conjunction(kept_predicates);

                    let new_left = left_pushdowns.map_or_else(
                        || left.clone(),
                        |left_pushdowns| {
                            Filter::try_new(left.clone(), left_pushdowns)
                                .unwrap()
                                .into()
                        },
                    );

                    let new_right = right_pushdowns.map_or_else(
                        || right.clone(),
                        |right_pushdowns| {
                            Filter::try_new(right.clone(), right_pushdowns)
                                .unwrap()
                                .into()
                        },
                    );

                    let new_join = Arc::new(LogicalPlan::Join(
                        Join::try_new(new_left, new_right, on.clone(), *join_type, *join_strategy)?
                            .with_key_filtering_config(key_filtering_config.clone()),
                    ));

                    if let Some(kept_predicates) = kept_predicates {
                        Filter::try_new(new_join, kept_predicates).unwrap().into()
                    } else {
                        new_join
                    }
                } else {
                    return Ok(Transformed::no(plan));
                }
            }
            LogicalPlan::Aggregate(..)
            | LogicalPlan::Window(..)
            | LogicalPlan::Limit(_)
            | LogicalPlan::Offset(_)
            | LogicalPlan::TopN(..)
            | LogicalPlan::Sample(..)
            | LogicalPlan::Explode(..)
            | LogicalPlan::Shard(..)
            | LogicalPlan::UDFProject(..)
            | LogicalPlan::Unpivot(..)
            | LogicalPlan::Pivot(..)
            | LogicalPlan::Intersect(..)
            | LogicalPlan::Union(..)
            | LogicalPlan::Sink(..)
            | LogicalPlan::MonotonicallyIncreasingId(..)
            | LogicalPlan::SubqueryAlias(..)
            | LogicalPlan::AsofJoin(..)
            | LogicalPlan::VLLMProject(..)
            | LogicalPlan::StageCheckpointKeys(..) => {
                return Ok(Transformed::no(plan));
            }
        };
        Ok(Transformed::yes(new_plan))
    }
}

/// The output columns of `exprs` that pass an input column through unchanged (a
/// column, or an alias of one), each mapped to the input column it reads.
pub(crate) fn passed_through_columns(exprs: &[ExprRef]) -> HashMap<String, ExprRef> {
    exprs
        .iter()
        .filter_map(|e| {
            e.input_mapping()
                .map(|input| (e.name().to_string(), resolved_col(input)))
        })
        .collect()
}

/// The largest share of a grouping's input rows a conjunct may be estimated to keep
/// and still move below the grouping.
///
/// Below, the conjunct copies every input row it keeps; above, only one row per
/// group. Over 10M rows in 100,000 groups, a filter keeping half the rows ran
/// faster below an aggregate (31 ms against 38), one keeping 90% or all of them
/// ran slower (55–59 ms against 37), and one pruning row groups ran 3× faster.
const MOST_KEPT_BELOW_GROUPING: f64 = 0.5;

/// Splits `predicate` into the conjunction that may run below a node, rewritten
/// onto its input, and the conjunction that stays above it; either is `None` when
/// no conjunct belongs to it.
///
/// A conjunct moves when every column it reads is one `passed_through` maps to
/// the node's input. Below a grouping, whose output schema is `grouping`, it must
/// also be deterministic, so that it runs once per row with the answer it would
/// give once per group, and estimated to keep at most
/// [`MOST_KEPT_BELOW_GROUPING`] of the rows, so that moving it saves more than it
/// costs.
fn split_conjuncts(
    predicate: &ExprRef,
    passed_through: &HashMap<String, ExprRef>,
    grouping: Option<&Schema>,
) -> (Option<ExprRef>, Option<ExprRef>) {
    let mut below = vec![];
    let mut above = vec![];
    for conjunct in split_conjunction(predicate) {
        let movable = get_required_columns(&conjunct)
            .iter()
            .all(|column| passed_through.contains_key(column))
            && grouping.is_none_or(|schema| {
                conjunct.is_deterministic()
                    && estimated_selectivity(&conjunct, schema) <= MOST_KEPT_BELOW_GROUPING
            });
        if movable {
            below.push(replace_columns_with_expressions(conjunct, passed_through));
        } else {
            above.push(conjunct);
        }
    }
    (combine_conjunction(below), combine_conjunction(above))
}

/// Rebuilds `node` over its only `input` filtered by `below`, and filters the
/// result by `above`, if given.
fn place_around(
    node: &LogicalPlan,
    input: &Arc<LogicalPlan>,
    below: ExprRef,
    above: Option<ExprRef>,
) -> DaftResult<Arc<LogicalPlan>> {
    let filtered_input: LogicalPlan = Filter::try_new(input.clone(), below)?.into();
    let rebuilt: Arc<LogicalPlan> = node.with_new_children(&[filtered_input.into()]).into();
    Ok(match above {
        Some(above) => Arc::new(Filter::try_new(rebuilt, above)?.into()),
        None => rebuilt,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use common_error::DaftResult;
    use daft_core::prelude::*;
    use daft_dsl::{ExprRef, functions::BuiltinScalarFn, lit, resolved_col, unresolved_col};
    use daft_functions_uri::download::UrlDownload;
    use daft_scan::Pushdowns;
    use rstest::rstest;

    use crate::{
        LogicalPlan,
        optimization::{
            optimizer::{RuleBatch, RuleExecutionStrategy},
            rules::PushDownFilter,
            test::assert_optimized_plan_with_rules_eq,
        },
        test::{dummy_scan_node, dummy_scan_node_with_pushdowns, dummy_scan_operator},
    };

    /// Helper that creates an optimizer with the PushDownFilter rule registered, optimizes
    /// the provided plan with said optimizer, and compares the optimized plan with
    /// the provided expected plan.
    fn assert_optimized_plan_eq(
        plan: Arc<LogicalPlan>,
        expected: Arc<LogicalPlan>,
        strict_pushdown: bool,
    ) -> DaftResult<()> {
        assert_optimized_plan_with_rules_eq(
            plan,
            expected,
            vec![RuleBatch::new(
                vec![Box::new(PushDownFilter::new(strict_pushdown))],
                RuleExecutionStrategy::Once,
            )],
        )
    }

    /// Tests that we can't pushdown a filter into a ScanOperator that has a limit.
    #[test]
    fn filter_not_pushed_down_into_scan_with_limit() -> DaftResult<()> {
        let scan_op = dummy_scan_operator(vec![
            Field::new("a", DataType::Int64),
            Field::new("b", DataType::Utf8),
        ]);
        let plan =
            dummy_scan_node_with_pushdowns(scan_op, Pushdowns::default().with_limit(Some(1)))
                .filter(resolved_col("a").lt(lit(2)))?
                .build();
        // Plan should be unchanged after optimization.
        let expected = plan.clone();
        assert_optimized_plan_eq(plan, expected, false)?;
        Ok(())
    }

    /// Tests combining of two Filters by merging their predicates.
    #[rstest]
    fn filter_combine_with_filter(#[values(false, true)] push_into_scan: bool) -> DaftResult<()> {
        use daft_dsl::resolved_col;

        let scan_op = dummy_scan_operator(vec![
            Field::new("a", DataType::Int64),
            Field::new("b", DataType::Utf8),
        ]);
        let scan_plan = dummy_scan_node_with_pushdowns(
            scan_op.clone(),
            Pushdowns::default().with_limit(if push_into_scan { None } else { Some(1) }),
        );
        let p1 = resolved_col("a").lt(lit(2));
        let p2 = resolved_col("b").eq(lit("foo"));
        let plan = scan_plan.filter(p1.clone())?.filter(p2.clone())?.build();
        let merged_filter = p2.and(p1);
        let expected = if push_into_scan {
            // Merged filter should be pushed into scan.
            dummy_scan_node_with_pushdowns(
                scan_op,
                Pushdowns::default().with_filters(Some(merged_filter)),
            )
            .build()
        } else {
            // Merged filter should not be pushed into scan.
            scan_plan.filter(merged_filter)?.build()
        };
        assert_optimized_plan_eq(plan, expected, false)?;
        Ok(())
    }

    /// Tests that we can't pushdown a filter into a ScanOperator if it has an udf-ish expression.
    #[test]
    fn filter_with_udf_not_pushed_down_into_scan() -> DaftResult<()> {
        let pred: ExprRef = BuiltinScalarFn::new_async(
            UrlDownload,
            vec![resolved_col("a"), lit(1), lit(true), lit(true)],
        )
        .into();
        let plan = dummy_scan_node(dummy_scan_operator(vec![
            Field::new("a", DataType::Int64),
            Field::new("b", DataType::Utf8),
        ]))
        .filter(pred.is_null())?
        .build();
        let expected = plan.clone();
        assert_optimized_plan_eq(plan, expected, false)?;
        Ok(())
    }

    /// Tests that Filter commutes with Projections.
    #[rstest]
    fn filter_commutes_with_projection(
        #[values(false, true)] push_into_scan: bool,
    ) -> DaftResult<()> {
        let scan_op = dummy_scan_operator(vec![
            Field::new("a", DataType::Int64),
            Field::new("b", DataType::Utf8),
        ]);
        let scan_plan = dummy_scan_node_with_pushdowns(
            scan_op.clone(),
            Pushdowns::default().with_limit(if push_into_scan { None } else { Some(1) }),
        );
        let pred = resolved_col("a").lt(lit(2));
        let proj = vec![resolved_col("a")];
        let plan = scan_plan
            .select(proj.clone())?
            .filter(pred.clone())?
            .build();
        let expected_scan_filter = if push_into_scan {
            dummy_scan_node_with_pushdowns(scan_op, Pushdowns::default().with_filters(Some(pred)))
        } else {
            scan_plan.filter(pred)?
        };
        let expected = expected_scan_filter.select(proj)?.build();
        assert_optimized_plan_eq(plan, expected, false)?;
        Ok(())
    }

    /// Tests that a Filter with multiple columns in its predicate commutes with a Projection on both of those columns.
    #[rstest]
    fn filter_commutes_with_projection_multi(
        #[values(false, true)] push_into_scan: bool,
    ) -> DaftResult<()> {
        let scan_op = dummy_scan_operator(vec![
            Field::new("a", DataType::Int64),
            Field::new("b", DataType::Utf8),
        ]);
        let scan_plan = dummy_scan_node_with_pushdowns(
            scan_op.clone(),
            Pushdowns::default().with_limit(if push_into_scan { None } else { Some(1) }),
        );
        let pred = resolved_col("a")
            .lt(lit(2))
            .and(resolved_col("b").eq(lit("foo")));
        let proj = vec![resolved_col("a"), resolved_col("b")];
        let plan = scan_plan
            .select(proj.clone())?
            .filter(pred.clone())?
            .build();
        let expected_scan_filter = if push_into_scan {
            dummy_scan_node_with_pushdowns(scan_op, Pushdowns::default().with_filters(Some(pred)))
        } else {
            scan_plan.filter(pred)?
        };
        let expected = expected_scan_filter.select(proj)?.build();
        assert_optimized_plan_eq(plan, expected, false)?;
        Ok(())
    }

    /// Tests that Filter does not commute with a Projection if the projection expression involves compute.
    #[test]
    fn filter_does_not_commute_with_projection_if_compute() -> DaftResult<()> {
        let plan = dummy_scan_node(dummy_scan_operator(vec![
            Field::new("a", DataType::Int64),
            Field::new("b", DataType::Utf8),
        ]))
        // Projection involves compute on filtered column "a".
        .select(vec![resolved_col("a").add(lit(1))])?
        .filter(resolved_col("a").lt(lit(2)))?
        .build();
        // Filter should NOT commute with Project, since this would involve redundant computation.
        let expected = plan.clone();
        assert_optimized_plan_eq(plan, expected, false)?;
        Ok(())
    }

    /// Tests that Filter commutes with Projection if projection expression involves deterministic compute.
    #[ignore = "No expression attribute indicating whether deterministic && (pure || idempotent)."]
    #[rstest]
    fn filter_commutes_with_projection_deterministic_compute(
        #[values(false, true)] push_into_scan: bool,
    ) -> DaftResult<()> {
        let scan_op = dummy_scan_operator(vec![
            Field::new("a", DataType::Int64),
            Field::new("b", DataType::Utf8),
        ]);
        let scan_plan = dummy_scan_node_with_pushdowns(
            scan_op.clone(),
            Pushdowns::default().with_limit(if push_into_scan { None } else { Some(1) }),
        );
        let pred = resolved_col("a").lt(lit(2));
        let proj = vec![resolved_col("a").add(lit(1))];
        let plan = scan_plan
            // Projection involves compute on filtered column "a".
            .select(proj.clone())?
            .filter(pred.clone())?
            .build();
        let expected_filter_scan = if push_into_scan {
            dummy_scan_node_with_pushdowns(scan_op, Pushdowns::default().with_filters(Some(pred)))
        } else {
            scan_plan.filter(pred)?
        };
        let expected = expected_filter_scan.select(proj)?.build();
        assert_optimized_plan_eq(plan, expected, false)?;
        Ok(())
    }

    /// Tests that Filter commutes with Sort.
    #[rstest]
    fn filter_commutes_with_sort(#[values(false, true)] push_into_scan: bool) -> DaftResult<()> {
        let scan_op = dummy_scan_operator(vec![
            Field::new("a", DataType::Int64),
            Field::new("b", DataType::Utf8),
        ]);
        let scan_plan = dummy_scan_node_with_pushdowns(
            scan_op.clone(),
            Pushdowns::default().with_limit(if push_into_scan { None } else { Some(1) }),
        );
        let pred = resolved_col("a").lt(lit(2));
        let sort_by = vec![resolved_col("a")];
        let descending = vec![true];
        let nulls_first = vec![false];
        let plan = scan_plan
            .sort(sort_by.clone(), descending.clone(), nulls_first.clone())?
            .filter(pred.clone())?
            .build();
        let expected_filter_scan = if push_into_scan {
            dummy_scan_node_with_pushdowns(scan_op, Pushdowns::default().with_filters(Some(pred)))
        } else {
            scan_plan.filter(pred)?
        };
        let expected = expected_filter_scan
            .sort(sort_by, descending, nulls_first)?
            .build();
        assert_optimized_plan_eq(plan, expected, false)?;
        Ok(())
    }

    /// Tests that Filter commutes with Repartition.
    #[rstest]
    fn filter_commutes_with_repartition(
        #[values(false, true)] push_into_scan: bool,
    ) -> DaftResult<()> {
        let scan_op = dummy_scan_operator(vec![
            Field::new("a", DataType::Int64),
            Field::new("b", DataType::Utf8),
        ]);
        let scan_plan = dummy_scan_node_with_pushdowns(
            scan_op.clone(),
            Pushdowns::default().with_limit(if push_into_scan { None } else { Some(1) }),
        );
        let pred = resolved_col("a").lt(lit(2));
        let num_partitions = 1;
        let repartition_by = vec![resolved_col("a")];
        let plan = scan_plan
            .hash_repartition(Some(num_partitions), repartition_by.clone())?
            .filter(pred.clone())?
            .build();
        let expected_filter_scan = if push_into_scan {
            dummy_scan_node_with_pushdowns(scan_op, Pushdowns::default().with_filters(Some(pred)))
        } else {
            scan_plan.filter(pred)?
        };
        let expected = expected_filter_scan
            .hash_repartition(Some(num_partitions), repartition_by)?
            .build();
        assert_optimized_plan_eq(plan, expected, false)?;
        Ok(())
    }

    /// Tests that Filter commutes with IntoBatches.
    #[rstest]
    fn filter_commutes_with_into_batches(
        #[values(false, true)] push_into_scan: bool,
    ) -> DaftResult<()> {
        let scan_op = dummy_scan_operator(vec![
            Field::new("a", DataType::Int64),
            Field::new("b", DataType::Utf8),
        ]);
        let scan_plan = dummy_scan_node_with_pushdowns(
            scan_op.clone(),
            Pushdowns::default().with_limit(if push_into_scan { None } else { Some(1) }),
        );
        let pred = resolved_col("a").lt(lit(2));
        let batch_size = 5usize;
        let plan = scan_plan
            .into_batches(batch_size)?
            .filter(pred.clone())?
            .build();
        let expected_filter_scan = if push_into_scan {
            dummy_scan_node_with_pushdowns(scan_op, Pushdowns::default().with_filters(Some(pred)))
        } else {
            scan_plan.filter(pred)?
        };
        let expected = expected_filter_scan.into_batches(batch_size)?.build();
        assert_optimized_plan_eq(plan, expected, false)?;
        Ok(())
    }

    /// Tests that Filter commutes with Concat.
    #[rstest]
    fn filter_commutes_with_concat(
        #[values(false, true)] push_into_left_scan: bool,
        #[values(false, true)] push_into_right_scan: bool,
    ) -> DaftResult<()> {
        let scan_op = dummy_scan_operator(vec![
            Field::new("a", DataType::Int64),
            Field::new("b", DataType::Utf8),
        ]);
        let left_scan_plan = dummy_scan_node_with_pushdowns(
            scan_op.clone(),
            Pushdowns::default().with_limit(if push_into_left_scan { None } else { Some(1) }),
        );
        let right_scan_plan = dummy_scan_node_with_pushdowns(
            scan_op.clone(),
            Pushdowns::default().with_limit(if push_into_right_scan { None } else { Some(1) }),
        );
        let pred = resolved_col("a").lt(lit(2));
        let plan = left_scan_plan
            .concat(&right_scan_plan)?
            .filter(pred.clone())?
            .build();
        let expected_left_filter_scan = if push_into_left_scan {
            dummy_scan_node_with_pushdowns(
                scan_op.clone(),
                Pushdowns::default().with_filters(Some(pred.clone())),
            )
        } else {
            left_scan_plan.filter(pred.clone())?
        };
        let expected_right_filter_scan = if push_into_right_scan {
            dummy_scan_node_with_pushdowns(scan_op, Pushdowns::default().with_filters(Some(pred)))
        } else {
            right_scan_plan.filter(pred)?
        };
        let expected = expected_left_filter_scan
            .concat(&expected_right_filter_scan)?
            .build();
        assert_optimized_plan_eq(plan, expected, false)?;
        Ok(())
    }

    /// Tests that Filter can be pushed into the left side of a Join.
    #[rstest]
    fn filter_commutes_with_join_left_side(
        #[values(false, true)] push_into_left_scan: bool,
        #[values(false, true)] null_equals_null: bool,
        #[values(JoinType::Inner, JoinType::Left, JoinType::Anti, JoinType::Semi)] how: JoinType,
    ) -> DaftResult<()> {
        let left_scan_op = dummy_scan_operator(vec![
            Field::new("a", DataType::Int64),
            Field::new("b", DataType::Utf8),
        ]);
        let right_scan_op = dummy_scan_operator(vec![
            Field::new("right.b", DataType::Utf8),
            Field::new("c", DataType::Float64),
        ]);
        let left_scan_plan = dummy_scan_node_with_pushdowns(
            left_scan_op.clone(),
            Pushdowns::default().with_limit(if push_into_left_scan { None } else { Some(1) }),
        );
        let right_scan_plan = dummy_scan_node(right_scan_op);
        let join_on = if null_equals_null {
            unresolved_col("b").eq_null_safe(unresolved_col("right.b"))
        } else {
            unresolved_col("b").eq(unresolved_col("right.b"))
        };

        let pred = resolved_col("a").lt(lit(2));
        let plan = left_scan_plan
            .join(
                &right_scan_plan,
                join_on.clone().into(),
                vec![],
                how,
                None,
                Default::default(),
            )?
            .filter(pred.clone())?
            .build();
        let expected_left_filter_scan = if push_into_left_scan {
            dummy_scan_node_with_pushdowns(
                left_scan_op,
                Pushdowns::default().with_filters(Some(pred)),
            )
        } else {
            left_scan_plan.filter(pred)?
        };
        let expected = expected_left_filter_scan
            .join(
                &right_scan_plan,
                join_on.into(),
                vec![],
                how,
                None,
                Default::default(),
            )?
            .build();
        assert_optimized_plan_eq(plan, expected, false)?;
        Ok(())
    }

    /// Tests that Filter can be pushed into the right side of a Join.
    #[rstest]
    fn filter_commutes_with_join_right_side(
        #[values(false, true)] push_into_right_scan: bool,
        #[values(false, true)] null_equals_null: bool,
        #[values(JoinType::Inner, JoinType::Right)] how: JoinType,
    ) -> DaftResult<()> {
        let left_scan_op = dummy_scan_operator(vec![
            Field::new("a", DataType::Int64),
            Field::new("b", DataType::Utf8),
        ]);
        let right_scan_op = dummy_scan_operator(vec![
            Field::new("right.b", DataType::Utf8),
            Field::new("c", DataType::Float64),
        ]);
        let left_scan_plan = dummy_scan_node(left_scan_op);
        let right_scan_plan = dummy_scan_node_with_pushdowns(
            right_scan_op.clone(),
            Pushdowns::default().with_limit(if push_into_right_scan { None } else { Some(1) }),
        );
        let join_on = if null_equals_null {
            unresolved_col("b").eq_null_safe(unresolved_col("right.b"))
        } else {
            unresolved_col("b").eq(unresolved_col("right.b"))
        };
        let pred = resolved_col("c").lt(lit(2.0));
        let plan = left_scan_plan
            .join(
                &right_scan_plan,
                join_on.clone().into(),
                vec![],
                how,
                None,
                Default::default(),
            )?
            .filter(pred.clone())?
            .build();
        let expected_right_filter_scan = if push_into_right_scan {
            dummy_scan_node_with_pushdowns(
                right_scan_op,
                Pushdowns::default().with_filters(Some(pred)),
            )
        } else {
            right_scan_plan.filter(pred)?
        };
        let expected = left_scan_plan
            .join(
                &expected_right_filter_scan,
                join_on.into(),
                vec![],
                how,
                None,
                Default::default(),
            )?
            .build();
        assert_optimized_plan_eq(plan, expected, false)?;
        Ok(())
    }

    /// Tests that Filter on join key commutes with Join.
    #[rstest]
    fn filter_commutes_with_join_on_join_key(
        #[values(false, true)] push_into_left_scan: bool,
        #[values(false, true)] push_into_right_scan: bool,
        #[values(
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Outer,
            JoinType::Anti,
            JoinType::Semi
        )]
        how: JoinType,
    ) -> DaftResult<()> {
        let left_scan_op = dummy_scan_operator(vec![
            Field::new("a", DataType::Utf8),
            Field::new("b", DataType::Int64),
            Field::new("c", DataType::Float64),
        ]);
        let right_scan_op = dummy_scan_operator(vec![
            Field::new("b", DataType::Int64),
            Field::new("d", DataType::Boolean),
        ]);
        let left_scan_plan = dummy_scan_node_with_pushdowns(
            left_scan_op.clone(),
            Pushdowns::default().with_limit(if push_into_left_scan { None } else { Some(1) }),
        );
        let right_scan_plan = dummy_scan_node_with_pushdowns(
            right_scan_op.clone(),
            Pushdowns::default().with_limit(if push_into_right_scan { None } else { Some(1) }),
        );
        let pred = resolved_col("b").is_null();
        let plan = left_scan_plan
            .join(
                &right_scan_plan,
                None,
                vec!["b".to_string()],
                how,
                None,
                Default::default(),
            )?
            .filter(pred.clone())?
            .build();
        let expected_left_filter_scan = if push_into_left_scan {
            dummy_scan_node_with_pushdowns(
                left_scan_op,
                Pushdowns::default().with_filters(Some(pred.clone())),
            )
        } else {
            left_scan_plan.filter(pred.clone())?
        };
        let expected_right_filter_scan = if push_into_right_scan {
            dummy_scan_node_with_pushdowns(
                right_scan_op,
                Pushdowns::default().with_filters(Some(pred)),
            )
        } else {
            right_scan_plan.filter(pred)?
        };
        let expected = expected_left_filter_scan
            .join(
                &expected_right_filter_scan,
                None,
                vec!["b".to_string()],
                how,
                None,
                Default::default(),
            )?
            .build();
        assert_optimized_plan_eq(plan, expected, false)?;
        Ok(())
    }

    /// Tests that Filter can be pushed into the left side of a Join.
    #[rstest]
    fn filter_does_not_commute_with_join_left_side(
        #[values(false, true)] null_equals_null: bool,
        #[values(JoinType::Right, JoinType::Outer)] how: JoinType,
    ) -> DaftResult<()> {
        let left_scan_op = dummy_scan_operator(vec![
            Field::new("a", DataType::Int64),
            Field::new("b", DataType::Utf8),
        ]);
        let right_scan_op = dummy_scan_operator(vec![
            Field::new("right.b", DataType::Utf8),
            Field::new("c", DataType::Float64),
        ]);
        let left_scan_plan = dummy_scan_node(left_scan_op);
        let right_scan_plan = dummy_scan_node(right_scan_op);
        let join_on = if null_equals_null {
            unresolved_col("b").eq_null_safe(unresolved_col("right.b"))
        } else {
            unresolved_col("b").eq(unresolved_col("right.b"))
        };
        let pred = resolved_col("a").is_null();
        let plan = left_scan_plan
            .join(
                &right_scan_plan,
                join_on.into(),
                vec![],
                how,
                None,
                Default::default(),
            )?
            .filter(pred)?
            .build();
        // should not push down filter
        let expected = plan.clone();
        assert_optimized_plan_eq(plan, expected, false)?;
        Ok(())
    }

    /// Tests that Filter can be pushed into the right side of a Join.
    #[rstest]
    fn filter_does_not_commute_with_join_right_side(
        #[values(false, true)] null_equals_null: bool,
        #[values(JoinType::Left, JoinType::Outer)] how: JoinType,
    ) -> DaftResult<()> {
        let left_scan_op = dummy_scan_operator(vec![
            Field::new("a", DataType::Int64),
            Field::new("b", DataType::Utf8),
        ]);
        let right_scan_op = dummy_scan_operator(vec![
            Field::new("right.b", DataType::Utf8),
            Field::new("c", DataType::Float64),
        ]);
        let left_scan_plan = dummy_scan_node(left_scan_op);
        let right_scan_plan = dummy_scan_node(right_scan_op);
        let join_on = if null_equals_null {
            unresolved_col("b").eq_null_safe(unresolved_col("right.b"))
        } else {
            unresolved_col("b").eq(unresolved_col("right.b"))
        };

        let pred = resolved_col("c").is_null();
        let plan = left_scan_plan
            .join(
                &right_scan_plan,
                join_on.into(),
                vec![],
                how,
                None,
                Default::default(),
            )?
            .filter(pred)?
            .build();
        // should not push down filter
        let expected = plan.clone();
        assert_optimized_plan_eq(plan, expected, false)?;
        Ok(())
    }

    /// Tests that a complex predicate can be separated so that it can be pushed down into one side of the join.
    /// Modeled after TPC-H Q7
    #[rstest]
    fn filter_commutes_with_join_complex() -> DaftResult<()> {
        let left_scan_op = dummy_scan_operator(vec![Field::new("a", DataType::Utf8)]);
        let right_scan_op = dummy_scan_operator(vec![Field::new("b", DataType::Utf8)]);

        let plan = dummy_scan_node(left_scan_op.clone())
            .join(
                dummy_scan_node(right_scan_op.clone()),
                None,
                vec![],
                JoinType::Inner,
                None,
                Default::default(),
            )?
            .filter(
                (resolved_col("a")
                    .eq(lit("FRANCE"))
                    .and(resolved_col("b").eq(lit("GERMANY"))))
                .or(resolved_col("a")
                    .eq(lit("GERMANY"))
                    .and(resolved_col("b").eq(lit("FRANCE")))),
            )?
            .build();

        let expected = dummy_scan_node_with_pushdowns(
            left_scan_op,
            Pushdowns::default().with_filters(Some(
                resolved_col("a")
                    .eq(lit("FRANCE"))
                    .or(resolved_col("a").eq(lit("GERMANY"))),
            )),
        )
        .join(
            dummy_scan_node_with_pushdowns(
                right_scan_op,
                Pushdowns::default().with_filters(Some(
                    resolved_col("b")
                        .eq(lit("GERMANY"))
                        .or(resolved_col("b").eq(lit("FRANCE"))),
                )),
            ),
            None,
            vec![],
            JoinType::Inner,
            None,
            Default::default(),
        )?
        .filter(
            (resolved_col("a")
                .eq(lit("FRANCE"))
                .or(resolved_col("b").eq(lit("FRANCE"))))
            .and(
                resolved_col("b")
                    .eq(lit("GERMANY"))
                    .or(resolved_col("a").eq(lit("GERMANY"))),
            ),
        )?
        .build();

        assert_optimized_plan_eq(plan, expected, false)?;

        Ok(())
    }

    fn filter_pushdown_strict_mode_scenario(enable_strict_pushdown: bool) -> DaftResult<()> {
        let pred = resolved_col("value").is_in(vec![lit(2)]);
        let scan_op = dummy_scan_operator(vec![
            Field::new("date_col", DataType::Date),
            Field::new("value", DataType::Int64),
        ]);

        let plan = dummy_scan_node(scan_op.clone())
            .filter(pred.clone())?
            .build();

        let expected = if enable_strict_pushdown {
            plan.clone()
        } else {
            dummy_scan_node_with_pushdowns(scan_op, Pushdowns::default().with_filters(Some(pred)))
                .build()
        };

        assert_optimized_plan_eq(plan, expected, enable_strict_pushdown)?;

        Ok(())
    }

    /// Tests that Filter pushdown respects the strict mode configuration.
    /// The main reason for not using rstest is that it seems unable to handle the setting of environment variables properly.
    #[test]
    fn filter_pushdown_strict_mode_true() -> DaftResult<()> {
        filter_pushdown_strict_mode_scenario(true)
    }

    #[test]
    fn filter_pushdown_strict_mode_false() -> DaftResult<()> {
        filter_pushdown_strict_mode_scenario(false)
    }

    /// Tests that Filter on a column that exists in both left and right input schemas
    /// is NOT pushed to the right side for anti/semi joins.
    /// This addresses GitHub issue #6086.
    #[rstest]
    fn filter_not_pushed_to_right_for_anti_semi_join_with_common_column(
        #[values(JoinType::Anti, JoinType::Semi)] how: JoinType,
    ) -> DaftResult<()> {
        // Left has columns: id, foo
        let left_scan_op = dummy_scan_operator(vec![
            Field::new("id", DataType::Int64),
            Field::new("foo", DataType::Int64),
        ]);
        // Right has columns: left_id, foo (foo exists in both!)
        let right_scan_op = dummy_scan_operator(vec![
            Field::new("left_id", DataType::Int64),
            Field::new("foo", DataType::Int64),
        ]);
        let left_scan_plan = dummy_scan_node(left_scan_op.clone());
        let right_scan_plan = dummy_scan_node(right_scan_op);

        // Filter on foo (which exists in both input schemas)
        let pred = resolved_col("foo").eq(lit(0));

        // Plan: Filter(Join(left, right))
        let plan = left_scan_plan
            .join(
                &right_scan_plan,
                Some(unresolved_col("id").eq(unresolved_col("left_id"))),
                vec![],
                how,
                None,
                Default::default(),
            )?
            .filter(pred.clone())?
            .build();

        // Expected: Filter should only be pushed to left side, NOT to right side
        // This is because anti/semi join output only contains left columns,
        // so filtering the right side would incorrectly change the join semantics.
        let expected = dummy_scan_node_with_pushdowns(
            left_scan_op,
            Pushdowns::default().with_filters(Some(pred)),
        )
        .join(
            &right_scan_plan,
            Some(unresolved_col("id").eq(unresolved_col("left_id"))),
            vec![],
            how,
            None,
            Default::default(),
        )?
        .build();

        assert_optimized_plan_eq(plan, expected, false)?;
        Ok(())
    }

    /// A scan of `k` and `v` whose limit keeps filters from being absorbed into it,
    /// so a filter pushed down stays visible directly above it. Plans compared
    /// with each other must share one scan.
    fn grouped_scan() -> crate::LogicalPlanBuilder {
        thread_local! {
            static SCAN: crate::LogicalPlanBuilder = dummy_scan_node_with_pushdowns(
                dummy_scan_operator(vec![
                    Field::new("k", DataType::Int64),
                    Field::new("v", DataType::Int64),
                ]),
                Pushdowns::default().with_limit(Some(1)),
            );
        }
        SCAN.with(Clone::clone)
    }

    /// A predicate the engine cannot know to be deterministic: a Python function.
    fn not_known_deterministic(input: ExprRef) -> ExprRef {
        ExprRef::from(daft_dsl::Expr::Function {
            func: daft_dsl::functions::FunctionExpr::Python(
                daft_dsl::functions::python::LegacyPythonUDF::new_testing_udf(),
            ),
            inputs: vec![input],
        })
        .gt(lit(0))
    }

    /// A conjunct on the group key runs below the aggregate; one on its result stays.
    #[test]
    fn filter_on_group_key_runs_below_aggregate() -> DaftResult<()> {
        let aggregations = vec![resolved_col("v").sum().alias("total")];
        let on_key = resolved_col("k").eq(lit(1));
        let on_total = resolved_col("total").gt(lit(10));
        let plan = grouped_scan()
            .aggregate(aggregations.clone(), vec![resolved_col("k")])?
            .filter(on_key.clone().and(on_total.clone()))?
            .build();
        let expected = grouped_scan()
            .filter(on_key)?
            .aggregate(aggregations, vec![resolved_col("k")])?
            .filter(on_total)?
            .build();
        assert_optimized_plan_eq(plan, expected, false)
    }

    /// A group key that renames its column is read by its input's name below.
    #[test]
    fn filter_on_renamed_group_key_runs_below_aggregate() -> DaftResult<()> {
        let aggregations = vec![resolved_col("v").sum().alias("total")];
        let groupby = vec![resolved_col("k").alias("key")];
        let plan = grouped_scan()
            .aggregate(aggregations.clone(), groupby.clone())?
            .filter(resolved_col("key").eq(lit(1)))?
            .build();
        let expected = grouped_scan()
            .filter(resolved_col("k").eq(lit(1)))?
            .aggregate(aggregations, groupby)?
            .build();
        assert_optimized_plan_eq(plan, expected, false)
    }

    /// A filter stays above an aggregate it cannot move below: a global one, which
    /// answers a row even over no rows; one over a computed group key; one whose
    /// predicate is not known to be deterministic; and one estimated to keep most
    /// rows, which costs more below than it saves.
    #[rstest]
    #[case::global(vec![], lit(false))]
    #[case::computed_key(vec![resolved_col("k").add(lit(1)).alias("k1")], resolved_col("k1").eq(lit(1)))]
    #[case::not_deterministic(vec![resolved_col("k")], not_known_deterministic(resolved_col("k")))]
    #[case::keeps_most_rows(vec![resolved_col("k")], resolved_col("k").not_null())]
    fn filter_stays_above_aggregate(
        #[case] groupby: Vec<ExprRef>,
        #[case] predicate: ExprRef,
    ) -> DaftResult<()> {
        let plan = grouped_scan()
            .aggregate(vec![resolved_col("v").sum().alias("total")], groupby)?
            .filter(predicate)?
            .build();
        let expected = plan.clone();
        assert_optimized_plan_eq(plan, expected, false)
    }

    fn windowed(
        scan: crate::LogicalPlanBuilder,
        partition_by: Vec<ExprRef>,
    ) -> DaftResult<Arc<LogicalPlan>> {
        use daft_dsl::{WindowExpr, expr::window::WindowSpec};

        let window_spec = Arc::new(WindowSpec {
            partition_by,
            ..Default::default()
        });
        let total: WindowExpr = resolved_col("v").sum().try_into()?;
        Ok(LogicalPlan::from(crate::ops::Window::try_new(
            scan.build(),
            vec![total],
            vec!["total".to_string()],
            window_spec,
        )?)
        .arced())
    }

    /// A conjunct on the partition key runs below the window; one on its result stays.
    #[test]
    fn filter_on_partition_key_runs_below_window() -> DaftResult<()> {
        let on_key = resolved_col("k").eq(lit(1));
        let on_total = resolved_col("total").gt(lit(10));
        let plan =
            crate::LogicalPlanBuilder::from(windowed(grouped_scan(), vec![resolved_col("k")])?)
                .filter(on_key.clone().and(on_total.clone()))?
                .build();
        let expected = crate::LogicalPlanBuilder::from(windowed(
            grouped_scan().filter(on_key)?,
            vec![resolved_col("k")],
        )?)
        .filter(on_total)?
        .build();
        assert_optimized_plan_eq(plan, expected, false)
    }

    /// A filter stays above a window with no partitions, and one estimated to keep
    /// most rows stays above any window.
    #[rstest]
    #[case::no_partitions(vec![], resolved_col("k").eq(lit(1)))]
    #[case::keeps_most_rows(vec![resolved_col("k")], resolved_col("k").not_null())]
    fn filter_stays_above_window(
        #[case] partition_by: Vec<ExprRef>,
        #[case] predicate: ExprRef,
    ) -> DaftResult<()> {
        let plan = crate::LogicalPlanBuilder::from(windowed(grouped_scan(), partition_by)?)
            .filter(predicate)?
            .build();
        let expected = plan.clone();
        assert_optimized_plan_eq(plan, expected, false)
    }

    /// Distinct over whole rows lets any deterministic conjunct below; distinct on
    /// some columns, only a conjunct on those.
    #[rstest]
    #[case::whole_rows(None, resolved_col("v").eq(lit(1)), true)]
    #[case::on_its_columns(Some(vec![resolved_col("k")]), resolved_col("k").eq(lit(1)), true)]
    #[case::on_other_columns(Some(vec![resolved_col("k")]), resolved_col("v").eq(lit(1)), false)]
    #[case::not_deterministic(None, not_known_deterministic(resolved_col("k")), false)]
    #[case::keeps_most_rows(None, resolved_col("k").not_null(), false)]
    fn filter_through_distinct(
        #[case] columns: Option<Vec<ExprRef>>,
        #[case] predicate: ExprRef,
        #[case] moves: bool,
    ) -> DaftResult<()> {
        let plan = grouped_scan()
            .distinct(columns.clone())?
            .filter(predicate.clone())?
            .build();
        let expected = if moves {
            grouped_scan().filter(predicate)?.distinct(columns)?.build()
        } else {
            plan.clone()
        };
        assert_optimized_plan_eq(plan, expected, false)
    }
}
