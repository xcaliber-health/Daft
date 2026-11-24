//! Pushes what every reader of a shared part needs into the part.
//!
//! A shared node is a barrier to pushdown, so after the pushdown rules each
//! reader's filters and projections stop right above its occurrence. This rule
//! gathers them and pushes into every occurrence alike:
//! - the columns any reader reads, so the part drops the rest;
//! - the conjuncts every reader's filter holds, and the OR of what remains of
//!   each, so the part drops the rows no reader keeps. Only deterministic
//!   conjuncts are lifted, and each reader keeps its own filter, so a filter is
//!   never evaluated more often than written in a way that changes an answer.
//!
//! Following the guard of other engines' common subplan elimination, a part is
//! no longer shared when every reader filters it differently and each filter
//! would reach below the part on its own: each reader then reads less by itself
//! than all of them read together.
//!
//! [`PushLimitsIntoSharedParts`] gives a part the largest of its readers' limits
//! when every reader takes one, so a shared sort still becomes a top-N.
//!
//! An id left with one occurrence is unwrapped; it has nobody to share with.

use std::{collections::HashMap, sync::Arc};

use common_error::DaftResult;
use common_treenode::{DynTreeNode, Transformed, TreeNode, TreeNodeRecursion};
use daft_algebra::boolean::{combine_conjunction, combine_disjunction, split_conjunction};
use daft_dsl::{ExprRef, optimization::get_required_columns, resolved_col};
use indexmap::IndexSet;

use super::{
    OptimizerRule, PushDownFilter,
    share_repeated_parts::{shared_occurrences, unwrap_shared},
};
use crate::{
    LogicalPlan, LogicalPlanRef,
    ops::{Filter, Limit, Project, Shared, TopN},
};

/// Most passes the probe of the guard runs filter pushdown for.
const PROBE_PASSES: usize = 20;

/// What one reader of a shared part reads, from the filters and projection right above it.
struct ReaderNeeds {
    /// The deterministic conjuncts of the filters between the part and the first projection.
    conjuncts: Vec<ExprRef>,
    /// The columns of the part it reads, or `None` for all of them.
    columns: Option<IndexSet<String>>,
}

/// Pushes the columns and rows every reader of a shared part needs into the part.
#[derive(Debug)]
pub struct PushDownIntoSharedParts {
    strict_pushdown: bool,
}

impl PushDownIntoSharedParts {
    /// A rule that pushes into each shared part what all of its readers need; with
    /// `strict_pushdown`, its probe pushes filters as filter pushdown does in strict mode.
    pub fn new(strict_pushdown: bool) -> Self {
        Self { strict_pushdown }
    }
}

impl OptimizerRule for PushDownIntoSharedParts {
    fn try_optimize(&self, plan: Arc<LogicalPlan>) -> DaftResult<Transformed<Arc<LogicalPlan>>> {
        let unwrapped = unwrap_lone(plan)?;
        let mut transformed = unwrapped.transformed;
        let mut plan = unwrapped.data;

        let mut readers: HashMap<usize, (LogicalPlanRef, Vec<ReaderNeeds>)> = HashMap::new();
        gather(&plan, &mut Vec::new(), &mut |shared, above| {
            let entry = readers
                .entry(shared.id)
                .or_insert_with(|| (shared.input.clone(), Vec::new()));
            entry.1.push(reader_needs(above));
        });

        let mut to_unwrap = Vec::new();
        let mut additions: HashMap<usize, (Option<ExprRef>, Option<Vec<ExprRef>>)> = HashMap::new();
        for (id, (body, needs)) in &readers {
            if self.each_reads_less_alone(body, needs)? {
                to_unwrap.push(*id);
                continue;
            }
            let predicate = shared_predicate(needs);
            let projection = shared_projection(body, needs);
            if predicate.is_some() || projection.is_some() {
                additions.insert(*id, (predicate, projection));
            }
        }

        let unwrapped = unwrap_shared(plan, &to_unwrap)?;
        transformed |= unwrapped.transformed;
        plan = unwrapped.data;

        if !additions.is_empty() {
            let pushed = plan.transform_down(|node| {
                let LogicalPlan::Shared(shared) = node.as_ref() else {
                    return Ok(Transformed::no(node));
                };
                let Some((predicate, projection)) = additions.get(&shared.id) else {
                    return Ok(Transformed::new(node, false, TreeNodeRecursion::Jump));
                };
                let mut body = shared.input.clone();
                if let Some(predicate) = predicate {
                    body = Filter::try_new(body, predicate.clone())?.into();
                }
                if let Some(projection) = projection {
                    body = Project::try_new(body, projection.clone())?.into();
                }
                Ok(Transformed::new(
                    Shared::new(body, shared.id).into(),
                    true,
                    TreeNodeRecursion::Jump,
                ))
            })?;
            transformed |= pushed.transformed;
            plan = pushed.data;
        }
        Ok(Transformed::new(
            plan,
            transformed,
            TreeNodeRecursion::Continue,
        ))
    }
}

impl PushDownIntoSharedParts {
    /// Whether every reader filters the part beyond what all of them share, those filters
    /// differ, and each would reach below the part on its own, so the readers read less
    /// apart than together. What every reader shares enters the part either way.
    fn each_reads_less_alone(
        &self,
        body: &LogicalPlanRef,
        needs: &[ReaderNeeds],
    ) -> DaftResult<bool> {
        let common = common_conjuncts(needs);
        let mut remainders = Vec::with_capacity(needs.len());
        for reader in needs {
            let own = reader
                .conjuncts
                .iter()
                .filter(|conjunct| !common.contains(conjunct))
                .cloned();
            match combine_conjunction(own) {
                Some(remainder) => remainders.push(remainder),
                None => return Ok(false),
            }
        }
        if remainders
            .iter()
            .all(|remainder| *remainder == remainders[0])
        {
            return Ok(false);
        }
        for remainder in remainders {
            if !self.reaches_below(body, &remainder)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Whether filter pushdown moves `predicate` off the top of `body`.
    fn reaches_below(&self, body: &LogicalPlanRef, predicate: &ExprRef) -> DaftResult<bool> {
        let rule = PushDownFilter::new(self.strict_pushdown);
        let mut probe: LogicalPlanRef = Filter::try_new(body.clone(), predicate.clone())?.into();
        for _ in 0..PROBE_PASSES {
            let pass = rule.try_optimize(probe)?;
            probe = pass.data;
            if !pass.transformed {
                break;
            }
        }
        let mut top = probe;
        while let LogicalPlan::Project(Project { input, .. }) = top.as_ref() {
            top = input.clone();
        }
        Ok(
            !matches!(top.as_ref(), LogicalPlan::Filter(Filter { predicate: left, .. }) if left == predicate),
        )
    }
}

/// Unwraps every id with fewer than two occurrences.
fn unwrap_lone(plan: LogicalPlanRef) -> DaftResult<Transformed<LogicalPlanRef>> {
    let lone: Vec<usize> = shared_occurrences(&plan)?
        .into_iter()
        .filter_map(|(id, count)| (count < 2).then_some(id))
        .collect();
    unwrap_shared(plan, &lone)
}

/// Calls `found` for each shared node with the nodes above it, nearest first.
#[recursive::recursive]
fn gather(
    node: &LogicalPlanRef,
    above: &mut Vec<LogicalPlanRef>,
    found: &mut impl FnMut(&Shared, &[LogicalPlanRef]),
) {
    if let LogicalPlan::Shared(shared) = node.as_ref() {
        let nearest_first: Vec<LogicalPlanRef> = above.iter().rev().cloned().collect();
        found(shared, &nearest_first);
        return;
    }
    above.push(node.clone());
    for child in node.arc_children() {
        gather(&child, above, found);
    }
    above.pop();
}

/// What a reader reads of the part, from the filters and the projection right above it.
fn reader_needs(above: &[LogicalPlanRef]) -> ReaderNeeds {
    let mut conjuncts = Vec::new();
    let mut filtered_columns = IndexSet::new();
    for node in above {
        match node.as_ref() {
            LogicalPlan::Filter(Filter { predicate, .. }) => {
                for conjunct in split_conjunction(predicate) {
                    filtered_columns.extend(get_required_columns(&conjunct));
                    if conjunct.is_deterministic() && !conjuncts.contains(&conjunct) {
                        conjuncts.push(conjunct);
                    }
                }
            }
            LogicalPlan::Project(project) => {
                let mut columns: IndexSet<String> = project
                    .projection
                    .iter()
                    .flat_map(get_required_columns)
                    .collect();
                columns.extend(filtered_columns);
                return ReaderNeeds {
                    conjuncts,
                    columns: Some(columns),
                };
            }
            _ => break,
        }
    }
    ReaderNeeds {
        conjuncts,
        columns: None,
    }
}

/// The conjuncts every reader's filter holds.
fn common_conjuncts(needs: &[ReaderNeeds]) -> Vec<ExprRef> {
    let Some(first) = needs.first() else {
        return Vec::new();
    };
    first
        .conjuncts
        .iter()
        .filter(|conjunct| {
            needs
                .iter()
                .all(|reader| reader.conjuncts.contains(conjunct))
        })
        .cloned()
        .collect()
}

/// The common conjuncts AND the OR of the rest, or `None` when some reader keeps every row.
fn shared_predicate(needs: &[ReaderNeeds]) -> Option<ExprRef> {
    if needs.iter().any(|reader| reader.conjuncts.is_empty()) {
        return None;
    }
    let common = common_conjuncts(needs);
    let remainders: Vec<Option<ExprRef>> = needs
        .iter()
        .map(|reader| {
            combine_conjunction(
                reader
                    .conjuncts
                    .iter()
                    .filter(|conjunct| !common.contains(conjunct))
                    .cloned(),
            )
        })
        .collect();
    // A reader whose filter is only the common part keeps every row the common part keeps.
    let alternatives = if remainders.iter().any(Option::is_none) {
        None
    } else {
        combine_disjunction(remainders.into_iter().flatten())
    };
    combine_conjunction(common.into_iter().chain(alternatives))
}

/// The columns any reader reads, in the part's order, or `None` when it reads all of them.
fn shared_projection(body: &LogicalPlanRef, needs: &[ReaderNeeds]) -> Option<Vec<ExprRef>> {
    let mut read: IndexSet<String> = IndexSet::new();
    for reader in needs {
        match &reader.columns {
            Some(columns) => read.extend(columns.iter().cloned()),
            None => return None,
        }
    }
    let schema = body.schema();
    if schema.field_names().all(|name| read.contains(name)) {
        return None;
    }
    let kept: Vec<ExprRef> = schema
        .field_names()
        .filter(|name| read.contains(*name))
        .map(resolved_col)
        .collect();
    // A part read for no column at all (a count of its rows) still keeps one.
    if kept.is_empty() {
        return schema
            .field_names()
            .next()
            .map(|name| vec![resolved_col(name)]);
    }
    Some(kept)
}

/// Gives a shared part the largest of its readers' limits, when every reader takes one
/// right above it, so a sort in the part becomes a top-N once.
#[derive(Default, Debug)]
pub struct PushLimitsIntoSharedParts {}

impl PushLimitsIntoSharedParts {
    /// A rule that limits a shared part when every reader limits it.
    pub fn new() -> Self {
        Self {}
    }
}

impl OptimizerRule for PushLimitsIntoSharedParts {
    fn try_optimize(&self, plan: Arc<LogicalPlan>) -> DaftResult<Transformed<Arc<LogicalPlan>>> {
        // For each id, the largest row count a reader takes, or `None` once one takes all.
        let mut taken: HashMap<usize, Option<u64>> = HashMap::new();
        gather(&plan, &mut Vec::new(), &mut |shared, above| {
            let rows = match above.first().map(AsRef::as_ref) {
                Some(LogicalPlan::Limit(Limit { limit, offset, .. })) => {
                    Some(limit.saturating_add(offset.unwrap_or(0)))
                }
                _ => None,
            };
            let entry = taken.entry(shared.id).or_insert(Some(0));
            *entry = entry.zip(rows).map(|(most, rows)| most.max(rows));
        });
        taken.retain(|_, rows| rows.is_some());
        if taken.is_empty() {
            return Ok(Transformed::no(plan));
        }
        plan.transform_down(|node| {
            let LogicalPlan::Shared(shared) = node.as_ref() else {
                return Ok(Transformed::no(node));
            };
            let Some(Some(rows)) = taken.get(&shared.id) else {
                return Ok(Transformed::new(node, false, TreeNodeRecursion::Jump));
            };
            // Already limited to as many rows, as a limit or a top-N; nothing to add.
            let limited = match shared.input.as_ref() {
                LogicalPlan::Limit(Limit { limit, offset, .. })
                | LogicalPlan::TopN(TopN { limit, offset, .. }) => {
                    limit.saturating_add(offset.unwrap_or(0)) <= *rows
                }
                _ => false,
            };
            if limited {
                return Ok(Transformed::new(node, false, TreeNodeRecursion::Jump));
            }
            let body: LogicalPlanRef = Limit::new(shared.input.clone(), *rows, None, false).into();
            Ok(Transformed::new(
                Shared::new(body, shared.id).into(),
                true,
                TreeNodeRecursion::Jump,
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use common_error::DaftResult;
    use common_treenode::{TreeNode, TreeNodeRecursion};
    use daft_core::prelude::*;
    use daft_dsl::{ExprRef, lit, unresolved_col};

    use super::{PushDownIntoSharedParts, PushLimitsIntoSharedParts};
    use crate::{
        LogicalPlan, LogicalPlanBuilder,
        ops::Shared,
        optimization::{
            optimizer::{RuleBatch, RuleExecutionStrategy},
            test::optimize_with_rules,
        },
        test::{dummy_scan_node, dummy_scan_operator},
    };

    fn col(name: &str) -> ExprRef {
        unresolved_col(name)
    }

    /// One shared aggregation of a scan, by `k`, with three columns.
    fn part() -> LogicalPlanBuilder {
        let rows = dummy_scan_node(dummy_scan_operator(vec![
            Field::new("k", DataType::Int64),
            Field::new("v", DataType::Int64),
            Field::new("w", DataType::Int64),
        ]));
        let aggregated = rows
            .aggregate(
                vec![col("v").sum().alias("s"), col("w").max().alias("m")],
                vec![col("k")],
            )
            .unwrap();
        LogicalPlanBuilder::from(Arc::new(LogicalPlan::Shared(Shared::new(
            aggregated.build(),
            0,
        ))))
    }

    fn pushed(plan: Arc<LogicalPlan>) -> DaftResult<Arc<LogicalPlan>> {
        optimize_with_rules(
            plan,
            vec![RuleBatch::new(
                vec![Box::new(PushDownIntoSharedParts::new(false))],
                RuleExecutionStrategy::Once,
            )],
        )
    }

    /// The body of each occurrence, as its top two nodes written out.
    fn bodies(plan: &Arc<LogicalPlan>) -> Vec<Vec<String>> {
        let mut found = vec![];
        plan.apply(|node| {
            if let LogicalPlan::Shared(shared) = node.as_ref() {
                let mut tops = vec![];
                let mut body = shared.input.clone();
                for _ in 0..2 {
                    tops.push(body.multiline_display()[0].clone());
                    let Some(child) = body.arc_children().into_iter().next() else {
                        break;
                    };
                    body = child;
                }
                found.push(tops);
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        found
    }

    fn concat(left: LogicalPlanBuilder, right: LogicalPlanBuilder) -> Arc<LogicalPlan> {
        left.concat(&right).unwrap().build()
    }

    use common_treenode::DynTreeNode;

    #[test]
    fn the_part_keeps_the_columns_any_reader_reads() -> DaftResult<()> {
        let plan = concat(
            part().select(vec![col("k"), col("s")])?,
            part().select(vec![col("k"), col("s")])?,
        );
        let plan = pushed(plan)?;
        assert_eq!(
            bodies(&plan),
            vec![
                vec![
                    "Project: col(k), col(s)".to_string(),
                    "Aggregation: sum(col(v)) as s, max(col(w)) as m".to_string()
                ];
                2
            ]
        );
        Ok(())
    }

    #[test]
    fn the_part_keeps_the_rows_any_reader_keeps() -> DaftResult<()> {
        let both = col("k").gt(lit(0));
        let plan = concat(
            part().filter(both.clone().and(col("s").gt(lit(1))))?,
            part().filter(both.and(col("m").lt(lit(5))))?,
        );
        let plan = pushed(plan)?;
        assert_eq!(
            bodies(&plan)[0][0],
            "Filter: [col(k) > lit(0)] & [[col(s) > lit(1)] | [col(m) < lit(5)]]"
        );
        assert_eq!(bodies(&plan)[0], bodies(&plan)[1]);
        Ok(())
    }

    #[test]
    fn a_reader_that_keeps_every_row_keeps_them_all_in_the_part() -> DaftResult<()> {
        let plan = concat(part().filter(col("s").gt(lit(1)))?, part());
        let plan = pushed(plan)?;
        assert_eq!(
            bodies(&plan)[0][0],
            "Aggregation: sum(col(v)) as s, max(col(w)) as m"
        );
        Ok(())
    }

    #[test]
    fn readers_that_each_read_less_alone_are_not_shared() -> DaftResult<()> {
        // Each filter is on the grouping column, so it reaches below the aggregation alone.
        let plan = concat(
            part().filter(col("k").gt(lit(10)))?,
            part().filter(col("k").lt(lit(5)))?,
        );
        assert!(bodies(&pushed(plan)?).is_empty());
        Ok(())
    }

    #[test]
    fn a_part_read_once_is_unwrapped() -> DaftResult<()> {
        assert!(bodies(&pushed(part().build())?).is_empty());
        Ok(())
    }

    #[test]
    fn every_reader_taking_a_limit_limits_the_part() -> DaftResult<()> {
        let plan = concat(part().limit(3, false)?, part().limit(7, false)?);
        let plan = optimize_with_rules(
            plan,
            vec![RuleBatch::new(
                vec![Box::new(PushLimitsIntoSharedParts::new())],
                RuleExecutionStrategy::FixedPoint(Some(3)),
            )],
        )?;
        assert_eq!(bodies(&plan)[0][0], "Limit: 7");
        assert_eq!(bodies(&plan)[1][0], "Limit: 7");
        Ok(())
    }

    #[test]
    fn a_reader_taking_every_row_leaves_the_part_unlimited() -> DaftResult<()> {
        let plan = concat(part().limit(3, false)?, part());
        let plan = optimize_with_rules(
            plan,
            vec![RuleBatch::new(
                vec![Box::new(PushLimitsIntoSharedParts::new())],
                RuleExecutionStrategy::Once,
            )],
        )?;
        assert_eq!(
            bodies(&plan)[0][0],
            "Aggregation: sum(col(v)) as s, max(col(w)) as m"
        );
        Ok(())
    }
}
