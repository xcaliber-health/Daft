//! Finds a part of a plan written more than once and marks each occurrence to be
//! computed once for all of them.
//!
//! A caller may build the same part twice from one source, for example once to
//! count a set of records and once to join to it. Each occurrence would compute
//! it again. This rule finds the largest repeated part that does real work and
//! wraps every occurrence in a [`Shared`] node with one id; the local executor
//! then runs the part once and hands its rows to every occurrence.
//!
//! It runs before any pushdown, while the occurrences are still alike: pushing a
//! reader's filter into its own occurrence would make them differ. From then on
//! a shared node is a barrier, and what every reader needs enters the part
//! through `PushDownIntoSharedParts`.
//!
//! A part is shared only when:
//! - it does real work: a join, an aggregation (distinct and pivot included), a
//!   window, or a sort; scanning, filtering and projecting are cheaper to repeat
//!   than to hold;
//! - every expression in it gives the same values each time it is evaluated,
//!   so a Python function, a model call or a random value is never shared;
//! - it holds no operator whose output is random or has a side effect (sampling,
//!   row ids, a random shuffle, a write, checkpoint staging);
//! - no reader would hold its whole output while waiting on another reader,
//!   unless that output is reduced (an aggregation, a top-N, a limit).
//!
//! The plan is left alone if it shards a scan: a shard can't pass the barrier,
//! and passing it would restrict every reader.

use std::{
    collections::{HashMap, HashSet},
    hash::{DefaultHasher, Hash, Hasher},
    sync::Arc,
};

use common_error::DaftResult;
use common_treenode::{Transformed, TreeNode, TreeNodeRecursion};
use daft_dsl::{Expr, ExprRef};

use super::OptimizerRule;
use crate::{
    LogicalPlan, LogicalPlanRef,
    ops::{
        AsofJoin, Concat, Distinct, Explode, Filter, Join, Pivot, Project, Shared, Sort, TopN,
        Unpivot, Window,
    },
    partitioning::RepartitionSpec,
    source_info::SourceInfo,
};

/// Wraps each occurrence of a part written more than once in a [`Shared`] node.
#[derive(Default, Debug)]
pub struct ShareRepeatedParts {}

impl ShareRepeatedParts {
    /// A rule that shares the largest repeated parts of a plan.
    pub fn new() -> Self {
        Self {}
    }
}

impl OptimizerRule for ShareRepeatedParts {
    fn try_optimize(&self, plan: Arc<LogicalPlan>) -> DaftResult<Transformed<Arc<LogicalPlan>>> {
        if any_node(&plan, &|node| matches!(node, LogicalPlan::Shard(_)))
            || !reads_a_source_twice(&plan)
        {
            return Ok(Transformed::no(plan));
        }
        let mut walk = Walk::default();
        walk.visit(&plan, Reading::Draining);
        let mut parts = Vec::new();
        walk.choose(&plan, &mut parts);
        if parts.is_empty() {
            return Ok(Transformed::no(plan));
        }
        let first_id = next_shared_id(&plan)?;
        Ok(match walk.wrap(&plan, &parts, first_id) {
            Some(wrapped) => Transformed::yes(wrapped),
            None => Transformed::no(plan),
        })
    }
}

// Walks here follow `LogicalPlan::children`, whose references are the plan's own nodes: a node
// reached through two parents is one address. (`arc_children` hands out fresh copies.)

/// Whether any node of `plan` satisfies `test`.
#[recursive::recursive]
fn any_node(plan: &LogicalPlan, test: &impl Fn(&LogicalPlan) -> bool) -> bool {
    test(plan)
        || plan
            .children()
            .into_iter()
            .any(|child| any_node(child, test))
}

/// Whether some source is read in two places: two occurrences of a part read equal sources,
/// so without such a pair there is nothing to share, and the plan's parts need not be
/// compared. Source information compares shallowly: a scan by its operator or task list, data
/// held in memory by its cache key.
fn reads_a_source_twice(plan: &LogicalPlan) -> bool {
    #[recursive::recursive]
    fn walk<'a>(node: &'a LogicalPlan, seen: &mut HashSet<&'a SourceInfo>) -> bool {
        if let LogicalPlan::Source(source) = node {
            return !seen.insert(source.source_info.as_ref());
        }
        node.children().into_iter().any(|child| walk(child, seen))
    }
    walk(plan, &mut HashSet::new())
}

/// How the reader above a node takes its rows.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reading {
    /// The rows flow, unheld, into an operator that takes all of them.
    Draining,
    /// The rows reach a join's input, or a concatenation's second input, before
    /// any such operator, so the reader may wait on another before taking them.
    Waiting,
}

/// What one walk of the plan learns about each part in it.
///
/// Parts are matched by fingerprint first: a hash of each node's own content and its
/// children's fingerprints, computed once per occurrence in one pass. Only parts whose
/// fingerprints agree are compared in full, on a stack sized for the plan, since a full
/// comparison recurses once per level of the part.
#[derive(Default)]
struct Walk<'a> {
    /// Each node's fingerprint, by its address.
    fingerprints: HashMap<*const LogicalPlan, u64>,
    /// Every occurrence of each fingerprint, once per path from the root.
    occurrences: HashMap<u64, Vec<&'a LogicalPlan>>,
    /// The fingerprints some occurrence of which a reader may wait on.
    waited_on: HashMap<u64, bool>,
    /// For each fingerprint compared so far, its occurrences grouped by equality: one
    /// occurrence standing for each group, and how many the group holds.
    classes: HashMap<u64, Vec<(&'a LogicalPlan, usize)>>,
}

impl<'a> Walk<'a> {
    /// Records every occurrence below `node` and returns `node`'s fingerprint: its name and
    /// display, and its children's fingerprints.
    #[recursive::recursive]
    fn visit(&mut self, node: &'a LogicalPlan, reading: Reading) -> u64 {
        let mut hasher = DefaultHasher::new();
        node.name().hash(&mut hasher);
        node.multiline_display().hash(&mut hasher);
        // A part already shared is a reader of its own; parts are not nested.
        if !matches!(node, LogicalPlan::Shared(_)) {
            let children: Vec<(&'a LogicalPlan, Reading)> = match node {
                LogicalPlan::Join(Join { left, right, .. })
                | LogicalPlan::AsofJoin(AsofJoin { left, right, .. }) => {
                    // The engine picks the build side as it runs, so either side may wait.
                    vec![
                        (left.as_ref(), Reading::Waiting),
                        (right.as_ref(), Reading::Waiting),
                    ]
                }
                LogicalPlan::Concat(Concat { input, other, .. }) => {
                    vec![
                        (input.as_ref(), reading),
                        (other.as_ref(), Reading::Waiting),
                    ]
                }
                _ => {
                    let below = if takes_all_rows(node) {
                        Reading::Draining
                    } else {
                        reading
                    };
                    node.children()
                        .into_iter()
                        .map(|child| (child, below))
                        .collect()
                }
            };
            for (child, below) in children {
                self.visit(child, below).hash(&mut hasher);
            }
        }
        let fingerprint = hasher.finish();
        self.fingerprints
            .insert(std::ptr::from_ref(node), fingerprint);
        if !matches!(node, LogicalPlan::Shared(_)) {
            self.occurrences.entry(fingerprint).or_default().push(node);
            *self.waited_on.entry(fingerprint).or_default() |= reading == Reading::Waiting;
        }
        fingerprint
    }

    /// Collects, top-down, the largest parts to share; a part refused is searched
    /// for smaller ones.
    #[recursive::recursive]
    fn choose(&mut self, node: &'a LogicalPlan, parts: &mut Vec<(u64, &'a LogicalPlan)>) {
        if matches!(node, LogicalPlan::Shared(_)) || self.chosen(node, parts).is_some() {
            return;
        }
        let Some(fingerprint) = self.fingerprints.get(&std::ptr::from_ref(node)).copied() else {
            return;
        };
        let repeated = self
            .occurrences
            .get(&fingerprint)
            .is_some_and(|occurrences| occurrences.len() > 1);
        // Cheap tests first: most repeated nodes (a scan, a projection) do no real work.
        if repeated && self.shareable(node, fingerprint) && self.alike(fingerprint, node) > 1 {
            parts.push((fingerprint, node));
            return;
        }
        for child in node.children() {
            self.choose(child, parts);
        }
    }

    /// How many occurrences of `node`'s fingerprint equal it, grouping them once per
    /// fingerprint so each occurrence is compared with one per group, not with every other.
    fn alike(&mut self, fingerprint: u64, node: &LogicalPlan) -> usize {
        if !self.classes.contains_key(&fingerprint) {
            let mut classes: Vec<(&'a LogicalPlan, usize)> = Vec::new();
            for occurrence in self.occurrences.get(&fingerprint).into_iter().flatten() {
                match classes
                    .iter_mut()
                    .find(|(standing, _)| self.same(standing, occurrence))
                {
                    Some((_, count)) => *count += 1,
                    None => classes.push((occurrence, 1)),
                }
            }
            self.classes.insert(fingerprint, classes);
        }
        self.classes
            .get(&fingerprint)
            .and_then(|classes| {
                classes
                    .iter()
                    .find(|(standing, _)| self.same(standing, node))
            })
            .map_or(0, |(_, count)| *count)
    }

    /// Which of `parts` the node is an occurrence of.
    fn chosen(&self, node: &LogicalPlan, parts: &[(u64, &LogicalPlan)]) -> Option<usize> {
        let fingerprint = self.fingerprints.get(&std::ptr::from_ref(node))?;
        parts
            .iter()
            .position(|(other, part)| other == fingerprint && self.same(part, node))
    }

    /// Whether two parts are the same: one node, or equal in full. The full comparison
    /// recurses once per level of the parts, so it runs on a stack grown for the plan.
    fn same(&self, left: &LogicalPlan, right: &LogicalPlan) -> bool {
        const STACK_PER_NODE: usize = 4 * 1024;
        std::ptr::eq(left, right) || {
            let stack = 1024 * 1024 + self.fingerprints.len() * STACK_PER_NODE;
            stacker::maybe_grow(stack, stack, || left == right)
        }
    }

    fn shareable(&self, part: &LogicalPlan, fingerprint: u64) -> bool {
        let waited_on = self.waited_on.get(&fingerprint).copied().unwrap_or(false);
        does_real_work(part) && always_alike(part) && (!waited_on || reduces_rows(part))
    }

    /// The plan with every occurrence of each part wrapped, or `None` if none occurs. Each
    /// occurrence gets its own node: two readers never hold one, which rules that key
    /// operators by pointer would take for one.
    #[recursive::recursive]
    fn wrap(
        &self,
        node: &LogicalPlan,
        parts: &[(u64, &LogicalPlan)],
        first_id: usize,
    ) -> Option<LogicalPlanRef> {
        if matches!(node, LogicalPlan::Shared(_)) {
            return None;
        }
        if let Some(index) = self.chosen(node, parts) {
            return Some(Shared::new(Arc::new(node.clone()), first_id + index).into());
        }
        let children = node.children();
        let wrapped: Vec<Option<LogicalPlanRef>> = children
            .iter()
            .map(|child| self.wrap(child, parts, first_id))
            .collect();
        if wrapped.iter().all(Option::is_none) {
            return None;
        }
        let children: Vec<LogicalPlanRef> = wrapped
            .into_iter()
            .zip(children)
            .map(|(wrapped, child)| wrapped.unwrap_or_else(|| Arc::new(child.clone())))
            .collect();
        Some(Arc::new(node.with_new_children(&children)))
    }
}

/// Whether the operator takes every row of its input before or as it emits, so
/// rows above it are never held waiting for another reader: an aggregation,
/// window, sort or top-N, or a limit, which stops taking.
fn takes_all_rows(node: &LogicalPlan) -> bool {
    matches!(
        node,
        LogicalPlan::Aggregate(_)
            | LogicalPlan::Distinct(_)
            | LogicalPlan::Pivot(_)
            | LogicalPlan::Window(_)
            | LogicalPlan::Sort(_)
            | LogicalPlan::TopN(_)
            | LogicalPlan::Limit(_)
    )
}

/// Whether the part's output is a reduction of its input, cheap to hold.
fn reduces_rows(part: &LogicalPlan) -> bool {
    matches!(
        part,
        LogicalPlan::Aggregate(_)
            | LogicalPlan::Distinct(_)
            | LogicalPlan::Pivot(_)
            | LogicalPlan::TopN(_)
            | LogicalPlan::Limit(_)
    )
}

/// Whether some operator in the part costs more to repeat than to hold its output.
fn does_real_work(part: &LogicalPlan) -> bool {
    any_node(part, &|node| {
        matches!(
            node,
            LogicalPlan::Join(_)
                | LogicalPlan::AsofJoin(_)
                | LogicalPlan::Aggregate(_)
                | LogicalPlan::Distinct(_)
                | LogicalPlan::Pivot(_)
                | LogicalPlan::Window(_)
                | LogicalPlan::Sort(_)
                | LogicalPlan::TopN(_)
        )
    })
}

/// Whether computing the part twice is known to give the same rows both times,
/// with no side effect: every operator qualifies and every expression is
/// deterministic.
fn always_alike(part: &LogicalPlan) -> bool {
    !any_node(part, &|node| !node_always_alike(node))
}

fn deterministic<'a>(exprs: impl IntoIterator<Item = &'a ExprRef>) -> bool {
    exprs.into_iter().all(Expr::is_deterministic)
}

fn node_always_alike(node: &LogicalPlan) -> bool {
    match node {
        LogicalPlan::Source(source) => {
            source.checkpoint.is_none()
                && match source.source_info.as_ref() {
                    SourceInfo::Physical(info) => deterministic(
                        info.pushdowns
                            .filters
                            .iter()
                            .chain(info.pushdowns.partition_filters.iter()),
                    ),
                    _ => true,
                }
        }
        LogicalPlan::Project(Project { projection, .. }) => deterministic(projection),
        LogicalPlan::Filter(Filter { predicate, .. }) => predicate.is_deterministic(),
        LogicalPlan::Explode(Explode { to_explode, .. }) => deterministic(to_explode),
        LogicalPlan::Unpivot(Unpivot { ids, values, .. }) => {
            deterministic(ids) && deterministic(values)
        }
        LogicalPlan::Sort(Sort { sort_by, .. }) | LogicalPlan::TopN(TopN { sort_by, .. }) => {
            deterministic(sort_by)
        }
        LogicalPlan::Distinct(Distinct { columns, .. }) => {
            columns.as_ref().is_none_or(deterministic)
        }
        LogicalPlan::Aggregate(aggregate) => {
            deterministic(&aggregate.aggregations) && deterministic(&aggregate.groupby)
        }
        LogicalPlan::PartialAggregate(aggregate) => {
            deterministic(&aggregate.aggregations) && deterministic(&aggregate.groupby)
        }
        LogicalPlan::Pivot(Pivot {
            group_by,
            pivot_column,
            value_column,
            aggregation,
            ..
        }) => {
            deterministic(group_by)
                && pivot_column.is_deterministic()
                && value_column.is_deterministic()
                && Arc::new(Expr::Agg(aggregation.clone())).is_deterministic()
        }
        LogicalPlan::Join(Join { on, .. }) => on.inner().is_none_or(Expr::is_deterministic),
        LogicalPlan::AsofJoin(AsofJoin {
            left_by,
            right_by,
            left_on,
            right_on,
            ..
        }) => {
            deterministic(left_by)
                && deterministic(right_by)
                && left_on.is_deterministic()
                && right_on.is_deterministic()
        }
        LogicalPlan::Window(Window {
            window_functions,
            window_spec,
            ..
        }) => {
            window_functions
                .iter()
                .all(|function| ExprRef::from(function).is_deterministic())
                && deterministic(&window_spec.partition_by)
                && deterministic(&window_spec.order_by)
        }
        LogicalPlan::Repartition(repartition) => {
            !matches!(repartition.repartition_spec, RepartitionSpec::Random(_))
        }
        LogicalPlan::Limit(_)
        | LogicalPlan::Offset(_)
        | LogicalPlan::Concat(_)
        | LogicalPlan::IntoBatches(_)
        | LogicalPlan::IntoPartitions(_)
        | LogicalPlan::SubqueryAlias(_) => true,
        // Their output is random, or computing them has a side effect, or they are
        // already shared or rewritten away by now.
        LogicalPlan::UDFProject(_)
        | LogicalPlan::VLLMProject(_)
        | LogicalPlan::Sample(_)
        | LogicalPlan::Shuffle(_)
        | LogicalPlan::MonotonicallyIncreasingId(_)
        | LogicalPlan::Sink(_)
        | LogicalPlan::StageCheckpointKeys(_)
        | LogicalPlan::Shard(_)
        | LogicalPlan::Intersect(_)
        | LogicalPlan::Union(_)
        | LogicalPlan::Shared(_) => false,
    }
}

/// One more than the largest id of a part already shared, so a plan optimized
/// again keeps its ids apart.
fn next_shared_id(plan: &LogicalPlanRef) -> DaftResult<usize> {
    let mut next = 0;
    plan.apply(|node| {
        if let LogicalPlan::Shared(shared) = node.as_ref() {
            next = next.max(shared.id + 1);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(next)
}

/// Runs a rule only on a plan that holds a shared part, so the batches that serve shared
/// parts leave every other plan exactly as the rest of the optimizer does.
#[derive(Debug)]
pub struct WithSharedParts<R>(pub R);

impl<R: OptimizerRule> OptimizerRule for WithSharedParts<R> {
    fn try_optimize(&self, plan: Arc<LogicalPlan>) -> DaftResult<Transformed<Arc<LogicalPlan>>> {
        if plan.exists(|node| matches!(node.as_ref(), LogicalPlan::Shared(_))) {
            self.0.try_optimize(plan)
        } else {
            Ok(Transformed::no(plan))
        }
    }
}

/// Replaces each occurrence of the parts `ids` names by its own body.
pub(crate) fn unwrap_shared(
    plan: LogicalPlanRef,
    ids: &[usize],
) -> DaftResult<Transformed<LogicalPlanRef>> {
    if ids.is_empty() {
        return Ok(Transformed::no(plan));
    }
    plan.transform_up(|node| match node.as_ref() {
        LogicalPlan::Shared(shared) if ids.contains(&shared.id) => {
            Ok(Transformed::yes(shared.input.clone()))
        }
        _ => Ok(Transformed::no(node)),
    })
}

/// Each shared id in the plan with the number of its occurrences.
pub(crate) fn shared_occurrences(plan: &LogicalPlanRef) -> DaftResult<HashMap<usize, usize>> {
    let mut counts: HashMap<usize, usize> = HashMap::new();
    plan.apply(|node| {
        if let LogicalPlan::Shared(shared) = node.as_ref() {
            *counts.entry(shared.id).or_default() += 1;
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use common_error::DaftResult;
    use common_treenode::{TreeNode, TreeNodeRecursion};
    use daft_core::{join::JoinType, prelude::*};
    use daft_dsl::{ExprRef, lit, unresolved_col};
    use rstest::rstest;

    use super::ShareRepeatedParts;
    use crate::{
        LogicalPlan, LogicalPlanBuilder,
        optimization::{
            optimizer::{RuleBatch, RuleExecutionStrategy},
            test::optimize_with_rules,
        },
        test::{dummy_scan_node, dummy_scan_operator},
    };

    fn scan() -> LogicalPlanBuilder {
        dummy_scan_node(dummy_scan_operator(vec![
            Field::new("k", DataType::Int64),
            Field::new("v", DataType::Int64),
        ]))
    }

    fn col(name: &str) -> ExprRef {
        unresolved_col(name)
    }

    fn sum_by_k(rows: &LogicalPlanBuilder) -> LogicalPlanBuilder {
        rows.aggregate(vec![col("v").sum().alias("s")], vec![col("k")])
            .unwrap()
    }

    fn join_on_k(left: &LogicalPlanBuilder, right: &LogicalPlanBuilder) -> LogicalPlanBuilder {
        left.join(
            right.clone(),
            None,
            vec!["k".to_string()],
            JoinType::Inner,
            None,
            crate::ops::join::JoinOptions::default().suffix("_r"),
        )
        .unwrap()
    }

    fn shared(plan: Arc<LogicalPlan>) -> DaftResult<Arc<LogicalPlan>> {
        optimize_with_rules(
            plan,
            vec![RuleBatch::new(
                vec![Box::new(ShareRepeatedParts::new())],
                RuleExecutionStrategy::Once,
            )],
        )
    }

    /// The ids of the shared nodes in `plan`, one per occurrence, with each occurrence's top.
    fn occurrences(plan: &Arc<LogicalPlan>) -> Vec<(usize, &'static str)> {
        let mut found = vec![];
        plan.apply(|node| {
            if let LogicalPlan::Shared(shared) = node.as_ref() {
                found.push((shared.id, shared.input.name()));
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        found
    }

    #[rstest]
    #[case::two_builds_from_one_frame(false)]
    #[case::one_object_twice(true)]
    fn a_repeated_aggregation_is_shared(#[case] same_object: bool) -> DaftResult<()> {
        let rows = scan();
        let first = sum_by_k(&rows);
        let second = if same_object {
            first.clone()
        } else {
            sum_by_k(&rows)
        };
        let plan = join_on_k(&first.filter(col("k").gt(lit(0)))?, &second).build();
        assert_eq!(
            occurrences(&shared(plan)?),
            vec![(0, "Aggregate"), (0, "Aggregate")]
        );
        Ok(())
    }

    #[test]
    fn each_occurrence_is_its_own_node() -> DaftResult<()> {
        let part = sum_by_k(&scan());
        let plan = shared(join_on_k(&part, &part).build())?;
        let mut nodes = vec![];
        plan.apply(|node| {
            if matches!(node.as_ref(), LogicalPlan::Shared(_)) {
                nodes.push(node.clone());
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        assert_eq!(nodes.len(), 2);
        assert!(!Arc::ptr_eq(&nodes[0], &nodes[1]));
        Ok(())
    }

    #[rstest]
    #[case::different_sources(sum_by_k(&scan()), sum_by_k(&scan()))]
    #[case::different_filters(
        sum_by_k(&scan().filter(col("v").gt(lit(1))).unwrap()),
        sum_by_k(&scan().filter(col("v").gt(lit(2))).unwrap())
    )]
    fn parts_that_differ_are_not_shared(
        #[case] left: LogicalPlanBuilder,
        #[case] right: LogicalPlanBuilder,
    ) -> DaftResult<()> {
        // Different sources are different scan objects, which compare unequal.
        assert!(occurrences(&shared(join_on_k(&left, &right).build())?).is_empty());
        Ok(())
    }

    #[rstest]
    #[case::plain_scan(|rows: &LogicalPlanBuilder| rows.clone())]
    #[case::filtered_scan(|rows: &LogicalPlanBuilder| rows.filter(col("v").gt(lit(1))).unwrap())]
    #[case::projection(|rows: &LogicalPlanBuilder| rows.select(vec![col("k"), col("v").alias("w")]).unwrap())]
    fn parts_without_real_work_are_not_shared(
        #[case] part: fn(&LogicalPlanBuilder) -> LogicalPlanBuilder,
    ) -> DaftResult<()> {
        let rows = scan();
        let plan = join_on_k(&part(&rows), &part(&rows)).build();
        assert!(occurrences(&shared(plan)?).is_empty());
        Ok(())
    }

    #[test]
    fn a_sampled_part_is_not_shared() -> DaftResult<()> {
        let rows = scan();
        let part = || sum_by_k(&rows.sample(Some(0.5), None, false, Some(7)).unwrap());
        let plan = join_on_k(&part(), &part()).build();
        assert!(occurrences(&shared(plan)?).is_empty());
        Ok(())
    }

    #[test]
    fn a_plan_that_shards_is_left_alone() -> DaftResult<()> {
        let part = sum_by_k(&scan().shard("file".to_string(), 2, 0)?);
        let plan = part.concat(&part)?.build();
        assert!(occurrences(&shared(plan)?).is_empty());
        Ok(())
    }

    #[test]
    fn a_part_a_waiting_reader_would_hold_in_full_is_not_shared() -> DaftResult<()> {
        // A sort keeps every row, and the join's readers may wait on each other.
        let rows = scan();
        let part = || rows.sort(vec![col("v")], vec![false], vec![false]).unwrap();
        let plan = join_on_k(&part(), &part()).build();
        assert!(occurrences(&shared(plan)?).is_empty());
        Ok(())
    }

    #[test]
    fn a_part_drained_by_blocking_readers_is_shared() -> DaftResult<()> {
        // Both readers aggregate the joined part before the join above them.
        let rows = scan();
        let part = || {
            join_on_k(
                &rows,
                &rows.select(vec![col("k"), col("v").alias("w")]).unwrap(),
            )
        };
        let counts = part().aggregate(
            vec![col("v").count(CountMode::Valid).alias("n")],
            vec![col("k")],
        )?;
        let maxima = part().aggregate(vec![col("w").max().alias("m")], vec![col("k")])?;
        let plan = join_on_k(&counts, &maxima).build();
        assert_eq!(occurrences(&shared(plan)?), vec![(0, "Join"), (0, "Join")]);
        Ok(())
    }

    #[test]
    fn the_largest_repeat_is_shared() -> DaftResult<()> {
        let rows = scan();
        let part = || sum_by_k(&rows).filter(col("s").gt(lit(10))).unwrap();
        // Both readers aggregate the part, so neither holds it waiting on the other.
        let counts = part().aggregate(
            vec![col("s").count(CountMode::Valid).alias("n")],
            vec![col("k")],
        )?;
        let maxima = part().aggregate(vec![col("s").max().alias("m")], vec![col("k")])?;
        let plan = join_on_k(&counts, &maxima).build();
        assert_eq!(
            occurrences(&shared(plan)?),
            vec![(0, "Filter"), (0, "Filter")]
        );
        Ok(())
    }

    #[test]
    fn sharing_again_changes_nothing() -> DaftResult<()> {
        let part = sum_by_k(&scan());
        let once = shared(join_on_k(&part, &part).build())?;
        let twice = shared(once.clone())?;
        assert_eq!(once, twice);
        Ok(())
    }
}
