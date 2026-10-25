//! Derives a restriction on one side of an equi-join from a restriction on the other side's key.
//!
//! When every row one side of `l = r` keeps satisfies `P(l)`, a row of the other side can only
//! match if `P(r)` holds, so the other side may be filtered by `P(r)` before the join wherever the
//! join type lets its unmatched rows go ([`JoinType::unmatched_rows_droppable`]). The derived
//! filter is placed on top of that side's input, and only [`PushDownFilter`] moves it from there:
//! it crosses a window only on partition keys and a grouping only on grouping keys, as any filter
//! does.
//!
//! [`PushDownFilter`]: super::PushDownFilter

use std::{collections::HashMap, sync::Arc};

use common_error::DaftResult;
use common_treenode::{DynTreeNode, Transformed, TreeNode, TreeNodeRecursion};
use daft_algebra::boolean::{combine_conjunction, split_conjunction, split_disjunction};
use daft_core::{join::JoinType, prelude::Operator};
use daft_dsl::{
    Column, Expr, ExprRef, ResolvedColumn, optimization::replace_columns_with_expressions,
    resolved_col,
};
use daft_scan::ScanState;
use daft_schema::{dtype::DataType, schema::Schema};

use super::{OptimizerRule, push_down_filter::passed_through_columns};
use crate::{
    LogicalPlan,
    ops::{Filter, Join, Source},
    source_info::SourceInfo,
};

/// Optimization rule that filters each side of an equi-join by the restrictions the other side's
/// key is known to satisfy.
///
/// A key pair takes part when both keys are plain columns compared by plain (not null-safe)
/// equality and share one dtype whose values compare the same in a join as in a filter (see
/// [`key_dtype_allowed`]). A restriction takes part when it compares that one column with literals
/// ([`restricted_column`]). The rule reads restrictions from filters and from the pushed filters
/// of scans that apply them exactly, and adds derived ones only to sides whose scans all apply
/// pushed filters exactly ([`ScanOperator::applies_filters_exactly`]).
///
/// [`ScanOperator::applies_filters_exactly`]: daft_scan::ScanOperator::applies_filters_exactly
#[derive(Default, Debug)]
pub struct InferJoinKeyFilters {}

impl InferJoinKeyFilters {
    /// Creates the rule.
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl OptimizerRule for InferJoinKeyFilters {
    fn try_optimize(&self, plan: Arc<LogicalPlan>) -> DaftResult<Transformed<Arc<LogicalPlan>>> {
        Ok(infer(plan)?.0)
    }
}

/// The restrictions known to hold on every row a plan outputs, by the one column each reads.
#[derive(Default)]
struct Restrictions(HashMap<String, Vec<ExprRef>>);

impl Restrictions {
    /// The restrictions known to hold on `column`.
    fn on(&self, column: &str) -> &[ExprRef] {
        self.0.get(column).map_or(&[], Vec::as_slice)
    }

    /// Whether `restriction` on `column` is already known, regardless of how it is written.
    fn holds(&self, column: &str, restriction: &ExprRef) -> bool {
        let wanted = canonical(restriction);
        self.on(column)
            .iter()
            .any(|known| canonical(known) == wanted)
    }

    /// Records each conjunct of `predicate` that is a restriction on a column of `schema`.
    fn add_conjuncts(&mut self, predicate: &ExprRef, schema: &Schema) {
        for conjunct in split_conjunction(predicate) {
            if let Some(column) = restricted_column(&conjunct)
                && schema.get_field(column).is_ok()
                && !self.holds(column, &conjunct)
            {
                self.0
                    .entry(column.to_string())
                    .or_default()
                    .push(conjunct.clone());
            }
        }
    }

    /// The restrictions on the columns of `schema`, a projection of `self`'s columns that keeps
    /// their names, dropping columns `schema` lacks or names more than once.
    fn restricted_to(&self, schema: &Schema) -> Self {
        Self(
            self.0
                .iter()
                .filter(|(column, _)| schema.get_field(column).is_ok())
                .map(|(column, restrictions)| (column.clone(), restrictions.clone()))
                .collect(),
        )
    }

    /// The restrictions on the columns `exprs` pass through unchanged, under their output names.
    fn passed_through(&self, exprs: &[ExprRef]) -> Self {
        let mut out = Self::default();
        for (output, input) in passed_through_columns(exprs) {
            let input = input.name();
            let renamed = HashMap::from([(input.to_string(), resolved_col(output.as_str()))]);
            let restrictions = self
                .on(input)
                .iter()
                .map(|r| replace_columns_with_expressions(r.clone(), &renamed))
                .collect::<Vec<_>>();
            if !restrictions.is_empty() {
                out.0.insert(output, restrictions);
            }
        }
        out
    }
}

/// What a walk learns about a plan: the restrictions on its output, and whether every scan in it
/// applies pushed filters exactly, so that a filter pushed anywhere into it keeps its meaning.
struct Facts {
    restrictions: Arc<Restrictions>,
    exact: bool,
}

/// Filters each join in `plan` by its keys' restrictions, bottom up, and returns the
/// restrictions on `plan`'s output.
#[recursive::recursive]
fn infer(plan: Arc<LogicalPlan>) -> DaftResult<(Transformed<Arc<LogicalPlan>>, Facts)> {
    let children = plan.arc_children();
    let mut child_facts = Vec::with_capacity(children.len());
    let mut new_children = Vec::with_capacity(children.len());
    let mut transformed = false;
    for child in children {
        let (child, facts) = infer(child)?;
        transformed |= child.transformed;
        new_children.push(child.data);
        child_facts.push(facts);
    }
    let plan = if transformed {
        Arc::new(plan.with_new_children(&new_children))
    } else {
        plan
    };
    let exact = child_facts.iter().all(|facts| facts.exact);

    if let LogicalPlan::Join(join) = plan.as_ref()
        && let [left, right] = child_facts.as_slice()
    {
        let (new_plan, facts) = infer_at_join(&plan, join, left, right)?;
        let transformed = transformed || new_plan.transformed;
        return Ok((
            Transformed::new(new_plan.data, transformed, TreeNodeRecursion::Continue),
            facts,
        ));
    }

    let child_restrictions = || {
        child_facts.first().map_or_else(
            || Arc::new(Restrictions::default()),
            |facts| facts.restrictions.clone(),
        )
    };
    let facts = match plan.as_ref() {
        LogicalPlan::Source(source) => source_facts(source),
        LogicalPlan::Filter(filter) => {
            let child = child_restrictions();
            let schema = filter.input.schema();
            let restrictions = if split_conjunction(&filter.predicate)
                .iter()
                .any(|conjunct| restricted_column(conjunct).is_some())
            {
                let mut restrictions = child.restricted_to(&schema);
                restrictions.add_conjuncts(&filter.predicate, &schema);
                Arc::new(restrictions)
            } else {
                child
            };
            Facts {
                restrictions,
                exact,
            }
        }
        LogicalPlan::Project(project) => Facts {
            restrictions: Arc::new(child_restrictions().passed_through(&project.projection)),
            exact,
        },
        LogicalPlan::Aggregate(aggregate) if !aggregate.groupby.is_empty() => Facts {
            restrictions: Arc::new(child_restrictions().passed_through(&aggregate.groupby)),
            exact,
        },
        // A window keeps every row and only adds columns; one that shadows a restricted column
        // leaves its name ambiguous, and `restricted_to` drops it.
        LogicalPlan::Window(_) => Facts {
            restrictions: Arc::new(child_restrictions().restricted_to(&plan.schema())),
            exact,
        },
        // Each keeps a subset of its input's rows, or all of them, with their values and schema
        // unchanged.
        LogicalPlan::Distinct(_)
        | LogicalPlan::Sort(_)
        | LogicalPlan::Limit(_)
        | LogicalPlan::Offset(_)
        | LogicalPlan::TopN(_)
        | LogicalPlan::Sample(_)
        | LogicalPlan::Repartition(_)
        | LogicalPlan::Shuffle(_)
        | LogicalPlan::IntoBatches(_)
        | LogicalPlan::IntoPartitions(_)
        | LogicalPlan::Shard(_) => Facts {
            restrictions: child_restrictions(),
            exact,
        },
        _ => Facts {
            restrictions: Arc::new(Restrictions::default()),
            exact,
        },
    };
    Ok((
        Transformed::new(plan, transformed, TreeNodeRecursion::Continue),
        facts,
    ))
}

/// The restrictions a source's pushed filters guarantee, if its scan applies them exactly.
///
/// A source the optimizer can't push a filter into keeps every filter above it, so it counts as
/// exact without contributing restrictions of its own.
fn source_facts(source: &Source) -> Facts {
    let none = |exact| Facts {
        restrictions: Arc::new(Restrictions::default()),
        exact,
    };
    match source.source_info.as_ref() {
        SourceInfo::InMemory(_) | SourceInfo::GlobScan(_) => none(true),
        SourceInfo::PlaceHolder(_) => none(false),
        SourceInfo::Physical(info) => {
            let ScanState::Operator(scan_op) = &info.scan_state else {
                return none(false);
            };
            if !scan_op.0.applies_filters_exactly() {
                return none(false);
            }
            if info.pushdowns.aggregation.is_some() {
                return none(true);
            }
            let mut restrictions = Restrictions::default();
            for predicate in [&info.pushdowns.filters, &info.pushdowns.partition_filters]
                .into_iter()
                .flatten()
            {
                restrictions.add_conjuncts(predicate, &source.output_schema);
            }
            Facts {
                restrictions: Arc::new(restrictions),
                exact: true,
            }
        }
    }
}

/// Filters `join`'s sides by the restrictions their keys inherit from each other, and returns the
/// restrictions on its output.
fn infer_at_join(
    plan: &Arc<LogicalPlan>,
    join: &Join,
    left: &Facts,
    right: &Facts,
) -> DaftResult<(Transformed<Arc<LogicalPlan>>, Facts)> {
    let (into_left, into_right) = join.join_type.unmatched_rows_droppable();
    let (_, left_keys, right_keys, null_equals_nulls) = join.on.split_eq_preds();
    let left_schema = join.left.schema();
    let right_schema = join.right.schema();

    let mut left_new: Vec<ExprRef> = vec![];
    let mut right_new: Vec<ExprRef> = vec![];
    let mut merged_keys: Vec<String> = vec![];
    for ((left_key, right_key), null_safe) in
        left_keys.iter().zip(&right_keys).zip(&null_equals_nulls)
    {
        if *null_safe {
            continue;
        }
        let (Some(l), Some(r)) = (left_key.input_mapping(), right_key.input_mapping()) else {
            continue;
        };
        let (Ok(l_field), Ok(r_field)) = (left_schema.get_field(&l), right_schema.get_field(&r))
        else {
            continue;
        };
        if l_field.dtype != r_field.dtype || !key_dtype_allowed(&l_field.dtype) {
            continue;
        }
        if l == r {
            merged_keys.push(l.clone());
        }
        if into_right && right.exact {
            right_new.extend(derived(&left.restrictions, &l, &right.restrictions, &r));
        }
        if into_left && left.exact {
            left_new.extend(derived(&right.restrictions, &r, &left.restrictions, &l));
        }
    }

    let mut left_restrictions = left.restrictions.restricted_to(&left_schema);
    let mut right_restrictions = right.restrictions.restricted_to(&right_schema);
    let new_left = filtered(&join.left, &left_new, &mut left_restrictions)?;
    let new_right = filtered(&join.right, &right_new, &mut right_restrictions)?;
    let new_plan = if new_left.is_some() || new_right.is_some() {
        Transformed::yes(Arc::new(plan.with_new_children(&[
            new_left.unwrap_or_else(|| join.left.clone()),
            new_right.unwrap_or_else(|| join.right.clone()),
        ])))
    } else {
        Transformed::no(plan.clone())
    };

    // A name both sides carry is output with the left side's values, except by a right join, and
    // by an inner join those values equal the right side's only for a merged key.
    let output_schema = new_plan.data.schema();
    let restrictions = match join.join_type {
        JoinType::Inner => {
            let mut out = left_restrictions;
            for (column, restrictions) in right_restrictions.0 {
                if !left_schema.has_field(&column) || merged_keys.contains(&column) {
                    for restriction in restrictions {
                        if !out.holds(&column, &restriction) {
                            out.0.entry(column.clone()).or_default().push(restriction);
                        }
                    }
                }
            }
            out.restricted_to(&output_schema)
        }
        JoinType::Left | JoinType::Semi | JoinType::Anti => {
            left_restrictions.restricted_to(&output_schema)
        }
        JoinType::Right => right_restrictions.restricted_to(&output_schema),
        JoinType::Outer => Restrictions::default(),
    };
    Ok((
        new_plan,
        Facts {
            restrictions: Arc::new(restrictions),
            exact: left.exact && right.exact,
        },
    ))
}

/// The restrictions `from`'s key `from_key` carries over to `to`'s key `to_key` that `to` doesn't
/// already hold.
fn derived(from: &Restrictions, from_key: &str, to: &Restrictions, to_key: &str) -> Vec<ExprRef> {
    let renamed = HashMap::from([(from_key.to_string(), resolved_col(to_key))]);
    from.on(from_key)
        .iter()
        .map(|restriction| replace_columns_with_expressions(restriction.clone(), &renamed))
        .filter(|restriction| !to.holds(to_key, restriction))
        .collect()
}

/// `input` filtered by `restrictions`, recording them in `known`, or `None` when there are none.
fn filtered(
    input: &Arc<LogicalPlan>,
    restrictions: &[ExprRef],
    known: &mut Restrictions,
) -> DaftResult<Option<Arc<LogicalPlan>>> {
    let Some(predicate) = combine_conjunction(restrictions.iter().cloned()) else {
        return Ok(None);
    };
    known.add_conjuncts(&predicate, &input.schema());
    Ok(Some(Arc::new(LogicalPlan::Filter(Filter::try_new(
        input.clone(),
        predicate,
    )?))))
}

/// Whether join keys of `dtype` match exactly when a filter's comparison of them agrees.
///
/// Floats are left out: a join matches them bit for bit, while comparisons and `is_in` treat
/// `-0.0` and `0.0`, or NaN, otherwise.
fn key_dtype_allowed(dtype: &DataType) -> bool {
    dtype.is_integer()
        || matches!(
            dtype,
            DataType::Utf8
                | DataType::Binary
                | DataType::FixedSizeBinary(_)
                | DataType::Boolean
                | DataType::Date
                | DataType::Timestamp(..)
                | DataType::Decimal128(..)
                | DataType::Duration(_)
        )
}

/// The one column `expr` restricts, when it compares a plain column with literals only: by `=`,
/// `<`, `<=`, `>`, `>=`, `between` or `is_in`, or an `or` of such comparisons of one column.
fn restricted_column(expr: &ExprRef) -> Option<&str> {
    match expr.as_ref() {
        Expr::BinaryOp {
            op: Operator::Or,
            left,
            right,
        } => {
            let column = restricted_column(left)?;
            (restricted_column(right)? == column).then_some(column)
        }
        Expr::BinaryOp {
            op: Operator::Eq | Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq,
            left,
            right,
        } => match (column_name(left), column_name(right)) {
            (Some(column), None) if is_literal(right) => Some(column),
            (None, Some(column)) if is_literal(left) => Some(column),
            _ => None,
        },
        Expr::Between(value, lower, upper) if is_literal(lower) && is_literal(upper) => {
            column_name(value)
        }
        Expr::IsIn(value, items) if items.iter().all(is_literal) => column_name(value),
        _ => None,
    }
}

/// The name of the plain column `expr` is, if it is one.
fn column_name(expr: &ExprRef) -> Option<&str> {
    match expr.as_ref() {
        Expr::Column(Column::Resolved(ResolvedColumn::Basic(name))) => Some(name),
        _ => None,
    }
}

/// Whether `expr` is a literal, or a cast of one.
fn is_literal(expr: &ExprRef) -> bool {
    match expr.as_ref() {
        Expr::Literal(_) => true,
        Expr::Cast(inner, ..) => is_literal(inner),
        _ => false,
    }
}

/// A form of `restriction` that is the same however it is written: its disjuncts, each with the
/// column on the left and casts of literals removed, in a fixed order.
///
/// Pushing a filter into a scan can write its literals as casts, and swap or regroup its
/// disjuncts. Comparing this form lets the rule recognise a restriction it derived earlier.
fn canonical(restriction: &ExprRef) -> Vec<String> {
    let mut disjuncts = split_disjunction(restriction)
        .iter()
        .map(|disjunct| uncast_literals(disjunct).to_string())
        .collect::<Vec<_>>();
    disjuncts.sort_unstable();
    disjuncts.dedup();
    disjuncts
}

/// `expr` with each cast of a literal replaced by the literal, and a comparison of a literal with
/// a column written with the column first.
fn uncast_literals(expr: &ExprRef) -> ExprRef {
    let expr = expr
        .clone()
        .transform_up(|e| {
            Ok(match e.as_ref() {
                Expr::Cast(inner, ..) if is_literal(inner) => Transformed::yes(inner.clone()),
                _ => Transformed::no(e),
            })
        })
        .map_or_else(|_| expr.clone(), |t| t.data);
    if let Expr::BinaryOp { op, left, right } = expr.as_ref()
        && is_literal(left)
        && column_name(right).is_some()
        && let Some(flipped) = flipped(*op)
    {
        return Arc::new(Expr::BinaryOp {
            op: flipped,
            left: right.clone(),
            right: left.clone(),
        });
    }
    expr
}

/// The comparison that holds with its operands swapped.
const fn flipped(op: Operator) -> Option<Operator> {
    match op {
        Operator::Eq => Some(Operator::Eq),
        Operator::Lt => Some(Operator::Gt),
        Operator::LtEq => Some(Operator::GtEq),
        Operator::Gt => Some(Operator::Lt),
        Operator::GtEq => Some(Operator::LtEq),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use common_error::DaftResult;
    use common_treenode::{TreeNode, TreeNodeRecursion};
    use daft_core::prelude::*;
    use daft_dsl::{ExprRef, binary_op, lit, resolved_col, unresolved_col};
    use daft_scan::{PartitionField, Pushdowns, ScanOperatorRef};
    use rstest::rstest;

    use super::InferJoinKeyFilters;
    use crate::{
        LogicalPlan, LogicalPlanBuilder,
        optimization::{
            optimizer::{RuleBatch, RuleExecutionStrategy},
            rules::PushDownFilter,
            test::{assert_optimized_plan_with_rules_eq, optimize_with_rules},
        },
        source_info::SourceInfo,
        test::{
            dummy_exact_scan_operator, dummy_scan_node, dummy_scan_node_with_pushdowns,
            dummy_scan_operator,
        },
    };

    fn assert_inferred_eq(plan: Arc<LogicalPlan>, expected: Arc<LogicalPlan>) -> DaftResult<()> {
        assert_optimized_plan_with_rules_eq(
            plan,
            expected,
            vec![RuleBatch::new(
                vec![Box::new(InferJoinKeyFilters::new())],
                RuleExecutionStrategy::Once,
            )],
        )
    }

    /// The batch the optimizer runs: inference and filter pushdown to a fixed point.
    fn infer_and_push(plan: Arc<LogicalPlan>) -> DaftResult<Arc<LogicalPlan>> {
        optimize_with_rules(
            plan,
            vec![RuleBatch::new(
                vec![
                    Box::new(InferJoinKeyFilters::new()),
                    Box::new(PushDownFilter::new(false)),
                ],
                RuleExecutionStrategy::FixedPoint(None),
            )],
        )
    }

    fn exact_scan(fields: &[(&str, DataType)]) -> LogicalPlanBuilder {
        dummy_scan_node(exact_op(fields, vec![]))
    }

    fn exact_op(fields: &[(&str, DataType)], keys: Vec<PartitionField>) -> ScanOperatorRef {
        dummy_exact_scan_operator(
            fields
                .iter()
                .map(|(name, dtype)| Field::new(*name, dtype.clone()))
                .collect(),
            keys,
        )
    }

    fn subjects() -> LogicalPlanBuilder {
        exact_scan(&[("id", DataType::Int64), ("name", DataType::Utf8)])
    }

    fn events() -> LogicalPlanBuilder {
        exact_scan(&[("sid", DataType::Int64), ("v", DataType::Int64)])
    }

    fn chosen(column: &str) -> ExprRef {
        unresolved_col(column).is_in(vec![lit(1i64), lit(2i64)])
    }

    fn join(
        left: &LogicalPlanBuilder,
        right: &LogicalPlanBuilder,
        on: ExprRef,
        join_type: JoinType,
    ) -> DaftResult<LogicalPlanBuilder> {
        left.join(
            right.clone(),
            Some(on),
            vec![],
            join_type,
            None,
            Default::default(),
        )
    }

    fn id_eq_sid() -> ExprRef {
        unresolved_col("id").eq(unresolved_col("sid"))
    }

    /// The filters pushed into the scan whose first column is `column`, as written.
    fn pushed_filters(plan: &Arc<LogicalPlan>, column: &str) -> Vec<String> {
        let mut found = vec![];
        plan.apply(|node| {
            if let LogicalPlan::Source(source) = node.as_ref()
                && let SourceInfo::Physical(info) = source.source_info.as_ref()
                && source.output_schema.fields().first().map(|f| &*f.name) == Some(column)
            {
                found.extend(info.pushdowns.filters.iter().map(ToString::to_string));
                found.extend(
                    info.pushdowns
                        .partition_filters
                        .iter()
                        .map(ToString::to_string),
                );
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        found
    }

    /// A restriction on one side's key reaches every side the join type lets drop unmatched rows.
    #[rstest]
    #[case::inner(JoinType::Inner, true, true)]
    #[case::left(JoinType::Left, false, true)]
    #[case::right(JoinType::Right, true, false)]
    #[case::outer(JoinType::Outer, false, false)]
    #[case::semi(JoinType::Semi, true, true)]
    #[case::anti(JoinType::Anti, false, true)]
    fn derives_into_the_sides_the_join_type_allows(
        #[case] join_type: JoinType,
        #[case] into_left: bool,
        #[case] into_right: bool,
    ) -> DaftResult<()> {
        let (subjects, events) = (subjects(), events());

        // From the left side's key into the right side.
        let chosen_subjects = subjects.filter(chosen("id"))?;
        let plan = join(&chosen_subjects, &events, id_eq_sid(), join_type)?.build();
        let right = if into_right {
            events.filter(chosen("sid"))?
        } else {
            events.clone()
        };
        let expected = join(&chosen_subjects, &right, id_eq_sid(), join_type)?.build();
        assert_inferred_eq(plan, expected)?;

        // From the right side's key into the left side.
        let chosen_events = events.filter(chosen("sid"))?;
        let plan = join(&subjects, &chosen_events, id_eq_sid(), join_type)?.build();
        let left = if into_left {
            subjects.filter(chosen("id"))?
        } else {
            subjects
        };
        let expected = join(&left, &chosen_events, id_eq_sid(), join_type)?.build();
        assert_inferred_eq(plan, expected)
    }

    /// Each restriction shape a comparison with literals takes is carried over.
    #[rstest]
    #[case::equality(|c: &str| unresolved_col(c).eq(lit(5i64)))]
    #[case::literal_first(|c: &str| lit(5i64).lt(unresolved_col(c)))]
    #[case::range(|c: &str| unresolved_col(c).gt_eq(lit(5i64)))]
    #[case::between(|c: &str| unresolved_col(c).between(lit(1i64), lit(9i64)))]
    #[case::or_chain(|c: &str| unresolved_col(c).eq(lit(1i64)).or(unresolved_col(c).eq(lit(4i64))))]
    #[case::is_in(|c: &str| unresolved_col(c).is_in((1..=6i64).map(lit).collect()))]
    fn derives_each_restriction_shape(#[case] restriction: fn(&str) -> ExprRef) -> DaftResult<()> {
        let (subjects, events) = (subjects().filter(restriction("id"))?, events());
        let plan = join(&subjects, &events, id_eq_sid(), JoinType::Inner)?.build();
        let expected = join(
            &subjects,
            &events.filter(restriction("sid"))?,
            id_eq_sid(),
            JoinType::Inner,
        )?
        .build();
        assert_inferred_eq(plan, expected)
    }

    /// Restrictions that aren't one plain column against literals are left where they are.
    #[rstest]
    #[case::not_equal(unresolved_col("id").not_eq(lit(5i64)))]
    #[case::not_in(unresolved_col("id").is_in(vec![lit(1i64), lit(2i64)]).not())]
    #[case::null_test(unresolved_col("id").is_null().not())]
    #[case::function_of_the_key(unresolved_col("id").cast(&DataType::Utf8).eq(lit("1")))]
    #[case::two_columns(unresolved_col("id").eq(lit(1i64)).or(unresolved_col("name").eq(lit("a"))))]
    fn leaves_other_restrictions_alone(#[case] restriction: ExprRef) -> DaftResult<()> {
        let plan = join(
            &subjects().filter(restriction)?,
            &events(),
            id_eq_sid(),
            JoinType::Inner,
        )?
        .build();
        assert_inferred_eq(plan.clone(), plan)
    }

    /// Keys whose equality in a join can disagree with a filter's comparison don't take part.
    #[rstest]
    #[case::null_safe(unresolved_col("a").eq_null_safe(unresolved_col("b")), DataType::Int64, DataType::Int64)]
    #[case::floats(unresolved_col("a").eq(unresolved_col("b")), DataType::Float64, DataType::Float64)]
    #[case::mismatched_dtypes(unresolved_col("a").eq(unresolved_col("b")), DataType::Int32, DataType::Int64)]
    #[case::expression_key(
        binary_op(Operator::Plus, unresolved_col("a"), lit(1i64)).eq(unresolved_col("b")),
        DataType::Int64,
        DataType::Int64
    )]
    fn skips_keys_that_may_disagree(
        #[case] on: ExprRef,
        #[case] left_dtype: DataType,
        #[case] right_dtype: DataType,
    ) -> DaftResult<()> {
        let left = exact_scan(&[("a", left_dtype.clone())]);
        let restriction = unresolved_col("a").eq(lit(1).cast(&left_dtype));
        let plan = join(
            &left.filter(restriction)?,
            &exact_scan(&[("b", right_dtype)]),
            on,
            JoinType::Inner,
        )?
        .build();
        assert_inferred_eq(plan.clone(), plan)
    }

    /// A side with a scan that may not apply a pushed filter exactly gets nothing derived, and a
    /// scan's pushed filters count only when it applies them exactly.
    #[test]
    fn trusts_only_scans_that_apply_filters_exactly() -> DaftResult<()> {
        let inexact_events = dummy_scan_node(dummy_scan_operator(vec![
            Field::new("sid", DataType::Int64),
            Field::new("v", DataType::Int64),
        ]));
        let plan = join(
            &subjects().filter(chosen("id"))?,
            &inexact_events,
            id_eq_sid(),
            JoinType::Inner,
        )?
        .build();
        assert_inferred_eq(plan.clone(), plan)?;

        let inexact_subjects = dummy_scan_node_with_pushdowns(
            dummy_scan_operator(vec![Field::new("id", DataType::Int64)]),
            Pushdowns::default().with_filters(Some(resolved_col("id").eq(lit(1i64)))),
        );
        let plan = join(&inexact_subjects, &events(), id_eq_sid(), JoinType::Inner)?.build();
        assert_inferred_eq(plan.clone(), plan)?;

        let exact_subjects = dummy_scan_node_with_pushdowns(
            exact_op(&[("id", DataType::Int64)], vec![]),
            Pushdowns::default().with_filters(Some(resolved_col("id").eq(lit(1i64)))),
        );
        let events = events();
        let plan = join(&exact_subjects, &events, id_eq_sid(), JoinType::Inner)?.build();
        let expected = join(
            &exact_subjects,
            &events.filter(resolved_col("sid").eq(lit(1i64)))?,
            id_eq_sid(),
            JoinType::Inner,
        )?
        .build();
        assert_inferred_eq(plan, expected)
    }

    /// Restrictions follow a column through renames and grouping on either side.
    #[test]
    fn follows_renames_and_groupings() -> DaftResult<()> {
        let renamed_subjects = subjects()
            .filter(chosen("id"))?
            .select(vec![unresolved_col("id").alias("subject")])?;
        let grouped_events = events().aggregate(
            vec![unresolved_col("v").sum()],
            vec![unresolved_col("sid").alias("key")],
        )?;
        let on = unresolved_col("subject").eq(unresolved_col("key"));
        let plan = join(
            &renamed_subjects,
            &grouped_events,
            on.clone(),
            JoinType::Left,
        )?
        .build();
        let expected = join(
            &renamed_subjects,
            &grouped_events.filter(chosen("key"))?,
            on,
            JoinType::Left,
        )?
        .build();
        assert_inferred_eq(plan, expected)
    }

    /// A restriction already held by the other side, however it was written, isn't added again.
    #[test]
    fn adds_nothing_the_other_side_holds() -> DaftResult<()> {
        let events_with_partition_filter = dummy_scan_node_with_pushdowns(
            exact_op(&[("sid", DataType::Int64)], vec![]),
            Pushdowns::default().with_partition_filters(Some(
                lit(5i64).cast(&DataType::Int64).eq(resolved_col("sid")),
            )),
        );
        let plan = join(
            &subjects().filter(unresolved_col("id").eq(lit(5i64)))?,
            &events_with_partition_filter,
            id_eq_sid(),
            JoinType::Inner,
        )?
        .build();
        assert_inferred_eq(plan.clone(), plan)
    }

    /// A restriction derived on top of a join's side reaches the joins inside that side, and the
    /// batch settles: optimizing its output again changes nothing.
    #[test]
    fn reaches_joins_inside_a_side_and_settles() -> DaftResult<()> {
        let details = exact_scan(&[("event_sid", DataType::Int64), ("code", DataType::Utf8)]);
        let event_details = join(
            &events(),
            &details,
            unresolved_col("sid").eq(unresolved_col("event_sid")),
            JoinType::Inner,
        )?;
        let plan = join(
            &subjects().filter(unresolved_col("id").eq(lit(5i64)))?,
            &event_details,
            id_eq_sid(),
            JoinType::Left,
        )?
        .build();

        let optimized = infer_and_push(plan)?;

        assert_eq!(
            pushed_filters(&optimized, "sid"),
            vec!["col(sid) == lit(5)".to_string()]
        );
        assert_eq!(
            pushed_filters(&optimized, "event_sid"),
            vec!["col(event_sid) == lit(5)".to_string()]
        );
        assert_eq!(infer_and_push(optimized.clone())?, optimized);
        Ok(())
    }

    /// A derived restriction on a partition column settles once it becomes a partition filter.
    #[test]
    fn settles_on_a_partition_column() -> DaftResult<()> {
        let key = PartitionField::new(Field::new("sid", DataType::Int64), None, None)?;
        let partitioned_events = dummy_scan_node(exact_op(&[("sid", DataType::Int64)], vec![key]));
        let plan = join(
            &subjects().filter(unresolved_col("id").eq(lit(5i64)))?,
            &partitioned_events,
            id_eq_sid(),
            JoinType::Inner,
        )?
        .build();

        let optimized = infer_and_push(plan)?;

        assert_eq!(pushed_filters(&optimized, "sid").len(), 1);
        assert_eq!(infer_and_push(optimized.clone())?, optimized);
        Ok(())
    }
}
