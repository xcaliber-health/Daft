//! The column sets a plan's rows are unique by.
//!
//! Two kinds of fact, never mixed up:
//!
//! - **Proven** keys follow from the plan itself: an aggregation is unique by its grouping,
//!   a distinct by its columns, and a filter keeping `row_number() == 1` over a window by
//!   that window's partitioning. A rule may rely on them to change a plan's shape in a way
//!   that would change its answer were they wrong: [`proven_keys`].
//! - **Assumed** keys add what a caller stated about a source without it being checked
//!   ([`crate::ops::Source::assumed_keys`]). They may inform cost decisions only, whose
//!   every outcome answers alike, so a wrong statement costs time and never a result:
//!   [`assumed_keys`].
//!
//! Both are carried through the operators that keep rows apart: filters, sorts, limits and
//! other operators that only reorder, split or drop rows, windows, renaming projections,
//! and joins whose other side cannot repeat a row.

use std::collections::HashMap;

use daft_algebra::boolean::split_conjunction;
use daft_core::{join::JoinType, prelude::Operator};
use daft_dsl::{Expr, ExprRef, WindowExpr};
use indexmap::IndexSet;

use super::rules::passed_through_columns;
use crate::{
    LogicalPlan,
    logical_plan::SubqueryAlias,
    ops::{
        Aggregate, Distinct, Filter, IntoBatches, IntoPartitions, Join, Limit, Offset, Project,
        Repartition, Sample, Shard, Shared, Sort, TopN, Window,
    },
};

/// Which keys a derivation may use.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Trust {
    /// Only keys that follow from the plan.
    Proven,
    /// Those, and keys a caller stated about a source.
    Assumed,
}

/// A set of columns no two of a plan's rows share values in.
///
/// A member `None` stands for a column the plan no longer outputs under any name, such as
/// one a projection dropped: the rows are still unique by it, but no set of the plan's
/// columns covers the key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Key(Vec<Option<String>>);

impl Key {
    fn of<'a>(columns: impl IntoIterator<Item = &'a str>) -> Self {
        Self(columns.into_iter().map(|c| Some(c.to_string())).collect())
    }

    /// Whether every column of the key is among `columns`.
    fn within(&self, columns: &IndexSet<String>) -> bool {
        self.0
            .iter()
            .all(|column| column.as_ref().is_some_and(|c| columns.contains(c)))
    }

    /// Both keys' columns: rows of a join are unique by them when each side's are.
    fn union(&self, other: &Self) -> Self {
        Self(self.0.iter().chain(&other.0).cloned().collect())
    }

    /// The key above a projection, given each input column's output name from
    /// [`outputs_of`]: a column the projection drops or computes from becomes unnamed.
    fn through(self, outputs: &HashMap<String, String>) -> Self {
        Self(
            self.0
                .into_iter()
                .map(|column| column.and_then(|c| outputs.get(&c).cloned()))
                .collect(),
        )
    }
}

/// Each input column a projection passes through unchanged, as the first output naming it.
fn outputs_of(projection: &[ExprRef]) -> HashMap<String, String> {
    let mut outputs = HashMap::new();
    for (output, input_column) in passed_through_columns(projection) {
        outputs
            .entry(input_column.name().to_string())
            .or_insert(output);
    }
    outputs
}

/// The keys that follow from `plan` itself. Any rule may rely on them.
pub(crate) fn proven_keys(plan: &LogicalPlan) -> Vec<Key> {
    keys(plan, Trust::Proven)
}

/// The keys of `plan`, including those a caller stated about its sources unchecked.
///
/// For cost decisions only, whose every outcome gives the same answer: a wrong stated key
/// may cost time and must never change a result.
pub(crate) fn assumed_keys(plan: &LogicalPlan) -> Vec<Key> {
    keys(plan, Trust::Assumed)
}

/// Whether one of `keys` lies within `columns`, so no two rows share values in them.
pub(crate) fn unique_by(keys: &[Key], columns: &IndexSet<String>) -> bool {
    keys.iter().any(|key| key.within(columns))
}

#[recursive::recursive]
fn keys(plan: &LogicalPlan, trust: Trust) -> Vec<Key> {
    match plan {
        LogicalPlan::Source(source) if trust == Trust::Assumed => source
            .assumed_keys
            .iter()
            .map(|key| {
                Key(key
                    .iter()
                    .map(|c| source.output_schema.has_field(c).then(|| c.clone()))
                    .collect())
            })
            .collect(),
        LogicalPlan::Aggregate(Aggregate { groupby, .. }) => {
            vec![Key::of(groupby.iter().map(|e| e.name()))]
        }
        LogicalPlan::Distinct(Distinct {
            columns: Some(columns),
            ..
        }) => vec![Key::of(columns.iter().map(|e| e.name()))],
        LogicalPlan::Distinct(Distinct { columns: None, .. }) => {
            vec![Key::of(plan.schema().field_names())]
        }
        LogicalPlan::Filter(Filter {
            input, predicate, ..
        }) => {
            let mut keys = keys(input, trust);
            keys.extend(
                split_conjunction(predicate)
                    .iter()
                    .filter_map(row_number_one)
                    .filter_map(|column| first_row_partition(input, column)),
            );
            keys
        }
        // Operators that only reorder, split or drop rows keep them apart.
        LogicalPlan::Sort(Sort { input, .. })
        | LogicalPlan::Limit(Limit { input, .. })
        | LogicalPlan::Offset(Offset { input, .. })
        | LogicalPlan::TopN(TopN { input, .. })
        | LogicalPlan::Shard(Shard { input, .. })
        | LogicalPlan::Repartition(Repartition { input, .. })
        | LogicalPlan::IntoPartitions(IntoPartitions { input, .. })
        | LogicalPlan::IntoBatches(IntoBatches { input, .. })
        | LogicalPlan::Window(Window { input, .. })
        | LogicalPlan::SubqueryAlias(SubqueryAlias { input, .. })
        // A shared part hands each reader its rows unchanged.
        | LogicalPlan::Shared(Shared { input, .. }) => keys(input, trust),
        // Sampling with replacement may draw a row twice.
        LogicalPlan::Sample(Sample {
            input,
            with_replacement: false,
            ..
        }) => keys(input, trust),
        LogicalPlan::Project(Project {
            input, projection, ..
        }) => {
            let outputs = outputs_of(projection);
            keys(input, trust)
                .into_iter()
                .map(|key| key.through(&outputs))
                .collect()
        }
        LogicalPlan::Join(join) => join_keys(join, trust),
        _ => Vec::new(),
    }
}

/// The keys of a join's output.
///
/// A side keeps its keys when the other side holds at most one row per value of its join
/// keys, so no row of the first side is repeated. Rows are also unique by a key of each
/// side together, whenever both sides have one.
fn join_keys(join: &Join, trust: Trust) -> Vec<Key> {
    let (_, left_on, right_on, _) = join.on.split_eq_preds();
    let left = keys(&join.left, trust);
    let right = keys(&join.right, trust);
    let left_unique = unique_by(&left, &column_names(&left_on));
    let right_unique = unique_by(&right, &column_names(&right_on));
    let both = left.first().zip(right.first()).map(|(l, r)| l.union(r));
    let mut out = Vec::new();
    match join.join_type {
        JoinType::Inner => {
            if right_unique {
                out.extend(left);
            }
            if left_unique {
                out.extend(right);
            }
            out.extend(both);
        }
        JoinType::Left => {
            if right_unique {
                out.extend(left);
            }
            out.extend(both);
        }
        JoinType::Right => {
            if left_unique {
                out.extend(right);
            }
            out.extend(both);
        }
        JoinType::Semi | JoinType::Anti => out.extend(left),
        JoinType::Outer => {}
    }
    out
}

/// The columns `exprs` read unchanged; an expression that computes adds none.
fn column_names(exprs: &[ExprRef]) -> IndexSet<String> {
    exprs.iter().filter_map(|e| e.input_mapping()).collect()
}

/// The column a conjunct keeps at 1, if it is `column == 1` for an integer 1.
fn row_number_one(conjunct: &ExprRef) -> Option<&str> {
    fn is_one(expr: &Expr) -> bool {
        match expr {
            Expr::Literal(value) => matches!(value.try_as_usize(), Ok(Some(1))),
            Expr::Cast(inner, ..) => is_one(inner),
            _ => false,
        }
    }
    let Expr::BinaryOp {
        op: Operator::Eq,
        left,
        right,
    } = conjunct.as_ref()
    else {
        return None;
    };
    let column = if is_one(right) {
        left
    } else if is_one(left) {
        right
    } else {
        return None;
    };
    match column.as_ref() {
        Expr::Column(_) => Some(column.name()),
        _ => None,
    }
}

/// The partitioning of the window below `plan` whose `row_number()` is output as `column`,
/// as a key: keeping row number 1 leaves one row per partition. `None` for any other
/// column, including `rank()` and `dense_rank()`, which number ties alike.
#[recursive::recursive]
fn first_row_partition(plan: &LogicalPlan, column: &str) -> Option<Key> {
    match plan {
        LogicalPlan::Window(Window {
            window_functions,
            aliases,
            window_spec,
            input,
            ..
        }) => {
            let numbered = window_functions
                .iter()
                .zip(aliases)
                .any(|(function, alias)| {
                    alias == column && matches!(function, WindowExpr::RowNumber)
                });
            if numbered {
                let partition = column_names(&window_spec.partition_by);
                (partition.len() == window_spec.partition_by.len())
                    .then(|| Key(partition.into_iter().map(Some).collect()))
            } else {
                first_row_partition(input, column)
            }
        }
        // Rows dropped or reordered between the window and the filter keep at most one
        // row per partition.
        LogicalPlan::Filter(Filter { input, .. }) | LogicalPlan::Sort(Sort { input, .. }) => {
            first_row_partition(input, column)
        }
        // Through a projection, the number is followed to its input column and the
        // partition brought back to the projection's names.
        LogicalPlan::Project(Project {
            input, projection, ..
        }) => {
            let input_column = passed_through_columns(projection).remove(column)?;
            first_row_partition(input, input_column.name())
                .map(|key| key.through(&outputs_of(projection)))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use common_error::DaftResult;
    use daft_core::{join::JoinType, prelude::*};
    use daft_dsl::{WindowExpr, expr::window::WindowSpec, lit, resolved_col, unresolved_col};
    use indexmap::IndexSet;
    use rstest::rstest;

    use super::{assumed_keys, proven_keys, unique_by};
    use crate::{
        LogicalPlan, LogicalPlanBuilder,
        ops::Window,
        test::{dummy_scan_node, dummy_scan_operator},
    };

    fn columns(names: &[&str]) -> IndexSet<String> {
        names.iter().map(ToString::to_string).collect()
    }

    /// Records of subjects, stated unique by `record_id`.
    fn records() -> LogicalPlanBuilder {
        dummy_scan_node(dummy_scan_operator(vec![
            Field::new("record_id", DataType::Int64),
            Field::new("subject", DataType::Int64),
            Field::new("version", DataType::Int64),
        ]))
        .with_assumed_key(vec!["record_id".to_string()])
        .unwrap()
    }

    fn events() -> LogicalPlanBuilder {
        dummy_scan_node(dummy_scan_operator(vec![
            Field::new("event_id", DataType::Int64),
            Field::new("subject", DataType::Int64),
        ]))
        .with_assumed_key(vec!["event_id".to_string()])
        .unwrap()
    }

    /// The records numbered within each subject by `function`, keeping number `kept`.
    fn numbered(function: WindowExpr, kept: u64) -> DaftResult<Arc<LogicalPlan>> {
        let spec = WindowSpec {
            partition_by: vec![resolved_col("subject")],
            order_by: vec![resolved_col("version")],
            descending: vec![true],
            nulls_first: vec![false],
            ..Default::default()
        };
        let window = Window::try_new(
            records().build(),
            vec![function],
            vec!["rn".to_string()],
            Arc::new(spec),
        )?;
        Ok(
            LogicalPlanBuilder::new(LogicalPlan::Window(window).into(), None)
                .filter(resolved_col("rn").eq(lit(kept)))?
                .build(),
        )
    }

    #[test]
    fn a_stated_key_is_assumed_and_never_proven() {
        let plan = records().build();

        assert!(unique_by(&assumed_keys(&plan), &columns(&["record_id"])));
        assert!(proven_keys(&plan).is_empty());
    }

    #[test]
    fn a_key_whose_column_is_dropped_is_known_but_covered_by_no_columns() -> DaftResult<()> {
        let plan = records().select(vec![unresolved_col("subject")])?.build();

        let keys = assumed_keys(&plan);

        assert_eq!(keys.len(), 1);
        assert!(!unique_by(&keys, &columns(&["subject"])));
        Ok(())
    }

    #[test]
    fn a_renamed_key_is_unique_by_its_new_name() -> DaftResult<()> {
        let plan = records()
            .select(vec![unresolved_col("record_id").alias("id")])?
            .build();

        assert!(unique_by(&assumed_keys(&plan), &columns(&["id"])));
        Ok(())
    }

    #[rstest]
    #[case::aggregate_by_its_grouping(&["subject"], true)]
    #[case::aggregate_not_by_part_of_it(&["version"], false)]
    fn an_aggregation_is_proven_unique_by_its_grouping(
        #[case] by: &[&str],
        #[case] unique: bool,
    ) -> DaftResult<()> {
        let plan = records()
            .aggregate(
                vec![unresolved_col("version").max()],
                vec![unresolved_col("subject")],
            )?
            .build();

        assert_eq!(unique_by(&proven_keys(&plan), &columns(by)), unique);
        Ok(())
    }

    #[test]
    fn a_distinct_is_proven_unique_by_its_columns() -> DaftResult<()> {
        let plan = records()
            .distinct(Some(vec![unresolved_col("subject")]))?
            .build();

        assert!(unique_by(&proven_keys(&plan), &columns(&["subject"])));
        Ok(())
    }

    #[rstest]
    #[case::first_row_number(WindowExpr::RowNumber, 1, true)]
    #[case::second_row_number(WindowExpr::RowNumber, 2, false)]
    // Ties share a rank: several rows of a subject can rank first.
    #[case::first_rank(WindowExpr::Rank, 1, false)]
    #[case::first_dense_rank(WindowExpr::DenseRank, 1, false)]
    fn keeping_the_first_row_number_is_proven_unique_by_the_partition(
        #[case] function: WindowExpr,
        #[case] kept: u64,
        #[case] unique: bool,
    ) -> DaftResult<()> {
        let plan = numbered(function, kept)?;

        assert_eq!(
            unique_by(&proven_keys(&plan), &columns(&["subject"])),
            unique
        );
        Ok(())
    }

    #[test]
    fn the_first_row_number_is_followed_through_a_projection_between_window_and_filter()
    -> DaftResult<()> {
        let spec = WindowSpec {
            partition_by: vec![resolved_col("subject")],
            order_by: vec![resolved_col("version")],
            descending: vec![true],
            nulls_first: vec![false],
            ..Default::default()
        };
        let window = Window::try_new(
            records().build(),
            vec![WindowExpr::RowNumber],
            vec!["rn".to_string()],
            Arc::new(spec),
        )?;
        let plan = LogicalPlanBuilder::new(LogicalPlan::Window(window).into(), None)
            .select(vec![resolved_col("subject").alias("s"), resolved_col("rn")])?
            .filter(resolved_col("rn").eq(lit(1u64)))?
            .build();

        assert!(unique_by(&proven_keys(&plan), &columns(&["s"])));
        Ok(())
    }

    #[rstest]
    // Each event names one subject, held once: events stay unique by their key.
    #[case::inner_on_a_unique_side(JoinType::Inner, true, true)]
    #[case::left_on_a_unique_side(JoinType::Left, true, true)]
    #[case::semi(JoinType::Semi, false, true)]
    #[case::outer(JoinType::Outer, true, false)]
    // Several records per subject repeat each event.
    #[case::inner_on_a_repeating_side(JoinType::Inner, false, false)]
    fn a_join_keeps_a_sides_key_when_the_other_cannot_repeat_its_rows(
        #[case] join_type: JoinType,
        #[case] chosen_record: bool,
        #[case] events_stay_unique: bool,
    ) -> DaftResult<()> {
        let other: LogicalPlanBuilder = if chosen_record {
            LogicalPlanBuilder::new(numbered(WindowExpr::RowNumber, 1)?, None)
        } else {
            records()
        };
        let other = other.select(vec![
            unresolved_col("subject").alias("record_subject"),
            unresolved_col("record_id"),
        ])?;
        let plan = events()
            .join(
                other,
                Some(unresolved_col("subject").eq(unresolved_col("record_subject"))),
                vec![],
                join_type,
                None,
                Default::default(),
            )?
            .build();

        assert_eq!(
            unique_by(&assumed_keys(&plan), &columns(&["event_id"])),
            events_stay_unique
        );
        Ok(())
    }

    #[test]
    fn an_inner_join_is_unique_by_a_key_of_each_side() -> DaftResult<()> {
        let plan = events()
            .join(
                records().select(vec![
                    unresolved_col("subject").alias("record_subject"),
                    unresolved_col("record_id"),
                ])?,
                Some(unresolved_col("subject").eq(unresolved_col("record_subject"))),
                vec![],
                JoinType::Inner,
                None,
                Default::default(),
            )?
            .build();

        assert!(unique_by(
            &assumed_keys(&plan),
            &columns(&["event_id", "record_id"])
        ));
        Ok(())
    }
}
