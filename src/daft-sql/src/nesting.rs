//! How deeply a SQL statement may nest, and the stack its syntax tree needs.
//!
//! The parser bounds the nesting it recurses into (parentheses, subqueries,
//! function calls), but builds a chain of infix operators, such as
//! `a AND b AND c …`, in a loop: the chain is as deep as it is long, and the
//! syntax tree holding it is dropped by recursing once per level. So each
//! statement is parsed, planned and dropped on a stack sized to its tokens, and
//! planning refuses a statement whose expressions nest deeper than
//! [`MAX_EXPRESSION_NESTING`] or whose queries nest deeper than
//! [`MAX_QUERY_NESTING`].

use std::{cell::Cell, rc::Rc};

use sqlparser::tokenizer::Token;

use crate::error::{PlannerError, SQLPlannerResult};

/// The most levels a statement's expressions may nest, each inside the one
/// before: a chain of `n` infix operators nests `n` levels, a `CASE` one per
/// `WHEN`, and a compound identifier one per part.
pub(crate) const MAX_EXPRESSION_NESTING: usize = 1_000;

/// The most levels a statement's queries may nest, each inside the one before:
/// each query, set operation, join and `FROM` item nests one level.
pub(crate) const MAX_QUERY_NESTING: usize = 2_000;

/// Stack reserved per token for the syntax tree, in bytes.
///
/// Each level of the tree takes at least two tokens (an operator and its
/// operand), and dropping a level takes about 100 bytes of stack.
const STACK_PER_TOKEN: usize = 128;

/// Stack reserved for the parser's own recursion, which its nesting limit bounds,
/// in bytes.
const STACK_FOR_PARSER: usize = 1024 * 1024;

/// The most stack a statement may reserve, in bytes. Only the pages a statement
/// uses are ever touched.
const MAX_STACK: usize = 1024 * 1024 * 1024;

/// Counts the tokens that can take a level of a syntax tree: all but whitespace.
pub(crate) fn significant<'a>(tokens: impl IntoIterator<Item = &'a Token>) -> usize {
    tokens
        .into_iter()
        .filter(|token| !matches!(token, Token::Whitespace(_)))
        .count()
}

/// Runs `parse_and_plan` with stack enough for the syntax tree of a statement
/// of `tokens` [significant] tokens.
///
/// The tree must be dropped within `parse_and_plan`. Nothing is allocated when
/// the current stack already has room.
///
/// # Errors
/// Returns an error, without running `parse_and_plan`, if the statement is too
/// long to reserve its stack, and otherwise whatever `parse_and_plan` returns.
pub(crate) fn with_stack_for<T>(
    tokens: usize,
    parse_and_plan: impl FnOnce() -> SQLPlannerResult<T>,
) -> SQLPlannerResult<T> {
    let needed = tokens
        .checked_mul(STACK_PER_TOKEN)
        .and_then(|bytes| bytes.checked_add(STACK_FOR_PARSER))
        .filter(|bytes| *bytes <= MAX_STACK)
        .ok_or_else(|| {
            PlannerError::invalid_operation(format!(
                "SQL statement of {tokens} tokens is too long to parse"
            ))
        })?;
    stacker::maybe_grow(needed, needed, parse_and_plan)
}

/// What a level of nesting is a level of.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Nested {
    /// An expression inside an expression.
    Expression,
    /// A query, or a relation it reads, inside a query.
    Query,
}

impl Nested {
    const fn limit(self) -> usize {
        match self {
            Self::Expression => MAX_EXPRESSION_NESTING,
            Self::Query => MAX_QUERY_NESTING,
        }
    }

    const fn noun(self) -> &'static str {
        match self {
            Self::Expression => "expressions",
            Self::Query => "queries",
        }
    }
}

/// The nesting of the statement being planned, shared by the planners of all
/// its scopes.
#[derive(Clone, Default)]
pub(crate) struct Nesting {
    expressions: Rc<Cell<usize>>,
    queries: Rc<Cell<usize>>,
}

impl Nesting {
    /// Enters `levels` more levels of `nested` until the returned guard drops.
    ///
    /// # Errors
    /// Returns an error, entering nothing, if the statement would nest deeper
    /// than the limit for `nested`.
    pub(crate) fn descend(&self, nested: Nested, levels: usize) -> SQLPlannerResult<Level> {
        let depth = match nested {
            Nested::Expression => &self.expressions,
            Nested::Query => &self.queries,
        };
        let deeper = depth.get().saturating_add(levels);
        if deeper > nested.limit() {
            return Err(PlannerError::invalid_operation(format!(
                "SQL {} may nest at most {} levels deep",
                nested.noun(),
                nested.limit()
            )));
        }
        depth.set(deeper);
        Ok(Level {
            depth: depth.clone(),
            levels,
        })
    }
}

/// Levels of nesting entered by [`Nesting::descend`], left when dropped.
pub(crate) struct Level {
    depth: Rc<Cell<usize>>,
    levels: usize,
}

impl Drop for Level {
    fn drop(&mut self) {
        self.depth.set(self.depth.get() - self.levels);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use daft_core::prelude::*;
    use daft_logical_plan::{
        ClusteringSpec, LogicalPlan, SourceInfo, logical_plan::Source, source_info::PlaceHolderInfo,
    };
    use daft_session::Session;
    use itertools::Itertools;
    use rstest::rstest;

    use super::*;
    use crate::planner::SQLPlanner;

    #[rstest]
    #[case::expressions(Nested::Expression)]
    #[case::queries(Nested::Query)]
    fn levels_are_left_when_their_guard_drops(#[case] nested: Nested) {
        let nesting = Nesting::default();
        let outer = nesting.descend(nested, 3).unwrap();
        {
            let _inner = nesting.descend(nested, nested.limit() - 3).unwrap();
            assert!(nesting.descend(nested, 1).is_err());
        }
        assert!(nesting.descend(nested, nested.limit() - 3).is_ok());
        drop(outer);
        assert!(nesting.descend(nested, nested.limit()).is_ok());
    }

    #[rstest]
    #[case::expressions(Nested::Expression)]
    #[case::queries(Nested::Query)]
    fn a_refused_descent_enters_nothing(#[case] nested: Nested) {
        let nesting = Nesting::default();
        let _level = nesting.descend(nested, nested.limit() - 1).unwrap();

        assert!(nesting.descend(nested, 2).is_err());
        assert!(nesting.descend(nested, 1).is_ok());
    }

    /// A statement of one construct chained `n` times, over a one-column table `t`.
    fn chained(shape: &str, n: usize) -> String {
        match shape {
            "and" => format!(
                "SELECT x FROM t WHERE {}",
                (0..n).map(|i| format!("x >= {i}")).join(" AND ")
            ),
            "plus" => format!("SELECT x{} AS x FROM t", " + 1".repeat(n)),
            "case" => format!(
                "SELECT CASE {} ELSE 0 END AS x FROM t",
                (0..n).map(|i| format!("WHEN x < {i} THEN {i}")).join(" ")
            ),
            "union" => vec!["SELECT x FROM t"; n].join(" UNION ALL "),
            "from" => format!("SELECT 1 AS x FROM {}", vec!["t"; n].join(", ")),
            _ => unreachable!("no shape {shape}"),
        }
    }

    /// Plans `sql` on a thread with a stack of `stack` bytes.
    fn plan_on(stack: usize, sql: String) -> SQLPlannerResult<()> {
        std::thread::Builder::new()
            .stack_size(stack)
            .spawn(move || {
                let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64)]));
                let table = LogicalPlan::Source(Source::new(
                    schema.clone(),
                    Arc::new(SourceInfo::PlaceHolder(PlaceHolderInfo {
                        source_schema: schema,
                        clustering_spec: Arc::new(ClusteringSpec::unknown(0)),
                    })),
                ))
                .arced();
                let session = Session::default();
                session
                    .create_temp_table("t", &table.into(), false)
                    .unwrap();
                SQLPlanner::new(&session).plan_sql(&sql).map(drop)
            })
            .unwrap()
            .join()
            .unwrap()
    }

    #[rstest]
    #[case::and("and", MAX_EXPRESSION_NESTING - 10)]
    #[case::plus("plus", MAX_EXPRESSION_NESTING - 10)]
    #[case::case("case", MAX_EXPRESSION_NESTING - 10)]
    #[case::union("union", MAX_QUERY_NESTING - 10)]
    #[case::from("from", MAX_QUERY_NESTING / 2)]
    fn a_statement_within_the_limits_plans_on_a_small_stack(#[case] shape: &str, #[case] n: usize) {
        plan_on(1024 * 1024, chained(shape, n)).unwrap();
    }

    #[rstest]
    #[case::and("and", 200_000)]
    #[case::plus("plus", 200_000)]
    #[case::case("case", 100_000)]
    #[case::union("union", 100_000)]
    #[case::from("from", 100_000)]
    fn a_statement_far_past_the_limits_is_refused_not_overflowed(
        #[case] shape: &str,
        #[case] n: usize,
    ) {
        let refused = plan_on(256 * 1024, chained(shape, n)).unwrap_err();

        assert!(
            refused.to_string().contains("may nest at most"),
            "{refused}"
        );
    }

    #[test]
    fn expressions_and_queries_nest_apart() {
        let nesting = Nesting::default();
        let _queries = nesting.descend(Nested::Query, MAX_QUERY_NESTING).unwrap();

        assert!(
            nesting
                .descend(Nested::Expression, MAX_EXPRESSION_NESTING)
                .is_ok()
        );
    }
}
