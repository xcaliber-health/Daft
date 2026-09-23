use std::sync::Arc;

use common_error::DaftResult;
use daft_core::prelude::*;
use daft_dsl::{WindowExpr, expr::window::WindowSpec};
use serde::{Deserialize, Serialize};

use crate::{
    logical_plan::{LogicalPlan, Result},
    stats::StatsState,
};

/// Window operator for computing window functions.
///
/// The Window operator represents window function operations in the logical plan.
/// When a user calls an expression like `df.select(col("a").sum().over(window))`,
/// it gets translated into this operator as follows:
///
/// 1. The aggregation function `col("a").sum()` is stored in the `window_functions` vector
/// 2. The window specification (partition by, order by, frame) is stored in the `window_spec` field
///
/// For example, `df.select(col("a").sum().over(window.partition_by("b")))` becomes:
/// ```
/// Window {
///   window_functions = [col("a").sum()],
///   window_spec = WindowSpec { partition_by: [col("b")], ... }
/// }
/// ```
///
/// Multiple window function expressions can be stored in a single Window operator
/// as long as they share the same window specification.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(debug_assertions, derive(Debug))]
pub struct Window {
    /// An id for the plan.
    pub plan_id: Option<usize>,
    pub node_id: Option<usize>,
    /// The input plan.
    pub input: Arc<LogicalPlan>,
    /// The window functions to compute.
    pub window_functions: Vec<WindowExpr>,
    /// The window function names to map to the output schema.
    pub aliases: Vec<String>,
    /// The window specification (partition by, order by, frame, etc.)
    pub window_spec: Arc<WindowSpec>,
    /// The output schema.
    pub schema: Arc<Schema>,
    /// The plan statistics.
    pub stats_state: StatsState,
}

impl Window {
    pub(crate) fn try_new(
        input: Arc<LogicalPlan>,
        window_functions: Vec<WindowExpr>,
        aliases: Vec<String>,
        window_spec: Arc<WindowSpec>,
    ) -> Result<Self> {
        for func in &window_functions {
            if matches!(
                func,
                WindowExpr::FirstValue(_, _) | WindowExpr::LastValue(_, _)
            ) {
                if window_spec.partition_by.is_empty() {
                    return Err(common_error::DaftError::ValueError(
                        "first_value() and last_value() require a partition_by in the window spec — use Window().partition_by(...).order_by(...).rows_between(...)".to_string(),
                    ).into());
                }
                if window_spec.order_by.is_empty() {
                    return Err(common_error::DaftError::ValueError(
                        "first_value() and last_value() require an order_by in the window spec — use Window().partition_by(...).order_by(...).rows_between(...)".to_string(),
                    ).into());
                }
                if window_spec.frame.is_none() {
                    return Err(common_error::DaftError::ValueError(
                        "first_value() and last_value() require a frame (rows_between) in the window spec — use Window().partition_by(...).order_by(...).rows_between(...)".to_string(),
                    ).into());
                }
            }
        }

        if window_spec.frame.is_some()
            && let Some(positional) = window_functions.iter().find_map(positional_function_name)
        {
            // These functions answer from a row's position within its partition, which a
            // frame does not change; accepting one would suggest it did.
            return Err(common_error::DaftError::ValueError(format!(
                "{positional} does not take a frame (rows_between or range_between) in its window spec — use Window().partition_by(...).order_by(...) without a frame"
            ))
            .into());
        }

        let input_schema = input.schema();

        let fields = input_schema
            .into_iter()
            .cloned()
            .map(Ok)
            .chain(
                aliases
                    .iter()
                    .zip(window_functions.iter())
                    .map(|(name, expr)| {
                        let dtype = expr.to_field(&input_schema)?.dtype;
                        Ok(Field::new(name.as_str(), dtype))
                    }),
            )
            .collect::<DaftResult<Vec<_>>>()?;

        let schema = Arc::new(Schema::new(fields));

        Ok(Self {
            plan_id: None,
            node_id: None,
            input,
            window_functions,
            aliases,
            window_spec,
            schema,
            stats_state: StatsState::NotMaterialized,
        })
    }

    pub fn with_materialized_stats(mut self) -> Self {
        // For now, just use the input's stats as an approximation
        let input_stats = self.input.materialized_stats();
        self.stats_state = StatsState::Materialized(input_stats.clone().into());
        self
    }

    pub fn with_plan_id(mut self, id: usize) -> Self {
        self.plan_id = Some(id);
        self
    }

    pub fn with_node_id(mut self, id: usize) -> Self {
        self.node_id = Some(id);
        self
    }
}

impl Window {
    pub fn multiline_display(&self) -> Vec<String> {
        let mut lines = vec!["Window:".to_string()];

        for (expr, name) in self.window_functions.iter().zip(self.aliases.iter()) {
            lines.push(format!("  {} as {}", expr, name));
        }

        if !self.window_spec.partition_by.is_empty() {
            let partition_cols = self
                .window_spec
                .partition_by
                .iter()
                .map(|e| e.name().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!("  Partition by: [{}]", partition_cols));
        }

        if !self.window_spec.order_by.is_empty() {
            let order_cols = self
                .window_spec
                .order_by
                .iter()
                .zip(self.window_spec.descending.iter())
                .zip(self.window_spec.nulls_first.iter())
                .map(|((e, desc), nulls_first)| {
                    format!(
                        "{} {} {}",
                        e.name(),
                        if *desc { "DESC" } else { "ASC" },
                        if *nulls_first {
                            "NULLS FIRST"
                        } else {
                            "NULLS LAST"
                        }
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!("  Order by: [{}]", order_cols));
        }

        if let Some(frame) = &self.window_spec.frame {
            let start = match &frame.start {
                daft_dsl::expr::window::WindowBoundary::UnboundedPreceding => {
                    "UNBOUNDED PRECEDING".to_string()
                }
                daft_dsl::expr::window::WindowBoundary::UnboundedFollowing => {
                    "UNBOUNDED FOLLOWING".to_string()
                }
                daft_dsl::expr::window::WindowBoundary::Offset(n) => match n.cmp(&0) {
                    std::cmp::Ordering::Equal => "CURRENT ROW".to_string(),
                    std::cmp::Ordering::Less => format!("{} PRECEDING", n.abs()),
                    std::cmp::Ordering::Greater => format!("{} FOLLOWING", n),
                },
                daft_dsl::expr::window::WindowBoundary::RangeOffset(n) => {
                    format!("RANGE {}", n)
                }
            };

            let end = match &frame.end {
                daft_dsl::expr::window::WindowBoundary::UnboundedPreceding => {
                    "UNBOUNDED PRECEDING".to_string()
                }
                daft_dsl::expr::window::WindowBoundary::UnboundedFollowing => {
                    "UNBOUNDED FOLLOWING".to_string()
                }
                daft_dsl::expr::window::WindowBoundary::Offset(n) => match n.cmp(&0) {
                    std::cmp::Ordering::Equal => "CURRENT ROW".to_string(),
                    std::cmp::Ordering::Less => format!("{} PRECEDING", n.abs()),
                    std::cmp::Ordering::Greater => format!("{} FOLLOWING", n),
                },
                daft_dsl::expr::window::WindowBoundary::RangeOffset(n) => {
                    format!("RANGE {}", n)
                }
            };

            lines.push(format!("  Frame: BETWEEN {} AND {}", start, end));
        }

        if self.window_spec.min_periods != 1 {
            lines.push(format!("  Min periods: {}", self.window_spec.min_periods));
        }

        if let StatsState::Materialized(stats) = &self.stats_state {
            lines.push(format!("Stats = {}", stats));
        }

        lines
    }
}

/// The name of a window function that answers from a row's position alone, if `function` is one.
fn positional_function_name(function: &WindowExpr) -> Option<&'static str> {
    match function {
        WindowExpr::RowNumber => Some("row_number()"),
        WindowExpr::Rank => Some("rank()"),
        WindowExpr::DenseRank => Some("dense_rank()"),
        WindowExpr::Offset { offset, .. } if *offset < 0 => Some("lag()"),
        WindowExpr::Offset { .. } => Some("lead()"),
        WindowExpr::Agg(_) | WindowExpr::FirstValue(..) | WindowExpr::LastValue(..) => None,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use daft_core::prelude::*;
    use daft_dsl::{
        WindowExpr,
        expr::window::{WindowBoundary, WindowFrame, WindowSpec},
        resolved_col,
    };
    use rstest::rstest;

    use super::Window;
    use crate::test::{dummy_scan_node, dummy_scan_operator};

    fn spec(frame: Option<WindowFrame>) -> Arc<WindowSpec> {
        Arc::new(WindowSpec {
            partition_by: vec![resolved_col("g")],
            order_by: vec![resolved_col("o")],
            descending: vec![false],
            nulls_first: vec![false],
            frame,
            ..Default::default()
        })
    }

    fn running() -> Option<WindowFrame> {
        Some(WindowFrame {
            start: WindowBoundary::UnboundedPreceding,
            end: WindowBoundary::Offset(0),
        })
    }

    fn offset(by: isize) -> WindowExpr {
        WindowExpr::Offset {
            input: resolved_col("o"),
            offset: by,
            default: None,
        }
    }

    #[rstest]
    #[case::row_number(WindowExpr::RowNumber, running(), Some("row_number()"))]
    #[case::rank(WindowExpr::Rank, running(), Some("rank()"))]
    #[case::dense_rank(WindowExpr::DenseRank, running(), Some("dense_rank()"))]
    #[case::lag(offset(-1), running(), Some("lag()"))]
    #[case::lead(offset(1), running(), Some("lead()"))]
    #[case::row_number_unframed(WindowExpr::RowNumber, None, None)]
    fn a_positional_function_takes_no_frame(
        #[case] function: WindowExpr,
        #[case] frame: Option<WindowFrame>,
        #[case] refused_as: Option<&str>,
    ) {
        let input = dummy_scan_node(dummy_scan_operator(vec![
            Field::new("g", DataType::Int64),
            Field::new("o", DataType::Int64),
        ]))
        .build();

        let built = Window::try_new(input, vec![function], vec!["w".to_string()], spec(frame));

        match refused_as {
            Some(name) => {
                let message = built.err().map(|e| e.to_string()).unwrap_or_default();
                assert!(
                    message.contains(&format!("{name} does not take a frame")),
                    "{message}"
                );
            }
            None => assert!(built.is_ok()),
        }
    }
}
