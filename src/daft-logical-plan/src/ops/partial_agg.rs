//! A partial aggregation: grouped aggregates that a later aggregation combines.

use std::sync::Arc;

use common_error::DaftError;
use daft_dsl::{ExprRef, expr::agg::extract_agg_expr, exprs_to_schema};
use daft_schema::schema::SchemaRef;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::{
    LogicalPlan,
    logical_plan::{self},
    stats::{PlanStats, StatsState},
};

/// Grouped aggregations whose rows a later aggregation combines into the whole result.
///
/// Combining its rows for each group with each aggregation's
/// [`daft_dsl::AggExpr::merge_of_partial`] gives the same answer as aggregating its input
/// whole. Unlike [`crate::ops::Aggregate`], it may answer any number of rows for a group,
/// so execution is free to aggregate only part of its input, or none, as it goes.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(debug_assertions, derive(Debug))]
pub struct PartialAggregate {
    pub plan_id: Option<usize>,
    pub node_id: Option<usize>,
    pub input: Arc<LogicalPlan>,
    /// Aggregations, each one that combines with an aggregation of the same kind.
    pub aggregations: Vec<ExprRef>,
    /// Grouping; never empty.
    pub groupby: Vec<ExprRef>,
    /// The grouping columns, then the aggregations.
    pub output_schema: SchemaRef,
    pub stats_state: StatsState,
}

impl PartialAggregate {
    /// A partial aggregation of `input`.
    ///
    /// # Errors
    ///
    /// Returns an error when `groupby` is empty, when an aggregation's partials do not
    /// combine with an aggregation of the same kind, or when an expression cannot be
    /// typed against `input`.
    pub(crate) fn try_new(
        input: Arc<LogicalPlan>,
        aggregations: Vec<ExprRef>,
        groupby: Vec<ExprRef>,
    ) -> logical_plan::Result<Self> {
        if groupby.is_empty() {
            return Err(DaftError::InternalError(
                "A partial aggregation must group by at least one expression".to_string(),
            )
            .into());
        }
        for aggregation in &aggregations {
            let agg = extract_agg_expr(aggregation)?;
            if agg.merge_of_partial(aggregation.clone()).is_none() {
                return Err(DaftError::InternalError(format!(
                    "A partial aggregation cannot hold {agg}, whose partials do not combine with an aggregation of the same kind"
                ))
                .into());
            }
        }
        let output_schema = exprs_to_schema(
            &[groupby.as_slice(), aggregations.as_slice()].concat(),
            input.schema(),
        )?;
        Ok(Self {
            plan_id: None,
            node_id: None,
            input,
            aggregations,
            groupby,
            output_schema,
            stats_state: StatsState::NotMaterialized,
        })
    }

    pub fn with_plan_id(mut self, plan_id: usize) -> Self {
        self.plan_id = Some(plan_id);
        self
    }

    pub fn with_node_id(mut self, node_id: usize) -> Self {
        self.node_id = Some(node_id);
        self
    }

    /// Takes its input's statistics: how far a partial aggregation reduces rows is only
    /// known as it runs, and keeping the input's figures leaves every decision made from
    /// them above (a join's build side, a broadcast) as it was without it.
    pub(crate) fn with_materialized_stats(mut self) -> Self {
        let input_stats = self.input.materialized_stats();
        self.stats_state =
            StatsState::Materialized(PlanStats::new(input_stats.approx_stats.clone()).into());
        self
    }

    pub fn multiline_display(&self) -> Vec<String> {
        let mut res = vec![
            format!(
                "PartialAggregation: {}",
                self.aggregations.iter().map(|e| e.to_string()).join(", ")
            ),
            format!(
                "Group by = {}",
                self.groupby.iter().map(|e| e.to_string()).join(", ")
            ),
            format!("Output schema = {}", self.output_schema.short_string()),
        ];
        if let StatsState::Materialized(stats) = &self.stats_state {
            res.push(format!("Stats = {}", stats));
        }
        res
    }
}
