use std::sync::Arc;

use daft_core::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    LogicalPlan,
    logical_plan::{self},
    merge_info::MergeRowsConfig,
    stats::StatsState,
};

/// Turns a joined stream of target and source rows into the rows a row-level
/// write applies: kept, replaced, removed and added rows, each tagged with what
/// it is.
#[derive(Hash, Eq, PartialEq, Clone, Serialize, Deserialize)]
#[cfg_attr(debug_assertions, derive(Debug))]
pub struct MergeRows {
    pub plan_id: Option<usize>,
    pub node_id: Option<usize>,
    pub input: Arc<LogicalPlan>,
    pub schema: Arc<Schema>,
    pub config: MergeRowsConfig,
    pub stats_state: StatsState,
}

impl MergeRows {
    pub(crate) fn try_new(
        input: Arc<LogicalPlan>,
        config: MergeRowsConfig,
    ) -> logical_plan::Result<Self> {
        let schema = config.output_schema(&input.schema())?;
        Ok(Self {
            plan_id: None,
            node_id: None,
            input,
            schema,
            config,
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

    pub(crate) fn with_materialized_stats(mut self) -> Self {
        // A merge emits at most one row per input pair, and inserts add none of
        // their own, so the input's size is the bound.
        let input_stats = self.input.materialized_stats();
        self.stats_state = StatsState::Materialized(input_stats.clone().into());
        self
    }

    pub fn multiline_display(&self) -> Vec<String> {
        let mut res = vec!["MergeRows".to_string()];
        res.extend(self.config.multiline_display());
        if let StatsState::Materialized(stats) = &self.stats_state {
            res.push(format!("Stats = {}", stats));
        }
        res
    }
}
