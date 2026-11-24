//! A part of a plan that several readers read, computed once.
//!
//! The same part can appear in a plan more than once, for example when one set
//! of records is both counted and joined to. Each occurrence is wrapped in a
//! [`Shared`] node carrying the same id, and the local executor computes the
//! part once and hands its rows to every occurrence.
//!
//! A shared node is a barrier to every rule that moves work across a node:
//! nothing above one occurrence may change what the part computes, since the
//! other occurrences read the same rows. Work enters the part only through
//! the rule that pushes what every reader needs into every occurrence alike,
//! so all occurrences of an id always compute the same rows.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{LogicalPlan, stats::StatsState};

/// One occurrence of a part of the plan that is computed once for all of its
/// readers.
///
/// Every occurrence of an id computes the same rows; they may differ in how
/// (for example in their join order), and the executor runs one of them.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(debug_assertions, derive(Debug))]
pub struct Shared {
    pub plan_id: Option<usize>,
    pub node_id: Option<usize>,
    /// Identifies the part; every occurrence of the part carries the same id.
    pub id: usize,
    /// The part itself.
    pub input: Arc<LogicalPlan>,
    pub stats_state: StatsState,
}

impl Shared {
    /// Wraps one occurrence of the part `id` computes.
    #[must_use]
    pub fn new(input: Arc<LogicalPlan>, id: usize) -> Self {
        Self {
            plan_id: None,
            node_id: None,
            id,
            input,
            stats_state: StatsState::NotMaterialized,
        }
    }

    /// Sets the id of the plan the node belongs to.
    #[must_use]
    pub fn with_plan_id(mut self, plan_id: usize) -> Self {
        self.plan_id = Some(plan_id);
        self
    }

    /// Sets the node's id within its plan.
    #[must_use]
    pub fn with_node_id(mut self, node_id: usize) -> Self {
        self.node_id = Some(node_id);
        self
    }

    /// Takes its input's statistics: a shared part passes its rows through.
    #[must_use]
    pub(crate) fn with_materialized_stats(mut self) -> Self {
        let input_stats = self.input.materialized_stats();
        self.stats_state = StatsState::Materialized(input_stats.clone().into());
        self
    }

    /// The node's lines in a plan's display.
    #[must_use]
    pub fn multiline_display(&self) -> Vec<String> {
        let mut res = vec![format!("Shared #{}", self.id)];
        if let StatsState::Materialized(stats) = &self.stats_state {
            res.push(format!("Stats = {stats}"));
        }
        res
    }
}
