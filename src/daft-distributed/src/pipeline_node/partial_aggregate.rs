//! A grouped partial aggregation over each task's rows, with no shuffle.

use std::sync::Arc;

use common_metrics::ops::{NodeCategory, NodeType};
use daft_dsl::expr::bound_expr::{BoundAggExpr, BoundExpr};
use daft_local_plan::{LocalNodeContext, LocalPhysicalPlan};
use daft_logical_plan::stats::StatsState;
use daft_schema::schema::SchemaRef;
use itertools::Itertools;

use super::PipelineNodeImpl;
use crate::{
    pipeline_node::{
        ClusteringStrategy, DistributedPipelineNode, NodeID, PipelineNodeConfig,
        PipelineNodeContext, TaskBuilderStream,
    },
    plan::{PlanConfig, PlanExecutionContext},
};

/// Partially aggregates the rows of each task where they are.
///
/// A partial aggregation may answer any number of rows per group, so each task aggregates
/// its own rows and the aggregation above merges them; no rows move between tasks.
pub(crate) struct PartialAggregateNode {
    config: PipelineNodeConfig,
    context: PipelineNodeContext,
    group_by: Vec<BoundExpr>,
    aggs: Vec<BoundAggExpr>,
    child: DistributedPipelineNode,
}

impl PartialAggregateNode {
    const NODE_NAME: &'static str = "PartialGroupedAggregate";

    pub fn new(
        node_id: NodeID,
        plan_config: &PlanConfig,
        group_by: Vec<BoundExpr>,
        aggs: Vec<BoundAggExpr>,
        output_schema: SchemaRef,
        child: DistributedPipelineNode,
    ) -> Self {
        let context = PipelineNodeContext::new(
            plan_config.query_idx,
            plan_config.query_id.clone(),
            node_id,
            Arc::from(Self::NODE_NAME),
            NodeType::PartialGroupByAgg,
            NodeCategory::StreamingSink,
        );
        let config = PipelineNodeConfig::new(
            output_schema,
            plan_config.config.clone(),
            ClusteringStrategy::Passthrough { child: &child },
        );
        Self {
            config,
            context,
            group_by,
            aggs,
            child,
        }
    }
}

impl PipelineNodeImpl for PartialAggregateNode {
    fn context(&self) -> &PipelineNodeContext {
        &self.context
    }

    fn config(&self) -> &PipelineNodeConfig {
        &self.config
    }

    fn children(&self) -> Vec<DistributedPipelineNode> {
        vec![self.child.clone()]
    }

    fn multiline_display(&self, _verbose: bool) -> Vec<String> {
        vec![
            format!(
                "Partial Group-By Aggregate: {}",
                self.group_by.iter().map(ToString::to_string).join(", ")
            ),
            format!(
                "Aggregations = {}",
                self.aggs.iter().map(ToString::to_string).join(", ")
            ),
            format!("Output Schema = {}", self.config.schema.short_string()),
        ]
    }

    fn produce_tasks(
        self: Arc<Self>,
        plan_context: &mut PlanExecutionContext,
    ) -> TaskBuilderStream {
        let input_node = self.child.clone().produce_tasks(plan_context);
        let this = self.clone();
        input_node.pipeline_instruction(self, move |input| {
            LocalPhysicalPlan::partial_hash_aggregate(
                input,
                this.aggs.clone(),
                this.group_by.clone(),
                this.config.schema.clone(),
                StatsState::NotMaterialized,
                LocalNodeContext::new(Some(this.node_id() as usize)),
            )
        })
    }
}
