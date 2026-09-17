use std::sync::Arc;

use common_metrics::ops::{NodeCategory, NodeType};
use daft_dsl::expr::bound_expr::BoundExpr;
use daft_local_plan::{LocalNodeContext, LocalPhysicalPlan};
use daft_logical_plan::{MergeRowsConfig, stats::StatsState};
use daft_schema::schema::SchemaRef;

use super::{DistributedPipelineNode, PipelineNodeImpl, TaskBuilderStream};
use crate::{
    pipeline_node::{ClusteringStrategy, NodeID, PipelineNodeConfig, PipelineNodeContext},
    plan::{PlanConfig, PlanExecutionContext},
};

/// Applies row-level merge rules inside each task.
///
/// Tasks arrive already clustered on the join keys, so every pair of one target
/// row lands in the same task and the one-match-per-target-row check stays exact.
pub(crate) struct MergeRowsNode {
    config: PipelineNodeConfig,
    context: PipelineNodeContext,
    merge_config: MergeRowsConfig<BoundExpr>,
    child: DistributedPipelineNode,
}

impl MergeRowsNode {
    const NODE_NAME: &'static str = "MergeRows";

    pub fn new(
        node_id: NodeID,
        plan_config: &PlanConfig,
        merge_config: MergeRowsConfig<BoundExpr>,
        schema: SchemaRef,
        child: DistributedPipelineNode,
    ) -> Self {
        let context = PipelineNodeContext::new(
            plan_config.query_idx,
            plan_config.query_id.clone(),
            node_id,
            Arc::from(Self::NODE_NAME),
            NodeType::MergeRows,
            NodeCategory::Intermediate,
        );
        let config = PipelineNodeConfig::new(
            schema,
            plan_config.config.clone(),
            ClusteringStrategy::Passthrough { child: &child },
        );
        Self {
            config,
            context,
            merge_config,
            child,
        }
    }
}

impl PipelineNodeImpl for MergeRowsNode {
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
        let mut lines = vec![Self::NODE_NAME.to_string()];
        lines.extend(self.merge_config.multiline_display());
        lines
    }

    fn produce_tasks(
        self: Arc<Self>,
        plan_context: &mut PlanExecutionContext,
    ) -> TaskBuilderStream {
        let input_node = self.child.clone().produce_tasks(plan_context);

        let merge_config = self.merge_config.clone();
        let schema = self.config.schema.clone();
        let node_id = self.node_id();
        input_node.pipeline_instruction(self, move |input| {
            LocalPhysicalPlan::merge_rows(
                input,
                merge_config.clone(),
                schema.clone(),
                StatsState::NotMaterialized,
                LocalNodeContext::new(Some(node_id as usize)),
            )
        })
    }
}
