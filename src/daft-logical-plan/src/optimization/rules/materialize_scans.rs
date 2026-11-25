#[derive(Default, Debug)]
pub struct MaterializeScans {}

impl MaterializeScans {
    pub fn new() -> Self {
        Self {}
    }
}
use std::{collections::HashMap, sync::Arc};

use common_error::DaftResult;
use common_treenode::{Transformed, TreeNode};
use daft_scan::ScanState;

use super::OptimizerRule;
use crate::{LogicalPlan, SourceInfo};

// Materialize scan tasks from scan operators for all physical scans.
//
// Scans of one operator with the same pushdowns are listed once: a part of a plan written
// more than once reads its sources once per occurrence until it is shared, and listing a
// source (a catalog's manifests, a directory) can cost as much as reading a small one. Each
// occurrence still gets its own node; only the listed tasks are shared.
impl OptimizerRule for MaterializeScans {
    fn try_optimize(&self, plan: Arc<LogicalPlan>) -> DaftResult<Transformed<Arc<LogicalPlan>>> {
        let mut listed: HashMap<Arc<LogicalPlan>, LogicalPlan> = HashMap::new();
        plan.transform_up(|node| Self::try_optimize_node(node, &mut listed))
    }
}

impl MaterializeScans {
    fn try_optimize_node(
        plan: Arc<LogicalPlan>,
        listed: &mut HashMap<Arc<LogicalPlan>, LogicalPlan>,
    ) -> DaftResult<Transformed<Arc<LogicalPlan>>> {
        let LogicalPlan::Source(source) = plan.as_ref() else {
            return Ok(Transformed::no(plan));
        };
        let SourceInfo::Physical(physical_scan_info) = source.source_info.as_ref() else {
            return Ok(Transformed::no(plan));
        };
        if matches!(physical_scan_info.scan_state, ScanState::Tasks(_)) {
            return Ok(Transformed::no(plan));
        }
        if let Some(materialized) = listed.get(&plan) {
            // A new node holding the same tasks: rules that key nodes by pointer see two scans.
            return Ok(Transformed::yes(Arc::new(materialized.clone())));
        }
        let materialized: LogicalPlan = source.clone().build_materialized_scan_source()?.into();
        listed.insert(plan, materialized.clone());
        Ok(Transformed::yes(Arc::new(materialized)))
    }
}

#[cfg(test)]
mod tests {
    use common_treenode::TransformedResult;
    use daft_scan::{
        Precision, Pushdowns, ScanOperatorRef, Statistics, test_utils::DummyScanOperator,
    };
    use daft_schema::{dtype::DataType, field::Field, schema::Schema};

    use super::*;
    use crate::{
        stats::StatsState,
        test::{dummy_scan_node_with_pushdowns, dummy_scan_operator},
    };

    fn scan_op_with_stats(stats: Option<Statistics>) -> ScanOperatorRef {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64)]));
        ScanOperatorRef(Arc::new(DummyScanOperator {
            schema,
            num_scan_tasks: 1,
            num_rows_per_task: None,
            supports_count_pushdown_flag: false,
            stats,
            ..Default::default()
        }))
    }

    fn run_materialize(scan_op: ScanOperatorRef) -> Arc<LogicalPlan> {
        let plan = dummy_scan_node_with_pushdowns(scan_op, Pushdowns::default()).build();
        MaterializeScans::new()
            .try_optimize(plan)
            .data()
            .expect("materialize_scans succeeds")
    }

    #[test]
    fn exact_num_rows_short_circuits_stats() {
        let scan_op = scan_op_with_stats(Some(Statistics {
            num_rows: Precision::Exact(42),
            size_bytes: Precision::Exact(1024),
            column_statistics: vec![],
        }));
        let plan = run_materialize(scan_op);
        let LogicalPlan::Source(source) = &*plan else {
            panic!("expected Source at root")
        };
        let stats = match &source.stats_state {
            StatsState::Materialized(s) => s,
            StatsState::NotMaterialized => {
                panic!("Exact source stats should materialize PlanStats during MaterializeScans")
            }
        };
        assert_eq!(stats.approx_stats.num_rows, 42);
        assert_eq!(stats.approx_stats.size_bytes, 1024);
    }

    #[test]
    fn inexact_num_rows_falls_back_to_task_iteration() {
        let scan_op = scan_op_with_stats(Some(Statistics {
            num_rows: Precision::Inexact(42),
            size_bytes: Precision::Absent,
            column_statistics: vec![],
        }));
        let plan = run_materialize(scan_op);
        let LogicalPlan::Source(source) = &*plan else {
            panic!("expected Source at root")
        };
        assert!(
            matches!(source.stats_state, StatsState::NotMaterialized),
            "Inexact stats should not short-circuit — EnrichWithStats runs next"
        );
    }

    #[test]
    fn no_source_stats_preserves_existing_behavior() {
        let scan_op = dummy_scan_operator(vec![Field::new("a", DataType::Int64)]);
        let plan = run_materialize(scan_op);
        let LogicalPlan::Source(source) = &*plan else {
            panic!("expected Source at root")
        };
        assert!(matches!(source.stats_state, StatsState::NotMaterialized));
    }
}
