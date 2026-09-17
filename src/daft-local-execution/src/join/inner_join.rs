use std::sync::Arc;

use common_error::DaftResult;
use daft_micropartition::MicroPartition;
use daft_recordbatch::{ProbeState, RecordBatch};

use crate::join::{
    hash_join::HashJoinParams,
    residual::{Candidates, take_build_rows, take_probe_rows},
};

pub(crate) fn probe_inner(
    input: &MicroPartition,
    probe_state: &ProbeState,
    params: &HashJoinParams,
) -> DaftResult<MicroPartition> {
    let build_side_tables = probe_state.get_record_batches().iter().collect::<Vec<_>>();

    let result_tables = input
        .record_batches()
        .iter()
        .map(|input_table| {
            let mut candidates = Candidates::collect(input_table, probe_state, &params.probe_on)?;
            if let Some(residual) = params.residual.as_ref() {
                candidates = candidates.retain(
                    residual,
                    input_table,
                    &build_side_tables,
                    params.build_on_left,
                )?;
            }
            assemble_pair(
                &take_build_rows(&build_side_tables, &candidates.build)?,
                &take_probe_rows(input_table, &candidates.probe)?,
                params,
            )
        })
        .collect::<DaftResult<Vec<_>>>()?;

    Ok(MicroPartition::new_loaded(
        params.output_schema.clone(),
        Arc::new(result_tables),
        None,
    ))
}

/// One row of the join for each pair, with the two sides put in join order.
fn assemble_pair(
    build_side_table: &RecordBatch,
    probe_side_table: &RecordBatch,
    params: &HashJoinParams,
) -> DaftResult<RecordBatch> {
    let (left_table, right_table) = if params.build_on_left {
        (build_side_table, probe_side_table)
    } else {
        (probe_side_table, build_side_table)
    };
    let common_join_keys: Vec<String> = params.common_join_cols.iter().cloned().collect();
    let left_non_join_columns: Vec<String> = params
        .left_schema
        .field_names()
        .filter(|c| !params.common_join_cols.contains(*c))
        .map(ToString::to_string)
        .collect();
    let right_non_join_columns: Vec<String> = params
        .right_schema
        .field_names()
        .filter(|c| !params.common_join_cols.contains(*c))
        .map(ToString::to_string)
        .collect();

    let join_keys_table = daft_recordbatch::get_columns_by_name(left_table, &common_join_keys)?;
    let left_non_join_columns =
        daft_recordbatch::get_columns_by_name(left_table, &left_non_join_columns)?;
    let right_non_join_columns =
        daft_recordbatch::get_columns_by_name(right_table, &right_non_join_columns)?;
    join_keys_table
        .union(&left_non_join_columns)?
        .union(&right_non_join_columns)
}
