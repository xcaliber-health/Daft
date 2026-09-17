use std::sync::Arc;

use common_error::DaftResult;
use daft_core::prelude::{Schema, Series, UInt64Array};
use daft_logical_plan::JoinType;
use daft_micropartition::MicroPartition;
use daft_recordbatch::{GrowableRecordBatch, ProbeState, RecordBatch, get_columns_by_name};

use crate::join::{
    hash_join::{HashJoinParams, HashJoinProbeState},
    index_bitmap::IndexBitmapBuilder,
    outer_join::merge_bitmaps_and_construct_null_table,
    residual::{Candidates, take_build_rows, take_probe_rows},
};

pub(crate) fn probe_left_right_with_bitmap(
    input: &MicroPartition,
    bitmap_builder: &mut IndexBitmapBuilder,
    probe_state: &ProbeState,
    params: &HashJoinParams,
) -> DaftResult<MicroPartition> {
    let build_side_tables = probe_state.get_record_batches().iter().collect::<Vec<_>>();

    let final_tables = input
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
            for (build_table_idx, build_row_idx) in &candidates.build {
                bitmap_builder.mark_used(*build_table_idx as usize, *build_row_idx as usize);
            }
            assemble_outer_build(
                &take_build_rows(&build_side_tables, &candidates.build)?,
                &take_probe_rows(input_table, &candidates.probe)?,
                params,
            )
        })
        .collect::<DaftResult<Vec<_>>>()?;

    Ok(MicroPartition::new_loaded(
        params.output_schema.clone(),
        Arc::new(final_tables),
        None,
    ))
}

/// One row of the join for each pair, where the outer side is the one built on.
fn assemble_outer_build(
    build_side_table: &RecordBatch,
    probe_side_table: &RecordBatch,
    params: &HashJoinParams,
) -> DaftResult<RecordBatch> {
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

    let join_table = get_columns_by_name(build_side_table, &common_join_keys)?;
    if params.join_type == JoinType::Left {
        let left = get_columns_by_name(build_side_table, &left_non_join_columns)?;
        let right = get_columns_by_name(probe_side_table, &right_non_join_columns)?;
        join_table.union(&left)?.union(&right)
    } else {
        let left = get_columns_by_name(probe_side_table, &left_non_join_columns)?;
        let right = get_columns_by_name(build_side_table, &right_non_join_columns)?;
        join_table.union(&left)?.union(&right)
    }
}

pub(crate) fn probe_left_right(
    input: &MicroPartition,
    probe_state: &ProbeState,
    params: &HashJoinParams,
) -> DaftResult<MicroPartition> {
    let build_side_tables = probe_state.get_record_batches().iter().collect::<Vec<_>>();

    let final_tables = input
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

            // Rows of the outer side that kept no pair are still emitted, with the
            // other side left empty.
            let mut build_side_growable = GrowableRecordBatch::new(
                &build_side_tables,
                true,
                build_side_tables.iter().map(|table| table.len()).sum(),
            )?;
            for (build_table_idx, build_row_idx) in &candidates.build {
                build_side_growable.extend(*build_table_idx as usize, *build_row_idx as usize, 1);
            }
            build_side_growable.add_nulls(candidates.unmatched.len());
            let build_side_table = build_side_growable.build()?;

            let mut probe_side_idxs = candidates.probe;
            probe_side_idxs.extend_from_slice(&candidates.unmatched);
            let probe_side_table = input_table.take(&UInt64Array::from_vec("", probe_side_idxs))?;

            assemble_outer_probe(&build_side_table, &probe_side_table, params)
        })
        .collect::<DaftResult<Vec<_>>>()?;

    Ok(MicroPartition::new_loaded(
        params.output_schema.clone(),
        Arc::new(final_tables),
        None,
    ))
}

/// One row of the join for each pair, where the outer side is the one probing.
fn assemble_outer_probe(
    build_side_table: &RecordBatch,
    probe_side_table: &RecordBatch,
    params: &HashJoinParams,
) -> DaftResult<RecordBatch> {
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

    let join_table = get_columns_by_name(probe_side_table, &common_join_keys)?;
    if params.join_type == JoinType::Left {
        let left = get_columns_by_name(probe_side_table, &left_non_join_columns)?;
        let right = get_columns_by_name(build_side_table, &right_non_join_columns)?;
        join_table.union(&left)?.union(&right)
    } else {
        let left = get_columns_by_name(build_side_table, &left_non_join_columns)?;
        let right = get_columns_by_name(probe_side_table, &right_non_join_columns)?;
        join_table.union(&left)?.union(&right)
    }
}

pub(crate) async fn finalize_left(
    states: Vec<HashJoinProbeState>,
    params: &HashJoinParams,
) -> DaftResult<Option<MicroPartition>> {
    let build_side_table = merge_bitmaps_and_construct_null_table(states).await?;

    // If build_side_table is empty, return empty result with correct schema
    if build_side_table.is_empty() {
        return Ok(Some(MicroPartition::empty(Some(
            params.output_schema.clone(),
        ))));
    }

    let common_join_cols: Vec<String> = params.common_join_cols.iter().cloned().collect();
    let left_non_join_columns: Vec<String> = params
        .left_schema
        .field_names()
        .filter(|c| !params.common_join_cols.contains(*c))
        .map(ToString::to_string)
        .collect();
    let right_non_join_schema = Arc::new(Schema::new(
        params
            .right_schema
            .fields()
            .iter()
            .filter(|f| !params.common_join_cols.contains(&*f.name))
            .cloned(),
    ));

    // For left join, we only finalize when build_on_left is true (needs_bitmap check ensures this)
    // So build_side_table has left columns
    let join_table = get_columns_by_name(&build_side_table, &common_join_cols)?;
    let left = get_columns_by_name(&build_side_table, &left_non_join_columns)?;
    let right = {
        let columns = right_non_join_schema
            .fields()
            .iter()
            .map(|field| Series::full_null(&field.name, &field.dtype, left.len()))
            .collect::<Vec<_>>();
        RecordBatch::new_unchecked(right_non_join_schema, columns, left.len())
    };
    let final_table = join_table.union(&left)?.union(&right)?;
    Ok(Some(MicroPartition::new_loaded(
        final_table.schema.clone(),
        Arc::new(vec![final_table]),
        None,
    )))
}

pub(crate) async fn finalize_right(
    states: Vec<HashJoinProbeState>,
    params: &HashJoinParams,
) -> DaftResult<Option<MicroPartition>> {
    let build_side_table = merge_bitmaps_and_construct_null_table(states).await?;

    // If build_side_table is empty, return empty result with correct schema
    if build_side_table.is_empty() {
        return Ok(Some(MicroPartition::empty(Some(
            params.output_schema.clone(),
        ))));
    }

    let common_join_cols: Vec<String> = params.common_join_cols.iter().cloned().collect();
    let right_non_join_columns: Vec<String> = params
        .right_schema
        .field_names()
        .filter(|c| !params.common_join_cols.contains(*c))
        .map(ToString::to_string)
        .collect();
    let left_non_join_schema = Arc::new(Schema::new(
        params
            .left_schema
            .fields()
            .iter()
            .filter(|f| !params.common_join_cols.contains(&*f.name))
            .cloned(),
    ));

    // For right join, we only finalize when build_on_left is false (needs_bitmap check ensures this)
    // So build_side_table has right columns
    let join_table = get_columns_by_name(&build_side_table, &common_join_cols)?;
    let left = {
        let columns = left_non_join_schema
            .fields()
            .iter()
            .map(|field| Series::full_null(&field.name, &field.dtype, build_side_table.len()))
            .collect::<Vec<_>>();
        RecordBatch::new_unchecked(left_non_join_schema, columns, build_side_table.len())
    };
    let right = get_columns_by_name(&build_side_table, &right_non_join_columns)?;
    let final_table = join_table.union(&left)?.union(&right)?;
    Ok(Some(MicroPartition::new_loaded(
        final_table.schema.clone(),
        Arc::new(vec![final_table]),
        None,
    )))
}
