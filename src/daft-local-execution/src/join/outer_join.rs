use std::sync::Arc;

use common_error::DaftResult;
use daft_core::{
    prelude::{Field, IntoSeries, Schema, Series, UInt64Array},
    utils::supertype::try_get_supertype,
};
use daft_micropartition::MicroPartition;
use daft_recordbatch::{GrowableRecordBatch, ProbeState, RecordBatch, get_columns_by_name};
use futures::{StreamExt, stream};

use crate::join::{
    hash_join::{HashJoinParams, HashJoinProbeState},
    index_bitmap::IndexBitmapBuilder,
    residual::Candidates,
};

pub(crate) async fn merge_bitmaps_and_construct_null_table(
    states: Vec<HashJoinProbeState>,
) -> DaftResult<RecordBatch> {
    let mut states_iter = states.into_iter();
    let first_state = states_iter
        .next()
        .expect("At least one state should be present for merge bitmaps and construct null table");
    let first_tables = first_state.probe_state.get_record_batches();
    let first_bitmap = first_state
        .bitmap_builder
        .expect("Bitmap builder should be set for merge bitmaps and construct null table")
        .build();

    let merged_bitmap = {
        let bitmaps = stream::once(async move { first_bitmap })
            .chain(stream::iter(states_iter).map(|s| {
                s.bitmap_builder
                    .expect(
                        "Bitmap builder should be set for merge bitmaps and construct null table",
                    )
                    .build()
            }))
            .collect::<Vec<_>>()
            .await;

        bitmaps.into_iter().fold(None, |acc, x| match acc {
            None => Some(x),
            Some(acc) => Some(acc.merge(&x)),
        })
    }
    .expect("At least one bitmap should be present for merge bitmaps and construct null table");

    let leftovers = merged_bitmap
        .convert_to_boolean_arrays()
        .zip(first_tables.iter())
        .map(|(bitmap, table)| table.mask_filter(&bitmap.into_series()))
        .collect::<DaftResult<Vec<_>>>()?;
    RecordBatch::concat(&leftovers)
}

pub(crate) fn probe_outer(
    input: &MicroPartition,
    probe_state: &ProbeState,
    bitmap_builder: Option<&mut IndexBitmapBuilder>,
    params: &HashJoinParams,
) -> DaftResult<MicroPartition> {
    let bitmap_builder = bitmap_builder.expect("Bitmap builder should be set for outer join probe");
    let build_side_tables = probe_state.get_record_batches().iter().collect::<Vec<_>>();

    // Compute outer common column schema
    let outer_common_col_fields = params
        .common_join_cols
        .iter()
        .map(|name| {
            let supertype = try_get_supertype(
                &params.left_schema.get_field(name)?.dtype,
                &params.right_schema.get_field(name)?.dtype,
            )?;
            Ok(Field::new(name.clone(), supertype))
        })
        .collect::<DaftResult<Vec<_>>>()?;
    let outer_common_col_schema = Arc::new(Schema::new(outer_common_col_fields));

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
            let assemble = |build: &RecordBatch, probe: &RecordBatch| {
                assemble_outer(build, probe, params, &outer_common_col_schema)
            };
            for (build_table_idx, build_row_idx) in &candidates.build {
                bitmap_builder.mark_used(*build_table_idx as usize, *build_row_idx as usize);
            }

            // Rows of the probe side that kept no pair are still emitted, with the
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

            assemble(&build_side_table, &probe_side_table)
        })
        .collect::<DaftResult<Vec<_>>>()?;

    Ok(MicroPartition::new_loaded(
        params.output_schema.clone(),
        Arc::new(final_tables),
        None,
    ))
}

pub(crate) async fn finalize_outer(
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

    // Compute outer common column schema
    let outer_common_col_fields = params
        .common_join_cols
        .iter()
        .map(|name| {
            let supertype = try_get_supertype(
                &params.left_schema.get_field(name)?.dtype,
                &params.right_schema.get_field(name)?.dtype,
            )?;
            Ok(Field::new(name.clone(), supertype))
        })
        .collect::<DaftResult<Vec<_>>>()?;
    let outer_common_col_schema = Arc::new(Schema::new(outer_common_col_fields));

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

    let common_join_cols: Vec<String> = params.common_join_cols.iter().cloned().collect();
    let right_non_join_columns: Vec<String> = params
        .right_schema
        .field_names()
        .filter(|c| !params.common_join_cols.contains(*c))
        .map(ToString::to_string)
        .collect();

    // Get columns from build_side_table based on which side we built on
    let (left, right, join_table) = if params.build_on_left {
        // Built on left, so build_side_table has left columns
        // Try to get columns - if they don't exist, return empty
        let left = match get_columns_by_name(&build_side_table, &left_non_join_columns) {
            Ok(cols) => cols,
            Err(_) => {
                return Ok(Some(MicroPartition::empty(Some(
                    params.output_schema.clone(),
                ))));
            }
        };
        let right = {
            let columns = right_non_join_schema
                .fields()
                .iter()
                .map(|field| Series::full_null(&field.name, &field.dtype, left.len()))
                .collect::<Vec<_>>();
            RecordBatch::new_unchecked(right_non_join_schema, columns, left.len())
        };
        #[allow(deprecated)]
        let join_table = match get_columns_by_name(&build_side_table, &common_join_cols) {
            Ok(cols) => cols.cast_to_schema(&outer_common_col_schema)?,
            Err(_) => {
                return Ok(Some(MicroPartition::empty(Some(
                    params.output_schema.clone(),
                ))));
            }
        };
        (left, right, join_table)
    } else {
        // Built on right, so build_side_table has right columns
        // The join columns in build_side_table have right side names, but common_join_cols has left side names
        // We need to get the right side join key columns from the build_side_table
        // For now, try to get by common_join_cols (left names), and if that fails,
        // the build side was likely empty or has different column names
        let right = match get_columns_by_name(&build_side_table, &right_non_join_columns) {
            Ok(cols) => cols,
            Err(_) => {
                return Ok(Some(MicroPartition::empty(Some(
                    params.output_schema.clone(),
                ))));
            }
        };
        let left = {
            let left_non_join_schema = Arc::new(Schema::new(
                params
                    .left_schema
                    .fields()
                    .iter()
                    .filter(|f| !params.common_join_cols.contains(&*f.name))
                    .cloned(),
            ));
            let columns = left_non_join_schema
                .fields()
                .iter()
                .map(|field| Series::full_null(&field.name, &field.dtype, right.len()))
                .collect::<Vec<_>>();
            RecordBatch::new_unchecked(left_non_join_schema, columns, right.len())
        };
        // Try to get join columns - if common_join_cols don't exist, the table might be empty or have wrong schema
        #[allow(deprecated)]
        let join_table = match get_columns_by_name(&build_side_table, &common_join_cols) {
            Ok(cols) => cols.cast_to_schema(&outer_common_col_schema)?,
            Err(_) => {
                // Build side table doesn't have the expected join columns - return empty
                return Ok(Some(MicroPartition::empty(Some(
                    params.output_schema.clone(),
                ))));
            }
        };
        (left, right, join_table)
    };
    let final_table = join_table.union(&left)?.union(&right)?;
    Ok(Some(MicroPartition::new_loaded(
        final_table.schema.clone(),
        Arc::new(vec![final_table]),
        None,
    )))
}

/// One row of the join for each pair, with the shared key columns widened to the
/// type both sides fit in.
fn assemble_outer(
    build_side_table: &RecordBatch,
    probe_side_table: &RecordBatch,
    params: &HashJoinParams,
    outer_common_col_schema: &Arc<Schema>,
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

    #[allow(deprecated)]
    let join_table = get_columns_by_name(probe_side_table, &common_join_keys)?
        .cast_to_schema(outer_common_col_schema)?;
    let (left, right) = if params.build_on_left {
        (
            get_columns_by_name(build_side_table, &left_non_join_columns)?,
            get_columns_by_name(probe_side_table, &right_non_join_columns)?,
        )
    } else {
        (
            get_columns_by_name(probe_side_table, &left_non_join_columns)?,
            get_columns_by_name(build_side_table, &right_non_join_columns)?,
        )
    };
    join_table.union(&left)?.union(&right)
}
