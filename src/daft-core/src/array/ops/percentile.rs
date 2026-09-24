use std::sync::Arc;

use arrow::array::Float64Builder;
use common_error::{DaftError, DaftResult};

use crate::{
    array::{
        ListArray,
        ops::{DaftPercentileAggable, GroupIndices},
    },
    datatypes::{DataType, Decimal128Array, Field, Float64Array},
    utils::{
        decimal::{
            DecimalAnswer, Percentage, exact_decimal_percentile, exact_decimal_percentile_of,
        },
        stats,
    },
};

impl DaftPercentileAggable for Float64Array {
    type Output = DaftResult<Self>;

    fn percentile(&self, percentage: f64) -> Self::Output {
        let mut builder = Float64Builder::with_capacity(1);
        builder.append_option(stats::exact_percentile(self, percentage)?);
        Self::from_arrow(self.field.clone(), Arc::new(builder.finish()))
    }

    fn grouped_percentile(&self, groups: &GroupIndices, percentage: f64) -> Self::Output {
        let mut builder = Float64Builder::with_capacity(groups.len());
        for group in groups {
            let values = group
                .iter()
                .map(|&index| self.get(index as usize))
                .collect();
            builder.append_option(stats::exact_percentile(&values, percentage)?);
        }
        Self::from_arrow(self.field.clone(), Arc::new(builder.finish()))
    }
}

impl DaftPercentileAggable for ListArray {
    type Output = DaftResult<Float64Array>;

    fn percentile(&self, percentage: f64) -> Self::Output {
        let mut row_iter = (0..self.len()).map(|i| i as u64);
        let percentile = percentile_for_rows(self, &mut row_iter, self.len(), percentage)?;

        let mut builder = Float64Builder::with_capacity(1);
        builder.append_option(percentile);
        Float64Array::from_arrow(
            Field::new(self.name(), DataType::Float64),
            Arc::new(builder.finish()),
        )
    }

    fn grouped_percentile(&self, groups: &GroupIndices, percentage: f64) -> Self::Output {
        let mut builder = Float64Builder::with_capacity(groups.len());
        for group in groups {
            let mut row_iter = group.iter().copied();
            builder.append_option(percentile_for_rows(
                self,
                &mut row_iter,
                group.len(),
                percentage,
            )?);
        }
        Float64Array::from_arrow(
            Field::new(self.name(), DataType::Float64),
            Arc::new(builder.finish()),
        )
    }
}

/// Calls `visit` with the child position of every value in `rows` of `list_array`, skipping null rows.
fn visit_child_positions(
    list_array: &ListArray,
    rows: &mut dyn Iterator<Item = u64>,
    mut visit: impl FnMut(usize),
) -> DaftResult<()> {
    let offsets = list_array.offsets();
    for row_idx in rows {
        let row_idx = row_idx as usize;
        if let Some(nulls) = list_array.nulls()
            && !nulls.is_valid(row_idx)
        {
            continue;
        }
        let (Some(&start), Some(&end)) = (offsets.get(row_idx), offsets.get(row_idx + 1)) else {
            return Err(DaftError::InternalError(format!(
                "list row {row_idx} has no offsets in an array of {} rows",
                list_array.len()
            )));
        };
        (start as usize..end as usize).for_each(&mut visit);
    }
    Ok(())
}

fn percentile_for_rows(
    list_array: &ListArray,
    rows: &mut dyn Iterator<Item = u64>,
    capacity: usize,
    percentage: f64,
) -> DaftResult<Option<f64>> {
    let child = list_array.flat_child.f64()?;
    let mut values_builder = Float64Builder::with_capacity(capacity);
    visit_child_positions(list_array, rows, |position| {
        values_builder.append_option(child.get(position));
    })?;

    let values = Float64Array::from_arrow(
        Field::new(list_array.name(), DataType::Float64),
        Arc::new(values_builder.finish()),
    )?;
    stats::exact_percentile(&values, percentage)
}

fn decimal_percentile_for_rows(
    list_array: &ListArray,
    rows: &mut dyn Iterator<Item = u64>,
    percentage: Percentage,
    input_scale: usize,
    answer: DecimalAnswer,
) -> DaftResult<Option<i128>> {
    let child = list_array.flat_child.decimal128()?;
    let mut values = Vec::new();
    visit_child_positions(list_array, rows, |position| {
        values.extend(child.get(position));
    })?;
    exact_decimal_percentile_of(values, percentage, input_scale, answer)
}

/// The scale of the decimal values a percentile reads.
fn decimal_scale(dtype: &DataType) -> DaftResult<usize> {
    match dtype {
        DataType::Decimal128(_, scale) => Ok(*scale),
        other => Err(DaftError::TypeError(format!(
            "A decimal percentile reads decimals, not {other}"
        ))),
    }
}

impl Decimal128Array {
    /// The exact percentile of these decimals, as the `answer` field's decimal type.
    pub fn decimal_percentile(&self, percentage: Percentage, answer: Field) -> DaftResult<Self> {
        let answer_type = DecimalAnswer::of(&answer.dtype)?;
        let percentile = exact_decimal_percentile(
            self,
            percentage,
            decimal_scale(self.data_type())?,
            answer_type,
        )?;
        Ok(Self::from_iter(
            Arc::new(answer),
            std::iter::once(percentile),
        ))
    }

    /// The exact percentile of each group of these decimals, as the `answer` field's decimal type.
    pub fn grouped_decimal_percentile(
        &self,
        groups: &GroupIndices,
        percentage: Percentage,
        answer: Field,
    ) -> DaftResult<Self> {
        let answer_type = DecimalAnswer::of(&answer.dtype)?;
        let input_scale = decimal_scale(self.data_type())?;
        let percentiles = groups
            .iter()
            .map(|group| {
                exact_decimal_percentile(
                    group.iter().map(|&index| self.get(index as usize)),
                    percentage,
                    input_scale,
                    answer_type,
                )
            })
            .collect::<DaftResult<Vec<_>>>()?;
        Ok(Self::from_iter(Arc::new(answer), percentiles))
    }
}

impl ListArray {
    /// The exact percentile of all decimal values in these lists, as the `answer` field's type.
    pub fn decimal_percentile(
        &self,
        percentage: Percentage,
        answer: Field,
    ) -> DaftResult<Decimal128Array> {
        let answer_type = DecimalAnswer::of(&answer.dtype)?;
        let input_scale = decimal_scale(self.child_data_type())?;
        let mut rows = (0..self.len()).map(|i| i as u64);
        let percentile =
            decimal_percentile_for_rows(self, &mut rows, percentage, input_scale, answer_type)?;
        Ok(Decimal128Array::from_iter(
            Arc::new(answer),
            std::iter::once(percentile),
        ))
    }

    /// The exact percentile of each group's decimal list values, as the `answer` field's type.
    pub fn grouped_decimal_percentile(
        &self,
        groups: &GroupIndices,
        percentage: Percentage,
        answer: Field,
    ) -> DaftResult<Decimal128Array> {
        let answer_type = DecimalAnswer::of(&answer.dtype)?;
        let input_scale = decimal_scale(self.child_data_type())?;
        let percentiles = groups
            .iter()
            .map(|group| {
                decimal_percentile_for_rows(
                    self,
                    &mut group.iter().copied(),
                    percentage,
                    input_scale,
                    answer_type,
                )
            })
            .collect::<DaftResult<Vec<_>>>()?;
        Ok(Decimal128Array::from_iter(Arc::new(answer), percentiles))
    }
}
