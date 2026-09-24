use common_error::DaftResult;

use super::{DaftSumAggable, as_arrow::AsArrow};
use crate::{
    array::ops::GroupIndices,
    datatypes::*,
    utils::decimal::{DecimalAnswer, exact_decimal_sum},
};
macro_rules! impl_daft_numeric_agg {
    ($T:ident, $AggType: ty) => {
        impl DaftSumAggable for &DataArray<$T> {
            type Output = DaftResult<DataArray<$T>>;

            fn sum(&self) -> Self::Output {
                let arrow_array = self.as_arrow()?;
                let sum_value = arrow::compute::sum(&arrow_array);
                Ok(DataArray::<$T>::from_iter(
                    self.field.clone(),
                    std::iter::once(sum_value),
                ))
            }

            fn grouped_sum(&self, groups: &GroupIndices) -> Self::Output {
                let sum_per_group = if self.null_count() > 0 {
                    DataArray::<$T>::from_iter(
                        self.field.clone(),
                        groups.iter().map(|g| {
                            g.iter().fold(None, |acc, index| {
                                let idx = *index as usize;
                                match (acc, self.get(idx)) {
                                    (acc, None) => acc,
                                    (None, Some(val)) => Some(val),
                                    (Some(acc), Some(val)) => Some(acc + val),
                                }
                            })
                        }),
                    )
                } else {
                    DataArray::<$T>::from_field_and_values(
                        self.field.clone(),
                        groups.iter().map(|g| {
                            g.iter()
                                .fold(<$AggType as num_traits::Zero>::zero(), |acc, index| {
                                    let idx = *index as usize;
                                    acc + self.get(idx).unwrap()
                                })
                        }),
                    )
                };

                Ok(sum_per_group)
            }
        }
    };
}

impl_daft_numeric_agg!(Int64Type, i64);
impl_daft_numeric_agg!(UInt64Type, u64);
impl_daft_numeric_agg!(Float16Type, half::f16);
impl_daft_numeric_agg!(Float32Type, f32);
impl_daft_numeric_agg!(Float64Type, f64);

/// A decimal sums exactly, and refuses a total its type cannot hold rather than wrapping.
impl DaftSumAggable for &DataArray<Decimal128Type> {
    type Output = DaftResult<DataArray<Decimal128Type>>;

    fn sum(&self) -> Self::Output {
        let answer = DecimalAnswer::of(self.data_type())?;
        let arrow_array = self.as_arrow()?;
        let total = if arrow::array::Array::null_count(&arrow_array) == 0 {
            exact_decimal_sum(arrow_array.values().iter().copied().map(Some), answer)?
        } else {
            exact_decimal_sum(arrow_array.iter(), answer)?
        };
        Ok(Decimal128Array::from_iter(
            self.field.clone(),
            std::iter::once(total),
        ))
    }

    fn grouped_sum(&self, groups: &GroupIndices) -> Self::Output {
        let answer = DecimalAnswer::of(self.data_type())?;
        let totals = groups
            .iter()
            .map(|group| {
                exact_decimal_sum(group.iter().map(|&index| self.get(index as usize)), answer)
            })
            .collect::<DaftResult<Vec<_>>>()?;
        Ok(Decimal128Array::from_iter(self.field.clone(), totals))
    }
}
