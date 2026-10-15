use std::{collections::BTreeSet, sync::Arc};

use arrow::{array::Array, buffer::BooleanBuffer};
use common_error::DaftResult;
use common_hashable_float_wrapper::FloatWrapper;

use super::{DaftIsIn, as_arrow::AsArrow, full::FullNull};
use crate::{
    array::{DataArray, prelude::*},
    datatypes::prelude::*,
};

/// Whether each value of `$self` is one of the non-null values of `$rhs`;
/// null where `$self` is null.
///
/// The membership bits are written straight into a bitmap, one per row, and
/// the input's validity is reused as the result's, so a row costs one hash
/// lookup and no per-row allocation.
macro_rules! collect_to_set_and_check_membership {
    ($self:expr, $rhs:expr) => {{
        let rhs_arrow = $rhs.as_arrow()?;
        let set = rhs_arrow
            .iter()
            .flatten()
            .collect::<hashbrown::HashSet<_>>();

        let self_arrow = $self.as_arrow()?;
        let members = BooleanBuffer::collect_bool(self_arrow.len(), |row| {
            set.contains(&self_arrow.value(row))
        });
        let mask = arrow::array::BooleanArray::new(members, self_arrow.nulls().cloned());
        BooleanArray::from_arrow(Field::new($self.name(), DataType::Boolean), Arc::new(mask))
    }};
}

impl<T> DaftIsIn<&Self> for DataArray<T>
where
    T: DaftIntegerType,
    <T as DaftNumericType>::Native: Ord + std::hash::Hash,
    <<<T as DaftNumericType>::Native as crate::datatypes::NumericNative>::ARROWTYPE as arrow::array::ArrowPrimitiveType>::Native:
        std::hash::Hash + std::cmp::Eq,
{
    type Output = DaftResult<BooleanArray>;

    fn is_in(&self, rhs: &Self) -> Self::Output {
        collect_to_set_and_check_membership!(self, rhs)
    }
}

macro_rules! impl_is_in_floating_array {
    ($arr:ident, $T:ty) => {
        impl DaftIsIn<&$arr> for $arr {
            type Output = DaftResult<BooleanArray>;

            fn is_in(&self, rhs: &$arr) -> Self::Output {
                let rhs_arrow = rhs.as_arrow()?;
                let set = rhs_arrow
                    .iter()
                    .filter_map(|item| item.map(FloatWrapper))
                    .collect::<BTreeSet<FloatWrapper<$T>>>();

                let self_arrow = self.as_arrow()?;
                let result = self_arrow
                    .iter()
                    .map(|option| option.map(|value| set.contains(&FloatWrapper(value))));

                let bools: BooleanArray = result.collect();
                Ok(bools.rename(self.name()))
            }
        }
    };
}
impl_is_in_floating_array!(Float16Array, half::f16);
impl_is_in_floating_array!(Float32Array, f32);
impl_is_in_floating_array!(Float64Array, f64);

macro_rules! impl_is_in_non_numeric_array {
    ($arr:ident) => {
        impl DaftIsIn<&$arr> for $arr {
            type Output = DaftResult<BooleanArray>;

            fn is_in(&self, rhs: &$arr) -> Self::Output {
                collect_to_set_and_check_membership!(self, rhs)
            }
        }
    };
}

impl_is_in_non_numeric_array!(Decimal128Array);
impl_is_in_non_numeric_array!(BooleanArray);
impl_is_in_non_numeric_array!(Utf8Array);
impl_is_in_non_numeric_array!(BinaryArray);
impl_is_in_non_numeric_array!(FixedSizeBinaryArray);

impl DaftIsIn<&Self> for NullArray {
    type Output = DaftResult<BooleanArray>;

    fn is_in(&self, _rhs: &Self) -> Self::Output {
        // If self and rhs are null array then return a full null array
        Ok(BooleanArray::full_null(
            self.name(),
            &DataType::Boolean,
            self.len(),
        ))
    }
}
