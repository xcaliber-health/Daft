use std::sync::Arc;

use common_error::{DaftError, DaftResult};

use crate::{
    array::ops::{DaftMeanAggable, GroupIndices},
    datatypes::*,
    utils::{
        decimal::{DecimalAnswer, exact_decimal_mean},
        stats,
    },
};

impl DaftMeanAggable for DataArray<Float64Type> {
    type Output = DaftResult<Self>;

    fn mean(&self) -> Self::Output {
        let stats = stats::calculate_stats(self)?;

        let field = Arc::new(Field::new(self.name(), DataType::Float64));
        Ok(Self::from_iter(field, std::iter::once(stats.mean)))
    }

    fn grouped_mean(&self, groups: &GroupIndices) -> Self::Output {
        let grouped_means = stats::grouped_stats(self, groups)?.map(|(stats, _)| stats.mean);

        let field = Arc::new(Field::new(self.name(), DataType::Float64));
        Ok(Self::from_iter(field, grouped_means))
    }
}

impl DataArray<Decimal128Type> {
    /// Divides these sums by `counts` into the `answer` field's decimal type.
    ///
    /// Each sum stays at its own scale until it is divided, so a sum is never widened
    /// out of range before the division brings it back. The mean is rounded half away
    /// from zero; a group without values answers null.
    ///
    /// # Errors
    ///
    /// Returns a compute error when a mean does not fit `answer`.
    pub fn merge_mean(&self, counts: &DataArray<UInt64Type>, answer: Field) -> DaftResult<Self> {
        let DataType::Decimal128(_, sum_scale) = *self.data_type() else {
            return Err(DaftError::TypeError(format!(
                "A decimal mean divides decimal sums, not {}",
                self.data_type()
            )));
        };
        let answer_type = DecimalAnswer::of(&answer.dtype)?;
        let means = self
            .into_iter()
            .zip(counts)
            .map(|(sum, count)| exact_decimal_mean(sum, sum_scale, count.unwrap_or(0), answer_type))
            .collect::<DaftResult<Vec<_>>>()?;
        Ok(Self::from_iter(Arc::new(answer), means))
    }
}
