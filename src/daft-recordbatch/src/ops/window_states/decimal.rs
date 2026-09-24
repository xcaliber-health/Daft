//! Window frames over decimals: sums and means that stay exact.
//!
//! A frame's running total is kept in 256 bits, so adding and removing values can
//! never wrap. Each row's answer is checked against its decimal type, and a total or
//! mean that does not fit is refused, as a decimal collapse outside a window is.

use arrow::{array::NullBufferBuilder, datatypes::i256};
use common_error::DaftResult;
use daft_core::{prelude::*, utils::decimal::DecimalAnswer};

use super::{CountWindowState, WindowAggStateOps};

/// The running sum of a frame of decimals, answered in the source's own decimal type.
pub struct DecimalSumWindowState {
    source: Decimal128Array,
    answer: DecimalAnswer,
    sum: i256,
    valid_count: usize,
    sums: Vec<i128>,
    nulls: NullBufferBuilder,
}

impl DecimalSumWindowState {
    /// Creates the state over `source`, whose decimal type is the sum's answer type.
    pub fn new(source: &Series, total_length: usize) -> DaftResult<Self> {
        Ok(Self {
            source: source.decimal128()?.clone(),
            answer: DecimalAnswer::of(source.data_type())?,
            sum: i256::ZERO,
            valid_count: 0,
            sums: Vec::with_capacity(total_length),
            nulls: NullBufferBuilder::new(total_length),
        })
    }
}

impl WindowAggStateOps for DecimalSumWindowState {
    fn add(&mut self, start_idx: usize, end_idx: usize) -> DaftResult<()> {
        for value in (start_idx..end_idx).filter_map(|i| self.source.get(i)) {
            // 256 bits hold far more 38-digit values than a frame can, so this cannot wrap.
            self.sum = self.sum.wrapping_add(i256::from_i128(value));
            self.valid_count += 1;
        }
        Ok(())
    }

    fn remove(&mut self, start_idx: usize, end_idx: usize) -> DaftResult<()> {
        for value in (start_idx..end_idx).filter_map(|i| self.source.get(i)) {
            self.sum = self.sum.wrapping_sub(i256::from_i128(value));
            self.valid_count -= 1;
        }
        Ok(())
    }

    fn evaluate(&mut self) -> DaftResult<()> {
        if self.valid_count > 0 {
            self.sums.push(self.answer.fit(self.sum, "sum")?);
            self.nulls.append_non_null();
        } else {
            self.sums.push(0);
            self.nulls.append_null();
        }
        Ok(())
    }

    fn build(&self) -> DaftResult<Series> {
        Decimal128Array::from_field_and_values(
            Field::new(self.source.name(), self.source.data_type().clone()),
            self.sums.iter().copied(),
        )
        .into_series()
        .with_nulls(self.nulls.finish_cloned())
    }
}

/// The running mean of a frame of decimals: its exact sum divided by its count.
pub struct DecimalMeanWindowState {
    sum: DecimalSumWindowState,
    count: CountWindowState,
    answer: Field,
}

impl DecimalMeanWindowState {
    /// Creates the state over `source`, at the decimal sum type, answering `answer_type`.
    pub fn new(source: &Series, answer_type: DataType, total_length: usize) -> DaftResult<Self> {
        Ok(Self {
            sum: DecimalSumWindowState::new(source, total_length)?,
            count: CountWindowState::new(source, total_length, CountMode::Valid),
            answer: Field::new(source.name(), answer_type),
        })
    }
}

impl WindowAggStateOps for DecimalMeanWindowState {
    fn add(&mut self, start_idx: usize, end_idx: usize) -> DaftResult<()> {
        self.sum.add(start_idx, end_idx)?;
        self.count.add(start_idx, end_idx)
    }

    fn remove(&mut self, start_idx: usize, end_idx: usize) -> DaftResult<()> {
        self.sum.remove(start_idx, end_idx)?;
        self.count.remove(start_idx, end_idx)
    }

    fn evaluate(&mut self) -> DaftResult<()> {
        self.sum.evaluate()?;
        self.count.evaluate()
    }

    fn build(&self) -> DaftResult<Series> {
        let sums = self.sum.build()?;
        let counts = self.count.build()?;
        Ok(sums
            .decimal128()?
            .merge_mean(counts.u64()?, self.answer.clone())?
            .into_series())
    }
}

#[cfg(test)]
mod tests {
    use common_error::DaftResult;
    use daft_core::prelude::*;

    use super::{DecimalMeanWindowState, DecimalSumWindowState};
    use crate::ops::window_states::WindowAggStateOps;

    fn decimals(values: &[Option<i128>], precision: usize, scale: usize) -> Series {
        Decimal128Array::from_iter(
            Field::new("v", DataType::Decimal128(precision, scale)),
            values.iter().copied(),
        )
        .into_series()
    }

    /// Runs a running frame (unbounded preceding to the current row) over every row.
    fn running(state: &mut dyn WindowAggStateOps, rows: usize) -> DaftResult<Series> {
        for row in 0..rows {
            state.add(row, row + 1)?;
            state.evaluate()?;
        }
        state.build()
    }

    #[test]
    fn a_running_sum_that_does_not_fit_is_refused() -> DaftResult<()> {
        let max = 10_i128.pow(38) - 1;
        let source = decimals(&[Some(max), Some(max)], 38, 0);
        let mut state = DecimalSumWindowState::new(&source, 2)?;

        let refused = running(&mut state, 2);

        assert!(refused.is_err());
        Ok(())
    }

    #[test]
    fn a_sliding_sum_passes_through_128_bits_and_back() -> DaftResult<()> {
        let max = 10_i128.pow(38) - 1;
        let source = decimals(&[Some(max), Some(max), Some(5)], 38, 0);
        let mut state = DecimalSumWindowState::new(&source, 2)?;

        state.add(0, 1)?;
        state.evaluate()?;
        // Adding before removing takes the total past i128::MAX for a moment.
        state.add(1, 2)?;
        state.remove(0, 1)?;
        state.evaluate()?;

        let sums: Vec<Option<i128>> = state.build()?.decimal128()?.into_iter().collect();
        assert_eq!(sums, vec![Some(max), Some(max)]);
        Ok(())
    }

    #[test]
    fn a_running_mean_rounds_half_away_from_zero() -> DaftResult<()> {
        let source = decimals(&[Some(100), Some(200), Some(200), None], 10, 2);
        let mut state = DecimalMeanWindowState::new(&source, DataType::Decimal128(38, 6), 4)?;

        let means: Vec<Option<i128>> = running(&mut state, 4)?.decimal128()?.into_iter().collect();

        // 1.00, 1.50, 1.666666.. → 1.666667, and the null row keeps the frame's mean.
        assert_eq!(
            means,
            vec![
                Some(1_000_000),
                Some(1_500_000),
                Some(1_666_667),
                Some(1_666_667)
            ]
        );
        Ok(())
    }
}
