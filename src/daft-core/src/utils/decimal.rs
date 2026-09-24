//! Exact arithmetic for collapsing decimals: sums, means and percentiles.
//!
//! A decimal is held as an integer of unscaled units at a fixed scale. These
//! collapses answer exactly while the answer fits its type, round half away from
//! zero where the answer has more places than its type keeps, and refuse with an
//! error where the answer does not fit. They never wrap and never answer null
//! for values that are present: null already means that a group had no values.

use arrow::datatypes::i256;
use common_error::{DaftError, DaftResult};

use crate::datatypes::DataType;

/// Most fractional digits a percentage keeps when it is read as a decimal.
///
/// With at most four places added to an answer's scale, every intermediate of the
/// interpolation then fits 256 bits for any 38-digit decimal input.
const PERCENTAGE_MAX_SCALE: usize = 34;

/// Most places a decimal collapse adds to its input's scale.
const MAX_ADDED_SCALE: usize = 4;

/// The decimal type a collapse answers, and the bound its answers must respect.
#[derive(Clone, Copy, Debug)]
pub struct DecimalAnswer {
    precision: usize,
    scale: usize,
    /// The largest magnitude the type holds, `10^precision − 1`.
    max: i128,
}

impl DecimalAnswer {
    /// Describes `dtype`, which must be a decimal.
    ///
    /// # Errors
    ///
    /// Returns [`DaftError::TypeError`] for a type that is not a decimal.
    pub fn of(dtype: &DataType) -> DaftResult<Self> {
        let DataType::Decimal128(precision, scale) = *dtype else {
            return Err(DaftError::TypeError(format!(
                "A decimal collapse must answer a decimal, not {dtype}"
            )));
        };
        let max = u32::try_from(precision)
            .ok()
            .and_then(|digits| 10_i128.checked_pow(digits))
            .map(|bound| bound - 1)
            .ok_or_else(|| {
                DaftError::TypeError(format!(
                    "A decimal's precision must be at most 38, not {precision}"
                ))
            })?;
        Ok(Self {
            precision,
            scale,
            max,
        })
    }

    /// The answer's scale.
    #[must_use]
    pub fn scale(self) -> usize {
        self.scale
    }

    /// Returns `value` as this type's unscaled integer, or refuses when it does not fit.
    ///
    /// # Errors
    ///
    /// Returns [`DaftError::ComputeError`], naming `collapse`, when `value` is out of range.
    pub fn fit(self, value: i256, collapse: &str) -> DaftResult<i128> {
        value
            .to_i128()
            .filter(|value| value.unsigned_abs() <= self.max.unsigned_abs())
            .ok_or_else(|| {
                DaftError::ComputeError(format!(
                    "The {collapse} of these decimals does not fit Decimal({}, {})",
                    self.precision, self.scale
                ))
            })
    }

    /// `10^(answer scale − input scale)`, the factor that carries an input value into the answer's scale.
    fn widening_from(self, input_scale: usize) -> DaftResult<i256> {
        let added = self
            .scale
            .checked_sub(input_scale)
            .filter(|&added| added <= MAX_ADDED_SCALE)
            .ok_or_else(|| {
                DaftError::InternalError(format!(
                    "A decimal collapse cannot answer scale {} from input scale {input_scale}",
                    self.scale
                ))
            })?;
        // At most four places, so the power always fits.
        Ok(i256::from_i128(10_i128.pow(added as u32)))
    }
}

fn intermediate_overflow() -> DaftError {
    DaftError::ComputeError("A decimal collapse overflowed its 256-bit intermediate".to_string())
}

/// Divides `numerator` by `denominator`, rounding half away from zero.
fn divide_rounding_half_away(numerator: i256, denominator: i256) -> DaftResult<i256> {
    let quotient = numerator
        .checked_div(denominator)
        .ok_or_else(intermediate_overflow)?;
    let remainder = numerator
        .checked_rem(denominator)
        .ok_or_else(intermediate_overflow)?;
    let twice_remainder = remainder
        .checked_abs()
        .and_then(|r| r.checked_mul(i256::from_i128(2)))
        .ok_or_else(intermediate_overflow)?;
    let divisor = denominator
        .checked_abs()
        .ok_or_else(intermediate_overflow)?;
    if twice_remainder >= divisor {
        let away_from_zero = numerator.signum().wrapping_mul(denominator.signum());
        quotient
            .checked_add(away_from_zero)
            .ok_or_else(intermediate_overflow)
    } else {
        Ok(quotient)
    }
}

/// Sums unscaled decimal values exactly into `answer`, skipping nulls.
///
/// Answers `None` when there are no values. The running total is kept in 128 bits
/// and moves to 256 bits only if it overflows them, so a total that passes out of
/// range and returns still answers exactly.
///
/// # Errors
///
/// Returns [`DaftError::ComputeError`] when the total does not fit `answer`.
pub fn exact_decimal_sum(
    values: impl IntoIterator<Item = Option<i128>>,
    answer: DecimalAnswer,
) -> DaftResult<Option<i128>> {
    let mut values = values.into_iter().flatten();
    let Some(mut total) = values.next() else {
        return Ok(None);
    };
    while let Some(value) = values.next() {
        match total.checked_add(value) {
            Some(sum) => total = sum,
            None => {
                let mut wide = i256::from_i128(total)
                    .checked_add(i256::from_i128(value))
                    .ok_or_else(intermediate_overflow)?;
                for value in values {
                    wide = wide
                        .checked_add(i256::from_i128(value))
                        .ok_or_else(intermediate_overflow)?;
                }
                return answer.fit(wide, "sum").map(Some);
            }
        }
    }
    answer.fit(i256::from_i128(total), "sum").map(Some)
}

/// Divides a sum at `sum_scale` by `count` into `answer`, rounding half away from zero.
///
/// Answers `None` when there are no values to average.
///
/// # Errors
///
/// Returns [`DaftError::ComputeError`] when the mean does not fit `answer`.
pub fn exact_decimal_mean(
    sum: Option<i128>,
    sum_scale: usize,
    count: u64,
    answer: DecimalAnswer,
) -> DaftResult<Option<i128>> {
    let Some(sum) = sum.filter(|_| count > 0) else {
        return Ok(None);
    };
    let widened = i256::from_i128(sum)
        .checked_mul(answer.widening_from(sum_scale)?)
        .ok_or_else(intermediate_overflow)?;
    let mean = divide_rounding_half_away(widened, i256::from_i128(i128::from(count)))?;
    answer.fit(mean, "mean").map(Some)
}

/// A percentage read as the decimal it prints as.
///
/// A percentage arrives as a binary float, which cannot hold most decimal
/// fractions: `0.9` is stored as 0.90000000000000002220... Reading it back as the
/// shortest decimal that prints the same keeps the value the caller wrote. Digits
/// beyond [`PERCENTAGE_MAX_SCALE`] places are cut.
#[derive(Clone, Copy, Debug)]
pub struct Percentage {
    value: i256,
    denominator: i256,
}

impl Percentage {
    /// Reads `percentage`, a value between 0 and 1.
    ///
    /// # Errors
    ///
    /// Returns [`DaftError::ValueError`] when it cannot be read as a decimal.
    pub fn parse(percentage: f64) -> DaftResult<Self> {
        let printed = percentage.to_string();
        let (whole, fraction) = printed.split_once('.').unwrap_or((printed.as_str(), ""));
        let fraction = &fraction[..fraction.len().min(PERCENTAGE_MAX_SCALE)];
        let value = format!("{whole}{fraction}").parse::<i128>().map_err(|_| {
            DaftError::ValueError(format!(
                "Provided percentile cannot be read as a decimal: {percentage}"
            ))
        })?;
        // At most 34 fractional digits, so the power always fits 256 bits.
        let denominator = i256::from_i128(10)
            .checked_pow(fraction.len() as u32)
            .ok_or_else(intermediate_overflow)?;
        Ok(Self {
            value: i256::from_i128(value),
            denominator,
        })
    }
}

/// Computes a percentile of unscaled decimal values at `input_scale` exactly, into `answer`.
///
/// The rank is `percentage × (n − 1)` and the value is interpolated linearly between
/// the two nearest values in order, at the input's own scale; only the answer is
/// carried to `answer`'s scale, and rounded half away from zero there. Nulls are
/// skipped, and no values answer `None`.
///
/// # Errors
///
/// Returns [`DaftError::ComputeError`] when the percentile does not fit `answer`.
pub fn exact_decimal_percentile(
    values: impl IntoIterator<Item = Option<i128>>,
    percentage: Percentage,
    input_scale: usize,
    answer: DecimalAnswer,
) -> DaftResult<Option<i128>> {
    exact_decimal_percentile_of(
        values.into_iter().flatten().collect(),
        percentage,
        input_scale,
        answer,
    )
}

/// As [`exact_decimal_percentile`], over values already gathered without nulls.
///
/// # Errors
///
/// Returns [`DaftError::ComputeError`] when the percentile does not fit `answer`.
pub fn exact_decimal_percentile_of(
    mut valid_values: Vec<i128>,
    percentage: Percentage,
    input_scale: usize,
    answer: DecimalAnswer,
) -> DaftResult<Option<i128>> {
    let Some(last_index) = valid_values.len().checked_sub(1) else {
        return Ok(None);
    };
    let widening = answer.widening_from(input_scale)?;

    let last_index =
        i256::from_i128(i128::try_from(last_index).map_err(|_| intermediate_overflow())?);
    let rank = percentage
        .value
        .checked_mul(last_index)
        .ok_or_else(intermediate_overflow)?;
    let lower_index = rank
        .checked_div(percentage.denominator)
        .and_then(i256::to_i128)
        .and_then(|index| usize::try_from(index).ok())
        .ok_or_else(intermediate_overflow)?;
    let remainder = rank
        .checked_rem(percentage.denominator)
        .ok_or_else(intermediate_overflow)?;

    let (_, lower_ref, greater_partition) = valid_values.select_nth_unstable(lower_index);
    let lower = i256::from_i128(*lower_ref);
    if remainder == i256::ZERO {
        let exact = lower
            .checked_mul(widening)
            .ok_or_else(intermediate_overflow)?;
        return answer.fit(exact, "percentile").map(Some);
    }
    // A fractional rank lies below the last index, so a greater value exists.
    let upper = greater_partition
        .iter()
        .copied()
        .min()
        .map(i256::from_i128)
        .ok_or_else(intermediate_overflow)?;

    // lower + (remainder / denominator) × (upper − lower), carried to the answer's scale.
    let scaled = lower
        .checked_mul(widening)
        .and_then(|l| l.checked_mul(percentage.denominator))
        .and_then(|l| {
            upper
                .checked_sub(lower)
                .and_then(|spread| spread.checked_mul(remainder))
                .and_then(|part| part.checked_mul(widening))
                .and_then(|part| l.checked_add(part))
        })
        .ok_or_else(intermediate_overflow)?;
    let percentile = divide_rounding_half_away(scaled, percentage.denominator)?;
    answer.fit(percentile, "percentile").map(Some)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn answer(precision: usize, scale: usize) -> DecimalAnswer {
        DecimalAnswer::of(&DataType::Decimal128(precision, scale)).expect("a decimal type")
    }

    const MAX_38: i128 = 10_i128.pow(38) - 1;

    #[rstest]
    #[case::exact(vec![Some(150), Some(250), None], Some(400))]
    #[case::all_null(vec![None, None], None)]
    #[case::empty(vec![], None)]
    // The running total passes i128::MAX and comes back: the answer still fits.
    #[case::out_and_back(vec![Some(MAX_38), Some(MAX_38), Some(-MAX_38)], Some(MAX_38))]
    fn sums_exactly(#[case] values: Vec<Option<i128>>, #[case] expected: Option<i128>) {
        assert_eq!(
            exact_decimal_sum(values, answer(38, 2)).expect("a sum"),
            expected
        );
    }

    #[rstest]
    #[case::past_the_type(vec![Some(MAX_38), Some(1)])]
    #[case::past_128_bits(vec![Some(MAX_38), Some(MAX_38)])]
    fn a_sum_that_does_not_fit_is_refused(#[case] values: Vec<Option<i128>>) {
        let refused = exact_decimal_sum(values, answer(38, 0));

        assert!(
            matches!(refused, Err(DaftError::ComputeError(message)) if message.contains("sum"))
        );
    }

    #[rstest]
    // 5.00 / 3 = 1.666666.. → 1.666667 at scale 6 (half away from zero, as Spark does).
    #[case::rounds_up(500, 3, Some(1_666_667))]
    #[case::rounds_negative_away(-500, 3, Some(-1_666_667))]
    // 1.00 / 8 = 0.125 exactly, carried to scale 6 as 0.125000.
    #[case::exact(100, 8, Some(125_000))]
    // 0.01 / 3 = 0.00333333.. → 0.003333.
    #[case::rounds_down(1, 3, Some(3_333))]
    fn means_round_half_away_from_zero(
        #[case] sum: i128,
        #[case] count: u64,
        #[case] expected: Option<i128>,
    ) {
        assert_eq!(
            exact_decimal_mean(Some(sum), 2, count, answer(38, 6)).expect("a mean"),
            expected
        );
    }

    #[test]
    fn a_mean_of_no_values_is_null() {
        assert_eq!(
            exact_decimal_mean(None, 2, 0, answer(38, 6)).expect("a mean"),
            None
        );
        assert_eq!(
            exact_decimal_mean(Some(0), 2, 0, answer(38, 6)).expect("a mean"),
            None
        );
    }

    #[test]
    fn a_mean_that_does_not_fit_is_refused() {
        // 33 digits before the point at scale 2 need 39 digits at scale 6.
        let x = 10_i128.pow(35) - 1;

        let refused = exact_decimal_mean(Some(x), 2, 1, answer(38, 6));

        assert!(
            matches!(refused, Err(DaftError::ComputeError(message)) if message.contains("mean"))
        );
    }

    fn percentile(values: Vec<i128>, percentage: f64) -> DaftResult<Option<i128>> {
        exact_decimal_percentile(
            values.into_iter().map(Some),
            Percentage::parse(percentage)?,
            2,
            answer(38, 6),
        )
    }

    #[rstest]
    // The median of 1.25 and 2.50 is 1.875, exactly 1.875000.
    #[case::median(vec![125, 250], 0.5, Some(1_875_000))]
    #[case::odd_count(vec![300, 100, 200], 0.5, Some(2_000_000))]
    #[case::lowest(vec![300, 100, 200], 0.0, Some(1_000_000))]
    #[case::highest(vec![300, 100, 200], 1.0, Some(3_000_000))]
    // Rank 0.9 × 10 = 9 exactly; 0.9 taken at its exact binary value would land just past it.
    #[case::percentage_read_as_written((0..=10).map(|v| v * 100).collect(), 0.9, Some(9_000_000))]
    // −1.00 + 0.5 × (0.01 − −1.00) = −0.495 exactly.
    #[case::negative(vec![-100, 1], 0.5, Some(-495_000))]
    // 1/3 of 0.01 is 0.00333.. → 0.003333; 2/3 is 0.00666.. → 0.006667.
    #[case::rounds_down(vec![0, 1, 1, 1], 1.0 / 9.0, Some(3_333))]
    #[case::rounds_up(vec![0, 1, 1], 1.0 / 3.0, Some(6_667))]
    #[case::one_value(vec![42], 0.3, Some(420_000))]
    fn interpolates_exactly(
        #[case] values: Vec<i128>,
        #[case] percentage: f64,
        #[case] expected: Option<i128>,
    ) {
        assert_eq!(
            percentile(values, percentage).expect("a percentile"),
            expected
        );
    }

    #[test]
    fn a_percentile_answers_where_only_its_inputs_are_too_wide_to_widen() {
        // 33 digits before the point: widening either input would not fit, but their median, 0, does.
        let x = 10_i128.pow(35) - 1;

        assert_eq!(percentile(vec![x, -x], 0.5).expect("a percentile"), Some(0));
    }

    #[test]
    fn a_percentile_that_does_not_fit_is_refused() {
        let x = 10_i128.pow(35) - 1;

        let refused = percentile(vec![x, x], 0.5);

        assert!(
            matches!(refused, Err(DaftError::ComputeError(message)) if message.contains("percentile"))
        );
    }

    #[test]
    fn extremes_do_not_overflow_the_intermediates() {
        let answer = answer(38, 38);
        let percentage = Percentage::parse(0.123_456_789).expect("a percentage");

        let exact = exact_decimal_percentile([Some(-MAX_38), Some(MAX_38)], percentage, 38, answer);

        // −(10^38 − 1) + 0.123456789 × 2(10^38 − 1), rounded half away from zero.
        assert_eq!(
            exact.expect("a percentile"),
            Some(-75_308_642_199_999_999_999_999_999_999_999_999_999)
        );
    }

    #[test]
    fn nulls_are_skipped_and_no_values_answer_null() {
        let answer = answer(38, 6);
        let median = Percentage::parse(0.5).expect("a percentage");

        assert_eq!(
            exact_decimal_percentile([None, Some(100), None, Some(300)], median, 2, answer)
                .expect("a percentile"),
            Some(2_000_000)
        );
        assert_eq!(
            exact_decimal_percentile([None, None], median, 2, answer).expect("a percentile"),
            None
        );
    }
}
