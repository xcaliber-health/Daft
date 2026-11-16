//! Exact decimal arithmetic: collapses (sums, means and percentiles) and the
//! row-wise sum, difference, quotient and product.
//!
//! A decimal is held as an integer of unscaled units at a fixed scale. Every
//! answer here is exact while it fits its type, rounds half away from zero where
//! it has more places than its type keeps, and never wraps.
//!
//! The two families differ where an answer does not fit:
//! - A collapse refuses with an error. It never answers null for values that are
//!   present, because null already means that a group had no values.
//! - A row answers null, as a decimal cast does, and so does a quotient whose
//!   divisor is zero. A row error could not be guarded against, because a
//!   conditional evaluates both of its branches over every row.

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
                "A decimal answer must have a decimal type, not {dtype}"
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

    /// Returns `value` when its magnitude fits this type, and `None` otherwise.
    #[must_use]
    pub fn holds(self, value: i128) -> Option<i128> {
        (value.unsigned_abs() <= self.max.unsigned_abs()).then_some(value)
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

/// Most digits a row-wise scale factor spans. `10^76` is the largest power of ten
/// that fits 256 bits, and it covers every factor two 38-digit decimals can need.
const MAX_FACTOR_DIGITS: usize = 76;

/// A power of ten that carries a value between scales, in 128 bits when it fits.
#[derive(Clone, Copy, Debug)]
struct ScaleFactor {
    narrow: Option<i128>,
    wide: i256,
}

impl ScaleFactor {
    /// `10^digits`.
    fn of(digits: usize) -> DaftResult<Self> {
        let wide = u32::try_from(digits)
            .ok()
            .filter(|_| digits <= MAX_FACTOR_DIGITS)
            .and_then(|digits| i256::from_i128(10).checked_pow(digits))
            .ok_or_else(|| {
                DaftError::InternalError(format!(
                    "A decimal scale factor of 10^{digits} does not fit 256 bits"
                ))
            })?;
        Ok(Self {
            narrow: wide.to_i128(),
            wide,
        })
    }

    /// Whether the factor is one, so that multiplying or dividing by it changes nothing.
    fn is_one(self) -> bool {
        self.narrow == Some(1)
    }
}

/// Divides `numerator` by a nonzero `denominator` in 128 bits, rounding half away from zero.
///
/// It divides magnitudes, biased by half the divisor, so it rounds without a
/// remainder or a branch: `⌊(n + ⌊d/2⌋) / d⌋` rounds up exactly when `2·(n mod d) ≥ d`.
/// When both fit 64 bits, it uses a native 64-bit division, which costs a fraction
/// of a 128-bit one. Answers `None` only for a magnitude of `2^127`, which no
/// decimal reaches.
fn divide_rounding_half_away_narrow(numerator: i128, denominator: i128) -> Option<i128> {
    let divisor = denominator.unsigned_abs();
    // At most 2^127 + 2^126, so the bias cannot overflow 128 bits.
    let biased = numerator.unsigned_abs() + divisor / 2;
    let magnitude = match (u64::try_from(biased), u64::try_from(divisor)) {
        (Ok(biased), Ok(divisor)) => u128::from(biased / divisor),
        _ => biased / divisor,
    };
    let magnitude = i128::try_from(magnitude).ok()?;
    Some(if (numerator ^ denominator) < 0 {
        -magnitude
    } else {
        magnitude
    })
}

/// The quotient of two decimal columns at a result type, row by row.
///
/// The dividend (unscaled, at scale `s1`) and the divisor (at scale `s2`) keep
/// their own scales. The quotient at result scale `s` is `dividend·10^k / divisor`
/// with `k = s − s1 + s2`; when `k` is negative, the divisor is scaled by `10^−k`
/// instead. Each row computes in 128 bits and moves to 256 bits only when a scaled
/// operand overflows 128, so no row wraps.
#[derive(Clone, Copy, Debug)]
pub struct DecimalQuotient {
    dividend_factor: ScaleFactor,
    divisor_factor: ScaleFactor,
    answer: DecimalAnswer,
}

impl DecimalQuotient {
    /// Prepares the quotient of a dividend at `dividend_scale` by a divisor at
    /// `divisor_scale`, answering `answer`.
    ///
    /// # Errors
    ///
    /// Returns [`DaftError::InternalError`] when a scale factor exceeds 256 bits,
    /// which no pair of 38-digit decimals reaches.
    pub fn new(
        dividend_scale: usize,
        divisor_scale: usize,
        answer: DecimalAnswer,
    ) -> DaftResult<Self> {
        let raised = answer.scale + divisor_scale;
        let (dividend_digits, divisor_digits) = match raised.checked_sub(dividend_scale) {
            Some(digits) => (digits, 0),
            None => (0, dividend_scale - raised),
        };
        Ok(Self {
            dividend_factor: ScaleFactor::of(dividend_digits)?,
            divisor_factor: ScaleFactor::of(divisor_digits)?,
            answer,
        })
    }

    /// `dividend / divisor` in unscaled units of the answer, rounded half away from zero.
    ///
    /// Answers `None` when the divisor is zero or the quotient does not fit the answer.
    #[must_use]
    pub fn of(&self, dividend: i128, divisor: i128) -> Option<i128> {
        if divisor == 0 {
            return None;
        }
        let narrow = self
            .dividend_factor
            .narrow
            .and_then(|factor| dividend.checked_mul(factor))
            .zip(
                self.divisor_factor
                    .narrow
                    .and_then(|factor| divisor.checked_mul(factor)),
            );
        let quotient = match narrow {
            Some((numerator, denominator)) => {
                divide_rounding_half_away_narrow(numerator, denominator)?
            }
            None => {
                let numerator = i256::from_i128(dividend).checked_mul(self.dividend_factor.wide)?;
                let denominator = i256::from_i128(divisor).checked_mul(self.divisor_factor.wide)?;
                divide_rounding_half_away(numerator, denominator)
                    .ok()?
                    .to_i128()?
            }
        };
        self.answer.holds(quotient)
    }
}

/// The product of two decimal columns at a result type, row by row.
///
/// The raw product of a value at scale `s1` and one at scale `s2` is already at
/// scale `s1 + s2`. A result scale below that divides it down, rounding half away
/// from zero. Each row computes in 128 bits and moves to 256 bits only when the
/// product overflows 128, so no row wraps.
#[derive(Clone, Copy, Debug)]
pub struct DecimalProduct {
    /// `10^(s1 + s2 − s)`; one when the result keeps every place.
    reduction: ScaleFactor,
    answer: DecimalAnswer,
}

impl DecimalProduct {
    /// Whether every product of a `left` and a `right` decimal, each `(precision,
    /// scale)`, fits `answer` exactly, so that no row needs a check or rounding.
    ///
    /// True when `answer` is the exact product type, `(p1 + p2, s1 + s2)` or wider:
    /// a `p1`-digit value times a `p2`-digit value has at most `p1 + p2` digits, which
    /// also bounds its 128-bit intermediate.
    #[must_use]
    pub fn always_fits(left: (usize, usize), right: (usize, usize), answer: DecimalAnswer) -> bool {
        answer.scale == left.1 + right.1 && answer.precision >= left.0 + right.0
    }

    /// Prepares the product of values at `left_scale` and `right_scale`, answering `answer`.
    ///
    /// # Errors
    ///
    /// Returns [`DaftError::InternalError`] when the answer has more places than the
    /// product, which no result type asks for.
    pub fn new(left_scale: usize, right_scale: usize, answer: DecimalAnswer) -> DaftResult<Self> {
        let product_scale = left_scale + right_scale;
        let reduction = product_scale.checked_sub(answer.scale).ok_or_else(|| {
            DaftError::InternalError(format!(
                "A decimal product at scale {product_scale} cannot answer scale {}",
                answer.scale
            ))
        })?;
        Ok(Self {
            reduction: ScaleFactor::of(reduction)?,
            answer,
        })
    }

    /// `left × right` in unscaled units of the answer, rounded half away from zero.
    ///
    /// Answers `None` when the product does not fit the answer.
    #[must_use]
    pub fn of(&self, left: i128, right: i128) -> Option<i128> {
        let product = match left.checked_mul(right) {
            Some(product) if self.reduction.is_one() => product,
            Some(product) => match self.reduction.narrow {
                Some(reduction) => divide_rounding_half_away_narrow(product, reduction)?,
                None => divide_rounding_half_away(i256::from_i128(product), self.reduction.wide)
                    .ok()?
                    .to_i128()?,
            },
            None => {
                // Two 38-digit values multiply to at most 76 digits, which fits 256 bits.
                let product = i256::from_i128(left).checked_mul(i256::from_i128(right))?;
                divide_rounding_half_away(product, self.reduction.wide)
                    .ok()?
                    .to_i128()?
            }
        };
        self.answer.holds(product)
    }
}

/// The sum or difference of two decimal columns at a result type, row by row.
///
/// Both sides are carried to the larger of their scales, `s_max`, and combined
/// exactly there. A result scale below `s_max` (a capped type) divides the result
/// down once, rounding half away from zero, so it never rounds each side
/// separately. Each row computes in 128 bits, and moves to 256 bits only when the
/// aligned values or their sum overflow 128, so no row wraps.
#[derive(Clone, Copy, Debug)]
pub struct DecimalSum {
    /// `10^(s_max − s1)`.
    left_factor: ScaleFactor,
    /// `10^(s_max − s2)`.
    right_factor: ScaleFactor,
    /// Both sides already share a scale, so they combine without a rescale.
    same_scale: bool,
    /// `10^(s_max − s)`; one when the result keeps every place.
    reduction: ScaleFactor,
    answer: DecimalAnswer,
}

impl DecimalSum {
    /// Whether every sum or difference of a `left` and a `right` decimal, each
    /// `(precision, scale)`, fits `answer` exactly, so that no row needs a check or
    /// rounding.
    ///
    /// True when `answer` is the exact type, `(max(p1 − s1, p2 − s2) + max(s1, s2) + 1,
    /// max(s1, s2))` or wider: both sides then align to that scale and combine within
    /// 38 digits, which also bounds the 128-bit intermediates.
    #[must_use]
    pub fn always_fits(left: (usize, usize), right: (usize, usize), answer: DecimalAnswer) -> bool {
        let scale = left.1.max(right.1);
        let integer_digits = (left.0 - left.1).max(right.0 - right.1) + 1;
        answer.scale == scale && answer.precision >= integer_digits + scale
    }

    /// Prepares the sum of values at `left_scale` and `right_scale`, answering `answer`.
    ///
    /// # Errors
    ///
    /// Returns [`DaftError::InternalError`] when the answer has more places than
    /// either side, which no result type asks for.
    pub fn new(left_scale: usize, right_scale: usize, answer: DecimalAnswer) -> DaftResult<Self> {
        let aligned_scale = left_scale.max(right_scale);
        let reduction = aligned_scale.checked_sub(answer.scale).ok_or_else(|| {
            DaftError::InternalError(format!(
                "A decimal sum at scale {aligned_scale} cannot answer scale {}",
                answer.scale
            ))
        })?;
        Ok(Self {
            left_factor: ScaleFactor::of(aligned_scale - left_scale)?,
            right_factor: ScaleFactor::of(aligned_scale - right_scale)?,
            same_scale: left_scale == right_scale,
            reduction: ScaleFactor::of(reduction)?,
            answer,
        })
    }

    /// `left + right` in unscaled units of the answer, rounded half away from zero.
    ///
    /// Answers `None` when the sum does not fit the answer.
    #[must_use]
    pub fn add(&self, left: i128, right: i128) -> Option<i128> {
        self.combine(left, right, i128::checked_add, i256::checked_add)
    }

    /// `left − right` in unscaled units of the answer, rounded half away from zero.
    ///
    /// Answers `None` when the difference does not fit the answer.
    #[must_use]
    pub fn subtract(&self, left: i128, right: i128) -> Option<i128> {
        self.combine(left, right, i128::checked_sub, i256::checked_sub)
    }

    #[inline]
    fn combine(
        &self,
        left: i128,
        right: i128,
        narrow: fn(i128, i128) -> Option<i128>,
        wide: fn(i256, i256) -> Option<i256>,
    ) -> Option<i128> {
        // Equal scales, the common case, combine directly, as arrow-rs does: skipping
        // the multiply by one is about 40% of the per-row cost.
        let aligned = if self.same_scale {
            narrow(left, right)
        } else {
            self.left_factor
                .narrow
                .and_then(|factor| left.checked_mul(factor))
                .zip(
                    self.right_factor
                        .narrow
                        .and_then(|factor| right.checked_mul(factor)),
                )
                .and_then(|(left, right)| narrow(left, right))
        };
        let combined = match aligned {
            Some(combined) if self.reduction.is_one() => combined,
            Some(combined) => match self.reduction.narrow {
                Some(reduction) => divide_rounding_half_away_narrow(combined, reduction)?,
                None => divide_rounding_half_away(i256::from_i128(combined), self.reduction.wide)
                    .ok()?
                    .to_i128()?,
            },
            None => {
                // Two 38-digit values aligned to a common scale stay within 76 digits.
                let left = i256::from_i128(left).checked_mul(self.left_factor.wide)?;
                let right = i256::from_i128(right).checked_mul(self.right_factor.wide)?;
                divide_rounding_half_away(wide(left, right)?, self.reduction.wide)
                    .ok()?
                    .to_i128()?
            }
        };
        self.answer.holds(combined)
    }
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

    fn quotient(
        dividend_scale: usize,
        divisor_scale: usize,
        answer: DecimalAnswer,
    ) -> DecimalQuotient {
        DecimalQuotient::new(dividend_scale, divisor_scale, answer).expect("a quotient")
    }

    #[rstest]
    // 6.00 / 2.00 and 9.20 / 9.20 at (18, 2) answer (37, 21).
    #[case::three(600, 2, 200, 2, answer(37, 21), 3 * 10_i128.pow(21))]
    #[case::one(920, 2, 920, 2, answer(37, 21), 10_i128.pow(21))]
    #[case::six(150, 2, 25, 2, answer(37, 21), 6 * 10_i128.pow(21))]
    // 1.500000 at (18, 6) / 0.25 at (18, 2) answers (37, 25).
    #[case::mixed_scales(1_500_000, 6, 25, 2, answer(37, 25), 6 * 10_i128.pow(25))]
    // 6.00 · 10^37 overflows 128 bits; the quotient 3 · 10^37 still fits 38 digits.
    #[case::past_128_bits(600, 2, 200, 2, answer(38, 37), 3 * 10_i128.pow(37))]
    // 1.0000000000 / 3 at scale 2: the divisor is scaled, not the dividend.
    #[case::divisor_scaled(10_i128.pow(10), 10, 3, 0, answer(10, 2), 33)]
    fn quotients_are_exact(
        #[case] dividend: i128,
        #[case] dividend_scale: usize,
        #[case] divisor: i128,
        #[case] divisor_scale: usize,
        #[case] answer: DecimalAnswer,
        #[case] expected: i128,
    ) {
        let quotient = quotient(dividend_scale, divisor_scale, answer);

        assert_eq!(quotient.of(dividend, divisor), Some(expected));
    }

    #[rstest]
    // 2 / 3 = 0.66666… → 0.6666666666667 at scale 13.
    #[case::rounds_up(200, 300, 6_666_666_666_667)]
    #[case::rounds_negative_away(-200, 300, -6_666_666_666_667)]
    #[case::rounds_negative_divisor_away(200, -300, -6_666_666_666_667)]
    // 1 / 3 = 0.33333… → 0.3333333333333.
    #[case::rounds_down(100, 300, 3_333_333_333_333)]
    // 1 / 8 = 0.125 exactly.
    #[case::exact(100, 800, 1_250_000_000_000)]
    fn quotients_round_half_away_from_zero(
        #[case] dividend: i128,
        #[case] divisor: i128,
        #[case] expected: i128,
    ) {
        let quotient = quotient(2, 2, answer(21, 13));

        assert_eq!(quotient.of(dividend, divisor), Some(expected));
    }

    #[rstest]
    #[case::zero_divisor(100, 0)]
    // 99999999.99 / 0.01 = 9999999999 needs 23 digits at scale 13.
    #[case::past_the_precision(9_999_999_999, 1)]
    fn quotients_without_an_answer_are_null(#[case] dividend: i128, #[case] divisor: i128) {
        let quotient = quotient(2, 2, answer(21, 13));

        assert_eq!(quotient.of(dividend, divisor), None);
    }

    fn product(left_scale: usize, right_scale: usize, answer: DecimalAnswer) -> DecimalProduct {
        DecimalProduct::new(left_scale, right_scale, answer).expect("a product")
    }

    const NINE_E17: i128 = 900_000_000_000_000_000;

    #[rstest]
    #[case::exact(150, 200, answer(36, 4), 30_000)]
    #[case::negative(-150, 200, answer(36, 4), -30_000)]
    // 9000000000000000.00² = 8.1 · 10^31, at scale 4.
    #[case::large(
        NINE_E17,
        NINE_E17,
        answer(36, 4),
        810_000_000_000_000_000_000_000_000_000_000_000
    )]
    // 1.25 × 0.10 = 0.125 → 0.13; 1.24 × 0.10 = 0.124 → 0.12.
    #[case::reduced_rounds_up(125, 10, answer(38, 2), 13)]
    #[case::reduced_rounds_negative_away(-125, 10, answer(38, 2), -13)]
    #[case::reduced_rounds_down(124, 10, answer(38, 2), 12)]
    fn products_are_exact(
        #[case] left: i128,
        #[case] right: i128,
        #[case] answer: DecimalAnswer,
        #[case] expected: i128,
    ) {
        let product = product(2, 2, answer);

        assert_eq!(product.of(left, right), Some(expected));
    }

    #[test]
    fn a_product_past_128_bits_is_reduced_exactly() {
        // 10^10 · 10^10 at scale 10 each is 10^40 at scale 20; at scale 2 it is 10^22.
        let product = product(10, 10, answer(38, 2));

        assert_eq!(
            product.of(10_i128.pow(20), 10_i128.pow(20)),
            Some(10_i128.pow(22))
        );
    }

    #[rstest]
    // 99.99 × 99.99 = 9998.0001 does not fit (4, 2).
    #[case::past_the_precision(9_999, 9_999, 2, answer(4, 2))]
    // 10^20 · 10^20 at scale 0 needs 41 digits.
    #[case::past_38_digits(10_i128.pow(20), 10_i128.pow(20), 0, answer(38, 0))]
    fn products_without_an_answer_are_null(
        #[case] left: i128,
        #[case] right: i128,
        #[case] scale: usize,
        #[case] answer: DecimalAnswer,
    ) {
        let product = product(scale, scale, answer);

        assert_eq!(product.of(left, right), None);
    }

    #[test]
    fn a_product_cannot_answer_more_places_than_it_has() {
        let refused = DecimalProduct::new(1, 1, answer(38, 3));

        assert!(matches!(refused, Err(DaftError::InternalError(_))));
    }

    fn sum(left_scale: usize, right_scale: usize, answer: DecimalAnswer) -> DecimalSum {
        DecimalSum::new(left_scale, right_scale, answer).expect("a sum")
    }

    #[rstest]
    // 1.50 + 0.25 at scales 2 and 2.
    #[case::same_scales(150, 2, 25, 2, answer(10, 2), 175)]
    // 1.50 at scale 2 + 0.250000 at scale 6 = 1.750000.
    #[case::different_scales(150, 2, 250_000, 6, answer(38, 6), 1_750_000)]
    // 1.50 + (−0.250000) = 1.250000.
    #[case::negative_operand(150, 2, -250_000, 6, answer(38, 6), 1_250_000)]
    // Capped at scale 5: 1.234565 + 0 → 1.23457, half away from zero.
    #[case::lowered_rounds_up(1_234_565, 6, 0, 6, answer(38, 5), 123_457)]
    #[case::lowered_rounds_negative_away(-1_234_565, 6, 0, 6, answer(38, 5), -123_457)]
    #[case::lowered_rounds_down(1_234_564, 6, 0, 6, answer(38, 5), 123_456)]
    fn sums_are_exact(
        #[case] left: i128,
        #[case] left_scale: usize,
        #[case] right: i128,
        #[case] right_scale: usize,
        #[case] answer: DecimalAnswer,
        #[case] expected: i128,
    ) {
        let sum = sum(left_scale, right_scale, answer);

        assert_eq!(sum.add(left, right), Some(expected));
    }

    #[rstest]
    #[case::exact(150, 25, 125)]
    #[case::below_zero(25, 150, -125)]
    fn differences_are_exact(#[case] left: i128, #[case] right: i128, #[case] expected: i128) {
        let sum = sum(2, 2, answer(10, 2));

        assert_eq!(sum.subtract(left, right), Some(expected));
    }

    #[test]
    fn a_sum_past_128_bits_is_reduced_exactly() {
        // (10^38 − 1) at scale 0 aligned to scale 2 overflows 128 bits; at a capped
        // scale 0 the sum of it and 1 is 10^38, which still exceeds 38 digits.
        let sum = sum(0, 2, answer(38, 0));

        assert_eq!(sum.add(MAX_38, 100), None);
        // 10^37 at scale 0 aligns to 10^39 at scale 2, past 128 bits; adding 10^36
        // (10^38 at scale 2) and reducing back to scale 0 answers 1.1 × 10^37 exactly.
        assert_eq!(
            sum.add(10_i128.pow(37), 10_i128.pow(38)),
            Some(11 * 10_i128.pow(36))
        );
    }

    #[rstest]
    // 99.99 + 0.01 does not fit (4, 2).
    #[case::past_the_precision(9_999, 1)]
    // −99.99 − 0.01 does not fit either.
    #[case::past_the_precision_below_zero(-9_999, -1)]
    fn sums_without_an_answer_are_null(#[case] left: i128, #[case] right: i128) {
        let sum = sum(2, 2, answer(4, 2));

        assert_eq!(sum.add(left, right), None);
    }

    #[test]
    fn a_sum_cannot_answer_more_places_than_it_has() {
        let refused = DecimalSum::new(2, 3, answer(38, 4));

        assert!(matches!(refused, Err(DaftError::InternalError(_))));
    }

    #[rstest]
    #[case::exact_type((10, 2), (10, 2), answer(11, 2), true)]
    #[case::wider_type((10, 2), (10, 4), answer(38, 4), true)]
    #[case::one_digit_short((10, 2), (10, 2), answer(10, 2), false)]
    // (18, 2) + (38, 6) is capped at (38, 6): 33 integer digits do not fit 32.
    #[case::capped((18, 2), (38, 6), answer(38, 6), false)]
    #[case::lowered_scale((38, 10), (30, 2), answer(38, 9), false)]
    fn a_sum_always_fits_only_its_exact_type(
        #[case] left: (usize, usize),
        #[case] right: (usize, usize),
        #[case] answer: DecimalAnswer,
        #[case] expected: bool,
    ) {
        assert_eq!(DecimalSum::always_fits(left, right, answer), expected);
    }

    #[rstest]
    #[case::exact_type((18, 2), (18, 2), answer(36, 4), true)]
    #[case::capped((18, 2), (38, 6), answer(38, 6), false)]
    #[case::lowered_scale((38, 4), (23, 2), answer(38, 6), false)]
    fn a_product_always_fits_only_its_exact_type(
        #[case] left: (usize, usize),
        #[case] right: (usize, usize),
        #[case] answer: DecimalAnswer,
        #[case] expected: bool,
    ) {
        assert_eq!(DecimalProduct::always_fits(left, right, answer), expected);
    }
}
