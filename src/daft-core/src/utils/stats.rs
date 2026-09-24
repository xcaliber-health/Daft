use arrow::datatypes::i256;
use common_error::{DaftError, DaftResult};

use crate::{
    array::{
        ops::{DaftCountAggable, DaftSumAggable, GroupIndices, VecIndices},
        prelude::{Float64Array, UInt64Array},
    },
    count_mode::CountMode,
};

#[derive(Clone, Copy, Default, Debug)]
pub struct Stats {
    pub sum: f64,
    pub count: f64,
    pub mean: Option<f64>,
}

pub fn calculate_stats(array: &Float64Array) -> DaftResult<Stats> {
    let sum = array.sum()?.get(0);
    let count = array.count(CountMode::Valid)?.get(0);
    let stats = sum
        .zip(count)
        .map_or_else(Default::default, |(sum, count)| Stats {
            sum,
            count: count as _,
            mean: calculate_mean(sum, count),
        });
    Ok(stats)
}

pub fn exact_percentile(values: &Float64Array, percentage: f64) -> DaftResult<Option<f64>> {
    let mut valid_values: Vec<f64> = values.into_iter().flatten().collect();

    if valid_values.is_empty() {
        return Ok(None);
    }

    let rank = percentage * (valid_values.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;

    let (_, lower_ref, greater_partition) =
        valid_values.select_nth_unstable_by(lower, f64::total_cmp);
    let lower_value = *lower_ref;

    if lower == upper {
        Ok(Some(lower_value))
    } else {
        // upper == lower + 1, so upper_value is the min of the greater partition.
        let upper_value = greater_partition
            .iter()
            .copied()
            .min_by(f64::total_cmp)
            .unwrap();
        let weight = rank - lower as f64;
        let percentile = (upper_value - lower_value).mul_add(weight, lower_value);
        Ok(Some(percentile))
    }
}

/// Most fractional digits a percentage keeps when it is read as a decimal.
///
/// Every intermediate of the interpolation then fits 256 bits for any
/// 38-digit decimal input.
const PERCENTAGE_MAX_SCALE: usize = 38;

/// Reads `percentage` as the decimal it prints as, returning its digits and scale.
///
/// A percentage arrives as a binary float, which cannot hold most decimal
/// fractions: `0.9` is stored as 0.90000000000000002220... Reading it back as the
/// shortest decimal that prints the same keeps the value the caller wrote.
/// Digits beyond [`PERCENTAGE_MAX_SCALE`] places are cut.
fn percentage_as_decimal(percentage: f64) -> DaftResult<(i256, u32)> {
    let printed = percentage.to_string();
    let (whole, fraction) = printed.split_once('.').unwrap_or((printed.as_str(), ""));
    let fraction = &fraction[..fraction.len().min(PERCENTAGE_MAX_SCALE)];
    let digits = format!("{whole}{fraction}");
    let value = digits.parse::<i128>().map_err(|_| {
        DaftError::ValueError(format!(
            "Provided percentile cannot be read as a decimal: {percentage}"
        ))
    })?;
    // At most 38 fractional digits, so the scale always fits a `u32`.
    let scale = u32::try_from(fraction.len()).unwrap_or(u32::MAX);
    Ok((i256::from_i128(value), scale))
}

fn overflow() -> DaftError {
    DaftError::ComputeError("Decimal percentile overflowed its intermediate precision".to_string())
}

/// Computes a percentile of decimal values exactly, by linear interpolation.
///
/// `values` are the unscaled integers of one decimal type; the answer is an
/// unscaled integer of that same type. The rank is `percentage × (n − 1)`, taken
/// with `percentage` read as the decimal it prints as, and the interpolated value
/// `lower + (rank − ⌊rank⌋) × (upper − lower)` is cut toward zero to the type's
/// scale, as a decimal mean is. Nulls are skipped; no values answer `None`.
pub fn exact_decimal_percentile(
    values: impl IntoIterator<Item = Option<i128>>,
    percentage: f64,
) -> DaftResult<Option<i128>> {
    let mut valid_values: Vec<i128> = values.into_iter().flatten().collect();
    if valid_values.is_empty() {
        return Ok(None);
    }

    let (percentage, scale) = percentage_as_decimal(percentage)?;
    let denominator = i256::from_i128(10)
        .checked_pow(scale)
        .ok_or_else(overflow)?;
    let last_index =
        i256::from_i128(i128::try_from(valid_values.len() - 1).map_err(|_| overflow())?);
    let rank = percentage.checked_mul(last_index).ok_or_else(overflow)?;
    let lower_index = rank
        .checked_div(denominator)
        .and_then(|index| index.to_i128())
        .and_then(|index| usize::try_from(index).ok())
        .ok_or_else(overflow)?;
    let remainder = rank.checked_rem(denominator).ok_or_else(overflow)?;

    let (_, lower_ref, greater_partition) = valid_values.select_nth_unstable(lower_index);
    let lower_value = *lower_ref;
    if remainder == i256::ZERO {
        return Ok(Some(lower_value));
    }
    // A fractional rank lies below the last index, so a greater value exists.
    let upper_value = greater_partition
        .iter()
        .copied()
        .min()
        .ok_or_else(overflow)?;

    let lower = i256::from_i128(lower_value);
    let spread = i256::from_i128(upper_value)
        .checked_sub(lower)
        .ok_or_else(overflow)?;
    let exact = lower
        .checked_mul(denominator)
        .and_then(|scaled| {
            spread
                .checked_mul(remainder)
                .and_then(|part| scaled.checked_add(part))
        })
        .ok_or_else(overflow)?;
    // Division truncates toward zero, as the decimal mean does.
    exact
        .checked_div(denominator)
        .and_then(|value| value.to_i128())
        .map(Some)
        .ok_or_else(overflow)
}

pub fn is_valid_percentile_percentage(percentage: f64) -> bool {
    (0.0..=1.0).contains(&percentage)
}

pub fn grouped_stats<'a>(
    array: &Float64Array,
    groups: &'a GroupIndices,
) -> DaftResult<impl Iterator<Item = (Stats, &'a VecIndices)>> {
    let grouped_sum = array.grouped_sum(groups)?;
    let grouped_count = array.grouped_count(groups, CountMode::Valid)?;
    debug_assert_eq!(grouped_sum.len(), grouped_count.len());
    debug_assert_eq!(grouped_sum.len(), groups.len());
    Ok(GroupedStats {
        grouped_sum,
        grouped_count,
        groups: groups.iter().enumerate(),
    })
}

struct GroupedStats<'a, I: Iterator<Item = (usize, &'a VecIndices)>> {
    grouped_sum: Float64Array,
    grouped_count: UInt64Array,
    groups: I,
}

impl<'a, I: Iterator<Item = (usize, &'a VecIndices)>> Iterator for GroupedStats<'a, I> {
    type Item = (Stats, &'a VecIndices);

    fn next(&mut self) -> Option<Self::Item> {
        let (index, group) = self.groups.next()?;
        let sum = self.grouped_sum.get(index);
        let count = self.grouped_count.get(index);
        let stats = sum
            .zip(count)
            .map_or_else(Default::default, |(sum, count)| Stats {
                sum,
                count: count as _,
                mean: calculate_mean(sum, count),
            });
        Some((stats, group))
    }
}

pub fn calculate_mean(sum: f64, count: u64) -> Option<f64> {
    match count {
        0 => None,
        _ => Some(sum / count as f64),
    }
}

pub fn calculate_stddev(
    stats: Stats,
    values: impl Iterator<Item = f64>,
    ddof: usize,
) -> Option<f64> {
    calculate_variance(stats, values, ddof).map(f64::sqrt)
}

pub fn calculate_variance(
    stats: Stats,
    values: impl Iterator<Item = f64>,
    ddof: usize,
) -> Option<f64> {
    stats.mean.and_then(|mean| {
        let n = stats.count as usize;
        if n <= ddof {
            return None; // Not enough data points for the requested ddof
        }
        let sum_of_squares = values.map(|value| (value - mean).powi(2)).sum::<f64>();
        Some(sum_of_squares / (n - ddof) as f64)
    })
}

pub fn calculate_skew(stats: Stats, values: impl Iterator<Item = f64>) -> Option<f64> {
    let count = stats.count;
    stats.mean.map(|mean| {
        // In order to use the same iterator for 2 different calculations
        let (m3, m2) = values.fold((0., 0.), |(m3_acc, m2_acc), v| {
            (
                m3_acc + (v - mean).powi(3),
                (v - mean).mul_add(v - mean, m2_acc),
            )
        });

        (m3 / count) / (m2 / count).powi(3).sqrt()
    })
}

#[cfg(test)]
mod decimal_percentile_tests {
    use rstest::rstest;

    use super::exact_decimal_percentile;

    #[rstest]
    // The median of 1.25 and 2.50 at scale 2 is 1.875, cut to 1.87.
    #[case::median_cut_toward_zero(vec![125, 250], 0.5, Some(187))]
    #[case::median_of_an_odd_count(vec![300, 100, 200], 0.5, Some(200))]
    #[case::lowest(vec![300, 100, 200], 0.0, Some(100))]
    #[case::highest(vec![300, 100, 200], 1.0, Some(300))]
    // Rank 0.9 × 10 = 9 exactly; 0.9 taken at its exact binary value would land just past index 9.
    #[case::percentage_read_as_written(
        (0..=10).map(|v| v * 100).collect(), 0.9, Some(900)
    )]
    // Rank 0.25 × 3 = 0.75: −2.00 + 0.75 × (−1.00 − −2.00) = −1.25 exactly.
    #[case::negative_values(vec![-200, -100, 0, 100], 0.25, Some(-125))]
    // −1.00 + 0.5 × (0.01 − −1.00) = −0.495, cut toward zero to −0.49.
    #[case::negative_cut_toward_zero(vec![-100, 1], 0.5, Some(-49))]
    #[case::one_value(vec![42], 0.3, Some(42))]
    #[case::extremes_do_not_overflow(
        vec![-(10_i128.pow(38) - 1), 10_i128.pow(38) - 1], 0.123_456_789, Some(-75_308_642_199_999_999_999_999_999_999_999_999_999_i128)
    )]
    fn interpolates_exactly(
        #[case] values: Vec<i128>,
        #[case] percentage: f64,
        #[case] expected: Option<i128>,
    ) {
        let answer = exact_decimal_percentile(values.into_iter().map(Some), percentage);

        assert_eq!(answer.expect("a percentile"), expected);
    }

    #[test]
    fn skips_nulls_and_answers_none_when_empty() {
        let with_nulls = exact_decimal_percentile([None, Some(100), None, Some(300)], 0.5);
        let empty = exact_decimal_percentile([None, None], 0.5);

        assert_eq!(with_nulls.expect("a percentile"), Some(200));
        assert_eq!(empty.expect("a percentile"), None);
    }
}
