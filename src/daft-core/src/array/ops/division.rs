//! Floored and truncated division and remainder of numeric arrays.
//!
//! The DataFrame operators `//` and `%` floor: the quotient rounds toward negative
//! infinity and the remainder takes the divisor's sign, so `a == b * (a // b) + a % b`
//! for every sign. SQL's `%` (and `MOD`) and `//` truncate toward zero instead, so the
//! remainder takes the dividend's sign.
//!
//! An integer row with a zero divisor answers null rather than aborting the query: a
//! row error could not be guarded against, because a conditional evaluates both of its
//! branches over every row. `MIN // −1` wraps to `MIN` and `MIN % −1` is 0, as integer
//! `+` and `*` wrap. Floats follow IEEE: dividing by zero gives an infinity or NaN.
//!
//! Each kernel computes every row without a branch, with zero divisors replaced by one
//! and masked out of the validity afterwards, so its loop vectorizes.

use std::iter::zip;

use arrow::buffer::{BooleanBuffer, NullBuffer};
use common_error::{DaftError, DaftResult};
use num_traits::{Float, Zero};

use super::full::FullNull;
use crate::{
    array::DataArray,
    datatypes::{DaftNumericType, NumericNative},
};

/// Floored and truncated division for one primitive type.
pub trait Division: NumericNative {
    /// Whether some divisor has no quotient (an integer zero), so that rows must be masked.
    const MAY_HAVE_NO_QUOTIENT: bool;

    /// Whether `self`, as a divisor, has no quotient.
    fn has_no_quotient(self) -> bool;

    /// A divisor that is safe to compute with: one in place of an integer zero, whose
    /// row is masked out of the validity.
    #[must_use]
    fn computable(self) -> Self;

    /// The quotient rounded toward negative infinity.
    #[must_use]
    fn floor_div(self, divisor: Self) -> Self;

    /// The remainder with the divisor's sign, matching [`Self::floor_div`].
    #[must_use]
    fn floor_rem(self, divisor: Self) -> Self;

    /// [`Self::floor_rem`] by a divisor every row shares, whose sign `divisor_negative` is
    /// decided once: the remainder then moves into the divisor's sign with one shift and
    /// add, instead of a comparison of both signs per row.
    #[must_use]
    fn floor_rem_by(self, divisor: Self, divisor_negative: bool) -> Self;

    /// The quotient rounded toward zero.
    #[must_use]
    fn trunc_div(self, divisor: Self) -> Self;

    /// The remainder with the dividend's sign, matching [`Self::trunc_div`].
    #[must_use]
    fn trunc_rem(self, divisor: Self) -> Self;
}

macro_rules! impl_signed_division {
    ($($t:ty),+) => {$(
        impl Division for $t {
            const MAY_HAVE_NO_QUOTIENT: bool = true;

            #[inline]
            fn has_no_quotient(self) -> bool {
                self == 0
            }

            #[inline]
            fn computable(self) -> Self {
                self | Self::from(self == 0)
            }

            #[inline]
            fn floor_div(self, divisor: Self) -> Self {
                let quotient = self.wrapping_div(divisor);
                // From the quotient, so that one division serves both.
                let remainder = self.wrapping_sub(quotient.wrapping_mul(divisor));
                // One less when the remainder is nonzero and its sign differs from the divisor's.
                quotient.wrapping_sub(Self::from(remainder != 0 && (remainder ^ divisor) < 0))
            }

            #[inline]
            fn floor_rem(self, divisor: Self) -> Self {
                let remainder = self.wrapping_rem(divisor);
                // Shifted into the divisor's sign when the signs differ, without a branch.
                let shift = Self::from(remainder != 0 && (remainder ^ divisor) < 0);
                remainder.wrapping_add(divisor.wrapping_mul(shift))
            }

            #[inline]
            fn floor_rem_by(self, divisor: Self, divisor_negative: bool) -> Self {
                let remainder = self.wrapping_rem(divisor);
                // All ones where the remainder's sign is the opposite of the divisor's: a
                // negative remainder under a positive divisor, or a positive one under a
                // negative divisor. The branch is on a value fixed across rows.
                let opposite = if divisor_negative {
                    Self::from(remainder > 0).wrapping_neg()
                } else {
                    remainder >> (Self::BITS - 1)
                };
                remainder.wrapping_add(divisor & opposite)
            }

            #[inline]
            fn trunc_div(self, divisor: Self) -> Self {
                self.wrapping_div(divisor)
            }

            #[inline]
            fn trunc_rem(self, divisor: Self) -> Self {
                self.wrapping_rem(divisor)
            }
        }
    )+};
}

macro_rules! impl_unsigned_division {
    ($($t:ty),+) => {$(
        impl Division for $t {
            const MAY_HAVE_NO_QUOTIENT: bool = true;

            #[inline]
            fn has_no_quotient(self) -> bool {
                self == 0
            }

            #[inline]
            fn computable(self) -> Self {
                self | Self::from(self == 0)
            }

            // Without negative values, flooring and truncating agree.
            #[inline]
            fn floor_div(self, divisor: Self) -> Self {
                self / divisor
            }

            #[inline]
            fn floor_rem(self, divisor: Self) -> Self {
                self % divisor
            }

            #[inline]
            fn floor_rem_by(self, divisor: Self, _divisor_negative: bool) -> Self {
                self % divisor
            }

            #[inline]
            fn trunc_div(self, divisor: Self) -> Self {
                self / divisor
            }

            #[inline]
            fn trunc_rem(self, divisor: Self) -> Self {
                self % divisor
            }
        }
    )+};
}

macro_rules! impl_float_division {
    ($($t:ty),+) => {$(
        impl Division for $t {
            const MAY_HAVE_NO_QUOTIENT: bool = false;

            #[inline]
            fn has_no_quotient(self) -> bool {
                false
            }

            #[inline]
            fn computable(self) -> Self {
                self
            }

            #[inline]
            fn floor_div(self, divisor: Self) -> Self {
                (self / divisor).floor()
            }

            // From `fmod`, which is exact, rather than `a − b·⌊a/b⌋`, which loses digits.
            #[inline]
            fn floor_rem(self, divisor: Self) -> Self {
                let remainder = self % divisor;
                if remainder == <$t>::zero() {
                    // A zero remainder takes the divisor's sign.
                    if divisor.is_sign_negative() {
                        -<$t>::zero()
                    } else {
                        <$t>::zero()
                    }
                } else if remainder.is_sign_negative() != divisor.is_sign_negative() {
                    remainder + divisor
                } else {
                    remainder
                }
            }

            #[inline]
            fn floor_rem_by(self, divisor: Self, _divisor_negative: bool) -> Self {
                self.floor_rem(divisor)
            }

            #[inline]
            fn trunc_div(self, divisor: Self) -> Self {
                (self / divisor).trunc()
            }

            #[inline]
            fn trunc_rem(self, divisor: Self) -> Self {
                self % divisor
            }
        }
    )+};
}

impl_signed_division!(i8, i16, i32, i64, i128);
impl_unsigned_division!(u8, u16, u32, u64);
impl_float_division!(half::f16, f32, f64);

/// The validity mask of the divisors that have a quotient.
fn divisors_with_a_quotient<N: Division>(divisors: &[N]) -> NullBuffer {
    NullBuffer::new(BooleanBuffer::collect_bool(divisors.len(), |i| {
        !divisors[i].has_no_quotient()
    }))
}

/// `op` of each dividend by its divisor, and whether any divisor had no quotient.
///
/// The check rides along in the same pass, so divisors are read once; the mask is built
/// only when a row needs it.
fn divide_each<N, I, F>(pairs: I, len: usize, op: F) -> (Vec<N>, bool)
where
    N: Division,
    I: Iterator<Item = (N, N)>,
    F: Fn(N, N) -> N,
{
    let mut any_without_quotient = false;
    let mut values = Vec::with_capacity(len);
    values.extend(pairs.map(|(dividend, divisor)| {
        if N::MAY_HAVE_NO_QUOTIENT {
            any_without_quotient |= divisor.has_no_quotient();
        }
        op(dividend, divisor.computable())
    }));
    (values, any_without_quotient)
}

/// `op` for any divisor sign: an operation that needs no sign decided ahead of the rows.
fn for_any_sign<N, F>(op: F) -> impl Fn(N, N, bool) -> N + Copy
where
    F: Fn(N, N) -> N + Copy,
{
    move |dividend, divisor, _| op(dividend, divisor)
}

/// Applies a division `op` to every row, broadcasting a single value on either side.
///
/// A divisor every row shares goes through `by_shared`, with its sign decided once. A
/// row is null where an input is null or its divisor has no quotient. The result takes
/// the dividend's name and type.
#[inline(never)]
fn divide_rows<T, F, G>(
    lhs: &DataArray<T>,
    rhs: &DataArray<T>,
    op: F,
    by_shared: G,
) -> DaftResult<DataArray<T>>
where
    T: DaftNumericType,
    T::Native: Division,
    F: Fn(T::Native, T::Native) -> T::Native + Copy,
    G: Fn(T::Native, T::Native, bool) -> T::Native + Copy,
{
    let (lhs_values, rhs_values) = (lhs.values(), rhs.values());
    match (lhs.len(), rhs.len()) {
        (a, b) if a == b => {
            let pairs = zip(lhs_values.iter().copied(), rhs_values.iter().copied());
            let (values, any_without_quotient) = divide_each(pairs, a, op);
            let inputs = NullBuffer::union(lhs.nulls(), rhs.nulls());
            let nulls = if any_without_quotient {
                NullBuffer::union(
                    inputs.as_ref(),
                    Some(&divisors_with_a_quotient(&rhs_values)),
                )
            } else {
                inputs
            };
            DataArray::from_field_and_values(lhs.field.clone(), values).with_nulls(nulls)
        }
        (len, 1) => match rhs.get(0) {
            Some(divisor) if !divisor.has_no_quotient() => {
                let negative = divisor < T::Native::zero();
                DataArray::from_field_and_values(
                    lhs.field.clone(),
                    lhs_values
                        .iter()
                        .map(|dividend| by_shared(*dividend, divisor, negative)),
                )
                .with_nulls(lhs.nulls().cloned())
            }
            _ => Ok(DataArray::full_null(lhs.name(), lhs.data_type(), len)),
        },
        (1, len) => match lhs.get(0) {
            Some(dividend) => {
                let pairs = rhs_values.iter().map(|divisor| (dividend, *divisor));
                let (values, any_without_quotient) = divide_each(pairs, len, op);
                let nulls = if any_without_quotient {
                    NullBuffer::union(rhs.nulls(), Some(&divisors_with_a_quotient(&rhs_values)))
                } else {
                    rhs.nulls().cloned()
                };
                DataArray::from_field_and_values(lhs.field.clone(), values).with_nulls(nulls)
            }
            None => Ok(DataArray::full_null(lhs.name(), lhs.data_type(), len)),
        },
        (a, b) => Err(DaftError::ValueError(format!(
            "Cannot apply operation on arrays of different lengths: {a} vs {b}"
        ))),
    }
}

impl<T> DataArray<T>
where
    T: DaftNumericType,
    T::Native: Division,
{
    /// `self // rhs`, rounded toward negative infinity.
    ///
    /// A row is null where either side is null or an integer divisor is zero; `MIN // −1`
    /// wraps to `MIN`. Either side may be a single value, which is broadcast.
    ///
    /// # Errors
    ///
    /// Returns [`DaftError::ValueError`] for lengths that cannot broadcast.
    pub fn floor_divide(&self, rhs: &Self) -> DaftResult<Self> {
        divide_rows(
            self,
            rhs,
            Division::floor_div,
            for_any_sign(Division::floor_div),
        )
    }

    /// `self % rhs` with the divisor's sign, so that `a == b * (a // b) + a % b`.
    ///
    /// A row is null where either side is null or an integer divisor is zero; `MIN % −1`
    /// is 0. Either side may be a single value, which is broadcast.
    ///
    /// # Errors
    ///
    /// Returns [`DaftError::ValueError`] for lengths that cannot broadcast.
    pub fn floored_remainder(&self, rhs: &Self) -> DaftResult<Self> {
        divide_rows(self, rhs, Division::floor_rem, Division::floor_rem_by)
    }

    /// `self / rhs` rounded toward zero, as SQL's integer division does.
    ///
    /// A row is null where either side is null or an integer divisor is zero; `MIN / −1`
    /// wraps to `MIN`. Either side may be a single value, which is broadcast.
    ///
    /// # Errors
    ///
    /// Returns [`DaftError::ValueError`] for lengths that cannot broadcast.
    pub fn truncated_divide(&self, rhs: &Self) -> DaftResult<Self> {
        divide_rows(
            self,
            rhs,
            Division::trunc_div,
            for_any_sign(Division::trunc_div),
        )
    }

    /// `self % rhs` with the dividend's sign, as SQL's `%` and `MOD` do.
    ///
    /// A row is null where either side is null or an integer divisor is zero; `MIN % −1`
    /// is 0. Either side may be a single value, which is broadcast.
    ///
    /// # Errors
    ///
    /// Returns [`DaftError::ValueError`] for lengths that cannot broadcast.
    pub fn truncated_remainder(&self, rhs: &Self) -> DaftResult<Self> {
        divide_rows(
            self,
            rhs,
            Division::trunc_rem,
            for_any_sign(Division::trunc_rem),
        )
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::datatypes::{
        DataType, Field, Float64Array, Int8Array, Int32Array, Int64Array, UInt32Array,
    };

    #[rstest]
    // (dividend, divisor) → (floored quotient, floored remainder, truncated quotient, truncated remainder)
    #[case::both_positive(7, 2, 3, 1, 3, 1)]
    #[case::negative_dividend(-7, 2, -4, 1, -3, -1)]
    #[case::negative_divisor(7, -2, -4, -1, -3, 1)]
    #[case::both_negative(-7, -2, 3, -1, 3, -1)]
    #[case::exact(-6, 2, -3, 0, -3, 0)]
    #[case::zero_dividend(0, -3, 0, 0, 0, 0)]
    fn signed_division_floors_and_truncates(
        #[case] a: i64,
        #[case] b: i64,
        #[case] floor_q: i64,
        #[case] floor_r: i64,
        #[case] trunc_q: i64,
        #[case] trunc_r: i64,
    ) {
        assert_eq!((a.floor_div(b), a.floor_rem(b)), (floor_q, floor_r));
        assert_eq!((a.trunc_div(b), a.trunc_rem(b)), (trunc_q, trunc_r));
        // Each pair keeps the identity a == b·q + r.
        assert_eq!(b * floor_q + floor_r, a);
        assert_eq!(b * trunc_q + trunc_r, a);
    }

    #[test]
    fn the_remainder_by_a_shared_divisor_is_the_floored_remainder() {
        for dividend in i8::MIN..=i8::MAX {
            for divisor in (i8::MIN..=i8::MAX).filter(|d| *d != 0) {
                assert_eq!(
                    dividend.floor_rem_by(divisor, divisor < 0),
                    dividend.floor_rem(divisor),
                    "{dividend} % {divisor}"
                );
            }
        }
        for (dividend, divisor) in [
            (i64::MIN, -1),
            (i64::MIN, 7),
            (i64::MAX, -7),
            (-1, i64::MIN),
        ] {
            assert_eq!(
                dividend.floor_rem_by(divisor, divisor < 0),
                dividend.floor_rem(divisor)
            );
        }
    }

    #[test]
    fn the_lowest_value_by_minus_one_wraps() {
        assert_eq!(i8::MIN.floor_div(-1), i8::MIN);
        assert_eq!(i8::MIN.floor_rem(-1), 0);
        assert_eq!(i64::MIN.trunc_div(-1), i64::MIN);
        assert_eq!(i64::MIN.trunc_rem(-1), 0);
    }

    #[rstest]
    #[case::negative_dividend(-7.0, 2.0, -4.0, 1.0)]
    #[case::negative_divisor(7.0, -2.0, -4.0, -1.0)]
    #[case::both_negative(-7.0, -2.0, 3.0, -1.0)]
    fn floats_floor(#[case] a: f64, #[case] b: f64, #[case] q: f64, #[case] r: f64) {
        assert_eq!(a.floor_div(b), q);
        assert_eq!(a.floor_rem(b), r);
        assert_eq!(a.trunc_rem(b), a % b);
    }

    #[test]
    fn a_zero_float_remainder_takes_the_divisor_sign_and_zero_divisors_are_ieee() {
        assert!(6.0_f64.floor_rem(-2.0).is_sign_negative());
        assert!((-6.0_f64).floor_rem(2.0).is_sign_positive());
        assert!(1.0_f64.floor_rem(0.0).is_nan());
        assert_eq!(1.0_f64.floor_div(0.0), f64::INFINITY);
    }

    #[test]
    fn unsigned_division_floors_and_truncates_alike() {
        assert_eq!((7_u32.floor_div(2), 7_u32.floor_rem(2)), (3, 1));
        assert_eq!((7_u32.trunc_div(2), 7_u32.trunc_rem(2)), (3, 1));
    }

    #[test]
    fn a_zero_integer_divisor_answers_null_and_keeps_the_other_rows() {
        let dividends = Int64Array::from_iter(
            Field::new("a", DataType::Int64),
            [Some(-7), Some(7), None, Some(9)],
        );
        let divisors = Int64Array::from_iter(
            Field::new("b", DataType::Int64),
            [Some(2), Some(0), Some(3), Some(-4)],
        );

        let quotients = dividends.floor_divide(&divisors).expect("a quotient");
        let remainders = dividends.floored_remainder(&divisors).expect("a remainder");

        assert_eq!(
            (&quotients).into_iter().collect::<Vec<_>>(),
            [Some(-4), None, None, Some(-3)]
        );
        assert_eq!(
            remainders.into_iter().collect::<Vec<_>>(),
            [Some(1), None, None, Some(-3)]
        );
        assert_eq!(quotients.name(), "a");
    }

    #[test]
    fn a_scalar_on_either_side_is_broadcast() {
        let column =
            Int32Array::from_iter(Field::new("a", DataType::Int32), [Some(-7), Some(0), None]);
        let two = Int32Array::from_iter(Field::new("b", DataType::Int32), [Some(2)]);
        let zero = Int32Array::from_iter(Field::new("b", DataType::Int32), [Some(0)]);

        let by_two = column.floor_divide(&two).expect("a quotient");
        let by_zero = column.floor_divide(&zero).expect("a quotient");
        let two_by = two.truncated_divide(&column).expect("a quotient");

        assert_eq!(
            by_two.into_iter().collect::<Vec<_>>(),
            [Some(-4), Some(0), None]
        );
        assert_eq!(by_zero.into_iter().collect::<Vec<_>>(), [None, None, None]);
        // 2 / −7 truncates to 0; 2 / 0 has no quotient.
        assert_eq!(
            two_by.into_iter().collect::<Vec<_>>(),
            [Some(0), None, None]
        );
    }

    #[test]
    fn narrow_and_unsigned_arrays_divide() {
        let narrow =
            Int8Array::from_iter(Field::new("a", DataType::Int8), [Some(i8::MIN), Some(-7)]);
        let minus_one = Int8Array::from_iter(Field::new("b", DataType::Int8), [Some(-1), Some(2)]);
        let unsigned =
            UInt32Array::from_iter(Field::new("a", DataType::UInt32), [Some(7), Some(1)]);
        let unsigned_zero =
            UInt32Array::from_iter(Field::new("b", DataType::UInt32), [Some(2), Some(0)]);

        let quotients = narrow.floor_divide(&minus_one).expect("a quotient");
        let remainders = unsigned
            .floored_remainder(&unsigned_zero)
            .expect("a remainder");

        assert_eq!(
            quotients.into_iter().collect::<Vec<_>>(),
            [Some(i8::MIN), Some(-4)]
        );
        assert_eq!(remainders.into_iter().collect::<Vec<_>>(), [Some(1), None]);
    }

    #[test]
    fn floats_keep_every_row() {
        let dividends =
            Float64Array::from_iter(Field::new("a", DataType::Float64), [Some(-7.0), Some(1.0)]);
        let divisors =
            Float64Array::from_iter(Field::new("b", DataType::Float64), [Some(2.0), Some(0.0)]);

        let remainders = dividends.floored_remainder(&divisors).expect("a remainder");

        assert_eq!(remainders.get(0), Some(1.0));
        assert!(remainders.get(1).is_some_and(f64::is_nan));
    }
}
