//! Exact comparison of decimal columns of different types.
//!
//! Two decimals whose common type would pass 38 digits have no decimal type that holds
//! every value of both. A comparison answers a boolean, so these compare each pair of
//! values aligned to the larger scale instead, exactly and without casting either side.

use std::{cmp::Ordering, sync::Arc};

use arrow::buffer::{BooleanBuffer, NullBuffer};
use common_error::{DaftError, DaftResult};

use crate::{
    datatypes::{BooleanArray, DataType, Decimal128Array, Field},
    utils::decimal::DecimalComparison,
};

fn decimal_scale(array: &Decimal128Array) -> DaftResult<usize> {
    match array.data_type() {
        DataType::Decimal128(_, scale) => Ok(*scale),
        other => Err(DaftError::TypeError(format!(
            "Expected a decimal operand, not {other}"
        ))),
    }
}

/// The common length of two broadcastable operands, and each side's index step:
/// 1 to walk a column, 0 to repeat a single value.
fn broadcast(lhs: usize, rhs: usize) -> DaftResult<(usize, usize, usize)> {
    match (lhs, rhs) {
        (a, b) if a == b => Ok((a, 1, 1)),
        (len, 1) => Ok((len, 1, 0)),
        (1, len) => Ok((len, 0, 1)),
        (a, b) => Err(DaftError::ValueError(format!(
            "Cannot compare arrays of different lengths: {a} vs {b}"
        ))),
    }
}

fn boolean_array(
    name: &str,
    values: BooleanBuffer,
    nulls: Option<NullBuffer>,
) -> DaftResult<BooleanArray> {
    BooleanArray::from_arrow(
        Field::new(name, DataType::Boolean),
        Arc::new(arrow::array::BooleanArray::new(values, nulls)),
    )
}

/// The validity of a broadcast operand over `len` rows.
fn validity(array: &Decimal128Array, len: usize) -> Option<NullBuffer> {
    if array.len() == len {
        array.nulls().cloned()
    } else if array.null_count() == 0 {
        None
    } else {
        Some(NullBuffer::new_null(len))
    }
}

impl Decimal128Array {
    /// Compares this column with `rhs`, row by row, answering `test` of each ordering.
    ///
    /// The two may have any decimal types: each pair is compared exactly at the larger
    /// scale. A row is null where either side is null. Either side may be a single
    /// value, which is broadcast. The result takes this column's name.
    ///
    /// # Errors
    ///
    /// Returns [`DaftError::ValueError`] for lengths that cannot broadcast.
    pub fn compare_exactly(
        &self,
        rhs: &Self,
        test: fn(Ordering) -> bool,
    ) -> DaftResult<BooleanArray> {
        let comparison = DecimalComparison::new(decimal_scale(self)?, decimal_scale(rhs)?)?;
        let (len, left_step, right_step) = broadcast(self.len(), rhs.len())?;
        let (left, right) = (self.values(), rhs.values());
        let values = BooleanBuffer::collect_bool(len, |row| {
            test(comparison.compare(left[row * left_step], right[row * right_step]))
        });
        let nulls = NullBuffer::union(validity(self, len).as_ref(), validity(rhs, len).as_ref());
        boolean_array(self.name(), values, nulls)
    }

    /// `self <=> rhs` exactly: true where both are null, false where one is, and
    /// otherwise whether the two are equal at the larger scale. Never null.
    ///
    /// # Errors
    ///
    /// Returns [`DaftError::ValueError`] for lengths that cannot broadcast.
    pub fn equal_null_safe_exactly(&self, rhs: &Self) -> DaftResult<BooleanArray> {
        let comparison = DecimalComparison::new(decimal_scale(self)?, decimal_scale(rhs)?)?;
        let (len, left_step, right_step) = broadcast(self.len(), rhs.len())?;
        let (left, right) = (self.values(), rhs.values());
        let (left_valid, right_valid) = (validity(self, len), validity(rhs, len));
        let is_valid =
            |nulls: &Option<NullBuffer>, row| nulls.as_ref().is_none_or(|n| n.is_valid(row));
        let values = BooleanBuffer::collect_bool(len, |row| {
            match (is_valid(&left_valid, row), is_valid(&right_valid, row)) {
                (true, true) => {
                    comparison.compare(left[row * left_step], right[row * right_step])
                        == Ordering::Equal
                }
                (left_valid, right_valid) => left_valid == right_valid,
            }
        });
        boolean_array(self.name(), values, None)
    }

    /// Whether each value of this column equals one of the non-null `items`, exactly at
    /// the larger scale. A row is null where its value is null.
    ///
    /// # Errors
    ///
    /// Returns [`DaftError::TypeError`] when either side is not a decimal.
    pub fn is_in_exactly(&self, items: &Self) -> DaftResult<BooleanArray> {
        let comparison = DecimalComparison::new(decimal_scale(self)?, decimal_scale(items)?)?;
        let item_values = items.values();
        let item_nulls = items.nulls();
        let aligned_items = (0..items.len())
            .filter(|&row| item_nulls.is_none_or(|nulls| nulls.is_valid(row)))
            .map(|row| comparison.align_right(item_values[row]));
        let values = self.values();
        let members = match aligned_items
            .clone()
            .map(|item| item.to_i128())
            .collect::<Option<hashbrown::HashSet<i128>>>()
        {
            // Every item fits 128 bits once aligned, so a value that does not cannot be one.
            Some(set) => BooleanBuffer::collect_bool(self.len(), |row| {
                comparison
                    .align_left_narrow(values[row])
                    .is_some_and(|value| set.contains(&value))
            }),
            None => {
                let set = aligned_items.collect::<hashbrown::HashSet<_>>();
                BooleanBuffer::collect_bool(self.len(), |row| {
                    set.contains(&comparison.align_left(values[row]))
                })
            }
        };
        boolean_array(self.name(), members, self.nulls().cloned())
    }
}
