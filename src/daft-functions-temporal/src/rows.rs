//! Aligning the rows of a function's arguments before they are paired.

use common_error::{DaftError, DaftResult};
use daft_core::series::Series;

/// Returns `args` with every one-row argument repeated to `row_count` rows.
///
/// A constant argument is evaluated once, as a single row. Pairing it row by row
/// with a column would stop after the first row, so it is repeated first.
///
/// # Errors
///
/// Returns [`DaftError::ValueError`] naming `function` when an argument has
/// neither one row nor `row_count` rows.
pub(crate) fn align_to_rows<const N: usize>(
    function: &str,
    args: [Series; N],
    row_count: usize,
) -> DaftResult<[Series; N]> {
    let mut aligned = args;
    for arg in &mut aligned {
        if arg.len() == row_count {
            continue;
        }
        if arg.len() != 1 {
            return Err(DaftError::ValueError(format!(
                "{function}() expects every argument to have {row_count} rows or one row, but {} has {}",
                arg.name(),
                arg.len()
            )));
        }
        *arg = arg.broadcast(row_count)?;
    }
    Ok(aligned)
}

#[cfg(test)]
mod tests {
    use daft_core::{
        prelude::{DataType, Int32Array},
        series::IntoSeries,
    };
    use rstest::rstest;

    use super::*;

    fn ints(name: &str, values: &[i32]) -> Series {
        Int32Array::from_slice(name, values).into_series()
    }

    #[rstest]
    #[case::constant_repeated(vec![1, 2, 3], vec![7], vec![7, 7, 7])]
    #[case::column_kept(vec![1, 2, 3], vec![4, 5, 6], vec![4, 5, 6])]
    #[case::one_row_batch(vec![1], vec![7], vec![7])]
    fn aligns_every_argument_to_the_batch(
        #[case] column: Vec<i32>,
        #[case] other: Vec<i32>,
        #[case] expected: Vec<i32>,
    ) -> DaftResult<()> {
        let rows = column.len();
        let [_, aligned] = align_to_rows("f", [ints("a", &column), ints("b", &other)], rows)?;

        assert_eq!(*aligned.data_type(), DataType::Int32);
        let observed: Vec<Option<i32>> = aligned.i32()?.into_iter().collect();
        assert_eq!(observed, expected.into_iter().map(Some).collect::<Vec<_>>());
        Ok(())
    }

    #[test]
    fn refuses_a_length_that_is_neither_one_nor_the_batch() {
        let refused = align_to_rows("f", [ints("a", &[1, 2, 3]), ints("b", &[1, 2])], 3);

        assert!(matches!(refused, Err(DaftError::ValueError(message)) if message.contains("f()")));
    }
}
