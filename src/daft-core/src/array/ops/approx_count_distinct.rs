use std::collections::HashSet;

use common_error::DaftResult;

use crate::{
    array::ops::DaftApproxCountDistinctAggable, datatypes::UInt64Array,
    utils::identity_hash_set::IdentityBuildHasher,
};

impl DaftApproxCountDistinctAggable for UInt64Array {
    type Output = DaftResult<Self>;

    fn approx_count_distinct(&self) -> Self::Output {
        let mut set = HashSet::with_capacity_and_hasher(self.len(), IdentityBuildHasher::default());
        for value in self.iter().flatten() {
            set.insert(value);
        }
        let count = set.len() as u64;
        let array = Self::from_slice(self.name(), &[count]);
        Ok(array)
    }

    /// Counts the distinct non-null hashes in each group.
    ///
    /// A null slot is skipped, as [`Self::approx_count_distinct`] does, so a group whose
    /// values are all null counts 0. Whether the array holds a null is decided once, and an
    /// array without one reads only its values.
    fn grouped_approx_count_distinct(&self, groups: &super::GroupIndices) -> Self::Output {
        let data = self.values();
        let nulls = self.nulls().filter(|nulls| nulls.null_count() > 0);
        let count_iter = groups.iter().map(|group| {
            let mut set = HashSet::<_, IdentityBuildHasher>::with_capacity_and_hasher(
                group.len(),
                IdentityBuildHasher::default(),
            );
            match nulls {
                None => {
                    for &index in group {
                        if let Some(value) = data.get(index as usize) {
                            set.insert(value);
                        }
                    }
                }
                Some(nulls) => {
                    for &index in group {
                        let index = index as usize;
                        if let Some(value) = data.get(index)
                            && nulls.is_valid(index)
                        {
                            set.insert(value);
                        }
                    }
                }
            }
            set.len() as u64
        });
        Ok(Self::from_values(self.name(), count_iter))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::{
        array::ops::{DaftApproxCountDistinctAggable, GroupIndices, VecIndices},
        datatypes::UInt64Array,
    };

    fn groups(indices: &[Vec<u64>]) -> GroupIndices {
        indices
            .iter()
            .map(|group| VecIndices::from_slice(group))
            .collect()
    }

    fn counts(array: &UInt64Array) -> Vec<Option<u64>> {
        array.iter().collect()
    }

    #[rstest]
    #[case::no_nulls(vec![Some(1), Some(2), Some(2), Some(3)], vec![vec![0, 1, 2], vec![3]], vec![2, 1])]
    #[case::nulls_skipped(vec![Some(1), None, Some(1), None], vec![vec![0, 1, 2, 3]], vec![1])]
    #[case::all_null_group(vec![Some(7), None, None], vec![vec![0], vec![1, 2]], vec![1, 0])]
    #[case::all_null_array(vec![None, None], vec![vec![0, 1]], vec![0])]
    #[case::empty_group(vec![Some(4)], vec![vec![], vec![0]], vec![0, 1])]
    fn grouped_counts_skip_nulls(
        #[case] values: Vec<Option<u64>>,
        #[case] indices: Vec<Vec<u64>>,
        #[case] expected: Vec<u64>,
    ) {
        let array = values.into_iter().collect::<UInt64Array>();
        let result = array
            .grouped_approx_count_distinct(&groups(&indices))
            .unwrap();
        assert_eq!(
            counts(&result),
            expected.into_iter().map(Some).collect::<Vec<_>>()
        );
    }

    #[test]
    fn one_group_agrees_with_ungrouped() {
        let array = [Some(5), None, Some(6), Some(5), None]
            .into_iter()
            .collect::<UInt64Array>();
        let grouped = array
            .grouped_approx_count_distinct(&groups(&[vec![0, 1, 2, 3, 4]]))
            .unwrap();
        let ungrouped = array.approx_count_distinct().unwrap();
        assert_eq!(counts(&grouped), counts(&ungrouped));
    }
}
