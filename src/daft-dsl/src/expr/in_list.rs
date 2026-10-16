//! The values an IN list tests membership against.
//!
//! An IN list's items are expressions. When every item is a literal, their
//! values do not depend on the batch being tested, so they are built into one
//! series the first time they are needed and reused by every batch and thread
//! evaluating the same expression after that.

use std::{
    fmt,
    hash::{Hash, Hasher},
    ops::Deref,
    sync::{Arc, OnceLock},
};

use daft_core::series::Series;
use serde::{Deserialize, Serialize};

use super::{Expr, ExprRef};

/// The items of an IN list, with their values built once when all are literals.
///
/// It serializes, compares and hashes as its items alone, so an expression
/// holding it reads, writes and compares as a plain list of items. The built
/// values are never sent: each process builds its own on first use.
#[derive(Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct InList {
    items: Vec<ExprRef>,
    #[serde(skip)]
    values: Arc<OnceLock<Option<Series>>>,
}

impl InList {
    /// The items' values as one series, built on first call and kept.
    ///
    /// Returns `None` unless every item is a literal, or when their values do
    /// not combine into one series. A caller then evaluates the items
    /// itself, which reports the same failure the combination would have.
    pub fn literal_values(&self) -> Option<&Series> {
        self.values
            .get_or_init(|| build_literal_values(&self.items))
            .as_ref()
    }

    /// A list of `items`, keeping these values when the items are these same
    /// expressions, as when a list's expression is rebuilt around it.
    #[must_use]
    pub fn with_items(&self, items: Vec<ExprRef>) -> Self {
        let same = items.len() == self.items.len()
            && items
                .iter()
                .zip(&self.items)
                .all(|(new, old)| Arc::ptr_eq(new, old));
        Self {
            values: if same {
                self.values.clone()
            } else {
                Arc::default()
            },
            items,
        }
    }
}

/// Every item's value as one series, or `None` unless all are literals and
/// their values concatenate. Each literal becomes its own series first, the
/// way evaluating it does, so the result is what evaluating the items gives.
fn build_literal_values(items: &[ExprRef]) -> Option<Series> {
    let values = items
        .iter()
        .map(|item| match item.as_ref() {
            Expr::Literal(value) => Series::from_literals(vec![value.clone()]).ok(),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    let first = values.first()?;
    if values.len() == 1 {
        return Some(first.clone());
    }
    Series::concat(&values.iter().collect::<Vec<_>>()).ok()
}

impl From<Vec<ExprRef>> for InList {
    fn from(items: Vec<ExprRef>) -> Self {
        Self {
            items,
            values: Arc::default(),
        }
    }
}

impl Deref for InList {
    type Target = [ExprRef];

    fn deref(&self) -> &Self::Target {
        &self.items
    }
}

impl PartialEq for InList {
    fn eq(&self, other: &Self) -> bool {
        self.items == other.items
    }
}

impl Eq for InList {}

impl Hash for InList {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.items.hash(state);
    }
}

impl fmt::Debug for InList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.items.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use std::hash::{DefaultHasher, Hash, Hasher};

    use daft_core::prelude::*;
    use rstest::rstest;

    use super::InList;
    use crate::{ExprRef, lit, null_lit, resolved_col};

    fn hash_of(value: &impl Hash) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    fn ints(values: &[i64]) -> Vec<ExprRef> {
        values.iter().map(|v| lit(*v)).collect()
    }

    #[test]
    fn literal_items_are_built_once_into_one_series() {
        let list = InList::from(ints(&[3, 1, 2]));

        let first = list.literal_values().unwrap();
        let again = list.literal_values().unwrap();

        assert_eq!(first.len(), 3);
        assert_eq!(first.data_type(), &DataType::Int64);
        assert!(std::ptr::eq(first, again));
    }

    #[rstest]
    #[case::a_column(vec![lit(1i64), resolved_col("x")])]
    #[case::no_items(vec![])]
    #[case::types_that_do_not_concatenate(vec![lit(1i64), lit("a")])]
    fn items_not_all_combinable_literals_have_no_built_values(#[case] items: Vec<ExprRef>) {
        assert!(InList::from(items).literal_values().is_none());
    }

    #[test]
    fn a_null_item_is_built_with_the_others() {
        let list = InList::from(vec![lit(1i64), null_lit(), lit(3i64)]);

        let values = list.literal_values();

        assert_eq!(values.map(Series::len), list_concat_len(&list));
    }

    /// What concatenating the items' own series gives, or `None` when it fails.
    fn list_concat_len(list: &InList) -> Option<usize> {
        let series = list
            .iter()
            .map(|item| match item.as_ref() {
                crate::Expr::Literal(value) => Series::from_literals(vec![value.clone()]).unwrap(),
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();
        Series::concat(&series.iter().collect::<Vec<_>>())
            .ok()
            .map(|s| s.len())
    }

    #[test]
    fn the_same_items_keep_their_built_values_and_new_items_do_not() {
        let list = InList::from(ints(&[1, 2]));
        let built = list.literal_values().unwrap() as *const Series;

        let rebuilt = list.with_items(list.to_vec());
        let changed = list.with_items(ints(&[1, 2]));

        assert!(std::ptr::eq(rebuilt.literal_values().unwrap(), built));
        assert!(!std::ptr::eq(changed.literal_values().unwrap(), built));
    }

    #[test]
    fn it_is_written_and_read_exactly_as_a_list_of_its_items() {
        let items = vec![lit(1i64), lit("a"), resolved_col("x")];
        let list = InList::from(items.clone());
        list.literal_values();
        let config = bincode::config::legacy();

        let as_json = serde_json::to_string(&list).unwrap();
        let as_bytes = bincode::serde::encode_to_vec(&list, config).unwrap();
        let (read, _): (InList, usize) =
            bincode::serde::decode_from_slice(&as_bytes, config).unwrap();

        assert_eq!(as_json, serde_json::to_string(&items).unwrap());
        assert_eq!(
            as_bytes,
            bincode::serde::encode_to_vec(&items, config).unwrap()
        );
        assert_eq!(read, list);
    }

    #[test]
    fn binding_an_expression_to_a_schema_keeps_its_lists_built_values() {
        let schema = Schema::new(vec![Field::new("x", DataType::Int64)]);
        let expr = resolved_col("x").is_in(ints(&[1, 2, 3]));
        let crate::Expr::IsIn(_, list) = expr.as_ref() else {
            unreachable!()
        };
        let built = list.literal_values().unwrap() as *const Series;

        let bound = crate::expr::bound_expr::BoundExpr::try_new(expr.clone(), &schema).unwrap();

        let crate::Expr::IsIn(_, bound_list) = bound.as_ref() else {
            unreachable!()
        };
        assert!(std::ptr::eq(bound_list.literal_values().unwrap(), built));
    }

    #[test]
    fn it_compares_and_hashes_as_its_items_whether_built_or_not() {
        let built = InList::from(ints(&[1, 2]));
        built.literal_values();
        let unbuilt = InList::from(ints(&[1, 2]));

        assert_eq!(built, unbuilt);
        assert_eq!(hash_of(&built), hash_of(&unbuilt));
        assert_ne!(built, InList::from(ints(&[2, 1])));
        assert_eq!(format!("{built:?}"), format!("{:?}", ints(&[1, 2])));
    }
}
