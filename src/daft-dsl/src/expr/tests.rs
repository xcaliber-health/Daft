use super::*;

#[test]
fn check_comparison_type() -> DaftResult<()> {
    let x = lit(10.);
    let y = lit(12);
    let schema = Schema::empty();

    let z = Expr::BinaryOp {
        left: x,
        right: y,
        op: Operator::Lt,
    };
    assert_eq!(z.get_type(&schema)?, DataType::Boolean);
    Ok(())
}

#[test]
fn check_alias_type() -> DaftResult<()> {
    let a = resolved_col("a");
    let b = a.alias("b");
    match b.as_ref() {
        Expr::Alias(..) => Ok(()),
        other => Err(common_error::DaftError::ValueError(format!(
            "expected expression to be a alias, got {other:?}"
        ))),
    }
}

#[test]
fn check_arithmetic_type() -> DaftResult<()> {
    let x = lit(10.);
    let y = lit(12);
    let schema = Schema::empty();

    let z = Expr::BinaryOp {
        left: x,
        right: y,
        op: Operator::Plus,
    };
    assert_eq!(z.get_type(&schema)?, DataType::Float64);

    let x = lit(10.);
    let y = lit(12);

    let z = Expr::BinaryOp {
        left: y,
        right: x,
        op: Operator::Plus,
    };
    assert_eq!(z.get_type(&schema)?, DataType::Float64);

    Ok(())
}

#[test]
fn check_arithmetic_type_with_columns() -> DaftResult<()> {
    let x = resolved_col("x");
    let y = resolved_col("y");
    let schema = Schema::new(vec![
        Field::new("x", DataType::Float64),
        Field::new("y", DataType::Int64),
    ]);

    let z = Expr::BinaryOp {
        left: x,
        right: y,
        op: Operator::Plus,
    };
    assert_eq!(z.get_type(&schema)?, DataType::Float64);

    let x = resolved_col("x");
    let y = resolved_col("y");

    let z = Expr::BinaryOp {
        left: y,
        right: x,
        op: Operator::Plus,
    };
    assert_eq!(z.get_type(&schema)?, DataType::Float64);

    Ok(())
}

// Partition-compatibility helpers operate on bound columns.
fn bound(index: usize, name: &str) -> crate::expr::bound_expr::BoundExpr {
    crate::expr::bound_expr::BoundExpr::new_unchecked(bound_col(
        index,
        Field::new(name, DataType::Int64),
    ))
}

#[test]
fn is_exact_partition_match_is_order_independent_set_equality() {
    let a = bound(0, "a");
    let b = bound(1, "b");
    let c = bound(2, "c");
    assert!(is_exact_partition_match(
        &[a.clone(), b.clone()],
        &[b.clone(), a.clone()]
    ));
    // A superset is NOT an exact match.
    assert!(!is_exact_partition_match(
        &[a.clone(), b.clone()],
        &[a, b, c]
    ));
}

#[test]
fn clustering_is_covered_by_equal_sets() {
    let a = bound(0, "a");
    let b = bound(1, "b");
    assert!(clustering_is_covered_by(&[a.clone(), b.clone()], &[b, a]));
}

#[test]
fn clustering_is_covered_by_op_is_strict_superset() {
    // op partitions by {a, b, c}, input is clustered by {a, b} => covered (refinement).
    let a = bound(0, "a");
    let b = bound(1, "b");
    let c = bound(2, "c");
    assert!(clustering_is_covered_by(
        &[a.clone(), b.clone()],
        &[a, b, c]
    ));
}

#[test]
fn clustering_is_covered_by_input_richer_is_not_covered() {
    // input clustered by {a, b, c}, op partitions by {a, b} => NOT covered (unsound to skip).
    let a = bound(0, "a");
    let b = bound(1, "b");
    let c = bound(2, "c");
    assert!(!clustering_is_covered_by(
        &[a.clone(), b.clone(), c],
        &[a, b]
    ));
}

#[test]
fn clustering_is_covered_by_empty_input_is_not_covered() {
    let a = bound(0, "a");
    let input: [crate::expr::bound_expr::BoundExpr; 0] = [];
    assert!(!clustering_is_covered_by(&input, &[a]));
}

#[test]
fn clustering_is_covered_by_expression_keys() {
    // Expression-valued keys must compare structurally, not just bare columns.
    let producer = bound(0, "producer");
    let hour = crate::expr::bound_expr::BoundExpr::new_unchecked(binary_op(
        Operator::FloorDivide,
        bound_col(1, Field::new("id", DataType::Int64)),
        lit(3_600_000i64),
    ));
    // op adds an extra sub-bucket key => still covered.
    assert!(clustering_is_covered_by(
        &[producer.clone(), hour.clone()],
        &[producer, hour, bound(2, "bucket")]
    ));
}

/// `x + 1 + 1 …`, nested `depth` levels on top of `base`.
fn chained(base: ExprRef, depth: usize) -> ExprRef {
    (0..depth).fold(base, |left, _| {
        Expr::BinaryOp {
            op: Operator::Plus,
            left,
            right: lit(1),
        }
        .arced()
    })
}

/// How many levels nest below `expr` along its left operands.
fn left_depth(expr: &ExprRef) -> usize {
    let mut depth = 0;
    let mut level = expr;
    while let Expr::BinaryOp { left, .. } = level.as_ref() {
        depth += 1;
        level = left;
    }
    depth
}

#[test]
fn an_expression_deeper_than_its_threads_stack_drops() {
    // Far more levels than a 256 KiB stack holds when each level's drop recurses.
    let depth = 1_000_000;
    std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(move || drop(chained(resolved_col("x"), depth)))
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn a_subtree_owned_elsewhere_survives_its_parent_dropping() {
    let shared = chained(resolved_col("x"), 1_000);
    let kept = shared.clone();

    std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(move || drop(chained(shared, 1_000_000)))
        .unwrap()
        .join()
        .unwrap();

    assert_eq!(Arc::strong_count(&kept), 1);
    assert_eq!(left_depth(&kept), 1_000);
}

mod selectivity {
    use rstest::rstest;

    use super::*;

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("a", DataType::Int64),
            Field::new("b", DataType::Int64),
            Field::new("s", DataType::Utf8),
        ])
    }

    fn values(k: i64) -> Vec<ExprRef> {
        (1..=k).map(lit).collect()
    }

    /// `e = l1 OR … OR e = lk`, as a short `is_in` list is simplified into.
    fn or_of_equalities(e: &ExprRef, items: Vec<ExprRef>) -> ExprRef {
        items
            .into_iter()
            .map(|item| e.clone().eq(item))
            .reduce(Expr::or)
            .unwrap()
    }

    #[rstest]
    fn an_or_of_equalities_is_estimated_as_its_in_list(
        #[values(1, 2, 3, 4, 5, 6, 50)] k: i64,
        #[values(resolved_col("a"), resolved_col("a").add(lit(1)))] e: ExprRef,
    ) {
        let in_list = e.clone().is_in(values(k));
        let or_form = or_of_equalities(&e, values(k));

        assert_eq!(
            estimated_selectivity(&or_form, &schema()),
            estimated_selectivity(&in_list, &schema())
        );
        assert_eq!(
            estimated_selectivity(&or_form.not(), &schema()),
            estimated_selectivity(&in_list.not(), &schema())
        );
    }

    #[rstest]
    #[case::literal_first(lit(1).eq(resolved_col("a")).or(lit(2).eq(resolved_col("a"))))]
    #[case::cast_literals(
        resolved_col("a")
            .eq(lit(1).cast(&DataType::Int64))
            .or(resolved_col("a").eq(lit(2).cast(&DataType::Int64)))
            .or(resolved_col("a").eq(lit(3).cast(&DataType::Int64)))
            .or(resolved_col("a").eq(lit(4).cast(&DataType::Int64)))
    )]
    #[case::nested(
        (resolved_col("s").eq(lit("x")).or(resolved_col("s").eq(lit("y"))))
            .or(resolved_col("s").eq(lit("z")).or(resolved_col("s").eq(lit("w"))))
    )]
    fn each_spelling_of_an_in_list_is_estimated_alike(#[case] or_form: ExprRef) {
        assert_eq!(
            estimated_selectivity(&or_form, &schema()),
            estimated_selectivity(&resolved_col("a").is_in(values(2)), &schema())
        );
    }

    /// An `or` that is not one expression's list of values keeps the general estimate.
    #[rstest]
    #[case::two_expressions(resolved_col("a").eq(lit(1)).or(resolved_col("b").eq(lit(2))))]
    #[case::mixed_comparisons(resolved_col("a").eq(lit(1)).or(resolved_col("a").gt(lit(5))))]
    #[case::null_safe(resolved_col("a").eq_null_safe(lit(1)).or(resolved_col("a").eq_null_safe(lit(2))))]
    #[case::two_columns_compared(resolved_col("a").eq(resolved_col("b")).or(resolved_col("a").eq(lit(1))))]
    fn other_ors_keep_the_general_estimate(#[case] or_form: ExprRef) {
        let Expr::BinaryOp { left, right, .. } = or_form.as_ref() else {
            unreachable!("each case is an or")
        };
        let (l, r) = (
            estimated_selectivity(left, &schema()),
            estimated_selectivity(right, &schema()),
        );

        assert_eq!(
            estimated_selectivity(&or_form, &schema()),
            l.mul_add(-r, l + r)
        );
    }
}
