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
