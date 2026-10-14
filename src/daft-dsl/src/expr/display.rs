use std::fmt::{Display, Write};

use itertools::Itertools;

use super::{Expr, ExprRef, Operator};

/// Display for Expr::BinaryOp
#[recursive::recursive]
pub fn expr_binary_op_display_without_formatter(
    op: &Operator,
    left: &ExprRef,
    right: &ExprRef,
) -> std::result::Result<String, std::fmt::Error> {
    let mut f = String::default();
    let write_out_expr = |f: &mut String, input: &Expr| match input {
        Expr::Alias(e, _) => write!(f, "{e}"),
        Expr::BinaryOp { .. } => write!(f, "[{input}]"),
        _ => write!(f, "{input}"),
    };
    write_out_expr(&mut f, left)?;
    write!(&mut f, " {op} ")?;
    write_out_expr(&mut f, right)?;
    Ok(f)
}

/// Display for Expr::IsIn
#[recursive::recursive]
pub fn expr_is_in_display_without_formatter(
    expr: &ExprRef,
    inputs: &[ExprRef],
) -> std::result::Result<String, std::fmt::Error> {
    let mut f = String::default();
    write!(&mut f, "{expr} IN (")?;
    for (i, input) in inputs.iter().enumerate() {
        if i != 0 {
            write!(&mut f, ", ")?;
        }
        write!(&mut f, "{input}")?;
    }
    write!(&mut f, ")")?;
    Ok(f)
}

/// Display for Expr::List
#[recursive::recursive]
pub fn expr_list_display_without_formatter(
    items: &[ExprRef],
) -> std::result::Result<String, std::fmt::Error> {
    let mut f = String::default();
    write!(
        &mut f,
        "list({})",
        items.iter().map(|x| x.to_string()).join(", ")
    )?;
    Ok(f)
}

/// Renders `T` on a stack grown when the current one runs low.
///
/// An expression renders each of its children inside its own rendering, so
/// a deep expression recurses once per level; this keeps that recursion from
/// overflowing the thread it renders on.
pub struct GrowingStack<T>(pub T);

impl<T: Display> Display for GrowingStack<T> {
    #[recursive::recursive]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
