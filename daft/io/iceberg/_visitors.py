from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any

from pyiceberg.expressions import (
    And,
    BooleanExpression,
    EqualTo,
    GreaterThan,
    GreaterThanOrEqual,
    In,
    IsNaN,
    IsNull,
    LessThan,
    LessThanOrEqual,
    Not,
    NotEqualTo,
    NotNaN,
    NotNull,
    Or,
    Reference,
    StartsWith,
)
from pyiceberg.expressions.literals import DateLiteral, Literal, StringLiteral, TimestampLiteral, literal
from pyiceberg.expressions.visitors import bind
from pyiceberg.types import TimestampType, TimestamptzType
from pyiceberg.utils.datetime import date_to_days, datetime_to_micros, days_to_date

from daft.expressions.visitor import ExpressionVisitor, PredicateVisitor

if TYPE_CHECKING:
    from pyiceberg.schema import Schema as IcebergSchema

    from daft.datatype import DataType
    from daft.expressions import Expression

try:
    from pyiceberg.types import TimestampNanoType, TimestamptzNanoType

    _TIMESTAMP_TYPES = (
        TimestampType,
        TimestamptzType,
        TimestampNanoType,
        TimestamptzNanoType,
    )
    _TIMESTAMPTZ_TYPES = (TimestamptzType, TimestamptzNanoType)
except ImportError:
    _TIMESTAMP_TYPES = (TimestampType, TimestamptzType)  # type: ignore[assignment]
    _TIMESTAMPTZ_TYPES = (TimestamptzType,)  # type: ignore[assignment]

_DATE_ONLY_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


class IcebergPredicateVisitor(PredicateVisitor[BooleanExpression]):
    def __init__(self, schema: IcebergSchema | None = None) -> None:
        self._schema = schema

    def visit_col(self, name: str) -> BooleanExpression:
        return Reference(name)

    def visit_lit(self, value: Any) -> BooleanExpression:
        if isinstance(value, datetime):
            return TimestampLiteral(datetime_to_micros(value))
        if isinstance(value, date):
            return DateLiteral(date_to_days(value))
        return literal(value)

    def visit_alias(self, expr: Expression, alias: str) -> BooleanExpression:
        return self.visit(expr)

    def visit_cast(self, expr: Expression, dtype: DataType) -> BooleanExpression:
        return self._visit_converted(expr, dtype, strict=True)

    def visit_try_cast(self, expr: Expression, dtype: DataType) -> BooleanExpression:
        return self._visit_converted(expr, dtype, strict=False)

    def visit_function(self, name: str, args: list[Expression]) -> BooleanExpression:
        # `is_nan`/`not_nan` have no dedicated `visit_*` hook on the base visitor,
        # so they arrive here as generic function calls with a single argument.
        if name in ("is_nan", "not_nan") and len(args) == 1:
            ref = self.visit_as_ref(args[0])
            return IsNaN(term=ref) if name == "is_nan" else NotNaN(term=ref)
        raise ValueError(f"Iceberg does not support function '{name}' in filter expressions")

    def visit_coalesce(self, args: list[Expression]) -> BooleanExpression:
        raise ValueError("Iceberg does not support coalesce in filter expressions")

    def visit_and(self, left: Expression, right: Expression) -> BooleanExpression:
        return And(left=self.visit(left), right=self.visit(right))

    def visit_or(self, left: Expression, right: Expression) -> BooleanExpression:
        return Or(left=self.visit(left), right=self.visit(right))

    def visit_not(self, expr: Expression) -> BooleanExpression:
        return Not(child=self.visit(expr))

    def visit_equal(self, left: Expression, right: Expression) -> BooleanExpression:
        ref, lit, _ = self.visit_lhs_rhs(left, right)
        return EqualTo(term=ref, literal=lit)

    def visit_not_equal(self, left: Expression, right: Expression) -> BooleanExpression:
        ref, lit, _ = self.visit_lhs_rhs(left, right)
        return NotEqualTo(term=ref, literal=lit)

    def visit_less_than(self, left: Expression, right: Expression) -> BooleanExpression:
        ref, lit, swapped = self.visit_lhs_rhs(left, right)
        if swapped:
            return GreaterThan(term=ref, literal=lit)
        else:
            return LessThan(term=ref, literal=lit)

    def visit_less_than_or_equal(self, left: Expression, right: Expression) -> BooleanExpression:
        ref, lit, swapped = self.visit_lhs_rhs(left, right)
        if swapped:
            return GreaterThanOrEqual(term=ref, literal=lit)
        else:
            return LessThanOrEqual(term=ref, literal=lit)

    def visit_greater_than(self, left: Expression, right: Expression) -> BooleanExpression:
        ref, lit, swapped = self.visit_lhs_rhs(left, right)
        if swapped:
            return LessThan(term=ref, literal=lit)
        else:
            return GreaterThan(term=ref, literal=lit)

    def visit_greater_than_or_equal(self, left: Expression, right: Expression) -> BooleanExpression:
        ref, lit, swapped = self.visit_lhs_rhs(left, right)
        if swapped:
            return LessThanOrEqual(term=ref, literal=lit)
        else:
            return GreaterThanOrEqual(term=ref, literal=lit)

    def visit_between(self, expr: Expression, lower: Expression, upper: Expression) -> BooleanExpression:
        ref = self.visit_as_ref(expr)
        lo = self.coerce(ref, self.visit_as_lit(lower))
        hi = self.coerce(ref, self.visit_as_lit(upper))
        return And(
            left=GreaterThanOrEqual(term=ref, literal=lo),
            right=LessThanOrEqual(term=ref, literal=hi),
        )

    def visit_is_in(self, expr: Expression, items: list[Expression]) -> BooleanExpression:
        ref = self.visit_as_ref(expr)
        literals = [self.coerce(ref, self.visit_as_lit(item)) for item in items]
        return In(term=ref, literals=set(literals))

    def visit_is_null(self, expr: Expression) -> BooleanExpression:
        return IsNull(term=self.visit_as_ref(expr))

    def visit_not_null(self, expr: Expression) -> BooleanExpression:
        return NotNull(term=self.visit_as_ref(expr))

    def visit_starts_with(self, input: Expression, prefix: Expression) -> BooleanExpression:
        ref = self.visit_as_ref(input)
        lit = self.visit_as_lit(prefix)
        return StartsWith(term=ref, literal=lit)

    ##
    # Helpers
    ##

    def _visit_converted(self, expr: Expression, dtype: DataType, *, strict: bool) -> BooleanExpression:
        """Translate a conversion only where the translation selects the same rows.

        A table predicate is evaluated against stored values, so a conversion of
        a column cannot simply be dropped: ``cast(x, int64) == 74`` holds for a
        stored ``74.5``, while ``x == 74`` does not, and pruning by the latter
        skips files that hold matching rows. A conversion is therefore kept only
        when it is a no-op on the stored column, or when it applies to a constant
        and can be folded into the constant it produces. Anything else is
        refused, which leaves that part of the filter to be applied after reading.
        """
        constant = _ConstantVisitor().converted(expr, dtype, strict=strict)
        if constant is not _NOT_CONSTANT:
            return self.visit_lit(constant)
        inner = self.visit(expr)
        if isinstance(inner, Reference) and self._stored_type(inner.name) == dtype:
            return inner
        raise ValueError(f"Iceberg cannot prune by a column converted to {dtype}")

    def _stored_type(self, name: str) -> DataType | None:
        """Return the type a column is read as, or ``None`` when it is not known."""
        if self._schema is None:
            return None
        from pyiceberg.io.pyarrow import schema_to_pyarrow
        from pyiceberg.schema import Schema

        from daft.datatype import DataType

        try:
            field = self._schema.find_field(name)
        except ValueError:
            return None
        if not field.field_type.is_primitive:
            return None
        return DataType.from_arrow_type(schema_to_pyarrow(Schema(field)).field(0).type)

    def coerce(self, ref: Reference, lit: Literal) -> Literal:
        """Coerce a literal to match the referenced column's type when pyiceberg can't."""
        if self._schema is None:
            return lit
        try:
            field = self._schema.find_field(ref.name)
        except ValueError:
            return lit
        field_type = field.field_type
        if not isinstance(field_type, _TIMESTAMP_TYPES):
            return lit
        # date-only string → full ISO-8601 timestamp string
        if isinstance(lit, StringLiteral) and _DATE_ONLY_RE.fullmatch(lit.value):
            suffix = "T00:00:00+00:00" if isinstance(field_type, _TIMESTAMPTZ_TYPES) else "T00:00:00"
            return literal(lit.value + suffix)
        # DateLiteral → TimestampLiteral via datetime
        if isinstance(lit, DateLiteral):
            dt = datetime.combine(days_to_date(lit.value), datetime.min.time())
            if isinstance(field_type, _TIMESTAMPTZ_TYPES):
                dt = dt.replace(tzinfo=timezone.utc)
            return TimestampLiteral(datetime_to_micros(dt))
        return lit

    def visit_lhs_rhs(
        self,
        lhs: Expression,
        rhs: Expression,
    ) -> tuple[Reference, Literal, bool]:
        """Visit a left-hand side and right-hand side expression, returning the reference, literal, and whether the left-hand side is the reference."""
        lv, rv = self.visit(lhs), self.visit(rhs)
        if isinstance(lv, Reference) and isinstance(rv, Literal):
            return lv, self.coerce(lv, rv), False
        if isinstance(rv, Reference) and isinstance(lv, Literal):
            return rv, self.coerce(rv, lv), True
        raise ValueError(
            f"Expected one column reference and one literal, got {type(lv).__name__} and {type(rv).__name__}"
        )

    def visit_as_ref(self, expr: Expression) -> Reference:
        result = self.visit(expr)
        if not isinstance(result, Reference):
            raise ValueError(f"Expected a column reference, got {type(result).__name__}")
        return result

    def visit_as_lit(self, expr: Expression) -> Literal:
        result = self.visit(expr)
        if not isinstance(result, Literal):
            raise ValueError(f"Expected a literal value, got {type(result).__name__}")
        return result


class IcebergPruningVisitor(IcebergPredicateVisitor):
    """Translate a filter into a predicate that holds for at least the rows it keeps.

    Such a predicate may only prune files, never rows the filter keeps, so a
    conjunct that cannot be translated or bound is dropped from an ``AND`` rather
    than losing the whole predicate. Dropping is sound only where weakening a
    part weakens the whole: under a ``NOT`` it would strengthen it, so there the
    translation is strict.
    """

    def __init__(self, schema: IcebergSchema | None = None) -> None:
        super().__init__(schema)
        self._weakening = True

    def visit_and(self, left: Expression, right: Expression) -> BooleanExpression:
        if not self._weakening:
            return super().visit_and(left, right)
        sides = [side for side in (self._bound_or_none(left), self._bound_or_none(right)) if side is not None]
        if not sides:
            raise ValueError("No part of the conjunction can be expressed as a table predicate")
        return sides[0] if len(sides) == 1 else And(left=sides[0], right=sides[1])

    def visit_not(self, expr: Expression) -> BooleanExpression:
        weakening, self._weakening = self._weakening, False
        try:
            return super().visit_not(expr)
        finally:
            self._weakening = weakening

    def _bound_or_none(self, expr: Expression) -> BooleanExpression | None:
        """Return the translated part if the table can bind it, else ``None``."""
        try:
            predicate = self.visit(expr)
            if self._schema is not None:
                bind(self._schema, predicate, case_sensitive=True)
        except (ValueError, TypeError, NotImplementedError):
            return None
        return predicate


_NOT_CONSTANT = object()


class _ConstantVisitor(ExpressionVisitor[Any]):
    """Return the value of a constant expression, or a sentinel when it is not one.

    Only literals and conversions of literals are constants here; that is the
    shape a typed constant takes in a filter.
    """

    def visit_col(self, name: str) -> Any:
        return _NOT_CONSTANT

    def visit_lit(self, value: Any) -> Any:
        return value

    def visit_alias(self, expr: Expression, alias: str) -> Any:
        return self.visit(expr)

    def visit_cast(self, expr: Expression, dtype: DataType) -> Any:
        return self.converted(expr, dtype, strict=True)

    def visit_try_cast(self, expr: Expression, dtype: DataType) -> Any:
        return self.converted(expr, dtype, strict=False)

    def visit_function(self, name: str, args: list[Expression]) -> Any:
        return _NOT_CONSTANT

    def visit_coalesce(self, args: list[Expression]) -> Any:
        return _NOT_CONSTANT

    def converted(self, expr: Expression, dtype: DataType, *, strict: bool) -> Any:
        """Return the value of ``expr`` converted to ``dtype``, or the sentinel."""
        from daft.series import Series

        value = self.visit(expr)
        if value is _NOT_CONSTANT:
            return _NOT_CONSTANT
        source = Series.from_pylist([value])
        return (source.cast(dtype) if strict else source.try_cast(dtype)).to_pylist()[0]
