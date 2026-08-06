from collections.abc import Callable

import polars as pl

from riskprobe.models import Condition, RiskRule

Comparison = Callable[[pl.Expr, object], pl.Expr]

_COMPARISONS: dict[str, Comparison] = {
    ">": lambda column, value: column > value,
    ">=": lambda column, value: column >= value,
    "<": lambda column, value: column < value,
    "<=": lambda column, value: column <= value,
    "==": lambda column, value: column == value,
    "!=": lambda column, value: column != value,
}


def _condition_expression(condition: Condition, reject_nan: bool) -> pl.Expr:
    column = pl.col(condition.feature)
    if condition.operator == "is_null":
        return column.is_null()
    comparison = _COMPARISONS[condition.operator](column, condition.value)
    return comparison & ~column.is_nan() if reject_nan else comparison


def evaluate_rule(frame: pl.DataFrame, rule: RiskRule) -> pl.Series:
    if not rule.conditions:
        return pl.Series("rule_match", [True] * frame.height, dtype=pl.Boolean)

    def expression_for(condition: Condition) -> pl.Expr:
        dtype = frame.schema.get(condition.feature)
        return _condition_expression(condition, dtype in (pl.Float32, pl.Float64))

    expression = expression_for(rule.conditions[0])
    for condition in rule.conditions[1:]:
        expression = expression & expression_for(condition)

    return frame.select(expression.fill_null(False).alias("rule_match")).to_series()
