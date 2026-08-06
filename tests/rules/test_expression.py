import polars as pl
import pytest
from pydantic import ValidationError

from riskprobe.models import Condition, RiskRule
from riskprobe.rules.expression import evaluate_rule


def test_two_condition_rule_handles_null_as_false() -> None:
    frame = pl.DataFrame({"a": [1.0, 4.0, None], "b": [0.1, 0.3, 0.2]})
    rule = RiskRule(
        rule_id="R_test",
        conditions=(
            Condition(feature="a", operator=">", value=3.0),
            Condition(feature="b", operator="<=", value=0.3),
        ),
        origin="quantile_pair",
    )

    assert evaluate_rule(frame, rule).to_list() == [False, True, False]


@pytest.mark.parametrize(
    ("operator", "value", "expected"),
    [
        (">", 2, [False, False, True, False]),
        (">=", 2, [False, True, True, False]),
        ("<", 2, [True, False, False, False]),
        ("<=", 2, [True, True, False, False]),
        ("==", 2, [False, True, False, False]),
        ("!=", 2, [True, False, True, False]),
        ("is_null", None, [False, False, False, True]),
    ],
)
def test_rule_supports_each_condition_operator(
    operator: str, value: int | None, expected: list[bool]
) -> None:
    frame = pl.DataFrame({"feature": [1, 2, 3, None]})
    condition = Condition.model_validate(
        {"feature": "feature", "operator": operator, "value": value}
    )
    rule = RiskRule(rule_id="R_operator", conditions=(condition,), origin="test")

    assert evaluate_rule(frame, rule).to_list() == expected


def test_rule_models_are_strict_and_frozen() -> None:
    with pytest.raises(ValidationError):
        Condition.model_validate({"feature": 1, "operator": ">", "value": 3})

    condition = Condition(feature="a", operator=">", value=3)
    with pytest.raises(ValidationError):
        condition.value = 4


def test_numeric_nan_does_not_match_comparison_rule() -> None:
    frame = pl.DataFrame({"feature": [1.0, float("nan"), None]})
    rule = RiskRule(
        rule_id="R_nan",
        conditions=(Condition(feature="feature", operator=">", value=0.0),),
        origin="test",
    )

    assert evaluate_rule(frame, rule).to_list() == [True, False, False]
