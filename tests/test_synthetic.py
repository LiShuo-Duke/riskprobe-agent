import polars as pl
import pytest

from riskprobe.models import Condition, RiskRule
from riskprobe.synthetic import generate_behavior_dataset


EXPECTED_COLUMNS = [
    "entity_id",
    "snapshot_date",
    "institution",
    "target",
    "order_cnt_7d",
    "order_cnt_30d",
    "order_amount_30d",
    "order_cancel_rate_30d",
    "browse_pv_7d",
    "browse_pv_30d",
    "browse_days_30d",
    "browse_night_ratio_30d",
    "browse_to_order_ratio_30d",
    "multi_platform_cnt_30d",
    "emb_00",
    "emb_01",
]

EXPECTED_RULES = (
    RiskRule(
        rule_id="hidden_order_cancellation",
        conditions=(
            Condition(feature="order_cancel_rate_30d", operator=">", value=0.45),
        ),
        origin="synthetic_truth",
    ),
    RiskRule(
        rule_id="hidden_night_browsing",
        conditions=(
            Condition(feature="browse_night_ratio_30d", operator=">", value=0.55),
            Condition(feature="browse_to_order_ratio_30d", operator=">", value=8),
        ),
        origin="synthetic_truth",
    ),
    RiskRule(
        rule_id="hidden_multi_platform_low_order",
        conditions=(
            Condition(feature="multi_platform_cnt_30d", operator=">=", value=4),
            Condition(feature="order_cnt_30d", operator="<=", value=1),
        ),
        origin="synthetic_truth",
    ),
)


def test_synthetic_behavior_is_reproducible_and_contains_hidden_rules() -> None:
    first, truth = generate_behavior_dataset(rows=2_000, seed=42)
    second, _ = generate_behavior_dataset(rows=2_000, seed=42)

    assert first.equals(second)
    assert 0.05 < first["target"].mean() < 0.35
    assert {"order_cancel_rate_30d", "browse_night_ratio_30d"}.issubset(first.columns)
    assert len(truth.hidden_rules) == 3


def test_synthetic_dataset_has_stable_schema_and_hidden_rule_expressions() -> None:
    frame, truth = generate_behavior_dataset(rows=100, seed=42)

    assert frame.columns == EXPECTED_COLUMNS
    assert frame.schema["snapshot_date"] == pl.Date
    assert frame.schema["target"] in (pl.Int8, pl.Int32, pl.Int64)
    assert truth.hidden_rules == EXPECTED_RULES


def test_synthetic_window_and_ratio_invariants_hold() -> None:
    frame, _ = generate_behavior_dataset(rows=5_000, seed=42)

    assert (frame["order_cnt_7d"] <= frame["order_cnt_30d"]).all()
    assert (frame["browse_pv_7d"] <= frame["browse_pv_30d"]).all()
    assert (frame["browse_days_30d"] <= 30).all()
    for column in ("order_cancel_rate_30d", "browse_night_ratio_30d"):
        assert frame[column].is_between(0.0, 1.0, closed="both").all()


@pytest.mark.parametrize("rows", [0, -1])
def test_synthetic_rejects_non_positive_rows(rows: int) -> None:
    with pytest.raises(ValueError, match="rows must be a positive integer"):
        generate_behavior_dataset(rows=rows, seed=42)


@pytest.mark.parametrize("rows", [True, 1.5, "10"])
def test_synthetic_rejects_non_integer_rows(rows: object) -> None:
    with pytest.raises(TypeError, match="rows must be a positive integer"):
        generate_behavior_dataset(rows=rows, seed=42)  # type: ignore[arg-type]


def test_non_default_seed_is_supported_and_reproducible() -> None:
    first, first_truth = generate_behavior_dataset(rows=500, seed=7)
    second, second_truth = generate_behavior_dataset(rows=500, seed=7)
    default, _ = generate_behavior_dataset(rows=500, seed=42)

    assert first.equals(second)
    assert not first.equals(default)
    assert first_truth == second_truth
    assert first_truth.hidden_rules == EXPECTED_RULES
