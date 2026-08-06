from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from riskprobe.models import Condition, RiskRule


@dataclass(frozen=True, slots=True)
class SyntheticTruth:
    hidden_rules: tuple[RiskRule, ...]


_HIDDEN_RULES = (
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


def generate_behavior_dataset(rows: int, seed: int) -> tuple[pl.DataFrame, SyntheticTruth]:
    if isinstance(rows, bool) or not isinstance(rows, int):
        raise TypeError("rows must be a positive integer")
    if rows <= 0:
        raise ValueError("rows must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be a non-negative integer")
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")

    rng = np.random.default_rng(seed)
    institutions = np.array(["bank_north", "bank_south", "bank_east", "bank_west"])
    institution_index = rng.integers(0, len(institutions), size=rows)
    month_index = rng.integers(0, 12, size=rows)

    order_cnt_30d = rng.poisson(3.2, size=rows)
    order_cnt_7d = rng.binomial(order_cnt_30d, 7 / 30)
    order_amount_30d = rng.gamma(shape=1.8, scale=85.0, size=rows) * order_cnt_30d
    order_cancel_rate_30d = np.clip(rng.beta(1.8, 5.2, size=rows), 0.0, 1.0)

    browse_pv_30d = rng.poisson(34.0, size=rows)
    browse_pv_7d = rng.binomial(browse_pv_30d, 7 / 30)
    browse_days_30d = rng.binomial(30, np.clip(browse_pv_30d / 90, 0.04, 0.82))
    browse_night_ratio_30d = np.clip(rng.beta(2.0, 4.5, size=rows), 0.0, 1.0)
    browse_to_order_ratio_30d = browse_pv_30d / np.maximum(order_cnt_30d, 1)
    multi_platform_cnt_30d = rng.choice(
        np.array([1, 2, 3, 4, 5]), size=rows, p=np.array([0.18, 0.3, 0.28, 0.16, 0.08])
    )

    cancellation_signal = order_cancel_rate_30d > 0.45
    browsing_signal = (browse_night_ratio_30d > 0.55) & (browse_to_order_ratio_30d > 8)
    platform_signal = (multi_platform_cnt_30d >= 4) & (order_cnt_30d <= 1)
    institution_offset = np.array([-0.12, 0.04, 0.1, -0.02])[institution_index]
    month_offset = (month_index - 5.5) * 0.015
    log_odds = (
        -2.75
        + 1.5 * cancellation_signal
        + 1.25 * browsing_signal
        + 1.4 * platform_signal
        + institution_offset
        + month_offset
    )
    bad_probability = 1.0 / (1.0 + np.exp(-log_odds))
    target = rng.binomial(1, bad_probability, size=rows).astype(np.int8)

    snapshot_dates = [date(2024, int(month) + 1, 1) for month in month_index]
    frame = pl.DataFrame(
        {
            "entity_id": np.arange(1, rows + 1, dtype=np.int64),
            "snapshot_date": snapshot_dates,
            "institution": institutions[institution_index],
            "target": target,
            "order_cnt_7d": order_cnt_7d,
            "order_cnt_30d": order_cnt_30d,
            "order_amount_30d": order_amount_30d.round(2),
            "order_cancel_rate_30d": order_cancel_rate_30d,
            "browse_pv_7d": browse_pv_7d,
            "browse_pv_30d": browse_pv_30d,
            "browse_days_30d": browse_days_30d,
            "browse_night_ratio_30d": browse_night_ratio_30d,
            "browse_to_order_ratio_30d": browse_to_order_ratio_30d,
            "multi_platform_cnt_30d": multi_platform_cnt_30d,
            "emb_00": rng.normal(size=rows),
            "emb_01": rng.normal(size=rows),
        }
    )
    return frame, SyntheticTruth(hidden_rules=_HIDDEN_RULES)
