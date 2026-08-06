import math

import numpy as np
import pytest
from pydantic import ValidationError

from riskprobe.metrics import adjust_pvalues, bootstrap_lift_ci, compute_rule_metrics
from riskprobe.models import RuleMetrics


def test_rule_metrics_match_hand_calculation() -> None:
    target = np.array([1, 1, 0, 0, 0, 0])
    mask = np.array([True, True, True, False, False, False])

    metrics = compute_rule_metrics(mask, target, positive_value=1)

    assert metrics.support_count == 3
    assert metrics.coverage == 0.5
    assert metrics.hit_bad_rate == 2 / 3
    assert metrics.base_bad_rate == 1 / 3
    assert metrics.lift == 2.0
    assert metrics.precision == 2 / 3
    assert metrics.recall == 1.0


def test_rule_metrics_reject_target_without_positive_samples() -> None:
    with pytest.raises(ValueError, match="^target has no positive samples$"):
        compute_rule_metrics(
            np.array([True, False]),
            np.array([0, 0]),
            positive_value=1,
        )


def test_rule_metrics_reject_empty_input() -> None:
    with pytest.raises(ValueError, match="^mask and target must not be empty$"):
        compute_rule_metrics(np.array([], dtype=bool), np.array([]), positive_value=1)


def test_rule_metrics_zero_support_has_finite_zero_rates() -> None:
    metrics = compute_rule_metrics(
        np.array([False, False, False]),
        np.array([1, 0, 0]),
        positive_value=1,
    )

    assert metrics.support_count == 0
    assert metrics.coverage == 0.0
    assert metrics.hit_bad_rate == 0.0
    assert metrics.lift == 0.0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert all(
        math.isfinite(value)
        for value in (
            metrics.coverage,
            metrics.base_bad_rate,
            metrics.hit_bad_rate,
            metrics.non_hit_bad_rate,
            metrics.lift,
            metrics.precision,
            metrics.recall,
            metrics.p_value,
        )
    )


def test_rule_metrics_all_hit_has_finite_non_hit_rate() -> None:
    metrics = compute_rule_metrics(
        np.array([True, True, True]),
        np.array([1, 0, 0]),
        positive_value=1,
    )

    assert metrics.non_hit_bad_rate == 0.0
    assert math.isfinite(metrics.non_hit_bad_rate)


def test_rule_metrics_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="^mask and target must have the same length$"):
        compute_rule_metrics(np.array([True]), np.array([1, 0]), positive_value=1)


def test_rule_metrics_model_rejects_non_finite_values() -> None:
    values = {
        "support_count": 1,
        "coverage": math.nan,
        "base_bad_rate": 0.5,
        "hit_bad_rate": 0.5,
        "non_hit_bad_rate": 0.5,
        "lift": 1.0,
        "precision": 0.5,
        "recall": 0.5,
        "p_value": 1.0,
    }

    with pytest.raises(ValidationError):
        RuleMetrics(**values)


def test_bootstrap_lift_ci_is_seeded_and_finite() -> None:
    target = np.array([1, 1, 0, 0, 0, 0])
    mask = np.array([True, True, True, False, False, False])

    first = bootstrap_lift_ci(mask, target, positive_value=1, rounds=100, random_seed=42)
    second = bootstrap_lift_ci(mask, target, positive_value=1, rounds=100, random_seed=42)

    assert first == second
    assert first[0] <= first[1]
    assert all(math.isfinite(bound) for bound in first)


def test_adjust_pvalues_uses_benjamini_hochberg() -> None:
    adjusted = adjust_pvalues([0.01, 0.02, 0.2])

    assert adjusted == pytest.approx([0.03, 0.03, 0.2])


def test_adjust_pvalues_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="^p-values must be finite$"):
        adjust_pvalues([0.1, math.inf])


def test_rule_metrics_rejects_non_boolean_mask_values() -> None:
    with pytest.raises(ValueError, match="^mask must contain only boolean values$"):
        compute_rule_metrics(np.array([True, math.nan]), np.array([1, 0]), positive_value=1)
