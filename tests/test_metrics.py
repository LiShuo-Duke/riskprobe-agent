import math

import numpy as np
import pytest
from pydantic import ValidationError

from riskprobe.metrics import adjust_pvalues, bootstrap_lift_ci, compute_rule_metrics
from riskprobe.models import RuleMetrics, SliceMetrics


def _sample_rule_metrics() -> RuleMetrics:
    return compute_rule_metrics(
        np.array([True, False]),
        np.array([1, 0]),
        positive_value=1,
    )


def test_slice_metrics_accepts_segment() -> None:
    slice_metrics = SliceMetrics(
        slice_type="segment",
        slice_value="small_business",
        metrics=_sample_rule_metrics(),
    )

    assert slice_metrics.slice_type == "segment"


def test_slice_metrics_rejects_institution() -> None:
    with pytest.raises(ValidationError):
        SliceMetrics(
            slice_type="institution",
            slice_value="bank_a",
            metrics=_sample_rule_metrics(),
        )


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


def test_bootstrap_lift_ci_matches_manual_stratified_low_positive_rate() -> None:
    target = np.array([1] + [0] * 19)
    mask = np.array([True, True, True, True, True] + [False] * 15)
    rounds = 40

    actual = bootstrap_lift_ci(
        mask,
        target,
        positive_value=1,
        rounds=rounds,
        random_seed=42,
    )

    rng = np.random.default_rng(42)
    positive_indices = np.flatnonzero(target == 1)
    negative_indices = np.flatnonzero(target != 1)
    manual_lifts = []
    for _ in range(rounds):
        sampled_indices = np.concatenate(
            (
                rng.choice(positive_indices, size=len(positive_indices), replace=True),
                rng.choice(negative_indices, size=len(negative_indices), replace=True),
            )
        )
        manual_lifts.append(
            compute_rule_metrics(
                mask[sampled_indices],
                target[sampled_indices],
                positive_value=1,
            ).lift
        )
    expected = np.quantile(manual_lifts, [0.025, 0.975])

    assert actual == pytest.approx(expected)


def test_bootstrap_lift_ci_positive_only_returns_finite_interval() -> None:
    interval = bootstrap_lift_ci(
        np.array([True, True, True]),
        np.array([1, 1, 1]),
        positive_value=1,
        rounds=20,
        random_seed=42,
    )

    assert interval == (1.0, 1.0)
    assert all(math.isfinite(bound) for bound in interval)


def test_adjust_pvalues_uses_benjamini_hochberg() -> None:
    adjusted = adjust_pvalues([0.01, 0.02, 0.2])

    assert adjusted == pytest.approx([0.03, 0.03, 0.2])


def test_adjust_pvalues_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="^p-values must be finite$"):
        adjust_pvalues([0.1, math.inf])


def test_rule_metrics_rejects_non_boolean_mask_values() -> None:
    with pytest.raises(ValueError, match="^mask must contain only boolean values$"):
        compute_rule_metrics(np.array([True, math.nan]), np.array([1, 0]), positive_value=1)


def test_balanced_weights_use_train_class_counts_only() -> None:
    from riskprobe.metrics import balanced_class_weights, balanced_sample_weights

    target = np.array([0, 0, 0, 1], dtype=np.int8)

    assert balanced_class_weights(target) == {0: 2 / 3, 1: 2.0}
    np.testing.assert_allclose(
        balanced_sample_weights(target),
        np.array([2 / 3, 2 / 3, 2 / 3, 2.0]),
    )


@pytest.mark.parametrize("target", [[], [0, 0], [1, 1], [0, 2], [0, np.nan]])
def test_balanced_weights_fail_closed_for_invalid_or_single_class_target(target: object) -> None:
    from riskprobe.metrics import balanced_sample_weights

    with pytest.raises(ValueError, match="binary target"):
        balanced_sample_weights(np.asarray(target, dtype=object))


def test_rule_metrics_expose_bad_good_ks_separation() -> None:
    metrics = compute_rule_metrics(
        np.array([True, True, False, False]),
        np.array([1, 0, 0, 0]),
        positive_value=1,
    )

    assert metrics.hit_good_rate == pytest.approx(1 / 3)
    assert metrics.ks_signed == pytest.approx(1 / 6)
    assert metrics.ks_stat == pytest.approx(1 / 6)
    assert metrics.p_value != metrics.ks_stat


def test_score_ks_filters_nonfinite_scores_and_preserves_direction() -> None:
    from riskprobe.metrics import compute_score_ks

    result = compute_score_ks(
        np.array([0.9, 0.8, np.nan, 0.1, 0.2]),
        np.array([1, 1, 0, 0, 0]),
    )

    assert result.statistic == pytest.approx(1.0)
    assert result.signed_statistic == pytest.approx(1.0)
    assert result.bad_count == 2
    assert result.good_count == 2
    assert result.excluded_count == 1
    assert result.p_value is not None


def test_score_ks_returns_unavailable_for_single_class_after_filtering() -> None:
    from riskprobe.metrics import compute_score_ks

    result = compute_score_ks([0.1, float("nan")], [1, 1])

    assert result.statistic is None
    assert result.p_value is None
    assert result.limitation == "single_class_or_no_finite_scores"


def test_score_ks_orients_signed_statistic_for_lower_bad_direction() -> None:
    from riskprobe.metrics import compute_score_ks

    result = compute_score_ks(
        [0.1, 0.2, 0.8, 0.9],
        [1, 1, 0, 0],
        direction="lower_is_bad",
    )

    assert result.statistic == pytest.approx(1.0)
    assert result.signed_statistic == pytest.approx(1.0)


def test_score_ks_rejects_non_binary_target() -> None:
    from riskprobe.metrics import compute_score_ks

    with pytest.raises(ValueError, match="binary target"):
        compute_score_ks([0.1, 0.2], [0, 2])
