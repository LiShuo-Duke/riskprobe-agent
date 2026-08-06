from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from riskprobe.models import RuleMetrics


def _validated_arrays(mask: Any, target: Any) -> tuple[np.ndarray, np.ndarray]:
    mask_array = np.asarray(mask)
    target_array = np.asarray(target)
    if mask_array.ndim != 1 or target_array.ndim != 1:
        raise ValueError("mask and target must be one-dimensional")
    if len(mask_array) != len(target_array):
        raise ValueError("mask and target must have the same length")
    if len(mask_array) == 0:
        raise ValueError("mask and target must not be empty")
    if mask_array.dtype != np.dtype(np.bool_):
        raise ValueError("mask must contain only boolean values")
    return mask_array, target_array


def compute_rule_metrics(mask: Any, target: Any, positive_value: Any) -> RuleMetrics:
    mask_array, target_array = _validated_arrays(mask, target)
    positive = target_array == positive_value
    positive_count = int(np.count_nonzero(positive))
    if positive_count == 0:
        raise ValueError("target has no positive samples")

    sample_count = len(target_array)
    support_count = int(np.count_nonzero(mask_array))
    hit_positive = int(np.count_nonzero(mask_array & positive))
    hit_negative = support_count - hit_positive
    non_hit_positive = positive_count - hit_positive
    non_hit_count = sample_count - support_count
    non_hit_negative = non_hit_count - non_hit_positive

    coverage = support_count / sample_count
    base_bad_rate = positive_count / sample_count
    hit_bad_rate = hit_positive / support_count if support_count else 0.0
    non_hit_bad_rate = non_hit_positive / non_hit_count if non_hit_count else 0.0
    lift = hit_bad_rate / base_bad_rate
    precision = hit_bad_rate
    recall = hit_positive / positive_count
    p_value = float(
        fisher_exact(
            [[hit_positive, hit_negative], [non_hit_positive, non_hit_negative]],
            alternative="two-sided",
        ).pvalue
    )

    return RuleMetrics(
        support_count=support_count,
        coverage=coverage,
        base_bad_rate=base_bad_rate,
        hit_bad_rate=hit_bad_rate,
        non_hit_bad_rate=non_hit_bad_rate,
        lift=lift,
        precision=precision,
        recall=recall,
        p_value=p_value,
    )


def bootstrap_lift_ci(
    mask: Any,
    target: Any,
    positive_value: Any,
    rounds: int = 500,
    random_seed: int = 42,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    mask_array, target_array = _validated_arrays(mask, target)
    if not np.any(target_array == positive_value):
        raise ValueError("target has no positive samples")
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if random_seed != 42:
        raise ValueError("random_seed must be 42")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    rng = np.random.default_rng(random_seed)
    lifts: list[float] = []
    for _ in range(rounds):
        indices = rng.integers(0, len(target_array), size=len(target_array))
        sampled_target = target_array[indices]
        if not np.any(sampled_target == positive_value):
            continue
        sampled_metrics = compute_rule_metrics(
            mask_array[indices],
            sampled_target,
            positive_value,
        )
        lifts.append(sampled_metrics.lift)

    if not lifts:
        raise ValueError("bootstrap samples contain no positive samples")
    tail_probability = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(lifts, [tail_probability, 1.0 - tail_probability])
    return float(lower), float(upper)


def adjust_pvalues(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("p-values must be one-dimensional")
    if values.size == 0:
        return []
    if not np.all(np.isfinite(values)):
        raise ValueError("p-values must be finite")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be between 0 and 1")
    return multipletests(values, method="fdr_bh")[1].tolist()
