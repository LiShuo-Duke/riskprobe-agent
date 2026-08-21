from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import fisher_exact, ks_2samp
from statsmodels.stats.multitest import multipletests

from riskprobe.models import RuleMetrics, ScoreSeparation


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


_BINARY_TARGET_ERROR = (
    "target must be a non-empty one-dimensional finite binary target containing 0 and 1"
)


def _validated_binary_target(target: Any) -> np.ndarray:
    try:
        target_array = np.asarray(target)
    except (TypeError, ValueError) as error:
        raise ValueError(_BINARY_TARGET_ERROR) from error
    if target_array.ndim != 1 or target_array.size == 0:
        raise ValueError(_BINARY_TARGET_ERROR)
    if target_array.dtype.kind not in "iuf":
        raise ValueError(_BINARY_TARGET_ERROR)
    if not np.all(np.isfinite(target_array)) or not np.all(
        (target_array == 0) | (target_array == 1)
    ):
        raise ValueError(_BINARY_TARGET_ERROR)
    if not np.any(target_array == 0) or not np.any(target_array == 1):
        raise ValueError(_BINARY_TARGET_ERROR)
    return target_array


def balanced_class_weights(target: Any) -> dict[int, float]:
    target_array = _validated_binary_target(target)
    sample_count = target_array.size
    return {
        class_value: float(
            sample_count / (2 * int(np.count_nonzero(target_array == class_value)))
        )
        for class_value in (0, 1)
    }


def balanced_sample_weights(target: Any) -> np.ndarray:
    target_array = _validated_binary_target(target)
    class_weights = balanced_class_weights(target_array)
    return np.where(
        target_array == 0,
        class_weights[0],
        class_weights[1],
    )


def compute_rule_metrics(mask: Any, target: Any, positive_value: Any) -> RuleMetrics:
    mask_array, target_array = _validated_arrays(mask, target)
    positive = target_array == positive_value
    positive_count = int(np.count_nonzero(positive))
    if positive_count == 0:
        raise ValueError("target has no positive samples")

    sample_count = len(target_array)
    negative_count = sample_count - positive_count
    support_count = int(np.count_nonzero(mask_array))
    hit_positive = int(np.count_nonzero(mask_array & positive))
    hit_negative = support_count - hit_positive
    non_hit_positive = positive_count - hit_positive
    non_hit_count = sample_count - support_count
    non_hit_negative = non_hit_count - non_hit_positive

    coverage = support_count / sample_count
    base_bad_rate = positive_count / sample_count
    hit_bad_rate = hit_positive / support_count if support_count else 0.0
    hit_good_rate = hit_negative / negative_count if negative_count else 0.0
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
    ks_signed = (
        None if negative_count == 0 else float(hit_bad_rate - hit_good_rate)
    )
    ks_stat = None if ks_signed is None else float(abs(ks_signed))

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
        hit_good_rate=hit_good_rate,
        ks_signed=ks_signed,
        ks_stat=ks_stat,
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
    positive_indices = np.flatnonzero(target_array == positive_value)
    negative_indices = np.flatnonzero(target_array != positive_value)
    lifts: list[float] = []
    for _ in range(rounds):
        sampled_positive_indices = rng.choice(
            positive_indices,
            size=len(positive_indices),
            replace=True,
        )
        if len(negative_indices):
            sampled_negative_indices = rng.choice(
                negative_indices,
                size=len(negative_indices),
                replace=True,
            )
            sampled_indices = np.concatenate(
                (sampled_positive_indices, sampled_negative_indices)
            )
        else:
            sampled_indices = sampled_positive_indices
        sampled_metrics = compute_rule_metrics(
            mask_array[sampled_indices],
            target_array[sampled_indices],
            positive_value,
        )
        lifts.append(sampled_metrics.lift)

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


_SCORE_KS_LIMITATION = "single_class_or_no_finite_scores"


def _validated_score_inputs(
    scores: Any,
    target: Any,
    positive_value: Any,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        score_array = np.asarray(scores, dtype=np.float64)
        target_array = np.asarray(target)
    except (TypeError, ValueError) as error:
        raise ValueError("scores and target must be one-dimensional arrays") from error
    if score_array.ndim != 1 or target_array.ndim != 1:
        raise ValueError("scores and target must be one-dimensional")
    if len(score_array) != len(target_array):
        raise ValueError("scores and target must have the same length")
    if len(score_array) == 0:
        raise ValueError("scores and target must not be empty")
    try:
        target_numeric = np.asarray(target_array, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(_BINARY_TARGET_ERROR) from error
    if (
        not np.all(np.isfinite(target_numeric))
        or not np.isin(target_numeric, (0.0, 1.0)).all()
        or isinstance(positive_value, bool)
        or positive_value not in (0, 1)
    ):
        raise ValueError(_BINARY_TARGET_ERROR)
    return score_array, target_numeric.astype(np.int8)


def _signed_ks_statistic(
    bad_scores: np.ndarray,
    good_scores: np.ndarray,
    direction: str,
) -> tuple[float, float]:
    combined = np.unique(np.concatenate((bad_scores, good_scores)))
    bad_sorted = np.sort(bad_scores)
    good_sorted = np.sort(good_scores)
    bad_cdf = np.searchsorted(bad_sorted, combined, side="right") / len(bad_sorted)
    good_cdf = np.searchsorted(good_sorted, combined, side="right") / len(good_sorted)
    differences = good_cdf - bad_cdf
    location = int(np.argmax(np.abs(differences)))
    statistic = float(np.max(np.abs(differences)))
    signed = float(differences[location])
    if direction == "lower_is_bad":
        signed = -signed
    return statistic, signed


def compute_score_ks(
    scores: Any,
    target: Any,
    *,
    positive_value: Any = 1,
    direction: str = "higher_is_bad",
) -> ScoreSeparation:
    """Compare frozen continuous scores between bad and good target classes."""

    if direction not in {"higher_is_bad", "lower_is_bad"}:
        raise ValueError("direction must be higher_is_bad or lower_is_bad")
    score_array, target_array = _validated_score_inputs(
        scores,
        target,
        positive_value,
    )
    finite = np.isfinite(score_array)
    filtered_scores = score_array[finite]
    filtered_target = target_array[finite]
    bad = filtered_target == positive_value
    bad_scores = filtered_scores[bad]
    good_scores = filtered_scores[~bad]
    bad_count = int(bad_scores.size)
    good_count = int(good_scores.size)
    excluded_count = int(np.count_nonzero(~finite))
    if bad_count == 0 or good_count == 0:
        return ScoreSeparation(
            statistic=None,
            signed_statistic=None,
            p_value=None,
            bad_count=bad_count,
            good_count=good_count,
            excluded_count=excluded_count,
            method="ks_2samp",
            limitation=_SCORE_KS_LIMITATION,
        )

    result = ks_2samp(bad_scores, good_scores, alternative="two-sided")
    statistic, signed_statistic = _signed_ks_statistic(
        bad_scores,
        good_scores,
        direction,
    )
    return ScoreSeparation(
        statistic=statistic,
        signed_statistic=signed_statistic,
        p_value=float(result.pvalue),
        bad_count=bad_count,
        good_count=good_count,
        excluded_count=excluded_count,
        method="ks_2samp",
        limitation=None,
    )
