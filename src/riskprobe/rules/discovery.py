import hashlib
import json
import math
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Any

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier

from riskprobe.config import DiscoveryConfig
from riskprobe.metrics import compute_rule_metrics
from riskprobe.models import Condition, RiskRule, RuleMetrics
from riskprobe.rules.expression import evaluate_rule

_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


@dataclass(frozen=True, slots=True)
class _Candidate:
    rule: RiskRule
    metrics: RuleMetrics
    mask: np.ndarray[Any, np.dtype[np.bool_]]
    expression: str


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    rules: tuple[RiskRule, ...]
    train_metrics: dict[str, RuleMetrics]
    single_candidates_before_cap: int
    single_rules_selected: int
    pair_candidates_before_diversity: int
    pair_rules_selected: int


def _condition_key(condition: Condition) -> tuple[str, str, str]:
    value = json.dumps(condition.value, allow_nan=False, separators=(",", ":"))
    return condition.feature, condition.operator, value


def _canonical_expression(conditions: tuple[Condition, ...]) -> str:
    payload = [
        {
            "feature": condition.feature,
            "operator": condition.operator,
            "value": condition.value,
        }
        for condition in sorted(conditions, key=_condition_key)
    ]
    return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _make_rule(conditions: tuple[Condition, ...], origin: str) -> tuple[RiskRule, str]:
    ordered_conditions = tuple(sorted(conditions, key=_condition_key))
    expression = _canonical_expression(ordered_conditions)
    rule_id = hashlib.sha256(expression.encode("utf-8")).hexdigest()[:12]
    return (
        RiskRule(rule_id=rule_id, conditions=ordered_conditions, origin=origin),
        expression,
    )


def _support_bounds(sample_count: int, min_support: float) -> tuple[int, int]:
    support = Decimal(str(min_support))
    minimum = int((support * sample_count).to_integral_value(rounding=ROUND_CEILING))
    maximum = int(
        ((Decimal(1) - support) * sample_count).to_integral_value(rounding=ROUND_FLOOR)
    )
    return minimum, maximum


def _tree_thresholds(
    values: np.ndarray[Any, np.dtype[np.float64]],
    target: np.ndarray[Any, np.dtype[Any]],
    min_samples_leaf: int,
    seed: int,
) -> list[float]:
    if values.size < 2 or np.unique(target).size < 2:
        return []
    classifier = DecisionTreeClassifier(
        max_depth=2,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
    )
    classifier.fit(values.reshape(-1, 1), target)
    return [
        float(threshold)
        for feature, threshold in zip(
            classifier.tree_.feature,
            classifier.tree_.threshold,
            strict=True,
        )
        if feature >= 0 and math.isfinite(float(threshold))
    ]


def _collect_lightgbm_thresholds(node: dict[str, Any], thresholds: list[float]) -> None:
    if "split_index" not in node:
        return
    if node.get("split_feature") == 0:
        threshold = float(node["threshold"])
        if math.isfinite(threshold):
            thresholds.append(threshold)
    _collect_lightgbm_thresholds(node["left_child"], thresholds)
    _collect_lightgbm_thresholds(node["right_child"], thresholds)


def _lightgbm_thresholds(
    values: np.ndarray[Any, np.dtype[np.float64]],
    target: np.ndarray[Any, np.dtype[Any]],
    seed: int,
) -> list[float]:
    if values.size < 2 or np.unique(target).size < 2:
        return []
    classifier = LGBMClassifier(
        n_estimators=30,
        max_depth=2,
        num_leaves=4,
        learning_rate=0.05,
        deterministic=True,
        force_col_wise=True,
        random_state=seed,
        verbosity=-1,
        n_jobs=1,
    )
    classifier.fit(values.reshape(-1, 1), target)
    thresholds: list[float] = []
    for tree in classifier.booster_.dump_model()["tree_info"]:
        _collect_lightgbm_thresholds(tree["tree_structure"], thresholds)
    return thresholds


def _feature_thresholds(
    train: pl.DataFrame,
    feature_name: str,
    target: np.ndarray[Any, np.dtype[Any]],
    config: DiscoveryConfig,
) -> list[float]:
    if not train.schema[feature_name].is_numeric():
        return []
    raw_values = train.get_column(feature_name).to_numpy()
    try:
        numeric_values = raw_values.astype(np.float64, copy=False)
    except (TypeError, ValueError):
        return []
    if np.isinf(numeric_values).any():
        return []
    finite_mask = np.isfinite(numeric_values)
    values = numeric_values[finite_mask]
    if values.size < 2 or np.unique(values).size < 2:
        return []

    finite_target = target[finite_mask]
    thresholds = [float(value) for value in np.quantile(values, _QUANTILES)]
    min_samples_leaf, _ = _support_bounds(train.height, config.min_support)
    thresholds.extend(
        _tree_thresholds(values, finite_target, min_samples_leaf, config.random_seed)
    )
    thresholds.extend(_lightgbm_thresholds(values, finite_target, config.random_seed))
    return sorted({threshold for threshold in thresholds if math.isfinite(threshold)})


def _candidate(
    train: pl.DataFrame,
    target: np.ndarray[Any, np.dtype[Any]],
    conditions: tuple[Condition, ...],
    origin: str,
    config: DiscoveryConfig,
    *,
    maximum_coverage: bool,
) -> _Candidate | None:
    rule, expression = _make_rule(conditions, origin)
    mask = evaluate_rule(train, rule).to_numpy()
    support_count = int(np.count_nonzero(mask))
    minimum_count, maximum_count = _support_bounds(train.height, config.min_support)
    if support_count < minimum_count:
        return None
    if maximum_coverage and support_count > maximum_count:
        return None
    metrics = compute_rule_metrics(mask, target, positive_value=1)
    return _Candidate(rule=rule, metrics=metrics, mask=mask, expression=expression)


def _ranking_key(candidate: _Candidate) -> tuple[float, int, str]:
    return candidate.metrics.lift, candidate.metrics.support_count, candidate.expression


def _single_candidates(
    train: pl.DataFrame,
    feature_names: list[str],
    target: np.ndarray[Any, np.dtype[Any]],
    config: DiscoveryConfig,
) -> list[_Candidate]:
    candidates: dict[str, _Candidate] = {}
    for feature_name in sorted(set(feature_names)):
        for threshold in _feature_thresholds(train, feature_name, target, config):
            for operator in ("<=", ">"):
                condition = Condition(
                    feature=feature_name,
                    operator=operator,
                    value=threshold,
                )
                candidate = _candidate(
                    train,
                    target,
                    (condition,),
                    "discovery_single",
                    config,
                    maximum_coverage=True,
                )
                if candidate is not None:
                    candidates[candidate.expression] = candidate
    return sorted(candidates.values(), key=_ranking_key, reverse=True)


def _pair_beam(singles: list[_Candidate], width: int) -> list[_Candidate]:
    grouped: dict[str, list[_Candidate]] = {}
    for candidate in singles:
        grouped.setdefault(candidate.rule.conditions[0].feature, []).append(candidate)
    selected: list[_Candidate] = []
    position = 0
    while len(selected) < width:
        added = False
        for feature in sorted(grouped):
            candidates = grouped[feature]
            if position >= len(candidates):
                continue
            selected.append(candidates[position])
            added = True
            if len(selected) == width:
                return selected
        if not added:
            return selected
        position += 1
    return selected


def _pair_candidates(
    train: pl.DataFrame,
    target: np.ndarray[Any, np.dtype[Any]],
    singles: list[_Candidate],
    config: DiscoveryConfig,
) -> list[_Candidate]:
    beam = _pair_beam(singles, config.beam_width)
    candidates: dict[str, _Candidate] = {}
    for left_index, left in enumerate(beam):
        for right in beam[left_index + 1 :]:
            if left.rule.conditions[0].feature == right.rule.conditions[0].feature:
                continue
            candidate = _candidate(
                train,
                target,
                left.rule.conditions + right.rule.conditions,
                "discovery_pair",
                config,
                maximum_coverage=False,
            )
            if candidate is None:
                continue
            if np.array_equal(candidate.mask, left.mask) or np.array_equal(
                candidate.mask, right.mask
            ):
                continue
            candidates[candidate.expression] = candidate
    return sorted(candidates.values(), key=_ranking_key, reverse=True)


def _diverse_pair_selection(
    pairs: list[_Candidate],
    maximum: int,
) -> list[_Candidate]:
    if maximum == 0:
        return []
    selected: list[_Candidate] = []
    selected_expressions: set[str] = set()
    represented_feature_pairs: set[tuple[str, str]] = set()
    for candidate in pairs:
        feature_pair = tuple(sorted(condition.feature for condition in candidate.rule.conditions))
        if feature_pair not in represented_feature_pairs:
            selected.append(candidate)
            selected_expressions.add(candidate.expression)
            represented_feature_pairs.add(feature_pair)
            if len(selected) == maximum:
                return sorted(selected, key=_ranking_key, reverse=True)
    for candidate in pairs:
        if candidate.expression not in selected_expressions:
            selected.append(candidate)
            if len(selected) == maximum:
                break
    return sorted(selected, key=_ranking_key, reverse=True)


def discover_with_metrics(
    train: pl.DataFrame,
    feature_names: list[str],
    target_col: str,
    config: DiscoveryConfig,
) -> DiscoveryResult:
    missing_features = sorted(set(feature_names) - set(train.columns))
    if missing_features:
        missing = ", ".join(missing_features)
        raise ValueError(f"missing feature columns: {missing}")
    if target_col not in train.columns:
        raise ValueError(f"missing target column: {target_col}")
    if train.is_empty():
        return DiscoveryResult((), {}, 0, 0, 0, 0)

    target = train.get_column(target_col).to_numpy()
    if np.unique(target).size < 2 or not np.any(target == 1):
        return DiscoveryResult((), {}, 0, 0, 0, 0)

    singles = _single_candidates(train, feature_names, target, config)
    selected_singles = singles[: config.max_single_rules]
    pairs = _pair_candidates(train, target, singles, config)
    selected_pairs = _diverse_pair_selection(pairs, config.max_pair_rules)
    selected = selected_singles + selected_pairs
    rules = tuple(candidate.rule for candidate in selected)
    train_metrics = {
        candidate.rule.rule_id: candidate.metrics for candidate in selected
    }
    return DiscoveryResult(
        rules=rules,
        train_metrics=train_metrics,
        single_candidates_before_cap=len(singles),
        single_rules_selected=len(selected_singles),
        pair_candidates_before_diversity=len(pairs),
        pair_rules_selected=len(selected_pairs),
    )


def discover_rules(
    train: pl.DataFrame,
    feature_names: list[str],
    target_col: str,
    config: DiscoveryConfig,
) -> list[RiskRule]:
    return list(discover_with_metrics(train, feature_names, target_col, config).rules)
