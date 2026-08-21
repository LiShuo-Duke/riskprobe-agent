"""Train-only WOE/IV binning with optional monotonic bad-rate constraints."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from riskprobe.models import RiskRule
from riskprobe.rules.expression import evaluate_rule

MonotonicDirection = Literal["none", "increasing", "decreasing", "auto"]
ResolvedDirection = Literal["none", "increasing", "decreasing"]


@dataclass(frozen=True, slots=True)
class WOEBinningModel:
    """Frozen transform learned from one training frame.

    ``monotonic`` describes the training bad-rate direction, not the WOE direction.
    Test and production frames can only be transformed with the stored edges.
    """

    feature: str
    edges: tuple[float, ...]
    woe_values: tuple[float, ...]
    bad_rates: tuple[float, ...]
    bin_counts: tuple[int, ...]
    missing_woe: float
    missing_bad_rate: float | None
    iv: float
    monotonic: ResolvedDirection

    def transform(self, values: pl.Series | Sequence[object]) -> np.ndarray:
        """Transform values without refitting edges or WOE."""

        numeric = _numeric_values(values)
        result = np.full(numeric.shape, self.missing_woe, dtype=np.float64)
        finite = np.isfinite(numeric)
        if finite.any():
            indices = np.searchsorted(
                np.asarray(self.edges, dtype=np.float64),
                numeric[finite],
                side="left",
            )
            result[finite] = np.asarray(self.woe_values, dtype=np.float64)[indices]
        return result


def fit_woe_binning(
    frame: pl.DataFrame,
    *,
    feature: str,
    target_col: str,
    max_bins: int = 10,
    min_bin_fraction: float = 0.05,
    smoothing: float = 0.5,
    monotonic: MonotonicDirection = "none",
) -> WOEBinningModel:
    """Fit one numeric feature's WOE/IV representation on training data only."""

    if not isinstance(frame, pl.DataFrame) or frame.is_empty():
        raise ValueError("training frame must be a non-empty DataFrame")
    if feature not in frame.columns or target_col not in frame.columns:
        raise ValueError("feature and target columns are required")
    if not frame.schema[feature].is_numeric():
        raise TypeError("WOE binning requires a numeric feature")
    if type(max_bins) is not int or not 2 <= max_bins <= 100:
        raise ValueError("max_bins must be between 2 and 100")
    if type(min_bin_fraction) is not float or not 0 < min_bin_fraction <= 1:
        raise ValueError("min_bin_fraction must be in (0, 1]")
    if type(smoothing) is not float or not 0 < smoothing <= 100:
        raise ValueError("smoothing must be in (0, 100]")
    if monotonic not in {"none", "increasing", "decreasing", "auto"}:
        raise ValueError("invalid monotonic direction")

    values = _numeric_values(frame.get_column(feature))
    targets = _target_values(frame.get_column(target_col))
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("feature has no finite values")
    if np.unique(targets).size < 2:
        raise ValueError("target must contain both classes")

    edges = _initial_edges(values[finite], max_bins)
    counts, bads = _bin_stats(values, targets, edges)
    min_count = max(1, math.ceil(int(finite.sum()) * min_bin_fraction))
    edges, counts, bads = _merge_small_bins(edges, counts, bads, min_count)

    resolved: ResolvedDirection
    if monotonic == "auto":
        resolved = _infer_direction(bads, counts)
    else:
        resolved = monotonic
    if resolved != "none":
        edges, counts, bads = _merge_monotonic_bins(
            edges,
            counts,
            bads,
            resolved,
        )

    missing_count = int((~finite).sum())
    missing_bad = int(targets[~finite].sum())
    woe_values, missing_woe, iv = _woe_values(
        counts,
        bads,
        missing_count,
        missing_bad,
        smoothing,
    )
    bad_rates = tuple(
        bad / count if count else 0.0
        for count, bad in zip(counts, bads, strict=True)
    )
    return WOEBinningModel(
        feature=feature,
        edges=tuple(edges),
        woe_values=tuple(woe_values),
        bad_rates=bad_rates,
        bin_counts=tuple(counts),
        missing_woe=missing_woe,
        missing_bad_rate=(missing_bad / missing_count if missing_count else None),
        iv=iv,
        monotonic=resolved,
    )


def _numeric_values(values: pl.Series | Sequence[object]) -> np.ndarray:
    raw = values.to_list() if isinstance(values, pl.Series) else list(values)
    result = np.full(len(raw), np.nan, dtype=np.float64)
    for index, value in enumerate(raw):
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result[index] = number
    return result


def _target_values(values: pl.Series) -> np.ndarray:
    raw = values.to_list()
    try:
        target = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("target must be binary numeric data") from error
    if not np.isfinite(target).all() or not np.isin(target, (0.0, 1.0)).all():
        raise ValueError("target must contain only 0 and 1")
    return target.astype(np.int8)


def _initial_edges(values: np.ndarray, max_bins: int) -> list[float]:
    unique = np.unique(values)
    if unique.size <= max_bins:
        candidates = unique[1:-1]
    else:
        quantiles = np.linspace(0.0, 1.0, max_bins + 1)[1:-1]
        candidates = np.unique(np.quantile(values, quantiles))
        candidates = candidates[(candidates > unique[0]) & (candidates < unique[-1])]
    return [float(value) for value in candidates]


def _bin_stats(
    values: np.ndarray,
    targets: np.ndarray,
    edges: Sequence[float],
) -> tuple[list[int], list[int]]:
    finite = np.isfinite(values)
    indices = np.searchsorted(np.asarray(edges, dtype=np.float64), values[finite], side="left")
    counts = [0] * (len(edges) + 1)
    bads = [0] * (len(edges) + 1)
    for index, target in zip(indices, targets[finite], strict=True):
        counts[int(index)] += 1
        bads[int(index)] += int(target)
    return counts, bads


def _merge_small_bins(
    edges: Sequence[float],
    counts: list[int],
    bads: list[int],
    minimum: int,
) -> tuple[list[float], list[int], list[int]]:
    edges = list(edges)
    while len(counts) > 1 and any(count < minimum for count in counts):
        index = min(
            (position for position, count in enumerate(counts) if count < minimum),
            key=lambda position: (counts[position], position),
        )
        if index == 0:
            left, right = 0, 1
        elif index == len(counts) - 1:
            left, right = index - 1, index
        else:
            left, right = (
                (index - 1, index)
                if abs(_rate(bads[index - 1], counts[index - 1]) - _rate(bads[index], counts[index]))
                <= abs(_rate(bads[index], counts[index]) - _rate(bads[index + 1], counts[index + 1]))
                else (index, index + 1)
            )
        counts[left] += counts[right]
        bads[left] += bads[right]
        del counts[right]
        del bads[right]
        del edges[right - 1]
    return edges, counts, bads


def _merge_monotonic_bins(
    edges: Sequence[float],
    counts: list[int],
    bads: list[int],
    direction: ResolvedDirection,
) -> tuple[list[float], list[int], list[int]]:
    edges = list(edges)
    while len(counts) > 1:
        violation = next(
            (
                index
                for index, (left, right) in enumerate(zip(bads, bads[1:]))
                if (
                    direction == "increasing"
                    and _rate(left, counts[index]) > _rate(right, counts[index + 1])
                )
                or (
                    direction == "decreasing"
                    and _rate(left, counts[index]) < _rate(right, counts[index + 1])
                )
            ),
            None,
        )
        if violation is None:
            return edges, counts, bads
        counts[violation] += counts[violation + 1]
        bads[violation] += bads[violation + 1]
        del counts[violation + 1]
        del bads[violation + 1]
        del edges[violation]
    return edges, counts, bads


def _infer_direction(bads: Sequence[int], counts: Sequence[int]) -> ResolvedDirection:
    if len(bads) < 2:
        return "none"
    first = _rate(bads[0], counts[0])
    last = _rate(bads[-1], counts[-1])
    if math.isclose(first, last):
        return "none"
    return "increasing" if last > first else "decreasing"


def _rate(bad: int, count: int) -> float:
    return bad / count if count else 0.0


def _woe_values(
    counts: Sequence[int],
    bads: Sequence[int],
    missing_count: int,
    missing_bad: int,
    smoothing: float,
) -> tuple[list[float], float, float]:
    regular_good = [count - bad for count, bad in zip(counts, bads, strict=True)]
    all_counts = list(counts) + ([missing_count] if missing_count else [])
    all_bads = list(bads) + ([missing_bad] if missing_count else [])
    all_good = regular_good + ([missing_count - missing_bad] if missing_count else [])
    bin_count = len(all_counts)
    total_good = sum(all_good)
    total_bad = sum(all_bads)
    woe: list[float] = []
    iv = 0.0
    for good, bad in zip(all_good, all_bads, strict=True):
        good_dist = (good + smoothing) / (total_good + smoothing * bin_count)
        bad_dist = (bad + smoothing) / (total_bad + smoothing * bin_count)
        value = math.log(good_dist / bad_dist)
        woe.append(value)
        iv += (good_dist - bad_dist) * value
    return woe[:-1] if missing_count else woe, woe[-1] if missing_count else 0.0, iv


@dataclass(frozen=True, slots=True)
class ScorecardPrediction:
    """Batch scorecard output with risk probability and explainability fields."""

    probabilities: tuple[float, ...]
    risk_scores: tuple[float, ...]
    risk_levels: tuple[str, ...]
    reason_codes: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class ScorecardModel:
    """Frozen WOE plus rule-hit logistic scorecard."""

    feature_names: tuple[str, ...]
    binning_models: tuple[WOEBinningModel, ...]
    rules: tuple[RiskRule, ...]
    coefficients: tuple[float, ...]
    intercept: float

    @property
    def rule_names(self) -> tuple[str, ...]:
        return tuple(rule.rule_id for rule in self.rules)

    @property
    def feature_coefficients(self) -> tuple[float, ...]:
        return self.coefficients[: len(self.feature_names)]

    @property
    def rule_coefficients(self) -> tuple[float, ...]:
        return self.coefficients[len(self.feature_names) :]

    def transform(self, frame: pl.DataFrame) -> np.ndarray:
        """Transform a frame using frozen WOE bins and rule-hit columns."""

        self._validate_frame(frame)
        return _build_scorecard_matrix(frame, self.binning_models, self.rules)

    def predict_proba(self, frame: pl.DataFrame) -> np.ndarray:
        """Return sklearn-style ``[non-bad, bad]`` probabilities."""

        probability = self._bad_probability(frame)
        return np.column_stack((1.0 - probability, probability))

    def predict(
        self,
        frame: pl.DataFrame,
        *,
        top_reason_codes: int = 3,
    ) -> ScorecardPrediction:
        """Score rows and return bounded risk explanations."""

        if type(top_reason_codes) is not int or top_reason_codes < 0:
            raise ValueError("top_reason_codes must be a non-negative integer")
        self._validate_frame(frame)
        matrix = _build_scorecard_matrix(frame, self.binning_models, self.rules)
        probability = _sigmoid(self.intercept + matrix @ np.asarray(self.coefficients))
        labels = self.feature_names + tuple(
            f"rule:{rule.rule_id}" for rule in self.rules
        )
        contributions = matrix * np.asarray(self.coefficients)
        reason_codes: list[tuple[str, ...]] = []
        for row in contributions:
            order = np.argsort(-np.abs(row), kind="stable")
            reasons = tuple(
                labels[index]
                for index in order[:top_reason_codes]
                if not math.isclose(float(row[index]), 0.0)
            )
            if not reasons and top_reason_codes and labels:
                reasons = (labels[int(order[0])],)
            reason_codes.append(reasons)
        return ScorecardPrediction(
            probabilities=tuple(float(value) for value in probability),
            risk_scores=tuple(float(value * 1000.0) for value in probability),
            risk_levels=tuple(_risk_level(float(value)) for value in probability),
            reason_codes=tuple(reason_codes),
        )

    def score(
        self,
        frame: pl.DataFrame,
        *,
        top_reason_codes: int = 3,
    ) -> ScorecardPrediction:
        """Alias for :meth:`predict` for scorecard-oriented call sites."""

        return self.predict(frame, top_reason_codes=top_reason_codes)

    def _bad_probability(self, frame: pl.DataFrame) -> np.ndarray:
        self._validate_frame(frame)
        matrix = _build_scorecard_matrix(frame, self.binning_models, self.rules)
        return _sigmoid(self.intercept + matrix @ np.asarray(self.coefficients))

    def _validate_frame(self, frame: pl.DataFrame) -> None:
        if not isinstance(frame, pl.DataFrame):
            raise TypeError("frame must be a polars DataFrame")
        missing = [name for name in self._required_columns if name not in frame.columns]
        if missing:
            raise ValueError(f"feature columns missing: {', '.join(missing)}")

    @property
    def _required_columns(self) -> tuple[str, ...]:
        names = list(self.feature_names)
        for rule in self.rules:
            names.extend(condition.feature for condition in rule.conditions)
        return tuple(dict.fromkeys(names))


def fit_scorecard(
    frame: pl.DataFrame,
    *,
    feature_names: Sequence[str],
    target_col: str,
    rules: Sequence[RiskRule] = (),
    max_bins: int = 10,
    min_bin_fraction: float = 0.05,
    smoothing: float = 0.5,
    monotonic: MonotonicDirection = "none",
    min_iv: float = 0.0,
    C: float = 1.0,
    max_iter: int = 1000,
) -> ScorecardModel:
    """Fit a train-only WOE scorecard with optional deterministic rule features."""

    if not isinstance(frame, pl.DataFrame) or frame.is_empty():
        raise ValueError("training frame must be a non-empty DataFrame")
    if not isinstance(feature_names, Sequence) or isinstance(feature_names, (str, bytes)):
        raise TypeError("feature_names must be a sequence of column names")
    names = tuple(feature_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("feature_names must contain at least one non-empty string")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    if target_col not in frame.columns:
        raise ValueError("target column is required")
    if target_col in names:
        raise ValueError("target column cannot also be a feature column")
    if type(min_iv) not in (int, float) or not math.isfinite(float(min_iv)) or min_iv < 0:
        raise ValueError("min_iv must be a non-negative finite number")
    if type(C) not in (int, float) or not math.isfinite(float(C)) or C <= 0:
        raise ValueError("C must be a positive finite number")
    if type(max_iter) is not int or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    fitted_rules = tuple(rules)
    if any(not isinstance(rule, RiskRule) for rule in fitted_rules):
        raise TypeError("rules must contain RiskRule instances")
    required = list(names)
    required.extend(
        condition.feature
        for rule in fitted_rules
        for condition in rule.conditions
    )
    missing = [name for name in dict.fromkeys(required) if name not in frame.columns]
    if missing:
        raise ValueError(f"feature columns missing: {', '.join(missing)}")

    target = _target_values(frame.get_column(target_col))
    binning_models = tuple(
        model
        for model in (
            fit_woe_binning(
                frame,
                feature=name,
                target_col=target_col,
                max_bins=max_bins,
                min_bin_fraction=min_bin_fraction,
                smoothing=smoothing,
                monotonic=monotonic,
            )
            for name in names
        )
        if model.iv >= float(min_iv)
    )
    selected_names = tuple(model.feature for model in binning_models)
    if not binning_models and not fitted_rules:
        raise ValueError("no scorecard features remain after IV filtering")

    design = _build_scorecard_matrix(frame, binning_models, fitted_rules)
    estimator = LogisticRegression(
        C=float(C),
        max_iter=max_iter,
        solver="lbfgs",
    )
    estimator.fit(design, target)
    coefficients = tuple(float(value) for value in estimator.coef_[0])
    return ScorecardModel(
        feature_names=selected_names,
        binning_models=binning_models,
        rules=fitted_rules,
        coefficients=coefficients,
        intercept=float(estimator.intercept_[0]),
    )


def _build_scorecard_matrix(
    frame: pl.DataFrame,
    binning_models: Sequence[WOEBinningModel],
    rules: Sequence[RiskRule],
) -> np.ndarray:
    columns = [model.transform(frame.get_column(model.feature)) for model in binning_models]
    columns.extend(
        np.asarray(evaluate_rule(frame, rule).to_numpy(), dtype=np.float64)
        for rule in rules
    )
    if not columns:
        return np.empty((frame.height, 0), dtype=np.float64)
    return np.column_stack(columns).astype(np.float64, copy=False)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _risk_level(probability: float) -> str:
    if probability < 0.25:
        return "low"
    if probability < 0.5:
        return "medium"
    if probability < 0.75:
        return "high"
    return "critical"


__all__ = [
    "MonotonicDirection",
    "ScorecardModel",
    "ScorecardPrediction",
    "WOEBinningModel",
    "fit_scorecard",
    "fit_woe_binning",
]
