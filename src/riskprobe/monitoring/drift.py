"""Reference-derived quantile PSI and aggregate shift diagnostics."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import polars as pl

from riskprobe.monitoring.models import (
    FindingKind,
    FindingSeverity,
    RiskFinding,
    finding_sort_key,
)

WARNING_THRESHOLD = 0.10
CRITICAL_THRESHOLD = 0.25
DEFAULT_EPSILON = 1e-6


def quantile_bin_edges(
    reference: Sequence[object] | pl.Series,
    *,
    bin_count: int = 10,
) -> tuple[float, ...]:
    """Fit deterministic quantile boundaries from finite reference values only."""

    if bin_count < 1:
        raise ValueError("bin_count must be positive")
    finite = _finite_values(reference)
    if finite.size == 0 or float(np.min(finite)) == float(np.max(finite)):
        return (-math.inf, math.inf)
    quantiles = np.linspace(0.0, 1.0, bin_count + 1)[1:-1]
    candidates = np.quantile(finite, quantiles)
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    internal = sorted(
        {
            float(value)
            for value in np.atleast_1d(candidates)
            if math.isfinite(float(value)) and minimum < float(value) < maximum
        }
    )
    return (-math.inf, *internal, math.inf)


def population_stability_index(
    reference: Sequence[object] | pl.Series,
    current: Sequence[object] | pl.Series,
    *,
    edges: Sequence[float] | None = None,
    bin_count: int = 10,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """Calculate PSI with additive epsilon smoothing for every bin."""

    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a positive finite number")
    reference_values = _finite_values(reference)
    current_values = _finite_values(current)
    if reference_values.size == 0 or current_values.size == 0:
        return 0.0
    fitted_edges = tuple(edges) if edges is not None else quantile_bin_edges(
        reference_values,
        bin_count=bin_count,
    )
    _validate_edges(fitted_edges)
    reference_counts, _ = np.histogram(reference_values, bins=fitted_edges)
    current_counts, _ = np.histogram(current_values, bins=fitted_edges)
    bin_total = len(fitted_edges) - 1
    reference_distribution = (reference_counts.astype(float) + epsilon) / (
        reference_counts.sum() + epsilon * bin_total
    )
    current_distribution = (current_counts.astype(float) + epsilon) / (
        current_counts.sum() + epsilon * bin_total
    )
    components = (current_distribution - reference_distribution) * np.log(
        current_distribution / reference_distribution
    )
    result = float(np.sum(components))
    return result if math.isfinite(result) else 0.0


def diagnose_drift(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str | None = None,
    positive_value: int = 1,
    bin_count: int = 10,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[RiskFinding, ...]:
    """Return thresholded PSI, missing-rate, and target-rate shift findings."""

    _require_columns(reference, current, feature_columns, target_column)
    findings: list[RiskFinding] = []
    for feature in sorted(set(feature_columns)):
        reference_series = reference.get_column(feature)
        current_series = current.get_column(feature)
        missing_shift = abs(
            _missing_rate(current_series) - _missing_rate(reference_series)
        )
        if missing_shift >= WARNING_THRESHOLD:
            findings.append(
                RiskFinding(
                    kind=FindingKind.POPULATION_SHIFT,
                    severity=severity_for_shift(missing_shift),
                    code="missing_rate_shift",
                    feature=feature,
                    metrics={
                        "reference_rate": _missing_rate(reference_series),
                        "current_rate": _missing_rate(current_series),
                        "rate_shift": float(missing_shift),
                    },
                )
            )

        if not reference_series.dtype.is_numeric() or not current_series.dtype.is_numeric():
            continue
        edges = quantile_bin_edges(reference_series, bin_count=bin_count)
        psi = population_stability_index(
            reference_series,
            current_series,
            edges=edges,
            epsilon=epsilon,
        )
        if psi >= WARNING_THRESHOLD:
            findings.append(
                RiskFinding(
                    kind=FindingKind.FEATURE_DRIFT,
                    severity=severity_for_shift(psi),
                    code="feature_psi",
                    feature=feature,
                    metrics={
                        "psi": psi,
                        "reference_count": int(_finite_values(reference_series).size),
                        "current_count": int(_finite_values(current_series).size),
                    },
                )
            )

    if target_column is not None:
        reference_rate = _target_rate(reference.get_column(target_column), positive_value)
        current_rate = _target_rate(current.get_column(target_column), positive_value)
        if reference_rate is not None and current_rate is not None:
            target_shift = abs(current_rate - reference_rate)
            if target_shift >= WARNING_THRESHOLD:
                findings.append(
                    RiskFinding(
                        kind=FindingKind.TARGET_SHIFT,
                        severity=severity_for_shift(target_shift),
                        code="target_rate_shift",
                        metrics={
                            "reference_rate": reference_rate,
                            "current_rate": current_rate,
                            "rate_shift": float(target_shift),
                        },
                    )
                )

    return tuple(sorted(findings, key=finding_sort_key))


def severity_for_shift(value: float) -> FindingSeverity:
    if value >= CRITICAL_THRESHOLD:
        return FindingSeverity.CRITICAL
    if value >= WARNING_THRESHOLD:
        return FindingSeverity.WARNING
    return FindingSeverity.INFO


def _finite_values(values: Sequence[object] | pl.Series) -> np.ndarray:
    source = values.to_list() if isinstance(values, pl.Series) else values
    finite: list[float] = []
    for value in source:
        if value is None or isinstance(value, bool):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            finite.append(numeric)
    return np.asarray(finite, dtype=float)


def _missing_rate(series: pl.Series) -> float:
    if len(series) == 0:
        return 0.0
    missing = series.null_count()
    if series.dtype.is_numeric():
        missing += _non_finite_count(series)
    return float(missing / len(series))


def _non_finite_count(series: pl.Series) -> int:
    return sum(
        1
        for value in series.drop_nulls().to_list()
        if not math.isfinite(float(value))
    )


def _target_rate(series: pl.Series, positive_value: int) -> float | None:
    values = tuple(value for value in series.to_list() if value is not None)
    if not values:
        return None
    return float(sum(value == positive_value for value in values) / len(values))


def _validate_edges(edges: Sequence[float]) -> None:
    if len(edges) < 2 or any(
        left >= right for left, right in zip(edges, edges[1:], strict=False)
    ):
        raise ValueError("PSI edges must be strictly increasing")


def _require_columns(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    feature_columns: Sequence[str],
    target_column: str | None,
) -> None:
    required = (*feature_columns, *((target_column,) if target_column else ()))
    if any(column not in reference.columns or column not in current.columns for column in required):
        raise ValueError("drift diagnostics require configured columns")


calculate_psi = population_stability_index
drift_findings = diagnose_drift

__all__ = [
    "CRITICAL_THRESHOLD",
    "DEFAULT_EPSILON",
    "WARNING_THRESHOLD",
    "calculate_psi",
    "diagnose_drift",
    "drift_findings",
    "population_stability_index",
    "quantile_bin_edges",
    "severity_for_shift",
]
