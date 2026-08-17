import math

import polars as pl
import pytest

from riskprobe.monitoring.drift import (
    diagnose_drift,
    population_stability_index,
    quantile_bin_edges,
)
from riskprobe.monitoring.models import FindingSeverity


def _find(findings: tuple[object, ...], code: str):
    return next(finding for finding in findings if finding.code == code)


def test_quantile_bins_are_fit_from_reference_only_and_cover_all_values() -> None:
    reference = [0.0, 1.0, 2.0, 3.0]

    edges = quantile_bin_edges(reference, bin_count=4)

    assert edges[0] == -math.inf
    assert edges[-1] == math.inf
    assert edges == quantile_bin_edges(reference, bin_count=4)
    assert edges != quantile_bin_edges([0.0, 1.0, 2.0, 3000.0], bin_count=4)


def test_psi_uses_epsilon_smoothing_for_empty_bins() -> None:
    epsilon = 1e-6
    reference = [0.0, 0.0, 1.0, 1.0]
    current = [1.0, 1.0, 1.0, 1.0]
    edges = (-math.inf, 0.5, math.inf)
    ref_low = (2 + epsilon) / (4 + 2 * epsilon)
    ref_high = ref_low
    cur_low = epsilon / (4 + 2 * epsilon)
    cur_high = (4 + epsilon) / (4 + 2 * epsilon)
    expected = (cur_low - ref_low) * math.log(cur_low / ref_low)
    expected += (cur_high - ref_high) * math.log(cur_high / ref_high)

    psi = population_stability_index(reference, current, edges=edges, epsilon=epsilon)

    assert psi == pytest.approx(expected)
    assert math.isfinite(psi)


def test_drift_severity_uses_warning_and_critical_thresholds() -> None:
    reference = pl.DataFrame({"score": [0.0] * 50 + [1.0] * 50})
    warning_current = pl.DataFrame({"score": [0.0] * 67 + [1.0] * 33})
    critical_current = pl.DataFrame({"score": [0.0] * 80 + [1.0] * 20})

    warning = _find(
        diagnose_drift(reference, warning_current, feature_columns=("score",)),
        "feature_psi",
    )
    critical = _find(
        diagnose_drift(reference, critical_current, feature_columns=("score",)),
        "feature_psi",
    )

    assert warning.severity is FindingSeverity.WARNING
    assert 0.10 <= warning.metrics["psi"] < 0.25
    assert critical.severity is FindingSeverity.CRITICAL
    assert critical.metrics["psi"] >= 0.25


def test_drift_detects_missing_rate_and_target_rate_shifts() -> None:
    reference = pl.DataFrame(
        {
            "score": [1.0] * 10,
            "target": [0] * 9 + [1],
        }
    )
    current = pl.DataFrame(
        {
            "score": [None] * 3 + [1.0] * 7,
            "target": [0] * 7 + [1] * 3,
        }
    )

    findings = diagnose_drift(
        reference,
        current,
        feature_columns=("score",),
        target_column="target",
    )

    missing = _find(findings, "missing_rate_shift")
    target = _find(findings, "target_rate_shift")
    assert missing.severity is FindingSeverity.CRITICAL
    assert missing.metrics["rate_shift"] == pytest.approx(0.3)
    assert target.severity is FindingSeverity.WARNING
    assert target.metrics["rate_shift"] == pytest.approx(0.2)


def test_all_null_and_empty_inputs_have_finite_zero_psi() -> None:
    edges = quantile_bin_edges([None, math.nan, math.inf], bin_count=10)

    assert edges == (-math.inf, math.inf)
    assert population_stability_index([], [], edges=edges) == 0.0
    assert population_stability_index([None], [None], edges=edges) == 0.0
