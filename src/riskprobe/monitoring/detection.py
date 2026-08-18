"""Deterministic aggregate drift detection for monitoring snapshots."""

import hashlib
import math
from collections.abc import Iterable
from typing import Literal

import polars as pl

from riskprobe.features.catalog import FeatureCatalog
from riskprobe.models import EvidenceCard

from .models import Alert, FeatureReference, ReferenceSnapshot

_MISSINGNESS_WARNING = 0.10
_MISSINGNESS_CRITICAL = 0.25
_PSI_WARNING = 0.20
_PSI_CRITICAL = 0.30
_POSITIVE_RATE_WARNING = 0.03
_POSITIVE_RATE_CRITICAL = 0.08
_RULE_LIFT_DECAY_WARNING = 0.20
_RULE_LIFT_DECAY_CRITICAL = 0.40
_INSTITUTION_SHARE_WARNING = 0.10
_PROBABILITY_FLOOR = 1e-6


def detect_anomalies(
    reference: ReferenceSnapshot,
    current_frame: pl.DataFrame,
    current_rule_cards: Iterable[EvidenceCard],
    catalog: FeatureCatalog,
) -> list[Alert]:
    """Compare current aggregate data with a reference snapshot.

    Alerts contain only aggregate measurements and stable deidentified scope
    codes. The result order and each alert identifier are deterministic.
    """
    alerts: list[Alert] = []
    anomalous_missingness: dict[str, list[Alert]] = {}

    for role, column in (("target", reference.target_column), ("segment", reference.segment_column)):
        if column not in current_frame.columns:
            alerts.append(
                _alert(
                    "schema",
                    "critical",
                    "dataset",
                    column,
                    "role_column",
                    role,
                    None,
                    None,
                    {},
                )
            )

    for feature in reference.features:
        if feature.feature not in current_frame.columns:
            alerts.append(
                _alert(
                    "schema",
                    "critical",
                    "feature",
                    feature.feature,
                    "column",
                    feature.dtype,
                    None,
                    None,
                    {},
                )
            )
            continue

        series = current_frame.get_column(feature.feature)
        if _dtype_family(series.dtype) != _dtype_family(feature.dtype):
            alerts.append(
                _alert(
                    "schema",
                    "critical",
                    "feature",
                    feature.feature,
                    "dtype_family",
                    _dtype_family(feature.dtype),
                    _dtype_family(series.dtype),
                    None,
                    {},
                )
            )
            continue

        current_missing_rate, numeric_values = _missing_rate_and_values(series)
        missingness_delta = current_missing_rate - feature.missing_rate
        missingness_severity = _increasing_severity(
            missingness_delta,
            _MISSINGNESS_WARNING,
            _MISSINGNESS_CRITICAL,
        )
        if missingness_severity is not None:
            alert = _alert(
                "missingness",
                missingness_severity,
                "feature",
                feature.feature,
                "missing_rate",
                feature.missing_rate,
                current_missing_rate,
                missingness_delta,
                {},
            )
            alerts.append(alert)
            anomalous_missingness.setdefault(feature.family, []).append(alert)

        psi = _population_stability_index(feature, numeric_values)
        psi_severity = _absolute_severity(psi, _PSI_WARNING, _PSI_CRITICAL)
        if psi_severity is not None:
            alerts.append(
                _alert(
                    "distribution",
                    psi_severity,
                    "feature",
                    feature.feature,
                    "psi",
                    0.0,
                    psi,
                    psi,
                    {"bucket_count": len(feature.histogram_counts)},
                )
            )

    alerts.extend(_family_missingness_alerts(anomalous_missingness, reference.features))
    alerts.extend(_label_alerts(reference, current_frame))
    alerts.extend(_population_alerts(reference, current_frame, catalog))
    alerts.extend(_rule_decay_alerts(reference, current_rule_cards))
    return sorted(alerts, key=lambda alert: (alert.alert_type, alert.scope, alert.scope_value, alert.metric))


def _family_missingness_alerts(
    anomalous_missingness: dict[str, list[Alert]],
    features: tuple[FeatureReference, ...],
) -> list[Alert]:
    alerts: list[Alert] = []
    feature_counts: dict[str, int] = {}
    for feature in features:
        feature_counts[feature.family] = feature_counts.get(feature.family, 0) + 1

    for family in sorted(anomalous_missingness):
        feature_alerts = anomalous_missingness[family]
        if len(feature_alerts) < 2:
            continue
        severity: Literal["warning", "critical"] = (
            "critical" if any(alert.severity == "critical" for alert in feature_alerts) else "warning"
        )
        alerts.append(
            _alert(
                "missingness",
                severity,
                "family",
                family,
                "missing_rate",
                None,
                None,
                None,
                {
                    "anomalous_feature_count": len(feature_alerts),
                    "family_feature_count": feature_counts[family],
                },
            )
        )
    return alerts


def _label_alerts(reference: ReferenceSnapshot, current_frame: pl.DataFrame) -> list[Alert]:
    if reference.target_column not in current_frame.columns or current_frame.height == 0:
        return []

    current_rate = sum(
        value == 1 for value in current_frame.get_column(reference.target_column).to_list()
    )
    current_rate /= current_frame.height
    delta = current_rate - reference.positive_rate
    severity = _absolute_severity(delta, _POSITIVE_RATE_WARNING, _POSITIVE_RATE_CRITICAL)
    if severity is None:
        return []
    return [
        _alert(
            "label",
            severity,
            "dataset",
            reference.dataset_id,
            "positive_rate",
            reference.positive_rate,
            current_rate,
            delta,
            {},
        )
    ]


def _population_alerts(
    reference: ReferenceSnapshot,
    current_frame: pl.DataFrame,
    catalog: FeatureCatalog,
) -> list[Alert]:
    del catalog
    if (
        reference.segment_column not in current_frame.columns
        or current_frame.height == 0
        or reference.row_count == 0
    ):
        return []

    current_values = current_frame.get_column(reference.segment_column).to_list()
    current_counts: dict[str, int] = {}
    for value in current_values:
        segment = str(value)
        current_counts[segment] = current_counts.get(segment, 0) + 1

    alerts: list[Alert] = []
    for segment in sorted(set(reference.segment_counts).union(current_counts)):
        reference_count = reference.segment_counts.get(segment, 0)
        current_count = current_counts.get(segment, 0)
        # A group is eligible when it is sufficiently represented on either side.
        # This preserves alerts for legal groups that appear or disappear while
        # suppressing groups that are small in both snapshots.
        if reference_count < reference.min_group_size and current_count < reference.min_group_size:
            continue
        if reference_count == 0 and current_count >= reference.min_group_size:
            reference_share = 0.0
            current_share = current_count / current_frame.height
        elif current_count == 0 and reference_count >= reference.min_group_size:
            reference_share = reference_count / reference.row_count
            current_share = 0.0
        else:
            if (
                reference_count < reference.min_group_size
                or current_count < reference.min_group_size
            ):
                continue
            reference_share = reference_count / reference.row_count
            current_share = current_count / current_frame.height
        delta = current_share - reference_share
        if abs(delta) < _INSTITUTION_SHARE_WARNING:
            continue
        alerts.append(
            _alert(
                "population",
                "warning",
                "institution",
                segment,
                "share",
                reference_share,
                current_share,
                delta,
                {},
            )
        )
    return alerts


def _rule_decay_alerts(
    reference: ReferenceSnapshot,
    current_rule_cards: Iterable[EvidenceCard],
) -> list[Alert]:
    current_lifts = {card.rule.rule_id: card.test.lift for card in current_rule_cards}
    alerts: list[Alert] = []
    for rule in reference.rules:
        current_lift = current_lifts.get(rule.rule_id)
        if current_lift is None or rule.lift <= 0:
            continue
        decay = (rule.lift - current_lift) / rule.lift
        severity = _increasing_severity(
            decay,
            _RULE_LIFT_DECAY_WARNING,
            _RULE_LIFT_DECAY_CRITICAL,
        )
        if severity is None:
            continue
        alerts.append(
            _alert(
                "rule_decay",
                severity,
                "rule",
                rule.rule_id,
                "lift_decay",
                rule.lift,
                current_lift,
                decay,
                {},
            )
        )
    return alerts


def _missing_rate_and_values(series: pl.Series) -> tuple[float, tuple[float, ...]]:
    values: list[float] = []
    invalid_count = 0
    for value in series.drop_nulls().to_list():
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            values.append(numeric_value)
        else:
            invalid_count += 1
    missing_rate = (series.null_count() + invalid_count) / len(series) if len(series) else 0.0
    return missing_rate, tuple(values)


def _population_stability_index(
    feature: FeatureReference,
    current_values: tuple[float, ...],
) -> float:
    if not feature.histogram_counts or not current_values:
        return 0.0
    current_counts = _histogram_counts(current_values, feature.quantile_edges)
    reference_total = sum(feature.histogram_counts)
    current_total = sum(current_counts)
    if reference_total == 0 or current_total == 0:
        return 0.0
    return sum(
        (current_probability - reference_probability)
        * math.log(current_probability / reference_probability)
        for reference_count, current_count in zip(feature.histogram_counts, current_counts)
        for reference_probability, current_probability in [
            (
                max(reference_count / reference_total, _PROBABILITY_FLOOR),
                max(current_count / current_total, _PROBABILITY_FLOOR),
            )
        ]
    )


def _histogram_counts(values: tuple[float, ...], edges: tuple[float, ...]) -> tuple[int, ...]:
    if len(edges) < 2:
        return ()
    counts = [0] * (len(edges) - 1)
    for value in values:
        for index, upper_edge in enumerate(edges[1:]):
            if value < upper_edge or index == len(counts) - 1:
                counts[index] += 1
                break
    return tuple(counts)


def _target_column(current_frame: pl.DataFrame, feature_names: set[str]) -> str | None:
    for name in ("target", "label", "outcome"):
        if name in current_frame.columns:
            return name
    for name in current_frame.columns:
        if name in feature_names or not current_frame.get_column(name).dtype.is_numeric():
            continue
        values = current_frame.get_column(name).drop_nulls().unique().to_list()
        if values and all(value in (0, 1) for value in values):
            return name
    return None


def _segment_column(
    reference: ReferenceSnapshot,
    current_frame: pl.DataFrame,
    catalog: FeatureCatalog,
) -> str | None:
    reference_segments = set(reference.segment_counts)
    candidates: list[tuple[int, str]] = []
    catalog_names = tuple(dict.fromkeys(spec.name for spec in catalog.features))
    for name in catalog_names:
        if name not in current_frame.columns:
            continue
        matching_values = sum(
            str(value) in reference_segments for value in current_frame.get_column(name).drop_nulls().to_list()
        )
        if matching_values:
            candidates.append((matching_values, name))
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: (-candidate[0], candidate[1]))[1]


def _dtype_family(dtype: pl.DataType | str) -> str:
    if not isinstance(dtype, str):
        if dtype.is_numeric():
            return "numeric"
        dtype = str(dtype)
    normalized = dtype.lower()
    if normalized.startswith(("int", "uint", "float", "decimal")):
        return "numeric"
    if normalized in {"bool", "boolean"}:
        return "boolean"
    if normalized.startswith(("str", "utf8", "categorical", "enum")):
        return "string"
    if normalized.startswith(("date", "datetime", "time", "duration")):
        return "temporal"
    return normalized


def _absolute_severity(
    value: float,
    warning: float,
    critical: float,
) -> Literal["warning", "critical"] | None:
    magnitude = abs(value)
    if magnitude >= critical:
        return "critical"
    if magnitude >= warning:
        return "warning"
    return None


def _increasing_severity(
    value: float,
    warning: float,
    critical: float,
) -> Literal["warning", "critical"] | None:
    if value >= critical:
        return "critical"
    if value >= warning:
        return "warning"
    return None


def _alert(
    alert_type: Literal[
        "schema", "missingness", "distribution", "population", "label", "rule_decay"
    ],
    severity: Literal["warning", "critical"],
    scope: Literal["dataset", "institution", "feature", "family", "rule"],
    scope_value: str,
    metric: str,
    reference_value: float | str | None,
    current_value: float | str | None,
    delta: float | None,
    evidence: dict[str, float | int | str],
) -> Alert:
    alert_id = hashlib.sha256(
        f"{alert_type}|{scope}|{scope_value}|{metric}".encode()
    ).hexdigest()[:12]
    return Alert(
        alert_id=alert_id,
        alert_type=alert_type,
        severity=severity,
        scope=scope,
        scope_value=scope_value,
        metric=metric,
        reference_value=reference_value,
        current_value=current_value,
        delta=delta,
        evidence=evidence,
    )
