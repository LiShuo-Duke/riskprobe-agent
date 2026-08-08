"""Deterministic, aggregate-only attribution for monitoring alerts."""

import math
from collections.abc import Iterable

import polars as pl

from riskprobe.features.catalog import FeatureCatalog

from .models import Alert, Diagnosis, FeatureReference, ReferenceSnapshot, RootCause

_CREATED_AT = "1970-01-01T00:00:00Z"


def diagnose_alerts(
    alerts: Iterable[Alert],
    reference: ReferenceSnapshot,
    current_frame: pl.DataFrame,
    catalog: FeatureCatalog,
    top_k: int,
) -> list[Diagnosis]:
    """Rank deterministic aggregate contributions for every alert type."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    references = {feature.feature: feature for feature in reference.features}
    diagnoses: list[Diagnosis] = []
    for alert in alerts:
        causes = _causes_for_alert(alert, reference, current_frame, references, catalog)
        diagnoses.append(
            Diagnosis(
                snapshot_id=reference.snapshot_id,
                alerts=(alert,),
                root_causes=tuple(causes[:top_k]),
                created_at=_CREATED_AT,
            )
        )
    return diagnoses


def _causes_for_alert(
    alert: Alert,
    reference: ReferenceSnapshot,
    current_frame: pl.DataFrame,
    references: dict[str, FeatureReference],
    catalog: FeatureCatalog,
) -> list[RootCause]:
    if alert.alert_type == "schema":
        return _rank(
            [("schema", alert.scope_value, 1.0, {"metric": alert.metric})]
        )
    if alert.alert_type == "label":
        causes = [
            (
                "target", reference.target_column, abs(alert.delta or 0.0),
                {"reference_rate": alert.reference_value, "current_rate": alert.current_value},
            )
        ]
        if reference.segment_column in current_frame.columns:
            for segment, subset in _eligible_segments(current_frame, reference):
                values = subset.get_column(reference.target_column).to_list()
                rate = sum(value == 1 for value in values) / len(values) if values else 0.0
                causes.append(
                    (
                        "segment", segment,
                        abs(rate - reference.positive_rate) * (subset.height / current_frame.height),
                        {"reference_rate": reference.positive_rate, "current_rate": rate},
                    )
                )
        return _rank(causes)
    if alert.alert_type == "population":
        return _rank(
            [
                (
                    "segment",
                    alert.scope_value,
                    abs(alert.delta or 0.0),
                    {"reference_share": alert.reference_value, "current_share": alert.current_value},
                )
            ]
        )
    if alert.alert_type == "rule_decay":
        return _rank(
            [
                (
                    "rule",
                    alert.scope_value,
                    abs(alert.delta or 0.0),
                    {"reference_lift": alert.reference_value, "current_lift": alert.current_value},
                )
            ]
        )
    if alert.alert_type == "missingness" and alert.scope == "family":
        features = [feature for feature in references.values() if feature.family == alert.scope_value]
        feature_causes: list[tuple[str, str, float, dict[str, float | int | str]]] = []
        segment_totals: dict[str, float] = {}
        for feature in sorted(features, key=lambda item: item.feature):
            feature_contributions = _missingness_contributions(current_frame, reference, feature)
            feature_total = sum(item[2] for item in feature_contributions)
            feature_causes.append(
                (
                    "feature",
                    feature.feature,
                    feature_total,
                    {"alert_metric": alert.metric, "family": alert.scope_value},
                )
            )
            for _, segment, contribution, _ in feature_contributions:
                segment_totals[segment] = segment_totals.get(segment, 0.0) + contribution
        causes = [
            (
                "segment",
                segment,
                contribution,
                {"alert_metric": alert.metric, "feature_count": len(features)},
            )
            for segment, contribution in segment_totals.items()
        ]
        causes.extend(feature_causes)
        causes.append(
            (
                "family",
                alert.scope_value,
                sum(item[2] for item in feature_causes),
                {"alert_metric": alert.metric, "feature_count": len(features)},
            )
        )
        return _rank(causes)
    if alert.scope != "feature" or alert.scope_value not in references:
        return []
    feature = references[alert.scope_value]
    if reference.segment_column not in current_frame.columns or feature.feature not in current_frame.columns:
        return []
    if alert.alert_type == "missingness":
        contributions = _missingness_contributions(current_frame, reference, feature)
    elif alert.alert_type == "distribution":
        contributions = _distribution_contributions(current_frame, reference, feature)
    else:
        contributions = []
    family = next(
        (spec.family for spec in catalog.features if spec.name == alert.scope_value), feature.family
    )
    if contributions:
        # Keep the feature as an explicit truth dimension.  Segment and family
        # aggregates remain additive, while the feature root cause prevents the
        # evaluator from comparing a feature-level truth to an opaque diagnosis.
        contributions.append(
            (
                "feature",
                feature.feature,
                abs(alert.delta or 0.0),
                {"alert_metric": alert.metric, "family": family},
            )
        )
        contributions.append(
            (
                "family",
                family,
                sum(item[2] for item in contributions if item[0] == "segment"),
                {"alert_metric": alert.metric, "feature_count": 1},
            )
        )
    return _rank(contributions)


def _missing_rate(series: pl.Series) -> float:
    invalid = sum(
        1 for value in series.drop_nulls().to_list()
        if not math.isfinite(float(value))
    )
    return (series.null_count() + invalid) / len(series) if len(series) else 0.0


def _missingness_contributions(
    frame: pl.DataFrame, reference: ReferenceSnapshot, feature: FeatureReference
) -> list[tuple[str, str, float, dict[str, float | int | str]]]:
    contributions = []
    for segment, subset in _eligible_segments(frame, reference):
        missing_rate = _missing_rate(subset.get_column(feature.feature))
        contribution = abs(missing_rate - feature.missing_rate) * (subset.height / frame.height)
        contributions.append(
            (
                "segment", segment, contribution,
                {
                    "reference_missing_rate": feature.missing_rate,
                    "current_missing_rate": missing_rate,
                    "current_share": subset.height / frame.height,
                },
            )
        )
    return contributions


def _distribution_contributions(
    frame: pl.DataFrame, reference: ReferenceSnapshot, feature: FeatureReference
) -> list[tuple[str, str, float, dict[str, float | int | str]]]:
    reference_center = (
        sum(feature.quantile_edges) / len(feature.quantile_edges)
        if feature.quantile_edges else 0.0
    )
    contributions = []
    for segment, subset in _eligible_segments(frame, reference):
        values = tuple(
            float(value) for value in subset.get_column(feature.feature).drop_nulls().to_list()
            if math.isfinite(float(value))
        )
        current_center = sum(values) / len(values) if values else 0.0
        contribution = abs(current_center - reference_center) * (subset.height / frame.height)
        contributions.append(
            (
                "segment", segment, contribution,
                {"reference_center": reference_center, "current_center": current_center},
            )
        )
    return contributions


def _eligible_segments(
    frame: pl.DataFrame, reference: ReferenceSnapshot
) -> list[tuple[str, pl.DataFrame]]:
    values = frame.get_column(reference.segment_column).to_list()
    result = []
    for segment in sorted({str(value) for value in values}):
        subset = frame.filter(pl.col(reference.segment_column).cast(pl.String) == segment)
        if subset.height >= reference.min_group_size:
            result.append((segment, subset))
    return result


def _rank(
    contributions: list[tuple[str, str, float, dict[str, float | int | str]]]
) -> list[RootCause]:
    dimension_order = {
        "segment": 0,
        "family": 1,
        "feature": 2,
        "target": 3,
        "rule": 4,
        "schema": 5,
    }
    ordered = sorted(
        contributions,
        key=lambda item: (-item[2], dimension_order.get(item[0], 99), item[0], item[1]),
    )
    return [
        RootCause(
            dimension=dimension,
            value=value,
            contribution=contribution,
            rank=index,
            evidence=evidence,
        )
        for index, (dimension, value, contribution, evidence) in enumerate(ordered, start=1)
    ]
