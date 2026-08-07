"""Deterministic, aggregate-only attribution for monitoring alerts."""

from collections.abc import Iterable

import polars as pl

from riskprobe.features.catalog import FeatureCatalog

from .models import Alert, Diagnosis, ReferenceSnapshot, RootCause

_CREATED_AT = "1970-01-01T00:00:00Z"


def diagnose_alerts(
    alerts: Iterable[Alert],
    reference: ReferenceSnapshot,
    current_frame: pl.DataFrame,
    catalog: FeatureCatalog,
    top_k: int,
) -> list[Diagnosis]:
    """Rank segment and feature-family aggregate contributions for each alert."""
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
    references: dict[str, object],
    catalog: FeatureCatalog,
) -> list[RootCause]:
    if (
        alert.scope != "feature"
        or alert.scope_value not in references
        or reference.segment_column not in current_frame.columns
    ):
        return []
    feature = references[alert.scope_value]
    if alert.alert_type == "missingness":
        contributions = _missingness_contributions(current_frame, reference, feature)
    elif alert.alert_type == "distribution":
        contributions = _distribution_contributions(current_frame, reference, feature)
    else:
        contributions = []
    family = next(
        (spec.family for spec in catalog.features if spec.name == alert.scope_value),
        feature.family,
    )
    if contributions:
        contributions.append(
            (
                "family",
                family,
                sum(item[2] for item in contributions) / (2 * len(contributions)),
                {"alert_metric": alert.metric, "feature_count": 1},
            )
        )
    return _rank(contributions)


def _missingness_contributions(
    frame: pl.DataFrame, reference: ReferenceSnapshot, feature: object
) -> list[tuple[str, str, float, dict[str, float | int | str]]]:
    reference_rate = feature.missing_rate
    contributions = []
    for segment, subset in _eligible_segments(frame, reference):
        missing_rate = subset.get_column(feature.feature).null_count() / subset.height
        contribution = abs(missing_rate - reference_rate) * (subset.height / frame.height)
        contributions.append(
            (
                "segment",
                segment,
                contribution,
                {
                    "reference_missing_rate": reference_rate,
                    "current_missing_rate": missing_rate,
                    "current_share": subset.height / frame.height,
                },
            )
        )
    return contributions


def _distribution_contributions(
    frame: pl.DataFrame, reference: ReferenceSnapshot, feature: object
) -> list[tuple[str, str, float, dict[str, float | int | str]]]:
    reference_mean = sum(feature.quantile_edges) / len(feature.quantile_edges) if feature.quantile_edges else 0.0
    contributions = []
    for segment, subset in _eligible_segments(frame, reference):
        values = subset.get_column(feature.feature).drop_nulls()
        current_mean = float(values.mean()) if values.len() else 0.0
        contribution = abs(current_mean - reference_mean) * (subset.height / frame.height)
        contributions.append(
            (
                "segment",
                segment,
                contribution,
                {"reference_center": reference_mean, "current_center": current_mean},
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
    ordered = sorted(contributions, key=lambda item: (-item[2], item[0], item[1]))
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
