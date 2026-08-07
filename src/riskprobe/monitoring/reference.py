import hashlib
import json
import math
from collections.abc import Iterable

import polars as pl

from riskprobe.config import ProjectConfig
from riskprobe.features.catalog import FeatureCatalog, FeatureSpec
from riskprobe.models import EvidenceCard
from riskprobe.profiling import DatasetProfile

from .models import FeatureReference, ReferenceSnapshot, RuleReference

_CREATED_AT = "1970-01-01T00:00:00Z"
_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)
_SEGMENT_NAMESPACE = "riskprobe-monitoring-segment-v1:"


def build_reference_snapshot(
    frame: pl.DataFrame,
    profile: DatasetProfile,
    evidence_cards: Iterable[EvidenceCard],
    catalog: FeatureCatalog,
    config: ProjectConfig,
) -> ReferenceSnapshot:
    """Build a reproducible snapshot from configured features and aggregate inputs only."""
    role_columns = (
        config.columns.entity,
        config.columns.snapshot,
        config.columns.segment,
        config.columns.target,
    )
    selected_features = tuple(config.features.select_columns(frame.columns, role_columns))
    catalog_by_name = {spec.name: spec for spec in catalog.features}
    features = tuple(
        _feature_reference(frame.get_column(feature), catalog_by_name.get(feature), feature)
        for feature in selected_features
    )
    rules = tuple(sorted((_rule_reference(card) for card in evidence_cards), key=lambda rule: rule.rule_id))
    snapshot_data = {
        "dataset_id": profile.dataset_id,
        "row_count": profile.row_count,
        "positive_rate": _positive_rate(profile),
        "segment_counts": _anonymize_segment_counts(profile.segment_counts.items()),
        "features": features,
        "rules": rules,
        "created_at": _CREATED_AT,
    }
    snapshot_id = _snapshot_id(snapshot_data)
    return ReferenceSnapshot(snapshot_id=snapshot_id, **snapshot_data)


def _feature_reference(
    series: pl.Series,
    spec: FeatureSpec | None,
    feature: str,
) -> FeatureReference:
    row_count = len(series)
    family = spec.family if spec is not None else "unknown"
    numeric_values = _finite_numeric_values(series)
    null_count = series.null_count()
    invalid_count = series.len() - null_count - len(numeric_values) if series.dtype.is_numeric() else 0
    missing_rate = (null_count + invalid_count) / row_count if row_count else 0.0
    zero_rate = (
        sum(value == 0.0 for value in numeric_values) / len(numeric_values) if numeric_values else 0.0
    )
    quantile_edges, histogram_counts = _histogram(numeric_values)
    return FeatureReference(
        feature=feature,
        family=family,
        dtype=str(series.dtype),
        missing_rate=float(missing_rate),
        zero_rate=float(zero_rate),
        quantile_edges=quantile_edges,
        histogram_counts=histogram_counts,
    )


def _finite_numeric_values(series: pl.Series) -> tuple[float, ...]:
    if not series.dtype.is_numeric():
        return ()
    return tuple(
        float(value)
        for value in series.drop_nulls().to_list()
        if math.isfinite(float(value))
    )


def _histogram(values: tuple[float, ...]) -> tuple[tuple[float, ...], tuple[int, ...]]:
    if not values:
        return (), ()
    ordered = tuple(sorted(values))
    quantile_edges = tuple(dict.fromkeys(_quantile(ordered, quantile) for quantile in _QUANTILES))
    if len(quantile_edges) == 1:
        return quantile_edges * 2, (len(ordered),)
    histogram_counts = tuple(
        sum(
            lower <= value < upper or (index == len(quantile_edges) - 2 and value == upper)
            for value in ordered
        )
        for index, (lower, upper) in enumerate(zip(quantile_edges, quantile_edges[1:]))
    )
    return quantile_edges, histogram_counts


def _quantile(values: tuple[float, ...], quantile: float) -> float:
    position = (len(values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower = values[lower_index]
    upper = values[upper_index]
    return lower + (upper - lower) * (position - lower_index)


def _rule_reference(card: EvidenceCard) -> RuleReference:
    return RuleReference(
        rule_id=card.rule.rule_id,
        coverage=card.test.coverage,
        bad_rate=card.test.hit_bad_rate,
        lift=card.test.lift,
    )


def _anonymize_segment_counts(segment_counts: Iterable[tuple[str, int]]) -> dict[str, int]:
    anonymized = {
        "segment_" + hashlib.sha256(f"{_SEGMENT_NAMESPACE}{segment}".encode()).hexdigest()[:16]: int(count)
        for segment, count in segment_counts
    }
    return dict(sorted(anonymized.items()))


def _positive_rate(profile: DatasetProfile) -> float:
    return 0.0 if profile.positive_rate is None else profile.positive_rate


def _snapshot_id(snapshot_data: dict[str, object]) -> str:
    canonical = json.dumps(
        snapshot_data,
        default=_json_default,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _json_default(value: object) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"unsupported snapshot aggregate type: {type(value)!r}")
