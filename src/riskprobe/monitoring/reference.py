import hashlib
import hmac
import json
import math
import re
from collections.abc import Iterable

import polars as pl

from riskprobe.config import ProjectConfig
from riskprobe.features.catalog import FeatureCatalog, FeatureSpec
from riskprobe.models import EvidenceCard
from riskprobe.profiling import DatasetProfile

from .models import FeatureReference, ReferenceSnapshot, RuleReference

_CREATED_AT = "1970-01-01T00:00:00Z"
_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)
_TOKEN_DOMAINS = {
    "dataset": b"riskprobe-monitoring-dataset-v1:\x00",
    "rule": b"riskprobe-monitoring-rule-v1:\x00",
    "segment": b"riskprobe-monitoring-segment-v1:\x00",
}
_TOKEN_NAMESPACE = re.compile(r"[a-z][a-z0-9-]{2,63}\Z")


def build_reference_snapshot(
    frame: pl.DataFrame,
    profile: DatasetProfile,
    evidence_cards: Iterable[EvidenceCard],
    catalog: FeatureCatalog,
    config: ProjectConfig,
    *,
    privacy_key: bytes,
    token_namespace: str,
) -> ReferenceSnapshot:
    """Build a reproducible aggregate snapshot from numeric configured features only.

    Callers must inject a non-empty secret ``privacy_key`` and a non-secret,
    stable ``token_namespace``. Every caller-provided dataset, rule, and segment
    value is HMAC-tokenized before it reaches the returned model; the key is
    transient and never included in model data or exception text. Consumers must
    reject comparisons across different namespaces with the model assertion.
    """
    privacy_key = _require_privacy_key(privacy_key)
    token_namespace = _require_token_namespace(token_namespace)
    dataset_id = _tokenize_identifier(profile.dataset_id, "dataset", privacy_key)
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
    rules = tuple(
        sorted(
            (_rule_reference(card, privacy_key) for card in evidence_cards),
            key=lambda rule: rule.rule_id,
        )
    )
    _require_unique_rule_ids(rules)
    snapshot_data = {
        "dataset_id": dataset_id,
        "token_namespace": token_namespace,
        "row_count": profile.row_count,
        "positive_rate": _positive_rate(profile),
        "segment_counts": _anonymize_segment_counts(profile.segment_counts.items(), privacy_key),
        "features": features,
        "rules": rules,
        "created_at": _CREATED_AT,
    }
    snapshot_id = _snapshot_id(snapshot_data)
    return ReferenceSnapshot(snapshot_id=snapshot_id, **snapshot_data)


def _require_privacy_key(privacy_key: bytes) -> bytes:
    if not isinstance(privacy_key, bytes) or not privacy_key:
        raise ValueError("privacy key must be non-empty bytes")
    return privacy_key


def _require_token_namespace(token_namespace: str) -> str:
    if not isinstance(token_namespace, str) or not _TOKEN_NAMESPACE.fullmatch(token_namespace):
        raise ValueError("token namespace must be a strict safe ID")
    return token_namespace


def _tokenize_identifier(identifier: str, kind: str, privacy_key: bytes) -> str:
    if not isinstance(identifier, str):
        raise ValueError(f"{kind} identifier must be a string")
    return f"{kind}_{_token_digest(identifier, kind, privacy_key)}"


def _token_digest(value: str, kind: str, privacy_key: bytes) -> str:
    return hmac.new(
        privacy_key,
        _TOKEN_DOMAINS[kind] + value.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _feature_reference(
    series: pl.Series,
    spec: FeatureSpec | None,
    feature: str,
) -> FeatureReference:
    if not series.dtype.is_numeric():
        raise ValueError(
            f"selected feature '{feature}' has unsupported dtype; numeric features are required"
        )
    row_count = len(series)
    family = spec.family if spec is not None else "unknown"
    numeric_values = _finite_numeric_values(series)
    null_count = series.null_count()
    invalid_count = series.len() - null_count - len(numeric_values)
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


def _rule_reference(card: EvidenceCard, privacy_key: bytes) -> RuleReference:
    return RuleReference(
        rule_id=_tokenize_identifier(card.rule.rule_id, "rule", privacy_key),
        coverage=card.test.coverage,
        bad_rate=card.test.hit_bad_rate,
        lift=card.test.lift,
    )


def _require_unique_rule_ids(rules: tuple[RuleReference, ...]) -> None:
    if len({rule.rule_id for rule in rules}) != len(rules):
        raise ValueError("duplicate rule identifier")


def _anonymize_segment_counts(
    segment_counts: Iterable[tuple[str, int]],
    privacy_key: bytes,
) -> dict[str, int]:
    anonymized: dict[str, int] = {}
    for segment, count in segment_counts:
        token = f"segment_{_token_digest(str(segment), 'segment', privacy_key)}"
        if token in anonymized:
            raise ValueError("token collision")
        anonymized[token] = int(count)
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
