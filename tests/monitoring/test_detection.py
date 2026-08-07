import hashlib

import polars as pl

from riskprobe.features.catalog import FeatureCatalog
from riskprobe.monitoring.detection import detect_anomalies
from riskprobe.monitoring.reference import build_reference_snapshot


def _reference_snapshot(reference_fixture):
    return build_reference_snapshot(**reference_fixture)


def _alerts(reference_fixture, current_frame, current_rule_cards=None):
    return detect_anomalies(
        _reference_snapshot(reference_fixture),
        current_frame,
        reference_fixture["evidence_cards"] if current_rule_cards is None else current_rule_cards,
        reference_fixture["catalog"],
    )


def test_missingness_jump_creates_feature_and_family_alerts(reference_fixture) -> None:
    reference = _reference_snapshot(reference_fixture)
    browse_features = [feature.feature for feature in reference.features if feature.family == "browse"]
    current = reference_fixture["frame"].with_columns(
        [pl.lit(None, dtype=reference_fixture["frame"].schema[name]).alias(name) for name in browse_features]
    )

    alerts = _alerts(reference_fixture, current)

    assert any(
        alert.alert_type == "missingness"
        and alert.scope == "feature"
        and alert.scope_value == "browse_pv_30d"
        and alert.severity == "critical"
        for alert in alerts
    )
    family_alert = next(
        alert
        for alert in alerts
        if alert.alert_type == "missingness" and alert.scope == "family" and alert.scope_value == "browse"
    )
    assert family_alert.evidence == {
        "anomalous_feature_count": len(browse_features),
        "family_feature_count": len(browse_features),
    }


def test_missing_feature_and_type_family_changes_are_critical_schema_alerts(reference_fixture) -> None:
    reference = _reference_snapshot(reference_fixture)
    missing_feature = reference.features[0].feature
    type_changed_feature = reference.features[1].feature
    current = (
        reference_fixture["frame"]
        .drop(missing_feature)
        .with_columns(pl.col(type_changed_feature).cast(pl.String))
    )

    alerts = _alerts(reference_fixture, current)

    assert any(
        alert.alert_type == "schema"
        and alert.scope == "feature"
        and alert.scope_value == missing_feature
        and alert.severity == "critical"
        for alert in alerts
    )
    assert any(
        alert.alert_type == "schema"
        and alert.scope == "feature"
        and alert.scope_value == type_changed_feature
        and alert.severity == "critical"
        for alert in alerts
    )


def test_psi_distribution_shift_uses_reference_bins(reference_fixture) -> None:
    reference = _reference_snapshot(reference_fixture)
    feature = next(feature for feature in reference.features if len(feature.quantile_edges) > 1)
    current = reference_fixture["frame"].with_columns(
        pl.lit(feature.quantile_edges[-1] + 1_000_000.0)
        .cast(reference_fixture["frame"].schema[feature.feature])
        .alias(feature.feature)
    )

    alerts = _alerts(reference_fixture, current)

    distribution_alert = next(
        alert
        for alert in alerts
        if alert.alert_type == "distribution" and alert.scope_value == feature.feature
    )
    assert distribution_alert.metric == "psi"
    assert distribution_alert.severity == "critical"
    assert distribution_alert.current_value > 0.30


def test_label_and_institution_population_changes_are_detected(reference_fixture) -> None:
    reference = _reference_snapshot(reference_fixture)
    frame = reference_fixture["frame"]
    target = reference_fixture["config"].columns.target
    institution = reference_fixture["config"].columns.segment
    institution_code = next(iter(reference.segment_counts))
    current = frame.with_columns(
        [
            pl.lit(1, dtype=frame.schema[target]).alias(target),
            pl.lit(institution_code, dtype=frame.schema[institution]).alias(institution),
        ]
    )

    alerts = _alerts(reference_fixture, current)

    assert any(
        alert.alert_type == "label"
        and alert.scope == "dataset"
        and alert.metric == "positive_rate"
        and alert.severity == "critical"
        for alert in alerts
    )
    assert any(
        alert.alert_type == "population"
        and alert.scope == "institution"
        and alert.scope_value == institution_code
        and alert.metric == "share"
        and alert.severity == "warning"
        for alert in alerts
    )


def test_population_detection_uses_snapshot_role_metadata_and_shared_group_threshold(
    reference_fixture,
) -> None:
    config = reference_fixture["config"].model_copy(
        update={
            "columns": reference_fixture["config"].columns.model_copy(
                update={"target": "event", "segment": "cohort"}
            )
        }
    )
    frame = reference_fixture["frame"].rename({"target": "event", "institution": "cohort"})
    reference = build_reference_snapshot(**dict(reference_fixture, frame=frame, config=config))
    reference_segment = next(iter(reference.segment_counts))
    current = frame.with_columns(pl.lit(reference_segment).alias("cohort"))
    feature_only_catalog = FeatureCatalog(
        features=tuple(
            spec
            for spec in reference_fixture["catalog"].features
            if spec.name in {feature.feature for feature in reference.features}
        )
    )

    alerts = detect_anomalies(reference, current, (), feature_only_catalog)

    assert any(
        alert.alert_type == "population"
        and alert.scope_value == reference_segment
        for alert in alerts
    )


def test_rule_lift_decay_is_detected_against_reference_lift(reference_fixture) -> None:
    reference = _reference_snapshot(reference_fixture)
    card = reference_fixture["evidence_cards"][0]
    current_card = card.model_copy(
        update={"test": card.test.model_copy(update={"lift": card.test.lift * 0.5})}
    )

    alerts = _alerts(reference_fixture, reference_fixture["frame"], (current_card,))

    rule_alert = next(alert for alert in alerts if alert.alert_type == "rule_decay")
    assert rule_alert.scope == "rule"
    assert rule_alert.scope_value == reference.rules[0].rule_id
    assert rule_alert.metric == "lift_decay"
    assert rule_alert.severity == "critical"
    assert rule_alert.reference_value == reference.rules[0].lift
    assert rule_alert.current_value == current_card.test.lift
    assert rule_alert.delta == 0.5


def test_alert_ids_are_stable_sha256_prefixes(reference_fixture) -> None:
    reference = _reference_snapshot(reference_fixture)
    feature = reference.features[0].feature
    current = reference_fixture["frame"].drop(feature)

    first = _alerts(reference_fixture, current)
    second = _alerts(reference_fixture, current)

    assert first == second
    for alert in first:
        expected = hashlib.sha256(
            f"{alert.alert_type}|{alert.scope}|{alert.scope_value}|{alert.metric}".encode()
        ).hexdigest()[:12]
        assert alert.alert_id == expected
