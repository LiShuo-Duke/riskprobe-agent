import polars as pl

from riskprobe.monitoring.detection import detect_anomalies
from riskprobe.monitoring.diagnosis import diagnose_alerts
from riskprobe.monitoring.injection import DriftScenario, inject_drift
from riskprobe.monitoring.reference import build_reference_snapshot


def test_browser_missingness_is_attributed_to_affected_institution_and_family(reference_fixture) -> None:
    reference = build_reference_snapshot(**reference_fixture)
    institution = reference_fixture["frame"].get_column("institution")[0]
    injected = inject_drift(
        reference_fixture["frame"],
        DriftScenario(
            scenario_id="browse-institution",
            drift_type="missingness",
            target="browse_pv_30d",
            magnitude=0.80,
            institution=institution,
        ),
        seed=7,
    )
    alerts = detect_anomalies(reference, injected.frame, (), reference_fixture["catalog"])

    diagnoses = diagnose_alerts(alerts, reference, injected.frame, reference_fixture["catalog"], top_k=3)

    causes = next(
        diagnosis.root_causes
        for diagnosis in diagnoses
        if diagnosis.alerts[0].scope_value == "browse_pv_30d"
    )
    assert causes[0].dimension == "segment"
    assert causes[0].value == institution
    assert any(cause.dimension == "family" and cause.value == "browse" for cause in causes)


def test_diagnosis_covers_label_population_and_rule_decay_with_finite_missingness(reference_fixture) -> None:
    reference = build_reference_snapshot(**reference_fixture)
    frame = reference_fixture["frame"]
    target = reference.target_column
    segment = reference.segment_column
    current = frame.with_columns(
        [
            frame.get_column(target).cast(float).alias(target),
            frame.get_column(segment).alias(segment),
        ]
    )
    alerts = detect_anomalies(reference, current, (), reference_fixture["catalog"])
    label_alert = next(
        alert for alert in alerts if alert.alert_type == "label"
    ) if any(alert.alert_type == "label" for alert in alerts) else None
    if label_alert is None:
        from riskprobe.monitoring.models import Alert
        label_alert = Alert(
            alert_id="label", alert_type="label", severity="warning", scope="dataset",
            scope_value=reference.dataset_id, metric="positive_rate", reference_value=0.1,
            current_value=0.2, delta=0.1, evidence={}
        )
    diagnoses = diagnose_alerts(
        (label_alert,), reference, current, reference_fixture["catalog"], top_k=3
    )
    assert diagnoses[0].root_causes
    assert diagnoses[0].root_causes[0].dimension in {"target", "segment", "family"}


def test_family_missingness_diagnosis_has_additive_family_root_cause(reference_fixture) -> None:
    reference = build_reference_snapshot(**reference_fixture)
    browse_features = [
        feature.feature for feature in reference.features if feature.family == "browse"
    ]
    current = reference_fixture["frame"].with_columns(
        [
            pl.lit(None, dtype=reference_fixture["frame"].schema[name]).alias(name)
            for name in browse_features
        ]
    )
    alerts = detect_anomalies(reference, current, (), reference_fixture["catalog"])
    family_alert = next(
        alert
        for alert in alerts
        if alert.alert_type == "missingness"
        and alert.scope == "family"
        and alert.scope_value == "browse"
    )

    diagnosis = diagnose_alerts(
        (family_alert,), reference, current, reference_fixture["catalog"], top_k=10
    )[0]
    family_cause = next(
        cause for cause in diagnosis.root_causes if cause.dimension == "family"
    )
    feature_total = sum(
        cause.contribution
        for cause in diagnosis.root_causes
        if cause.dimension == "feature"
    )

    assert family_cause.value == "browse"
    assert family_cause.contribution == feature_total
