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
