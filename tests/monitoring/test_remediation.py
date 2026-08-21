from __future__ import annotations

from riskprobe.monitoring.models import (
    Alert,
    Diagnosis,
    FeatureReference,
    FindingKind,
    FindingSeverity,
    ReferenceSnapshot,
    RootCause,
    RiskFinding,
)
from riskprobe.monitoring.remediation import (
    RemediationStatus,
    apply_verification,
    build_monitoring_remediation_plan,
    create_remediation,
    verify_remediation,
)


def _reference() -> ReferenceSnapshot:
    return ReferenceSnapshot(
        snapshot_id="snapshot-1",
        dataset_id="dataset-1",
        row_count=100,
        positive_rate=0.1,
        target_column="target",
        segment_column="segment",
        min_group_size=5,
        segment_counts={"token-a": 100},
        features=(
            FeatureReference(
                feature="amount",
                family="transaction",
                dtype="float64",
                missing_rate=0.0,
                zero_rate=0.0,
                quantile_edges=(0.0, 1.0),
                histogram_counts=(100,),
            ),
        ),
        rules=(),
        created_at="2026-08-21T00:00:00Z",
    )


def _alert_and_diagnosis() -> tuple[Alert, Diagnosis]:
    alert = Alert(
        alert_id="alert-1",
        alert_type="distribution",
        severity="warning",
        scope="feature",
        scope_value="amount",
        metric="psi",
        reference_value=0.1,
        current_value=0.3,
        delta=0.2,
        evidence={"psi": 0.2},
    )
    diagnosis = Diagnosis(
        snapshot_id="snapshot-1",
        alerts=(alert,),
        root_causes=(
            RootCause(
                dimension="feature",
                value="amount",
                contribution=0.2,
                rank=1,
                evidence={"psi": 0.2},
            ),
        ),
        created_at="1970-01-01T00:00:00Z",
    )
    return alert, diagnosis


def test_alert_diagnosis_recommendation_and_retest_form_one_chain() -> None:
    alert, diagnosis = _alert_and_diagnosis()
    plan = build_monitoring_remediation_plan(
        _reference(),
        alerts=(alert,),
        diagnoses=(diagnosis,),
        metadata_grade="B",
    )

    assert plan.report.findings[0].kind is FindingKind.FEATURE_DRIFT
    assert plan.recommendations[0].action_code == "investigate_feature_drift"
    assert plan.recommendations[0].decision_eligibility == "analysis_only"

    remediation = create_remediation(plan.recommendations[0])
    before = plan.report.findings
    after = (
        RiskFinding(
            kind=before[0].kind,
            severity=FindingSeverity.INFO,
            code=before[0].code,
            feature=before[0].feature,
            metrics={"reference_value": 0.1, "current_value": 0.12, "delta": 0.02},
            limitations=before[0].limitations,
        ),
    )
    verification = verify_remediation(remediation, before, after)
    updated = apply_verification(remediation, verification)

    assert verification.status == "verified"
    assert verification.resolved_finding_ids == (before[0].finding_id,)
    assert updated.status is RemediationStatus.VERIFIED
    assert updated.verification_id == verification.verification_id
