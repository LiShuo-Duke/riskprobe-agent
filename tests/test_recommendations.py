import pytest
from pydantic import ValidationError

from riskprobe.monitoring.models import (
    DiagnosticReport,
    FindingKind,
    FindingSeverity,
    RiskFinding,
    SafeProfile,
)
from riskprobe.privacy import assert_safe_payload
from riskprobe.recommendations import (
    DecisionEligibility,
    Recommendation,
    build_recommendations,
)


def _profile(grade: str = "A") -> SafeProfile:
    return SafeProfile(
        dataset_id="dataset-1",
        row_count=100,
        feature_count=2,
        positive_rate=0.2,
        segment_count=2,
        min_segment_size=40,
        max_segment_size=60,
        snapshot_min="2024-01-01",
        snapshot_max="2024-02-01",
        metadata_grade=grade,
        issue_codes=(),
        issue_count=0,
    )


def _finding(kind: FindingKind, code: str) -> RiskFinding:
    return RiskFinding(
        kind=kind,
        severity=FindingSeverity.WARNING,
        code=code,
        metrics={"affected_count": 10, "affected_rate": 0.1},
    )


@pytest.mark.parametrize(
    ("kind", "expected_action"),
    [
        (FindingKind.DATA_QUALITY, "remediate_data_quality"),
        (FindingKind.FEATURE_DRIFT, "investigate_feature_drift"),
        (FindingKind.POPULATION_SHIFT, "investigate_population_shift"),
        (FindingKind.TARGET_SHIFT, "investigate_target_shift"),
        (FindingKind.SEGMENT_RISK, "review_segment_risk"),
        (FindingKind.TIME_INSTABILITY, "monitor_time_stability"),
        (FindingKind.RULE_EVIDENCE, "review_rule_evidence"),
    ],
)
def test_each_finding_kind_produces_an_evidence_linked_action(
    kind: FindingKind,
    expected_action: str,
) -> None:
    finding = _finding(kind, f"{kind.value}_code")
    report = DiagnosticReport(profile=_profile(), findings=(finding,))

    recommendations = build_recommendations(report, metadata_grade="A")

    assert len(recommendations) == 1
    recommendation = recommendations[0]
    assert recommendation.action_code == expected_action
    assert recommendation.finding_ids == (finding.finding_id,)
    assert recommendation.human_approval_required is True
    assert recommendation.decision_eligibility is DecisionEligibility.HUMAN_REVIEW_REQUIRED


def test_grade_b_forces_analysis_only_and_never_suggests_automatic_change() -> None:
    findings = (
        _finding(FindingKind.DATA_QUALITY, "missing_values"),
        _finding(FindingKind.FEATURE_DRIFT, "feature_psi"),
        _finding(FindingKind.SEGMENT_RISK, "segment_rate"),
        _finding(FindingKind.TIME_INSTABILITY, "monthly_target_rate_change"),
        _finding(FindingKind.RULE_EVIDENCE, "weak_rule"),
    )
    report = DiagnosticReport(profile=_profile("B"), findings=findings)

    recommendations = build_recommendations(report, metadata_grade="B")

    assert recommendations
    assert all(
        item.decision_eligibility is DecisionEligibility.ANALYSIS_ONLY
        for item in recommendations
    )
    assert all(item.human_approval_required is True for item in recommendations)
    assert all(
        forbidden not in item.action_code
        for item in recommendations
        for forbidden in ("deploy", "production", "threshold_change", "auto")
    )


def test_grade_argument_cannot_bypass_report_grade() -> None:
    finding = _finding(FindingKind.DATA_QUALITY, "missing_values")
    report = DiagnosticReport(profile=_profile("B"), findings=(finding,))

    with pytest.raises(ValueError, match="metadata grade"):
        build_recommendations(report, metadata_grade="A")


def test_recommendation_requires_real_finding_evidence_and_is_strict() -> None:
    with pytest.raises(ValidationError, match="finding_ids"):
        Recommendation(
            action_code="remediate_data_quality",
            priority="high",
            finding_ids=(),
            rationale_code="quality_finding_present",
            human_approval_required=True,
            decision_eligibility=DecisionEligibility.HUMAN_REVIEW_REQUIRED,
        )
    with pytest.raises(ValidationError):
        Recommendation.model_validate(
            {
                "action_code": "remediate_data_quality",
                "priority": "high",
                "finding_ids": ["a" * 64],
                "rationale_code": "quality_finding_present",
                "human_approval_required": True,
                "decision_eligibility": "human_review_required",
            }
        )


def test_recommendations_are_grouped_ordered_and_privacy_safe() -> None:
    findings = (
        _finding(FindingKind.TIME_INSTABILITY, "time_b"),
        _finding(FindingKind.DATA_QUALITY, "quality_b"),
        _finding(FindingKind.DATA_QUALITY, "quality_a"),
    )
    report = DiagnosticReport(profile=_profile(), findings=findings)

    first = build_recommendations(report, metadata_grade="A")
    second = build_recommendations(report, metadata_grade="A")

    assert first == second
    quality = next(item for item in first if item.action_code == "remediate_data_quality")
    assert quality.finding_ids == tuple(sorted(quality.finding_ids))
    assert len(quality.finding_ids) == 2
    assert all(len(item.recommendation_id) == 64 for item in first)
    for recommendation in first:
        assert_safe_payload(recommendation.model_dump(mode="json"))
