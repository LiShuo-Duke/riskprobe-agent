from datetime import date
from types import MappingProxyType

import pytest
from pydantic import ValidationError

from riskprobe.features.catalog import QualityIssue
from riskprobe.monitoring.models import (
    DiagnosticReport,
    FindingKind,
    FindingSeverity,
    RiskFinding,
    SafeProfile,
)
from riskprobe.privacy import assert_safe_payload
from riskprobe.profiling import DatasetProfile


def _profile() -> SafeProfile:
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
        metadata_grade="A",
        issue_codes=(),
        issue_count=0,
    )


def _finding(*, code: str = "missing_values", count: int = 2) -> RiskFinding:
    return RiskFinding(
        kind=FindingKind.DATA_QUALITY,
        severity=FindingSeverity.WARNING,
        code=code,
        feature="score_1",
        metrics={"affected_count": count, "affected_rate": 0.02},
    )


def test_finding_id_is_stable_for_canonical_safe_payload() -> None:
    first = _finding()
    second = RiskFinding(
        kind=FindingKind.DATA_QUALITY,
        severity=FindingSeverity.WARNING,
        code="missing_values",
        feature="score_1",
        metrics={"affected_rate": 0.02, "affected_count": 2},
    )

    assert first.finding_id == second.finding_id
    assert len(first.finding_id) == 64
    assert first.finding_id != _finding(count=3).finding_id


def test_finding_rejects_a_supplied_noncanonical_id() -> None:
    with pytest.raises(ValidationError, match="finding_id"):
        RiskFinding(
            finding_id="f" * 64,
            kind=FindingKind.DATA_QUALITY,
            severity=FindingSeverity.WARNING,
            code="missing_values",
            metrics={"affected_count": 2},
        )


def test_models_are_strict_frozen_and_forbid_unknown_fields() -> None:
    finding = _finding()

    with pytest.raises(ValidationError):
        RiskFinding.model_validate(
            {
                "kind": "data_quality",
                "severity": "warning",
                "code": "missing_values",
                "metrics": {"affected_count": "2"},
            }
        )
    with pytest.raises(ValidationError):
        SafeProfile.model_validate({**_profile().model_dump(), "issue_codes": []})
    with pytest.raises(ValidationError):
        RiskFinding.model_validate({**finding.model_dump(), "unexpected": True})
    with pytest.raises(ValidationError):
        finding.code = "changed"


def test_safe_profile_removes_real_segment_values_and_issue_messages() -> None:
    source = DatasetProfile(
        dataset_id="dataset-1",
        row_count=100,
        feature_count=2,
        positive_rate=0.2,
        segment_counts=MappingProxyType({"secret-bank": 60, "vip-clients": 40}),
        snapshot_min=date(2024, 1, 1),
        snapshot_max=date(2024, 2, 1),
        metadata_grade="B",
        issues=(
            QualityIssue(
                code="SINGLE_CLASS_SLICE",
                severity="warning",
                family="institution",
                features=(),
                affected_rows=40,
                message="institution slice 'secret-bank' contains a single target class",
            ),
        ),
    )

    safe = SafeProfile.from_profile(source)
    dumped = safe.model_dump(mode="json")

    assert safe.segment_count == 2
    assert safe.min_segment_size == 40
    assert safe.max_segment_size == 60
    assert safe.issue_codes == ("SINGLE_CLASS_SLICE",)
    assert "secret-bank" not in str(dumped)
    assert "vip-clients" not in str(dumped)
    assert_safe_payload(dumped)


def test_report_normalizes_finding_order_and_is_privacy_safe() -> None:
    lower = RiskFinding(
        kind=FindingKind.DATA_QUALITY,
        severity=FindingSeverity.WARNING,
        code="constant_feature",
        metrics={"affected_count": 100, "affected_rate": 1.0},
    )
    higher = RiskFinding(
        kind=FindingKind.TARGET_SHIFT,
        severity=FindingSeverity.CRITICAL,
        code="target_rate_shift",
        metrics={"rate_shift": 0.3},
    )

    report = DiagnosticReport(profile=_profile(), findings=(lower, higher))

    assert report.dataset_id == "dataset-1"
    assert report.metadata_grade == "A"
    assert report.findings == (higher, lower)
    assert_safe_payload(report.model_dump(mode="json"))
