"""Bounded alert-to-recommendation remediation and retest contracts."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

from riskprobe.monitoring.models import (
    Alert,
    Diagnosis,
    DiagnosticReport,
    FindingKind,
    FindingSeverity,
    ReferenceSnapshot,
    RiskFinding,
    SafeProfile,
)
from riskprobe.privacy import canonical_payload_hash
from riskprobe.recommendations.engine import build_recommendations
from riskprobe.recommendations.models import Recommendation
from riskprobe.recommendations.policy import ActionCode

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

AlertMetricDirection = Literal["decrease", "increase", "toward"]
VerificationStatus = Literal["verified", "remaining", "inconclusive"]

_ALERT_KIND = {
    "schema": FindingKind.DATA_QUALITY,
    "missingness": FindingKind.DATA_QUALITY,
    "distribution": FindingKind.FEATURE_DRIFT,
    "population": FindingKind.POPULATION_SHIFT,
    "label": FindingKind.TARGET_SHIFT,
    "rule_decay": FindingKind.RULE_EVIDENCE,
}
_ALERT_RISK_METRICS: Mapping[str, AlertMetricDirection] = {
    "affected_count": "decrease",
    "affected_rate": "decrease",
    "delta": "decrease",
    "drop_rate": "decrease",
    "missing_rate": "decrease",
    "psi": "decrease",
    "rate_shift": "decrease",
    "lift": "toward",
}
_SEVERITY_RANK = {
    FindingSeverity.CRITICAL: 0,
    FindingSeverity.WARNING: 1,
    FindingSeverity.INFO: 2,
}


class RemediationStatus(StrEnum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    AWAITING_VERIFICATION = "awaiting_verification"
    VERIFIED = "verified"
    FAILED = "failed"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )


class MonitoringRemediationPlan(_FrozenModel):
    """The deterministic output of the monitor diagnosis-to-recommendation bridge."""

    reference_snapshot_id: str
    diagnoses: tuple[Diagnosis, ...]
    report: DiagnosticReport
    recommendations: tuple[Recommendation, ...]


class RemediationRecord(_FrozenModel):
    """Append-only remediation state linked to one recommendation."""

    remediation_id: str = ""
    recommendation_id: str
    action_code: str
    finding_ids: tuple[str, ...]
    status: RemediationStatus = RemediationStatus.OPEN
    verification_id: str | None = None

    @field_validator("remediation_id", "recommendation_id", "verification_id")
    @classmethod
    def validate_hash_id(cls, value: str | None) -> str | None:
        if value is not None and value and _SHA256.fullmatch(value) is None:
            raise ValueError("remediation identifiers must be SHA-256 values")
        return value

    @field_validator("action_code")
    @classmethod
    def validate_action_code(cls, value: str) -> str:
        if value not in {action.value for action in ActionCode}:
            raise ValueError("action code is not allowlisted")
        return value

    @field_validator("finding_ids")
    @classmethod
    def validate_finding_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("finding_ids must be non-empty and unique")
        if any(_SHA256.fullmatch(item) is None for item in value):
            raise ValueError("finding_ids must contain SHA-256 identifiers")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def derive_id(self) -> RemediationRecord:
        payload = self.model_dump(mode="json", exclude={"remediation_id"})
        expected = canonical_payload_hash(payload)
        if self.remediation_id and self.remediation_id != expected:
            raise ValueError("remediation_id does not match canonical payload")
        object.__setattr__(self, "remediation_id", expected)
        return self


class RemediationVerification(_FrozenModel):
    """Aggregate before/after result for one remediation record."""

    verification_id: str = ""
    remediation_id: str
    status: VerificationStatus
    resolved_finding_ids: tuple[str, ...] = ()
    remaining_finding_ids: tuple[str, ...] = ()
    inconclusive_finding_ids: tuple[str, ...] = ()
    metric_deltas: Mapping[str, float] = Field(default_factory=dict)

    @field_validator("verification_id")
    @classmethod
    def validate_verification_id(cls, value: str) -> str:
        if value and _SHA256.fullmatch(value) is None:
            raise ValueError("verification identifiers must be SHA-256 values")
        return value

    @field_validator("remediation_id")
    @classmethod
    def validate_remediation_id(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("remediation identifiers must be SHA-256 values")
        return value

    @field_validator(
        "resolved_finding_ids",
        "remaining_finding_ids",
        "inconclusive_finding_ids",
    )
    @classmethod
    def validate_finding_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or any(_SHA256.fullmatch(item) is None for item in value):
            raise ValueError("verification finding IDs must be unique SHA-256 values")
        return tuple(sorted(value))

    @field_validator("metric_deltas")
    @classmethod
    def freeze_metric_deltas(cls, value: Mapping[str, float]) -> Mapping[str, float]:
        normalized: dict[str, float] = {}
        for key, metric in value.items():
            if not key or len(key) > 256 or not math.isfinite(float(metric)):
                raise ValueError("metric deltas must be bounded finite numbers")
            normalized[key] = float(metric)
        return dict(sorted(normalized.items()))

    @field_serializer("metric_deltas")
    def serialize_metric_deltas(self, value: Mapping[str, float]) -> dict[str, float]:
        return dict(value)

    @model_validator(mode="after")
    def validate_sets_and_id(self) -> RemediationVerification:
        sets = (
            set(self.resolved_finding_ids),
            set(self.remaining_finding_ids),
            set(self.inconclusive_finding_ids),
        )
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError("verification finding IDs must be disjoint")
        if self.status == "verified" and (sets[1] or sets[2]):
            raise ValueError("verified remediation cannot have unresolved findings")
        payload = self.model_dump(mode="json", exclude={"verification_id"})
        expected = canonical_payload_hash(payload)
        if self.verification_id and self.verification_id != expected:
            raise ValueError("verification_id does not match canonical payload")
        object.__setattr__(self, "verification_id", expected)
        return self


def build_monitoring_remediation_plan(
    reference: ReferenceSnapshot,
    *,
    alerts: Iterable[Alert],
    diagnoses: Iterable[Diagnosis],
    metadata_grade: Literal["A", "B"] = "A",
) -> MonitoringRemediationPlan:
    """Bridge monitor alerts and diagnoses into existing finding recommendations."""

    alert_values = tuple(alerts)
    if any(not isinstance(alert, Alert) for alert in alert_values):
        raise TypeError("alerts must contain Alert values")
    alert_by_id = {alert.alert_id: alert for alert in alert_values}
    if len(alert_by_id) != len(alert_values):
        raise ValueError("alerts must have unique alert IDs")

    diagnosis_values = tuple(diagnoses)
    diagnosis_by_alert: dict[str, Diagnosis] = {}
    for diagnosis in diagnosis_values:
        if not isinstance(diagnosis, Diagnosis):
            raise TypeError("diagnoses must contain Diagnosis values")
        if diagnosis.snapshot_id != reference.snapshot_id:
            raise ValueError("diagnosis snapshot does not match reference snapshot")
        for alert in diagnosis.alerts:
            if alert.alert_id not in alert_by_id:
                raise ValueError("diagnosis references an unavailable alert")
            if alert.alert_id in diagnosis_by_alert:
                raise ValueError("each alert may have only one diagnosis")
            diagnosis_by_alert[alert.alert_id] = diagnosis

    missing_diagnoses = [alert.alert_id for alert in alert_values if alert.alert_id not in diagnosis_by_alert]
    if missing_diagnoses:
        raise ValueError("every alert requires a diagnosis")

    findings = tuple(
        alert_diagnosis_to_finding(alert, diagnosis_by_alert[alert.alert_id])
        for alert in sorted(alert_values, key=lambda item: item.alert_id)
    )
    profile = _profile_from_reference(reference, metadata_grade)
    report = DiagnosticReport(
        profile=profile,
        findings=findings,
        limitations=("monitor_alert_bridge",),
    )
    recommendations = build_recommendations(report, metadata_grade)
    return MonitoringRemediationPlan(
        reference_snapshot_id=reference.snapshot_id,
        diagnoses=tuple(
            sorted(
                diagnosis_values,
                key=lambda item: tuple(sorted(alert.alert_id for alert in item.alerts)),
            )
        ),
        report=report,
        recommendations=recommendations,
    )


def alert_diagnosis_to_finding(alert: Alert, diagnosis: Diagnosis) -> RiskFinding:
    """Convert one diagnosed alert into the existing privacy-safe finding contract."""

    if not any(item.alert_id == alert.alert_id for item in diagnosis.alerts):
        raise ValueError("diagnosis does not contain the alert")
    kind = _ALERT_KIND[alert.alert_type]
    metrics: dict[str, int | float] = {
        "alert_evidence_count": len(alert.evidence),
        "root_cause_count": len(diagnosis.root_causes),
    }
    for name, value in (
        ("reference_value", alert.reference_value),
        ("current_value", alert.current_value),
        ("delta", alert.delta),
    ):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            metrics[name] = value
    if diagnosis.root_causes:
        metrics["top_root_cause_contribution"] = max(
            float(cause.contribution) for cause in diagnosis.root_causes
        )
    digest = canonical_payload_hash({"alert_id": alert.alert_id})[:16]
    return RiskFinding(
        kind=kind,
        severity=(
            FindingSeverity.CRITICAL
            if alert.severity == "critical"
            else FindingSeverity.WARNING
        ),
        code=f"monitor_{alert.alert_type}_{alert.scope}_{digest}",
        feature=alert.scope_value if alert.scope == "feature" else None,
        metrics=metrics,
        limitations=("monitor_alert_bridge",),
    )


def create_remediation(recommendation: Recommendation) -> RemediationRecord:
    """Open a remediation record without executing the recommended action."""

    if not isinstance(recommendation, Recommendation):
        raise TypeError("recommendation must be a Recommendation")
    return RemediationRecord(
        recommendation_id=recommendation.recommendation_id,
        action_code=recommendation.action_code,
        finding_ids=recommendation.finding_ids,
    )


def verify_remediation(
    remediation: RemediationRecord,
    before: Sequence[RiskFinding],
    after: Sequence[RiskFinding],
    *,
    metric_directions: Mapping[str, AlertMetricDirection] | None = None,
    tolerance: float = 0.0,
) -> RemediationVerification:
    """Compare aggregate findings before/after a remediation using stable semantic keys."""

    if not isinstance(remediation, RemediationRecord):
        raise TypeError("remediation must be a RemediationRecord")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a non-negative finite number")
    directions = dict(_ALERT_RISK_METRICS)
    directions.update(metric_directions or {})
    if any(direction not in {"decrease", "increase", "toward"} for direction in directions.values()):
        raise ValueError("metric directions are invalid")
    before_values = _validate_findings(before)
    after_values = _validate_findings(after)
    selected = tuple(finding for finding in before_values if finding.finding_id in remediation.finding_ids)
    if not selected:
        raise ValueError("remediation findings are unavailable in before findings")
    after_by_key = {_finding_key(finding): finding for finding in after_values}
    if len(after_by_key) != len(after_values):
        raise ValueError("after findings must have unique semantic keys")

    resolved: list[str] = []
    remaining: list[str] = []
    inconclusive: list[str] = []
    metric_deltas: dict[str, float] = {}
    for finding in selected:
        current = after_by_key.get(_finding_key(finding))
        if current is None:
            resolved.append(finding.finding_id)
            continue
        common_metrics = set(finding.metrics) & set(current.metrics)
        applicable = []
        for metric in sorted(common_metrics):
            before_value = finding.metrics[metric]
            after_value = current.metrics[metric]
            if not isinstance(before_value, (int, float)) or not isinstance(after_value, (int, float)):
                continue
            delta = float(after_value) - float(before_value)
            metric_deltas[f"{finding.finding_id}:{metric}"] = delta
            if metric in directions:
                applicable.append(
                    _metric_improved(
                        float(before_value),
                        float(after_value),
                        directions[metric],
                        tolerance,
                    )
                )
        if applicable:
            if all(applicable):
                resolved.append(finding.finding_id)
            else:
                remaining.append(current.finding_id)
        elif _SEVERITY_RANK[current.severity] < _SEVERITY_RANK[finding.severity]:
            resolved.append(finding.finding_id)
        else:
            inconclusive.append(current.finding_id)

    status: VerificationStatus
    if remaining:
        status = "remaining"
    elif inconclusive:
        status = "inconclusive"
    else:
        status = "verified"
    return RemediationVerification(
        remediation_id=remediation.remediation_id,
        status=status,
        resolved_finding_ids=tuple(resolved),
        remaining_finding_ids=tuple(remaining),
        inconclusive_finding_ids=tuple(inconclusive),
        metric_deltas=metric_deltas,
    )


def apply_verification(
    remediation: RemediationRecord,
    verification: RemediationVerification,
) -> RemediationRecord:
    """Return the next immutable lifecycle state after a verification result."""

    if verification.remediation_id != remediation.remediation_id:
        raise ValueError("verification does not belong to remediation")
    status = {
        "verified": RemediationStatus.VERIFIED,
        "remaining": RemediationStatus.FAILED,
        "inconclusive": RemediationStatus.AWAITING_VERIFICATION,
    }[verification.status]
    return RemediationRecord(
        recommendation_id=remediation.recommendation_id,
        action_code=remediation.action_code,
        finding_ids=remediation.finding_ids,
        status=status,
        verification_id=verification.verification_id,
    )


def _profile_from_reference(
    reference: ReferenceSnapshot,
    metadata_grade: Literal["A", "B"],
) -> SafeProfile:
    counts = tuple(reference.segment_counts.values())
    return SafeProfile(
        dataset_id=reference.dataset_id,
        row_count=reference.row_count,
        feature_count=len(reference.features),
        positive_rate=reference.positive_rate,
        segment_count=len(counts),
        min_segment_size=min(counts) if counts else None,
        max_segment_size=max(counts) if counts else None,
        metadata_grade=metadata_grade,
    )


def _validate_findings(findings: Sequence[RiskFinding]) -> tuple[RiskFinding, ...]:
    values = tuple(findings)
    if any(not isinstance(finding, RiskFinding) for finding in values):
        raise TypeError("findings must contain RiskFinding values")
    keys = [_finding_key(finding) for finding in values]
    if len(keys) != len(set(keys)):
        raise ValueError("findings must have unique semantic keys")
    return values


def _finding_key(finding: RiskFinding) -> tuple[str, str, str, str, str]:
    return (
        finding.kind.value,
        finding.code,
        finding.feature or "",
        finding.period or "",
        finding.segment_token.token if finding.segment_token is not None else "",
    )


def _metric_improved(
    before: float,
    after: float,
    direction: AlertMetricDirection,
    tolerance: float,
) -> bool:
    if direction == "decrease":
        return after < before - tolerance
    if direction == "increase":
        return after > before + tolerance
    return abs(after - 1.0) < abs(before - 1.0) - tolerance


__all__ = [
    "AlertMetricDirection",
    "MonitoringRemediationPlan",
    "RemediationRecord",
    "RemediationStatus",
    "RemediationVerification",
    "VerificationStatus",
    "alert_diagnosis_to_finding",
    "apply_verification",
    "build_monitoring_remediation_plan",
    "create_remediation",
    "verify_remediation",
]
