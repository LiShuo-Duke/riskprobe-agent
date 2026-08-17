"""Deterministic recommendation templates linked to diagnostic findings."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

from riskprobe.monitoring.models import (
    DiagnosticReport,
    FindingKind,
    FindingSeverity,
    RiskFinding,
)
from riskprobe.recommendations.models import DecisionEligibility, Recommendation
from riskprobe.recommendations.policy import ACTION_TEMPLATE_BY_FINDING_KIND_V1

_PRIORITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def build_recommendations(
    report: DiagnosticReport,
    metadata_grade: Literal["A", "B"],
) -> tuple[Recommendation, ...]:
    """Build human-gated recommendations and reject metadata-grade bypasses."""

    if metadata_grade != report.metadata_grade:
        raise ValueError("metadata grade must match diagnostic report")
    eligibility = (
        DecisionEligibility.ANALYSIS_ONLY
        if metadata_grade == "B"
        else DecisionEligibility.HUMAN_REVIEW_REQUIRED
    )
    grouped: dict[FindingKind, list[RiskFinding]] = defaultdict(list)
    for finding in report.findings:
        grouped[finding.kind].append(finding)

    recommendations: list[Recommendation] = []
    report_finding_ids = {finding.finding_id for finding in report.findings}
    for kind in sorted(grouped, key=lambda item: item.value):
        action_code, rationale_code = ACTION_TEMPLATE_BY_FINDING_KIND_V1[kind]
        findings = grouped[kind]
        finding_ids = tuple(sorted(finding.finding_id for finding in findings))
        if not finding_ids or any(item not in report_finding_ids for item in finding_ids):
            raise ValueError("recommendation evidence is unavailable")
        recommendations.append(
            Recommendation(
                action_code=action_code,
                priority=_priority(findings),
                finding_ids=finding_ids,
                rationale_code=rationale_code,
                human_approval_required=True,
                decision_eligibility=eligibility,
            )
        )
    return tuple(
        sorted(
            recommendations,
            key=lambda item: (
                _PRIORITY_RANK[item.priority],
                item.action_code,
                item.finding_ids,
                item.recommendation_id,
            ),
        )
    )


def _priority(findings: list[RiskFinding]) -> Literal["medium", "high", "critical"]:
    severities = {finding.severity for finding in findings}
    if FindingSeverity.CRITICAL in severities:
        return "critical"
    if FindingSeverity.WARNING in severities:
        return "high"
    return "medium"


__all__ = ["build_recommendations"]
