"""Versioned allowlist for deterministic recommendation templates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from types import MappingProxyType

from riskprobe.monitoring.models import FindingKind, RiskFinding

RECOMMENDATION_POLICY_VERSION = "recommendation-policy-v1"


class ActionCode(StrEnum):
    """Only recommendation templates an untrusted proposal may select."""

    REMEDIATE_DATA_QUALITY = "remediate_data_quality"
    INVESTIGATE_FEATURE_DRIFT = "investigate_feature_drift"
    INVESTIGATE_POPULATION_SHIFT = "investigate_population_shift"
    INVESTIGATE_TARGET_SHIFT = "investigate_target_shift"
    REVIEW_SEGMENT_RISK = "review_segment_risk"
    MONITOR_TIME_STABILITY = "monitor_time_stability"
    REVIEW_RULE_EVIDENCE = "review_rule_evidence"


ACTION_TEMPLATE_BY_FINDING_KIND_V1: Mapping[
    FindingKind, tuple[ActionCode, str]
] = MappingProxyType(
    {
        FindingKind.DATA_QUALITY: (
            ActionCode.REMEDIATE_DATA_QUALITY,
            "data_quality_finding_present",
        ),
        FindingKind.FEATURE_DRIFT: (
            ActionCode.INVESTIGATE_FEATURE_DRIFT,
            "feature_drift_finding_present",
        ),
        FindingKind.POPULATION_SHIFT: (
            ActionCode.INVESTIGATE_POPULATION_SHIFT,
            "population_shift_finding_present",
        ),
        FindingKind.TARGET_SHIFT: (
            ActionCode.INVESTIGATE_TARGET_SHIFT,
            "target_shift_finding_present",
        ),
        FindingKind.SEGMENT_RISK: (
            ActionCode.REVIEW_SEGMENT_RISK,
            "segment_risk_finding_present",
        ),
        FindingKind.TIME_INSTABILITY: (
            ActionCode.MONITOR_TIME_STABILITY,
            "time_instability_finding_present",
        ),
        FindingKind.RULE_EVIDENCE: (
            ActionCode.REVIEW_RULE_EVIDENCE,
            "rule_evidence_finding_present",
        ),
    }
)
ALL_ACTION_CODES: tuple[ActionCode, ...] = tuple(
    sorted(ActionCode, key=lambda action: action.value)
)


def applicable_action_codes(findings: Iterable[RiskFinding]) -> tuple[ActionCode, ...]:
    """Return the canonical allowlisted actions applicable to these findings."""

    actions: set[ActionCode] = set()
    for finding in findings:
        if type(finding) is not RiskFinding:
            raise TypeError("findings must contain RiskFinding values")
        actions.add(ACTION_TEMPLATE_BY_FINDING_KIND_V1[finding.kind][0])
    return tuple(sorted(actions, key=lambda action: action.value))


__all__ = [
    "ACTION_TEMPLATE_BY_FINDING_KIND_V1",
    "ALL_ACTION_CODES",
    "ActionCode",
    "RECOMMENDATION_POLICY_VERSION",
    "applicable_action_codes",
]
