"""Evidence-linked deterministic recommendation engine."""

from riskprobe.recommendations.engine import build_recommendations
from riskprobe.recommendations.models import DecisionEligibility, Recommendation
from riskprobe.recommendations.policy import (
    ACTION_TEMPLATE_BY_FINDING_KIND_V1,
    ALL_ACTION_CODES,
    RECOMMENDATION_POLICY_VERSION,
    ActionCode,
    applicable_action_codes,
)

__all__ = [
    "ACTION_TEMPLATE_BY_FINDING_KIND_V1",
    "ALL_ACTION_CODES",
    "ActionCode",
    "DecisionEligibility",
    "RECOMMENDATION_POLICY_VERSION",
    "Recommendation",
    "applicable_action_codes",
    "build_recommendations",
]
