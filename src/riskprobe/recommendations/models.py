"""Strict evidence-linked recommendation contracts."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from riskprobe.privacy import canonical_payload_hash

_CODE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DecisionEligibility(StrEnum):
    """Deterministic upper bound on how a recommendation may be used."""

    ANALYSIS_ONLY = "analysis_only"
    HUMAN_REVIEW_REQUIRED = "human_review_required"


RecommendationPriority = Literal["low", "medium", "high", "critical"]


class Recommendation(BaseModel):
    """A human-gated action whose complete evidence is a set of finding IDs."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )

    recommendation_id: str = ""
    action_code: str
    priority: RecommendationPriority
    finding_ids: tuple[str, ...]
    rationale_code: str
    human_approval_required: Literal[True] = True
    decision_eligibility: DecisionEligibility

    @field_validator("recommendation_id")
    @classmethod
    def validate_recommendation_id(cls, value: str) -> str:
        if value and _SHA256.fullmatch(value) is None:
            raise ValueError("recommendation_id must be a SHA-256 identifier")
        return value

    @field_validator("action_code", "rationale_code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        if _CODE.fullmatch(value) is None:
            raise ValueError("recommendation codes must be lower snake case")
        return value

    @field_validator("finding_ids")
    @classmethod
    def validate_finding_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("finding_ids must contain evidence")
        if len(value) != len(set(value)):
            raise ValueError("finding_ids must be unique")
        if any(_SHA256.fullmatch(item) is None for item in value):
            raise ValueError("finding_ids must contain SHA-256 identifiers")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def derive_and_validate_recommendation_id(self) -> Recommendation:
        payload = self.model_dump(mode="json", exclude={"recommendation_id"})
        expected = canonical_payload_hash(payload)
        if self.recommendation_id and self.recommendation_id != expected:
            raise ValueError("recommendation_id does not match canonical payload")
        object.__setattr__(self, "recommendation_id", expected)
        return self


__all__ = ["DecisionEligibility", "Recommendation", "RecommendationPriority"]
