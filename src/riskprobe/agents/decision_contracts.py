"""Strict aggregate-only contracts for controlled local decision proposals."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from riskprobe.monitoring.models import RiskFinding
from riskprobe.privacy import canonical_payload_hash
from riskprobe.recommendations.policy import (
    ALL_ACTION_CODES,
    RECOMMENDATION_POLICY_VERSION,
    ActionCode,
    applicable_action_codes,
)

_PUBLIC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VERSION_KEY = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_VERSION_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}$")
_REQUIRED_COMPONENTS = frozenset(
    {"diagnostics", "orchestrator", "planner", "recommendations"}
)


class _StrictDTO(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )


class DecisionSource(StrEnum):
    DETERMINISTIC = "deterministic"
    EXTERNAL_HOST = "external_host"


class DecisionStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class DecisionReason(StrEnum):
    CONTEXT_MISMATCH = "context_mismatch"
    CONTEXT_EXPIRED = "context_expired"
    EVIDENCE_MISMATCH = "evidence_mismatch"
    ACTION_COUNT_INVALID = "action_count_invalid"
    ACTION_NOT_ALLOWED = "action_not_allowed"
    ACTION_NOT_APPLICABLE = "action_not_applicable"
    GRADE_B_ACTION_NOT_ALLOWED = "grade_b_action_not_allowed"


class DecisionFinding(_StrictDTO):
    evidence_id: str
    finding: RiskFinding

    @field_validator("evidence_id")
    @classmethod
    def validate_evidence_id(cls, value: str) -> str:
        return _validated_sha(value, "evidence_id")

    @model_validator(mode="after")
    def validate_privacy(self) -> DecisionFinding:
        canonical_payload_hash(self.model_dump(mode="json"))
        return self


class DecisionPolicy(_StrictDTO):
    schema_version: Literal["riskprobe.decision-policy.v1"] = (
        "riskprobe.decision-policy.v1"
    )
    policy_id: str = ""
    policy_version: str = RECOMMENDATION_POLICY_VERSION
    allowed_action_codes: tuple[ActionCode, ...] = ALL_ACTION_CODES
    grade_b_allowed_action_codes: tuple[ActionCode, ...] = ALL_ACTION_CODES
    min_action_count: int = Field(default=1, ge=1, le=len(ActionCode))
    max_action_count: int = Field(
        default=len(ActionCode), ge=1, le=len(ActionCode)
    )
    context_ttl_seconds: int = Field(default=600, ge=30, le=3600)
    require_complete_diagnosis_evidence: Literal[True] = True
    grade_b_analysis_only: Literal[True] = True

    @field_validator("policy_id")
    @classmethod
    def validate_policy_id(cls, value: str) -> str:
        return _validated_optional_sha(value, "policy_id")

    @field_validator("policy_version")
    @classmethod
    def validate_policy_version(cls, value: str) -> str:
        return _validated_version(value, "policy_version")

    @field_validator("allowed_action_codes", "grade_b_allowed_action_codes")
    @classmethod
    def normalize_actions(
        cls, value: tuple[ActionCode, ...]
    ) -> tuple[ActionCode, ...]:
        if len(value) != len(set(value)):
            raise ValueError("policy action codes must be unique")
        return tuple(sorted(value, key=lambda action: action.value))

    @model_validator(mode="after")
    def validate_and_derive_policy(self) -> DecisionPolicy:
        if not self.allowed_action_codes:
            raise ValueError("policy must allow at least one action")
        if not set(self.grade_b_allowed_action_codes).issubset(
            self.allowed_action_codes
        ):
            raise ValueError("Grade-B actions must be an allowed subset")
        if self.min_action_count > self.max_action_count or self.max_action_count > len(
            self.allowed_action_codes
        ):
            raise ValueError("policy action count bounds are invalid")
        return _derive_id(self, "policy_id")


class DecisionContext(_StrictDTO):
    schema_version: Literal["riskprobe.decision-context.v1"] = (
        "riskprobe.decision-context.v1"
    )
    context_id: str = ""
    privacy_class: Literal["aggregate"] = "aggregate"
    session_id: str
    attempt: int = Field(ge=0, le=1)
    anchor_node_id: str
    dataset_id: str
    objective: Literal["comprehensive"] = "comprehensive"
    metadata_grade: Literal["A", "B"]
    row_count: int = Field(ge=0)
    feature_count: int = Field(ge=0)
    issue_codes: tuple[str, ...] = ()
    rule_ids: tuple[str, ...] = ()
    diagnosis_evidence_ids: tuple[str, ...]
    findings: tuple[DecisionFinding, ...]
    policy: DecisionPolicy
    issued_at: datetime
    expires_at: datetime
    component_versions: Mapping[str, str]

    @field_validator("context_id")
    @classmethod
    def validate_context_id(cls, value: str) -> str:
        return _validated_optional_sha(value, "context_id")

    @field_validator("session_id", "dataset_id")
    @classmethod
    def validate_public_ids(cls, value: str, info: object) -> str:
        if _PUBLIC_ID.fullmatch(value) is None:
            raise ValueError(f"{getattr(info, 'field_name', 'identifier')} is invalid")
        return value

    @field_validator("anchor_node_id")
    @classmethod
    def validate_anchor_node_id(cls, value: str) -> str:
        return _validated_sha(value, "anchor_node_id")

    @field_validator("issue_codes", "rule_ids")
    @classmethod
    def normalize_public_codes(
        cls, value: tuple[str, ...], info: object
    ) -> tuple[str, ...]:
        if len(value) != len(set(value)) or any(
            _PUBLIC_ID.fullmatch(item) is None for item in value
        ):
            raise ValueError(
                f"{getattr(info, 'field_name', 'codes')} must be unique public identifiers"
            )
        return tuple(sorted(value))

    @field_validator("diagnosis_evidence_ids")
    @classmethod
    def normalize_diagnosis_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validated_sha_tuple(value, "diagnosis_evidence_ids", require=True)

    @field_validator("findings")
    @classmethod
    def normalize_findings(
        cls, value: tuple[DecisionFinding, ...]
    ) -> tuple[DecisionFinding, ...]:
        evidence_ids = tuple(item.evidence_id for item in value)
        finding_ids = tuple(item.finding.finding_id for item in value)
        if (
            not value
            or len(evidence_ids) != len(set(evidence_ids))
            or len(finding_ids) != len(set(finding_ids))
        ):
            raise ValueError("decision findings must be non-empty and unique")
        return tuple(sorted(value, key=lambda item: item.evidence_id))

    @field_validator("issued_at", "expires_at")
    @classmethod
    def normalize_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("decision timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("component_versions")
    @classmethod
    def normalize_component_versions(
        cls, value: Mapping[str, str]
    ) -> Mapping[str, str]:
        normalized = dict(value)
        if set(normalized) != _REQUIRED_COMPONENTS or any(
            _VERSION_KEY.fullmatch(key) is None
            or _VERSION_VALUE.fullmatch(component_version) is None
            for key, component_version in normalized.items()
        ):
            raise ValueError("component_versions are invalid")
        return MappingProxyType(dict(sorted(normalized.items())))

    @field_serializer("component_versions")
    def serialize_component_versions(
        self, value: Mapping[str, str]
    ) -> dict[str, str]:
        return dict(value)

    @model_validator(mode="after")
    def validate_and_derive_context(self) -> DecisionContext:
        if tuple(item.evidence_id for item in self.findings) != self.diagnosis_evidence_ids:
            raise ValueError("decision context must contain the complete diagnosis set")
        if not self.issued_at < self.expires_at:
            raise ValueError("decision context expiry is invalid")
        if (self.expires_at - self.issued_at).total_seconds() > self.policy.context_ttl_seconds:
            raise ValueError("decision context exceeds policy TTL")
        return _derive_id(self, "context_id")


class DecisionProposal(_StrictDTO):
    schema_version: Literal["riskprobe.decision-proposal.v1"] = (
        "riskprobe.decision-proposal.v1"
    )
    proposal_id: str = ""
    context_id: str
    diagnosis_evidence_ids: tuple[str, ...]
    action_codes: tuple[ActionCode, ...]
    source: DecisionSource
    source_version: str

    @field_validator("proposal_id")
    @classmethod
    def validate_proposal_id(cls, value: str) -> str:
        return _validated_optional_sha(value, "proposal_id")

    @field_validator("context_id")
    @classmethod
    def validate_context_id(cls, value: str) -> str:
        return _validated_sha(value, "context_id")

    @field_validator("diagnosis_evidence_ids")
    @classmethod
    def normalize_diagnosis_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validated_sha_tuple(value, "diagnosis_evidence_ids", require=False)

    @field_validator("action_codes")
    @classmethod
    def normalize_actions(
        cls, value: tuple[ActionCode, ...]
    ) -> tuple[ActionCode, ...]:
        if len(value) != len(set(value)):
            raise ValueError("proposal action codes must be unique")
        return tuple(sorted(value, key=lambda action: action.value))

    @field_validator("source_version")
    @classmethod
    def validate_source_version(cls, value: str) -> str:
        return _validated_version(value, "source_version")

    @model_validator(mode="after")
    def validate_and_derive_proposal(self) -> DecisionProposal:
        return _derive_id(self, "proposal_id")


class DecisionResult(_StrictDTO):
    schema_version: Literal["riskprobe.decision-result.v1"] = (
        "riskprobe.decision-result.v1"
    )
    decision_id: str = ""
    context_id: str
    proposal_id: str
    policy_id: str
    status: DecisionStatus
    reason_codes: tuple[DecisionReason, ...] = ()
    diagnosis_evidence_ids: tuple[str, ...]
    action_codes: tuple[ActionCode, ...] = ()
    source: DecisionSource
    source_version: str

    @field_validator("decision_id")
    @classmethod
    def validate_decision_id(cls, value: str) -> str:
        return _validated_optional_sha(value, "decision_id")

    @field_validator("context_id", "proposal_id", "policy_id")
    @classmethod
    def validate_hash_ids(cls, value: str, info: object) -> str:
        return _validated_sha(value, getattr(info, "field_name", "identifier"))

    @field_validator("reason_codes")
    @classmethod
    def normalize_reasons(
        cls, value: tuple[DecisionReason, ...]
    ) -> tuple[DecisionReason, ...]:
        order = {reason: index for index, reason in enumerate(DecisionReason)}
        return tuple(sorted(set(value), key=order.__getitem__))

    @field_validator("diagnosis_evidence_ids")
    @classmethod
    def normalize_diagnosis_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validated_sha_tuple(value, "diagnosis_evidence_ids", require=True)

    @field_validator("action_codes")
    @classmethod
    def normalize_actions(
        cls, value: tuple[ActionCode, ...]
    ) -> tuple[ActionCode, ...]:
        if len(value) != len(set(value)):
            raise ValueError("decision action codes must be unique")
        return tuple(sorted(value, key=lambda action: action.value))

    @field_validator("source_version")
    @classmethod
    def validate_source_version(cls, value: str) -> str:
        return _validated_version(value, "source_version")

    @model_validator(mode="after")
    def validate_and_derive_result(self) -> DecisionResult:
        if self.status is DecisionStatus.ACCEPTED:
            if self.reason_codes or not self.action_codes:
                raise ValueError("accepted decision must contain actions without reasons")
        elif not self.reason_codes or self.action_codes:
            raise ValueError("rejected decision must contain reasons without actions")
        return _derive_id(self, "decision_id")


class ProposalValidator:
    """Apply server-owned context, evidence, action, Grade-B, and expiry gates."""

    version = "proposal-validator-v1"

    def validate(
        self,
        context: DecisionContext,
        proposal: DecisionProposal,
        *,
        now: datetime,
    ) -> DecisionResult:
        if type(context) is not DecisionContext:
            raise TypeError("context must be a DecisionContext")
        if type(proposal) is not DecisionProposal:
            raise TypeError("proposal must be a DecisionProposal")
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        current_time = now.astimezone(UTC)
        reasons: list[DecisionReason] = []
        if proposal.context_id != context.context_id:
            reasons.append(DecisionReason.CONTEXT_MISMATCH)
        if current_time < context.issued_at or current_time >= context.expires_at:
            reasons.append(DecisionReason.CONTEXT_EXPIRED)
        if proposal.diagnosis_evidence_ids != context.diagnosis_evidence_ids:
            reasons.append(DecisionReason.EVIDENCE_MISMATCH)

        policy = context.policy
        actions = proposal.action_codes
        if not policy.min_action_count <= len(actions) <= policy.max_action_count:
            reasons.append(DecisionReason.ACTION_COUNT_INVALID)
        if not set(actions).issubset(policy.allowed_action_codes):
            reasons.append(DecisionReason.ACTION_NOT_ALLOWED)
        if context.metadata_grade == "B" and not set(actions).issubset(
            policy.grade_b_allowed_action_codes
        ):
            reasons.append(DecisionReason.GRADE_B_ACTION_NOT_ALLOWED)
        applicable = set(
            applicable_action_codes(item.finding for item in context.findings)
        )
        if not set(actions).issubset(applicable):
            reasons.append(DecisionReason.ACTION_NOT_APPLICABLE)

        return DecisionResult(
            context_id=context.context_id,
            proposal_id=proposal.proposal_id,
            policy_id=policy.policy_id,
            status=(
                DecisionStatus.REJECTED if reasons else DecisionStatus.ACCEPTED
            ),
            reason_codes=tuple(reasons),
            diagnosis_evidence_ids=context.diagnosis_evidence_ids,
            action_codes=() if reasons else actions,
            source=proposal.source,
            source_version=proposal.source_version,
        )


def default_decision_policy() -> DecisionPolicy:
    return DecisionPolicy()


def _validated_sha(value: str, field_name: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a SHA-256 identifier")
    return value


def _validated_optional_sha(value: str, field_name: str) -> str:
    if value:
        return _validated_sha(value, field_name)
    return value


def _validated_sha_tuple(
    value: tuple[str, ...], field_name: str, *, require: bool
) -> tuple[str, ...]:
    if (require and not value) or len(value) != len(set(value)) or any(
        _SHA256.fullmatch(item) is None for item in value
    ):
        raise ValueError(f"{field_name} must contain unique SHA-256 identifiers")
    return tuple(sorted(value))


def _validated_version(value: str, field_name: str) -> str:
    if _VERSION_VALUE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a public version token")
    return value


def _derive_id(model: BaseModel, field_name: str):
    payload = model.model_dump(mode="json", exclude={field_name})
    expected = canonical_payload_hash(payload)
    current = getattr(model, field_name)
    if current and current != expected:
        raise ValueError(f"{field_name} does not match canonical payload")
    object.__setattr__(model, field_name, expected)
    return model


__all__ = [
    "DecisionContext",
    "DecisionFinding",
    "DecisionPolicy",
    "DecisionProposal",
    "DecisionReason",
    "DecisionResult",
    "DecisionSource",
    "DecisionStatus",
    "ProposalValidator",
    "default_decision_policy",
]
