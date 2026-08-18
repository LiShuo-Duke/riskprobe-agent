"""Strict, path-free contracts for deterministic agent execution."""

from __future__ import annotations

import re
from collections.abc import Mapping
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

from riskprobe.privacy import assert_safe_payload
from riskprobe.tools.models import (
    DiagnoseRequest,
    DiscoverRequest,
    InspectRequest,
    RecommendRequest,
    RunRequest,
    ToolRequest,
)

_PUBLIC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CODE = re.compile(r"^[a-z][a-z0-9_-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VERSION_KEY = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_VERSION_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}$")
_REQUEST_TYPES: dict[str, type[ToolRequest]] = {
    "diagnose": DiagnoseRequest,
    "discover": DiscoverRequest,
    "inspect": InspectRequest,
    "recommend": RecommendRequest,
    "run": RunRequest,
}


class AgentState(StrEnum):
    PLANNING = "planning"
    EXECUTING = "executing"
    COLLECTING_EVIDENCE = "collecting_evidence"
    REVIEWING = "reviewing"
    RETRYING = "retrying"
    COMPLETED = "completed"
    REJECTED = "rejected"


class AgentStatus(StrEnum):
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"


class ReviewReason(StrEnum):
    MISSING_EVIDENCE = "missing_evidence"
    PERMISSION_DENIED = "permission_denied"
    UNSAFE_PAYLOAD = "unsafe_payload"
    GRADE_B_PRODUCTION_ACTION = "grade_b_production_action"
    RETRY_LIMIT_EXCEEDED = "retry_limit_exceeded"
    MISSING_DIAGNOSIS = "missing_diagnosis"
    EVIDENCE_MISMATCH = "evidence_mismatch"
    TOOL_FAILURE = "tool_failure"


class _StrictDTO(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )


class PlanStep(_StrictDTO):
    """One typed tool selection, or the terminal deterministic review step."""

    step_id: str
    tool_name: str
    request: ToolRequest | None = None
    requires_evidence: bool = False
    production_action: bool = False

    @field_validator("step_id", "tool_name")
    @classmethod
    def validate_codes(cls, value: str) -> str:
        if _CODE.fullmatch(value) is None:
            raise ValueError("plan step identifiers must be public codes")
        return value

    @model_validator(mode="after")
    def validate_typed_request(self) -> PlanStep:
        if self.tool_name == "review":
            if self.request is not None or self.production_action:
                raise ValueError("review step cannot contain a tool request or production action")
            return self
        expected = _REQUEST_TYPES.get(self.tool_name)
        if expected is None or type(self.request) is not expected:
            raise ValueError("tool name must match its strict typed request")
        assert_safe_payload(self.request.model_dump(mode="json"))
        return self


class ExecutionPlan(_StrictDTO):
    """Deterministic, immutable plan containing typed allowlisted requests only."""

    objective: str
    dataset_id: str
    steps: tuple[PlanStep, ...]
    component_versions: Mapping[str, str] = Field(default_factory=dict)

    @field_validator("objective")
    @classmethod
    def validate_objective(cls, value: str) -> str:
        if _CODE.fullmatch(value) is None:
            raise ValueError("objective must be a safe public code")
        return value

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, value: str) -> str:
        if _PUBLIC_ID.fullmatch(value) is None:
            raise ValueError("dataset_id must be a public identifier")
        return value

    @field_validator("steps")
    @classmethod
    def validate_steps(cls, value: tuple[PlanStep, ...]) -> tuple[PlanStep, ...]:
        if not value or value[-1].tool_name != "review":
            raise ValueError("execution plan must end with review")
        if len({step.step_id for step in value}) != len(value):
            raise ValueError("plan step IDs must be unique")
        if any(step.tool_name == "review" for step in value[:-1]):
            raise ValueError("review must be the terminal plan step")
        return value

    @field_validator("component_versions")
    @classmethod
    def validate_component_versions(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        normalized = dict(value)
        if any(
            _VERSION_KEY.fullmatch(key) is None or _VERSION_VALUE.fullmatch(version) is None
            for key, version in normalized.items()
        ):
            raise ValueError("component versions must contain public version tokens")
        return MappingProxyType(dict(sorted(normalized.items())))

    @field_serializer("component_versions")
    def serialize_component_versions(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)

    @model_validator(mode="after")
    def validate_privacy(self) -> ExecutionPlan:
        assert_safe_payload(self.model_dump(mode="json"))
        return self

    @property
    def tool_sequence(self) -> tuple[str, ...]:
        return tuple(step.tool_name for step in self.steps)


class ReviewDecision(_StrictDTO):
    """Deterministic review outcome with machine-readable denial reasons."""

    approved: bool
    reason_codes: tuple[ReviewReason, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    retry_allowed: bool = False

    @field_validator("reason_codes")
    @classmethod
    def normalize_reasons(cls, value: tuple[ReviewReason, ...]) -> tuple[ReviewReason, ...]:
        order = {reason: index for index, reason in enumerate(ReviewReason)}
        return tuple(sorted(set(value), key=order.__getitem__))

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or any(_SHA256.fullmatch(item) is None for item in value):
            raise ValueError("evidence IDs must be unique SHA-256 identifiers")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def validate_decision(self) -> ReviewDecision:
        if self.approved and (self.reason_codes or self.retry_allowed):
            raise ValueError("approved review cannot contain denial reasons or retry")
        if not self.approved and not self.reason_codes:
            raise ValueError("rejected review must contain a reason")
        return self


class AgentResult(_StrictDTO):
    """Safe terminal projection of an agent run; tool payloads are intentionally absent."""

    session_id: str
    status: AgentStatus
    plan: ExecutionPlan
    review: ReviewDecision
    tool_sequence: tuple[str, ...]
    evidence_ids: tuple[str, ...] = ()
    diagnosis_evidence_ids: tuple[str, ...] = ()
    retry_count: int = Field(ge=0, le=1)
    state_history: tuple[AgentState, ...]
    leaf_node_id: str
    redacted_summary: str

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        if _PUBLIC_ID.fullmatch(value) is None:
            raise ValueError("session_id must be a public identifier")
        return value

    @field_validator("tool_sequence")
    @classmethod
    def validate_tool_sequence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(_CODE.fullmatch(item) is None for item in value):
            raise ValueError("tool sequence must contain public tool codes")
        return value

    @field_validator("evidence_ids", "diagnosis_evidence_ids")
    @classmethod
    def validate_result_evidence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or any(_SHA256.fullmatch(item) is None for item in value):
            raise ValueError("result evidence IDs must be unique SHA-256 identifiers")
        return tuple(sorted(value))

    @field_validator("leaf_node_id")
    @classmethod
    def validate_leaf_node_id(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("leaf_node_id must be a SHA-256 identifier")
        return value

    @field_validator("redacted_summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        if not value or len(value) > 1_024:
            raise ValueError("redacted summary must be bounded and non-empty")
        return value

    @model_validator(mode="after")
    def validate_result(self) -> AgentResult:
        if (self.status is AgentStatus.SUCCEEDED) != self.review.approved:
            raise ValueError("agent status must agree with review decision")
        if not set(self.diagnosis_evidence_ids).issubset(self.evidence_ids):
            raise ValueError("diagnosis evidence must be included in result evidence")
        assert_safe_payload(self.model_dump(mode="json"))
        return self


MetadataGrade = Literal["A", "B"]

__all__ = [
    "AgentResult",
    "AgentState",
    "AgentStatus",
    "ExecutionPlan",
    "MetadataGrade",
    "PlanStep",
    "ReviewDecision",
    "ReviewReason",
]
