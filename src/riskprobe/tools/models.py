"""Strict path-free request and aggregate-only response DTOs for RiskProbe tools."""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


_DATASET_ID = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")
_PUBLIC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class _StrictDTO(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )


class _DatasetRequest(_StrictDTO):
    dataset_id: str

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, value: str) -> str:
        if _DATASET_ID.fullmatch(value) is None:
            raise ValueError("dataset_id must be a registered identifier")
        return value


class _RunRequest(_StrictDTO):
    run_id: str

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        return _validated_public_id(value, "run_id")


class InspectRequest(_DatasetRequest):
    pass


class DiscoverRequest(_DatasetRequest):
    pass


class DiagnoseRequest(_DatasetRequest):
    pass


class RecommendRequest(_DatasetRequest):
    evidence_ids: tuple[str, ...] = ()

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or any(_SHA256.fullmatch(item) is None for item in value):
            raise ValueError("evidence_ids must contain unique SHA-256 identifiers")
        return value


class _ControlledRecommendRequest(RecommendRequest):
    """Private server-side binding; intentionally absent from the public union."""

    decision_result_evidence_id: str

    @field_validator("decision_result_evidence_id")
    @classmethod
    def validate_decision_result_evidence_id(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("decision_result_evidence_id must be a SHA-256 identifier")
        return value


class RunRequest(_DatasetRequest):
    pass


class StatusRequest(_RunRequest):
    pass


class TraceRequest(_RunRequest):
    pass


class EvidenceLookupRequest(_StrictDTO):
    evidence_id: str

    @field_validator("evidence_id")
    @classmethod
    def validate_evidence_id(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("evidence_id must be a SHA-256 identifier")
        return value


class _DatasetResponse(_StrictDTO):
    dataset_id: str

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, value: str) -> str:
        if _DATASET_ID.fullmatch(value) is None:
            raise ValueError("dataset_id must be a registered identifier")
        return value


class InspectResponse(_DatasetResponse):
    row_count: int = Field(ge=0)
    feature_count: int = Field(ge=0)
    metadata_grade: Literal["A", "B"]
    issue_codes: tuple[str, ...] = ()

    @field_validator("issue_codes")
    @classmethod
    def validate_issue_codes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validated_ids(value, "issue_codes")


class DiscoverResponse(_DatasetResponse):
    rule_ids: tuple[str, ...]

    @field_validator("rule_ids")
    @classmethod
    def validate_rule_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validated_ids(value, "rule_ids")

    @property
    def candidate_rule_count(self) -> int:
        return len(self.rule_ids)


class DiagnoseResponse(_DatasetResponse):
    finding_ids: tuple[str, ...]

    @field_validator("finding_ids")
    @classmethod
    def validate_finding_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validated_ids(value, "finding_ids")

    @property
    def finding_count(self) -> int:
        return len(self.finding_ids)


class RecommendResponse(_DatasetResponse):
    recommendation_ids: tuple[str, ...]

    @field_validator("recommendation_ids")
    @classmethod
    def validate_recommendation_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validated_ids(value, "recommendation_ids")

    @property
    def recommendation_count(self) -> int:
        return len(self.recommendation_ids)


class RunResponse(_DatasetResponse):
    run_id: str
    reused: bool
    metadata_grade: Literal["A", "B"] | None = None
    artifact_count: int = Field(default=6, ge=0)

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        return _validated_public_id(value, "run_id")


ToolStatus: TypeAlias = Literal["pending", "running", "succeeded", "failed", "cancelled"]


class StatusResponse(_StrictDTO):
    run_id: str
    status: ToolStatus

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        return _validated_public_id(value, "run_id")


class TraceEvent(_StrictDTO):
    sequence: int = Field(ge=1)
    node_id: str
    event_type: str
    status: ToolStatus
    attempt: int = Field(ge=1)
    error_class: str | None = None

    @field_validator("node_id", "event_type", "error_class")
    @classmethod
    def validate_event_token(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        return _validated_public_id(value, getattr(info, "field_name", "event token"))


class TraceResponse(_StrictDTO):
    run_id: str
    events: tuple[TraceEvent, ...]

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        return _validated_public_id(value, "run_id")

    @property
    def event_count(self) -> int:
        return len(self.events)


class EvidenceLookupResponse(_StrictDTO):
    evidence_id: str
    run_id: str
    kind: str
    payload: Mapping[str, object]
    parent_ids: tuple[str, ...] = ()
    artifact_hashes: Mapping[str, str] = Field(default_factory=dict)
    privacy_class: Literal["aggregate"] = "aggregate"
    producer_version: str | None = None

    @field_validator("evidence_id")
    @classmethod
    def validate_evidence_id(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("evidence_id must be a SHA-256 identifier")
        return value

    @field_validator("run_id", "kind")
    @classmethod
    def validate_public_fields(cls, value: str, info: object) -> str:
        return _validated_public_id(value, getattr(info, "field_name", "identifier"))

    @field_validator("payload")
    @classmethod
    def copy_payload(cls, value: Mapping[str, object]) -> Mapping[str, object]:
        return MappingProxyType(dict(value))

    @field_validator("parent_ids")
    @classmethod
    def validate_parent_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or any(_SHA256.fullmatch(item) is None for item in value):
            raise ValueError("parent_ids must contain unique SHA-256 identifiers")
        return value

    @field_validator("artifact_hashes")
    @classmethod
    def validate_artifact_hashes(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        normalized = dict(value)
        for name, digest in normalized.items():
            if (
                not name
                or name in {".", ".."}
                or "/" in name
                or "\\" in name
                or _SHA256.fullmatch(digest) is None
            ):
                raise ValueError("artifact_hashes must contain public names and SHA-256 values")
        return MappingProxyType(normalized)

    @field_serializer("payload", "artifact_hashes")
    def serialize_mappings(self, value: Mapping[str, object]) -> dict[str, object]:
        return dict(value)

    @field_validator("producer_version")
    @classmethod
    def validate_producer_version(cls, value: str | None) -> str | None:
        if value is not None and (
            not value or len(value) > 128 or any(character.isspace() for character in value)
        ):
            raise ValueError("producer_version must be a version token")
        return value


ToolRequest: TypeAlias = (
    InspectRequest
    | DiscoverRequest
    | DiagnoseRequest
    | RecommendRequest
    | RunRequest
    | StatusRequest
    | TraceRequest
    | EvidenceLookupRequest
)
ToolResponse: TypeAlias = (
    InspectResponse
    | DiscoverResponse
    | DiagnoseResponse
    | RecommendResponse
    | RunResponse
    | StatusResponse
    | TraceResponse
    | EvidenceLookupResponse
)
EvidenceRequest = EvidenceLookupRequest
EvidenceResponse = EvidenceLookupResponse


def _validated_public_id(value: str, field_name: str) -> str:
    if _PUBLIC_ID.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a public identifier")
    return value


def _validated_ids(value: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if field_name in {"finding_ids", "recommendation_ids"}:
        if len(value) != len(set(value)) or any(_SHA256.fullmatch(item) is None for item in value):
            raise ValueError(f"{field_name} must contain unique lowercase SHA-256 identifiers")
        return value
    if len(value) != len(set(value)) or any(_PUBLIC_ID.fullmatch(item) is None for item in value):
        raise ValueError(f"{field_name} must contain unique public identifiers")
    return value
