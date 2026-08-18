"""Strict frozen DTOs for deterministic offline agent evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
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

from riskprobe.privacy import assert_safe_payload, canonical_payload_hash

DEFAULT_EVAL_SEED = 42
_PUBLIC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CODE = re.compile(r"^[a-z][a-z0-9_-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_METRIC_NAMES = (
    "task_success",
    "tool_sequence",
    "evidence_completeness",
    "policy_compliance",
    "privacy_compliance",
    "replay_determinism",
)


class _StrictDTO(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )


class EvalCase(_StrictDTO):
    case_id: str
    objective: str
    expected_tool_sequence: tuple[str, ...]
    required_evidence_ids: tuple[str, ...] = ()
    require_diagnosis: bool = True

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str) -> str:
        if _PUBLIC_ID.fullmatch(value) is None:
            raise ValueError("case_id must be a public identifier")
        return value

    @field_validator("objective")
    @classmethod
    def validate_objective(cls, value: str) -> str:
        if _CODE.fullmatch(value) is None:
            raise ValueError("objective must be a safe public code")
        return value

    @field_validator("expected_tool_sequence")
    @classmethod
    def validate_sequence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(_CODE.fullmatch(item) is None for item in value):
            raise ValueError("expected tool sequence must contain public tool codes")
        return value

    @field_validator("required_evidence_ids")
    @classmethod
    def validate_required_evidence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or any(_SHA256.fullmatch(item) is None for item in value):
            raise ValueError("required evidence must contain unique SHA-256 IDs")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def validate_privacy(self) -> EvalCase:
        assert_safe_payload(self.model_dump(mode="json"))
        return self


class EvalSuite(_StrictDTO):
    suite_id: str
    seed: int = Field(default=DEFAULT_EVAL_SEED, ge=0)
    cases: tuple[EvalCase, ...]
    frozen: Literal[True] = True
    suite_hash: str = ""

    @field_validator("suite_id")
    @classmethod
    def validate_suite_id(cls, value: str) -> str:
        if _PUBLIC_ID.fullmatch(value) is None:
            raise ValueError("suite_id must be a public identifier")
        return value

    @field_validator("cases")
    @classmethod
    def validate_cases(cls, value: tuple[EvalCase, ...]) -> tuple[EvalCase, ...]:
        if not value or len({case.case_id for case in value}) != len(value):
            raise ValueError("eval suite requires unique non-empty cases")
        return value

    @field_validator("suite_hash")
    @classmethod
    def validate_suite_hash(cls, value: str) -> str:
        if value and _SHA256.fullmatch(value) is None:
            raise ValueError("suite_hash must be a SHA-256 identifier")
        return value

    @model_validator(mode="after")
    def derive_suite_hash(self) -> EvalSuite:
        payload = self.model_dump(mode="json", exclude={"suite_hash"})
        expected = canonical_payload_hash(payload)
        if self.suite_hash and self.suite_hash != expected:
            raise ValueError("suite_hash does not match frozen suite")
        object.__setattr__(self, "suite_hash", expected)
        return self

    def verify_integrity(self) -> bool:
        payload = self.model_dump(mode="json", exclude={"suite_hash"})
        return self.frozen is True and canonical_payload_hash(payload) == self.suite_hash


class EvalObservation(_StrictDTO):
    """Safe semantic projection returned by an evaluated candidate."""

    case_id: str
    task_succeeded: bool
    tool_sequence: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    diagnosis_evidence_ids: tuple[str, ...]
    policy_violations: int = Field(ge=0)
    privacy_violations: int = Field(ge=0)

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str) -> str:
        if _PUBLIC_ID.fullmatch(value) is None:
            raise ValueError("case_id must be a public identifier")
        return value

    @field_validator("tool_sequence")
    @classmethod
    def validate_tool_sequence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(_CODE.fullmatch(item) is None for item in value):
            raise ValueError("tool sequence must contain public tool codes")
        return value

    @field_validator("evidence_ids", "diagnosis_evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or any(_SHA256.fullmatch(item) is None for item in value):
            raise ValueError("observation evidence must contain unique SHA-256 IDs")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def validate_observation(self) -> EvalObservation:
        if not set(self.diagnosis_evidence_ids).issubset(self.evidence_ids):
            raise ValueError("diagnosis evidence must be included in evidence IDs")
        assert_safe_payload(self.model_dump(mode="json"))
        return self


class EvalCaseResult(_StrictDTO):
    case_id: str
    task_succeeded: bool
    tool_sequence_matched: bool
    evidence_complete: bool
    policy_compliant: bool
    privacy_compliant: bool
    replay_deterministic: bool
    passed: bool

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str) -> str:
        if _PUBLIC_ID.fullmatch(value) is None:
            raise ValueError("case_id must be a public identifier")
        return value

    @model_validator(mode="after")
    def validate_passed(self) -> EvalCaseResult:
        expected = all(
            (
                self.task_succeeded,
                self.tool_sequence_matched,
                self.evidence_complete,
                self.policy_compliant,
                self.privacy_compliant,
                self.replay_deterministic,
            )
        )
        if self.passed != expected:
            raise ValueError("case pass flag must equal all metric gates")
        return self


class EvalMetrics(_StrictDTO):
    case_count: int = Field(ge=1)
    task_success: float = Field(ge=0, le=1)
    tool_sequence: float = Field(ge=0, le=1)
    evidence_completeness: float = Field(ge=0, le=1)
    policy_compliance: float = Field(ge=0, le=1)
    privacy_compliance: float = Field(ge=0, le=1)
    replay_determinism: float = Field(ge=0, le=1)
    overall: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_overall(self) -> EvalMetrics:
        expected = sum(getattr(self, name) for name in _METRIC_NAMES) / len(_METRIC_NAMES)
        if abs(self.overall - expected) > 1e-12:
            raise ValueError("overall metric must be the mean of all gates")
        return self


class EvalReport(_StrictDTO):
    suite_id: str
    suite_hash: str
    seed: int = Field(ge=0)
    candidate_version: str
    case_results: tuple[EvalCaseResult, ...]
    metrics: EvalMetrics
    passed: bool
    frozen: Literal[True] = True
    report_hash: str = ""

    @field_validator("suite_id", "candidate_version")
    @classmethod
    def validate_public_ids(cls, value: str) -> str:
        if _PUBLIC_ID.fullmatch(value) is None:
            raise ValueError("report identifiers must be public")
        return value

    @field_validator("suite_hash", "report_hash")
    @classmethod
    def validate_hashes(cls, value: str) -> str:
        if value and _SHA256.fullmatch(value) is None:
            raise ValueError("report hashes must be SHA-256 identifiers")
        return value

    @model_validator(mode="after")
    def derive_report_hash(self) -> EvalReport:
        if not self.case_results or self.metrics.case_count != len(self.case_results):
            raise ValueError("report metric count must match case results")
        expected_metrics = _metrics_from_results(self.case_results)
        if any(
            abs(getattr(self.metrics, name) - expected_metrics[name]) > 1e-12
            for name in _METRIC_NAMES
        ):
            raise ValueError("report metrics must match frozen case results")
        if abs(self.metrics.overall - expected_metrics["overall"]) > 1e-12:
            raise ValueError("report overall metric must match frozen case results")
        if self.passed != all(result.passed for result in self.case_results):
            raise ValueError("report pass flag must equal all case gates")
        payload = self.model_dump(mode="json", exclude={"report_hash"})
        expected = _report_payload_hash(payload)
        if self.report_hash and self.report_hash != expected:
            raise ValueError("report_hash does not match frozen report")
        object.__setattr__(self, "report_hash", expected)
        return self

    def verify_integrity(self) -> bool:
        payload = self.model_dump(mode="json", exclude={"report_hash"})
        return self.frozen is True and _report_payload_hash(payload) == self.report_hash


class EvalComparison(_StrictDTO):
    baseline_version: str
    candidate_version: str
    baseline_report_hash: str
    candidate_report_hash: str
    compatible: bool
    candidate_passed: bool
    deltas: Mapping[str, float]
    regressed_metrics: tuple[str, ...] = ()

    @field_validator("baseline_version", "candidate_version")
    @classmethod
    def validate_versions(cls, value: str) -> str:
        if _PUBLIC_ID.fullmatch(value) is None:
            raise ValueError("comparison versions must be public identifiers")
        return value

    @field_validator("baseline_report_hash", "candidate_report_hash")
    @classmethod
    def validate_report_hashes(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("comparison report hashes must be SHA-256 identifiers")
        return value

    @field_validator("deltas")
    @classmethod
    def validate_deltas(cls, value: Mapping[str, float]) -> Mapping[str, float]:
        normalized = dict(value)
        if set(normalized) != set(_METRIC_NAMES):
            raise ValueError("comparison must contain every eval metric")
        return MappingProxyType(dict(sorted(normalized.items())))

    @field_serializer("deltas")
    def serialize_deltas(self, value: Mapping[str, float]) -> dict[str, float]:
        return dict(value)

    @field_validator("regressed_metrics")
    @classmethod
    def validate_regressions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(item not in _METRIC_NAMES for item in value):
            raise ValueError("regressions must name known metrics")
        return tuple(sorted(set(value)))


def _metrics_from_results(results: tuple[EvalCaseResult, ...]) -> dict[str, float]:
    count = len(results)
    values = {
        "task_success": sum(result.task_succeeded for result in results) / count,
        "tool_sequence": sum(result.tool_sequence_matched for result in results) / count,
        "evidence_completeness": sum(result.evidence_complete for result in results) / count,
        "policy_compliance": sum(result.policy_compliant for result in results) / count,
        "privacy_compliance": sum(result.privacy_compliant for result in results) / count,
        "replay_determinism": sum(result.replay_deterministic for result in results) / count,
    }
    values["overall"] = sum(values.values()) / len(values)
    return values


def _report_payload_hash(payload: Mapping[str, object]) -> str:
    privacy_projection = dict(payload)
    privacy_projection["findings"] = privacy_projection.pop("case_results", ())
    assert_safe_payload(privacy_projection)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "DEFAULT_EVAL_SEED",
    "EvalCase",
    "EvalCaseResult",
    "EvalComparison",
    "EvalMetrics",
    "EvalObservation",
    "EvalReport",
    "EvalSuite",
]
