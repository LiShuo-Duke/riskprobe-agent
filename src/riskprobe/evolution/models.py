"""Immutable hash-only models for gated offline evolution."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)

from riskprobe.evals import EvalComparison, EvalReport
from riskprobe.privacy import canonical_payload_hash

_PUBLIC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$")
_REASON = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ContentKind(StrEnum):
    PROMPT = "prompt"
    TEMPLATE = "template"
    POLICY = "policy"
    CONFIG = "config"


class _StrictDTO(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )


class EvaluationGate(_StrictDTO):
    suite_id: str
    suite_hash: str
    report_hash: str
    seed: int = Field(ge=0)
    passed: bool
    frozen: Literal[True] = True
    regressed_metrics: tuple[str, ...] = ()

    @field_validator("suite_id")
    @classmethod
    def validate_suite_id(cls, value: str) -> str:
        if _PUBLIC_ID.fullmatch(value) is None:
            raise ValueError("suite_id must be a public identifier")
        return value

    @field_validator("suite_hash", "report_hash")
    @classmethod
    def validate_hashes(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("evaluation gate hashes must be SHA-256 identifiers")
        return value

    @field_validator("regressed_metrics")
    @classmethod
    def validate_regressions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(_REASON.fullmatch(item) is None for item in value):
            raise ValueError("regressed metrics must be public metric names")
        return tuple(sorted(set(value)))

    @classmethod
    def from_report(
        cls,
        report: EvalReport,
        comparison: EvalComparison | None = None,
    ) -> EvaluationGate:
        if type(report) is not EvalReport:
            raise TypeError("eval report must be an EvalReport v1 instance")
        if not report.verify_integrity():
            raise ValueError("eval report is not a frozen integrity-checked report")
        regressions: tuple[str, ...] = ()
        comparison_passed = True
        if comparison is not None:
            if type(comparison) is not EvalComparison:
                raise TypeError("comparison must be an EvalComparison instance")
            if comparison.candidate_report_hash != report.report_hash:
                raise ValueError("comparison does not reference the candidate report")
            if comparison.candidate_version != report.candidate_version:
                raise ValueError("comparison does not reference the candidate version")
            regressions = comparison.regressed_metrics
            comparison_passed = comparison.compatible and comparison.candidate_passed
        return cls(
            suite_id=report.suite_id,
            suite_hash=report.suite_hash,
            report_hash=report.report_hash,
            seed=report.seed,
            passed=report.passed and comparison_passed and not regressions,
            regressed_metrics=regressions,
        )


class CandidateVersion(_StrictDTO):
    """An immutable version containing hashes only, never prompt/config bodies."""

    version_id: str = ""
    content_hashes: Mapping[ContentKind, str]
    eval_gate: EvaluationGate
    parent_version_id: str | None = None

    @field_validator("version_id", "parent_version_id")
    @classmethod
    def validate_version_hash(cls, value: str | None) -> str | None:
        if value is not None and value != "" and _SHA256.fullmatch(value) is None:
            raise ValueError("version identifiers must be SHA-256 identifiers")
        return value

    @field_validator("content_hashes", mode="before")
    @classmethod
    def normalize_content_keys(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized: dict[ContentKind, object] = {}
        for key, digest in value.items():
            if isinstance(key, ContentKind):
                kind = key
            elif isinstance(key, str):
                try:
                    kind = ContentKind(key)
                except ValueError as error:
                    raise ValueError("only prompt/template/policy/config hashes are allowed") from error
            else:
                raise ValueError("content hash keys must be allowlisted kinds")
            normalized[kind] = digest
        return normalized

    @field_validator("content_hashes")
    @classmethod
    def validate_content_hashes(
        cls,
        value: Mapping[ContentKind, str],
    ) -> Mapping[ContentKind, str]:
        normalized = dict(value)
        if not normalized or any(_SHA256.fullmatch(digest) is None for digest in normalized.values()):
            raise ValueError("content versions must contain SHA-256 hashes only")
        return MappingProxyType(dict(sorted(normalized.items(), key=lambda item: item[0].value)))

    @field_serializer("content_hashes")
    def serialize_content_hashes(
        self,
        value: Mapping[ContentKind, str],
    ) -> dict[str, str]:
        return {kind.value: digest for kind, digest in value.items()}

    @model_validator(mode="after")
    def derive_version_id(self) -> CandidateVersion:
        payload = self.model_dump(mode="json", exclude={"version_id"})
        expected = _candidate_payload_hash(payload)
        if self.version_id and self.version_id != expected:
            raise ValueError("version_id does not match immutable candidate content")
        object.__setattr__(self, "version_id", expected)
        return self


class HumanApproval(_StrictDTO):
    """Explicit, deterministic approval attestation bound to one action and candidate."""

    action: Literal["promote", "rollback"]
    candidate_version_id: str
    approver_id: str
    approved: Literal[True] = True
    attestation_hash: str

    @field_validator("candidate_version_id", "attestation_hash")
    @classmethod
    def validate_hashes(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("approval identifiers must be SHA-256 values")
        return value

    @field_validator("approver_id")
    @classmethod
    def validate_approver(cls, value: str) -> str:
        if _PUBLIC_ID.fullmatch(value) is None:
            raise ValueError("approver_id must be a public identifier")
        return value

    @classmethod
    def attest(
        cls,
        *,
        action: Literal["promote", "rollback"],
        candidate_version_id: str,
        approver_id: str,
    ) -> HumanApproval:
        payload = {
            "action": action,
            "approved": True,
            "approver_id": approver_id,
            "candidate_version_id": candidate_version_id,
        }
        return cls(**payload, attestation_hash=canonical_payload_hash(payload))

    @model_validator(mode="after")
    def verify_attestation(self) -> HumanApproval:
        payload = self.model_dump(mode="json", exclude={"attestation_hash"})
        if self.attestation_hash != canonical_payload_hash(payload):
            raise ValueError("approval attestation hash mismatch")
        return self


class PromotionReport(_StrictDTO):
    action: Literal["promote", "rollback"]
    candidate_version_id: str
    previous_active_version_id: str | None = None
    active_version_id: str | None = None
    promoted: bool
    eval_passed: bool
    human_approved: bool
    reason_codes: tuple[str, ...] = ()
    suite_id: str | None = None
    suite_hash: str | None = None
    seed: int | None = Field(default=None, ge=0)
    approver_id: str | None = None
    approval_attestation_hash: str | None = None
    report_hash: str = ""

    @field_validator(
        "candidate_version_id",
        "previous_active_version_id",
        "active_version_id",
        "suite_hash",
        "approval_attestation_hash",
        "report_hash",
    )
    @classmethod
    def validate_hash_ids(cls, value: str | None) -> str | None:
        if value is not None and value != "" and _SHA256.fullmatch(value) is None:
            raise ValueError("promotion identifiers must be SHA-256 values")
        return value

    @field_validator("suite_id", "approver_id")
    @classmethod
    def validate_public_ids(cls, value: str | None) -> str | None:
        if value is not None and _PUBLIC_ID.fullmatch(value) is None:
            raise ValueError("promotion identities must be public identifiers")
        return value

    @field_validator("reason_codes")
    @classmethod
    def validate_reasons(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(_REASON.fullmatch(item) is None for item in value):
            raise ValueError("promotion reasons must be public codes")
        return tuple(sorted(set(value)))

    @model_validator(mode="after")
    def validate_and_hash_report(self) -> PromotionReport:
        if self.promoted:
            if self.active_version_id != self.candidate_version_id or self.reason_codes:
                raise ValueError("successful promotion must activate candidate without denial reasons")
        elif self.active_version_id != self.previous_active_version_id:
            raise ValueError("denied promotion cannot change active version")

        v2_fields = {
            "approval_attestation_hash",
            "approver_id",
            "seed",
            "suite_hash",
            "suite_id",
        }
        is_legacy = all(
            value is None
            for value in (
                self.approval_attestation_hash,
                self.approver_id,
                self.seed,
                self.suite_hash,
                self.suite_id,
            )
        )
        if not is_legacy:
            suite_values = (self.suite_id, self.suite_hash, self.seed)
            if any(value is None for value in suite_values):
                raise ValueError("promotion report requires a complete suite binding")
            identity_values = (self.approver_id, self.approval_attestation_hash)
            if (self.approver_id is None) != (self.approval_attestation_hash is None):
                raise ValueError("promotion approver and attestation must be recorded together")
            if self.promoted and (not self.human_approved or any(v is None for v in identity_values)):
                raise ValueError("successful promotion requires a trusted approval attestation")

        payload = self.model_dump(mode="json", exclude={"report_hash"})
        if is_legacy:
            for field in v2_fields:
                payload.pop(field, None)
        expected = canonical_payload_hash(payload)
        if self.report_hash and self.report_hash != expected:
            raise ValueError("promotion report hash mismatch")
        object.__setattr__(self, "report_hash", expected)
        return self

    def verify_integrity(self) -> bool:
        try:
            type(self).model_validate_json(self.model_dump_json())
        except ValidationError:
            return False
        return True


class AuditEvent(_StrictDTO):
    """A stable append-only activation event linked to the previous event hash."""

    sequence: int = Field(ge=1)
    event_type: Literal["promote", "rollback"]
    candidate_version_id: str
    previous_active_version_id: str | None = None
    suite_id: str
    suite_hash: str
    seed: int = Field(ge=0)
    approver_id: str
    approval_attestation_hash: str
    promotion_report_hash: str
    previous_event_hash: str | None = None
    event_hash: str = ""

    @field_validator(
        "candidate_version_id",
        "previous_active_version_id",
        "suite_hash",
        "approval_attestation_hash",
        "promotion_report_hash",
        "previous_event_hash",
        "event_hash",
    )
    @classmethod
    def validate_hash_ids(cls, value: str | None) -> str | None:
        if value is not None and value != "" and _SHA256.fullmatch(value) is None:
            raise ValueError("audit identifiers must be SHA-256 values")
        return value

    @field_validator("suite_id", "approver_id")
    @classmethod
    def validate_public_ids(cls, value: str) -> str:
        if _PUBLIC_ID.fullmatch(value) is None:
            raise ValueError("audit identities must be public identifiers")
        return value

    @model_validator(mode="after")
    def validate_and_hash_event(self) -> AuditEvent:
        payload = self.model_dump(mode="json", exclude={"event_hash"})
        expected = canonical_payload_hash(payload)
        if self.event_hash and self.event_hash != expected:
            raise ValueError("audit event hash mismatch")
        object.__setattr__(self, "event_hash", expected)
        return self


def _candidate_payload_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "AuditEvent",
    "CandidateVersion",
    "ContentKind",
    "EvaluationGate",
    "HumanApproval",
    "PromotionReport",
]
