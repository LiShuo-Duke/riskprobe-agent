"""Strict privacy-safe contracts for aggregate risk diagnostics."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import date
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

from riskprobe.privacy import SegmentToken, assert_safe_payload, canonical_payload_hash
from riskprobe.profiling import DatasetProfile

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CODE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FindingKind(StrEnum):
    DATA_QUALITY = "data_quality"
    FEATURE_DRIFT = "feature_drift"
    POPULATION_SHIFT = "population_shift"
    TARGET_SHIFT = "target_shift"
    SEGMENT_RISK = "segment_risk"
    TIME_INSTABILITY = "time_instability"
    RULE_EVIDENCE = "rule_evidence"


class FindingSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )


class SafeProfile(_FrozenModel):
    """Dataset profile containing aggregates only, with no segment labels or paths."""

    dataset_id: str
    row_count: int = Field(ge=0)
    feature_count: int = Field(ge=0)
    positive_rate: float | None = Field(default=None, ge=0, le=1)
    segment_count: int = Field(default=0, ge=0)
    min_segment_size: int | None = Field(default=None, ge=0)
    max_segment_size: int | None = Field(default=None, ge=0)
    snapshot_min: str | None = None
    snapshot_max: str | None = None
    metadata_grade: Literal["A", "B"]
    issue_codes: tuple[str, ...] = ()
    issue_count: int = Field(default=0, ge=0)

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, value: str) -> str:
        if _IDENTIFIER.fullmatch(value) is None:
            raise ValueError("dataset_id must be a public identifier")
        return value

    @field_validator("snapshot_min", "snapshot_max")
    @classmethod
    def validate_snapshot_date(cls, value: str | None) -> str | None:
        if value is not None:
            date.fromisoformat(value)
        return value

    @field_validator("issue_codes")
    @classmethod
    def normalize_issue_codes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item or len(item) > 128 for item in value):
            raise ValueError("issue codes must be non-empty public codes")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def validate_aggregate_consistency(self) -> SafeProfile:
        if self.segment_count == 0:
            if self.min_segment_size is not None or self.max_segment_size is not None:
                raise ValueError("empty segment summary cannot have size bounds")
        elif self.min_segment_size is None or self.max_segment_size is None:
            raise ValueError("non-empty segment summary requires size bounds")
        elif self.min_segment_size > self.max_segment_size:
            raise ValueError("segment size bounds are invalid")
        if self.issue_count < len(self.issue_codes):
            raise ValueError("issue_count cannot be lower than unique issue codes")
        assert_safe_payload(self.model_dump(mode="json"))
        return self

    @classmethod
    def from_profile(cls, profile: DatasetProfile) -> SafeProfile:
        counts = tuple(int(count) for count in profile.segment_counts.values())
        issue_codes = tuple(sorted({issue.code for issue in profile.issues}))
        return cls(
            dataset_id=profile.dataset_id,
            row_count=profile.row_count,
            feature_count=profile.feature_count,
            positive_rate=profile.positive_rate,
            segment_count=len(counts),
            min_segment_size=min(counts) if counts else None,
            max_segment_size=max(counts) if counts else None,
            snapshot_min=_date_text(profile.snapshot_min),
            snapshot_max=_date_text(profile.snapshot_max),
            metadata_grade=profile.metadata_grade,
            issue_codes=issue_codes,
            issue_count=len(profile.issues),
        )


class RiskFinding(_FrozenModel):
    """A deterministic aggregate finding identified by its canonical safe payload."""

    finding_id: str = ""
    kind: FindingKind
    severity: FindingSeverity
    code: str
    feature: str | None = None
    period: str | None = None
    segment_token: SegmentToken | None = None
    metrics: Mapping[str, int | float] = Field(default_factory=dict)
    limitations: tuple[str, ...] = ()

    @field_validator("finding_id")
    @classmethod
    def validate_finding_id(cls, value: str) -> str:
        if value and _SHA256.fullmatch(value) is None:
            raise ValueError("finding_id must be a SHA-256 identifier")
        return value

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        if _CODE.fullmatch(value) is None:
            raise ValueError("finding code must be lower snake case")
        return value

    @field_validator("feature")
    @classmethod
    def validate_feature(cls, value: str | None) -> str | None:
        if value is not None and (not value or len(value) > 256):
            raise ValueError("feature must be a bounded name")
        return value

    @field_validator("period")
    @classmethod
    def validate_period(cls, value: str | None) -> str | None:
        if value is not None and re.fullmatch(r"\d{4}-\d{2}", value) is None:
            raise ValueError("period must use YYYY-MM")
        return value

    @field_validator("metrics")
    @classmethod
    def freeze_metrics(cls, value: Mapping[str, int | float]) -> Mapping[str, int | float]:
        normalized: dict[str, int | float] = {}
        for key, metric in value.items():
            if _CODE.fullmatch(key) is None or isinstance(metric, bool):
                raise ValueError("metrics must contain public numeric aggregates")
            if isinstance(metric, float) and not math.isfinite(metric):
                raise ValueError("metrics must be finite")
            normalized[key] = metric
        return MappingProxyType(dict(sorted(normalized.items())))

    @field_serializer("metrics")
    def serialize_metrics(self, value: Mapping[str, int | float]) -> dict[str, int | float]:
        return dict(value)

    @field_validator("limitations")
    @classmethod
    def normalize_limitations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(_CODE.fullmatch(item) is None for item in value):
            raise ValueError("limitations must be public codes")
        return tuple(sorted(set(value)))

    @model_validator(mode="after")
    def derive_and_validate_finding_id(self) -> RiskFinding:
        payload = self.model_dump(mode="json", exclude={"finding_id"})
        expected = canonical_payload_hash(payload)
        if self.finding_id and self.finding_id != expected:
            raise ValueError("finding_id does not match canonical payload")
        object.__setattr__(self, "finding_id", expected)
        return self


class DiagnosticReport(_FrozenModel):
    """Deterministically ordered aggregate diagnosis for one registered dataset."""

    profile: SafeProfile
    findings: tuple[RiskFinding, ...]
    limitations: tuple[str, ...] = ()
    dataset_id: str = ""
    metadata_grade: Literal["A", "B"] = "A"

    @model_validator(mode="before")
    @classmethod
    def inherit_profile_identity(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        profile = normalized.get("profile")
        if isinstance(profile, SafeProfile):
            normalized.setdefault("dataset_id", profile.dataset_id)
            normalized.setdefault("metadata_grade", profile.metadata_grade)
        elif isinstance(profile, Mapping):
            normalized.setdefault("dataset_id", profile.get("dataset_id"))
            normalized.setdefault("metadata_grade", profile.get("metadata_grade"))
        return normalized

    @field_validator("findings")
    @classmethod
    def normalize_findings(cls, value: tuple[RiskFinding, ...]) -> tuple[RiskFinding, ...]:
        if len({finding.finding_id for finding in value}) != len(value):
            raise ValueError("findings must have unique finding_ids")
        return tuple(sorted(value, key=finding_sort_key))

    @field_validator("limitations")
    @classmethod
    def normalize_report_limitations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(_CODE.fullmatch(item) is None for item in value):
            raise ValueError("limitations must be public codes")
        return tuple(sorted(set(value)))

    @model_validator(mode="after")
    def validate_report_identity_and_privacy(self) -> DiagnosticReport:
        if self.dataset_id != self.profile.dataset_id:
            raise ValueError("report dataset_id must match profile")
        if self.metadata_grade != self.profile.metadata_grade:
            raise ValueError("report metadata_grade must match profile")
        assert_safe_payload(self.model_dump(mode="json"))
        return self


def finding_sort_key(finding: RiskFinding) -> tuple[object, ...]:
    severity_rank = {
        FindingSeverity.CRITICAL: 0,
        FindingSeverity.WARNING: 1,
        FindingSeverity.INFO: 2,
    }
    return (
        severity_rank[finding.severity],
        finding.kind.value,
        finding.code,
        finding.feature or "",
        finding.period or "",
        finding.segment_token.token if finding.segment_token is not None else "",
        finding.finding_id,
    )


def _date_text(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


__all__ = [
    "DiagnosticReport",
    "FindingKind",
    "FindingSeverity",
    "RiskFinding",
    "SafeProfile",
    "finding_sort_key",
]
