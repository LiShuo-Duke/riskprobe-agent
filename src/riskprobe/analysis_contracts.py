"""Strict, bounded aggregate-only contracts for full RiskProbe analysis output."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from riskprobe.privacy import assert_safe_payload

_PUBLIC_CODE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_PUBLIC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_FEATURE_TOKEN = re.compile(r"^model-feature-[0-9a-f]{16}$")
_SENSITIVE_FEATURE_PART = re.compile(
    r"(?:account|borrower|customer|entity|member|person|user|loan|path|email|phone)",
    re.IGNORECASE,
)
_MAX_LIMITATIONS = 32
_MAX_TOP_ITEMS = 100
_MAX_MODEL_TERMS = 1024


class _StrictDTO(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )


class StageStatus(StrEnum):
    NOT_RUN = "not_run"
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    REUSED = "reused"


class StageName(StrEnum):
    CONFIG = "config"
    SNAPSHOT = "snapshot"
    PROFILE = "profile"
    PARTITION = "partition"
    DISCOVERY = "discovery"
    WOE = "woe"
    SCORECARD = "scorecard"
    VALIDATION = "validation"
    INSTITUTION_ANALYSIS = "institution_analysis"
    REPORT = "report"
    FINALIZE = "finalize"
    INSPECT = "inspect"
    DIAGNOSE_QUALITY = "diagnose_quality"
    DIAGNOSE_FEATURE_DRIFT = "diagnose_feature_drift"
    DIAGNOSE_POPULATION_SHIFT = "diagnose_population_shift"
    DIAGNOSE_TARGET_SHIFT = "diagnose_target_shift"
    DIAGNOSE_SEGMENT_RISK = "diagnose_segment_risk"
    DIAGNOSE_TIME_STABILITY = "diagnose_time_stability"
    DIAGNOSE_RULE_EVIDENCE = "diagnose_rule_evidence"
    DISCOVER_RESTORE = "discover_restore"
    DECISION_CONTEXT = "decision_context"
    RECOMMEND = "recommend"
    REVIEW = "review"
    TERMINAL = "terminal"


APPROVED_STAGE_NAMES = tuple(stage.value for stage in StageName)


def _finite(value: float | None, field_name: str) -> float | None:
    if value is not None and not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


def _public_code(value: str, field_name: str) -> str:
    if _PUBLIC_CODE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a public code")
    return value


def _codes(
    value: Sequence[str],
    field_name: str,
    *,
    maximum: int = _MAX_TOP_ITEMS,
) -> tuple[str, ...]:
    values = tuple(value)
    if len(values) > maximum:
        raise ValueError(f"{field_name} exceeds the public output limit")
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must be unique")
    return tuple(sorted(_public_code(item, field_name) for item in values))


class StageSummary(_StrictDTO):
    name: StageName
    status: StageStatus
    enabled: bool
    output_available: bool
    reason_code: str | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    limitations: tuple[str, ...] = ()

    @field_validator("reason_code")
    @classmethod
    def validate_reason_code(cls, value: str | None) -> str | None:
        return None if value is None else _public_code(value, "reason_code")

    @field_validator("limitations")
    @classmethod
    def validate_limitations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _codes(value, "limitations", maximum=_MAX_LIMITATIONS)

    @model_validator(mode="after")
    def validate_stage(self) -> StageSummary:
        if self.status in {StageStatus.SUCCEEDED, StageStatus.REUSED}:
            if not self.output_available or self.reason_code is not None:
                raise ValueError("successful stages require output_available without reason_code")
        elif self.status in {StageStatus.FAILED, StageStatus.UNAVAILABLE}:
            if self.output_available or self.reason_code is None:
                raise ValueError("failed or unavailable stages require reason_code without output_available")
        elif self.output_available:
            raise ValueError("skipped or not_run stages cannot have output_available")
        assert_safe_payload(self.model_dump(mode="json"))
        return self


class FeatureRef(_StrictDTO):
    value: str
    tokenized: bool = False

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: str) -> str:
        if _FEATURE_TOKEN.fullmatch(value) is not None:
            return value
        if _PUBLIC_CODE.fullmatch(value) is None or _SENSITIVE_FEATURE_PART.search(value):
            raise ValueError("feature reference must be public or tokenized")
        return value

    @model_validator(mode="after")
    def validate_token_flag(self) -> FeatureRef:
        is_token = _FEATURE_TOKEN.fullmatch(self.value) is not None
        if is_token != self.tokenized:
            raise ValueError("feature tokenized flag is inconsistent")
        assert_safe_payload(self.model_dump(mode="json"))
        return self

    @classmethod
    def from_name(cls, value: str) -> FeatureRef:
        if not isinstance(value, str) or not value:
            raise ValueError("feature name must be non-empty")
        if _PUBLIC_CODE.fullmatch(value) is not None and not _SENSITIVE_FEATURE_PART.search(value):
            return cls(value=value)
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
        return cls(value=f"model-feature-{digest}", tokenized=True)


class InputSummary(_StrictDTO):
    dataset_id: str
    read_only: Literal[True] = True
    selected_feature_count: int = Field(ge=0)
    target_positive_meaning: str | None = None
    performance_window_known: bool

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, value: str) -> str:
        return _public_code(value, "dataset_id")

    @field_validator("target_positive_meaning")
    @classmethod
    def validate_target_meaning(cls, value: str | None) -> str | None:
        return None if value is None else _public_code(value, "target_positive_meaning")


class ProfileSummary(_StrictDTO):
    row_count: int = Field(ge=0)
    feature_count: int = Field(ge=0)
    numeric_feature_count: int = Field(ge=0)
    positive_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    segment_count: int = Field(ge=0)
    segment_min_size: int | None = Field(default=None, ge=0)
    segment_max_size: int | None = Field(default=None, ge=0)
    metadata_grade: Literal["A", "B"]
    issue_codes: tuple[str, ...] = ()

    @field_validator("issue_codes")
    @classmethod
    def validate_issue_codes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _codes(value, "issue_codes")


class PartitionSummary(_StrictDTO):
    requested_time_validation: bool
    applied_time_validation: bool
    mode: Literal["strict", "auto", "disabled"]
    strategy: str
    train_rows: int = Field(ge=0)
    test_rows: int = Field(ge=0)
    holdout_rows: int = Field(ge=0)
    excluded_null_snapshot_rows: int = Field(ge=0)
    fallback_reason_code: str | None = None

    @field_validator("strategy", "fallback_reason_code")
    @classmethod
    def validate_codes(cls, value: str | None, info: object) -> str | None:
        return None if value is None else _public_code(value, getattr(info, "field_name", "code"))


class MetricDistribution(_StrictDTO):
    count: int = Field(ge=0)
    minimum: float | None = None
    median: float | None = None
    maximum: float | None = None

    @field_validator("minimum", "median", "maximum")
    @classmethod
    def validate_finite(cls, value: float | None, info: object) -> float | None:
        return _finite(value, getattr(info, "field_name", "metric"))

    @model_validator(mode="after")
    def validate_distribution(self) -> MetricDistribution:
        values = (self.minimum, self.median, self.maximum)
        if self.count == 0 and any(value is not None for value in values):
            raise ValueError("empty distributions cannot contain metrics")
        if self.count > 0 and any(value is None for value in values):
            raise ValueError("non-empty distributions require all metrics")
        if self.minimum is not None and not self.minimum <= self.median <= self.maximum:
            raise ValueError("distribution values are invalid")
        return self


class DiscoverySummary(_StrictDTO):
    sampled: bool
    sample_rows: int | None = Field(default=None, ge=0)
    input_feature_count: int = Field(ge=0)
    eligible_feature_count: int = Field(ge=0)
    skipped_feature_reason_counts: Mapping[str, int] = Field(default_factory=dict)
    candidate_rule_count: int = Field(ge=0)
    selected_rule_count: int = Field(ge=0)
    single_candidate_count: int = Field(ge=0)
    single_rule_count: int = Field(ge=0)
    pair_candidate_count: int = Field(ge=0)
    pair_rule_count: int = Field(ge=0)
    rule_ids: tuple[str, ...] = ()
    lift: MetricDistribution
    support: MetricDistribution
    precision: MetricDistribution

    @field_validator("skipped_feature_reason_counts")
    @classmethod
    def validate_skip_counts(cls, value: Mapping[str, int]) -> dict[str, int]:
        if len(value) > _MAX_TOP_ITEMS:
            raise ValueError("skipped feature reasons exceed the public output limit")
        normalized = {_public_code(key, "skip reason"): item for key, item in value.items()}
        if any(type(item) is not int or item < 0 for item in normalized.values()):
            raise ValueError("skipped feature counts must be non-negative integers")
        return dict(sorted(normalized.items()))

    @field_validator("rule_ids")
    @classmethod
    def validate_rule_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > _MAX_MODEL_TERMS:
            raise ValueError("rule_ids exceed the public output limit")
        if len(value) != len(set(value)) or any(
            _PUBLIC_ID.fullmatch(item) is None for item in value
        ):
            raise ValueError("rule_ids must contain unique public identifiers")
        normalized = tuple(sorted(value))
        assert_safe_payload({"rule_ids": normalized})
        return normalized


class ScorecardFeatureSummary(_StrictDTO):
    feature: FeatureRef
    iv: float = Field(ge=0.0)
    bin_count: int = Field(ge=0)
    monotonic: Literal["none", "increasing", "decreasing"]
    has_missing_bin: bool


class ScorecardTerm(_StrictDTO):
    term: FeatureRef
    kind: Literal["feature", "rule"]
    coefficient: float

    @field_validator("coefficient")
    @classmethod
    def validate_coefficient(cls, value: float) -> float:
        return _finite(value, "coefficient")  # type: ignore[return-value]


class ScorecardSplitMetrics(_StrictDTO):
    split: Literal["train", "test", "holdout"]
    status: StageStatus
    reason_code: str | None = None
    sample_count: int = Field(ge=0)
    positive_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    probability_min: float | None = Field(default=None, ge=0.0, le=1.0)
    probability_mean: float | None = Field(default=None, ge=0.0, le=1.0)
    probability_max: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_score: float | None = None
    risk_level_counts: Mapping[str, int] = Field(default_factory=dict)
    auc: float | None = Field(default=None, ge=0.0, le=1.0)
    ks: float | None = Field(default=None, ge=0.0, le=1.0)
    gini: float | None = Field(default=None, ge=-1.0, le=1.0)

    @field_validator("reason_code")
    @classmethod
    def validate_reason_code(cls, value: str | None) -> str | None:
        return None if value is None else _public_code(value, "reason_code")

    @field_validator("mean_score")
    @classmethod
    def validate_mean_score(cls, value: float | None) -> float | None:
        return _finite(value, "mean_score")

    @field_validator("risk_level_counts")
    @classmethod
    def validate_risk_counts(cls, value: Mapping[str, int]) -> dict[str, int]:
        normalized = {_public_code(key, "risk level"): item for key, item in value.items()}
        if any(type(item) is not int or item < 0 for item in normalized.values()):
            raise ValueError("risk level counts must be non-negative integers")
        return dict(sorted(normalized.items()))

    @model_validator(mode="after")
    def validate_metrics(self) -> ScorecardSplitMetrics:
        if sum(self.risk_level_counts.values()) != self.sample_count:
            raise ValueError("risk level counts must equal sample_count")
        values = (
            self.positive_rate,
            self.probability_min,
            self.probability_mean,
            self.probability_max,
            self.mean_score,
            self.auc,
            self.ks,
            self.gini,
        )
        if self.status is StageStatus.SUCCEEDED:
            if self.reason_code is not None or any(value is None for value in values):
                raise ValueError("successful split metrics require complete metrics without reason_code")
            if not self.probability_min <= self.probability_mean <= self.probability_max:
                raise ValueError("probability metrics are invalid")
            if not math.isclose(self.gini, 2.0 * self.auc - 1.0, abs_tol=1e-12):
                raise ValueError("gini must equal 2 * auc - 1")
        elif self.status is StageStatus.UNAVAILABLE:
            if self.reason_code is None or any(value is not None for value in (self.auc, self.ks, self.gini)):
                raise ValueError("unavailable split metrics require reason_code without discrimination metrics")
        else:
            raise ValueError("split metrics must be succeeded or unavailable")
        assert_safe_payload(self.model_dump(mode="json"))
        return self


class ScorecardSummary(_StrictDTO):
    enabled: bool
    status: StageStatus
    reason_code: str | None = None
    model_type: Literal["logistic_regression"] | None = None
    calibrated: bool | None = None
    parameters: Mapping[str, float | int | str | bool] = Field(default_factory=dict)
    imbalance_strategy: str | None = None
    class_counts: Mapping[str, int] = Field(default_factory=dict)
    input_features: tuple[FeatureRef, ...] = ()
    included_features: tuple[FeatureRef, ...] = ()
    excluded_features: Mapping[str, str] = Field(default_factory=dict)
    feature_summaries: tuple[ScorecardFeatureSummary, ...] = ()
    terms: tuple[ScorecardTerm, ...] = ()
    intercept: float | None = None
    splits: tuple[ScorecardSplitMetrics, ...] = ()

    @field_validator("reason_code", "imbalance_strategy")
    @classmethod
    def validate_optional_codes(cls, value: str | None, info: object) -> str | None:
        return None if value is None else _public_code(value, getattr(info, "field_name", "code"))

    @field_validator("intercept")
    @classmethod
    def validate_intercept(cls, value: float | None) -> float | None:
        return _finite(value, "intercept")

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, value: Mapping[str, float | int | str | bool]) -> dict[str, float | int | str | bool]:
        if len(value) > _MAX_TOP_ITEMS:
            raise ValueError("scorecard parameters exceed the public output limit")
        normalized = {_public_code(key, "parameter"): item for key, item in value.items()}
        if any(isinstance(item, float) and not math.isfinite(item) for item in normalized.values()):
            raise ValueError("scorecard parameters must be finite")
        return dict(sorted(normalized.items()))

    @model_validator(mode="after")
    def validate_scorecard(self) -> ScorecardSummary:
        if len(self.terms) > _MAX_MODEL_TERMS:
            raise ValueError("complete scorecard terms exceed the public output limit")
        term_names = [term.term.value for term in self.terms]
        if len(term_names) != len(set(term_names)):
            raise ValueError("scorecard terms must be unique")
        if self.status is StageStatus.SUCCEEDED:
            if not self.enabled or self.reason_code is not None or self.model_type is None or self.calibrated is None or self.intercept is None:
                raise ValueError("fitted scorecard requires complete model metadata")
            input_names = {feature.value for feature in self.input_features}
            included_names = {feature.value for feature in self.included_features}
            excluded_names = set(self.excluded_features)
            if len(input_names) != len(self.input_features) or len(included_names) != len(self.included_features):
                raise ValueError("scorecard feature references must be unique")
            if included_names & excluded_names or input_names != included_names | excluded_names:
                raise ValueError("scorecard feature sets are inconsistent")
            summary_names = {item.feature.value for item in self.feature_summaries}
            feature_term_names = {term.term.value for term in self.terms if term.kind == "feature"}
            if summary_names != included_names or feature_term_names != included_names:
                raise ValueError("scorecard feature content is incomplete")
            split_names = [split.split for split in self.splits]
            if len(split_names) != len(set(split_names)) or set(split_names) != {"train", "test", "holdout"}:
                raise ValueError("scorecard splits are incomplete")
            if any(type(value) is not int or value < 0 for value in self.class_counts.values()):
                raise ValueError("scorecard class counts are invalid")
        elif self.status in {StageStatus.SKIPPED, StageStatus.UNAVAILABLE}:
            if self.status is StageStatus.UNAVAILABLE and self.reason_code is None:
                raise ValueError("unavailable scorecard requires reason_code")
        else:
            raise ValueError("scorecard status must be succeeded, skipped, or unavailable")
        assert_safe_payload(self.model_dump(mode="json"))
        return self


class ValidationSummary(_StrictDTO):
    evidence_count: int = Field(ge=0)
    grade_counts: Mapping[str, int] = Field(default_factory=dict)
    holdout_status: StageStatus
    limitation_counts: Mapping[str, int] = Field(default_factory=dict)
    lift: MetricDistribution
    adjusted_p_value: MetricDistribution
    segment_consistency: MetricDistribution
    time_decay: MetricDistribution


class ArtifactSummary(_StrictDTO):
    logical_artifacts: tuple[str, ...]
    artifact_count: int = Field(ge=0)
    published: bool
    reused: bool
    integrity_verified: bool

    @field_validator("logical_artifacts")
    @classmethod
    def validate_artifacts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _codes(value, "logical_artifacts", maximum=_MAX_TOP_ITEMS)


class DiagnosticsSummary(_StrictDTO):
    diagnostic_time_enabled: bool
    finding_counts_by_kind: Mapping[str, int] = Field(default_factory=dict)
    finding_counts_by_severity: Mapping[str, int] = Field(default_factory=dict)
    finding_counts_by_code: Mapping[str, int] = Field(default_factory=dict)


class AnalysisSummary(_StrictDTO):
    schema_version: Literal["riskprobe.analysis-summary.v1"] = "riskprobe.analysis-summary.v1"
    stages: tuple[StageSummary, ...]
    input: InputSummary | None = None
    profile: ProfileSummary | None = None
    partition: PartitionSummary | None = None
    discovery: DiscoverySummary | None = None
    scorecard: ScorecardSummary | None = None
    validation: ValidationSummary | None = None
    artifacts: ArtifactSummary | None = None
    diagnostics: DiagnosticsSummary | None = None

    @field_validator("stages")
    @classmethod
    def validate_stages(cls, value: tuple[StageSummary, ...]) -> tuple[StageSummary, ...]:
        names = tuple(stage.name.value for stage in value)
        if len(value) != len(StageName) or set(names) != set(APPROVED_STAGE_NAMES):
            raise ValueError("analysis summary must contain every approved stage exactly once")
        return tuple(sorted(value, key=lambda stage: APPROVED_STAGE_NAMES.index(stage.name.value)))

    @model_validator(mode="after")
    def validate_privacy(self) -> AnalysisSummary:
        assert_safe_payload(self.model_dump(mode="json"))
        return self


class RecommendationSummary(_StrictDTO):
    action_code: str
    parent_finding_ids: tuple[str, ...]
    evidence_id: str
    human_approval_required: Literal[True] = True
    analysis_only: bool
    limitations: tuple[str, ...] = ()

    @field_validator("action_code")
    @classmethod
    def validate_action_code(cls, value: str) -> str:
        return _public_code(value, "action_code")

    @field_validator("parent_finding_ids")
    @classmethod
    def validate_parent_finding_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) != len(set(value)) or any(
            re.fullmatch(r"[0-9a-f]{64}", item) is None for item in value
        ):
            raise ValueError("parent_finding_ids must contain unique SHA-256 identifiers")
        return tuple(sorted(value))

    @field_validator("limitations")
    @classmethod
    def validate_limitations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _codes(value, "limitations")

    @field_validator("evidence_id")
    @classmethod
    def validate_evidence_id(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("evidence_id must be a SHA-256 identifier")
        return value


class DecisionSummary(_StrictDTO):
    selected_action_codes: tuple[str, ...] = ()
    recommendations: tuple[RecommendationSummary, ...] = ()
    recommendation_status: StageStatus
    review_approved: bool
    review_reason_codes: tuple[str, ...] = ()
    no_action_required: bool
    retry_count: int = Field(ge=0, le=1)
    tool_sequence: tuple[str, ...]
    evidence_complete: bool
    decision_status: Literal["accepted", "rejected", "no_action"]
    final_status: Literal["succeeded", "rejected", "failed"]

    @field_validator("selected_action_codes", "review_reason_codes")
    @classmethod
    def validate_public_codes(cls, value: tuple[str, ...], info: object) -> tuple[str, ...]:
        return _codes(value, getattr(info, "field_name", "codes"))

    @field_validator("tool_sequence")
    @classmethod
    def validate_tool_sequence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > _MAX_TOP_ITEMS or len(value) != len(set(value)):
            raise ValueError("tool_sequence must contain bounded unique codes")
        return tuple(_public_code(item, "tool_sequence") for item in value)

    @model_validator(mode="after")
    def validate_decision(self) -> DecisionSummary:
        if self.no_action_required and (self.selected_action_codes or self.recommendations):
            raise ValueError("no-action decisions cannot have actions or recommendations")
        if self.decision_status == "accepted" and not self.review_approved:
            raise ValueError("accepted decisions require approved review")
        assert_safe_payload(self.model_dump(mode="json"))
        return self


__all__ = [
    "APPROVED_STAGE_NAMES",
    "AnalysisSummary",
    "ArtifactSummary",
    "DecisionSummary",
    "DiagnosticsSummary",
    "DiscoverySummary",
    "FeatureRef",
    "InputSummary",
    "MetricDistribution",
    "PartitionSummary",
    "ProfileSummary",
    "RecommendationSummary",
    "ScorecardFeatureSummary",
    "ScorecardSplitMetrics",
    "ScorecardSummary",
    "ScorecardTerm",
    "StageName",
    "StageStatus",
    "StageSummary",
    "ValidationSummary",
]
