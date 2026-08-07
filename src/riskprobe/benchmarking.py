"""Auditable measurements for locally executed RiskProbe workflows."""

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class StageTiming(_StrictModel):
    stage: Literal["inspect", "discover", "validate", "monitor", "report"]
    seconds: float = Field(ge=0)


class RuleReviewSummary(_StrictModel):
    candidate_rule_count: int = Field(ge=0)
    evidence_passed_count: int = Field(ge=0)
    reviewed_rule_count: int = Field(ge=0)
    accepted_rule_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> "RuleReviewSummary":
        if self.evidence_passed_count > self.candidate_rule_count:
            raise ValueError("evidence_passed_count cannot exceed candidate_rule_count")
        if self.accepted_rule_count > self.reviewed_rule_count:
            raise ValueError("accepted_rule_count cannot exceed reviewed_rule_count")
        return self


class AnomalyEvaluationSummary(_StrictModel):
    true_positive_count: int = Field(ge=0)
    false_positive_count: int = Field(ge=0)
    false_negative_count: int = Field(ge=0)
    root_cause_top3_hit_count: int = Field(ge=0)
    root_cause_case_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> "AnomalyEvaluationSummary":
        if self.root_cause_top3_hit_count > self.root_cause_case_count:
            raise ValueError("root_cause_top3_hit_count cannot exceed root_cause_case_count")
        return self


class BenchmarkRecord(_StrictModel):
    run_id: str
    task_id: str
    dataset_id: str
    measured_at: str
    code_version: str
    config_hash: str
    data_fingerprint: str
    manual_minutes: float | None = Field(ge=0)
    agent_minutes: float = Field(ge=0)
    stage_timings: tuple[StageTiming, ...]
    candidate_rule_count: int = Field(ge=0)
    evidence_passed_count: int = Field(ge=0)
    reviewed_rule_count: int = Field(ge=0)
    accepted_rule_count: int = Field(ge=0)
    anomaly_true_positive_count: int = Field(ge=0)
    anomaly_false_positive_count: int = Field(ge=0)
    anomaly_false_negative_count: int = Field(ge=0)
    root_cause_top3_hit_count: int = Field(ge=0)
    root_cause_case_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_consistency(self) -> "BenchmarkRecord":
        RuleReviewSummary(
            candidate_rule_count=self.candidate_rule_count,
            evidence_passed_count=self.evidence_passed_count,
            reviewed_rule_count=self.reviewed_rule_count,
            accepted_rule_count=self.accepted_rule_count,
        )
        AnomalyEvaluationSummary(
            true_positive_count=self.anomaly_true_positive_count,
            false_positive_count=self.anomaly_false_positive_count,
            false_negative_count=self.anomaly_false_negative_count,
            root_cause_top3_hit_count=self.root_cause_top3_hit_count,
            root_cause_case_count=self.root_cause_case_count,
        )
        if len({timing.stage for timing in self.stage_timings}) != len(self.stage_timings):
            raise ValueError("stage_timings must contain each stage at most once")
        return self

    @property
    def precision(self) -> float | None:
        denominator = self.anomaly_true_positive_count + self.anomaly_false_positive_count
        return self.anomaly_true_positive_count / denominator if denominator else None

    @property
    def recall(self) -> float | None:
        denominator = self.anomaly_true_positive_count + self.anomaly_false_negative_count
        return self.anomaly_true_positive_count / denominator if denominator else None

    @property
    def rule_review_acceptance(self) -> float | None:
        return self.accepted_rule_count / self.reviewed_rule_count if self.reviewed_rule_count else None

    @property
    def root_cause_top3_hit_rate(self) -> float | None:
        return (
            self.root_cause_top3_hit_count / self.root_cause_case_count
            if self.root_cause_case_count
            else None
        )


def calculate_efficiency(record: BenchmarkRecord) -> float:
    if record.manual_minutes is None:
        raise ValueError("manual baseline is required to calculate efficiency")
    if record.manual_minutes == 0:
        raise ValueError("manual baseline must be greater than zero to calculate efficiency")
    return (record.manual_minutes - record.agent_minutes) / record.manual_minutes


def total_agent_minutes(timings: Iterable[StageTiming]) -> float:
    return sum(timing.seconds for timing in timings) / 60
