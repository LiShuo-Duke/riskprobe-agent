from typing import Literal

from pydantic import BaseModel, ConfigDict

Operator = Literal[">", ">=", "<", "<=", "==", "!=", "is_null"]
EvidenceGrade = Literal["Stable", "Local", "Unstable", "Suspicious"]


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class Condition(FrozenModel):
    feature: str
    operator: Operator
    value: float | int | str | None = None


class RiskRule(FrozenModel):
    rule_id: str
    conditions: tuple[Condition, ...]
    origin: str


class RuleMetrics(FrozenModel):
    support_count: int
    coverage: float
    base_bad_rate: float
    hit_bad_rate: float
    non_hit_bad_rate: float
    lift: float
    precision: float
    recall: float
    p_value: float


class SliceMetrics(FrozenModel):
    slice_type: Literal["dataset", "institution", "time"]
    slice_value: str
    metrics: RuleMetrics


class EvidenceCard(FrozenModel):
    rule: RiskRule
    train: RuleMetrics
    test: RuleMetrics
    slices: tuple[SliceMetrics, ...]
    lift_ci: tuple[float, float]
    adjusted_p_value: float
    segment_consistency: float
    max_time_decay: float
    grade: EvidenceGrade
    limitations: tuple[str, ...] = ()
