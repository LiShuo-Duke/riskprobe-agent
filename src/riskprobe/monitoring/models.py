from typing import Literal

from riskprobe.models import FrozenModel


class FeatureReference(FrozenModel):
    feature: str
    family: str
    dtype: str
    missing_rate: float
    zero_rate: float
    quantile_edges: tuple[float, ...]
    histogram_counts: tuple[int, ...]


class RuleReference(FrozenModel):
    rule_id: str
    coverage: float
    bad_rate: float
    lift: float


class ReferenceSnapshot(FrozenModel):
    """Aggregate reference data with no temporal bucket or month dimension.

    Monitoring conclusions are limited to the dimensions represented below;
    rule-validation time splits are a separate capability and are not encoded as
    monitoring drift evidence.
    """

    snapshot_id: str
    dataset_id: str
    row_count: int
    positive_rate: float
    target_column: str
    segment_column: str
    min_group_size: int
    segment_counts: dict[str, int]
    features: tuple[FeatureReference, ...]
    rules: tuple[RuleReference, ...]
    created_at: str


class Alert(FrozenModel):
    alert_id: str
    alert_type: Literal[
        "schema", "missingness", "distribution", "population", "label", "rule_decay"
    ]
    severity: Literal["warning", "critical"]
    scope: Literal["dataset", "institution", "feature", "family", "rule"]
    scope_value: str
    metric: str
    reference_value: float | str | None
    current_value: float | str | None
    delta: float | None
    evidence: dict[str, float | int | str]


class RootCause(FrozenModel):
    dimension: str
    value: str
    contribution: float
    rank: int
    evidence: dict[str, float | int | str]


class Diagnosis(FrozenModel):
    snapshot_id: str
    alerts: tuple[Alert, ...]
    root_causes: tuple[RootCause, ...]
    created_at: str
