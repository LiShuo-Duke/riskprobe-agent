import re
from typing import Literal

from pydantic import field_validator

from riskprobe.models import FrozenModel

_TOKEN_NAMESPACE = re.compile(r"[a-z][a-z0-9-]{2,63}\Z")


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
    snapshot_id: str
    dataset_id: str
    token_namespace: str
    row_count: int
    positive_rate: float
    segment_counts: dict[str, int]
    features: tuple[FeatureReference, ...]
    rules: tuple[RuleReference, ...]
    created_at: str

    @field_validator("token_namespace")
    @classmethod
    def validate_token_namespace(cls, value: str) -> str:
        if not _TOKEN_NAMESPACE.fullmatch(value):
            raise ValueError("token namespace must be a strict safe ID")
        return value

    def assert_comparable_token_namespace(self, other: "ReferenceSnapshot") -> None:
        """Reject a comparison unless both snapshots declare the same token namespace."""
        if not isinstance(other, ReferenceSnapshot) or self.token_namespace != other.token_namespace:
            raise ValueError("token namespaces do not match")


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
    scope: str
    metric: str
    contribution: float
    evidence: dict[str, float | int | str]


class Diagnosis(FrozenModel):
    snapshot_id: str
    alerts: tuple[Alert, ...]
    root_causes: tuple[RootCause, ...]
    created_at: str
