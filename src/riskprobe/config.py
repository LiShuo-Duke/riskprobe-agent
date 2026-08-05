from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetConfig(StrictModel):
    id: str
    path: Path
    read_only: bool = True


class ColumnRoles(StrictModel):
    entity: str
    snapshot: str
    segment: str
    target: str


class TargetConfig(StrictModel):
    positive_value: int = 1
    positive_meaning: Literal["bad_debt"]
    performance_window_days: int | None = Field(default=None, gt=0)


class SnapshotConfig(StrictModel):
    meaning: Literal[
        "customer_specified_feature_cutoff",
        "public_relative_reference",
    ]


class FeatureFamilyConfig(StrictModel):
    families: dict[str, list[str]]
    explicit_catalog: Path | None = None


class DiscoveryConfig(StrictModel):
    min_support: float = Field(default=0.05, gt=0, lt=1)
    max_single_rules: int = Field(default=100, ge=1)
    beam_width: int = Field(default=20, ge=1)
    max_pair_rules: int = Field(default=50, ge=0)
    random_seed: int = 42


class ValidationConfig(StrictModel):
    alpha: float = Field(default=0.05, gt=0, lt=1)
    min_segment_consistency: float = Field(default=0.6, ge=0, le=1)
    max_lift_decay: float = Field(default=0.3, ge=0)
    bootstrap_rounds: int = Field(default=500, ge=100)
    min_group_size: int = Field(default=100, ge=20)


class ProjectConfig(StrictModel):
    dataset: DatasetConfig
    columns: ColumnRoles
    target: TargetConfig
    snapshot: SnapshotConfig
    features: FeatureFamilyConfig
    segment_display_name: Literal["institution", "customer_segment"] = "institution"
    time_validation_enabled: bool = True
    discovery: DiscoveryConfig = DiscoveryConfig()
    validation: ValidationConfig = ValidationConfig()

    @property
    def metadata_grade(self) -> Literal["A", "B"]:
        return "A" if self.target.performance_window_days is not None else "B"

    @classmethod
    def from_yaml(cls, path: Path) -> "ProjectConfig":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(payload)
