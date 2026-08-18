from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DatasetConfig(StrictModel):
    id: str
    path: Path
    read_only: Literal[True] = True

    @field_validator("path", mode="before")
    @classmethod
    def parse_dataset_path(cls, path: object) -> object:
        return Path(path) if isinstance(path, str) else path

    @field_validator("path")
    @classmethod
    def require_local_parquet(cls, path: Path) -> Path:
        path_text = str(path)
        if path_text.startswith(("//", "\\\\")):
            raise ValueError("dataset path must be local")
        is_windows_drive_path = (
            len(path_text) >= 3
            and path_text[0].isalpha()
            and path_text[1] == ":"
            and path_text[2] in ("/", "\\")
        )
        if urlparse(path_text).scheme and not is_windows_drive_path:
            raise ValueError("dataset path must be local")
        if path.suffix.lower() != ".parquet":
            raise ValueError("dataset path must have a .parquet extension")
        return path


class ColumnRoles(StrictModel):
    entity: str
    snapshot: str
    segment: str
    target: str


class TargetConfig(StrictModel):
    positive_value: Literal[1] = 1
    positive_meaning: Literal["bad_debt"]
    performance_window_days: int | None = Field(default=None, gt=0)


class SnapshotConfig(StrictModel):
    meaning: Literal[
        "customer_specified_feature_cutoff",
        "public_relative_reference",
    ]


class FeatureFamilyConfig(StrictModel):
    families: Mapping[str, tuple[str, ...]]
    exact_columns: tuple[str, ...] | None = None
    explicit_catalog: Path | None = None

    @field_validator("explicit_catalog", mode="before")
    @classmethod
    def parse_explicit_catalog_path(cls, path: object) -> object:
        return Path(path) if isinstance(path, str) else path

    @field_validator("families", mode="before")
    @classmethod
    def convert_family_lists_to_tuples(cls, families: object) -> object:
        if not isinstance(families, Mapping):
            return families
        return {
            name: tuple(prefixes) if isinstance(prefixes, list) else prefixes
            for name, prefixes in families.items()
        }

    @field_validator("exact_columns", mode="before")
    @classmethod
    def convert_exact_column_lists_to_tuples(cls, columns: object) -> object:
        return tuple(columns) if isinstance(columns, list) else columns

    @field_validator("families")
    @classmethod
    def freeze_families(
        cls, families: Mapping[str, tuple[str, ...]]
    ) -> Mapping[str, tuple[str, ...]]:
        if any(not prefix for prefixes in families.values() for prefix in prefixes):
            raise ValueError("feature family prefixes must be non-empty")
        return MappingProxyType(dict(families))

    @field_validator("exact_columns")
    @classmethod
    def validate_exact_columns(
        cls, columns: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        if columns is None:
            return None
        if not columns or any(not column for column in columns):
            raise ValueError("exact column names must be non-empty")
        if len(columns) != len(set(columns)):
            raise ValueError("exact column names must be unique")
        return columns

    @field_serializer("families")
    def serialize_families(self, families: Mapping[str, tuple[str, ...]]) -> dict[str, list[str]]:
        return {name: list(prefixes) for name, prefixes in families.items()}

    def select_columns(
        self,
        columns: Iterable[str],
        role_columns: Iterable[str],
    ) -> list[str]:
        candidates = sorted(set(columns).difference(role_columns))
        if self.exact_columns is not None:
            exact = frozenset(self.exact_columns)
            return [column for column in candidates if column in exact]
        if self.explicit_catalog is not None:
            return [
                column
                for column in candidates
                if column in _catalog_feature_names(self.explicit_catalog)
            ]
        prefixes = tuple(
            dict.fromkeys(prefix for family in self.families.values() for prefix in family)
        )
        return [
            column
            for column in candidates
            if any(column.startswith(prefix) for prefix in prefixes)
        ]


def _catalog_feature_names(path: Path) -> frozenset[str]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError("explicit feature catalog must be a readable YAML file") from error
    names = payload
    if isinstance(payload, Mapping):
        names = payload.get("features", payload.get("columns"))
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("explicit feature catalog must list feature names")
    return frozenset(names)


class DiscoveryConfig(StrictModel):
    min_support: float = Field(default=0.05, gt=0, lt=1)
    max_single_rules: int = Field(default=100, ge=1)
    beam_width: int = Field(default=20, ge=1)
    max_pair_rules: int = Field(default=50, ge=0)
    random_seed: Literal[42] = 42


class ValidationConfig(StrictModel):
    alpha: float = Field(default=0.05, gt=0, lt=1)
    min_segment_consistency: float = Field(default=0.6, ge=0, le=1)
    max_lift_decay: float = Field(default=0.3, ge=0)
    bootstrap_rounds: int = Field(default=500, ge=100)
    min_group_size: int = Field(default=100, ge=20)


class PrivacyConfig(StrictModel):
    expose_segment_values: bool = True


class ProjectConfig(StrictModel):
    dataset: DatasetConfig
    columns: ColumnRoles
    target: TargetConfig
    snapshot: SnapshotConfig
    features: FeatureFamilyConfig
    segment_display_name: Literal["institution", "customer_segment"] = "institution"
    time_validation_enabled: bool = True
    metadata_grade_override: Literal["A", "B", "C", "D"] | None = None
    discovery: DiscoveryConfig = DiscoveryConfig()
    validation: ValidationConfig = ValidationConfig()
    privacy: PrivacyConfig = PrivacyConfig()

    @property
    def metadata_grade(self) -> Literal["A", "B", "C", "D"]:
        if self.metadata_grade_override is not None:
            return self.metadata_grade_override
        return "A" if self.target.performance_window_days is not None else "B"

    @classmethod
    def from_yaml(cls, path: Path) -> "ProjectConfig":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = cls.model_validate(payload)
        catalog = config.features.explicit_catalog
        if catalog is None or catalog.is_absolute():
            return config
        return config.model_copy(
            update={
                "features": config.features.model_copy(
                    update={"explicit_catalog": path.resolve().parent / catalog}
                )
            }
        )
