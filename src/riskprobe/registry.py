"""Startup-loaded dataset ID allowlist for local RiskProbe tools."""

import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import polars as pl
import yaml

from riskprobe.config import ProjectConfig
from riskprobe.io.parquet import ParquetDataset


_DATASET_ID = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")


class DatasetNotRegisteredError(ValueError):
    """Raised when a tool request does not name an allowlisted dataset ID."""


class DatasetAlreadyRegisteredError(ValueError):
    """Raised when a session registration would replace an existing dataset ID."""


def _allowed_roots(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    if not roots:
        raise ValueError("allowed local data roots are required")
    resolved: list[Path] = []
    for root in roots:
        try:
            candidate = Path(root).resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ValueError("allowed local data roots must be existing directories") from error
        if not candidate.is_dir():
            raise ValueError("allowed local data roots must be existing directories")
        resolved.append(candidate)
    return tuple(dict.fromkeys(resolved))


def _resolve_under_roots(path: Path, roots: tuple[Path, ...], label: str) -> Path:
    try:
        resolved = Path(path).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"{label} must be an existing local file") from error
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise ValueError(f"{label} is outside the allowed local data roots")
    return resolved


@dataclass(frozen=True, slots=True)
class DatasetRegistry:
    _configs: Mapping[str, ProjectConfig]

    @classmethod
    def from_yaml(cls, path: Path) -> "DatasetRegistry":
        try:
            payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ValueError("dataset registry must be readable YAML") from error
        datasets = payload.get("datasets") if isinstance(payload, dict) else None
        if not isinstance(datasets, dict):
            raise ValueError("dataset registry must contain a datasets mapping")
        configs: dict[str, ProjectConfig] = {}
        registry_root = Path(path).resolve().parent
        for dataset_id, entry in datasets.items():
            if not isinstance(dataset_id, str) or not _DATASET_ID.fullmatch(dataset_id):
                raise ValueError("dataset registry has an invalid dataset ID")
            if not isinstance(entry, dict) or set(entry) != {"config"} or not isinstance(entry["config"], str):
                raise ValueError("dataset registry entries require only a config path")
            config_path = Path(entry["config"])
            if not config_path.is_absolute():
                config_path = registry_root / config_path
            configs[dataset_id] = ProjectConfig.from_yaml(config_path)
        return cls(_configs=MappingProxyType(configs))

    def register_local_config(
        self,
        dataset_id: str,
        config_path: Path,
        allowed_roots: tuple[Path, ...],
    ) -> "DatasetRegistry":
        if not isinstance(dataset_id, str) or not _DATASET_ID.fullmatch(dataset_id):
            raise DatasetNotRegisteredError("dataset requests must use a registered dataset ID")
        if dataset_id in self._configs:
            raise DatasetAlreadyRegisteredError("dataset ID is already registered")
        roots = _allowed_roots(allowed_roots)
        safe_config_path = _resolve_under_roots(Path(config_path), roots, "config path")
        try:
            config = ProjectConfig.from_yaml(safe_config_path)
        except (OSError, ValueError, yaml.YAMLError) as error:
            raise ValueError("local dataset config is invalid") from error

        dataset_path = config.dataset.path
        if not dataset_path.is_absolute():
            dataset_path = safe_config_path.parent / dataset_path
        safe_dataset_path = _resolve_under_roots(dataset_path, roots, "dataset path")
        if config.features.explicit_catalog is not None:
            _resolve_under_roots(config.features.explicit_catalog, roots, "feature catalog path")
        try:
            schema = ParquetDataset(safe_dataset_path).schema()
        except (OSError, ValueError, pl.exceptions.PolarsError) as error:
            raise ValueError("local dataset Parquet file is not readable") from error
        required_columns = {
            config.columns.entity,
            config.columns.snapshot,
            config.columns.segment,
            config.columns.target,
        }
        if not required_columns.issubset(schema.names()):
            raise ValueError("local dataset schema is incompatible with configured roles")

        normalized_config = config.model_copy(
            update={
                "dataset": config.dataset.model_copy(
                    update={"path": safe_dataset_path}
                )
            }
        )
        configs = dict(self._configs)
        configs[dataset_id] = normalized_config
        return DatasetRegistry(_configs=MappingProxyType(configs))

    def register_local_parquet(
        self,
        dataset_id: str,
        parquet_path: Path,
        *,
        entity_column: str,
        target_column: str,
        segment_column: str,
        snapshot_column: str | None,
        feature_columns: list[str] | tuple[str, ...] | None = None,
        allowed_roots: tuple[Path, ...],
    ) -> "DatasetRegistry":
        if not isinstance(dataset_id, str) or not _DATASET_ID.fullmatch(dataset_id):
            raise DatasetNotRegisteredError("dataset requests must use a registered dataset ID")
        if dataset_id in self._configs:
            raise DatasetAlreadyRegisteredError("dataset ID is already registered")

        role_values = (entity_column, target_column, segment_column)
        if any(not isinstance(column, str) or not column.strip() for column in role_values):
            raise ValueError("entity, target, and segment columns are required")
        normalized_entity, normalized_target, normalized_segment = tuple(
            column.strip() for column in role_values
        )
        if snapshot_column is not None and (
            not isinstance(snapshot_column, str) or not snapshot_column.strip()
        ):
            raise ValueError("snapshot column must be a non-empty string or null")
        normalized_snapshot = (
            snapshot_column.strip() if snapshot_column is not None else None
        )
        snapshot_role = normalized_snapshot or normalized_entity
        real_roles = (normalized_entity, normalized_target, normalized_segment)
        if normalized_snapshot is not None:
            real_roles += (normalized_snapshot,)
        if len(real_roles) != len(set(real_roles)):
            raise ValueError("explicit role columns must be distinct")

        explicit_features = feature_columns is not None
        if explicit_features:
            if not isinstance(feature_columns, (list, tuple)) or not feature_columns:
                raise ValueError("feature columns must be a non-empty list or tuple")
            if any(not isinstance(column, str) or not column.strip() for column in feature_columns):
                raise ValueError("feature columns must be non-empty strings")
            normalized_features = tuple(feature_columns)
            if len(normalized_features) != len(set(normalized_features)):
                raise ValueError("feature columns must not contain duplicates")
        else:
            normalized_features = ()

        roots = _allowed_roots(allowed_roots)
        safe_path = _resolve_under_roots(Path(parquet_path), roots, "parquet path")
        if safe_path.suffix.lower() != ".parquet":
            raise ValueError("parquet path must have a .parquet extension")
        try:
            schema = ParquetDataset(safe_path).schema()
        except (OSError, ValueError, pl.exceptions.PolarsError) as error:
            raise ValueError("local dataset Parquet file is not readable") from error

        role_columns = (normalized_entity, snapshot_role, normalized_segment, normalized_target)
        schema_names = set(schema.names())
        if not set(role_columns).issubset(schema_names):
            raise ValueError("local dataset schema is incompatible with explicit role columns")

        role_names = set(real_roles)
        if explicit_features:
            if not set(normalized_features).issubset(schema_names):
                raise ValueError("feature columns must exist in the local dataset schema")
            if any(not schema[name].is_numeric() for name in normalized_features):
                raise ValueError("feature columns must be numeric")
            if role_names.intersection(normalized_features):
                raise ValueError("feature columns must not overlap role columns")
            feature_names = normalized_features
        else:
            feature_names = tuple(
                name
                for name, dtype in schema.items()
                if name not in role_names and dtype.is_numeric()
            )
        if not feature_names:
            raise ValueError("local dataset must contain numeric feature columns")

        features = {"families": {"parquet_numeric": feature_names}, "exact_columns": feature_names}
        config = ProjectConfig.model_validate(
            {
                "dataset": {"id": dataset_id, "path": safe_path},
                "columns": {
                    "entity": normalized_entity,
                    "snapshot": snapshot_role,
                    "segment": normalized_segment,
                    "target": normalized_target,
                },
                "target": {
                    "positive_value": 1,
                    "positive_meaning": "bad_debt",
                },
                "snapshot": {"meaning": "public_relative_reference"},
                "features": features,
                "segment_display_name": "customer_segment",
                "time_validation_enabled": normalized_snapshot is not None,
                "privacy": {"expose_segment_values": True},
            }
        )
        configs = dict(self._configs)
        configs[dataset_id] = config
        return DatasetRegistry(_configs=MappingProxyType(configs))

    def get_config(self, dataset_id: str) -> ProjectConfig:
        if not isinstance(dataset_id, str) or not _DATASET_ID.fullmatch(dataset_id):
            raise DatasetNotRegisteredError("dataset requests must use a registered dataset ID")
        try:
            return self._configs[dataset_id]
        except KeyError as error:
            raise DatasetNotRegisteredError("dataset ID is not registered") from error
