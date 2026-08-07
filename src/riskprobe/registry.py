"""Startup-loaded dataset ID allowlist for local RiskProbe tools."""

import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml

from riskprobe.config import ProjectConfig


_DATASET_ID = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")


class DatasetNotRegisteredError(ValueError):
    """Raised when a tool request does not name an allowlisted dataset ID."""


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

    def get_config(self, dataset_id: str) -> ProjectConfig:
        if not isinstance(dataset_id, str) or not _DATASET_ID.fullmatch(dataset_id):
            raise DatasetNotRegisteredError("dataset requests must use a registered dataset ID")
        try:
            return self._configs[dataset_id]
        except KeyError as error:
            raise DatasetNotRegisteredError("dataset ID is not registered") from error
