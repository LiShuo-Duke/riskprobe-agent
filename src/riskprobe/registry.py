"""Private configuration registry addressed only by public dataset IDs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from types import MappingProxyType

import yaml

from riskprobe.config import ProjectConfig


_DATASET_ID = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")
_INVALID_REGISTRY = "dataset registry is invalid"


class DatasetNotRegisteredError(ValueError):
    """Raised without echoing an untrusted dataset selector."""


@dataclass(frozen=True, slots=True, repr=False)
class DatasetHandle:
    """Internal handle passed to trusted handlers, never serialized as a tool response."""

    dataset_id: str
    _config: ProjectConfig = field(repr=False)

    @property
    def config(self) -> ProjectConfig:
        return self._config

    def __repr__(self) -> str:
        return f"DatasetHandle(dataset_id={self.dataset_id!r})"


@dataclass(frozen=True, slots=True, repr=False)
class _RegistryEntry:
    config: ProjectConfig = field(repr=False)
    source_path: Path | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True, repr=False)
class DatasetRegistry:
    """An immutable startup snapshot of allowlisted dataset configurations."""

    _entries: Mapping[str, _RegistryEntry] = field(repr=False)

    @classmethod
    def from_mapping(
        cls,
        entries: Mapping[str, ProjectConfig | Path],
    ) -> DatasetRegistry:
        if not isinstance(entries, Mapping):
            raise TypeError("entries must be a mapping")
        normalized: dict[str, _RegistryEntry] = {}
        for dataset_id, source in entries.items():
            if not isinstance(dataset_id, str) or _DATASET_ID.fullmatch(dataset_id) is None:
                raise ValueError(_INVALID_REGISTRY)
            if isinstance(source, ProjectConfig):
                normalized[dataset_id] = _RegistryEntry(config=source)
                continue
            if type(source) is Path or isinstance(source, Path):
                normalized[dataset_id] = _entry_from_path(source)
                continue
            raise ValueError(_INVALID_REGISTRY)
        return cls(_entries=MappingProxyType(normalized))

    @classmethod
    def from_yaml(cls, path: Path) -> DatasetRegistry:
        registry_path = Path(path)
        try:
            payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or set(payload) != {"datasets"}:
                raise ValueError(_INVALID_REGISTRY)
            datasets = payload["datasets"]
            if not isinstance(datasets, dict):
                raise ValueError(_INVALID_REGISTRY)
            sources: dict[str, Path] = {}
            for dataset_id, item in datasets.items():
                if (
                    not isinstance(dataset_id, str)
                    or _DATASET_ID.fullmatch(dataset_id) is None
                    or not isinstance(item, dict)
                    or set(item) != {"config"}
                    or not isinstance(item["config"], str)
                    or not item["config"]
                ):
                    raise ValueError(_INVALID_REGISTRY)
                config_path = Path(item["config"])
                if not config_path.is_absolute():
                    config_path = registry_path.resolve().parent / config_path
                sources[dataset_id] = config_path
            return cls.from_mapping(sources)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
            raise ValueError(_INVALID_REGISTRY) from error

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def resolve(self, dataset_id: str) -> DatasetHandle:
        if not isinstance(dataset_id, str) or _DATASET_ID.fullmatch(dataset_id) is None:
            raise DatasetNotRegisteredError("dataset ID is not registered")
        try:
            entry = self._entries[dataset_id]
        except KeyError as error:
            raise DatasetNotRegisteredError("dataset ID is not registered") from error
        return DatasetHandle(dataset_id=dataset_id, _config=entry.config)

    def get_config(self, dataset_id: str) -> ProjectConfig:
        """Compatibility helper; tool-facing code should use ``resolve``."""

        return self.resolve(dataset_id).config

    def __repr__(self) -> str:
        return f"DatasetRegistry(dataset_ids={self.dataset_ids!r})"


def _entry_from_path(path: Path) -> _RegistryEntry:
    source_path = Path(path).resolve()
    try:
        config = ProjectConfig.from_yaml(source_path)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError(_INVALID_REGISTRY) from error
    dataset_path = config.dataset.path
    if not dataset_path.is_absolute() and not PureWindowsPath(str(dataset_path)).is_absolute():
        config = config.model_copy(
            update={
                "dataset": config.dataset.model_copy(
                    update={"path": (source_path.parent / dataset_path).resolve()}
                )
            }
        )
    return _RegistryEntry(config=config, source_path=source_path)
