from pathlib import Path

import pytest
import yaml

from riskprobe.config import ProjectConfig
from riskprobe.registry import (
    DatasetNotRegisteredError,
    DatasetRegistry,
)


def _write_config(path: Path, *, dataset_path: str = "data/demo.parquet") -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"id": "private-source", "path": dataset_path},
                "columns": {
                    "entity": "entity_id",
                    "snapshot": "snapshot_date",
                    "segment": "institution",
                    "target": "target",
                },
                "target": {
                    "positive_value": 1,
                    "positive_meaning": "bad_debt",
                },
                "snapshot": {"meaning": "customer_specified_feature_cutoff"},
                "features": {"families": {"orders": ["order_"]}},
            }
        ),
        encoding="utf-8",
    )


def test_in_memory_registry_resolves_only_public_dataset_ids(
    synthetic_config: ProjectConfig,
) -> None:
    registry = DatasetRegistry.from_mapping({"synthetic_demo": synthetic_config})

    handle = registry.resolve("synthetic_demo")

    assert handle.dataset_id == "synthetic_demo"
    assert handle.config is synthetic_config
    assert registry.get_config("synthetic_demo") is synthetic_config
    assert str(synthetic_config.dataset.path) not in repr(handle)


@pytest.mark.parametrize(
    "dataset_id",
    [
        "/tmp/company.parquet",
        "../company",
        "file:///tmp/company.parquet",
        "unknown_demo",
        "UPPERCASE",
        "ab",
    ],
)
def test_registry_rejects_paths_invalid_ids_and_unknown_ids_without_echoing_input(
    synthetic_config: ProjectConfig,
    dataset_id: str,
) -> None:
    registry = DatasetRegistry.from_mapping({"synthetic_demo": synthetic_config})

    with pytest.raises(DatasetNotRegisteredError) as exc_info:
        registry.resolve(dataset_id)

    assert str(exc_info.value) == "dataset ID is not registered"
    assert dataset_id not in str(exc_info.value)


def test_yaml_registry_keeps_config_path_private_and_resolves_relative_paths(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "private-configs"
    config_dir.mkdir()
    config_path = config_dir / "demo.yaml"
    _write_config(config_path)
    registry_path = tmp_path / "datasets.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {"datasets": {"public_demo": {"config": "private-configs/demo.yaml"}}}
        ),
        encoding="utf-8",
    )

    registry = DatasetRegistry.from_yaml(registry_path)
    handle = registry.resolve("public_demo")

    assert handle.config.dataset.path == (config_dir / "data/demo.parquet").resolve()
    assert not hasattr(handle, "config_path")
    assert str(config_path) not in repr(registry)
    assert str(config_path) not in repr(handle)


def test_path_backed_in_memory_entry_is_loaded_without_exposing_its_path(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "private.yaml"
    _write_config(config_path, dataset_path="demo.parquet")

    registry = DatasetRegistry.from_mapping({"public_demo": config_path})

    assert registry.resolve("public_demo").config.dataset.path == (
        tmp_path / "demo.parquet"
    ).resolve()
    assert str(config_path) not in repr(registry)


def test_registry_is_an_immutable_snapshot_of_input_mapping(
    synthetic_config: ProjectConfig,
) -> None:
    entries = {"synthetic_demo": synthetic_config}
    registry = DatasetRegistry.from_mapping(entries)
    entries.clear()

    assert registry.resolve("synthetic_demo").config is synthetic_config


def test_yaml_registry_is_strict_and_sanitizes_private_config_errors(
    tmp_path: Path,
) -> None:
    private_path = tmp_path / "missing-private.yaml"
    registry_path = tmp_path / "datasets.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {"datasets": {"public_demo": {"config": str(private_path), "extra": True}}}
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        DatasetRegistry.from_yaml(registry_path)

    assert str(exc_info.value) == "dataset registry is invalid"
    assert str(private_path) not in str(exc_info.value)
