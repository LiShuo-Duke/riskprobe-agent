from pathlib import Path

import polars as pl
import pytest
import yaml

from riskprobe.config import ProjectConfig
from riskprobe.io.parquet import ParquetDataset
from riskprobe.profiling import profile_dataset
from riskprobe.registry import (
    DatasetAlreadyRegisteredError,
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
def test_registry_rejects_path_instead_of_dataset_id(tmp_path, synthetic_config) -> None:
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(yaml.safe_dump(synthetic_config.model_dump(mode="json")), encoding="utf-8")
    registry_path = tmp_path / "datasets.yaml"
    registry_path.write_text(
        yaml.safe_dump({"datasets": {"synthetic_demo": {"config": str(config_path)}}}),
        encoding="utf-8",
    )
    registry = DatasetRegistry.from_yaml(registry_path)

    with pytest.raises(DatasetNotRegisteredError):
        registry.get_config("/tmp/company.parquet")

    assert registry.get_config("synthetic_demo").dataset.id == synthetic_config.dataset.id


def _write_registry(tmp_path, config_path: str) -> DatasetRegistry:
    registry_path = tmp_path / "datasets.yaml"
    registry_path.write_text(
        yaml.safe_dump({"datasets": {"synthetic_demo": {"config": config_path}}}),
        encoding="utf-8",
    )
    return DatasetRegistry.from_yaml(registry_path)


def test_register_local_config_adds_dataset_without_writing_registry(
    tmp_path, synthetic_config
) -> None:
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(
        yaml.safe_dump(synthetic_config.model_dump(mode="json")), encoding="utf-8"
    )
    registry = _write_registry(tmp_path, str(config_path))
    registry_text = (tmp_path / "datasets.yaml").read_text(encoding="utf-8")

    registered = registry.register_local_config(
        "local_demo", config_path, (tmp_path.resolve(),)
    )

    assert registered.get_config("local_demo").dataset.id == synthetic_config.dataset.id
    assert (tmp_path / "datasets.yaml").read_text(encoding="utf-8") == registry_text
    with pytest.raises(DatasetNotRegisteredError):
        registry.get_config("local_demo")


def test_register_local_config_rejects_config_outside_allowed_root(
    tmp_path, synthetic_config
) -> None:
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(
        yaml.safe_dump(synthetic_config.model_dump(mode="json")), encoding="utf-8"
    )
    registry = _write_registry(tmp_path, str(config_path))

    with pytest.raises(ValueError, match="allowed local data roots"):
        registry.register_local_config(
            "local_demo", config_path, ((tmp_path / "allowed").resolve(),)
        )


def test_register_local_config_rejects_dataset_outside_allowed_root(
    tmp_path, synthetic_config
) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    config_path = allowed_root / "synthetic.yaml"
    payload = synthetic_config.model_dump(mode="json")
    payload["dataset"]["path"] = str(synthetic_config.dataset.path)
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    registry = _write_registry(tmp_path, str(config_path))

    with pytest.raises(ValueError, match="allowed local data roots"):
        registry.register_local_config("local_demo", config_path, (allowed_root,))


def test_register_local_config_rejects_duplicate_dataset_id(tmp_path, synthetic_config) -> None:
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(
        yaml.safe_dump(synthetic_config.model_dump(mode="json")), encoding="utf-8"
    )
    registry = _write_registry(tmp_path, str(config_path))

    with pytest.raises(DatasetAlreadyRegisteredError):
        registry.register_local_config("synthetic_demo", config_path, (tmp_path,))


def test_register_local_config_rejects_empty_allowed_roots(tmp_path, synthetic_config) -> None:
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(
        yaml.safe_dump(synthetic_config.model_dump(mode="json")), encoding="utf-8"
    )
    registry = _write_registry(tmp_path, str(config_path))

    with pytest.raises(ValueError, match="allowed local data roots"):
        registry.register_local_config("local_demo", config_path, ())


def test_register_local_parquet_builds_read_only_config_without_snapshot(
    tmp_path, synthetic_config
) -> None:
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(
        yaml.safe_dump(synthetic_config.model_dump(mode="json")), encoding="utf-8"
    )
    registry = _write_registry(tmp_path, str(config_path))
    registered = registry.register_local_parquet(
        "local_parquet",
        synthetic_config.dataset.path,
        entity_column="entity_id",
        target_column="target",
        segment_column="institution",
        snapshot_column=None,
        allowed_roots=(tmp_path,),
    )

    config = registered.get_config("local_parquet")
    assert config.dataset.path == synthetic_config.dataset.path.resolve()
    assert config.dataset.read_only is True
    assert config.columns.entity == "entity_id"
    assert config.columns.target == "target"
    assert config.columns.segment == "institution"
    assert config.columns.snapshot == "entity_id"
    assert config.time_validation_enabled is False
    assert config.privacy.expose_segment_values is True
    schema_names = ParquetDataset(config.dataset.path).schema().names()
    feature_names = config.features.select_columns(
        schema_names,
        (config.columns.entity, config.columns.snapshot, config.columns.segment, config.columns.target),
    )
    assert feature_names
    assert set(feature_names).isdisjoint(
        {config.columns.entity, config.columns.segment, config.columns.target}
    )


def test_register_local_parquet_rejects_path_outside_allowed_root(
    tmp_path, synthetic_config
) -> None:
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(
        yaml.safe_dump(synthetic_config.model_dump(mode="json")), encoding="utf-8"
    )
    registry = _write_registry(tmp_path, str(config_path))
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()

    with pytest.raises(ValueError, match="allowed local data roots"):
        registry.register_local_parquet(
            "local_parquet",
            synthetic_config.dataset.path,
            entity_column="entity_id",
            target_column="target",
            segment_column="institution",
            snapshot_column=None,
            allowed_roots=(allowed_root,),
        )


def test_register_local_parquet_requires_explicit_role_columns(
    tmp_path, synthetic_config
) -> None:
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(
        yaml.safe_dump(synthetic_config.model_dump(mode="json")), encoding="utf-8"
    )
    registry = _write_registry(tmp_path, str(config_path))

    with pytest.raises(ValueError, match="schema"):
        registry.register_local_parquet(
            "local_parquet",
            synthetic_config.dataset.path,
            entity_column="entity_id",
            target_column="missing_target",
            segment_column="institution",
            snapshot_column=None,
            allowed_roots=(tmp_path,),
        )


def test_register_local_parquet_keeps_exact_feature_columns(tmp_path, synthetic_config) -> None:
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(
        yaml.safe_dump(synthetic_config.model_dump(mode="json")), encoding="utf-8"
    )
    registry = _write_registry(tmp_path, str(config_path))
    registered = registry.register_local_parquet(
        "exact_features",
        synthetic_config.dataset.path,
        entity_column="entity_id",
        target_column="target",
        segment_column="institution",
        snapshot_column=None,
        feature_columns=["order_cnt_7d", "order_amount_30d"],
        allowed_roots=(tmp_path,),
    )

    config = registered.get_config("exact_features")
    assert config.features.exact_columns == (
        "order_cnt_7d",
        "order_amount_30d",
    )
    assert config.features.select_columns(
        ["entity_id", "order_cnt_7d", "order_amount_30d", "order_cnt_30d"],
        ("entity_id", "entity_id", "institution", "target"),
    ) == ["order_amount_30d", "order_cnt_7d"]


@pytest.mark.parametrize(
    "feature_columns, message",
    [
        ([], "feature columns"),
        ((), "feature columns"),
        (["missing_feature"], "feature columns"),
        (["target"], "role"),
        (["order_cnt_7d", "order_cnt_7d"], "duplicate"),
        (["institution"], "numeric"),
    ],
)
def test_register_local_parquet_rejects_invalid_exact_features(
    tmp_path, synthetic_config, feature_columns, message
) -> None:
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(
        yaml.safe_dump(synthetic_config.model_dump(mode="json")), encoding="utf-8"
    )
    registry = _write_registry(tmp_path, str(config_path))

    with pytest.raises(ValueError, match=message):
        registry.register_local_parquet(
            "invalid_features",
            synthetic_config.dataset.path,
            entity_column="entity_id",
            target_column="target",
            segment_column="institution",
            snapshot_column=None,
            feature_columns=feature_columns,
            allowed_roots=(tmp_path,),
        )


@pytest.mark.parametrize(
    "roles",
    [
        {"entity_column": "target"},
        {"target_column": "institution"},
        {"segment_column": "entity_id"},
        {"snapshot_column": "institution"},
    ],
)
def test_register_local_parquet_rejects_duplicate_real_roles(
    tmp_path, synthetic_config, roles
) -> None:
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(
        yaml.safe_dump(synthetic_config.model_dump(mode="json")), encoding="utf-8"
    )
    registry = _write_registry(tmp_path, str(config_path))

    with pytest.raises(ValueError, match="role"):
        registry.register_local_parquet(
            "duplicate_roles",
            synthetic_config.dataset.path,
            entity_column=roles.get("entity_column", "entity_id"),
            target_column=roles.get("target_column", "target"),
            segment_column=roles.get("segment_column", "institution"),
            snapshot_column=roles.get("snapshot_column", "snapshot_date"),
            allowed_roots=(tmp_path,),
        )


def test_register_local_parquet_legacy_numeric_selection_excludes_prefix_collision(
    tmp_path, synthetic_config
) -> None:
    parquet_path = tmp_path / "age-columns.parquet"
    pl.DataFrame(
        {
            "entity_id": ["e1", "e2", "e3", "e4"],
            "target": [0, 1, 0, 1],
            "institution": ["a", "a", "b", "b"],
            "age": [20, 30, 40, 50],
            "age_label": ["young", "adult", "adult", "senior"],
        }
    ).write_parquet(parquet_path)
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(
        yaml.safe_dump(synthetic_config.model_dump(mode="json")), encoding="utf-8"
    )
    registry = _write_registry(tmp_path, str(config_path))

    registered = registry.register_local_parquet(
        "age_columns",
        parquet_path,
        entity_column="entity_id",
        target_column="target",
        segment_column="institution",
        snapshot_column=None,
        allowed_roots=(tmp_path,),
    )

    config = registered.get_config("age_columns")
    dataset = ParquetDataset(config.dataset.path)
    roles = (
        config.columns.entity,
        config.columns.snapshot,
        config.columns.segment,
        config.columns.target,
    )
    selected_features = config.features.select_columns(dataset.schema().names(), roles)
    profile = profile_dataset(dataset, config)
    assert selected_features == ["age"]
    assert profile.feature_count == 1
    assert "age_label" not in selected_features


def test_register_local_parquet_rejects_unreadable_file_without_path(
    tmp_path, synthetic_config
) -> None:
    broken_path = tmp_path / "broken.parquet"
    broken_path.write_bytes(b"not a parquet file")
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(
        yaml.safe_dump(synthetic_config.model_dump(mode="json")), encoding="utf-8"
    )
    registry = _write_registry(tmp_path, str(config_path))

    with pytest.raises(ValueError) as error:
        registry.register_local_parquet(
            "broken_parquet",
            broken_path,
            entity_column="entity_id",
            target_column="target",
            segment_column="institution",
            snapshot_column=None,
            allowed_roots=(tmp_path,),
        )

    assert str(error.value) == "local dataset Parquet file is not readable"
    assert str(broken_path) not in str(error.value)


def test_register_local_config_rejects_unreadable_parquet_without_path(
    tmp_path, synthetic_config
) -> None:
    broken_path = tmp_path / "broken.parquet"
    broken_path.write_bytes(b"not a parquet file")
    payload = synthetic_config.model_dump(mode="json")
    payload["dataset"]["path"] = str(broken_path)
    config_path = tmp_path / "broken.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    registry = _write_registry(tmp_path, str(config_path))

    with pytest.raises(ValueError) as error:
        registry.register_local_config("broken_config", config_path, (tmp_path,))

    assert str(error.value) == "local dataset Parquet file is not readable"
    assert str(broken_path) not in str(error.value)
