import yaml
import pytest

from riskprobe.registry import DatasetNotRegisteredError, DatasetRegistry


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
