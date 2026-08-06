from pathlib import Path

import pytest
from pydantic import ValidationError

from riskprobe.config import (
    DatasetConfig,
    DiscoveryConfig,
    FeatureFamilyConfig,
    ProjectConfig,
    TargetConfig,
)


def test_company_metadata_without_performance_window_is_grade_b(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  id: demo
  path: /tmp/demo.parquet
columns:
  entity: entity_id
  snapshot: snapshot_date
  segment: institution
  target: target
target:
  positive_value: 1
  positive_meaning: bad_debt
  performance_window_days: null
snapshot:
  meaning: customer_specified_feature_cutoff
features:
  families:
    order: [order_]
    browse: [browse_]
""".strip(),
        encoding="utf-8",
    )

    config = ProjectConfig.from_yaml(config_path)

    assert config.dataset.id == "demo"
    assert config.metadata_grade == "B"


def test_unknown_snapshot_semantics_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
dataset: {id: demo, path: /tmp/demo.parquet}
columns: {entity: id, snapshot: dt, segment: org, target: y}
target: {positive_value: 1, positive_meaning: bad_debt}
snapshot: {meaning: bad_debt_date}
features: {families: {order: [order_]}}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        ProjectConfig.from_yaml(config_path)


def test_strict_models_reject_coercible_values() -> None:
    with pytest.raises(ValidationError):
        DiscoveryConfig(max_single_rules="100")


@pytest.mark.parametrize(
    ("path", "read_only"),
    [
        (Path("/tmp/demo.csv"), True),
        (Path("s3://bucket/demo.parquet"), True),
        (Path("/tmp/demo.parquet"), False),
    ],
)
def test_dataset_requires_local_read_only_parquet(path: Path, read_only: bool) -> None:
    with pytest.raises(ValidationError):
        DatasetConfig(id="demo", path=path, read_only=read_only)


def test_positive_value_must_be_one() -> None:
    with pytest.raises(ValidationError):
        TargetConfig(positive_value=0, positive_meaning="bad_debt")


def test_random_seed_must_be_42() -> None:
    with pytest.raises(ValidationError):
        DiscoveryConfig(random_seed=7)


def test_feature_families_are_deeply_immutable() -> None:
    config = FeatureFamilyConfig(families={"order": ["order_"]})

    with pytest.raises(TypeError):
        config.families["browse"] = ["browse_"]
    with pytest.raises(AttributeError):
        config.families["order"].append("late_")
