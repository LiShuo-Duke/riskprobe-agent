from pathlib import Path

import pytest
from pydantic import ValidationError

from riskprobe.config import ProjectConfig


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
