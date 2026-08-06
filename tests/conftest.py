from pathlib import Path

import pytest

from riskprobe.config import ProjectConfig
from riskprobe.synthetic import generate_behavior_dataset


@pytest.fixture
def synthetic_config(tmp_path: Path) -> ProjectConfig:
    data_path = tmp_path / "synthetic.parquet"
    frame, _ = generate_behavior_dataset(5_000, seed=42)
    frame.write_parquet(data_path)
    return ProjectConfig.model_validate(
        {
            "dataset": {"id": "synthetic_test", "path": data_path},
            "columns": {
                "entity": "entity_id",
                "snapshot": "snapshot_date",
                "segment": "institution",
                "target": "target",
            },
            "target": {"positive_value": 1, "positive_meaning": "bad_debt"},
            "snapshot": {"meaning": "customer_specified_feature_cutoff"},
            "features": {"families": {"order": ["order_"], "browse": ["browse_"]}},
            "segment_display_name": "institution",
            "time_validation_enabled": True,
        }
    )
