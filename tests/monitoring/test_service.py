from pathlib import Path

import polars as pl

from riskprobe.config import ProjectConfig
from riskprobe.io.parquet import ParquetDataset
from riskprobe.monitoring.models import FindingSeverity
from riskprobe.monitoring.service import diagnose_dataset
from riskprobe.privacy import assert_safe_payload


def _config(path: Path) -> ProjectConfig:
    return ProjectConfig.model_validate(
        {
            "dataset": {"id": "diagnostic-fixture", "path": path},
            "columns": {
                "entity": "entity_id",
                "snapshot": "snapshot_date",
                "segment": "institution",
                "target": "target",
            },
            "target": {"positive_value": 1, "positive_meaning": "bad_debt"},
            "snapshot": {"meaning": "customer_specified_feature_cutoff"},
            "features": {"families": {"risk": ("risk_",)}},
            "validation": {
                "alpha": 0.05,
                "min_segment_consistency": 0.6,
                "max_lift_decay": 0.3,
                "bootstrap_rounds": 100,
                "min_group_size": 20,
            },
        }
    )


def test_diagnose_dataset_combines_findings_deterministically_and_safely(
    tmp_path: Path,
) -> None:
    path = tmp_path / "private-source.parquet"
    rows = 40
    frame = pl.DataFrame(
        {
            "entity_id": ["borrower-secret"] * 2 + [f"entity-{index:06d}" for index in range(38)],
            "snapshot_date": ["2024-01-01"] * 20 + ["2024-02-01"] * 20,
            "institution": ["secret-bank-a"] * 20 + ["secret-bank-b"] * 20,
            "target": [0] * 18 + [1] * 2 + [0] * 10 + [1] * 10,
            "risk_score": [0.0] * 20 + [1.0] * 20,
            "risk_constant": [7] * rows,
        }
    )
    frame.write_parquet(path)
    config = _config(path)
    dataset = ParquetDataset(path)

    first = diagnose_dataset(dataset, config)
    second = diagnose_dataset(dataset, config)

    assert first == second
    assert first.dataset_id == "diagnostic-fixture"
    assert first.metadata_grade == "B"
    assert first.findings
    severity_rank = {
        FindingSeverity.CRITICAL: 0,
        FindingSeverity.WARNING: 1,
        FindingSeverity.INFO: 2,
    }
    assert list(first.findings) == sorted(
        first.findings,
        key=lambda item: (
            severity_rank[item.severity],
            item.kind.value,
            item.code,
            item.feature or "",
            item.period or "",
            item.finding_id,
        ),
    )
    dumped = first.model_dump(mode="json")
    assert_safe_payload(dumped)
    rendered = str(dumped)
    assert "borrower-secret" not in rendered
    assert "secret-bank-a" not in rendered
    assert str(path) not in rendered
