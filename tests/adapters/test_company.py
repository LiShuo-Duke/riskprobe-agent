from datetime import date

import polars as pl

from riskprobe.adapters.company import preflight_company_dataset
from riskprobe.config import ProjectConfig


def test_preflight_does_not_modify_source(tmp_path) -> None:
    path = tmp_path / "company.parquet"
    pl.DataFrame(
        {
            "anonymous_id": [f"u{index}" for index in range(200)],
            "cutoff_date": [date(2026, 1, 1)] * 200,
            "org_code": ["A"] * 100 + ["B"] * 100,
            "bad_label": [0, 1] * 100,
            "ord_x_cnt_30d": list(range(200)),
            "brw_x_pv_30d": list(range(200, 400)),
        }
    ).write_parquet(path)
    config = ProjectConfig.model_validate(
        {
            "dataset": {"id": "company_test", "path": path},
            "columns": {
                "entity": "anonymous_id",
                "snapshot": "cutoff_date",
                "segment": "org_code",
                "target": "bad_label",
            },
            "target": {"positive_value": 1, "positive_meaning": "bad_debt"},
            "snapshot": {"meaning": "customer_specified_feature_cutoff"},
            "features": {
                "families": {"order": ["ord_x_"], "browse": ["brw_x_"]}
            },
            "segment_display_name": "institution",
        }
    )
    before = path.stat().st_mtime_ns

    result = preflight_company_dataset(config)

    after = path.stat().st_mtime_ns
    assert before == after
    assert result.feature_family_counts["order"] > 0
    assert result.feature_family_counts["browse"] > 0
    assert result.batch_count == 1
    assert result.metadata_grade == "B"
    assert result.row_count == 200
    assert result.segment_count == 2
