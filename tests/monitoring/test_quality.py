import math

import polars as pl

from riskprobe.monitoring.quality import diagnose_quality
from riskprobe.privacy import assert_safe_payload


def _find(findings: tuple[object, ...], code: str, feature: str | None = None):
    return next(
        finding
        for finding in findings
        if finding.code == code and (feature is None or finding.feature == feature)
    )


def test_quality_detects_missing_and_constant_features_with_aggregate_metrics() -> None:
    frame = pl.DataFrame(
        {
            "entity_id": ["e1", "e2", "e3", "e4"],
            "snapshot": ["2024-01-01"] * 4,
            "missing_feature": [1.0, None, None, 4.0],
            "constant_feature": [7, 7, 7, 7],
        }
    )

    findings = diagnose_quality(
        frame,
        entity_column="entity_id",
        snapshot_column="snapshot",
        feature_columns=("missing_feature", "constant_feature"),
    )

    missing = _find(findings, "missing_values", "missing_feature")
    constant = _find(findings, "constant_feature", "constant_feature")
    assert missing.metrics == {"affected_count": 2, "affected_rate": 0.5}
    assert constant.metrics == {"affected_count": 4, "affected_rate": 1.0}
    for finding in findings:
        assert_safe_payload(finding.model_dump(mode="json"))
        assert "e1" not in str(finding.model_dump(mode="json"))


def test_quality_counts_only_extra_duplicate_entity_snapshot_rows() -> None:
    frame = pl.DataFrame(
        {
            "entity_id": ["private-a", "private-a", "private-a", "private-b"],
            "snapshot": ["2024-01-01", "2024-01-01", "2024-02-01", "2024-01-01"],
            "score": [1, 2, 3, 4],
        }
    )

    findings = diagnose_quality(
        frame,
        entity_column="entity_id",
        snapshot_column="snapshot",
        feature_columns=("score",),
    )

    duplicate = _find(findings, "duplicate_entity_snapshot")
    assert duplicate.metrics == {"affected_count": 1, "affected_rate": 0.25}
    assert "private-a" not in str(duplicate.model_dump(mode="json"))


def test_quality_detects_numeric_range_and_non_finite_values() -> None:
    frame = pl.DataFrame(
        {
            "entity_id": ["e1", "e2", "e3", "e4", "e5"],
            "snapshot": ["2024-01-01"] * 5,
            "score": [-1.0, 0.2, 1.2, math.inf, math.nan],
        }
    )

    findings = diagnose_quality(
        frame,
        entity_column="entity_id",
        snapshot_column="snapshot",
        feature_columns=("score",),
        numeric_ranges={"score": (0.0, 1.0)},
    )

    out_of_range = _find(findings, "numeric_out_of_range", "score")
    non_finite = _find(findings, "non_finite_values", "score")
    assert out_of_range.metrics == {"affected_count": 2, "affected_rate": 0.4}
    assert non_finite.metrics == {"affected_count": 2, "affected_rate": 0.4}


def test_quality_empty_frame_produces_no_nonfinite_metrics() -> None:
    frame = pl.DataFrame(
        schema={"entity_id": pl.String, "snapshot": pl.String, "score": pl.Float64}
    )

    findings = diagnose_quality(
        frame,
        entity_column="entity_id",
        snapshot_column="snapshot",
        feature_columns=("score",),
    )

    assert findings == ()
