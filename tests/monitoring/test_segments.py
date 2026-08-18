import polars as pl
import pytest

from riskprobe.monitoring.segments import diagnose_segments
from riskprobe.privacy import assert_safe_payload


def test_segment_risk_suppresses_small_groups_and_uses_explicit_tokens() -> None:
    frame = pl.DataFrame(
        {
            "institution": ["secret-bank"] * 3 + ["tiny-private-bank"],
            "target": [1, 1, 0, 0],
        }
    )

    findings = diagnose_segments(
        frame,
        segment_column="institution",
        target_column="target",
        min_group_size=3,
        token_namespace="dataset-1",
    )

    assert len(findings) == 1
    finding = findings[0]
    assert finding.segment_token is not None
    assert finding.segment_token.token.startswith("segment-")
    assert finding.metrics["sample_count"] == 3
    assert finding.metrics["target_rate"] == pytest.approx(2 / 3)
    assert finding.metrics["lift"] == pytest.approx(4 / 3)
    assert finding.metrics["population_share"] == pytest.approx(0.75)
    dumped = finding.model_dump(mode="json")
    assert "secret-bank" not in str(dumped)
    assert "tiny-private-bank" not in str(dumped)
    assert_safe_payload(dumped)


def test_segment_at_minimum_size_is_included_and_order_is_deterministic() -> None:
    frame = pl.DataFrame(
        {
            "segment": ["z-private"] * 2 + ["a-private"] * 2,
            "target": [1, 0, 1, 1],
        }
    )

    first = diagnose_segments(
        frame,
        segment_column="segment",
        target_column="target",
        min_group_size=2,
        token_namespace="dataset-1",
    )
    second = diagnose_segments(
        frame.reverse(),
        segment_column="segment",
        target_column="target",
        min_group_size=2,
        token_namespace="dataset-1",
    )

    assert first == second
    assert tuple(item.finding_id for item in first) == tuple(
        sorted(item.finding_id for item in first)
    )


def test_zero_base_target_rate_returns_finite_zero_lift() -> None:
    frame = pl.DataFrame({"segment": ["private"] * 3, "target": [0, 0, 0]})

    finding = diagnose_segments(
        frame,
        segment_column="segment",
        target_column="target",
        min_group_size=3,
    )[0]

    assert finding.metrics["target_rate"] == 0.0
    assert finding.metrics["lift"] == 0.0
