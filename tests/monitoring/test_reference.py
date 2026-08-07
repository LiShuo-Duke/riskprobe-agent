import polars as pl
import pytest
from pydantic import ValidationError

from riskprobe.monitoring.models import Alert
from riskprobe.monitoring.reference import build_reference_snapshot


def test_reference_snapshot_contains_aggregates_not_entities(reference_fixture) -> None:
    snapshot = build_reference_snapshot(**reference_fixture)
    payload = snapshot.model_dump_json()

    assert "entity_id" not in payload
    assert "user_0001" not in payload
    assert "private-file-marker" not in payload
    assert "emb_00" not in payload
    assert str(reference_fixture["config"].dataset.path) not in payload
    assert "bank_north" not in payload
    assert snapshot.row_count > 0
    assert snapshot.features[0].histogram_counts
    assert all(key.startswith("segment_") for key in snapshot.segment_counts)


def test_reference_snapshot_has_deterministic_aggregates_and_identifier(reference_fixture) -> None:
    first = build_reference_snapshot(**reference_fixture)
    shuffled_fixture = dict(reference_fixture)
    shuffled_fixture["frame"] = reference_fixture["frame"].sample(
        fraction=1.0, shuffle=True, seed=9
    )
    second = build_reference_snapshot(**shuffled_fixture)

    assert first == second
    assert first.snapshot_id == second.snapshot_id
    assert len(first.snapshot_id) == 64
    assert first.created_at == "1970-01-01T00:00:00Z"


def test_reference_features_use_only_fail_closed_config_selection(reference_fixture) -> None:
    snapshot = build_reference_snapshot(**reference_fixture)
    feature_names = {feature.feature for feature in snapshot.features}

    assert feature_names == set(
        reference_fixture["config"].features.select_columns(
            reference_fixture["frame"].columns,
            (
                reference_fixture["config"].columns.entity,
                reference_fixture["config"].columns.snapshot,
                reference_fixture["config"].columns.segment,
                reference_fixture["config"].columns.target,
            ),
        )
    )
    assert "emb_00" not in feature_names
    assert "undeclared_text" not in feature_names


def test_reference_feature_histograms_use_quantile_boundaries(reference_fixture) -> None:
    snapshot = build_reference_snapshot(**reference_fixture)

    for feature in snapshot.features:
        assert len(feature.quantile_edges) == len(feature.histogram_counts) + 1
        assert sum(feature.histogram_counts) <= snapshot.row_count
        assert feature.missing_rate >= 0.0
        assert feature.zero_rate >= 0.0


def test_reference_snapshot_converts_rule_metrics_to_aggregates(reference_fixture) -> None:
    snapshot = build_reference_snapshot(**reference_fixture)

    assert snapshot.rules[0].rule_id == "rule_order_cancel"
    assert snapshot.rules[0].coverage == pytest.approx(0.02)
    assert snapshot.rules[0].bad_rate == pytest.approx(0.3)
    assert snapshot.rules[0].lift == pytest.approx(3.0)


def test_reference_snapshot_handles_all_missing_and_constant_features(reference_fixture) -> None:
    frame = reference_fixture["frame"].with_columns(
        [
            pl.lit(None, dtype=pl.Float64).alias("order_cnt_7d"),
            pl.lit(0.0).alias("order_cnt_30d"),
        ]
    )
    fixture = dict(reference_fixture, frame=frame)

    snapshot = build_reference_snapshot(**fixture)
    missing = next(feature for feature in snapshot.features if feature.feature == "order_cnt_7d")
    constant = next(feature for feature in snapshot.features if feature.feature == "order_cnt_30d")

    assert missing.missing_rate == 1.0
    assert missing.zero_rate == 0.0
    assert missing.quantile_edges == ()
    assert missing.histogram_counts == ()
    assert constant.missing_rate == 0.0
    assert constant.zero_rate == 1.0
    assert constant.quantile_edges == (0.0, 0.0)
    assert constant.histogram_counts == (snapshot.row_count,)


def test_alert_model_is_strict_and_immutable() -> None:
    alert = Alert(
        alert_id="alert-1",
        alert_type="population",
        severity="warning",
        scope="dataset",
        scope_value="synthetic_test",
        metric="row_count",
        reference_value=10,
        current_value=11,
        delta=0.1,
        evidence={"reference_rows": 10},
    )

    with pytest.raises(ValidationError):
        alert.severity = "critical"
    with pytest.raises(ValidationError):
        Alert.model_validate({**alert.model_dump(), "unexpected": "value"})
