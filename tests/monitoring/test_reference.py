from dataclasses import replace
import hashlib
import hmac

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
    assert reference_fixture["segment_token_key"].decode() not in payload
    assert snapshot.row_count > 0
    assert snapshot.features[0].histogram_counts
    assert all(key.startswith("segment_") for key in snapshot.segment_counts)


def test_reference_segment_tokens_are_keyed_stable_and_not_legacy_hashes(reference_fixture) -> None:
    token_key = reference_fixture["segment_token_key"]
    first = build_reference_snapshot(**reference_fixture)
    second = build_reference_snapshot(**reference_fixture)
    expected = "segment_" + hmac.new(
        token_key,
        b"riskprobe-monitoring-segment-v1:\x00bank_north",
        hashlib.sha256,
    ).hexdigest()[:16]
    legacy = "segment_" + hashlib.sha256(
        b"riskprobe-monitoring-segment-v1:bank_north"
    ).hexdigest()[:16]
    alternate = build_reference_snapshot(
        **dict(reference_fixture, segment_token_key=b"independent-test-segment-key")
    )

    assert first.segment_counts == second.segment_counts
    assert expected in first.segment_counts
    assert legacy not in first.segment_counts
    assert first.segment_counts != alternate.segment_counts


def test_reference_omits_segment_aggregates_without_a_key(reference_fixture) -> None:
    fixture_without_key = dict(reference_fixture)
    fixture_without_key.pop("segment_token_key")

    snapshot = build_reference_snapshot(**fixture_without_key)

    assert snapshot.segment_counts == {}


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


@pytest.mark.parametrize(
    "series",
    [
        lambda rows: pl.Series("order_cnt_7d", [True] * rows),
        lambda rows: pl.Series("order_cnt_7d", ["category"] * rows),
        lambda rows: pl.Series("order_cnt_7d", ["category"] * rows).cast(pl.Categorical),
        lambda rows: pl.Series("order_cnt_7d", [object()] * rows, dtype=pl.Object),
    ],
    ids=["boolean", "string", "categorical", "object"],
)
def test_reference_rejects_selected_non_numeric_features(
    reference_fixture,
    series: object,
) -> None:
    frame = reference_fixture["frame"].with_columns(series(reference_fixture["frame"].height))

    with pytest.raises(
        ValueError,
        match="selected feature 'order_cnt_7d' has unsupported dtype; numeric features are required",
    ):
        build_reference_snapshot(**dict(reference_fixture, frame=frame))


def test_reference_feature_histograms_use_quantile_boundaries(reference_fixture) -> None:
    snapshot = build_reference_snapshot(**reference_fixture)

    for feature in snapshot.features:
        assert len(feature.quantile_edges) == len(feature.histogram_counts) + 1
        assert sum(feature.histogram_counts) <= snapshot.row_count
        assert feature.missing_rate >= 0.0
        assert feature.zero_rate >= 0.0


def test_reference_treats_nan_and_infinities_as_missing_not_distribution_values(
    reference_fixture,
) -> None:
    rows = reference_fixture["frame"].height
    values = [0.0, float("nan"), float("inf"), float("-inf"), 2.0] + [None] * (rows - 5)
    frame = reference_fixture["frame"].with_columns(pl.Series("order_cnt_7d", values))

    snapshot = build_reference_snapshot(**dict(reference_fixture, frame=frame))
    feature = next(feature for feature in snapshot.features if feature.feature == "order_cnt_7d")

    assert feature.missing_rate == pytest.approx((rows - 2) / rows)
    assert feature.zero_rate == pytest.approx(0.5)
    assert feature.quantile_edges == (0.0, 0.5, 1.0, 1.5, 2.0)
    assert feature.histogram_counts == (1, 0, 0, 1)


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


def test_reference_snapshot_handles_empty_frames(reference_fixture) -> None:
    empty_profile = replace(
        reference_fixture["profile"],
        row_count=0,
        positive_rate=None,
        segment_counts={},
    )
    snapshot = build_reference_snapshot(
        **dict(reference_fixture, frame=reference_fixture["frame"].head(0), profile=empty_profile)
    )

    assert snapshot.row_count == 0
    assert snapshot.positive_rate == 0.0
    assert snapshot.segment_counts == {}
    assert all(feature.missing_rate == 0.0 for feature in snapshot.features)
    assert all(feature.histogram_counts == () for feature in snapshot.features)


def test_reference_snapshot_converts_rule_metrics_to_aggregates(reference_fixture) -> None:
    snapshot = build_reference_snapshot(**reference_fixture)

    assert snapshot.rules[0].rule_id == "6c1469285066"
    assert snapshot.rules[0].coverage == pytest.approx(0.02)
    assert snapshot.rules[0].bad_rate == pytest.approx(0.3)
    assert snapshot.rules[0].lift == pytest.approx(3.0)


def test_reference_rejects_duplicate_rule_ids(reference_fixture) -> None:
    duplicate = reference_fixture["evidence_cards"][0]

    with pytest.raises(ValueError, match="duplicate rule identifier"):
        build_reference_snapshot(**dict(reference_fixture, evidence_cards=(duplicate, duplicate)))


@pytest.mark.parametrize(
    "unsafe_identifier",
    ["/private/customer.parquet", "https://example.test/dataset", "user_0001", "dataset\nlog"],
    ids=["path", "uri", "entity", "control-character"],
)
def test_reference_rejects_unsafe_dataset_identifiers(reference_fixture, unsafe_identifier: str) -> None:
    profile = replace(reference_fixture["profile"], dataset_id=unsafe_identifier)

    with pytest.raises(ValueError, match="dataset identifier must be an opaque safe ID"):
        build_reference_snapshot(**dict(reference_fixture, profile=profile))


@pytest.mark.parametrize(
    "unsafe_identifier",
    ["/private/rule", "mailto:user@example.test", "entity_0001", "rule\tlog"],
    ids=["path", "uri", "entity", "control-character"],
)
def test_reference_rejects_unsafe_rule_identifiers(reference_fixture, unsafe_identifier: str) -> None:
    evidence_card = reference_fixture["evidence_cards"][0]
    unsafe_rule = evidence_card.rule.model_copy(update={"rule_id": unsafe_identifier})
    unsafe_card = evidence_card.model_copy(update={"rule": unsafe_rule})

    with pytest.raises(ValueError, match="rule identifier must be an opaque safe ID"):
        build_reference_snapshot(**dict(reference_fixture, evidence_cards=(unsafe_card,)))


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
