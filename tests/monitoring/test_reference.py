import base64
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
    assert reference_fixture["profile"].dataset_id not in payload
    assert reference_fixture["evidence_cards"][0].rule.rule_id not in payload
    assert "bank_north" not in payload
    assert reference_fixture["privacy_key"].decode() not in payload
    assert snapshot.row_count > 0
    assert snapshot.features[0].histogram_counts
    assert snapshot.dataset_id.startswith("dataset_")
    assert all(rule.rule_id.startswith("rule_") for rule in snapshot.rules)
    assert all(key.startswith("segment_") for key in snapshot.segment_counts)


def test_reference_tokens_are_keyed_domain_separated_and_full_length(reference_fixture) -> None:
    privacy_key = reference_fixture["privacy_key"]
    profile = replace(reference_fixture["profile"], dataset_id="same-private-value")
    evidence_card = reference_fixture["evidence_cards"][0]
    matching_rule = evidence_card.model_copy(
        update={"rule": evidence_card.rule.model_copy(update={"rule_id": "same-private-value"})}
    )
    fixture = dict(reference_fixture, profile=profile, evidence_cards=(matching_rule,))
    first = build_reference_snapshot(**fixture)
    second = build_reference_snapshot(**fixture)
    expected_dataset = "dataset_" + hmac.new(
        privacy_key,
        b"riskprobe-monitoring-dataset-v1:\x00same-private-value",
        hashlib.sha256,
    ).hexdigest()
    expected_rule = "rule_" + hmac.new(
        privacy_key,
        b"riskprobe-monitoring-rule-v1:\x00same-private-value",
        hashlib.sha256,
    ).hexdigest()
    legacy_segment = "segment_" + hashlib.sha256(
        b"riskprobe-monitoring-segment-v1:bank_north"
    ).hexdigest()[:16]
    alternate = build_reference_snapshot(
        **dict(reference_fixture, privacy_key=b"independent-test-privacy-key")
    )

    assert first == second
    assert first.dataset_id == expected_dataset
    assert first.rules[0].rule_id == expected_rule
    assert first.dataset_id.removeprefix("dataset_") != first.rules[0].rule_id.removeprefix("rule_")
    assert all(len(token.removeprefix("segment_")) == 64 for token in first.segment_counts)
    assert legacy_segment not in first.segment_counts
    assert first.segment_counts != alternate.segment_counts


def test_reference_snapshot_never_serializes_encoded_path_or_entity_identifiers(
    reference_fixture,
) -> None:
    encoded_path = "/private/customer.parquet".encode().hex()
    encoded_entity = base64.b64encode(b"user_0001").decode()
    profile = replace(reference_fixture["profile"], dataset_id=encoded_path)
    evidence_card = reference_fixture["evidence_cards"][0]
    encoded_rule = evidence_card.model_copy(
        update={"rule": evidence_card.rule.model_copy(update={"rule_id": encoded_entity})}
    )

    snapshot = build_reference_snapshot(
        **dict(reference_fixture, profile=profile, evidence_cards=(encoded_rule,))
    )
    payload = snapshot.model_dump_json()

    assert encoded_path not in payload
    assert encoded_entity not in payload
    assert "/private/customer.parquet" not in payload
    assert "user_0001" not in payload
    assert snapshot.dataset_id.startswith("dataset_")
    assert snapshot.rules[0].rule_id.startswith("rule_")


@pytest.mark.parametrize("privacy_key", [None, b""], ids=["none", "empty"])
def test_reference_snapshot_requires_non_empty_bytes_privacy_key(
    reference_fixture, privacy_key: bytes | None
) -> None:
    with pytest.raises(ValueError, match="privacy key must be non-empty bytes"):
        build_reference_snapshot(**dict(reference_fixture, privacy_key=privacy_key))


def test_reference_snapshot_requires_privacy_key_keyword_only(reference_fixture) -> None:
    fixture = dict(reference_fixture)
    fixture.pop("privacy_key")

    with pytest.raises(TypeError, match="privacy_key"):
        build_reference_snapshot(**fixture)


@pytest.mark.parametrize(
    "token_namespace",
    ["", "bad namespace", "tenant/path", "Namespace-v1"],
    ids=["empty", "space", "path", "uppercase"],
)
def test_reference_snapshot_requires_a_strict_safe_token_namespace(
    reference_fixture, token_namespace: str
) -> None:
    with pytest.raises(ValueError, match="token namespace must be a strict safe ID"):
        build_reference_snapshot(**dict(reference_fixture, token_namespace=token_namespace))


def test_reference_snapshot_records_namespace_and_fails_closed_for_mismatch(reference_fixture) -> None:
    first = build_reference_snapshot(**reference_fixture)
    second = build_reference_snapshot(
        **dict(reference_fixture, token_namespace="riskprobe-prod-v2")
    )

    assert first.token_namespace == reference_fixture["token_namespace"]
    assert first.assert_comparable_token_namespace(first) is None
    with pytest.raises(ValueError, match="token namespaces do not match"):
        first.assert_comparable_token_namespace(second)


def test_reference_snapshot_rejects_actual_segment_token_collisions(
    monkeypatch: pytest.MonkeyPatch, reference_fixture
) -> None:
    class CollidingDigest:
        def hexdigest(self) -> str:
            return "a" * 64

    profile = replace(reference_fixture["profile"], segment_counts={"north": 1, "south": 2})
    monkeypatch.setattr("riskprobe.monitoring.reference.hmac.new", lambda *args: CollidingDigest())

    with pytest.raises(ValueError, match="token collision"):
        build_reference_snapshot(**dict(reference_fixture, profile=profile))


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

    assert snapshot.rules[0].rule_id.startswith("rule_")
    assert snapshot.rules[0].coverage == pytest.approx(0.02)
    assert snapshot.rules[0].bad_rate == pytest.approx(0.3)
    assert snapshot.rules[0].lift == pytest.approx(3.0)


def test_reference_rejects_duplicate_rule_ids(reference_fixture) -> None:
    duplicate = reference_fixture["evidence_cards"][0]

    with pytest.raises(ValueError, match="duplicate rule identifier"):
        build_reference_snapshot(**dict(reference_fixture, evidence_cards=(duplicate, duplicate)))


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
