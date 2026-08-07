from dataclasses import replace

import polars as pl
import pytest
from pydantic import ValidationError

from riskprobe.monitoring.models import Alert
from riskprobe.monitoring.reference import build_reference_snapshot


_PATH_LIKE_IDENTIFIERS = (
    "/private/customer-data.parquet",
    r"C:\\private\\customer-data.parquet",
    "file:///private/customer-data.parquet",
    "private/customer-data.parquet",
    r"..\\private\\customer-data.parquet",
    r"C:private\\customer-data.parquet",
)
_PATH_LIKE_IDENTIFIER_IDS = (
    "posix-path",
    "windows-path",
    "file-uri",
    "relative-posix-path",
    "relative-windows-path",
    "drive-relative-windows-path",
)
_REPEATEDLY_ENCODED_PATH_LIKE_IDENTIFIERS = (
    "%252Fprivate%252Fcustomer-data.parquet",
    "file%253A%252F%252Fprivate%252Fcustomer-data.parquet",
)
_REPEATEDLY_ENCODED_PATH_LIKE_IDENTIFIER_IDS = (
    "twice-encoded-posix-path",
    "twice-encoded-file-uri",
)


def test_reference_snapshot_contains_aggregates_not_entities(reference_fixture) -> None:
    snapshot = build_reference_snapshot(**reference_fixture)
    payload = snapshot.model_dump_json()

    assert "entity_id" not in payload
    assert "user_0001" not in payload
    assert "private-file-marker" not in payload
    assert "emb_00" not in payload
    assert str(reference_fixture["config"].dataset.path) not in payload
    assert snapshot.row_count > 0
    assert snapshot.features[0].histogram_counts


@pytest.mark.parametrize(
    "dataset_id",
    _PATH_LIKE_IDENTIFIERS,
    ids=_PATH_LIKE_IDENTIFIER_IDS,
)
def test_reference_snapshot_rejects_path_like_dataset_ids(
    reference_fixture,
    dataset_id: str,
) -> None:
    profile = replace(reference_fixture["profile"], dataset_id=dataset_id)

    with pytest.raises(ValueError, match="path-like identifier"):
        build_reference_snapshot(**dict(reference_fixture, profile=profile))


@pytest.mark.parametrize(
    "segment_code",
    _PATH_LIKE_IDENTIFIERS,
    ids=_PATH_LIKE_IDENTIFIER_IDS,
)
def test_reference_snapshot_rejects_path_like_segment_codes(
    reference_fixture,
    segment_code: str,
) -> None:
    minimum = reference_fixture["config"].validation.min_group_size
    profile = replace(reference_fixture["profile"], segment_counts={segment_code: minimum})

    with pytest.raises(ValueError, match="path-like identifier"):
        build_reference_snapshot(**dict(reference_fixture, profile=profile))


@pytest.mark.parametrize(
    "rule_id",
    _PATH_LIKE_IDENTIFIERS,
    ids=_PATH_LIKE_IDENTIFIER_IDS,
)
def test_reference_snapshot_rejects_path_like_rule_ids(
    reference_fixture,
    rule_id: str,
) -> None:
    card = reference_fixture["evidence_cards"][0]
    path_like_card = card.model_copy(
        update={"rule": card.rule.model_copy(update={"rule_id": rule_id})}
    )

    with pytest.raises(ValueError, match="path-like identifier"):
        build_reference_snapshot(**dict(reference_fixture, evidence_cards=(path_like_card,)))


@pytest.mark.parametrize(
    "dataset_id",
    _REPEATEDLY_ENCODED_PATH_LIKE_IDENTIFIERS,
    ids=_REPEATEDLY_ENCODED_PATH_LIKE_IDENTIFIER_IDS,
)
def test_reference_snapshot_rejects_repeatedly_encoded_path_like_dataset_ids(
    reference_fixture,
    dataset_id: str,
) -> None:
    profile = replace(reference_fixture["profile"], dataset_id=dataset_id)

    with pytest.raises(ValueError, match="path-like identifier"):
        build_reference_snapshot(**dict(reference_fixture, profile=profile))


@pytest.mark.parametrize(
    "segment_code",
    _REPEATEDLY_ENCODED_PATH_LIKE_IDENTIFIERS,
    ids=_REPEATEDLY_ENCODED_PATH_LIKE_IDENTIFIER_IDS,
)
def test_reference_snapshot_rejects_repeatedly_encoded_path_like_segment_codes(
    reference_fixture,
    segment_code: str,
) -> None:
    minimum = reference_fixture["config"].validation.min_group_size
    profile = replace(reference_fixture["profile"], segment_counts={segment_code: minimum})

    with pytest.raises(ValueError, match="path-like identifier"):
        build_reference_snapshot(**dict(reference_fixture, profile=profile))


@pytest.mark.parametrize(
    "rule_id",
    _REPEATEDLY_ENCODED_PATH_LIKE_IDENTIFIERS,
    ids=_REPEATEDLY_ENCODED_PATH_LIKE_IDENTIFIER_IDS,
)
def test_reference_snapshot_rejects_repeatedly_encoded_path_like_rule_ids(
    reference_fixture,
    rule_id: str,
) -> None:
    card = reference_fixture["evidence_cards"][0]
    path_like_card = card.model_copy(
        update={"rule": card.rule.model_copy(update={"rule_id": rule_id})}
    )

    with pytest.raises(ValueError, match="path-like identifier"):
        build_reference_snapshot(**dict(reference_fixture, evidence_cards=(path_like_card,)))


@pytest.mark.parametrize(
    "segment_code",
    _PATH_LIKE_IDENTIFIERS + _REPEATEDLY_ENCODED_PATH_LIKE_IDENTIFIERS,
    ids=_PATH_LIKE_IDENTIFIER_IDS + _REPEATEDLY_ENCODED_PATH_LIKE_IDENTIFIER_IDS,
)
def test_reference_snapshot_rejects_path_like_codes_below_group_threshold(
    reference_fixture,
    segment_code: str,
) -> None:
    minimum = reference_fixture["config"].validation.min_group_size
    profile = replace(
        reference_fixture["profile"],
        segment_counts={"ordinary-small-group": minimum - 1, segment_code: minimum - 1},
    )

    with pytest.raises(ValueError, match="path-like identifier"):
        build_reference_snapshot(**dict(reference_fixture, profile=profile))


@pytest.mark.parametrize("stable_code", ["d1", "1ab", "A12"])
def test_reference_snapshot_preserves_non_path_stable_codes_verbatim(
    reference_fixture,
    stable_code: str,
) -> None:
    minimum = reference_fixture["config"].validation.min_group_size
    card = reference_fixture["evidence_cards"][0]
    stable_code_card = card.model_copy(
        update={"rule": card.rule.model_copy(update={"rule_id": stable_code})}
    )
    profile = replace(
        reference_fixture["profile"],
        dataset_id=stable_code,
        segment_counts={stable_code: minimum},
    )

    snapshot = build_reference_snapshot(
        **dict(reference_fixture, profile=profile, evidence_cards=(stable_code_card,))
    )

    assert snapshot.dataset_id == stable_code
    assert snapshot.segment_counts == {stable_code: minimum}
    assert snapshot.rules[0].rule_id == stable_code


def test_reference_snapshot_suppresses_small_segment_counts(reference_fixture) -> None:
    minimum = reference_fixture["config"].validation.min_group_size
    profile = replace(
        reference_fixture["profile"],
        segment_counts={
            "inst-c-deid": minimum + 1,
            "inst-a-deid": minimum,
            "inst-b-deid": minimum - 1,
        },
    )

    snapshot = build_reference_snapshot(**dict(reference_fixture, profile=profile))

    assert snapshot.segment_counts == {
        "inst-a-deid": minimum,
        "inst-c-deid": minimum + 1,
    }
    assert list(snapshot.segment_counts) == ["inst-a-deid", "inst-c-deid"]


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
    snapshot = build_reference_snapshot(**dict(reference_fixture, frame=frame))
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

    assert snapshot.rules[0].rule_id == reference_fixture["evidence_cards"][0].rule.rule_id
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
