from datetime import date, datetime
import math

import polars as pl
import pytest

from riskprobe.config import ValidationConfig
from riskprobe.models import Condition, RiskRule
from riskprobe.rules.validation import validate_rules

_CONFIG = ValidationConfig(bootstrap_rounds=100, min_group_size=20)


def _frame(
    groups: list[tuple[str, object, int, int, int, int]],
) -> pl.DataFrame:
    data: dict[str, list[object]] = {
        "f": [],
        "target": [],
        "institution": [],
        "snapshot_date": [],
    }
    for segment, snapshot, hit_bad, hit_good, miss_bad, miss_good in groups:
        counts = (hit_bad, hit_good, miss_bad, miss_good)
        data["f"].extend(
            [1] * (hit_bad + hit_good) + [0] * (miss_bad + miss_good)
        )
        data["target"].extend(
            [1] * hit_bad
            + [0] * hit_good
            + [1] * miss_bad
            + [0] * miss_good
        )
        group_size = sum(counts)
        data["institution"].extend([segment] * group_size)
        data["snapshot_date"].extend([snapshot] * group_size)
    return pl.DataFrame(data)


def _rule(rule_id: str = "R1", feature: str = "f") -> RiskRule:
    return RiskRule(
        rule_id=rule_id,
        conditions=(Condition(feature=feature, operator=">", value=0.5),),
        origin="test",
    )


def _validate(
    train: pl.DataFrame,
    test: pl.DataFrame,
    rules: list[RiskRule] | None = None,
    *,
    segment_display_name: str = "institution",
    time_validation_enabled: bool = True,
    metadata_grade: str = "A",
    config: ValidationConfig = _CONFIG,
):
    return validate_rules(
        train,
        test,
        rules if rules is not None else [_rule()],
        target_col="target",
        segment_col="institution",
        snapshot_col="snapshot_date",
        segment_display_name=segment_display_name,
        time_validation_enabled=time_validation_enabled,
        config=config,
        metadata_grade=metadata_grade,
    )


def test_rule_with_test_lift_decay_is_not_stable() -> None:
    train = pl.DataFrame(
        {
            "f": [1] * 100 + [0] * 100,
            "target": [1] * 50 + [0] * 50 + [1] * 10 + [0] * 90,
            "institution": ["A"] * 200,
            "snapshot_date": ["2026-01-01"] * 200,
        }
    )
    test = pl.DataFrame(
        {
            "f": [1] * 100 + [0] * 100,
            "target": [1] * 20 + [0] * 80 + [1] * 20 + [0] * 80,
            "institution": ["A"] * 200,
            "snapshot_date": ["2026-02-01"] * 200,
        }
    )

    cards = _validate(train, test, metadata_grade="B")

    assert cards[0].grade in {"Unstable", "Suspicious"}
    assert "label performance window unknown" in cards[0].limitations


def test_empty_rules_return_empty_cards() -> None:
    assert _validate(pl.DataFrame(), pl.DataFrame(), rules=[]) == []


@pytest.mark.parametrize("empty_dataset", ["train", "test"])
def test_global_dataset_without_positive_target_is_rejected(
    empty_dataset: str,
) -> None:
    valid = _frame([("A", "2026-01-01", 80, 20, 20, 80)])
    no_positive = valid.with_columns(pl.lit(0).alias("target"))
    train = no_positive if empty_dataset == "train" else valid
    test = no_positive if empty_dataset == "test" else valid

    with pytest.raises(ValueError, match="^target has no positive samples$"):
        _validate(train, test)


def test_global_dataset_below_minimum_size_is_suspicious() -> None:
    small = _frame([("A", "2026-01-01", 4, 1, 1, 4)])

    card = _validate(small, small, time_validation_enabled=False)[0]

    assert card.grade == "Suspicious"
    assert all(math.isfinite(value) for value in card.lift_ci)


@pytest.mark.parametrize("single_class_target", [0, 1])
def test_segment_slices_skip_single_class_symmetrically_with_auditable_limitation(
    single_class_target: int,
) -> None:
    single_class_counts = (
        (0, 10, 0, 10) if single_class_target == 0 else (10, 0, 10, 0)
    )
    frame = _frame(
        [
            ("exactly_twenty", "2026-01-01", 8, 2, 2, 8),
            ("nineteen", "2026-01-01", 8, 2, 2, 7),
            ("single_class", "2026-01-01", *single_class_counts),
        ]
    )

    card = _validate(
        frame,
        frame,
        segment_display_name="customer_segment",
        time_validation_enabled=False,
    )[0]
    segment_slices = [item for item in card.slices if item.slice_type == "segment"]

    assert [item.slice_value for item in segment_slices] == ["exactly_twenty"]
    assert card.segment_consistency == 1.0
    assert card.grade == "Suspicious"
    assert "single-class customer_segment: single_class" in card.limitations
    assert all(math.isfinite(item.metrics.lift) for item in card.slices)


@pytest.mark.parametrize("single_class_target", [0, 1])
def test_time_slices_skip_single_class_symmetrically_with_stable_limitation_name(
    single_class_target: int,
) -> None:
    single_class_counts = (
        (0, 10, 0, 10) if single_class_target == 0 else (10, 0, 10, 0)
    )
    frame = _frame(
        [
            ("A", "2026-01-01", 40, 10, 10, 40),
            ("A", "2026-02-01", *single_class_counts),
        ]
    )

    card = _validate(frame, frame)[0]
    time_slices = [item for item in card.slices if item.slice_type == "time"]

    assert [item.slice_value for item in time_slices] == ["2026-01"]
    assert card.grade == "Suspicious"
    assert "single-class time: 2026-02" in card.limitations


@pytest.mark.parametrize(
    "snapshots, expected",
    [
        (("2026-01-31", "2026-02-01"), ["2026-01", "2026-02"]),
        ((date(2026, 1, 31), date(2026, 2, 1)), ["2026-01", "2026-02"]),
        (
            (datetime(2026, 1, 31, 23, 59), datetime(2026, 2, 1, 0, 1)),
            ["2026-01", "2026-02"],
        ),
    ],
)
def test_time_slices_parse_supported_dates_to_year_month(
    snapshots: tuple[object, object], expected: list[str]
) -> None:
    frame = _frame(
        [
            ("A", snapshots[0], 40, 10, 10, 40),
            ("A", snapshots[1], 40, 10, 10, 40),
        ]
    )

    card = _validate(frame, frame)[0]

    assert [
        item.slice_value for item in card.slices if item.slice_type == "time"
    ] == expected


def test_validation_accepts_categorical_snapshots_and_ignores_original_nulls() -> None:
    frame = _frame(
        [
            ("A", "2026-01-01", 40, 10, 10, 40),
            ("A", None, 8, 2, 2, 8),
        ]
    ).with_columns(pl.col("snapshot_date").cast(pl.Categorical))

    card = _validate(frame, frame)[0]

    assert [
        item.slice_value for item in card.slices if item.slice_type == "time"
    ] == ["2026-01"]
    assert "missing time values: 20 rows excluded" in card.limitations


def test_invalid_snapshot_date_is_rejected_when_time_validation_is_enabled() -> None:
    frame = _frame([("A", "not-a-date", 80, 20, 20, 80)])

    with pytest.raises(ValueError, match="^snapshot column contains invalid dates$"):
        _validate(frame, frame)


@pytest.mark.parametrize("empty_dataset", ["train", "test"])
def test_zero_row_dataset_is_rejected_with_stable_error(empty_dataset: str) -> None:
    valid = _frame([("A", "2026-01-01", 80, 20, 20, 80)])
    empty = valid.clear()
    train = empty if empty_dataset == "train" else valid
    test = empty if empty_dataset == "test" else valid

    with pytest.raises(ValueError, match="^mask and target must not be empty$"):
        _validate(train, test)


def test_adjusted_pvalues_are_assigned_to_corresponding_input_rules() -> None:
    target = [1] * 20 + [0] * 20
    frame = pl.DataFrame(
        {
            "strong": [1] * 10 + [0] * 30,
            "medium": [1] * 8 + [0] * 12 + [1] * 2 + [0] * 18,
            "all_rows": [1] * 40,
            "target": target,
            "institution": ["A"] * 40,
            "snapshot_date": ["2026-01-01"] * 40,
        }
    )
    rules = [
        _rule("strong", "strong"),
        _rule("medium", "medium"),
        _rule("all_rows", "all_rows"),
    ]

    cards = _validate(frame, frame, rules=rules)

    assert [card.rule for card in cards] == rules
    assert [card.adjusted_p_value for card in cards] == pytest.approx(
        [0.0013077593722755016, 0.09724974241103275, 1.0]
    )


def test_suspicious_grade_takes_priority_over_decay_and_locality() -> None:
    train = _frame([("A", "2026-01-01", 90, 10, 10, 90)])
    test = _frame(
        [
            ("A", "2026-02-01", 90, 10, 10, 90),
            ("B", "2026-02-01", 10, 90, 90, 10),
        ]
    )

    card = _validate(train, test)[0]

    assert card.max_time_decay > _CONFIG.max_lift_decay
    assert card.segment_consistency < _CONFIG.min_segment_consistency
    assert card.grade == "Suspicious"


def test_unstable_grade_takes_priority_over_locality() -> None:
    train = _frame([("A", "2026-01-01", 360, 40, 40, 360)])
    test = _frame(
        [
            ("A", "2026-02-01", 180, 20, 20, 180),
            ("B", "2026-02-01", 60, 140, 140, 60),
        ]
    )

    card = _validate(train, test)[0]

    assert card.lift_ci[0] > 1.0
    assert card.segment_consistency == 0.5
    assert card.max_time_decay > _CONFIG.max_lift_decay
    assert card.grade == "Unstable"


def test_low_segment_consistency_with_stable_segment_is_local() -> None:
    train = _frame([("A", "2026-01-01", 90, 10, 10, 90)])
    test = _frame(
        [
            ("A", "2026-02-01", 90, 10, 10, 90),
            ("B", "2026-02-01", 20, 30, 30, 20),
        ]
    )

    card = _validate(train, test)[0]

    assert card.lift_ci[0] > 1.0
    assert card.segment_consistency == 0.5
    assert card.max_time_decay <= _CONFIG.max_lift_decay
    assert card.grade == "Local"


def test_zero_train_lift_produces_finite_zero_decay() -> None:
    frame = _frame([("A", "2026-01-01", 0, 100, 100, 0)])

    card = _validate(frame, frame)[0]

    assert card.train.lift == 0.0
    assert card.max_time_decay == 0.0
    assert math.isfinite(card.max_time_decay)


def test_improved_time_lift_clamps_negative_decay_to_zero() -> None:
    train = _frame([("A", "2026-01-01", 60, 40, 40, 60)])
    test = _frame([("A", "2026-02-01", 90, 10, 10, 90)])

    card = _validate(train, test)[0]

    assert card.test.lift > card.train.lift
    assert card.max_time_decay == 0.0
    assert card.grade == "Stable"


def test_time_validation_disabled_does_not_parse_or_create_time_slices() -> None:
    frame = _frame([("A", "not-a-date", 80, 20, 20, 80)])

    card = _validate(frame, frame, time_validation_enabled=False)[0]

    assert all(item.slice_type != "time" for item in card.slices)
    assert card.max_time_decay == 0.0


def test_metadata_grade_b_adds_limitation_without_downgrading_stable_card() -> None:
    frame = _frame([("A", "2026-01-01", 80, 20, 20, 80)])

    card = _validate(frame, frame, metadata_grade="B")[0]

    assert card.grade == "Stable"
    assert card.limitations == ("label performance window unknown",)
