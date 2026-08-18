import polars as pl
import pytest

from riskprobe.config import DiscoveryConfig
from riskprobe.models import RiskRule
from riskprobe.rules.discovery import discover_rules
from riskprobe.rules.expression import evaluate_rule
from riskprobe.synthetic import generate_behavior_dataset


def _expressions(rules: list[RiskRule]) -> list[tuple[tuple[str, str, object], ...]]:
    return [
        tuple(
            (condition.feature, condition.operator, condition.value)
            for condition in rule.conditions
        )
        for rule in rules
    ]


def test_discovery_finds_cancel_rate_signal_deterministically() -> None:
    frame, _ = generate_behavior_dataset(rows=20_000, seed=42)
    train = frame.sort("snapshot_date").head(int(frame.height * 0.7))
    config = DiscoveryConfig(min_support=0.03, max_single_rules=40, beam_width=10)

    first = discover_rules(train, ["order_cancel_rate_30d"], "target", config)
    second = discover_rules(train, ["order_cancel_rate_30d"], "target", config)

    assert [rule.model_dump() for rule in first] == [rule.model_dump() for rule in second]
    assert any(
        condition.feature == "order_cancel_rate_30d" and condition.operator == ">"
        for rule in first
        for condition in rule.conditions
    )


def test_discovery_generates_pair_rules_from_different_features() -> None:
    x = [-2.0, -1.0, 1.0, 2.0] * 100
    y = [-2.0, 1.0, -1.0, 2.0] * 100
    train = pl.DataFrame(
        {
            "x": x,
            "y": y,
            "target": [int(left > 0 and right > 0) for left, right in zip(x, y, strict=True)],
        }
    )
    config = DiscoveryConfig(
        min_support=0.1,
        max_single_rules=20,
        beam_width=10,
        max_pair_rules=10,
    )

    rules = discover_rules(train, ["x", "y"], "target", config)

    assert any(
        len(rule.conditions) == 2
        and {condition.feature for condition in rule.conditions} == {"x", "y"}
        for rule in rules
    )


def test_pair_beam_is_independent_of_single_rule_output_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = [-1.0] + [1.0] * 4
    train = pl.DataFrame(
        {
            "x": x * 3,
            "y": [-1.0] * 5 + [1.0] * 5 + [2.0] * 5,
            "target": [0] * 10 + [0, 1, 1, 1, 1],
        }
    )

    def fixed_thresholds(
        _train: pl.DataFrame,
        feature_name: str,
        _target: object,
        _config: DiscoveryConfig,
    ) -> list[float]:
        return {"x": [0.0], "y": [0.0, 1.0]}[feature_name]

    monkeypatch.setattr(
        "riskprobe.rules.discovery._feature_thresholds", fixed_thresholds
    )
    config = DiscoveryConfig(
        min_support=0.1,
        max_single_rules=2,
        beam_width=3,
        max_pair_rules=10,
    )

    rules = discover_rules(train, ["x", "y"], "target", config)
    singles = [rule for rule in rules if len(rule.conditions) == 1]
    pairs = [rule for rule in rules if len(rule.conditions) == 2]

    assert {rule.conditions[0].feature for rule in singles} == {"y"}
    assert any(
        {condition.feature for condition in rule.conditions} == {"x", "y"}
        for rule in pairs
    )


def test_discovery_skips_all_null_constant_nan_and_infinite_features() -> None:
    train = pl.DataFrame(
        {
            "all_null": pl.Series([None] * 20, dtype=pl.Float64),
            "constant": [3.0] * 20,
            "all_nan": [float("nan")] * 20,
            "infinite": [float("inf"), float("-inf")] * 10,
            "target": [0, 1] * 10,
        }
    )

    rules = discover_rules(
        train,
        ["all_null", "constant", "all_nan", "infinite"],
        "target",
        DiscoveryConfig(min_support=0.1),
    )

    assert rules == []


def test_discovery_skips_feature_when_infinite_values_mix_with_finite_values() -> None:
    train = pl.DataFrame(
        {
            "feature": [float("nan"), float("inf"), float("-inf")]
            + [float(value) for value in range(20)],
            "target": [0, 0, 0] + [0] * 10 + [1] * 10,
        }
    )

    rules = discover_rules(
        train,
        ["feature"],
        "target",
        DiscoveryConfig(min_support=0.1, max_pair_rules=0),
    )

    assert rules == []


def test_discovery_rejects_a_missing_feature() -> None:
    train = pl.DataFrame({"present": [0.0, 1.0], "target": [0, 1]})

    with pytest.raises(ValueError, match="missing feature.*absent"):
        discover_rules(train, ["present", "absent"], "target", DiscoveryConfig())


@pytest.mark.parametrize("label", [0, 1])
def test_discovery_returns_no_rules_for_a_single_target_label(label: int) -> None:
    train = pl.DataFrame(
        {"feature": [float(value) for value in range(20)], "target": [label] * 20}
    )

    assert discover_rules(train, ["feature"], "target", DiscoveryConfig()) == []


def test_discovery_deduplicates_repeated_thresholds_and_rule_ids() -> None:
    train = pl.DataFrame(
        {
            "feature": [0.0] * 10 + [1.0] * 10 + [2.0] * 10 + [3.0] * 10,
            "target": [0] * 20 + [1] * 20,
        }
    )

    rules = discover_rules(
        train,
        ["feature"],
        "target",
        DiscoveryConfig(min_support=0.1, max_single_rules=100, max_pair_rules=0),
    )

    expressions = _expressions(rules)
    assert len(expressions) == len(set(expressions))
    assert len({rule.rule_id for rule in rules}) == len(rules)


@pytest.mark.parametrize(
    ("row_count", "min_support", "lower_count", "upper_count"),
    [(10, 0.2, 2, 8), (100, 0.07, 7, 93), (90, 0.3, 27, 63)],
)
def test_min_support_boundaries_are_inclusive(
    row_count: int,
    min_support: float,
    lower_count: int,
    upper_count: int,
) -> None:
    train = pl.DataFrame(
        {
            "feature": [float(value) for value in range(row_count)],
            "target": [1] * lower_count + [0] * (row_count - lower_count),
        }
    )
    config = DiscoveryConfig(
        min_support=min_support,
        max_single_rules=100,
        max_pair_rules=0,
    )

    rules = discover_rules(train, ["feature"], "target", config)
    support_counts = [evaluate_rule(train, rule).sum() for rule in rules]

    assert lower_count in support_counts
    assert upper_count in support_counts
    assert all(lower_count <= count <= upper_count for count in support_counts)


def test_feature_input_order_does_not_change_rules_or_tie_order() -> None:
    values = [float(value) for value in range(40)]
    train = pl.DataFrame(
        {
            "alpha": values,
            "beta": values,
            "target": [0] * 20 + [1] * 20,
        }
    )
    config = DiscoveryConfig(
        min_support=0.1,
        max_single_rules=30,
        beam_width=8,
        max_pair_rules=10,
    )

    forward = discover_rules(train, ["alpha", "beta"], "target", config)
    reverse = discover_rules(train, ["beta", "alpha"], "target", config)

    assert [rule.model_dump() for rule in forward] == [rule.model_dump() for rule in reverse]


def test_lightgbm_produces_no_stdout_or_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    train = pl.DataFrame(
        {
            "feature": [float(value) for value in range(100)],
            "target": [0] * 50 + [1] * 50,
        }
    )

    discover_rules(train, ["feature"], "target", DiscoveryConfig(max_pair_rules=0))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_discovery_does_not_expand_to_categorical_features() -> None:
    train = pl.DataFrame(
        {
            "category": ["0"] * 10 + ["1"] * 10,
            "target": [0] * 10 + [1] * 10,
        }
    )

    rules = discover_rules(train, ["category"], "target", DiscoveryConfig())

    assert rules == []


def test_discover_with_metrics_returns_counts_and_train_metrics() -> None:
    x = [-2.0, -1.0, 1.0, 2.0] * 100
    y = [-2.0, 1.0, -1.0, 2.0] * 100
    train = pl.DataFrame(
        {
            "x": x,
            "y": y,
            "target": [int(left > 0 and right > 0) for left, right in zip(x, y, strict=True)],
        }
    )
    config = DiscoveryConfig(
        min_support=0.1,
        max_single_rules=3,
        beam_width=10,
        max_pair_rules=2,
    )

    from riskprobe.rules.discovery import discover_with_metrics

    result = discover_with_metrics(train, ["x", "y"], "target", config)

    assert result.single_candidates_before_cap >= result.single_rules_selected
    assert result.pair_candidates_before_diversity >= result.pair_rules_selected
    assert len(result.rules) == result.single_rules_selected + result.pair_rules_selected
    assert set(result.train_metrics) == {rule.rule_id for rule in result.rules}
    assert all(metric.lift >= 0 for metric in result.train_metrics.values())
    assert any(len(rule.conditions) == 2 for rule in result.rules)
