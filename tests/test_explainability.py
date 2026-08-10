import pytest

from riskprobe.config import ValidationConfig
from riskprobe.explainability import (
    stable_reasons,
    summarize_candidate_rules,
    summarize_conditions,
    summarize_evidence_cards,
)
from riskprobe.models import Condition, EvidenceCard, RiskRule, RuleMetrics
from riskprobe.privacy import stable_token
from riskprobe.rules.discovery import DiscoveryResult


def _metrics(lift: float, support_count: int = 40) -> RuleMetrics:
    return RuleMetrics(
        support_count=support_count,
        coverage=support_count / 200,
        base_bad_rate=0.2,
        hit_bad_rate=0.4,
        non_hit_bad_rate=0.1,
        lift=lift,
        precision=0.4,
        recall=0.2,
        p_value=0.01,
    )


def _rule(rule_id: str, conditions: tuple[Condition, ...]) -> RiskRule:
    return RiskRule(rule_id=rule_id, conditions=conditions, origin="discovery_single")


def test_candidate_summary_ranks_by_train_lift_then_support() -> None:
    first = _rule(
        "rule-a", (Condition(feature="age", operator=">", value=50.0),)
    )
    second = _rule(
        "rule-b", (Condition(feature="income", operator="<=", value=100.0),)
    )
    pair = _rule(
        "rule-c",
        (
            Condition(feature="age", operator=">", value=50.0),
            Condition(feature="income", operator="<=", value=100.0),
        ),
    )
    result = DiscoveryResult(
        rules=(first, second, pair),
        train_metrics={
            "rule-a": _metrics(3.0, 40),
            "rule-b": _metrics(2.0, 50),
            "rule-c": _metrics(2.5, 30),
        },
        single_candidates_before_cap=2,
        single_rules_selected=2,
        pair_candidates_before_diversity=1,
        pair_rules_selected=1,
    )

    summary = summarize_candidate_rules(result, frozenset({"age", "income"}))

    assert [item["rank"] for item in summary["top_rules"]] == [1, 2, 3]
    assert summary["top_rules"][0]["train"]["lift"] == 3.0
    assert {item["condition_count"] for item in summary["top_two_condition_rules"]} == {2}
    assert summary["single_rule_count"] == 2
    assert summary["two_condition_rule_count"] == 1


def test_evidence_summary_ranks_by_test_lift_and_preserves_grade() -> None:
    single = _rule(
        "rule-a", (Condition(feature="age", operator=">", value=50.0),)
    )
    pair = _rule(
        "rule-b",
        (
            Condition(feature="age", operator=">", value=50.0),
            Condition(feature="income", operator="<=", value=100.0),
        ),
    )
    cards = [
        EvidenceCard(
            rule=single,
            train=_metrics(2.0),
            test=_metrics(3.0),
            slices=(),
            lift_ci=(1.2, 3.8),
            adjusted_p_value=0.01,
            segment_consistency=0.8,
            max_time_decay=0.0,
            grade="Stable",
        ),
        EvidenceCard(
            rule=pair,
            train=_metrics(2.5),
            test=_metrics(2.0),
            slices=(),
            lift_ci=(1.1, 2.8),
            adjusted_p_value=0.02,
            segment_consistency=0.7,
            max_time_decay=0.0,
            grade="Local",
        ),
    ]

    summary = summarize_evidence_cards(
        cards, frozenset({"age", "income"}), ValidationConfig()
    )

    assert summary["top_rules"][0]["test"]["lift"] == 3.0
    assert summary["top_rules"][0]["grade"] == "Stable"
    assert all(len(item["conditions"]) == 2 for item in summary["top_two_condition_rules"])
    assert summary["grade_counts"] == {
        "Stable": 1,
        "Local": 1,
        "Unstable": 0,
        "Suspicious": 0,
    }
    assert set(summary["top_rules"][0]["train"]) == {
        "support_count", "coverage", "base_bad_rate", "hit_bad_rate",
        "non_hit_bad_rate", "lift", "precision", "recall", "p_value",
    }
    assert set(summary["top_rules"][0]) >= {
        "lift_ci", "adjusted_p_value", "segment_consistency",
        "max_time_decay", "grade", "reason_codes",
    }


def test_condition_summary_rejects_unconfirmed_or_string_values() -> None:
    unknown = _rule(
        "rule-a", (Condition(feature="private_identifier", operator=">", value=1.0),)
    )
    string_value = _rule(
        "rule-b", (Condition(feature="age", operator="==", value="adult"),)
    )

    with pytest.raises(ValueError, match="confirmed feature"):
        summarize_conditions(unknown, frozenset({"age"}))
    with pytest.raises(ValueError, match="numeric"):
        summarize_conditions(string_value, frozenset({"age"}))


def test_stable_reasons_are_codes_without_free_text_values() -> None:
    card = EvidenceCard(
        rule=_rule("rule-a", (Condition(feature="age", operator=">", value=50.0),)),
        train=_metrics(2.0),
        test=_metrics(1.1, support_count=10),
        slices=(),
        lift_ci=(0.9, 1.5),
        adjusted_p_value=0.2,
        segment_consistency=0.2,
        max_time_decay=0.5,
        grade="Suspicious",
    )

    reasons = stable_reasons(card, ValidationConfig(min_group_size=20))

    assert reasons == [
        "adjusted_p_value_above_alpha",
        "lift_ci_lower_not_above_one",
        "insufficient_samples",
        "time_decay_above_limit",
    ]



def test_evidence_summary_includes_institution_metrics_and_interpretation() -> None:
    from riskprobe.models import SliceMetrics

    card = EvidenceCard(
        rule=_rule("local-rule", (Condition(feature="age", operator=">", value=50.0),)),
        train=_metrics(2.0),
        test=_metrics(2.5),
        slices=(
            SliceMetrics(
                slice_type="segment",
                slice_value="institution-a",
                metrics=_metrics(3.0),
            ),
        ),
        lift_ci=(1.2, 3.8),
        adjusted_p_value=0.01,
        segment_consistency=1.0,
        max_time_decay=0.0,
        grade="Stable",
    )

    summary = summarize_evidence_cards(
        [card], frozenset({"age"}), ValidationConfig(min_group_size=20)
    )

    item = summary["top_rules"][0]
    assert item["institution_results"][0]["institution_token"].startswith("tok_")
    assert item["institution_results"][0]["institution_name"] == "institution-a"
    assert item["institution_results"][0]["metrics"]["lift"] == 3.0
    assert summary["institution_summary"]["institution_count"] == 1
    assert summary["institution_summary"]["institution_names"] == ["institution-a"]
    assert "interpretation" in summary["institution_summary"]

    hidden = summarize_evidence_cards(
        [card],
        frozenset({"age"}),
        ValidationConfig(min_group_size=20),
        expose_segment_values=False,
    )
    hidden_item = hidden["top_rules"][0]
    assert "institution_name" not in hidden_item["institution_results"][0]
    assert "institution_names" not in hidden["institution_summary"]


def test_alert_summary_exposes_authorized_institution_name_only() -> None:
    from riskprobe.explainability import summarize_alerts
    from riskprobe.monitoring.models import Alert

    alert = Alert(
        alert_id="alert-a",
        alert_type="population",
        severity="warning",
        scope="institution",
        scope_value="institution-a",
        metric="population_share",
        reference_value=0.2,
        current_value=0.4,
        delta=0.2,
        evidence={"reference_count": 20, "current_count": 40},
    )

    hidden = summarize_alerts(
        [alert],
        reference_row_count=100,
        reference_positive_rate=0.2,
        reference_feature_count=1,
        current_row_count=100,
        current_positive_rate=0.2,
        current_feature_count=1,
        expose_segment_values=False,
    )
    assert hidden["institution_alerts"][0]["scope_value"] == stable_token("institution-a")
    assert "institution_name" not in hidden["institution_alerts"][0]

    exposed = summarize_alerts(
        [alert],
        reference_row_count=100,
        reference_positive_rate=0.2,
        reference_feature_count=1,
        current_row_count=100,
        current_positive_rate=0.2,
        current_feature_count=1,
    )

    assert exposed["institution_alerts"][0]["institution_name"] == "institution-a"
