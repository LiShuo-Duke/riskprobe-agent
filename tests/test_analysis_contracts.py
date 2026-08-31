from __future__ import annotations

import pytest
from pydantic import ValidationError

from riskprobe.analysis_contracts import (
    APPROVED_STAGE_NAMES,
    AnalysisSummary,
    DiscoverySummary,
    FeatureRef,
    MetricDistribution,
    ScorecardSplitMetrics,
    StageName,
    StageStatus,
    StageSummary,
)


def _stages() -> tuple[StageSummary, ...]:
    return tuple(
        StageSummary(
            name=name,
            status=StageStatus.NOT_RUN,
            enabled=False,
            output_available=False,
        )
        for name in StageName
    )


def test_stage_summary_accepts_all_public_statuses() -> None:
    assert {status.value for status in StageStatus} == {
        "not_run",
        "succeeded",
        "skipped",
        "unavailable",
        "failed",
        "reused",
    }
    assert len(APPROVED_STAGE_NAMES) == 24
    assert StageSummary(
        name=StageName.PROFILE,
        status=StageStatus.SUCCEEDED,
        enabled=True,
        output_available=True,
    ).status is StageStatus.SUCCEEDED


def test_analysis_summary_requires_each_approved_stage_once() -> None:
    summary = AnalysisSummary(stages=_stages())

    assert tuple(stage.name.value for stage in summary.stages) == APPROVED_STAGE_NAMES
    with pytest.raises(ValidationError, match="stage"):
        AnalysisSummary(stages=summary.stages[:-1])
    with pytest.raises(ValidationError, match="stage"):
        AnalysisSummary(stages=summary.stages + (summary.stages[0],))


def test_scorecard_metrics_validate_gini_and_unavailable_reason() -> None:
    metrics = ScorecardSplitMetrics(
        split="train",
        status=StageStatus.SUCCEEDED,
        sample_count=20,
        positive_rate=0.25,
        probability_min=0.1,
        probability_mean=0.25,
        probability_max=0.9,
        mean_score=550.0,
        risk_level_counts={"low": 20},
        auc=0.8,
        ks=0.6,
        gini=0.6,
    )

    assert metrics.gini == pytest.approx(0.6)
    with pytest.raises(ValidationError, match="gini"):
        ScorecardSplitMetrics(
            split="train",
            status=StageStatus.SUCCEEDED,
            sample_count=20,
            positive_rate=0.25,
            probability_min=0.1,
            probability_mean=0.25,
            probability_max=0.9,
            mean_score=550.0,
            risk_level_counts={"low": 20},
            auc=0.8,
            ks=0.6,
            gini=0.5,
        )
    with pytest.raises(ValidationError, match="reason_code"):
        ScorecardSplitMetrics(
            split="holdout",
            status=StageStatus.UNAVAILABLE,
            sample_count=0,
        )


def test_feature_ref_preserves_safe_names_and_tokens_unsafe_names() -> None:
    assert FeatureRef.from_name("PAY_AMT1").value == "PAY_AMT1"
    tokenized = FeatureRef.from_name("customer_id")

    assert tokenized.value.startswith("model-feature-")
    assert tokenized.value != "customer_id"
    assert tokenized == FeatureRef.from_name("customer_id")


def test_stage_summary_rejects_invalid_status_payload_combinations() -> None:
    with pytest.raises(ValidationError, match="output_available"):
        StageSummary(
            name=StageName.SCORECARD,
            status=StageStatus.FAILED,
            enabled=True,
            output_available=True,
            reason_code="scorecard_failed",
        )
    with pytest.raises(ValidationError, match="reason_code"):
        StageSummary(
            name=StageName.SCORECARD,
            status=StageStatus.UNAVAILABLE,
            enabled=True,
            output_available=False,
        )


def test_discovery_summary_accepts_digit_prefixed_public_rule_ids() -> None:
    empty = MetricDistribution(count=0)

    summary = DiscoverySummary(
        sampled=False,
        input_feature_count=1,
        eligible_feature_count=1,
        candidate_rule_count=1,
        selected_rule_count=1,
        single_candidate_count=1,
        single_rule_count=1,
        pair_candidate_count=0,
        pair_rule_count=0,
        rule_ids=("003c5ae36862", "rule-460472387795"),
        lift=empty,
        support=empty,
        precision=empty,
    )

    assert summary.rule_ids == ("003c5ae36862", "rule-460472387795")


def test_discovery_summary_rejects_unprefixed_long_numeric_rule_id() -> None:
    empty = MetricDistribution(count=0)

    with pytest.raises(ValidationError):
        DiscoverySummary(
            sampled=False,
            input_feature_count=1,
            eligible_feature_count=1,
            candidate_rule_count=1,
            selected_rule_count=1,
            single_candidate_count=1,
            single_rule_count=1,
            pair_candidate_count=0,
            pair_rule_count=0,
            rule_ids=("460472387795",),
            lift=empty,
            support=empty,
            precision=empty,
        )
