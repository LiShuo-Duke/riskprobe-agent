import pytest

from riskprobe.benchmarking import BenchmarkRecord, StageTiming
from riskprobe.resume_evidence import aggregate_benchmarks, render_resume_bullets


def make_record(task_id: str, manual: float = 60.0, agent: float = 30.0) -> BenchmarkRecord:
    return BenchmarkRecord(
        run_id=f"run-{task_id}",
        task_id=task_id,
        dataset_id="company_current",
        measured_at="2026-08-05T10:00:00Z",
        code_version="0.1.0",
        config_hash="cfg123",
        data_fingerprint=f"data-{task_id}",
        manual_minutes=manual,
        agent_minutes=agent,
        stage_timings=(StageTiming(stage="inspect", seconds=12.0),),
        candidate_rule_count=30,
        evidence_passed_count=15,
        reviewed_rule_count=10,
        accepted_rule_count=4,
        anomaly_true_positive_count=9,
        anomaly_false_positive_count=2,
        anomaly_false_negative_count=1,
        root_cause_top3_hit_count=7,
        root_cause_case_count=10,
    )


def test_resume_evidence_requires_three_completed_tasks() -> None:
    with pytest.raises(ValueError, match="at least 3 completed tasks"):
        aggregate_benchmarks([make_record("001"), make_record("002")])


def test_resume_metrics_use_recorded_values() -> None:
    records = [make_record("001"), make_record("002"), make_record("003")]

    evidence = aggregate_benchmarks(records)

    assert evidence.task_count == 3
    assert evidence.total_reviewed_rules == 30
    assert evidence.total_accepted_rules == 12
    assert evidence.efficiency_rate == 0.5
    assert evidence.anomaly_recall == 27 / 30
    assert evidence.root_cause_top3_hit_rate == 21 / 30
    assert evidence.source_run_ids == ("run-001", "run-002", "run-003")


def test_resume_draft_uses_only_aggregated_measurements() -> None:
    evidence = aggregate_benchmarks([make_record("001"), make_record("002"), make_record("003")])

    draft = render_resume_bullets(evidence)

    assert "50.0%" in draft.company_experience
    assert "3 completed" in draft.company_experience
    assert "Run ID" not in draft.public_project
