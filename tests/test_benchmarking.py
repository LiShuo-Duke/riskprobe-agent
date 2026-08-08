import pytest

from riskprobe.benchmarking import BenchmarkRecord, StageTiming, calculate_efficiency


def _record(manual_minutes: float | None = None) -> BenchmarkRecord:
    return BenchmarkRecord(
        run_id="run001",
        task_id="joint-validation-001",
        dataset_id="company_current",
        measured_at="2026-08-05T10:00:00Z",
        code_version="0.1.0",
        config_hash="cfg123",
        data_fingerprint="data123",
        baseline_task_id="joint-validation-001",
        manual_minutes=manual_minutes,
        agent_minutes=32.0,
        stage_timings=(StageTiming(stage="inspect", seconds=12.0),),
        candidate_rule_count=30,
        evidence_passed_count=15,
        reviewed_rule_count=15,
        accepted_rule_count=6,
        anomaly_true_positive_count=9,
        anomaly_false_positive_count=2,
        anomaly_false_negative_count=1,
        root_cause_top3_hit_count=7,
        root_cause_case_count=10,
    )


def test_efficiency_requires_measured_manual_baseline() -> None:
    with pytest.raises(ValueError, match="manual baseline"):
        calculate_efficiency(_record())


def test_benchmark_metrics_are_derived_from_raw_counts() -> None:
    record = _record(manual_minutes=64.0)

    assert calculate_efficiency(record) == 0.5
    assert record.precision == 9 / 11
    assert record.recall == 0.9
    assert record.rule_review_acceptance == 0.4
    assert record.root_cause_top3_hit_rate == 0.7


def test_benchmark_record_rejects_accepted_rules_above_reviewed_rules() -> None:
    with pytest.raises(ValueError, match="accepted_rule_count"):
        _record(manual_minutes=64.0).model_copy(
            update={"accepted_rule_count": 16}
        ).validate_consistency()


def test_benchmark_record_requires_manual_minutes_field_even_when_null() -> None:
    payload = _record().model_dump(exclude={"manual_minutes"})

    with pytest.raises(ValueError, match="manual_minutes"):
        BenchmarkRecord.model_validate(payload)


def test_benchmark_record_rejects_duplicate_stage_timings() -> None:
    with pytest.raises(ValueError, match="stage_timings"):
        _record(manual_minutes=64.0).model_copy(
            update={
                "stage_timings": (
                    StageTiming(stage="inspect", seconds=12.0),
                    StageTiming(stage="inspect", seconds=13.0),
                )
            }
        ).validate_consistency()


def test_benchmark_record_requires_baseline_task_binding() -> None:
    payload = _record(manual_minutes=64.0).model_dump(exclude={"baseline_task_id"})

    with pytest.raises(ValueError, match="baseline_task_id"):
        BenchmarkRecord.model_validate(payload)


def test_benchmark_record_rejects_baseline_task_mismatch() -> None:
    with pytest.raises(ValueError, match="baseline_task_id"):
        _record(manual_minutes=64.0).model_copy(
            update={"baseline_task_id": "other-task"}
        ).validate_consistency()


    record = _record(manual_minutes=64.0).model_copy(
        update={"baseline_fingerprint": "sha256:baseline", "baseline_task_id": "joint-validation-001"}
    )
    assert record.baseline_fingerprint == "sha256:baseline"
    assert record.baseline_task_id == record.task_id
