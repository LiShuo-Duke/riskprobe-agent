import pytest
from pydantic import ValidationError

from riskprobe.evals import (
    DEFAULT_EVAL_SEED,
    EvalCase,
    EvalHarness,
    EvalObservation,
    EvalSuite,
)

_EVIDENCE_A = "a" * 64
_SEQUENCE = ("inspect", "diagnose", "discover", "recommend", "review")


def _suite(*, seed: int = DEFAULT_EVAL_SEED) -> EvalSuite:
    return EvalSuite(
        suite_id="comprehensive-offline-v1",
        seed=seed,
        cases=(
            EvalCase(
                case_id="complete-diagnosis",
                objective="comprehensive",
                expected_tool_sequence=_SEQUENCE,
                required_evidence_ids=(_EVIDENCE_A,),
                require_diagnosis=True,
            ),
        ),
    )


def _observation(**updates: object) -> EvalObservation:
    payload: dict[str, object] = {
        "case_id": "complete-diagnosis",
        "task_succeeded": True,
        "tool_sequence": _SEQUENCE,
        "evidence_ids": (_EVIDENCE_A,),
        "diagnosis_evidence_ids": (_EVIDENCE_A,),
        "policy_violations": 0,
        "privacy_violations": 0,
    }
    payload.update(updates)
    return EvalObservation(**payload)


def test_harness_uses_fixed_seed_and_scores_all_required_dimensions() -> None:
    seeds: list[int] = []

    def runner(case: EvalCase, seed: int) -> EvalObservation:
        seeds.append(seed)
        assert case.case_id == "complete-diagnosis"
        return _observation()

    report = EvalHarness(seed=1234).evaluate(
        _suite(seed=1234),
        runner,
        candidate_version="candidate-v1",
    )

    assert seeds == [1234, 1234]
    assert report.passed is True
    assert report.metrics.task_success == 1.0
    assert report.metrics.tool_sequence == 1.0
    assert report.metrics.evidence_completeness == 1.0
    assert report.metrics.policy_compliance == 1.0
    assert report.metrics.privacy_compliance == 1.0
    assert report.metrics.replay_determinism == 1.0
    assert len(report.report_hash) == 64


def test_harness_detects_nondeterministic_replay_with_same_seed() -> None:
    calls = 0

    def runner(case: EvalCase, seed: int) -> EvalObservation:
        nonlocal calls
        del case, seed
        calls += 1
        if calls == 1:
            return _observation()
        return _observation(evidence_ids=(), diagnosis_evidence_ids=())

    report = EvalHarness().evaluate(_suite(), runner, candidate_version="candidate-v1")

    assert report.passed is False
    assert report.metrics.replay_determinism == 0.0
    assert report.case_results[0].replay_deterministic is False


def test_harness_scores_policy_privacy_evidence_sequence_and_task_failures() -> None:
    def runner(case: EvalCase, seed: int) -> EvalObservation:
        del case, seed
        return _observation(
            task_succeeded=False,
            tool_sequence=("diagnose",),
            evidence_ids=(),
            diagnosis_evidence_ids=(),
            policy_violations=1,
            privacy_violations=1,
        )

    report = EvalHarness().evaluate(_suite(), runner, candidate_version="candidate-v1")

    assert report.passed is False
    assert report.metrics.model_dump() == {
        "case_count": 1,
        "task_success": 0.0,
        "tool_sequence": 0.0,
        "evidence_completeness": 0.0,
        "policy_compliance": 0.0,
        "privacy_compliance": 0.0,
        "replay_determinism": 1.0,
        "overall": pytest.approx(1 / 6),
    }


def test_reports_compare_baseline_and_candidate_without_hidden_state() -> None:
    harness = EvalHarness()

    def failed(case: EvalCase, seed: int) -> EvalObservation:
        del case, seed
        return _observation(task_succeeded=False)

    baseline = harness.evaluate(_suite(), failed, candidate_version="baseline-v1")
    candidate = harness.evaluate(
        _suite(),
        lambda case, seed: _observation(),
        candidate_version="candidate-v2",
    )

    comparison = harness.compare(baseline, candidate)

    assert comparison.compatible is True
    assert comparison.candidate_passed is True
    assert comparison.regressed_metrics == ()
    assert comparison.deltas["task_success"] == 1.0


def test_eval_dtos_are_strict_frozen_and_privacy_safe() -> None:
    with pytest.raises(ValidationError):
        EvalCase(
            case_id="unsafe-case",
            objective="/private/customer/data.parquet",
            expected_tool_sequence=_SEQUENCE,
        )
    with pytest.raises(ValidationError):
        EvalObservation.model_validate(
            {
                "case_id": "complete-diagnosis",
                "task_succeeded": "yes",
                "tool_sequence": _SEQUENCE,
                "evidence_ids": (),
                "diagnosis_evidence_ids": (),
                "policy_violations": 0,
                "privacy_violations": 0,
            }
        )
    with pytest.raises(ValidationError):
        EvalSuite.model_validate(
            {
                "suite_id": "suite-v1",
                "seed": DEFAULT_EVAL_SEED,
                "cases": (),
                "frozen": False,
            }
        )


def test_harness_rejects_suite_seed_different_from_its_fixed_seed() -> None:
    with pytest.raises(ValueError, match="eval seed does not match frozen suite"):
        EvalHarness(seed=7).evaluate(
            _suite(seed=8),
            lambda case, seed: _observation(),
            candidate_version="candidate-v1",
        )


def test_report_rejects_metrics_that_do_not_match_frozen_case_results() -> None:
    report = EvalHarness().evaluate(
        _suite(),
        lambda case, seed: _observation(),
        candidate_version="candidate-v1",
    )
    payload = report.model_dump(mode="python")
    payload["metrics"]["task_success"] = 0.0
    payload["metrics"]["overall"] = 5 / 6
    payload["report_hash"] = ""

    with pytest.raises(ValidationError, match="metrics must match"):
        type(report).model_validate(payload)
