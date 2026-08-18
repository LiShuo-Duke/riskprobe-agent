import math

import pytest
from pydantic import ValidationError

from riskprobe.evals import (
    DEFAULT_EVAL_SEED,
    EvalCase,
    EvalCaseV2,
    EvalHarness,
    EvalHarnessV2,
    EvalObservation,
    EvalObservationV2,
    EvalSuite,
    EvalSuiteV2,
)

_EVIDENCE_A = "a" * 64
_SEQUENCE = ("inspect", "diagnose", "discover", "recommend", "review")
_V1_SUITE_HASH = "61cf802a6b5360be2b8c6188a45a6947c96937fd40280b9f2f00f9c6949fe958"
_V1_REPORT_HASH = "85eaa6c8ed0b3751906142dc3aa0c8871faf7e668c4daed78ebe7b469aeba897"


def _base_case(case_id: str = "case-v2", *, require_diagnosis: bool = True) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        objective="comprehensive",
        expected_tool_sequence=_SEQUENCE,
        required_evidence_ids=(_EVIDENCE_A,) if require_diagnosis else (),
        require_diagnosis=require_diagnosis,
    )


def _base_observation(
    case_id: str = "case-v2",
    *,
    require_diagnosis: bool = True,
) -> EvalObservation:
    return EvalObservation(
        case_id=case_id,
        task_succeeded=True,
        tool_sequence=_SEQUENCE,
        evidence_ids=(_EVIDENCE_A,) if require_diagnosis else (),
        diagnosis_evidence_ids=(_EVIDENCE_A,) if require_diagnosis else (),
        policy_violations=0,
        privacy_violations=0,
    )


def _case(case_id: str = "case-v2", **updates: object) -> EvalCaseV2:
    payload: dict[str, object] = {
        "base_case": _base_case(case_id),
        "expected_rule_ids": ("rule-a", "rule-b"),
        "drift_universe_ids": ("drift-a", "drift-b", "drift-c", "drift-d"),
        "drift_ground_truth_ids": ("drift-a", "drift-b"),
        "diagnosis_relevant_ids": ("cause-a", "cause-b"),
        "diagnosis_k": 2,
        "expected_recommendation_ids": ("action-a", "action-b"),
    }
    payload.update(updates)
    return EvalCaseV2(**payload)


def _observation(case_id: str = "case-v2", **updates: object) -> EvalObservationV2:
    payload: dict[str, object] = {
        "base_observation": _base_observation(case_id),
        "recovered_rule_ids": ("rule-a", "rule-extra"),
        "detected_drift_ids": ("drift-a", "drift-c"),
        "diagnosis_ranked_ids": ("cause-b", "cause-x", "cause-a"),
        "recommendation_ids": ("action-a", "action-extra"),
    }
    payload.update(updates)
    return EvalObservationV2(**payload)


def _evaluate(
    cases: tuple[EvalCaseV2, ...],
    observations: dict[str, EvalObservationV2],
):
    suite = EvalSuiteV2(suite_id="offline-v2", cases=cases)

    def runner(case: EvalCaseV2, seed: int) -> EvalObservationV2:
        assert seed == DEFAULT_EVAL_SEED
        return observations[case.base_case.case_id]

    return EvalHarnessV2().evaluate(suite, runner, candidate_version="candidate-v2")


def test_v2_scores_rule_drift_diagnosis_and_recommendation_formulas() -> None:
    report = _evaluate((_case(),), {"case-v2": _observation()})
    result = report.case_results[0]

    assert result.rule_recovery.model_dump() == {
        "true_positives": 1,
        "false_positives": 1,
        "false_negatives": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    assert result.drift_confusion.model_dump() == {
        "true_positives": 1,
        "false_positives": 1,
        "false_negatives": 1,
        "true_negatives": 1,
        "precision": 0.5,
        "recall": 0.5,
        "false_positive_rate": 0.5,
    }
    assert result.diagnosis.model_dump() == {
        "k": 2,
        "relevant_count": 2,
        "retrieved_relevant_count": 1,
        "hit_at_k": 1.0,
        "recall_at_k": 0.5,
    }
    assert result.recommendation.model_dump() == {
        "true_positives": 1,
        "false_positives": 1,
        "false_negatives": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "exact": False,
    }
    assert result.base_result.passed is True


def test_v2_zero_denominators_are_zero_and_empty_recommendations_are_exact() -> None:
    case = _case(
        "empty-case",
        base_case=_base_case("empty-case", require_diagnosis=False),
        expected_rule_ids=(),
        drift_universe_ids=(),
        drift_ground_truth_ids=(),
        diagnosis_relevant_ids=(),
        diagnosis_k=3,
        expected_recommendation_ids=(),
    )
    observation = _observation(
        "empty-case",
        base_observation=_base_observation("empty-case", require_diagnosis=False),
        recovered_rule_ids=(),
        detected_drift_ids=(),
        diagnosis_ranked_ids=(),
        recommendation_ids=(),
    )

    result = _evaluate((case,), {"empty-case": observation}).case_results[0]

    assert (result.rule_recovery.precision, result.rule_recovery.recall, result.rule_recovery.f1) == (
        0.0,
        0.0,
        0.0,
    )
    assert result.drift_confusion.model_dump() == {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "true_negatives": 0,
        "precision": 0.0,
        "recall": 0.0,
        "false_positive_rate": 0.0,
    }
    assert (result.diagnosis.hit_at_k, result.diagnosis.recall_at_k) == (0.0, 0.0)
    assert (result.recommendation.precision, result.recommendation.recall) == (0.0, 0.0)
    assert result.recommendation.f1 == 0.0
    assert result.recommendation.exact is True


def test_diagnosis_preserves_ranking_and_has_deterministic_k_boundaries() -> None:
    first = _observation(
        diagnosis_ranked_ids=("cause-a", "cause-b"),
        recovered_rule_ids=(),
        detected_drift_ids=(),
        recommendation_ids=(),
    )
    second = _observation(
        diagnosis_ranked_ids=("cause-b", "cause-a"),
        recovered_rule_ids=(),
        detected_drift_ids=(),
        recommendation_ids=(),
    )
    k_one = _case(
        expected_rule_ids=(),
        drift_ground_truth_ids=(),
        diagnosis_relevant_ids=("cause-b",),
        diagnosis_k=1,
        expected_recommendation_ids=(),
    )

    first_result = _evaluate((k_one,), {"case-v2": first}).case_results[0].diagnosis
    second_result = _evaluate((k_one,), {"case-v2": second}).case_results[0].diagnosis

    assert first.diagnosis_ranked_ids == ("cause-a", "cause-b")
    assert second.diagnosis_ranked_ids == ("cause-b", "cause-a")
    assert (first_result.hit_at_k, first_result.recall_at_k) == (0.0, 0.0)
    assert (second_result.hit_at_k, second_result.recall_at_k) == (1.0, 1.0)
    assert first.observation_hash != second.observation_hash

    k_past_end = _case(
        expected_rule_ids=(),
        drift_ground_truth_ids=(),
        diagnosis_relevant_ids=("cause-b",),
        diagnosis_k=5,
        expected_recommendation_ids=(),
    )
    past_end = _evaluate((k_past_end,), {"case-v2": first}).case_results[0].diagnosis
    assert (past_end.hit_at_k, past_end.recall_at_k) == (1.0, 1.0)


@pytest.mark.parametrize("invalid_k", [0, -1, True, 1.0, "1"])
def test_v2_rejects_non_positive_or_non_integer_k(invalid_k: object) -> None:
    with pytest.raises(ValidationError):
        _case(diagnosis_k=invalid_k)


@pytest.mark.parametrize(
    ("field", "duplicate"),
    [
        ("expected_rule_ids", "rule-a"),
        ("drift_universe_ids", "drift-a"),
        ("drift_ground_truth_ids", "drift-a"),
        ("diagnosis_relevant_ids", "cause-a"),
        ("expected_recommendation_ids", "action-a"),
    ],
)
def test_v2_case_rejects_duplicate_ids(field: str, duplicate: str) -> None:
    with pytest.raises(ValidationError, match="unique"):
        _case(**{field: (duplicate, duplicate)})


@pytest.mark.parametrize(
    ("field", "duplicate"),
    [
        ("recovered_rule_ids", "rule-a"),
        ("detected_drift_ids", "drift-a"),
        ("diagnosis_ranked_ids", "cause-a"),
        ("recommendation_ids", "action-a"),
    ],
)
def test_v2_observation_rejects_duplicate_ids(field: str, duplicate: str) -> None:
    with pytest.raises(ValidationError, match="unique"):
        _observation(**{field: (duplicate, duplicate)})


def test_drift_requires_explicit_valid_universe_and_rejects_outside_ids() -> None:
    payload = _case().model_dump(mode="python", exclude={"case_hash", "drift_universe_ids"})
    with pytest.raises(ValidationError):
        EvalCaseV2.model_validate(payload)

    with pytest.raises(ValidationError, match="ground truth.*universe"):
        _case(drift_ground_truth_ids=("drift-outside",))

    with pytest.raises(ValueError, match="outside.*universe"):
        _evaluate((_case(),), {"case-v2": _observation(detected_drift_ids=("drift-outside",))})


def test_v2_dtos_are_strict_frozen_extra_forbid_finite_and_hash_checked() -> None:
    case = _case()
    observation = _observation()
    report = _evaluate((case,), {"case-v2": observation})

    with pytest.raises(ValidationError):
        EvalCaseV2.model_validate(
            {
                **case.model_dump(mode="python", exclude={"case_hash"}),
                "expected_rule_ids": ["rule-a"],
            }
        )
    with pytest.raises(ValidationError):
        EvalObservationV2.model_validate(
            {
                **observation.model_dump(mode="python", exclude={"observation_hash"}),
                "unexpected": True,
            }
        )
    with pytest.raises(ValidationError):
        observation.recovered_rule_ids = ()

    metric = report.case_results[0].rule_recovery
    with pytest.raises(ValidationError):
        type(metric).model_validate({**metric.model_dump(), "precision": float("nan")})
    with pytest.raises(ValidationError, match="observation_hash does not match"):
        EvalObservationV2.model_validate(
            {**observation.model_dump(mode="python"), "observation_hash": "0" * 64}
        )


def test_v2_aggregate_is_micro_recomputed_from_total_counts() -> None:
    case_one = _case(
        "micro-one",
        base_case=_base_case("micro-one"),
        expected_rule_ids=("rule-one",),
        drift_universe_ids=("drift-one", "drift-two"),
        drift_ground_truth_ids=("drift-one",),
        diagnosis_relevant_ids=("cause-one",),
        diagnosis_k=1,
        expected_recommendation_ids=("action-one",),
    )
    observation_one = _observation(
        "micro-one",
        base_observation=_base_observation("micro-one"),
        recovered_rule_ids=("rule-one",),
        detected_drift_ids=("drift-one",),
        diagnosis_ranked_ids=("cause-one",),
        recommendation_ids=("action-one",),
    )
    case_two = _case(
        "micro-two",
        base_case=_base_case("micro-two"),
        expected_rule_ids=("rule-two",),
        drift_universe_ids=("drift-three", "drift-four", "drift-five", "drift-six"),
        drift_ground_truth_ids=("drift-three",),
        diagnosis_relevant_ids=("cause-two", "cause-three", "cause-four"),
        diagnosis_k=1,
        expected_recommendation_ids=("action-two",),
    )
    observation_two = _observation(
        "micro-two",
        base_observation=_base_observation("micro-two"),
        recovered_rule_ids=("rule-two", "rule-x", "rule-y", "rule-z"),
        detected_drift_ids=("drift-three", "drift-four", "drift-five", "drift-six"),
        diagnosis_ranked_ids=("cause-two",),
        recommendation_ids=("action-two", "action-x", "action-y", "action-z"),
    )

    report = _evaluate(
        (case_one, case_two),
        {"micro-one": observation_one, "micro-two": observation_two},
    )
    aggregate = report.aggregate

    assert len(report.case_results) == 2
    assert aggregate.rule_recovery.true_positives == 2
    assert aggregate.rule_recovery.false_positives == 3
    assert aggregate.rule_recovery.precision == 0.4
    assert aggregate.rule_recovery.precision != pytest.approx((1.0 + 0.25) / 2)
    assert aggregate.rule_recovery.recall == 1.0
    assert aggregate.rule_recovery.f1 == pytest.approx(4 / 7)
    assert aggregate.drift_confusion.true_negatives == 1
    assert aggregate.drift_confusion.false_positive_rate == 0.75
    assert aggregate.diagnosis.hit_at_k == 1.0
    assert aggregate.diagnosis.recall_at_k == 0.5
    assert aggregate.recommendation.precision == 0.4
    assert aggregate.recommendation.exact_rate == 0.5
    assert aggregate.base_metrics.case_count == 2


def test_v2_hashes_are_canonical_stable_and_replay_includes_diagnosis_order() -> None:
    case = _case()
    reordered_case = _case(
        expected_rule_ids=("rule-b", "rule-a"),
        drift_universe_ids=("drift-d", "drift-c", "drift-b", "drift-a"),
        drift_ground_truth_ids=("drift-b", "drift-a"),
        diagnosis_relevant_ids=("cause-b", "cause-a"),
        expected_recommendation_ids=("action-b", "action-a"),
    )
    observation = _observation()
    reordered_sets = _observation(
        recovered_rule_ids=("rule-extra", "rule-a"),
        detected_drift_ids=("drift-c", "drift-a"),
        recommendation_ids=("action-extra", "action-a"),
    )

    assert case.case_hash == reordered_case.case_hash
    assert observation.observation_hash == reordered_sets.observation_hash
    assert case.verify_integrity() is True
    assert observation.verify_integrity() is True

    suite = EvalSuiteV2(suite_id="stable-v2", cases=(case,))
    first_report = EvalHarnessV2().evaluate(
        suite,
        lambda evaluated_case, seed: observation,
        candidate_version="candidate-v2",
    )
    second_report = EvalHarnessV2().evaluate(
        suite,
        lambda evaluated_case, seed: observation,
        candidate_version="candidate-v2",
    )
    assert first_report.report_hash == second_report.report_hash
    assert first_report.verify_integrity() is True

    calls = 0

    def reordered_replay(evaluated_case: EvalCaseV2, seed: int) -> EvalObservationV2:
        nonlocal calls
        del evaluated_case, seed
        calls += 1
        if calls == 1:
            return _observation(diagnosis_ranked_ids=("cause-a", "cause-b"))
        return _observation(diagnosis_ranked_ids=("cause-b", "cause-a"))

    replay_report = EvalHarnessV2().evaluate(
        suite,
        reordered_replay,
        candidate_version="candidate-v2",
    )
    assert replay_report.case_results[0].base_result.replay_deterministic is False


def test_v1_suite_and_report_hashes_remain_unchanged_after_v2_import() -> None:
    suite = EvalSuite(
        suite_id="comprehensive-offline-v1",
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
    observation = EvalObservation(
        case_id="complete-diagnosis",
        task_succeeded=True,
        tool_sequence=_SEQUENCE,
        evidence_ids=(_EVIDENCE_A,),
        diagnosis_evidence_ids=(_EVIDENCE_A,),
        policy_violations=0,
        privacy_violations=0,
    )
    harness = EvalHarness()
    first = harness.evaluate(
        suite,
        lambda case, seed: observation,
        candidate_version="candidate-v1",
    )
    second = harness.evaluate(
        suite,
        lambda case, seed: observation,
        candidate_version="candidate-v1",
    )

    assert suite.suite_hash == _V1_SUITE_HASH
    assert first.report_hash == _V1_REPORT_HASH
    assert second.report_hash == _V1_REPORT_HASH
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert math.isfinite(first.metrics.overall)
