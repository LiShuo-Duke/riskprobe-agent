"""Versioned deterministic evaluation contracts with task-specific metrics."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from riskprobe.evals.models import (
    DEFAULT_EVAL_SEED,
    EvalCase,
    EvalCaseResult,
    EvalMetrics,
    EvalObservation,
)
from riskprobe.privacy import canonical_payload_hash

_PUBLIC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BASE_METRIC_NAMES = (
    "task_success",
    "tool_sequence",
    "evidence_completeness",
    "policy_compliance",
    "privacy_compliance",
    "replay_determinism",
)


class _StrictV2DTO(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )


def _validated_ids(value: tuple[str, ...], *, label: str, ordered: bool = False) -> tuple[str, ...]:
    if len(value) != len(set(value)):
        raise ValueError(f"{label} must contain unique IDs")
    if any(_PUBLIC_ID.fullmatch(item) is None for item in value):
        raise ValueError(f"{label} must contain opaque public IDs")
    return value if ordered else tuple(sorted(value))


def _validate_public_id(value: str, *, label: str) -> str:
    if _PUBLIC_ID.fullmatch(value) is None:
        raise ValueError(f"{label} must be a public identifier")
    return value


def _validate_hash(value: str, *, label: str) -> str:
    if value and _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA-256 identifier")
    return value


def _ratio(numerator: int, denominator: int) -> float:
    """Return zero for every zero denominator used by Eval v2."""

    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _rates(true_positives: int, false_positives: int, false_negatives: int) -> tuple[float, float, float]:
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    return precision, recall, _f1(precision, recall)


def _matches(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= 1e-12


class EvalCaseV2(_StrictV2DTO):
    """A v2 case composed with, but not projected into, the frozen v1 case."""

    base_case: EvalCase
    expected_rule_ids: tuple[str, ...]
    drift_universe_ids: tuple[str, ...]
    drift_ground_truth_ids: tuple[str, ...]
    diagnosis_relevant_ids: tuple[str, ...]
    diagnosis_k: int = Field(gt=0)
    expected_recommendation_ids: tuple[str, ...]
    case_hash: str = ""

    @field_validator(
        "expected_rule_ids",
        "drift_universe_ids",
        "drift_ground_truth_ids",
        "diagnosis_relevant_ids",
        "expected_recommendation_ids",
    )
    @classmethod
    def validate_id_sets(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validated_ids(value, label="case ID collection")

    @field_validator("case_hash")
    @classmethod
    def validate_case_hash(cls, value: str) -> str:
        return _validate_hash(value, label="case_hash")

    @model_validator(mode="after")
    def validate_and_hash(self) -> EvalCaseV2:
        if not set(self.drift_ground_truth_ids).issubset(self.drift_universe_ids):
            raise ValueError("drift ground truth must be contained in the universe")
        payload = self.model_dump(mode="json", exclude={"case_hash"})
        expected = canonical_payload_hash(payload)
        if self.case_hash and self.case_hash != expected:
            raise ValueError("case_hash does not match frozen v2 case")
        object.__setattr__(self, "case_hash", expected)
        return self

    @property
    def case_id(self) -> str:
        return self.base_case.case_id

    def verify_integrity(self) -> bool:
        payload = self.model_dump(mode="json", exclude={"case_hash"})
        return canonical_payload_hash(payload) == self.case_hash


class EvalSuiteV2(_StrictV2DTO):
    suite_id: str
    seed: int = Field(default=DEFAULT_EVAL_SEED, ge=0)
    cases: tuple[EvalCaseV2, ...]
    frozen: Literal[True] = True
    suite_hash: str = ""

    @field_validator("suite_id")
    @classmethod
    def validate_suite_id(cls, value: str) -> str:
        return _validate_public_id(value, label="suite_id")

    @field_validator("cases")
    @classmethod
    def validate_cases(cls, value: tuple[EvalCaseV2, ...]) -> tuple[EvalCaseV2, ...]:
        if not value or len({case.case_id for case in value}) != len(value):
            raise ValueError("v2 eval suite requires unique non-empty cases")
        if any(not case.verify_integrity() for case in value):
            raise ValueError("v2 eval suite contains an invalid case")
        return value

    @field_validator("suite_hash")
    @classmethod
    def validate_suite_hash(cls, value: str) -> str:
        return _validate_hash(value, label="suite_hash")

    @model_validator(mode="after")
    def derive_suite_hash(self) -> EvalSuiteV2:
        payload = self.model_dump(mode="json", exclude={"suite_hash"})
        expected = canonical_payload_hash(payload)
        if self.suite_hash and self.suite_hash != expected:
            raise ValueError("suite_hash does not match frozen v2 suite")
        object.__setattr__(self, "suite_hash", expected)
        return self

    def verify_integrity(self) -> bool:
        payload = self.model_dump(mode="json", exclude={"suite_hash"})
        return self.frozen is True and canonical_payload_hash(payload) == self.suite_hash


class EvalObservationV2(_StrictV2DTO):
    """V2 output; only diagnosis_ranked_ids has order-dependent semantics."""

    base_observation: EvalObservation
    recovered_rule_ids: tuple[str, ...]
    detected_drift_ids: tuple[str, ...]
    diagnosis_ranked_ids: tuple[str, ...]
    recommendation_ids: tuple[str, ...]
    observation_hash: str = ""

    @field_validator("recovered_rule_ids", "detected_drift_ids", "recommendation_ids")
    @classmethod
    def validate_id_sets(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validated_ids(value, label="observation ID collection")

    @field_validator("diagnosis_ranked_ids")
    @classmethod
    def validate_ranked_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validated_ids(value, label="diagnosis ranking", ordered=True)

    @field_validator("observation_hash")
    @classmethod
    def validate_observation_hash(cls, value: str) -> str:
        return _validate_hash(value, label="observation_hash")

    @model_validator(mode="after")
    def derive_observation_hash(self) -> EvalObservationV2:
        payload = self.model_dump(mode="json", exclude={"observation_hash"})
        expected = canonical_payload_hash(payload)
        if self.observation_hash and self.observation_hash != expected:
            raise ValueError("observation_hash does not match frozen v2 observation")
        object.__setattr__(self, "observation_hash", expected)
        return self

    @property
    def case_id(self) -> str:
        return self.base_observation.case_id

    def verify_integrity(self) -> bool:
        payload = self.model_dump(mode="json", exclude={"observation_hash"})
        return canonical_payload_hash(payload) == self.observation_hash


class _SetMetricsV2(_StrictV2DTO):
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_rates(self) -> _SetMetricsV2:
        precision, recall, f1 = _rates(
            self.true_positives,
            self.false_positives,
            self.false_negatives,
        )
        if not all(
            (
                _matches(self.precision, precision),
                _matches(self.recall, recall),
                _matches(self.f1, f1),
            )
        ):
            raise ValueError("set metric rates must match confusion counts")
        return self


class RuleRecoveryMetricsV2(_SetMetricsV2):
    pass


class DriftConfusionMetricsV2(_StrictV2DTO):
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    true_negatives: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    false_positive_rate: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_rates(self) -> DriftConfusionMetricsV2:
        precision = _ratio(self.true_positives, self.true_positives + self.false_positives)
        recall = _ratio(self.true_positives, self.true_positives + self.false_negatives)
        false_positive_rate = _ratio(
            self.false_positives,
            self.false_positives + self.true_negatives,
        )
        if not all(
            (
                _matches(self.precision, precision),
                _matches(self.recall, recall),
                _matches(self.false_positive_rate, false_positive_rate),
            )
        ):
            raise ValueError("drift rates must match confusion counts")
        return self


class DiagnosisMetricsV2(_StrictV2DTO):
    k: int = Field(gt=0)
    relevant_count: int = Field(ge=0)
    retrieved_relevant_count: int = Field(ge=0)
    hit_at_k: float = Field(ge=0, le=1)
    recall_at_k: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_rates(self) -> DiagnosisMetricsV2:
        if self.retrieved_relevant_count > min(self.relevant_count, self.k):
            raise ValueError("retrieved relevant count exceeds the Top-K boundary")
        hit_at_k = float(self.retrieved_relevant_count > 0)
        recall_at_k = _ratio(self.retrieved_relevant_count, self.relevant_count)
        if not (
            _matches(self.hit_at_k, hit_at_k) and _matches(self.recall_at_k, recall_at_k)
        ):
            raise ValueError("diagnosis rates must match Top-K counts")
        return self


class RecommendationMetricsV2(_SetMetricsV2):
    exact: bool

    @model_validator(mode="after")
    def validate_exact(self) -> RecommendationMetricsV2:
        expected = self.false_positives == 0 and self.false_negatives == 0
        if self.exact is not expected:
            raise ValueError("recommendation exact flag must match confusion counts")
        return self


class DiagnosisAggregateV2(_StrictV2DTO):
    case_count: int = Field(ge=1)
    hit_count: int = Field(ge=0)
    relevant_count: int = Field(ge=0)
    retrieved_relevant_count: int = Field(ge=0)
    hit_at_k: float = Field(ge=0, le=1)
    recall_at_k: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_rates(self) -> DiagnosisAggregateV2:
        if self.hit_count > self.case_count or self.retrieved_relevant_count > self.relevant_count:
            raise ValueError("diagnosis aggregate counts are inconsistent")
        hit_at_k = _ratio(self.hit_count, self.case_count)
        recall_at_k = _ratio(self.retrieved_relevant_count, self.relevant_count)
        if not (
            _matches(self.hit_at_k, hit_at_k) and _matches(self.recall_at_k, recall_at_k)
        ):
            raise ValueError("diagnosis aggregate rates must match total counts")
        return self


class RecommendationAggregateV2(_SetMetricsV2):
    case_count: int = Field(ge=1)
    exact_count: int = Field(ge=0)
    exact_rate: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_exact_rate(self) -> RecommendationAggregateV2:
        if self.exact_count > self.case_count:
            raise ValueError("recommendation exact count exceeds case count")
        if not _matches(self.exact_rate, _ratio(self.exact_count, self.case_count)):
            raise ValueError("recommendation exact rate must match total counts")
        return self


class EvalCaseResultV2(_StrictV2DTO):
    case_id: str
    case_hash: str
    observation_hash: str
    base_result: EvalCaseResult
    rule_recovery: RuleRecoveryMetricsV2
    drift_confusion: DriftConfusionMetricsV2
    diagnosis: DiagnosisMetricsV2
    recommendation: RecommendationMetricsV2

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str) -> str:
        return _validate_public_id(value, label="case_id")

    @field_validator("case_hash", "observation_hash")
    @classmethod
    def validate_hashes(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("v2 case result hashes must be SHA-256 identifiers")
        return value

    @model_validator(mode="after")
    def validate_base_result(self) -> EvalCaseResultV2:
        if self.base_result.case_id != self.case_id:
            raise ValueError("v1 base result must match the v2 case")
        return self


class EvalAggregateV2(_StrictV2DTO):
    base_metrics: EvalMetrics
    rule_recovery: RuleRecoveryMetricsV2
    drift_confusion: DriftConfusionMetricsV2
    diagnosis: DiagnosisAggregateV2
    recommendation: RecommendationAggregateV2


class EvalReportV2(_StrictV2DTO):
    suite_id: str
    suite_hash: str
    seed: int = Field(ge=0)
    candidate_version: str
    case_results: tuple[EvalCaseResultV2, ...]
    aggregate: EvalAggregateV2
    frozen: Literal[True] = True
    report_hash: str = ""

    @field_validator("suite_id", "candidate_version")
    @classmethod
    def validate_public_ids(cls, value: str) -> str:
        return _validate_public_id(value, label="report identifier")

    @field_validator("suite_hash", "report_hash")
    @classmethod
    def validate_hashes(cls, value: str) -> str:
        return _validate_hash(value, label="report hash")

    @field_validator("case_results")
    @classmethod
    def validate_case_results(
        cls,
        value: tuple[EvalCaseResultV2, ...],
    ) -> tuple[EvalCaseResultV2, ...]:
        if not value or len({result.case_id for result in value}) != len(value):
            raise ValueError("v2 report requires unique non-empty case results")
        return value

    @model_validator(mode="after")
    def validate_and_hash(self) -> EvalReportV2:
        expected_aggregate = _aggregate(self.case_results)
        if self.aggregate != expected_aggregate:
            raise ValueError("v2 aggregate must be recomputed from total case counts")
        payload = self.model_dump(mode="json", exclude={"report_hash"})
        expected_hash = canonical_payload_hash(payload)
        if self.report_hash and self.report_hash != expected_hash:
            raise ValueError("report_hash does not match frozen v2 report")
        object.__setattr__(self, "report_hash", expected_hash)
        return self

    def verify_integrity(self) -> bool:
        payload = self.model_dump(mode="json", exclude={"report_hash"})
        return self.frozen is True and canonical_payload_hash(payload) == self.report_hash


@runtime_checkable
class EvalRunnerV2(Protocol):
    def run_case(self, case: EvalCaseV2, seed: int) -> EvalObservationV2: ...


RunnerCallableV2 = Callable[[EvalCaseV2, int], EvalObservationV2]


class EvalHarnessV2:
    """Replay v2 observations and compute deterministic per-case and micro metrics."""

    def __init__(self, *, seed: int = DEFAULT_EVAL_SEED) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("eval seed must be a non-negative integer")
        self.seed = seed

    def evaluate(
        self,
        suite: EvalSuiteV2,
        runner: EvalRunnerV2 | RunnerCallableV2 | object,
        *,
        candidate_version: str,
    ) -> EvalReportV2:
        if not isinstance(suite, EvalSuiteV2) or not suite.verify_integrity():
            raise ValueError("v2 eval suite is not frozen")
        if suite.seed != self.seed:
            raise ValueError("eval seed does not match frozen v2 suite")

        results: list[EvalCaseResultV2] = []
        for case in suite.cases:
            first = self._run_case(runner, case)
            replayed = self._run_case(runner, case)
            self._validate_observation(case, first)
            self._validate_observation(case, replayed)
            replay_deterministic = first.observation_hash == replayed.observation_hash
            base_result = _score_base(case.base_case, first.base_observation, replay_deterministic)
            results.append(
                EvalCaseResultV2(
                    case_id=case.case_id,
                    case_hash=case.case_hash,
                    observation_hash=first.observation_hash,
                    base_result=base_result,
                    rule_recovery=_score_rule_recovery(case, first),
                    drift_confusion=_score_drift(case, first),
                    diagnosis=_score_diagnosis(case, first),
                    recommendation=_score_recommendation(case, first),
                )
            )

        case_results = tuple(results)
        return EvalReportV2(
            suite_id=suite.suite_id,
            suite_hash=suite.suite_hash,
            seed=self.seed,
            candidate_version=candidate_version,
            case_results=case_results,
            aggregate=_aggregate(case_results),
        )

    run = evaluate

    def _run_case(
        self,
        runner: EvalRunnerV2 | RunnerCallableV2 | object,
        case: EvalCaseV2,
    ) -> EvalObservationV2:
        method = getattr(runner, "run_case", None)
        if callable(method):
            result = method(case, seed=self.seed)
        elif callable(runner):
            result = runner(case, seed=self.seed)
        else:
            raise TypeError("v2 runner must be callable or implement run_case")
        if not isinstance(result, EvalObservationV2):
            raise TypeError("v2 runner must return EvalObservationV2")
        if not result.verify_integrity():
            raise ValueError("v2 observation integrity check failed")
        return result

    @staticmethod
    def _validate_observation(case: EvalCaseV2, observation: EvalObservationV2) -> None:
        if observation.case_id != case.case_id:
            raise ValueError("runner returned a v2 observation for a different case")
        outside = set(observation.detected_drift_ids).difference(case.drift_universe_ids)
        if outside:
            raise ValueError("detected drift IDs are outside the case universe")


def _set_counts(expected: tuple[str, ...], observed: tuple[str, ...]) -> tuple[int, int, int]:
    expected_set = set(expected)
    observed_set = set(observed)
    return (
        len(expected_set & observed_set),
        len(observed_set - expected_set),
        len(expected_set - observed_set),
    )


def _score_base(
    case: EvalCase,
    observation: EvalObservation,
    replay_deterministic: bool,
) -> EvalCaseResult:
    evidence_complete = set(case.required_evidence_ids).issubset(observation.evidence_ids)
    if case.require_diagnosis:
        evidence_complete = evidence_complete and bool(observation.diagnosis_evidence_ids)
    gates = {
        "task_succeeded": observation.task_succeeded,
        "tool_sequence_matched": observation.tool_sequence == case.expected_tool_sequence,
        "evidence_complete": evidence_complete,
        "policy_compliant": observation.policy_violations == 0,
        "privacy_compliant": observation.privacy_violations == 0,
        "replay_deterministic": replay_deterministic,
    }
    return EvalCaseResult(case_id=case.case_id, **gates, passed=all(gates.values()))


def _score_rule_recovery(case: EvalCaseV2, observation: EvalObservationV2) -> RuleRecoveryMetricsV2:
    true_positives, false_positives, false_negatives = _set_counts(
        case.expected_rule_ids,
        observation.recovered_rule_ids,
    )
    precision, recall, f1 = _rates(true_positives, false_positives, false_negatives)
    return RuleRecoveryMetricsV2(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _score_drift(case: EvalCaseV2, observation: EvalObservationV2) -> DriftConfusionMetricsV2:
    true_positives, false_positives, false_negatives = _set_counts(
        case.drift_ground_truth_ids,
        observation.detected_drift_ids,
    )
    ground_truth = set(case.drift_ground_truth_ids)
    detected = set(observation.detected_drift_ids)
    true_negatives = len(set(case.drift_universe_ids) - ground_truth - detected)
    return _drift_metrics(true_positives, false_positives, false_negatives, true_negatives)


def _score_diagnosis(case: EvalCaseV2, observation: EvalObservationV2) -> DiagnosisMetricsV2:
    top_k = observation.diagnosis_ranked_ids[: case.diagnosis_k]
    relevant_count = len(case.diagnosis_relevant_ids)
    retrieved_relevant_count = len(set(top_k).intersection(case.diagnosis_relevant_ids))
    return DiagnosisMetricsV2(
        k=case.diagnosis_k,
        relevant_count=relevant_count,
        retrieved_relevant_count=retrieved_relevant_count,
        hit_at_k=float(retrieved_relevant_count > 0),
        recall_at_k=_ratio(retrieved_relevant_count, relevant_count),
    )


def _score_recommendation(
    case: EvalCaseV2,
    observation: EvalObservationV2,
) -> RecommendationMetricsV2:
    true_positives, false_positives, false_negatives = _set_counts(
        case.expected_recommendation_ids,
        observation.recommendation_ids,
    )
    precision, recall, f1 = _rates(true_positives, false_positives, false_negatives)
    return RecommendationMetricsV2(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        exact=false_positives == 0 and false_negatives == 0,
    )


def _drift_metrics(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
    true_negatives: int,
) -> DriftConfusionMetricsV2:
    return DriftConfusionMetricsV2(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
        precision=_ratio(true_positives, true_positives + false_positives),
        recall=_ratio(true_positives, true_positives + false_negatives),
        false_positive_rate=_ratio(false_positives, false_positives + true_negatives),
    )


def _aggregate(results: tuple[EvalCaseResultV2, ...]) -> EvalAggregateV2:
    count = len(results)
    base_values = {
        "task_success": sum(result.base_result.task_succeeded for result in results) / count,
        "tool_sequence": sum(result.base_result.tool_sequence_matched for result in results) / count,
        "evidence_completeness": sum(result.base_result.evidence_complete for result in results) / count,
        "policy_compliance": sum(result.base_result.policy_compliant for result in results) / count,
        "privacy_compliance": sum(result.base_result.privacy_compliant for result in results) / count,
        "replay_determinism": sum(result.base_result.replay_deterministic for result in results) / count,
    }
    base_metrics = EvalMetrics(
        case_count=count,
        **base_values,
        overall=sum(base_values.values()) / len(_BASE_METRIC_NAMES),
    )

    rule_tp = sum(result.rule_recovery.true_positives for result in results)
    rule_fp = sum(result.rule_recovery.false_positives for result in results)
    rule_fn = sum(result.rule_recovery.false_negatives for result in results)
    rule_precision, rule_recall, rule_f1 = _rates(rule_tp, rule_fp, rule_fn)
    rule_recovery = RuleRecoveryMetricsV2(
        true_positives=rule_tp,
        false_positives=rule_fp,
        false_negatives=rule_fn,
        precision=rule_precision,
        recall=rule_recall,
        f1=rule_f1,
    )

    drift_confusion = _drift_metrics(
        sum(result.drift_confusion.true_positives for result in results),
        sum(result.drift_confusion.false_positives for result in results),
        sum(result.drift_confusion.false_negatives for result in results),
        sum(result.drift_confusion.true_negatives for result in results),
    )

    diagnosis_hit_count = sum(result.diagnosis.retrieved_relevant_count > 0 for result in results)
    diagnosis_relevant = sum(result.diagnosis.relevant_count for result in results)
    diagnosis_retrieved = sum(result.diagnosis.retrieved_relevant_count for result in results)
    diagnosis = DiagnosisAggregateV2(
        case_count=count,
        hit_count=diagnosis_hit_count,
        relevant_count=diagnosis_relevant,
        retrieved_relevant_count=diagnosis_retrieved,
        hit_at_k=_ratio(diagnosis_hit_count, count),
        recall_at_k=_ratio(diagnosis_retrieved, diagnosis_relevant),
    )

    recommendation_tp = sum(result.recommendation.true_positives for result in results)
    recommendation_fp = sum(result.recommendation.false_positives for result in results)
    recommendation_fn = sum(result.recommendation.false_negatives for result in results)
    recommendation_precision, recommendation_recall, recommendation_f1 = _rates(
        recommendation_tp,
        recommendation_fp,
        recommendation_fn,
    )
    exact_count = sum(result.recommendation.exact for result in results)
    recommendation = RecommendationAggregateV2(
        true_positives=recommendation_tp,
        false_positives=recommendation_fp,
        false_negatives=recommendation_fn,
        precision=recommendation_precision,
        recall=recommendation_recall,
        f1=recommendation_f1,
        case_count=count,
        exact_count=exact_count,
        exact_rate=_ratio(exact_count, count),
    )

    return EvalAggregateV2(
        base_metrics=base_metrics,
        rule_recovery=rule_recovery,
        drift_confusion=drift_confusion,
        diagnosis=diagnosis,
        recommendation=recommendation,
    )


__all__ = [
    "DiagnosisAggregateV2",
    "DiagnosisMetricsV2",
    "DriftConfusionMetricsV2",
    "EvalAggregateV2",
    "EvalCaseResultV2",
    "EvalCaseV2",
    "EvalHarnessV2",
    "EvalObservationV2",
    "EvalReportV2",
    "EvalRunnerV2",
    "EvalSuiteV2",
    "RecommendationAggregateV2",
    "RecommendationMetricsV2",
    "RuleRecoveryMetricsV2",
    "RunnerCallableV2",
]
