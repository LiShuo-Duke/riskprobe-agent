from pathlib import Path

import pytest
from pydantic import ValidationError

from riskprobe.agents.contracts import (
    ExecutionPlan,
    PlanStep,
    ReviewReason,
)
from riskprobe.agents.planner import Planner, PlanningError
from riskprobe.agents.reviewer import Reviewer
from riskprobe.tools import (
    DiagnoseRequest,
    DiscoverRequest,
    InspectRequest,
    RecommendRequest,
)

_EVIDENCE_A = "a" * 64
_EVIDENCE_B = "b" * 64
_TOOL_TYPES = {
    "inspect": InspectRequest,
    "diagnose": DiagnoseRequest,
    "discover": DiscoverRequest,
    "recommend": RecommendRequest,
}


def _plan() -> ExecutionPlan:
    return Planner(allowed_tools=_TOOL_TYPES).plan(
        objective="comprehensive",
        dataset_id="synthetic_demo",
    )


def test_planner_builds_only_the_injected_typed_comprehensive_sequence() -> None:
    plan = _plan()

    assert plan.objective == "comprehensive"
    assert plan.tool_sequence == (
        "inspect",
        "diagnose",
        "discover",
        "recommend",
        "review",
    )
    assert isinstance(plan.steps[0].request, InspectRequest)
    assert isinstance(plan.steps[1].request, DiagnoseRequest)
    assert isinstance(plan.steps[2].request, DiscoverRequest)
    assert isinstance(plan.steps[3].request, RecommendRequest)
    assert plan.steps[-1].request is None


def test_planner_fails_closed_when_required_tool_is_not_injected() -> None:
    with pytest.raises(PlanningError, match="required typed tool is unavailable"):
        Planner(allowed_tools={"inspect": InspectRequest}).plan(
            objective="comprehensive",
            dataset_id="synthetic_demo",
        )


@pytest.mark.parametrize(
    "objective",
    [
        "/private/data.parquet",
        "run shell command",
        "SELECT * FROM customers",
        "```python import os```",
    ],
)
def test_planner_never_turns_paths_shell_sql_or_code_into_a_plan(objective: str) -> None:
    with pytest.raises(PlanningError, match="unsupported safe objective"):
        Planner(allowed_tools=_TOOL_TYPES).plan(
            objective=objective,
            dataset_id="synthetic_demo",
        )


def test_plan_step_requires_matching_typed_request_and_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        PlanStep(
            step_id="inspect",
            tool_name="inspect",
            request=DiscoverRequest(dataset_id="synthetic_demo"),
        )
    with pytest.raises(ValidationError):
        PlanStep.model_validate(
            {
                "step_id": "inspect",
                "tool_name": "inspect",
                "request": {"dataset_id": "synthetic_demo"},
                "shell": "rm -rf /",
            }
        )
    with pytest.raises(ValidationError):
        InspectRequest(dataset_id=Path("relative.parquet"))


def test_reviewer_approves_complete_safe_diagnosis_evidence() -> None:
    decision = Reviewer().review(
        _plan(),
        evidence_ids=(_EVIDENCE_A, _EVIDENCE_B),
        diagnosis_evidence_ids=(_EVIDENCE_A,),
        claimed_evidence_ids=(_EVIDENCE_A,),
        metadata_grade="A",
        payloads=({"finding_count": 1},),
        retry_count=0,
    )

    assert decision.approved is True
    assert decision.reason_codes == ()
    assert decision.retry_allowed is False


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        (
            {"evidence_ids": (), "diagnosis_evidence_ids": ()},
            ReviewReason.MISSING_EVIDENCE,
        ),
        (
            {
                "evidence_ids": (_EVIDENCE_A,),
                "diagnosis_evidence_ids": (_EVIDENCE_A,),
                "permission_denied": True,
            },
            ReviewReason.PERMISSION_DENIED,
        ),
        (
            {
                "evidence_ids": (_EVIDENCE_A,),
                "diagnosis_evidence_ids": (_EVIDENCE_A,),
                "payloads": ({"path": "/private/data.parquet"},),
            },
            ReviewReason.UNSAFE_PAYLOAD,
        ),
        (
            {
                "evidence_ids": (_EVIDENCE_A,),
                "diagnosis_evidence_ids": (_EVIDENCE_A,),
                "retry_count": 2,
            },
            ReviewReason.RETRY_LIMIT_EXCEEDED,
        ),
        (
            {"evidence_ids": (_EVIDENCE_A,), "diagnosis_evidence_ids": ()},
            ReviewReason.MISSING_DIAGNOSIS,
        ),
        (
            {
                "evidence_ids": (_EVIDENCE_A,),
                "diagnosis_evidence_ids": (_EVIDENCE_A,),
                "claimed_evidence_ids": (_EVIDENCE_B,),
            },
            ReviewReason.EVIDENCE_MISMATCH,
        ),
    ],
)
def test_reviewer_deterministically_rejects_unsafe_or_unsupported_decisions(
    kwargs: dict[str, object],
    reason: ReviewReason,
) -> None:
    call_kwargs: dict[str, object] = {
        "evidence_ids": (_EVIDENCE_A,),
        "diagnosis_evidence_ids": (_EVIDENCE_A,),
        "metadata_grade": "A",
        "retry_count": 0,
    }
    call_kwargs.update(kwargs)
    decision = Reviewer().review(_plan(), **call_kwargs)

    assert decision.approved is False
    assert reason in decision.reason_codes


def test_reviewer_rejects_grade_b_production_action() -> None:
    base = _plan()
    production_step = base.steps[3].model_copy(update={"production_action": True})
    plan = ExecutionPlan(
        objective=base.objective,
        dataset_id=base.dataset_id,
        steps=(*base.steps[:3], production_step, base.steps[4]),
        component_versions=base.component_versions,
    )

    decision = Reviewer().review(
        plan,
        evidence_ids=(_EVIDENCE_A,),
        diagnosis_evidence_ids=(_EVIDENCE_A,),
        metadata_grade="B",
    )

    assert decision.approved is False
    assert ReviewReason.GRADE_B_PRODUCTION_ACTION in decision.reason_codes


def test_reviewer_rejects_diagnosis_reference_not_present_in_evidence() -> None:
    decision = Reviewer().review(
        _plan(),
        evidence_ids=(_EVIDENCE_A,),
        diagnosis_evidence_ids=(_EVIDENCE_B,),
        metadata_grade="A",
    )

    assert decision.approved is False
    assert ReviewReason.EVIDENCE_MISMATCH in decision.reason_codes


def test_planner_repair_is_reason_coded_and_bounded() -> None:
    planner = Planner(allowed_tools=_TOOL_TYPES)
    decision = Reviewer().review(
        _plan(),
        evidence_ids=(),
        diagnosis_evidence_ids=(),
        metadata_grade="A",
        retry_count=0,
    )

    repaired = planner.repair(
        objective="comprehensive",
        dataset_id="synthetic_demo",
        decision=decision,
        retry_count=0,
    )

    assert repaired == _plan()
    with pytest.raises(PlanningError, match="repair retry limit exceeded"):
        planner.repair(
            objective="comprehensive",
            dataset_id="synthetic_demo",
            decision=decision,
            retry_count=1,
        )


def test_planner_repair_rejects_non_recoverable_review() -> None:
    planner = Planner(allowed_tools=_TOOL_TYPES)
    decision = Reviewer().review(
        _plan(),
        evidence_ids=(_EVIDENCE_A,),
        diagnosis_evidence_ids=(_EVIDENCE_A,),
        metadata_grade="A",
        permission_denied=True,
        retry_count=0,
    )

    with pytest.raises(PlanningError, match="review reason is not repairable"):
        planner.repair(
            objective="comprehensive",
            dataset_id="synthetic_demo",
            decision=decision,
            retry_count=0,
        )
