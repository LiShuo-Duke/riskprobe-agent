import hashlib
import json
from pathlib import Path

import pytest

from riskprobe.analysis_contracts import (
    AnalysisSummary,
    StageName,
    StageStatus,
    StageSummary,
)
from riskprobe.agents.contracts import (
    AgentResult,
    AgentState,
    AgentStatus,
    ExecutionPlan,
    PlanStep,
    ReviewDecision,
)
from riskprobe.agents.results import AgentResultIntegrityError, AgentResultStore
from riskprobe.tools import (
    DiagnoseRequest,
    DiscoverRequest,
    InspectRequest,
    RecommendRequest,
)


def _analysis_summary() -> AnalysisSummary:
    return AnalysisSummary(
        stages=tuple(
            StageSummary(
                name=name,
                status=StageStatus.NOT_RUN,
                enabled=True,
                output_available=False,
            )
            for name in StageName
        )
    )


def _result(*, analysis_summary: AnalysisSummary | None) -> AgentResult:
    return AgentResult(
        session_id="result-run",
        status=AgentStatus.SUCCEEDED,
        plan=ExecutionPlan(
            objective="comprehensive",
            dataset_id="synthetic_demo",
            steps=(
                PlanStep(
                    step_id="inspect",
                    tool_name="inspect",
                    request=InspectRequest(dataset_id="synthetic_demo"),
                ),
                PlanStep(
                    step_id="diagnose",
                    tool_name="diagnose",
                    request=DiagnoseRequest(dataset_id="synthetic_demo"),
                ),
                PlanStep(
                    step_id="discover",
                    tool_name="discover",
                    request=DiscoverRequest(dataset_id="synthetic_demo"),
                ),
                PlanStep(
                    step_id="recommend",
                    tool_name="recommend",
                    request=RecommendRequest(dataset_id="synthetic_demo"),
                ),
                PlanStep(step_id="review", tool_name="review"),
            ),
            component_versions={"planner": "planner-v1"},
        ),
        review=ReviewDecision(approved=True, no_action_required=True),
        tool_sequence=("inspect", "diagnose", "discover", "recommend", "review"),
        retry_count=0,
        state_history=(
            AgentState.PLANNING,
            AgentState.EXECUTING,
            AgentState.COLLECTING_EVIDENCE,
            AgentState.REVIEWING,
            AgentState.COMPLETED,
        ),
        leaf_node_id="a" * 64,
        redacted_summary="no actionable diagnosis",
        analysis_summary=analysis_summary,
    )


def test_agent_result_store_round_trips_current_analysis_summary(
    tmp_path: Path,
) -> None:
    result = _result(analysis_summary=_analysis_summary())
    store = AgentResultStore(tmp_path / "agent-result.json")

    store.publish(result)

    envelope = json.loads(store.path.read_text(encoding="utf-8"))
    assert envelope["result"]["analysis_summary"] == result.analysis_summary.model_dump(
        mode="json"
    )
    assert store.load() == result


def test_agent_result_store_loads_legacy_shape_without_rewriting(
    tmp_path: Path,
) -> None:
    result = _result(analysis_summary=None)
    store = AgentResultStore(tmp_path / "agent-result.json")

    store.publish(result)
    before = store.path.read_bytes()
    envelope = json.loads(before)

    assert "analysis_summary" not in envelope["result"]
    assert store.load() == result
    assert store.path.read_bytes() == before


def test_agent_result_store_rejects_null_current_analysis_summary(
    tmp_path: Path,
) -> None:
    store = AgentResultStore(tmp_path / "agent-result.json")
    store.publish(_result(analysis_summary=_analysis_summary()))
    envelope = json.loads(store.path.read_text(encoding="utf-8"))
    envelope["result"]["analysis_summary"] = None
    result_json = json.dumps(
        envelope["result"],
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    envelope["result_sha256"] = hashlib.sha256(
        result_json.encode("utf-8")
    ).hexdigest()
    store.path.write_text(
        json.dumps(
            envelope,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        AgentResultIntegrityError,
        match="^agent result is unavailable$",
    ):
        store.load()
