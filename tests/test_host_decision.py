from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event, Thread

import pytest

from riskprobe.agents.contracts import (
    AgentResult,
    AgentState,
    AgentStatus,
    ExecutionPlan,
    PlanStep,
    ReviewDecision,
)
from riskprobe.agents.decision_contracts import (
    DecisionContext,
    DecisionFinding,
    DecisionPolicy,
    DecisionProposal,
    DecisionSource,
)
from riskprobe.monitoring.models import FindingKind, FindingSeverity, RiskFinding
from riskprobe.recommendations.policy import ActionCode
from riskprobe.agents.decision_providers import (
    DecisionProviderConfig,
    DecisionProviderMode,
    DeterministicDecisionProvider,
)
from riskprobe.host_decision import (
    HostDecisionCoordinator,
    HostDecisionError,
)
from riskprobe.policy import Budget, Principal, Role
from riskprobe.service import RiskProbeService
from riskprobe.tools import (
    DiagnoseRequest,
    DiscoverRequest,
    InspectRequest,
    RecommendRequest,
)


def test_host_decision_pauses_for_context_then_resumes_fixed_flow(
    tmp_path: Path,
    synthetic_config: object,
) -> None:
    runs_dir = tmp_path / "runs"
    state_dir = tmp_path / "state"
    coordinator = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=state_dir,
    )
    service = RiskProbeService(
        config=synthetic_config,
        runs_dir=runs_dir,
        state_dir=state_dir,
        decision_provider_config=DecisionProviderConfig(
            mode=DecisionProviderMode.EXTERNAL_HOST,
            provider_id=coordinator.provider_id,
            provider_version=coordinator.version,
        ),
        decision_provider=coordinator,
    )
    principal = Principal(principal_id="kiro-host", role=Role.ANALYST)

    def run_agent():
        return service.orchestrate(
            dataset_id=synthetic_config.dataset.id,
            principal=principal,
            budget=Budget(max_queries=16),
        )

    pending = coordinator.get_context(
        idempotency_key="decision-1",
        runner=run_agent,
    )
    deterministic = DeterministicDecisionProvider().resolve(
        context=pending.context
    ).proposal
    assert deterministic is not None
    proposal = DecisionProposal(
        context_id=pending.context.context_id,
        diagnosis_evidence_ids=pending.context.diagnosis_evidence_ids,
        action_codes=deterministic.action_codes,
        source=DecisionSource.EXTERNAL_HOST,
        source_version=coordinator.version,
    )

    outcome = coordinator.submit_proposal(
        idempotency_key="decision-1",
        proposal=proposal,
    )

    assert pending.phase == "awaiting_proposal"
    assert outcome.phase == "terminal"
    assert outcome.decision_status == "accepted"
    assert outcome.reason_codes == ()
    assert outcome.action_codes == deterministic.action_codes
    assert outcome.context_evidence_id
    assert outcome.proposal_evidence_id
    assert outcome.result_evidence_id
    assert outcome.expires_at == pending.context.expires_at
    assert set(outcome.model_dump()) == {
        "protocol_version",
        "phase",
        "context_id",
        "agent_result",
        "decision_status",
        "reason_codes",
        "action_codes",
        "context_evidence_id",
        "proposal_evidence_id",
        "result_evidence_id",
        "expires_at",
    }
    assert outcome.agent_result.review.approved is True
    assert outcome.agent_result.tool_sequence == (
        "inspect",
        "diagnose",
        "discover",
        "recommend",
        "review",
    )
    assert coordinator.get_context(
        idempotency_key="decision-1",
        runner=run_agent,
    ) == pending
    assert coordinator.submit_proposal(
        idempotency_key="decision-1",
        proposal=proposal,
    ) == outcome
    assert {path.name for path in (runs_dir / outcome.agent_result.session_id).iterdir()} == {
        "manifest.json",
        "metadata_report.json",
        "data_profile.json",
        "candidate_rules.parquet",
        "evidence_cards.json",
        "risk_report.md",
    }


def _host_context(session_id: str, *, ttl_seconds: float = 30.0) -> DecisionContext:
    finding = RiskFinding(
        kind=FindingKind.FEATURE_DRIFT,
        severity=FindingSeverity.WARNING,
        code="feature_psi",
        metrics={"affected_rate": 0.25},
    )
    evidence_id = "a" * 64
    now = datetime.now(UTC)
    return DecisionContext(
        session_id=session_id,
        attempt=0,
        anchor_node_id="f" * 64,
        dataset_id="synthetic_demo",
        metadata_grade="A",
        row_count=100,
        feature_count=8,
        diagnosis_evidence_ids=(evidence_id,),
        findings=(DecisionFinding(evidence_id=evidence_id, finding=finding),),
        policy=DecisionPolicy(context_ttl_seconds=30),
        issued_at=now,
        expires_at=now + timedelta(seconds=ttl_seconds),
        component_versions={
            "diagnostics": "diagnostics-v1",
            "orchestrator": "orchestrator-v1",
            "planner": "planner-v1",
            "recommendations": "recommendations-v1",
        },
    )


def _host_result(session_id: str) -> AgentResult:
    plan = ExecutionPlan(
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
                request=RecommendRequest(
                    dataset_id="synthetic_demo",
                    evidence_ids=(),
                ),
                requires_evidence=True,
            ),
            PlanStep(step_id="review", tool_name="review"),
        ),
        component_versions={"planner": "planner-v1"},
    )
    return AgentResult(
        session_id=session_id,
        status=AgentStatus.SUCCEEDED,
        plan=plan,
        review=ReviewDecision(approved=True),
        tool_sequence=("review",),
        retry_count=0,
        state_history=(AgentState.PLANNING, AgentState.REVIEWING, AgentState.COMPLETED),
        leaf_node_id="b" * 64,
        redacted_summary="host decision accepted",
    )


def _runner(coordinator: HostDecisionCoordinator, context: DecisionContext):
    def run() -> AgentResult:
        coordinator.resolve(context=context)
        return _host_result(context.session_id)

    return run


def _proposal_for(context: DecisionContext) -> DecisionProposal:
    return DecisionProposal(
        context_id=context.context_id,
        diagnosis_evidence_ids=context.diagnosis_evidence_ids,
        action_codes=(ActionCode.INVESTIGATE_FEATURE_DRIFT,),
        source=DecisionSource.EXTERNAL_HOST,
        source_version="gpt5.6sol",
    )


def test_coordinator_supports_independent_idempotency_keys(tmp_path: Path) -> None:
    coordinator = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=tmp_path,
    )
    first_context = _host_context("first-run", ttl_seconds=0.05)
    second_context = _host_context("second-run", ttl_seconds=0.05)

    first = coordinator.get_context(
        idempotency_key="first-key",
        runner=_runner(coordinator, first_context),
    )
    second = coordinator.get_context(
        idempotency_key="second-key",
        runner=_runner(coordinator, second_context),
    )

    assert first.context.context_id != second.context.context_id


def test_completed_session_replays_after_coordinator_restart_and_rejects_conflict(
    tmp_path: Path,
) -> None:
    context_coordinator = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=tmp_path,
    )
    context = _host_context("restart-run")
    pending = context_coordinator.get_context(
        idempotency_key="restart-key",
        runner=_runner(context_coordinator, context),
    )
    proposal = _proposal_for(pending.context)
    outcome = context_coordinator.submit_proposal(
        idempotency_key="restart-key",
        proposal=proposal,
    )

    restarted = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=tmp_path,
    )
    assert restarted.get_context(
        idempotency_key="restart-key",
        runner=lambda: _host_result("unused"),
    ) == pending
    assert restarted.submit_proposal(
        idempotency_key="restart-key",
        proposal=proposal,
    ) == outcome

    with pytest.raises(HostDecisionError, match="^host decision is unavailable$"):
        restarted.submit_proposal(
            idempotency_key="restart-key",
            proposal=proposal.model_copy(
                update={"action_codes": (ActionCode.REVIEW_RULE_EVIDENCE,)}
            ),
        )


def test_terminal_wait_is_bounded_by_context_expiry(tmp_path: Path) -> None:
    coordinator = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=tmp_path,
    )
    context = _host_context("expiry-run", ttl_seconds=0.05)
    pending = coordinator.get_context(
        idempotency_key="expiry-key",
        runner=lambda: (
            coordinator.resolve(context=context),
            Event().wait(),
            _host_result(context.session_id),
        )[-1],
    )
    proposal = _proposal_for(pending.context)
    finished = Event()
    error: list[BaseException] = []

    def submit() -> None:
        try:
            coordinator.submit_proposal(
                idempotency_key="expiry-key",
                proposal=proposal,
            )
        except BaseException as caught:
            error.append(caught)
        finally:
            finished.set()

    Thread(target=submit, daemon=True).start()
    assert finished.wait(timeout=0.5)
    assert error and isinstance(error[0], HostDecisionError)
