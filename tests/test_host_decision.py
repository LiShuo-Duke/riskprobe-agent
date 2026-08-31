import hashlib
import json
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
from riskprobe.agents.results import AgentResultStore
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
from riskprobe import host_decision
from riskprobe.analysis_contracts import (
    AnalysisSummary,
    DecisionSummary,
    StageName,
    StageStatus,
    StageSummary,
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


def _host_stack(
    tmp_path: Path,
    synthetic_config: object,
) -> tuple[HostDecisionCoordinator, RiskProbeService, Principal]:
    state_dir = tmp_path / "state"
    coordinator = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=state_dir,
    )
    service = RiskProbeService(
        config=synthetic_config,
        runs_dir=tmp_path / "runs",
        state_dir=state_dir,
        decision_provider_config=DecisionProviderConfig(
            mode=DecisionProviderMode.EXTERNAL_HOST,
            provider_id=coordinator.provider_id,
            provider_version=coordinator.version,
        ),
        decision_provider=coordinator,
    )
    return (
        coordinator,
        service,
        Principal(principal_id="kiro-host", role=Role.ANALYST),
    )


def _get_host_context(
    coordinator: HostDecisionCoordinator,
    service: RiskProbeService,
    principal: Principal,
    synthetic_config: object,
    *,
    idempotency_key: str,
):
    return coordinator.get_context(
        idempotency_key=idempotency_key,
        runner=lambda: service.orchestrate(
            dataset_id=synthetic_config.dataset.id,
            principal=principal,
            budget=Budget(max_queries=16),
        ),
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
        "analysis_summary",
        "decision_summary",
    }
    assert outcome.analysis_summary is not None
    assert len(outcome.analysis_summary.stages) == 24
    assert next(
        stage for stage in outcome.analysis_summary.stages if stage.name.value == "finalize"
    ).status.value == "succeeded"
    assert outcome.decision_summary is not None
    assert outcome.decision_summary.recommendations
    assert outcome.decision_summary.selected_action_codes == tuple(
        action.value for action in outcome.action_codes
    )
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

    replayed_context = coordinator.get_context(
        idempotency_key="decision-2",
        runner=run_agent,
    )
    assert replayed_context == pending
    assert coordinator.report_subject(idempotency_key="decision-2") is None
    restarted = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=state_dir,
    )
    with pytest.raises(HostDecisionError, match="^host decision is unavailable$"):
        restarted.submit_proposal(
            idempotency_key="decision-2",
            proposal=proposal.model_copy(
                update={"action_codes": (ActionCode.REVIEW_RULE_EVIDENCE,)}
            ),
        )
    assert restarted.submit_proposal(
        idempotency_key="decision-2",
        proposal=proposal,
    ) == outcome

    store = host_decision._HostSessionStore(state_dir)
    for legacy_review in (False, True):
        legacy_outcome = {
            key: value
            for key, value in outcome.model_dump(mode="json").items()
            if key not in {"analysis_summary", "decision_summary"}
        }
        legacy_result = dict(legacy_outcome["agent_result"])
        legacy_result.pop("analysis_summary")
        if legacy_review:
            review = dict(legacy_result["review"])
            review.pop("no_action_required")
            legacy_result["review"] = review
        legacy_outcome["agent_result"] = legacy_result
        legacy_key = f"legacy-evidence-{legacy_review}"
        store.save(
            host_decision._StoredHostSession(
                key=legacy_key,
                provider_id="kiro",
                provider_version="gpt5.6sol",
                lifecycle="terminal",
                report_run_id=outcome.agent_result.session_id,
                context=pending,
                proposal=proposal,
                outcome=legacy_outcome,
            )
        )
        before = store.path.read_bytes()
        legacy_coordinator = HostDecisionCoordinator(
            provider_id="kiro",
            version="gpt5.6sol",
            state_dir=state_dir,
        )
        assert legacy_coordinator.get_context(
            idempotency_key=legacy_key,
            runner=lambda: (_ for _ in ()).throw(
                AssertionError("legacy replay must not rerun")
            ),
        ) == pending
        assert legacy_coordinator.report_subject(idempotency_key=legacy_key) is None
        assert legacy_coordinator.submit_proposal(
            idempotency_key=legacy_key,
            proposal=proposal,
        ).model_dump(mode="json") == legacy_outcome
        assert store.path.read_bytes() == before

    assert {path.name for path in (runs_dir / outcome.agent_result.session_id).iterdir()} == {
        "manifest.json",
        "metadata_report.json",
        "data_profile.json",
        "candidate_rules.parquet",
        "evidence_cards.json",
        "risk_report.md",
        "analysis_summary.json",
    }


def test_finalized_replay_rolls_back_proposal_after_persistence_failure(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    context = _host_context("finalized-persist-retry")
    outcome = HostDecisionCoordinator._outcome_from_payload(
        _current_outcome_payload(context)
    )
    coordinator = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=tmp_path,
    )
    session = host_decision._SessionState(
        key="finalized-persist-retry-key",
        report_run_id=context.session_id,
        context=host_decision.HostDecisionContext(
            provider_id="kiro",
            provider_version="gpt5.6sol",
            context=context,
        ),
        outcome=outcome,
        done=True,
    )
    coordinator._sessions[session.key] = session
    persist_calls = 0

    def fail_once(candidate: object) -> None:
        nonlocal persist_calls
        assert candidate is session
        persist_calls += 1
        if persist_calls == 1:
            raise HostDecisionError("host decision is unavailable")

    monkeypatch.setattr(coordinator, "_persist", fail_once)
    monkeypatch.setattr(
        coordinator,
        "_validated_terminal_replay",
        lambda *_args: True,
    )
    proposal = _proposal_for(context)

    with pytest.raises(HostDecisionError, match="^host decision is unavailable$"):
        coordinator.submit_proposal(
            idempotency_key=session.key,
            proposal=proposal,
        )

    assert session.proposal is None
    assert coordinator.submit_proposal(
        idempotency_key=session.key,
        proposal=proposal,
    ) == outcome
    assert persist_calls == 2


def test_report_subject_rejects_deleted_persisted_recommendations(
    tmp_path: Path,
    synthetic_config: object,
) -> None:
    coordinator, service, principal = _host_stack(tmp_path, synthetic_config)
    pending = _get_host_context(
        coordinator,
        service,
        principal,
        synthetic_config,
        idempotency_key="tampered-recommendations-key",
    )
    proposal = DeterministicDecisionProvider().resolve(
        context=pending.context
    ).proposal
    assert proposal is not None
    outcome = coordinator.submit_proposal(
        idempotency_key="tampered-recommendations-key",
        proposal=DecisionProposal(
            context_id=pending.context.context_id,
            diagnosis_evidence_ids=pending.context.diagnosis_evidence_ids,
            action_codes=proposal.action_codes,
            source=DecisionSource.EXTERNAL_HOST,
            source_version=coordinator.version,
        ),
    )
    assert outcome.decision_summary.recommendations
    assert coordinator.report_subject(
        idempotency_key="tampered-recommendations-key"
    ) is not None

    payload = outcome.model_dump(mode="json")
    decision_summary = dict(payload["decision_summary"])
    decision_summary["recommendations"] = []
    payload["decision_summary"] = decision_summary
    host_decision._HostSessionStore(tmp_path / "state").save(
        host_decision._StoredHostSession(
            key="tampered-recommendations-key",
            provider_id="kiro",
            provider_version="gpt5.6sol",
            lifecycle="terminal",
            report_run_id=outcome.agent_result.session_id,
            context=pending,
            proposal=DecisionProposal(
                context_id=pending.context.context_id,
                diagnosis_evidence_ids=pending.context.diagnosis_evidence_ids,
                action_codes=proposal.action_codes,
                source=DecisionSource.EXTERNAL_HOST,
                source_version=coordinator.version,
            ),
            outcome=payload,
        )
    )
    restarted = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=tmp_path / "state",
    )

    with pytest.raises(HostDecisionError, match="^host decision is unavailable$"):
        restarted.report_subject(idempotency_key="tampered-recommendations-key")


def test_report_subject_accepts_policy_rejected_four_step_terminal(
    tmp_path: Path,
    synthetic_config: object,
) -> None:
    coordinator, service, principal = _host_stack(tmp_path, synthetic_config)
    pending = _get_host_context(
        coordinator,
        service,
        principal,
        synthetic_config,
        idempotency_key="policy-rejected-key",
    )
    proposal = DecisionProposal(
        context_id=pending.context.context_id,
        diagnosis_evidence_ids=pending.context.diagnosis_evidence_ids,
        action_codes=(),
        source=DecisionSource.EXTERNAL_HOST,
        source_version=coordinator.version,
    )

    outcome = coordinator.submit_proposal(
        idempotency_key="policy-rejected-key",
        proposal=proposal,
    )

    assert outcome.decision_status is host_decision.DecisionStatus.REJECTED
    assert outcome.reason_codes == (
        host_decision.DecisionReason.ACTION_COUNT_INVALID,
    )
    assert outcome.action_codes == ()
    assert outcome.agent_result.status is AgentStatus.REJECTED
    assert outcome.agent_result.tool_sequence == (
        "inspect",
        "diagnose",
        "discover",
        "review",
    )
    subject = coordinator.report_subject(idempotency_key="policy-rejected-key")
    assert subject is not None
    assert subject.terminal_status == "rejected"


def _agent_analysis_summary() -> AnalysisSummary:
    succeeded = {
        StageName.CONFIG,
        StageName.SNAPSHOT,
        StageName.PROFILE,
        StageName.PARTITION,
        StageName.DISCOVERY,
        StageName.WOE,
        StageName.SCORECARD,
        StageName.VALIDATION,
        StageName.INSTITUTION_ANALYSIS,
        StageName.REPORT,
        StageName.FINALIZE,
        StageName.INSPECT,
        StageName.DIAGNOSE_QUALITY,
        StageName.DIAGNOSE_FEATURE_DRIFT,
        StageName.DIAGNOSE_POPULATION_SHIFT,
        StageName.DIAGNOSE_TARGET_SHIFT,
        StageName.DIAGNOSE_SEGMENT_RISK,
        StageName.DISCOVER_RESTORE,
    }
    stages = []
    for name in StageName:
        if name in succeeded:
            stages.append(
                StageSummary(
                    name=name,
                    status=StageStatus.SUCCEEDED,
                    enabled=True,
                    output_available=True,
                )
            )
        elif name is StageName.DIAGNOSE_TIME_STABILITY:
            stages.append(
                StageSummary(
                    name=name,
                    status=StageStatus.SKIPPED,
                    enabled=False,
                    output_available=False,
                    reason_code="time_validation_disabled",
                )
            )
        elif name is StageName.DIAGNOSE_RULE_EVIDENCE:
            stages.append(
                StageSummary(
                    name=name,
                    status=StageStatus.NOT_RUN,
                    enabled=False,
                    output_available=False,
                    reason_code="rule_evidence_not_run",
                )
            )
        else:
            stages.append(
                StageSummary(
                    name=name,
                    status=StageStatus.NOT_RUN,
                    enabled=True,
                    output_available=False,
                )
            )
    return AnalysisSummary(stages=tuple(stages))


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


def _host_result(session_id: str, *, no_action_required: bool = False) -> AgentResult:
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
        review=ReviewDecision(
            approved=True,
            no_action_required=no_action_required,
        ),
        tool_sequence=("review",),
        retry_count=0,
        state_history=(AgentState.PLANNING, AgentState.REVIEWING, AgentState.COMPLETED),
        leaf_node_id="b" * 64,
        redacted_summary="host decision accepted",
    )


def _clean_no_action_result(session_id: str) -> AgentResult:
    base = _host_result(session_id, no_action_required=True)
    return AgentResult(
        session_id=base.session_id,
        status=base.status,
        plan=base.plan,
        review=base.review,
        tool_sequence=(
            "inspect",
            "diagnose",
            "discover",
            "recommend",
            "review",
        ),
        evidence_ids=base.evidence_ids,
        diagnosis_evidence_ids=base.diagnosis_evidence_ids,
        retry_count=base.retry_count,
        state_history=base.state_history,
        leaf_node_id=base.leaf_node_id,
        redacted_summary=base.redacted_summary,
        analysis_summary=_agent_analysis_summary(),
    )


def _legacy_clean_no_action_result(session_id: str) -> AgentResult:
    return _host_result(session_id, no_action_required=True).model_copy(
        update={
            "tool_sequence": (
                "inspect",
                "diagnose",
                "discover",
                "recommend",
                "review",
            )
        }
    )


def test_no_action_terminal_replays_without_context_or_proposal(tmp_path: Path) -> None:
    coordinator = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=tmp_path,
    )

    outcome = coordinator.get_context(
        idempotency_key="no-action-key",
        runner=lambda: _clean_no_action_result("no-action-run"),
    )

    outcome_type = getattr(host_decision, "HostDecisionNoActionOutcome", None)
    assert outcome_type is not None
    assert type(outcome) is outcome_type
    assert set(outcome.model_dump()) == {
        "protocol_version",
        "phase",
        "terminal_reason",
        "action_codes",
        "agent_result",
        "analysis_summary",
        "decision_summary",
    }
    assert outcome.phase == "terminal"
    assert outcome.terminal_reason == "no_actionable_diagnosis"
    assert outcome.action_codes == ()
    assert outcome.analysis_summary is not None
    assert len(outcome.analysis_summary.stages) == 24
    assert len({stage.name for stage in outcome.analysis_summary.stages}) == 24
    stages = {stage.name: stage for stage in outcome.analysis_summary.stages}
    source_stages = {
        stage.name: stage for stage in outcome.agent_result.analysis_summary.stages
    }
    for name in (
        StageName.DIAGNOSE_QUALITY,
        StageName.DIAGNOSE_FEATURE_DRIFT,
        StageName.DIAGNOSE_POPULATION_SHIFT,
        StageName.DIAGNOSE_TARGET_SHIFT,
        StageName.DIAGNOSE_SEGMENT_RISK,
        StageName.DIAGNOSE_TIME_STABILITY,
        StageName.DIAGNOSE_RULE_EVIDENCE,
    ):
        assert stages[name] == source_stages[name]
    assert stages[StageName.DIAGNOSE_RULE_EVIDENCE].status is StageStatus.NOT_RUN
    assert stages[StageName.DIAGNOSE_TIME_STABILITY].status is StageStatus.SKIPPED
    assert stages[StageName.DECISION_CONTEXT].status is StageStatus.SKIPPED
    assert stages[StageName.DECISION_CONTEXT].reason_code == "no_action_required"
    assert stages[StageName.RECOMMEND].status is StageStatus.SKIPPED
    assert stages[StageName.RECOMMEND].reason_code == "no_action_required"
    assert stages[StageName.REVIEW].status is StageStatus.SUCCEEDED
    assert stages[StageName.TERMINAL].status is StageStatus.SUCCEEDED
    assert outcome.decision_summary == DecisionSummary(
        selected_action_codes=(),
        recommendations=(),
        recommendation_status=StageStatus.SKIPPED,
        review_approved=True,
        review_reason_codes=(),
        no_action_required=True,
        retry_count=0,
        tool_sequence=(
            "inspect",
            "diagnose",
            "discover",
            "recommend",
            "review",
        ),
        evidence_complete=True,
        decision_status="no_action",
        final_status="succeeded",
    )
    assert not hasattr(outcome, "context_id")
    assert not hasattr(outcome, "proposal")
    assert not hasattr(outcome, "diagnosis_evidence_ids")
    assert not hasattr(outcome, "artifact_location")
    assert coordinator.get_context(
        idempotency_key="no-action-key",
        runner=lambda: _host_result("unused"),
    ) == outcome
    assert coordinator.submit_proposal(idempotency_key="no-action-key") == outcome

    restarted = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=tmp_path,
    )
    assert restarted.get_context(
        idempotency_key="no-action-key",
        runner=lambda: _host_result("unused"),
    ) == outcome
    assert restarted.submit_proposal(idempotency_key="no-action-key") == outcome


def test_no_action_outcome_rejects_dirty_results_and_hybrid_payloads(
    tmp_path: Path,
) -> None:
    outcome_type = getattr(host_decision, "HostDecisionNoActionOutcome", None)
    assert outcome_type is not None
    clean = _clean_no_action_result("clean-no-action-run")
    coordinator = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=tmp_path,
    )
    terminal = coordinator.get_context(
        idempotency_key="clean-no-action-shape",
        runner=lambda: clean,
    )
    assert type(terminal) is outcome_type

    for dirty in (
        clean.model_copy(update={"status": AgentStatus.REJECTED}),
        clean.model_copy(
            update={"review": ReviewDecision(approved=True, no_action_required=False)}
        ),
        clean.model_copy(update={"evidence_ids": ("a" * 64,)}),
        clean.model_copy(update={"diagnosis_evidence_ids": ("a" * 64,)}),
        clean.model_copy(update={"tool_sequence": ("review",)}),
    ):
        with pytest.raises(ValueError):
            outcome_type(
                agent_result=dirty,
                analysis_summary=terminal.analysis_summary,
                decision_summary=terminal.decision_summary,
            )

    payload = terminal.model_dump(mode="json")
    from pydantic_core import PydanticSerializationError

    with pytest.raises(
        PydanticSerializationError,
        match="current no-action outcome requires canonical payload",
    ):
        terminal.model_dump(
            mode="json",
            exclude={
                "analysis_summary": True,
                "decision_summary": True,
                "agent_result": {"analysis_summary": True},
            },
        )

    tampered_summary = dict(payload["analysis_summary"])
    tampered_stages = list(tampered_summary["stages"])
    terminal_index = next(
        index
        for index, stage in enumerate(tampered_stages)
        if stage["name"] == "terminal"
    )
    tampered_stages[terminal_index] = {
        **tampered_stages[terminal_index],
        "status": "not_run",
        "output_available": False,
    }
    tampered_summary["stages"] = tampered_stages
    missing_recommendations = dict(payload["decision_summary"])
    missing_recommendations.pop("recommendations")
    for malformed in (
        {key: value for key, value in payload.items() if key != "terminal_reason"},
        {key: value for key, value in payload.items() if key != "analysis_summary"},
        {key: value for key, value in payload.items() if key != "decision_summary"},
        {**payload, "analysis_summary": None},
        {**payload, "decision_summary": None},
        {**payload, "decision_summary": missing_recommendations},
        {**payload, "analysis_summary": tampered_summary},
        {**payload, "context_id": "a" * 64},
        {**payload, "terminal_reason": "unknown"},
    ):
        with pytest.raises(HostDecisionError, match="^host decision is unavailable$"):
            HostDecisionCoordinator._outcome_from_payload(malformed)


def test_legacy_no_action_replays_exactly_without_rewrite(tmp_path: Path) -> None:
    result = _legacy_clean_no_action_result("legacy-no-action-run")
    result_payload = result.model_dump(mode="json")
    result_payload.pop("analysis_summary", None)
    legacy_outcome = {
        "protocol_version": "riskprobe.host-decision.v1",
        "phase": "terminal",
        "terminal_reason": "no_actionable_diagnosis",
        "action_codes": [],
        "agent_result": result_payload,
    }
    store = host_decision._HostSessionStore(tmp_path)
    store.save(
        host_decision._StoredHostSession(
            key="legacy-no-action-key",
            provider_id="kiro",
            provider_version="gpt5.6sol",
            lifecycle="terminal",
            report_run_id=result.session_id,
            outcome=legacy_outcome,
        )
    )
    before = store.path.read_bytes()
    runner_calls = 0

    def reject_rerun() -> AgentResult:
        nonlocal runner_calls
        runner_calls += 1
        raise AssertionError("legacy no-action replay must not rerun")

    coordinator = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=tmp_path,
    )
    replayed = coordinator.get_context(
        idempotency_key="legacy-no-action-key",
        runner=reject_rerun,
    )

    assert type(replayed) is not host_decision.HostDecisionNoActionOutcome
    assert replayed.model_dump(mode="json") == legacy_outcome
    assert coordinator.submit_proposal(
        idempotency_key="legacy-no-action-key"
    ).model_dump(mode="json") == legacy_outcome
    assert coordinator.report_subject(idempotency_key="legacy-no-action-key") is None
    assert store.path.read_bytes() == before
    assert runner_calls == 0


def test_no_action_decoder_rejects_mixed_current_and_legacy_shapes(
    tmp_path: Path,
) -> None:
    coordinator = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=tmp_path,
    )
    current = coordinator.get_context(
        idempotency_key="current-no-action-mix",
        runner=lambda: _clean_no_action_result("current-no-action-mix"),
    ).model_dump(mode="json")
    legacy = dict(current)
    legacy.pop("analysis_summary")
    legacy.pop("decision_summary")
    legacy_result = dict(legacy["agent_result"])
    legacy_result.pop("analysis_summary")
    legacy["agent_result"] = legacy_result

    mixed_payloads = (
        {**legacy, "agent_result": current["agent_result"]},
        {**current, "agent_result": legacy["agent_result"]},
        {key: value for key, value in current.items() if key != "analysis_summary"},
        {key: value for key, value in current.items() if key != "decision_summary"},
    )
    for payload in mixed_payloads:
        with pytest.raises(HostDecisionError, match="^host decision is unavailable$"):
            HostDecisionCoordinator._outcome_from_payload(payload)


def test_terminal_lifecycle_requires_an_exact_terminal_payload() -> None:
    with pytest.raises(ValueError):
        host_decision._StoredHostSession(
            key="missing-terminal-payload",
            provider_id="kiro",
            provider_version="gpt5.6sol",
            lifecycle="terminal",
        )


@pytest.mark.parametrize(
    ("include_context", "include_proposal"),
    [(True, False), (False, True), (True, True)],
)
def test_no_action_terminal_lifecycle_rejects_outer_context_or_proposal(
    tmp_path: Path,
    include_context: bool,
    include_proposal: bool,
) -> None:
    context = _host_context("stored-no-action")
    source = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=tmp_path / "source",
    )
    outcome = source.get_context(
        idempotency_key="stored-no-action-source",
        runner=lambda: _clean_no_action_result("stored-no-action"),
    ).model_dump(mode="json")
    outer_context = host_decision.HostDecisionContext(
        provider_id="kiro",
        provider_version="gpt5.6sol",
        context=context,
    )
    proposal = _proposal_for(context)

    with pytest.raises(ValueError):
        host_decision._StoredHostSession(
            key="stored-no-action",
            provider_id="kiro",
            provider_version="gpt5.6sol",
            lifecycle="terminal",
            context=outer_context if include_context else None,
            proposal=proposal if include_proposal else None,
            outcome=outcome,
        )

    invalid = host_decision._StoredHostSession.model_construct(
        key="stored-no-action-reload",
        provider_id="kiro",
        provider_version="gpt5.6sol",
        lifecycle="terminal",
        context=outer_context,
        outcome=outcome,
    )
    host_decision._HostSessionStore(tmp_path).save(invalid)

    restarted = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=tmp_path,
    )
    assert restarted.get_context(
        idempotency_key="stored-no-action-reload",
        runner=lambda: _host_result("unused"),
    ) == host_decision.HostDecisionFailure(
        error_code="session_state_unavailable"
    )
    fresh = restarted.get_context(
        idempotency_key="fresh-after-invalid-session",
        runner=_runner(restarted, _host_context("fresh-after-invalid-session")),
    )
    assert fresh.phase == "awaiting_proposal"


@pytest.mark.parametrize("no_action_required", [False, True])
def test_agent_result_store_round_trips_default_and_no_action_reviews(
    tmp_path: Path,
    no_action_required: bool,
) -> None:
    result = _host_result(
        "terminal-result",
        no_action_required=no_action_required,
    )
    store = AgentResultStore(tmp_path / "agent-result.json")

    store.publish(result)

    assert store.load() == result


def test_agent_result_store_loads_legacy_v1_review_without_no_action_field(
    tmp_path: Path,
) -> None:
    result = _host_result("legacy-terminal-result")
    store = AgentResultStore(tmp_path / "agent-result.json")
    store.publish(result)
    envelope = json.loads(store.path.read_text(encoding="utf-8"))

    assert envelope["format_version"] == "riskprobe.agent-result.v1"
    assert envelope["result"]["review"].pop("no_action_required") is False
    result_json = json.dumps(
        envelope["result"],
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    envelope["result_sha256"] = hashlib.sha256(result_json.encode("utf-8")).hexdigest()
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

    assert store.load() == result
    assert store.load().review.no_action_required is False


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


def test_terminal_replay_waits_for_matching_inflight_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=tmp_path,
    )
    context = _host_context("inflight-run")
    host_context = host_decision.HostDecisionContext(
        provider_id=coordinator.provider_id,
        provider_version=coordinator.version,
        context=context,
    )
    proposal = _proposal_for(context)
    outcome = HostDecisionCoordinator._outcome_from_payload(
        _current_outcome_payload(context)
    )
    assert type(outcome) is host_decision.HostDecisionOutcome
    result = outcome.agent_result
    candidate = host_decision._SessionState(
        key="first-key",
        context=host_context,
        proposal=proposal,
        runner_started=True,
    )
    coordinator._sessions[candidate.key] = candidate
    monkeypatch.setattr(
        coordinator,
        "_validated_terminal_replay",
        lambda *_args: True,
    )
    entered = Event()
    finished = Event()
    replay: list[object] = []

    def reconcile() -> None:
        with coordinator._condition:
            entered.set()
            replay.append(coordinator._terminal_replay_for(result))
        finished.set()

    Thread(target=reconcile, daemon=True).start()
    assert entered.wait(timeout=0.5)
    assert not finished.wait(timeout=0.05)
    with coordinator._condition:
        candidate.outcome = outcome
        candidate.done = True
        coordinator._condition.notify_all()
    assert finished.wait(timeout=0.5)
    assert replay == [(host_context, outcome)]


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


def test_runner_without_decision_context_is_projected_as_incomplete_state(
    tmp_path: Path,
) -> None:
    coordinator = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=tmp_path,
    )

    failure = coordinator.get_context(
        idempotency_key="missing-context-key",
        runner=lambda: _host_result("missing-context-run"),
    )

    assert failure.model_dump() == {
        "phase": "context",
        "error_code": "agent_state_incomplete",
    }


def test_runner_tool_failure_without_context_is_projected_as_orchestration_failure(
    tmp_path: Path,
) -> None:
    from riskprobe.agents.contracts import ReviewReason

    coordinator = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=tmp_path,
    )
    failed_result = _host_result("tool-failure-run").model_copy(
        update={
            "status": AgentStatus.REJECTED,
            "review": ReviewDecision(
                approved=False,
                reason_codes=(ReviewReason.TOOL_FAILURE,),
            ),
        }
    )

    failure = coordinator.get_context(
        idempotency_key="tool-failure-key",
        runner=lambda: failed_result,
    )

    assert failure.model_dump() == {
        "phase": "context",
        "error_code": "agent_orchestration_failed",
    }


def test_failed_runner_returns_replayable_safe_projection_without_details(
    tmp_path: Path,
) -> None:
    sentinel = "private-runner-sentinel"
    coordinator = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=tmp_path,
    )

    def fail() -> AgentResult:
        raise RuntimeError(sentinel)

    failure = coordinator.get_context(idempotency_key="failure-key", runner=fail)

    assert type(failure).__name__ == "HostDecisionFailure"
    assert failure.model_dump() == {
        "phase": "context",
        "error_code": "agent_session_failed",
    }
    assert sentinel not in failure.model_dump_json()
    assert sentinel not in repr(failure)

    restarted = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=tmp_path,
    )
    assert restarted.get_context(
        idempotency_key="failure-key",
        runner=lambda: _host_result("unused"),
    ) == failure
    with pytest.raises(HostDecisionError, match="^host decision is unavailable$"):
        restarted.submit_proposal(
            idempotency_key="failure-key",
            proposal=_proposal_for(_host_context("failed-run")),
        )


def test_safe_service_stage_failure_is_projected_as_its_allowlisted_code(
    tmp_path: Path,
) -> None:
    from riskprobe.service import HostSafeStageError

    coordinator = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=tmp_path,
    )
    failure = coordinator.get_context(
        idempotency_key="profile-failure-key",
        runner=lambda: (_ for _ in ()).throw(
            HostSafeStageError("profile_contract_failed")
        ),
    )

    assert failure.model_dump() == {
        "phase": "context",
        "error_code": "profile_contract_failed",
    }


@pytest.mark.parametrize(
    "error_code",
    (
        "agent_state_unavailable",
        "agent_state_incomplete",
        "agent_orchestration_failed",
    ),
)
def test_new_agent_failure_codes_are_allowlisted_and_replayable(
    tmp_path: Path,
    error_code: str,
) -> None:
    from riskprobe.service import HostSafeStageError

    coordinator = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=tmp_path,
    )
    failure = coordinator.get_context(
        idempotency_key=f"{error_code}-key",
        runner=lambda: (_ for _ in ()).throw(HostSafeStageError(error_code)),
    )

    assert failure.model_dump() == {
        "phase": "context",
        "error_code": error_code,
    }
    restarted = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=tmp_path,
    )
    assert restarted.get_context(
        idempotency_key=f"{error_code}-key",
        runner=lambda: _host_result("unused"),
    ) == failure


def _current_outcome_payload(context: DecisionContext) -> dict[str, object]:
    result = _host_result(context.session_id).model_copy(
        update={"analysis_summary": _agent_analysis_summary()}
    )
    proposal = _proposal_for(context)
    return host_decision.HostDecisionOutcome(
        context_id=context.context_id,
        agent_result=result,
        decision_status=host_decision.DecisionStatus.ACCEPTED,
        action_codes=proposal.action_codes,
        context_evidence_id="a" * 64,
        proposal_evidence_id="b" * 64,
        result_evidence_id="c" * 64,
        expires_at=context.expires_at,
        analysis_summary=host_decision._terminal_analysis_summary(
            result.analysis_summary,
            findings=context.findings,
            result=result,
            decision_context_succeeded=True,
        ),
        decision_summary=host_decision._decision_summary(
            proposal=proposal,
            result=result,
            recommendations=(),
            decision_status=host_decision.DecisionStatus.ACCEPTED,
            selected_action_codes=proposal.action_codes,
        ),
    ).model_dump(mode="json")


def _tampered_diagnosis_summary(summary: AnalysisSummary) -> AnalysisSummary:
    payload = summary.model_dump(mode="python")
    payload["stages"] = tuple(
        stage.model_copy(update={"limitations": ("tampered_summary",)})
        if stage.name is StageName.DIAGNOSE_QUALITY
        else stage
        for stage in summary.stages
    )
    return AnalysisSummary.model_validate(payload)


def test_report_subject_rejects_coordinated_normal_summary_tampering(
    tmp_path: Path,
) -> None:
    context = _host_context("tampered-normal-summary-run")
    proposal = _proposal_for(context)
    authoritative = _host_result(context.session_id).model_copy(
        update={
            "tool_sequence": (
                "inspect",
                "diagnose",
                "discover",
                "recommend",
                "review",
            ),
            "diagnosis_evidence_ids": context.diagnosis_evidence_ids,
            "evidence_ids": context.diagnosis_evidence_ids,
            "analysis_summary": _agent_analysis_summary(),
        }
    )
    outcome = host_decision.HostDecisionOutcome(
        context_id=context.context_id,
        agent_result=authoritative,
        decision_status=host_decision.DecisionStatus.ACCEPTED,
        action_codes=proposal.action_codes,
        context_evidence_id="a" * 64,
        proposal_evidence_id="b" * 64,
        result_evidence_id="c" * 64,
        expires_at=context.expires_at,
        analysis_summary=host_decision._terminal_analysis_summary(
            authoritative.analysis_summary,
            findings=context.findings,
            result=authoritative,
            decision_context_succeeded=True,
        ),
        decision_summary=host_decision._decision_summary(
            proposal=proposal,
            result=authoritative,
            recommendations=(),
            decision_status=host_decision.DecisionStatus.ACCEPTED,
            selected_action_codes=proposal.action_codes,
        ),
    )
    AgentResultStore(
        tmp_path / f".{context.session_id}.agent-result.json"
    ).publish(authoritative)
    tampered_result = authoritative.model_copy(
        update={
            "analysis_summary": _tampered_diagnosis_summary(
                authoritative.analysis_summary
            )
        }
    )
    tampered_analysis = host_decision._terminal_analysis_summary(
        tampered_result.analysis_summary,
        findings=context.findings,
        result=tampered_result,
        decision_context_succeeded=True,
    )
    payload = outcome.model_dump(mode="json")
    payload["agent_result"] = tampered_result.model_dump(mode="json")
    payload["analysis_summary"] = tampered_analysis.model_dump(mode="json")
    host_decision._HostSessionStore(tmp_path).save(
        host_decision._StoredHostSession(
            key="tampered-normal-summary-key",
            provider_id="kiro",
            provider_version="gpt5.6sol",
            lifecycle="terminal",
            report_run_id=context.session_id,
            context=host_decision.HostDecisionContext(
                provider_id="kiro",
                provider_version="gpt5.6sol",
                context=context,
            ),
            proposal=_proposal_for(context),
            outcome=payload,
        )
    )

    coordinator = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=tmp_path,
    )
    with pytest.raises(HostDecisionError, match="^host decision is unavailable$"):
        coordinator.report_subject(
            idempotency_key="tampered-normal-summary-key"
        )


def test_report_subject_rejects_coordinated_no_action_summary_tampering(
    tmp_path: Path,
) -> None:
    result = _clean_no_action_result("tampered-no-action-summary-run")
    AgentResultStore(tmp_path / f".{result.session_id}.agent-result.json").publish(
        result
    )
    tampered_result = result.model_copy(
        update={
            "analysis_summary": _tampered_diagnosis_summary(
                result.analysis_summary
            )
        }
    )
    tampered_analysis = host_decision._terminal_analysis_summary(
        tampered_result.analysis_summary,
        findings=(),
        result=tampered_result,
        decision_context_succeeded=False,
    )
    outcome = host_decision.HostDecisionNoActionOutcome(
        agent_result=tampered_result,
        analysis_summary=tampered_analysis,
        decision_summary=host_decision._no_action_decision_summary(
            tampered_result
        ),
    )
    host_decision._HostSessionStore(tmp_path).save(
        host_decision._StoredHostSession(
            key="tampered-no-action-summary-key",
            provider_id="kiro",
            provider_version="gpt5.6sol",
            lifecycle="terminal",
            report_run_id=result.session_id,
            outcome=outcome.model_dump(mode="json"),
        )
    )

    coordinator = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=tmp_path,
    )
    with pytest.raises(HostDecisionError, match="^host decision is unavailable$"):
        coordinator.report_subject(
            idempotency_key="tampered-no-action-summary-key"
        )


def test_fresh_summary_less_normal_outcome_fails_closed_before_legacy_replay() -> None:
    from pydantic import ValidationError

    context = _host_context("fresh-summary-less-run")
    proposal = _proposal_for(context)
    current = _current_outcome_payload(context)

    with pytest.raises(
        ValidationError,
        match="current outcome requires agent analysis summary",
    ):
        host_decision.HostDecisionOutcome(
            context_id=context.context_id,
            agent_result=_host_result(context.session_id),
            decision_status=host_decision.DecisionStatus.ACCEPTED,
            action_codes=proposal.action_codes,
            expires_at=context.expires_at,
            analysis_summary=AnalysisSummary.model_validate_json(
                json.dumps(current["analysis_summary"], sort_keys=True)
            ),
            decision_summary=DecisionSummary.model_validate_json(
                json.dumps(current["decision_summary"], sort_keys=True)
            ),
        )


def test_current_outcome_serializer_rejects_excluded_generation_markers() -> None:
    from pydantic_core import PydanticSerializationError

    context = _host_context("excluded-current-markers-run")
    outcome = HostDecisionCoordinator._outcome_from_payload(
        _current_outcome_payload(context)
    )

    with pytest.raises(
        PydanticSerializationError,
        match="current outcome requires canonical payload",
    ):
        outcome.model_dump(
            mode="json",
            exclude={
                "analysis_summary": True,
                "decision_summary": True,
                "agent_result": {"analysis_summary": True},
            },
        )


def test_unvalidated_current_outcome_cannot_forge_legacy_generation() -> None:
    from pydantic_core import PydanticSerializationError

    context = _host_context("forged-legacy-generation-run")
    proposal = _proposal_for(context)
    outcome = host_decision.HostDecisionOutcome.model_construct(
        context_id=context.context_id,
        agent_result=_host_result(context.session_id),
        decision_status=host_decision.DecisionStatus.ACCEPTED,
        action_codes=proposal.action_codes,
        expires_at=context.expires_at,
    ).model_copy(update={"_legacy_generation": True})

    with pytest.raises(
        PydanticSerializationError,
        match="current outcome requires canonical payload",
    ):
        outcome.model_dump(mode="json")


def test_current_outcome_decoder_rejects_null_and_mixed_generation_summaries() -> None:
    current = _current_outcome_payload(_host_context("current-summary-run"))
    legacy = {
        key: value
        for key, value in current.items()
        if key not in {"analysis_summary", "decision_summary"}
    }
    legacy_result = dict(legacy["agent_result"])
    legacy_result.pop("analysis_summary")
    legacy_review = dict(legacy_result["review"])
    legacy_review.pop("no_action_required")
    legacy_result["review"] = legacy_review
    legacy["agent_result"] = legacy_result
    current_result_with_legacy_review = dict(current["agent_result"])
    current_result_with_legacy_review["review"] = legacy_review
    missing_decision_default = dict(current["decision_summary"])
    missing_decision_default.pop("review_reason_codes")

    malformed = (
        {**current, "analysis_summary": None},
        {**current, "decision_summary": None},
        {**current, "decision_summary": missing_decision_default},
        {**current, "agent_result": legacy_result},
        {**legacy, "agent_result": current["agent_result"]},
        {**current, "agent_result": current_result_with_legacy_review},
    )
    for payload in malformed:
        with pytest.raises(HostDecisionError, match="^host decision is unavailable$"):
            HostDecisionCoordinator._outcome_from_payload(payload)


def test_legacy_normal_decoder_rejects_noncanonical_terminal_state() -> None:
    current = _current_outcome_payload(_host_context("legacy-noncanonical-terminal"))
    legacy = {
        key: value
        for key, value in current.items()
        if key not in {"analysis_summary", "decision_summary"}
    }
    result = dict(legacy["agent_result"])
    result.pop("analysis_summary")
    legacy["agent_result"] = result

    with pytest.raises(HostDecisionError, match="^host decision is unavailable$"):
        HostDecisionCoordinator._outcome_from_payload(legacy)


def test_legacy_normal_decoder_rejects_no_action_result() -> None:
    current = _current_outcome_payload(_host_context("legacy-normal-no-action"))
    legacy = {
        key: value
        for key, value in current.items()
        if key not in {"analysis_summary", "decision_summary"}
    }
    result = dict(legacy["agent_result"])
    result.pop("analysis_summary")
    review = dict(result["review"])
    review["no_action_required"] = True
    result["review"] = review
    legacy["agent_result"] = result

    with pytest.raises(HostDecisionError, match="^host decision is unavailable$"):
        HostDecisionCoordinator._outcome_from_payload(legacy)


@pytest.mark.parametrize("legacy_review", [False, True])
def test_legacy_normal_without_evidence_fails_closed_on_terminal_replay(
    tmp_path: Path,
    legacy_review: bool,
) -> None:
    context = _host_context("legacy-summary-run")
    outcome = _current_outcome_payload(context)
    legacy_fields = {
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
    legacy_outcome = {
        key: value for key, value in outcome.items() if key in legacy_fields
    }
    legacy_result = dict(legacy_outcome["agent_result"])
    legacy_result.pop("analysis_summary")
    legacy_result["tool_sequence"] = [
        "inspect",
        "diagnose",
        "discover",
        "recommend",
        "review",
    ]
    if legacy_review:
        review = dict(legacy_result["review"])
        review.pop("no_action_required")
        legacy_result["review"] = review
    legacy_outcome["agent_result"] = legacy_result

    restored = HostDecisionCoordinator._outcome_from_payload(legacy_outcome)

    assert restored.model_dump(mode="json") == legacy_outcome
    assert not hasattr(restored, "analysis_summary")
    assert not hasattr(restored, "decision_summary")

    store = host_decision._HostSessionStore(tmp_path)
    store.save(
        host_decision._StoredHostSession(
            key="legacy-summary-key",
            provider_id="kiro",
            provider_version="gpt5.6sol",
            lifecycle="terminal",
            report_run_id=context.session_id,
            context=host_decision.HostDecisionContext(
                provider_id="kiro",
                provider_version="gpt5.6sol",
                context=context,
            ),
            proposal=_proposal_for(context),
            outcome=legacy_outcome,
        )
    )
    before = store.path.read_bytes()
    coordinator = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=tmp_path,
    )
    assert coordinator.report_subject(idempotency_key="legacy-summary-key") is None
    replayed_context = coordinator.get_context(
        idempotency_key="legacy-summary-key",
        runner=lambda: (_ for _ in ()).throw(
            AssertionError("legacy replay must not rerun")
        ),
    )
    assert replayed_context.context == context
    with pytest.raises(HostDecisionError, match="^host decision is unavailable$"):
        coordinator.submit_proposal(
            idempotency_key="legacy-summary-key",
            proposal=_proposal_for(context),
        )
    assert coordinator.report_subject(idempotency_key="legacy-summary-key") is None
    assert store.path.read_bytes() == before


@pytest.mark.parametrize(
    "error_code",
    (
        "analysis_summary_failed",
        "artifact_finalize_failed",
        "scorecard_terms_failed",
        "scorecard_features_failed",
        "scorecard_metrics_failed",
        "scorecard_refs_failed",
        "scorecard_metadata_failed",
        "scorecard_refs_contract_failed",
        "scorecard_content_failed",
        "scorecard_splits_contract_failed",
        "scorecard_contract_failed",
    ),
)
def test_analysis_artifact_failure_codes_are_safe_and_replayable(
    tmp_path: Path,
    error_code: str,
) -> None:
    from riskprobe.service import HostSafeStageError

    coordinator = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=tmp_path,
    )
    failure = coordinator.get_context(
        idempotency_key=f"{error_code}-key",
        runner=lambda: (_ for _ in ()).throw(HostSafeStageError(error_code)),
    )

    assert failure.model_dump() == {"phase": "context", "error_code": error_code}
