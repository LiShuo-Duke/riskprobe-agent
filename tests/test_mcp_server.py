import asyncio
import json
from pathlib import Path

from mcp import Client

from riskprobe import host_decision
from riskprobe.agents.contracts import (
    AgentResult,
    AgentState,
    AgentStatus,
    ExecutionPlan,
    PlanStep,
    ReviewDecision,
)
from riskprobe.agents.decision_contracts import DecisionContext
from riskprobe.analysis_contracts import (
    AnalysisSummary,
    StageName,
    StageStatus,
    StageSummary,
)
from riskprobe.agents.decision_providers import (
    DecisionProviderConfig,
    DecisionProviderMode,
    DeterministicDecisionProvider,
)
from riskprobe.agents.results import AgentResultStore
from riskprobe.host_decision import HostDecisionCoordinator
from riskprobe.mcp_server import create_mcp_server
from riskprobe.policy import Principal, Role
from riskprobe.service import RiskProbeService
from riskprobe.terminal_reports import TerminalReportError
from riskprobe.tools import (
    DiagnoseRequest,
    DiscoverRequest,
    InspectRequest,
    RecommendRequest,
)


def _agent_analysis_summary() -> AnalysisSummary:
    succeeded = {
        *tuple(StageName)[:11],
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


def _clean_no_action_result() -> AgentResult:
    return AgentResult(
        session_id="mcp-no-action",
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
                    request=RecommendRequest(
                        dataset_id="synthetic_demo", evidence_ids=()
                    ),
                    requires_evidence=True,
                ),
                PlanStep(step_id="review", tool_name="review"),
            ),
            component_versions={"planner": "planner-v1"},
        ),
        review=ReviewDecision(approved=True, no_action_required=True),
        tool_sequence=("inspect", "diagnose", "discover", "recommend", "review"),
        retry_count=0,
        state_history=(AgentState.PLANNING, AgentState.REVIEWING, AgentState.COMPLETED),
        leaf_node_id="b" * 64,
        redacted_summary="no actionable diagnosis",
        analysis_summary=_agent_analysis_summary(),
    )


def _server(
    *,
    tmp_path: Path,
    synthetic_config: object,
) -> tuple[object, Path]:
    state_dir = tmp_path / "state"
    coordinator = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
        state_dir=state_dir,
    )
    runs_dir = tmp_path / "runs"
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
    return (
        create_mcp_server(
            service=service,
            coordinator=coordinator,
            dataset_id=synthetic_config.dataset.id,
            principal=Principal(principal_id="kiro-host", role=Role.ANALYST),
            max_queries=16,
        ),
        runs_dir,
    )


def test_mcp_server_exposes_only_two_high_level_tools(
    tmp_path: Path,
    synthetic_config: object,
) -> None:
    server, _ = _server(tmp_path=tmp_path, synthetic_config=synthetic_config)

    async def inspect_server() -> None:
        async with Client(server, raise_exceptions=True) as client:
            tools = await client.list_tools()
            resources = await client.list_resources()
            prompts = await client.list_prompts()

        assert [tool.name for tool in tools.tools] == [
            "riskprobe_get_decision_context",
            "riskprobe_submit_decision_proposal",
        ]
        schemas = {tool.name: tool.input_schema for tool in tools.tools}
        assert set(schemas["riskprobe_get_decision_context"]["properties"]) == {
            "idempotency_key"
        }
        assert set(
            schemas["riskprobe_submit_decision_proposal"]["properties"]
        ) == {
            "idempotency_key",
            "context_id",
            "diagnosis_evidence_ids",
            "action_codes",
        }
        assert resources.resources == []
        assert prompts.prompts == []

    asyncio.run(inspect_server())


def test_mcp_tools_complete_the_fixed_pipeline(
    tmp_path: Path,
    synthetic_config: object,
) -> None:
    server, runs_dir = _server(tmp_path=tmp_path, synthetic_config=synthetic_config)

    async def run_tools() -> dict[str, object]:
        async with Client(server, raise_exceptions=True) as client:
            pending = await client.call_tool(
                "riskprobe_get_decision_context",
                {"idempotency_key": "mcp-decision-1"},
            )
            assert pending.is_error is False
            assert isinstance(pending.structured_content, dict)
            context_payload = pending.structured_content["result"]["context"]
            assert "analysis_summary" in context_payload
            assert len(context_payload["analysis_summary"]["stages"]) == 24
            context = DecisionContext.model_validate_json(
                json.dumps(context_payload, sort_keys=True)
            )
            deterministic = DeterministicDecisionProvider().resolve(
                context=context
            ).proposal
            assert deterministic is not None

            terminal = await client.call_tool(
                "riskprobe_submit_decision_proposal",
                {
                    "idempotency_key": "mcp-decision-1",
                    "context_id": context.context_id,
                    "diagnosis_evidence_ids": list(context.diagnosis_evidence_ids),
                    "action_codes": [code.value for code in deterministic.action_codes],
                },
            )
            assert terminal.is_error is False
            assert isinstance(terminal.structured_content, dict)
            return terminal.structured_content

    outcome = asyncio.run(run_tools())
    result = outcome["agent_result"]
    assert isinstance(result, dict)
    assert outcome["phase"] == "terminal"
    assert "analysis_summary" in outcome
    assert "decision_summary" in outcome
    assert len(outcome["analysis_summary"]["stages"]) == 24
    assert result["review"]["approved"] is True
    assert result["tool_sequence"] == [
        "inspect",
        "diagnose",
        "discover",
        "recommend",
        "review",
    ]
    assert {path.name for path in (runs_dir / result["session_id"]).iterdir()} == {
        "manifest.json",
        "metadata_report.json",
        "data_profile.json",
        "candidate_rules.parquet",
        "evidence_cards.json",
        "risk_report.md",
        "analysis_summary.json",
    }



def test_mcp_no_action_terminal_replays_without_decision_artifacts(
    tmp_path: Path,
    synthetic_config: object,
    monkeypatch: object,
) -> None:
    clean_result = _clean_no_action_result()

    def return_no_action(self: object, **kwargs: object) -> AgentResult:
        del self, kwargs
        return clean_result

    monkeypatch.setattr(RiskProbeService, "orchestrate", return_no_action)
    monkeypatch.setattr(
        RiskProbeService,
        "ensure_terminal_report",
        lambda *_args, **_kwargs: None,
    )
    server, runs_dir = _server(tmp_path=tmp_path, synthetic_config=synthetic_config)
    AgentResultStore(
        tmp_path / "state" / f".{clean_result.session_id}.agent-result.json"
    ).publish(clean_result)

    async def run_tools() -> tuple[dict[str, object], object]:
        async with Client(server, raise_exceptions=True) as client:
            context_response = await client.call_tool(
                "riskprobe_get_decision_context",
                {"idempotency_key": "mcp-no-action-key"},
            )
            submit_response = await client.call_tool(
                "riskprobe_submit_decision_proposal",
                {
                    "idempotency_key": "mcp-no-action-key",
                    "context_id": "f" * 64,
                    "diagnosis_evidence_ids": ["a" * 64],
                    "action_codes": ["investigate_feature_drift"],
                },
            )
        assert context_response.is_error is False
        assert isinstance(context_response.structured_content, dict)
        return context_response.structured_content, submit_response

    context_payload, submit_response = asyncio.run(run_tools())
    terminal = context_payload["result"]
    assert isinstance(terminal, dict)
    assert set(terminal) == {
        "protocol_version",
        "phase",
        "terminal_reason",
        "action_codes",
        "agent_result",
        "analysis_summary",
        "decision_summary",
    }
    assert terminal["phase"] == "terminal"
    assert terminal["terminal_reason"] == "no_actionable_diagnosis"
    assert terminal["action_codes"] == []
    assert len(terminal["analysis_summary"]["stages"]) == 24
    assert terminal["decision_summary"] == {
        "selected_action_codes": [],
        "recommendations": [],
        "recommendation_status": "skipped",
        "review_approved": True,
        "review_reason_codes": [],
        "no_action_required": True,
        "retry_count": 0,
        "tool_sequence": [
            "inspect",
            "diagnose",
            "discover",
            "recommend",
            "review",
        ],
        "evidence_complete": True,
        "decision_status": "no_action",
        "final_status": "succeeded",
    }
    result = terminal["agent_result"]
    assert isinstance(result, dict)
    assert result["status"] == "succeeded"
    assert result["review"] == {
        "approved": True,
        "reason_codes": [],
        "evidence_ids": [],
        "retry_allowed": False,
        "no_action_required": True,
    }
    assert result["evidence_ids"] == []
    assert result["diagnosis_evidence_ids"] == []
    assert result["tool_sequence"] == [
        "inspect",
        "diagnose",
        "discover",
        "recommend",
        "review",
    ]
    assert submit_response.is_error is False
    assert submit_response.structured_content == terminal
    assert not (runs_dir / clean_result.session_id).exists()


def test_mcp_replays_report_bound_legacy_no_action_exactly(
    tmp_path: Path,
    synthetic_config: object,
    monkeypatch: object,
) -> None:
    result = _clean_no_action_result()
    result_payload = result.model_dump(mode="json")
    result_payload.pop("analysis_summary")
    legacy_outcome = {
        "protocol_version": "riskprobe.host-decision.v1",
        "phase": "terminal",
        "terminal_reason": "no_actionable_diagnosis",
        "action_codes": [],
        "agent_result": result_payload,
    }
    store = host_decision._HostSessionStore(tmp_path / "state")
    store.save(
        host_decision._StoredHostSession(
            key="mcp-legacy-no-action-key",
            provider_id="kiro",
            provider_version="gpt5.6sol",
            lifecycle="terminal",
            report_run_id=result.session_id,
            outcome=legacy_outcome,
        )
    )
    before = store.path.read_bytes()
    orchestration_calls = 0

    def reject_rerun(self: object, **kwargs: object) -> AgentResult:
        nonlocal orchestration_calls
        del self, kwargs
        orchestration_calls += 1
        raise AssertionError("legacy no-action replay must not rerun")

    monkeypatch.setattr(RiskProbeService, "orchestrate", reject_rerun)
    server, _ = _server(tmp_path=tmp_path, synthetic_config=synthetic_config)

    async def replay() -> dict[str, object]:
        async with Client(server, raise_exceptions=True) as client:
            response = await client.call_tool(
                "riskprobe_get_decision_context",
                {"idempotency_key": "mcp-legacy-no-action-key"},
            )
        assert response.is_error is False
        assert isinstance(response.structured_content, dict)
        return response.structured_content

    assert asyncio.run(replay()) == {"result": legacy_outcome}
    assert orchestration_calls == 0
    assert store.path.read_bytes() == before


def test_mcp_no_action_unknown_report_gate_failure_is_safe_and_retryable(
    tmp_path: Path,
    synthetic_config: object,
    monkeypatch: object,
) -> None:
    clean_result = _clean_no_action_result()
    orchestration_calls = 0
    report_calls = 0
    sentinel = "private-terminal-report-sentinel"

    def return_no_action(self: object, **kwargs: object) -> AgentResult:
        nonlocal orchestration_calls
        del self, kwargs
        orchestration_calls += 1
        return clean_result

    def fail_once(self: object, subject: object) -> None:
        nonlocal report_calls
        del self, subject
        report_calls += 1
        if report_calls == 1:
            raise RuntimeError(sentinel)

    monkeypatch.setattr(RiskProbeService, "orchestrate", return_no_action)
    monkeypatch.setattr(RiskProbeService, "ensure_terminal_report", fail_once)
    server, _ = _server(tmp_path=tmp_path, synthetic_config=synthetic_config)
    AgentResultStore(
        tmp_path / "state" / f".{clean_result.session_id}.agent-result.json"
    ).publish(clean_result)

    async def retry_get() -> tuple[dict[str, object], dict[str, object]]:
        async with Client(server, raise_exceptions=True) as client:
            first = await client.call_tool(
                "riskprobe_get_decision_context",
                {"idempotency_key": "mcp-no-action-report-retry"},
            )
            second = await client.call_tool(
                "riskprobe_get_decision_context",
                {"idempotency_key": "mcp-no-action-report-retry"},
            )
        assert first.is_error is False
        assert second.is_error is False
        assert isinstance(first.structured_content, dict)
        assert isinstance(second.structured_content, dict)
        return first.structured_content, second.structured_content

    first, second = asyncio.run(retry_get())

    assert first == {
        "result": {
            "phase": "context",
            "error_code": "agent_orchestration_failed",
        }
    }
    assert sentinel not in json.dumps(first, sort_keys=True)
    assert second["result"]["terminal_reason"] == "no_actionable_diagnosis"
    assert orchestration_calls == 1
    assert report_calls == 2


def test_mcp_actionable_report_gate_failure_is_safe_and_retryable(
    tmp_path: Path,
    synthetic_config: object,
    monkeypatch: object,
) -> None:
    original_orchestrate = RiskProbeService.orchestrate
    orchestration_calls = 0
    report_calls = 0
    sentinel = "private-actionable-report-sentinel"

    def count_orchestration(
        self: RiskProbeService,
        **kwargs: object,
    ) -> AgentResult:
        nonlocal orchestration_calls
        orchestration_calls += 1
        return original_orchestrate(self, **kwargs)

    def fail_once(self: object, subject: object) -> None:
        nonlocal report_calls
        del self, subject
        report_calls += 1
        if report_calls == 1:
            raise TerminalReportError(sentinel)

    monkeypatch.setattr(RiskProbeService, "orchestrate", count_orchestration)
    monkeypatch.setattr(RiskProbeService, "ensure_terminal_report", fail_once)
    server, _ = _server(tmp_path=tmp_path, synthetic_config=synthetic_config)

    async def retry_submit() -> tuple[dict[str, object], dict[str, object]]:
        async with Client(server, raise_exceptions=True) as client:
            pending = await client.call_tool(
                "riskprobe_get_decision_context",
                {"idempotency_key": "mcp-actionable-report-retry"},
            )
            assert isinstance(pending.structured_content, dict)
            context = DecisionContext.model_validate_json(
                json.dumps(
                    pending.structured_content["result"]["context"],
                    sort_keys=True,
                )
            )
            proposal = DeterministicDecisionProvider().resolve(context=context).proposal
            assert proposal is not None
            arguments = {
                "idempotency_key": "mcp-actionable-report-retry",
                "context_id": context.context_id,
                "diagnosis_evidence_ids": list(context.diagnosis_evidence_ids),
                "action_codes": [code.value for code in proposal.action_codes],
            }
            first = await client.call_tool(
                "riskprobe_submit_decision_proposal",
                arguments,
            )
            second = await client.call_tool(
                "riskprobe_submit_decision_proposal",
                arguments,
            )
        assert first.is_error is False
        assert second.is_error is False
        assert isinstance(first.structured_content, dict)
        assert isinstance(second.structured_content, dict)
        return first.structured_content, second.structured_content

    first, second = asyncio.run(retry_submit())

    assert first == {
        "phase": "context",
        "error_code": "agent_orchestration_failed",
    }
    assert sentinel not in json.dumps(first, sort_keys=True)
    assert second["phase"] == "terminal"
    assert second["decision_status"] == "accepted"
    assert orchestration_calls == 1
    assert report_calls == 2


def test_mcp_context_failure_is_a_safe_structured_payload(
    tmp_path: Path,
    synthetic_config: object,
    monkeypatch: object,
) -> None:
    sentinel = "private-mcp-runner-sentinel"

    def fail_orchestration(self: object, **kwargs: object) -> object:
        del self, kwargs
        raise RuntimeError(sentinel)

    monkeypatch.setattr(RiskProbeService, "orchestrate", fail_orchestration)
    server, _ = _server(tmp_path=tmp_path, synthetic_config=synthetic_config)

    async def get_failure() -> dict[str, object]:
        async with Client(server, raise_exceptions=True) as client:
            response = await client.call_tool(
                "riskprobe_get_decision_context",
                {"idempotency_key": "mcp-failure-key"},
            )
        assert response.is_error is False
        assert isinstance(response.structured_content, dict)
        return response.structured_content

    payload = asyncio.run(get_failure())
    assert payload == {
        "result": {"phase": "context", "error_code": "agent_session_failed"}
    }
    assert sentinel not in json.dumps(payload, sort_keys=True)


def test_mcp_submit_after_context_failure_returns_safe_structured_payload(
    tmp_path: Path,
    synthetic_config: object,
    monkeypatch: object,
) -> None:
    sentinel = "private-mcp-submit-sentinel"

    def fail_orchestration(self: object, **kwargs: object) -> object:
        del self, kwargs
        raise RuntimeError(sentinel)

    monkeypatch.setattr(RiskProbeService, "orchestrate", fail_orchestration)
    server, _ = _server(tmp_path=tmp_path, synthetic_config=synthetic_config)

    async def submit_after_failure() -> tuple[dict[str, object], object]:
        async with Client(server, raise_exceptions=True) as client:
            context_response = await client.call_tool(
                "riskprobe_get_decision_context",
                {"idempotency_key": "mcp-submit-failure-key"},
            )
            submit_response = await client.call_tool(
                "riskprobe_submit_decision_proposal",
                {
                    "idempotency_key": "mcp-submit-failure-key",
                    "context_id": "f" * 64,
                    "diagnosis_evidence_ids": ["a" * 64],
                    "action_codes": ["investigate_feature_drift"],
                },
            )
        assert context_response.is_error is False
        assert isinstance(context_response.structured_content, dict)
        return context_response.structured_content, submit_response

    context_payload, submit_response = asyncio.run(submit_after_failure())
    assert submit_response.is_error is False
    assert context_payload == {
        "result": {"phase": "context", "error_code": "agent_session_failed"}
    }
    assert submit_response.structured_content == {
        "phase": "context", "error_code": "agent_session_failed"
    }
    assert sentinel not in json.dumps(
        [context_payload, submit_response.structured_content], sort_keys=True
    )


def test_mcp_exactly_projects_agent_orchestration_failure(
    tmp_path: Path,
    synthetic_config: object,
    monkeypatch: object,
) -> None:
    from riskprobe.service import HostSafeStageError

    sentinel = "private-mcp-orchestration-sentinel"

    def fail_orchestration(self: object, **kwargs: object) -> object:
        del self, kwargs
        raise HostSafeStageError("agent_orchestration_failed") from RuntimeError(sentinel)

    monkeypatch.setattr(RiskProbeService, "orchestrate", fail_orchestration)
    server, _ = _server(tmp_path=tmp_path, synthetic_config=synthetic_config)

    async def get_failure() -> dict[str, object]:
        async with Client(server, raise_exceptions=True) as client:
            response = await client.call_tool(
                "riskprobe_get_decision_context",
                {"idempotency_key": "mcp-orchestration-failure-key"},
            )
        assert response.is_error is False
        assert isinstance(response.structured_content, dict)
        return response.structured_content

    payload = asyncio.run(get_failure())
    assert payload == {
        "result": {
            "phase": "context",
            "error_code": "agent_orchestration_failed",
        }
    }
    assert sentinel not in json.dumps(payload, sort_keys=True)
