import socket
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import riskprobe.agents.orchestrator as orchestrator_module
from riskprobe.analysis_contracts import (
    AnalysisSummary,
    PartitionSummary,
    ProfileSummary,
    StageName,
    StageStatus,
    StageSummary,
)
from riskprobe.agents.contracts import (
    AgentState,
    AgentStatus,
    ReviewDecision,
    ReviewReason,
)
from riskprobe.agents.decision_contracts import (
    DecisionContext,
    DecisionProposal,
    DecisionSource,
    DecisionStatus,
)
from riskprobe.agents.decision_controller import DecisionController
from riskprobe.agents.decision_providers import (
    DecisionDisposition,
    DecisionProviderError,
    DecisionProviderMode,
    DecisionProviderResolution,
    DeterministicDecisionProvider,
)
from riskprobe.agents.orchestrator import AgentOrchestrator
from riskprobe.agents.planner import Planner
from riskprobe.agents.reviewer import Reviewer
from riskprobe.agents.sessions import (
    SessionNodeKind,
    SessionStore,
    SessionToolCall,
)
from riskprobe.evidence import EvidenceRecord, EvidenceStore
from riskprobe.monitoring.models import FindingKind, FindingSeverity, RiskFinding
from riskprobe.policy import Budget, PolicyDeniedError, Principal, Role
from riskprobe.privacy import tokenize_segment
from riskprobe.recommendations.models import DecisionEligibility, Recommendation
from riskprobe.recommendations.policy import ActionCode
from riskprobe.tools import (
    DiagnoseRequest,
    DiagnoseResponse,
    DiscoverRequest,
    DiscoverResponse,
    InspectRequest,
    InspectResponse,
    RecommendRequest,
    RecommendResponse,
)

_FINDING_A = "a" * 64
_FINDING_B = "b" * 64
_RECOMMENDATION_A = "c" * 64
_TOOL_TYPES = {
    "inspect": InspectRequest,
    "diagnose": DiagnoseRequest,
    "discover": DiscoverRequest,
    "recommend": RecommendRequest,
}


def _record(
    *,
    run_id: str = "session-agent",
    kind: str = "diagnostic.finding",
    payload: dict[str, object] | None = None,
    parent_ids: tuple[str, ...] = (),
) -> EvidenceRecord:
    return EvidenceRecord(
        run_id=run_id,
        kind=kind,
        payload=payload or {"finding_id": _FINDING_A, "severity": "warning"},
        parent_ids=parent_ids,
        artifact_hashes={},
        producer_version="orchestrator-test/1",
    )


class RecordingGateway:
    def __init__(self, *, missing_diagnosis_attempts: int = 0) -> None:
        self.requests: list[object] = []
        self.missing_diagnosis_attempts = missing_diagnosis_attempts
        self.diagnosis_calls = 0
        self.latest_findings: tuple[str, ...] = ()
        self.latest_recommendations: tuple[str, ...] = ()
        self.evidence_store: EvidenceStore | None = None

    def bind(self, evidence_store: EvidenceStore) -> None:
        self.evidence_store = evidence_store

    def invoke(self, principal: Principal, request: object, budget: Budget) -> object:
        del principal
        assert budget.consume()
        self.requests.append(request)
        if isinstance(request, InspectRequest):
            return InspectResponse(
                dataset_id=request.dataset_id,
                row_count=100,
                feature_count=8,
                metadata_grade="A",
            )
        if isinstance(request, DiagnoseRequest):
            self.diagnosis_calls += 1
            if self.diagnosis_calls <= self.missing_diagnosis_attempts:
                findings: tuple[str, ...] = ()
            else:
                assert self.evidence_store is not None
                findings = (self.evidence_store.append(_record()),)
            self.latest_findings = findings
            return DiagnoseResponse(dataset_id=request.dataset_id, finding_ids=findings)
        if isinstance(request, DiscoverRequest):
            return DiscoverResponse(dataset_id=request.dataset_id, rule_ids=("rule-1",))
        if isinstance(request, RecommendRequest):
            assert request.evidence_ids == self.latest_findings
            if not self.latest_findings:
                recommendations: tuple[str, ...] = ()
            else:
                assert self.evidence_store is not None
                recommendation = _record(
                    kind="recommendation",
                    payload={
                        "action_code": "remediate_data_quality",
                        "decision_eligibility": "human_review_required",
                        "finding_ids": (_FINDING_A,),
                        "human_approval_required": True,
                        "recommendation_id": _RECOMMENDATION_A,
                    },
                    parent_ids=self.latest_findings,
                )
                recommendations = (self.evidence_store.append(recommendation),)
            self.latest_recommendations = recommendations
            return RecommendResponse(
                dataset_id=request.dataset_id,
                recommendation_ids=recommendations,
            )
        raise AssertionError("unexpected request")


class StaticGateway:
    def __init__(
        self,
        *,
        diagnosis_ids: tuple[str, ...],
        recommendation_ids: tuple[str, ...],
    ) -> None:
        self.diagnosis_ids = diagnosis_ids
        self.recommendation_ids = recommendation_ids

    def invoke(self, principal: Principal, request: object, budget: Budget) -> object:
        del principal
        assert budget.consume()
        if isinstance(request, InspectRequest):
            return InspectResponse(
                dataset_id=request.dataset_id,
                row_count=100,
                feature_count=8,
                metadata_grade="A",
            )
        if isinstance(request, DiagnoseRequest):
            return DiagnoseResponse(
                dataset_id=request.dataset_id,
                finding_ids=self.diagnosis_ids,
            )
        if isinstance(request, DiscoverRequest):
            return DiscoverResponse(dataset_id=request.dataset_id, rule_ids=("rule-1",))
        if isinstance(request, RecommendRequest):
            assert request.evidence_ids == self.diagnosis_ids
            return RecommendResponse(
                dataset_id=request.dataset_id,
                recommendation_ids=self.recommendation_ids,
            )
        raise AssertionError("unexpected request")


def _orchestrator(
    tmp_path: Path,
    gateway: object,
    *,
    evidence_resolver: object | None = None,
) -> tuple[AgentOrchestrator, SessionStore]:
    sessions = SessionStore(tmp_path / "sessions.sqlite3")
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    bind = getattr(gateway, "bind", None)
    if callable(bind):
        bind(store)
    return (
        AgentOrchestrator(
            planner=Planner(allowed_tools=_TOOL_TYPES),
            reviewer=Reviewer(),
            gateway=gateway,
            sessions=sessions,
            evidence_resolver=store if evidence_resolver is None else evidence_resolver,
        ),
        sessions,
    )


def _run(orchestrator: AgentOrchestrator):
    return orchestrator.run(
        objective="comprehensive",
        dataset_id="synthetic_demo",
        principal=Principal(principal_id="agent-analyst", role=Role.ANALYST),
        budget=Budget(max_queries=16),
        session_id="session-agent",
    )


def test_orchestrator_executes_typed_state_machine_without_network_or_raw_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def network_forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "getaddrinfo", network_forbidden)
    monkeypatch.setattr(socket.socket, "connect", network_forbidden)
    gateway = RecordingGateway()
    orchestrator, sessions = _orchestrator(tmp_path, gateway)

    result = _run(orchestrator)

    assert result.status is AgentStatus.SUCCEEDED
    assert result.tool_sequence == (
        "inspect",
        "diagnose",
        "discover",
        "recommend",
        "review",
    )
    assert result.evidence_ids == tuple(
        sorted((*gateway.latest_findings, *gateway.latest_recommendations))
    )
    assert result.review.approved is True
    assert result.retry_count == 0
    assert [type(request) for request in gateway.requests] == [
        InspectRequest,
        DiagnoseRequest,
        DiscoverRequest,
        RecommendRequest,
    ]
    replay = sessions.replay("session-agent")
    assert replay[0].kind is SessionNodeKind.ROOT
    assert replay[-1].tool_call is not None
    assert replay[-1].tool_call.tool_name == "review"
    assert "/private" not in result.model_dump_json()


def test_orchestrator_completes_explicit_empty_diagnosis_without_retry(
    tmp_path: Path,
) -> None:
    gateway = RecordingGateway(missing_diagnosis_attempts=1)
    orchestrator, sessions = _orchestrator(tmp_path, gateway)

    result = _run(orchestrator)

    assert result.status is AgentStatus.SUCCEEDED
    assert result.retry_count == 0
    assert gateway.diagnosis_calls == 1
    assert result.review.no_action_required is True
    assert not any(
        node.kind is SessionNodeKind.RETRY
        for node in sessions.replay("session-agent")
    )


@pytest.mark.parametrize(
    "tamper",
    ("arguments", "evidence", "component"),
)
def test_terminal_validation_rejects_tampered_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    gateway = RecordingGateway()
    orchestrator, sessions = _orchestrator(tmp_path, gateway)
    result = _run(orchestrator)
    assert orchestrator.validate_terminal_result(
        result,
        objective="comprehensive",
        dataset_id="synthetic_demo",
        session_id="session-agent",
        metadata_grade="A",
    ) == result
    persisted_replay = sessions.replay
    first_review = next(
        node
        for node in persisted_replay("session-agent")
        if node.tool_call is not None
        and node.tool_call.tool_name == "review"
    )
    if tamper == "arguments":
        replacement = first_review.model_copy(
            update={
                "tool_call": SessionToolCall(
                    tool_name="review",
                    arguments={
                        **dict(first_review.tool_call.arguments),
                        "retry_count": 1,
                    },
                )
            }
        )
    elif tamper == "evidence":
        replacement = first_review.model_copy(
            update={"evidence_ids": ("a" * 64,)}
        )
    else:
        replacement = first_review.model_copy(
            update={"component_versions": {"reviewer": "other-v1"}}
        )

    def tampered_replay(
        session_id: str,
        *,
        branch_id: str | None = None,
        leaf_node_id: str | None = None,
    ):
        nodes = persisted_replay(
            session_id,
            branch_id=branch_id,
            leaf_node_id=leaf_node_id,
        )
        return tuple(
            replacement if node.node_id == first_review.node_id else node
            for node in nodes
        )

    monkeypatch.setattr(sessions, "replay", tampered_replay)

    with pytest.raises(RuntimeError, match="^agent result is unavailable$"):
        orchestrator.validate_terminal_result(
            result,
            objective="comprehensive",
            dataset_id="synthetic_demo",
            session_id="session-agent",
            metadata_grade="A",
        )


def test_orchestrator_routes_permission_denial_through_reviewer(tmp_path: Path) -> None:
    class DeniedGateway(RecordingGateway):
        def invoke(self, principal: Principal, request: object, budget: Budget) -> object:
            if isinstance(request, DiagnoseRequest):
                raise PolicyDeniedError("private details must not escape")
            return super().invoke(principal, request, budget)

    orchestrator, _ = _orchestrator(tmp_path, DeniedGateway())

    result = _run(orchestrator)

    assert result.status is AgentStatus.REJECTED
    assert ReviewReason.PERMISSION_DENIED in result.review.reason_codes
    assert "private details" not in result.model_dump_json()


def test_orchestrator_independently_rejects_unsafe_gateway_payload(tmp_path: Path) -> None:
    class UnsafeGateway(RecordingGateway):
        def invoke(self, principal: Principal, request: object, budget: Budget) -> object:
            if isinstance(request, InspectRequest):
                assert budget.consume()
                return InspectResponse(
                    dataset_id=request.dataset_id,
                    row_count=100,
                    feature_count=8,
                    metadata_grade="A",
                    issue_codes=("customer_123456",),
                )
            return super().invoke(principal, request, budget)

    orchestrator, _ = _orchestrator(tmp_path, UnsafeGateway())

    result = _run(orchestrator)

    assert result.status is AgentStatus.REJECTED
    assert ReviewReason.UNSAFE_PAYLOAD in result.review.reason_codes
    assert "customer_123456" not in result.model_dump_json()


def test_orchestrator_requires_a_callable_evidence_resolver(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="evidence_resolver"):
        AgentOrchestrator(
            planner=Planner(allowed_tools=_TOOL_TYPES),
            reviewer=Reviewer(),
            gateway=RecordingGateway(),
            sessions=SessionStore(tmp_path / "sessions.sqlite3"),
            evidence_resolver=None,
        )


def test_orchestrator_fails_closed_when_resolver_cannot_return_evidence(tmp_path: Path) -> None:
    class MissingResolver:
        def get(self, evidence_id: str) -> None:
            del evidence_id
            return None

    gateway = RecordingGateway()
    orchestrator, _ = _orchestrator(
        tmp_path,
        gateway,
        evidence_resolver=MissingResolver(),
    )

    result = _run(orchestrator)

    assert result.status is AgentStatus.REJECTED
    assert result.retry_count == 0
    assert result.evidence_ids == ()
    assert ReviewReason.MISSING_EVIDENCE in result.review.reason_codes
    assert ReviewReason.TOOL_FAILURE in result.review.reason_codes


def test_orchestrator_rejects_unsafe_resolved_evidence_without_echo(tmp_path: Path) -> None:
    private_path = "/private/customer/data.parquet"

    class UnsafeResolver:
        def get(self, evidence_id: str) -> EvidenceRecord:
            del evidence_id
            return _record(payload={"path": private_path})

    orchestrator, _ = _orchestrator(
        tmp_path,
        RecordingGateway(),
        evidence_resolver=UnsafeResolver(),
    )

    result = _run(orchestrator)

    assert result.status is AgentStatus.REJECTED
    assert result.retry_count == 0
    assert ReviewReason.UNSAFE_PAYLOAD in result.review.reason_codes
    assert private_path not in result.model_dump_json()


def test_orchestrator_rejects_unvalidated_evidence_ids_from_gateway(tmp_path: Path) -> None:
    class InvalidIdGateway(RecordingGateway):
        def invoke(self, principal: Principal, request: object, budget: Budget) -> object:
            if isinstance(request, DiagnoseRequest):
                del principal
                assert budget.consume()
                return DiagnoseResponse.model_construct(
                    dataset_id=request.dataset_id,
                    finding_ids=("A" * 64,),
                )
            return super().invoke(principal, request, budget)

    orchestrator, _ = _orchestrator(tmp_path, InvalidIdGateway())

    result = _run(orchestrator)

    assert result.status is AgentStatus.REJECTED
    assert result.retry_count == 0
    assert result.evidence_ids == ()
    assert ReviewReason.TOOL_FAILURE in result.review.reason_codes


@pytest.mark.parametrize("mismatch", ["content", "run", "type"])
def test_orchestrator_rejects_record_identity_run_and_type_mismatch(
    tmp_path: Path,
    mismatch: str,
) -> None:
    store = EvidenceStore(tmp_path / "graph.sqlite3")
    run_id = "other-run" if mismatch == "run" else "session-agent"
    diagnosis = _record(
        run_id=run_id,
        kind="recommendation" if mismatch == "type" else "diagnostic.finding",
    )
    diagnosis_id = store.append(diagnosis)
    recommendation = _record(
        run_id=run_id,
        kind="recommendation",
        payload={
            "finding_ids": (_FINDING_A,),
            "human_approval_required": True,
            "recommendation_id": _RECOMMENDATION_A,
        },
        parent_ids=(diagnosis_id,),
    )
    recommendation_id = store.append(recommendation)

    resolver: object = store
    if mismatch == "content":
        original_get = store.get

        class ContentMismatchResolver:
            def get(self, evidence_id: str) -> EvidenceRecord | None:
                record = original_get(evidence_id)
                if evidence_id == diagnosis_id and record is not None:
                    return record.model_copy(
                        update={"payload": {"finding_id": _FINDING_A, "severity": "critical"}}
                    )
                return record

        resolver = ContentMismatchResolver()

    gateway = StaticGateway(
        diagnosis_ids=(diagnosis_id,),
        recommendation_ids=(recommendation_id,),
    )
    orchestrator, _ = _orchestrator(tmp_path, gateway, evidence_resolver=resolver)

    result = _run(orchestrator)

    assert result.status is AgentStatus.REJECTED
    assert result.retry_count == 0
    assert result.evidence_ids == ()
    assert ReviewReason.TOOL_FAILURE in result.review.reason_codes


def test_orchestrator_requires_exact_recommendation_parent_set(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "graph.sqlite3")
    first_id = store.append(_record(payload={"finding_id": _FINDING_A}))
    second_id = store.append(_record(payload={"finding_id": _FINDING_B}))
    recommendation_id = store.append(
        _record(
            kind="recommendation",
            payload={
                "finding_ids": (_FINDING_A,),
                "human_approval_required": True,
                "recommendation_id": _RECOMMENDATION_A,
            },
            parent_ids=tuple(sorted((first_id, second_id))),
        )
    )
    gateway = StaticGateway(
        diagnosis_ids=tuple(sorted((first_id, second_id))),
        recommendation_ids=(recommendation_id,),
    )
    orchestrator, _ = _orchestrator(tmp_path, gateway, evidence_resolver=store)

    result = _run(orchestrator)

    assert result.status is AgentStatus.REJECTED
    assert result.retry_count == 0
    assert result.evidence_ids == ()
    assert ReviewReason.TOOL_FAILURE in result.review.reason_codes


def test_orchestrator_rejects_non_analysis_only_grade_b_recommendation(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path / "grade-b.sqlite3")
    diagnosis_id = store.append(_record(payload={"finding_id": _FINDING_A}))
    recommendation_id = store.append(
        _record(
            kind="recommendation",
            payload={
                "decision_eligibility": "human_review_required",
                "finding_ids": (_FINDING_A,),
                "human_approval_required": True,
                "recommendation_id": _RECOMMENDATION_A,
            },
            parent_ids=(diagnosis_id,),
        )
    )

    class GradeBGateway(StaticGateway):
        def invoke(self, principal: Principal, request: object, budget: Budget) -> object:
            if isinstance(request, InspectRequest):
                del principal
                assert budget.consume()
                return InspectResponse(
                    dataset_id=request.dataset_id,
                    row_count=100,
                    feature_count=8,
                    metadata_grade="B",
                )
            return super().invoke(principal, request, budget)

    gateway = GradeBGateway(
        diagnosis_ids=(diagnosis_id,),
        recommendation_ids=(recommendation_id,),
    )
    orchestrator, _ = _orchestrator(tmp_path, gateway, evidence_resolver=store)

    result = _run(orchestrator)

    assert result.status is AgentStatus.REJECTED
    assert result.retry_count == 0
    assert result.evidence_ids == ()
    assert ReviewReason.TOOL_FAILURE in result.review.reason_codes


class _DecisionGateway:
    def __init__(
        self,
        findings: Mapping[ActionCode, tuple[RiskFinding, ...]] | None = None,
    ) -> None:
        self.requests: list[object] = []
        self.evidence_store: EvidenceStore | None = None
        self.findings = dict(findings) if findings is not None else {
            ActionCode.INVESTIGATE_FEATURE_DRIFT: (
                RiskFinding(
                    kind=FindingKind.FEATURE_DRIFT,
                    severity=FindingSeverity.WARNING,
                    code="feature_psi",
                    feature="order_count",
                    metrics={"psi": 0.25},
                ),
            ),
            ActionCode.REMEDIATE_DATA_QUALITY: (
                RiskFinding(
                    kind=FindingKind.DATA_QUALITY,
                    severity=FindingSeverity.WARNING,
                    code="missing_values",
                    metrics={"affected_rate": 0.1},
                ),
            ),
        }
        self.finding_evidence: dict[
            ActionCode,
            tuple[tuple[str, RiskFinding], ...],
        ] = {}
        self.latest_recommendations: tuple[str, ...] = ()

    def bind(self, evidence_store: EvidenceStore) -> None:
        self.evidence_store = evidence_store

    def invoke(self, principal: Principal, request: object, budget: Budget) -> object:
        del principal
        assert budget.consume()
        self.requests.append(request)
        if isinstance(request, InspectRequest):
            return InspectResponse(
                dataset_id=request.dataset_id,
                row_count=100,
                feature_count=8,
                metadata_grade="A",
                issue_codes=("LABEL_PERFORMANCE_WINDOW_UNKNOWN",),
            )
        if isinstance(request, DiagnoseRequest):
            assert self.evidence_store is not None
            self.finding_evidence = {}
            for action, findings in self.findings.items():
                action_evidence: list[tuple[str, RiskFinding]] = []
                for finding in findings:
                    evidence_id = self.evidence_store.append(
                        EvidenceRecord(
                            run_id="session-agent",
                            kind="diagnostic.finding",
                            payload={
                                **finding.model_dump(mode="json"),
                                "dataset_id": request.dataset_id,
                            },
                            producer_version="diagnostics-v1",
                        )
                    )
                    action_evidence.append((evidence_id, finding))
                self.finding_evidence[action] = tuple(action_evidence)
            return DiagnoseResponse(
                dataset_id=request.dataset_id,
                finding_ids=tuple(
                    sorted(
                        evidence_id
                        for action_evidence in self.finding_evidence.values()
                        for evidence_id, _ in action_evidence
                    )
                ),
            )
        if isinstance(request, DiscoverRequest):
            return DiscoverResponse(
                dataset_id=request.dataset_id,
                rule_ids=("rule-1",),
            )
        if isinstance(request, RecommendRequest):
            assert self.evidence_store is not None
            result_evidence_id = getattr(
                request,
                "decision_result_evidence_id",
                None,
            )
            assert isinstance(result_evidence_id, str)
            submission = DecisionController(self.evidence_store).replay(
                result_evidence_id=result_evidence_id,
                expected_run_id="session-agent",
            )
            recommendation_ids: list[str] = []
            for action in submission.result.action_codes:
                action_evidence = self.finding_evidence[action]
                recommendation = Recommendation(
                    action_code=action.value,
                    priority="high",
                    finding_ids=tuple(
                        finding.finding_id for _, finding in action_evidence
                    ),
                    rationale_code=(
                        f"{action_evidence[0][1].kind.value}_finding_present"
                    ),
                    human_approval_required=True,
                    decision_eligibility=DecisionEligibility.HUMAN_REVIEW_REQUIRED,
                )
                recommendation_ids.append(
                    self.evidence_store.append(
                        EvidenceRecord(
                            run_id="session-agent",
                            kind="recommendation",
                            payload=recommendation.model_dump(mode="json"),
                            parent_ids=tuple(
                                sorted(
                                    evidence_id
                                    for evidence_id, _ in action_evidence
                                )
                            ),
                            producer_version="recommendations-v1",
                        )
                    )
                )
            self.latest_recommendations = tuple(recommendation_ids)
            return RecommendResponse(
                dataset_id=request.dataset_id,
                recommendation_ids=self.latest_recommendations,
            )
        raise AssertionError("unexpected request")


class _ExternalProposalProvider:
    mode = DecisionProviderMode.EXTERNAL_HOST
    provider_id = "test-external-host"
    version = "test-external-host-v1"

    def __init__(
        self,
        action_code: ActionCode,
        *additional_action_codes: ActionCode,
    ) -> None:
        self.action_code = action_code
        self.action_codes = (action_code, *additional_action_codes)
        self.calls = 0

    def resolve(self, *, context: DecisionContext) -> DecisionProviderResolution:
        self.calls += 1
        return DecisionProviderResolution(
            disposition=DecisionDisposition.PROPOSAL,
            proposal=DecisionProposal(
                context_id=context.context_id,
                diagnosis_evidence_ids=context.diagnosis_evidence_ids,
                action_codes=self.action_codes,
                source=DecisionSource.EXTERNAL_HOST,
                source_version=self.version,
            ),
        )


class _PendingDecisionProvider:
    mode = DecisionProviderMode.EXTERNAL_HOST
    provider_id = "test-pending-host"
    version = "test-pending-host-v1"

    def __init__(self) -> None:
        self.calls = 0

    def resolve(self, *, context: DecisionContext) -> DecisionProviderResolution:
        del context
        self.calls += 1
        return DecisionProviderResolution(disposition=DecisionDisposition.PENDING)


class _ErrorDecisionProvider:
    mode = DecisionProviderMode.EXTERNAL_HOST
    provider_id = "test-error-host"
    version = "test-error-host-v1"

    def __init__(self) -> None:
        self.calls = 0

    def resolve(self, *, context: DecisionContext) -> DecisionProviderResolution:
        del context
        self.calls += 1
        raise DecisionProviderError("private provider failure detail")


class _ErrorFallbackProvider:
    mode = DecisionProviderMode.DETERMINISTIC
    provider_id = "test-error-fallback"
    version = "test-error-fallback-v1"

    def __init__(self) -> None:
        self.calls = 0

    def resolve(self, *, context: DecisionContext) -> DecisionProviderResolution:
        del context
        self.calls += 1
        raise DecisionProviderError("private fallback failure detail")


class _MutableDecisionClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value


def _controlled_orchestrator(
    tmp_path: Path,
    gateway: _DecisionGateway,
    *,
    provider: object | None = None,
    fallback: object | None = None,
) -> tuple[AgentOrchestrator, SessionStore, EvidenceStore, _MutableDecisionClock]:
    sessions = SessionStore(tmp_path / "controlled-sessions.sqlite3")
    store = EvidenceStore(tmp_path / "controlled-evidence.sqlite3")
    gateway.bind(store)
    clock = _MutableDecisionClock()
    kwargs: dict[str, object] = {}
    if provider is not None:
        kwargs["decision_provider"] = provider
    if fallback is not None:
        kwargs["decision_fallback"] = fallback
    orchestrator = AgentOrchestrator(
        planner=Planner(allowed_tools=_TOOL_TYPES),
        reviewer=Reviewer(),
        gateway=gateway,
        sessions=sessions,
        evidence_resolver=store,
        decision_controller=DecisionController(store, clock=clock),
        **kwargs,
    )
    return orchestrator, sessions, store, clock


def test_orchestrator_default_fallback_preserves_v1_surface_and_audits_decision(
    tmp_path: Path,
) -> None:
    gateway = _DecisionGateway()
    orchestrator, sessions, _, _ = _controlled_orchestrator(tmp_path, gateway)

    result = _run(orchestrator)

    assert result.status is AgentStatus.SUCCEEDED
    assert result.tool_sequence == (
        "inspect",
        "diagnose",
        "discover",
        "recommend",
        "review",
    )
    assert tuple(step.tool_name for step in result.plan.steps) == result.tool_sequence
    assert isinstance(gateway.requests[3], RecommendRequest)
    assert type(gateway.requests[3]) is not RecommendRequest
    assert [
        type(request)
        for request in gateway.requests[:3]
    ] == [InspectRequest, DiagnoseRequest, DiscoverRequest]
    replay = sessions.replay("session-agent")
    tool_names = [
        node.tool_call.tool_name
        for node in replay
        if node.tool_call is not None
    ]
    assert tool_names == [
        "inspect",
        "diagnose",
        "discover",
        "decision",
        "recommend",
        "review",
    ]
    audit = next(
        node
        for node in replay
        if node.tool_call is not None and node.tool_call.tool_name == "decision"
    )
    assert set(audit.tool_call.arguments) == {
        "context_evidence_id",
        "proposal_evidence_id",
        "result_evidence_id",
    }
    assert all(
        isinstance(value, str) and len(value) == 64
        for value in audit.tool_call.arguments.values()
    )
    assert audit.evidence_ids == tuple(sorted(audit.tool_call.arguments.values()))
    assert set(audit.evidence_ids).isdisjoint(result.evidence_ids)
    assert set(result.model_dump()) == {
        "session_id",
        "status",
        "plan",
        "review",
        "tool_sequence",
        "evidence_ids",
        "diagnosis_evidence_ids",
        "retry_count",
        "state_history",
        "leaf_node_id",
        "redacted_summary",
        "analysis_summary",
    }


def test_orchestrator_filters_to_one_server_verified_action(tmp_path: Path) -> None:
    gateway = _DecisionGateway()
    provider = _ExternalProposalProvider(ActionCode.INVESTIGATE_FEATURE_DRIFT)
    orchestrator, _, store, _ = _controlled_orchestrator(
        tmp_path,
        gateway,
        provider=provider,
    )

    result = _run(orchestrator)

    assert result.status is AgentStatus.SUCCEEDED
    assert provider.calls == 1
    assert len(gateway.latest_recommendations) == 1
    recommendation = store.get(gateway.latest_recommendations[0])
    assert recommendation is not None
    assert recommendation.payload["action_code"] == "investigate_feature_drift"


def test_orchestrator_aggregates_segment_findings_for_two_action_terminal(
    tmp_path: Path,
) -> None:
    segment_findings = (
        RiskFinding(
            kind=FindingKind.SEGMENT_RISK,
            severity=FindingSeverity.INFO,
            code="segment_target_rate",
            segment_token=tokenize_segment("private-segment-a", namespace="test"),
            metrics={"sample_count": 100, "target_rate": 0.2},
        ),
        RiskFinding(
            kind=FindingKind.SEGMENT_RISK,
            severity=FindingSeverity.INFO,
            code="segment_target_rate",
            segment_token=tokenize_segment("private-segment-b", namespace="test"),
            metrics={"sample_count": 120, "target_rate": 0.3},
        ),
    )
    time_finding = RiskFinding(
        kind=FindingKind.TIME_INSTABILITY,
        severity=FindingSeverity.CRITICAL,
        code="monthly_sample_drop",
        period="2026-08",
        metrics={
            "current_count": 154,
            "previous_count": 4468,
            "drop_rate": 0.96,
        },
    )
    gateway = _DecisionGateway(
        findings={
            ActionCode.REVIEW_SEGMENT_RISK: segment_findings,
            ActionCode.MONITOR_TIME_STABILITY: (time_finding,),
        }
    )
    provider = _ExternalProposalProvider(
        ActionCode.REVIEW_SEGMENT_RISK,
        ActionCode.MONITOR_TIME_STABILITY,
    )
    orchestrator, sessions, store, _ = _controlled_orchestrator(
        tmp_path,
        gateway,
        provider=provider,
    )

    result = _run(orchestrator)

    assert result.status is AgentStatus.SUCCEEDED
    assert result.review.approved is True
    assert result.tool_sequence == (
        "inspect",
        "diagnose",
        "discover",
        "recommend",
        "review",
    )
    assert len(result.diagnosis_evidence_ids) == 3
    recommendations = tuple(
        record
        for record in store.list_run("session-agent")
        if record.kind == "recommendation"
    )
    assert len(recommendations) == len(gateway.latest_recommendations) == 2
    assert {record.payload["action_code"] for record in recommendations} == {
        ActionCode.REVIEW_SEGMENT_RISK.value,
        ActionCode.MONITOR_TIME_STABILITY.value,
    }
    segment_recommendation = next(
        record
        for record in recommendations
        if record.payload["action_code"] == ActionCode.REVIEW_SEGMENT_RISK.value
    )
    time_recommendation = next(
        record
        for record in recommendations
        if record.payload["action_code"] == ActionCode.MONITOR_TIME_STABILITY.value
    )
    assert segment_recommendation.payload["finding_ids"] == sorted(
        finding.finding_id for finding in segment_findings
    )
    assert time_recommendation.payload["finding_ids"] == [time_finding.finding_id]
    assert len(segment_recommendation.parent_ids) == 2
    assert len(time_recommendation.parent_ids) == 1
    assert set(
        (*segment_recommendation.parent_ids, *time_recommendation.parent_ids)
    ) == set(result.diagnosis_evidence_ids)
    audit = next(
        node
        for node in sessions.replay("session-agent")
        if node.tool_call is not None and node.tool_call.tool_name == "decision"
    )
    assert audit.tool_call is not None
    submission = DecisionController(store).replay(
        result_evidence_id=str(audit.tool_call.arguments["result_evidence_id"]),
        expected_run_id="session-agent",
    )
    assert submission.result.status is DecisionStatus.ACCEPTED
    assert submission.result.action_codes == (
        ActionCode.MONITOR_TIME_STABILITY,
        ActionCode.REVIEW_SEGMENT_RISK,
    )
    assert orchestrator.validate_terminal_result(
        result,
        objective="comprehensive",
        dataset_id="synthetic_demo",
        session_id="session-agent",
        metadata_grade="A",
    ) == result
    assert orchestrator.recover_terminal_result(
        objective="comprehensive",
        dataset_id="synthetic_demo",
        session_id="session-agent",
        metadata_grade="A",
    ) == result
    assert provider.calls == 1


@pytest.mark.parametrize(
    "provider",
    (
        _PendingDecisionProvider(),
        _ErrorDecisionProvider(),
        _ExternalProposalProvider(ActionCode.REVIEW_SEGMENT_RISK),
    ),
    ids=("pending", "error", "rejected"),
)
def test_orchestrator_fails_closed_without_recommend_for_nonaccepted_decision(
    tmp_path: Path,
    provider: object,
) -> None:
    gateway = _DecisionGateway()
    orchestrator, _, _, _ = _controlled_orchestrator(
        tmp_path,
        gateway,
        provider=provider,
    )

    result = _run(orchestrator)

    assert result.status is AgentStatus.REJECTED
    assert ReviewReason.TOOL_FAILURE in result.review.reason_codes
    assert not any(isinstance(request, RecommendRequest) for request in gateway.requests)
    assert "private provider failure detail" not in result.model_dump_json()


@pytest.mark.parametrize(
    ("provider", "reason"),
    (
        (_PendingDecisionProvider(), "provider_pending"),
        (_ErrorDecisionProvider(), "provider_error"),
    ),
    ids=("pending", "error"),
)
def test_terminal_validation_replays_provider_unavailable_without_provider_call(
    tmp_path: Path,
    provider: object,
    reason: str,
) -> None:
    gateway = _DecisionGateway()
    orchestrator, sessions, store, _ = _controlled_orchestrator(
        tmp_path,
        gateway,
        provider=provider,
    )

    result = _run(orchestrator)

    assert result.status is AgentStatus.REJECTED
    assert ReviewReason.TOOL_FAILURE in result.review.reason_codes
    assert not any(
        isinstance(request, RecommendRequest) for request in gateway.requests
    )
    records = store.list_run("session-agent")
    contexts = [record for record in records if record.kind == "decision.context"]
    unavailable = [
        record for record in records if record.kind == "decision.unavailable"
    ]
    assert len(contexts) == len(unavailable) == 1
    assert not any(
        record.kind in {"decision.proposal", "decision.result"}
        for record in records
    )
    context_id = EvidenceStore.content_id(contexts[0])
    outcome_id = EvidenceStore.content_id(unavailable[0])
    assert unavailable[0].parent_ids == (context_id,)
    assert unavailable[0].payload["reason"] == reason
    assert "private" not in unavailable[0].model_dump_json()
    audit = next(
        node
        for node in sessions.replay("session-agent")
        if node.tool_call is not None and node.tool_call.tool_name == "decision"
    )
    assert audit.tool_call is not None
    assert dict(audit.tool_call.arguments) == {
        "context_evidence_id": context_id,
        "outcome_evidence_id": outcome_id,
    }
    assert audit.evidence_ids == tuple(sorted((context_id, outcome_id)))
    assert audit.redacted_summary == "controlled decision unavailable audited"
    calls = getattr(provider, "calls")

    validated = orchestrator.validate_terminal_result(
        result,
        objective="comprehensive",
        dataset_id="synthetic_demo",
        session_id="session-agent",
        metadata_grade="A",
    )

    assert validated == result
    assert getattr(provider, "calls") == calls == 1


def test_submit_failure_records_replayable_unavailable_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _DecisionGateway()
    orchestrator, _, store, _ = _controlled_orchestrator(tmp_path, gateway)

    def fail_submit(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("private submit failure detail")

    monkeypatch.setattr(DecisionController, "submit", fail_submit)

    result = _run(orchestrator)

    assert result.status is AgentStatus.REJECTED
    assert result.review.reason_codes == (ReviewReason.TOOL_FAILURE,)
    outcome_records = [
        record
        for record in store.list_run("session-agent")
        if record.kind == "decision.unavailable"
    ]
    assert len(outcome_records) == 1
    outcome_record = outcome_records[0]
    assert outcome_record.payload["reason"] == "submission_error"
    assert "private submit failure detail" not in outcome_record.model_dump_json()
    assert "private submit failure detail" not in result.model_dump_json()
    assert orchestrator.validate_terminal_result(
        result,
        objective="comprehensive",
        dataset_id="synthetic_demo",
        session_id="session-agent",
        metadata_grade="A",
    ) == result
    assert orchestrator.recover_terminal_result(
        objective="comprehensive",
        dataset_id="synthetic_demo",
        session_id="session-agent",
        metadata_grade="A",
    ) == result


@pytest.mark.parametrize(
    ("provider", "reason"),
    (
        (_PendingDecisionProvider(), "provider_pending"),
        (_ErrorDecisionProvider(), "provider_error"),
    ),
    ids=("pending", "error"),
)
def test_orchestrator_audits_disabled_primary_unavailable(
    tmp_path: Path,
    provider: object,
    reason: str,
) -> None:
    provider.mode = DecisionProviderMode.DISABLED
    gateway = _DecisionGateway()
    orchestrator, _, store, _ = _controlled_orchestrator(
        tmp_path,
        gateway,
        provider=provider,
    )

    result = _run(orchestrator)

    assert result.status is AgentStatus.REJECTED
    records = store.list_run("session-agent")
    unavailable = [
        record for record in records if record.kind == "decision.unavailable"
    ]
    assert len(unavailable) == 1
    assert unavailable[0].payload["reason"] == reason
    binding = unavailable[0].payload["provider_binding"]
    assert binding["selected_role"] == "primary"
    assert binding["selected"]["provider_id"] == provider.provider_id
    assert binding["selected"]["mode"] == "disabled"
    assert not any(
        record.kind in {"decision.proposal", "decision.result"}
        for record in records
    )
    assert orchestrator.validate_terminal_result(
        result,
        objective="comprehensive",
        dataset_id="synthetic_demo",
        session_id="session-agent",
        metadata_grade="A",
    ) == result


def test_terminal_validation_requires_unavailable_review_to_be_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _DecisionGateway()
    provider = _PendingDecisionProvider()
    orchestrator, sessions, _, _ = _controlled_orchestrator(
        tmp_path,
        gateway,
        provider=provider,
    )
    result = _run(orchestrator)
    persisted_replay = sessions.replay
    review = persisted_replay("session-agent")[-1]
    approved_review = review.model_copy(
        update={
            "tool_call": SessionToolCall(
                tool_name="review",
                arguments={
                    "approved": True,
                    "reason_codes": (),
                    "retry_count": 0,
                },
            ),
            "redacted_summary": "deterministic review approved",
        }
    )
    forged = result.model_copy(
        update={
            "status": AgentStatus.SUCCEEDED,
            "review": ReviewDecision(
                approved=True,
                evidence_ids=result.evidence_ids,
            ),
            "state_history": (*result.state_history[:-1], AgentState.COMPLETED),
            "redacted_summary": (
                "comprehensive objective approved from aggregate evidence"
            ),
        }
    )

    def tampered_replay(
        session_id: str,
        *,
        branch_id: str | None = None,
        leaf_node_id: str | None = None,
    ):
        nodes = persisted_replay(
            session_id,
            branch_id=branch_id,
            leaf_node_id=leaf_node_id,
        )
        return tuple(
            approved_review if node.node_id == review.node_id else node
            for node in nodes
        )

    monkeypatch.setattr(sessions, "replay", tampered_replay)

    with pytest.raises(RuntimeError, match="^agent result is unavailable$"):
        orchestrator.validate_terminal_result(
            forged,
            objective="comprehensive",
            dataset_id="synthetic_demo",
            session_id="session-agent",
            metadata_grade="A",
        )


def test_terminal_validation_replays_fallback_provider_error(
    tmp_path: Path,
) -> None:
    gateway = _DecisionGateway()
    fallback = _ErrorFallbackProvider()
    orchestrator, _, store, _ = _controlled_orchestrator(
        tmp_path,
        gateway,
        fallback=fallback,
    )

    result = _run(orchestrator)

    unavailable = next(
        record
        for record in store.list_run("session-agent")
        if record.kind == "decision.unavailable"
    )
    binding = unavailable.payload["provider_binding"]
    assert isinstance(binding, dict)
    assert binding["selected_role"] == "fallback"
    assert binding["selected"]["provider_id"] == fallback.provider_id
    assert unavailable.payload["reason"] == "provider_error"
    assert fallback.calls == 1
    assert orchestrator.validate_terminal_result(
        result,
        objective="comprehensive",
        dataset_id="synthetic_demo",
        session_id="session-agent",
        metadata_grade="A",
    ) == result
    assert fallback.calls == 1


def test_terminal_validation_replays_persisted_submission_time_and_anchor(
    tmp_path: Path,
) -> None:
    gateway = _DecisionGateway()
    orchestrator, sessions, store, clock = _controlled_orchestrator(tmp_path, gateway)
    result = _run(orchestrator)
    replay = sessions.replay("session-agent")
    discover_node = next(
        node
        for node in replay
        if node.tool_call is not None and node.tool_call.tool_name == "discover"
    )
    audit = next(
        node
        for node in replay
        if node.tool_call is not None and node.tool_call.tool_name == "decision"
    )
    assert audit.tool_call is not None
    submission = DecisionController(store).replay(
        result_evidence_id=str(audit.tool_call.arguments["result_evidence_id"]),
        expected_run_id="session-agent",
    )
    assert submission.context.attempt == 0
    assert submission.context.anchor_node_id == discover_node.node_id
    clock.value = submission.context.expires_at + timedelta(days=30)

    validated = orchestrator.validate_terminal_result(
        result,
        objective="comprehensive",
        dataset_id="synthetic_demo",
        session_id="session-agent",
        metadata_grade="A",
    )

    assert validated == result


def test_terminal_validation_rejects_duplicate_decision_audit(tmp_path: Path) -> None:
    gateway = _DecisionGateway()
    orchestrator, sessions, _, _ = _controlled_orchestrator(tmp_path, gateway)
    result = _run(orchestrator)
    audit = next(
        node
        for node in sessions.replay("session-agent")
        if node.tool_call is not None and node.tool_call.tool_name == "decision"
    )
    assert audit.tool_call is not None
    sessions.append_child(
        result.leaf_node_id,
        tool_call=audit.tool_call,
        evidence_ids=audit.evidence_ids,
        redacted_summary="controlled decision audited",
        component_versions=audit.component_versions,
    )

    with pytest.raises(RuntimeError, match="^agent result is unavailable$"):
        orchestrator.validate_terminal_result(
            result,
            objective="comprehensive",
            dataset_id="synthetic_demo",
            session_id="session-agent",
            metadata_grade="A",
        )


class _BypassDecisionGateway(_DecisionGateway):
    def invoke(self, principal: Principal, request: object, budget: Budget) -> object:
        response = super().invoke(principal, request, budget)
        if not isinstance(request, RecommendRequest):
            return response
        assert self.evidence_store is not None
        evidence_id, finding = self.finding_evidence[
            ActionCode.REMEDIATE_DATA_QUALITY
        ][0]
        extra = Recommendation(
            action_code=ActionCode.REMEDIATE_DATA_QUALITY.value,
            priority="high",
            finding_ids=(finding.finding_id,),
            rationale_code="data_quality_finding_present",
            human_approval_required=True,
            decision_eligibility=DecisionEligibility.HUMAN_REVIEW_REQUIRED,
        )
        extra_id = self.evidence_store.append(
            EvidenceRecord(
                run_id="session-agent",
                kind="recommendation",
                payload=extra.model_dump(mode="json"),
                parent_ids=(evidence_id,),
                producer_version="recommendations-v1",
            )
        )
        assert isinstance(response, RecommendResponse)
        self.latest_recommendations = (*response.recommendation_ids, extra_id)
        return RecommendResponse(
            dataset_id=response.dataset_id,
            recommendation_ids=self.latest_recommendations,
        )


def test_orchestrator_rejects_gateway_action_filter_bypass(tmp_path: Path) -> None:
    gateway = _BypassDecisionGateway()
    provider = _ExternalProposalProvider(ActionCode.INVESTIGATE_FEATURE_DRIFT)
    orchestrator, _, _, _ = _controlled_orchestrator(
        tmp_path,
        gateway,
        provider=provider,
    )

    result = _run(orchestrator)

    assert result.status is AgentStatus.REJECTED
    assert ReviewReason.TOOL_FAILURE in result.review.reason_codes


def test_orchestrator_completes_clean_explicit_empty_diagnosis_without_decision(
    tmp_path: Path,
) -> None:
    gateway = RecordingGateway(missing_diagnosis_attempts=10)
    sessions = SessionStore(tmp_path / "missing-controlled-sessions.sqlite3")
    store = EvidenceStore(tmp_path / "missing-controlled-evidence.sqlite3")
    gateway.bind(store)
    provider = _ExternalProposalProvider(ActionCode.REMEDIATE_DATA_QUALITY)
    orchestrator = AgentOrchestrator(
        planner=Planner(allowed_tools=_TOOL_TYPES),
        reviewer=Reviewer(),
        gateway=gateway,
        sessions=sessions,
        evidence_resolver=store,
        decision_controller=DecisionController(
            store,
            clock=lambda: datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        ),
        decision_provider=provider,
    )

    result = _run(orchestrator)

    assert result.status is AgentStatus.SUCCEEDED
    assert result.tool_sequence == (
        "inspect",
        "diagnose",
        "discover",
        "recommend",
        "review",
    )
    assert result.retry_count == 0
    assert gateway.diagnosis_calls == 1
    assert result.evidence_ids == ()
    assert result.diagnosis_evidence_ids == ()
    assert result.review.no_action_required is True
    assert provider.calls == 0
    assert not any(
        record.kind.startswith("decision.")
        for record in store.list_run("session-agent")
    )
    assert not any(
        node.tool_call is not None and node.tool_call.tool_name == "decision"
        for node in sessions.replay("session-agent")
    )
    assert orchestrator.validate_terminal_result(
        result,
        objective="comprehensive",
        dataset_id="synthetic_demo",
        session_id="session-agent",
        metadata_grade="A",
    ) == result


def test_orchestrator_rejects_custom_gateway_response_type_mismatch_before_next_tool(
    tmp_path: Path,
) -> None:
    class WrongTypeGateway(_DecisionGateway):
        def invoke(
            self,
            principal: Principal,
            request: object,
            budget: Budget,
        ) -> object:
            if isinstance(request, DiscoverRequest):
                del principal
                assert budget.consume()
                self.requests.append(request)
                return InspectResponse(
                    dataset_id=request.dataset_id,
                    row_count=100,
                    feature_count=8,
                    metadata_grade="A",
                )
            return super().invoke(principal, request, budget)

    gateway = WrongTypeGateway()
    orchestrator, _, _, _ = _controlled_orchestrator(tmp_path, gateway)

    result = _run(orchestrator)

    assert result.status is AgentStatus.REJECTED
    assert ReviewReason.TOOL_FAILURE in result.review.reason_codes
    assert [type(request) for request in gateway.requests] == [
        InspectRequest,
        DiagnoseRequest,
        DiscoverRequest,
    ]


def test_terminal_validation_rejects_empty_diagnose_with_hidden_decision_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _DecisionGateway()
    orchestrator, sessions, _, _ = _controlled_orchestrator(tmp_path, gateway)
    result = _run(orchestrator)
    persisted_replay = sessions.replay
    diagnose = next(
        node
        for node in persisted_replay("session-agent")
        if node.tool_call is not None
        and node.tool_call.tool_name == "diagnose"
    )

    def tampered_replay(
        session_id: str,
        *,
        branch_id: str | None = None,
        leaf_node_id: str | None = None,
    ):
        nodes = persisted_replay(
            session_id,
            branch_id=branch_id,
            leaf_node_id=leaf_node_id,
        )
        return tuple(
            node.model_copy(update={"evidence_ids": ()})
            if node.node_id == diagnose.node_id
            else node
            for node in nodes
            if node.tool_call is None
            or node.tool_call.tool_name != "decision"
        )

    monkeypatch.setattr(sessions, "replay", tampered_replay)

    with pytest.raises(RuntimeError, match="^agent result is unavailable$"):
        orchestrator.validate_terminal_result(
            result,
            objective="comprehensive",
            dataset_id="synthetic_demo",
            session_id="session-agent",
            metadata_grade="A",
        )


def test_terminal_validation_allows_failed_discover_without_decision_audit(
    tmp_path: Path,
) -> None:
    class WrongDiscoverResponseGateway(_DecisionGateway):
        def invoke(
            self,
            principal: Principal,
            request: object,
            budget: Budget,
        ) -> object:
            if isinstance(request, DiscoverRequest):
                del principal
                assert budget.consume()
                self.requests.append(request)
                return InspectResponse(
                    dataset_id=request.dataset_id,
                    row_count=100,
                    feature_count=8,
                    metadata_grade="A",
                )
            return super().invoke(principal, request, budget)

    gateway = WrongDiscoverResponseGateway()
    orchestrator, sessions, store, _ = _controlled_orchestrator(tmp_path, gateway)
    result = _run(orchestrator)
    assert result.status is AgentStatus.REJECTED
    assert not any(
        node.tool_call is not None and node.tool_call.tool_name == "decision"
        for node in sessions.replay("session-agent")
    )
    assert not any(
        record.kind.startswith("decision.")
        for record in store.list_run("session-agent")
    )

    assert orchestrator.validate_terminal_result(
        result,
        objective="comprehensive",
        dataset_id="synthetic_demo",
        session_id="session-agent",
        metadata_grade="A",
    ) == result


def test_terminal_validation_requires_failed_discover_review_to_be_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WrongDiscoverResponseGateway(_DecisionGateway):
        def invoke(
            self,
            principal: Principal,
            request: object,
            budget: Budget,
        ) -> object:
            if isinstance(request, DiscoverRequest):
                del principal
                assert budget.consume()
                self.requests.append(request)
                return InspectResponse(
                    dataset_id=request.dataset_id,
                    row_count=100,
                    feature_count=8,
                    metadata_grade="A",
                )
            return super().invoke(principal, request, budget)

    gateway = WrongDiscoverResponseGateway()
    orchestrator, sessions, _, _ = _controlled_orchestrator(tmp_path, gateway)
    result = _run(orchestrator)
    persisted_replay = sessions.replay
    review = persisted_replay("session-agent")[-1]
    approved_review = review.model_copy(
        update={
            "tool_call": SessionToolCall(
                tool_name="review",
                arguments={
                    "approved": True,
                    "reason_codes": (),
                    "retry_count": 0,
                },
            ),
            "redacted_summary": "deterministic review approved",
        }
    )
    forged = result.model_copy(
        update={
            "status": AgentStatus.SUCCEEDED,
            "review": ReviewDecision(
                approved=True,
                evidence_ids=result.evidence_ids,
            ),
            "state_history": (*result.state_history[:-1], AgentState.COMPLETED),
            "redacted_summary": (
                "comprehensive objective approved from aggregate evidence"
            ),
        }
    )

    def tampered_replay(
        session_id: str,
        *,
        branch_id: str | None = None,
        leaf_node_id: str | None = None,
    ):
        nodes = persisted_replay(
            session_id,
            branch_id=branch_id,
            leaf_node_id=leaf_node_id,
        )
        return tuple(
            approved_review if node.node_id == review.node_id else node
            for node in nodes
        )

    monkeypatch.setattr(sessions, "replay", tampered_replay)

    with pytest.raises(RuntimeError, match="^agent result is unavailable$"):
        orchestrator.validate_terminal_result(
            forged,
            objective="comprehensive",
            dataset_id="synthetic_demo",
            session_id="session-agent",
            metadata_grade="A",
        )


def test_terminal_validation_rejects_succeeded_plan_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _DecisionGateway()
    orchestrator, sessions, _, _ = _controlled_orchestrator(tmp_path, gateway)
    result = _run(orchestrator)
    persisted = sessions.replay("session-agent")
    root = persisted[0]
    inspect = next(
        node
        for node in persisted
        if node.tool_call is not None
        and node.tool_call.tool_name == "inspect"
    )
    diagnose = next(
        node
        for node in persisted
        if node.tool_call is not None
        and node.tool_call.tool_name == "diagnose"
    )
    review = persisted[-1].model_copy(
        update={
            "parent_node_id": diagnose.node_id,
            "sequence": diagnose.sequence + 1,
            "evidence_ids": diagnose.evidence_ids,
        }
    )
    forged = result.model_copy(
        update={
            "review": result.review.model_copy(
                update={"evidence_ids": diagnose.evidence_ids}
            ),
            "evidence_ids": diagnose.evidence_ids,
            "leaf_node_id": review.node_id,
        }
    )
    shortened_replay = (root, inspect, diagnose, review)

    def tampered_replay(
        session_id: str,
        *,
        branch_id: str | None = None,
        leaf_node_id: str | None = None,
    ):
        del branch_id, leaf_node_id
        assert session_id == "session-agent"
        return shortened_replay

    monkeypatch.setattr(sessions, "replay", tampered_replay)

    with pytest.raises(RuntimeError, match="^agent result is unavailable$"):
        orchestrator.validate_terminal_result(
            forged,
            objective="comprehensive",
            dataset_id="synthetic_demo",
            session_id="session-agent",
            metadata_grade="A",
        )


def test_terminal_validation_rejects_same_mode_version_different_provider_id(
    tmp_path: Path,
) -> None:
    gateway = _DecisionGateway()
    original = _ExternalProposalProvider(ActionCode.INVESTIGATE_FEATURE_DRIFT)
    orchestrator, sessions, store, _ = _controlled_orchestrator(
        tmp_path,
        gateway,
        provider=original,
    )
    result = _run(orchestrator)
    replacement = _ExternalProposalProvider(
        ActionCode.INVESTIGATE_FEATURE_DRIFT
    )
    replacement.provider_id = "different-external-host"
    fresh = AgentOrchestrator(
        planner=Planner(allowed_tools=_TOOL_TYPES),
        reviewer=Reviewer(),
        gateway=gateway,
        sessions=sessions,
        evidence_resolver=store,
        decision_controller=DecisionController(store),
        decision_provider=replacement,
    )

    with pytest.raises(RuntimeError, match="^agent result is unavailable$"):
        fresh.validate_terminal_result(
            result,
            objective="comprehensive",
            dataset_id="synthetic_demo",
            session_id="session-agent",
            metadata_grade="A",
        )


def test_terminal_validation_rejects_disabled_fallback_cache_for_external_primary(
    tmp_path: Path,
) -> None:
    gateway = _DecisionGateway()
    orchestrator, sessions, store, _ = _controlled_orchestrator(tmp_path, gateway)
    result = _run(orchestrator)
    external = _ExternalProposalProvider(ActionCode.INVESTIGATE_FEATURE_DRIFT)
    fresh = AgentOrchestrator(
        planner=Planner(allowed_tools=_TOOL_TYPES),
        reviewer=Reviewer(),
        gateway=gateway,
        sessions=sessions,
        evidence_resolver=store,
        decision_controller=DecisionController(store),
        decision_provider=external,
        decision_fallback=DeterministicDecisionProvider(),
    )

    with pytest.raises(RuntimeError, match="^agent result is unavailable$"):
        fresh.validate_terminal_result(
            result,
            objective="comprehensive",
            dataset_id="synthetic_demo",
            session_id="session-agent",
            metadata_grade="A",
        )


def test_terminal_validation_rejects_cache_from_different_provider_provenance(
    tmp_path: Path,
) -> None:
    gateway = _DecisionGateway()
    external = _ExternalProposalProvider(ActionCode.INVESTIGATE_FEATURE_DRIFT)
    orchestrator, sessions, store, _ = _controlled_orchestrator(
        tmp_path,
        gateway,
        provider=external,
    )
    result = _run(orchestrator)
    fresh = AgentOrchestrator(
        planner=Planner(allowed_tools=_TOOL_TYPES),
        reviewer=Reviewer(),
        gateway=gateway,
        sessions=sessions,
        evidence_resolver=store,
        decision_controller=DecisionController(store),
    )

    with pytest.raises(RuntimeError, match="^agent result is unavailable$"):
        fresh.validate_terminal_result(
            result,
            objective="comprehensive",
            dataset_id="synthetic_demo",
            session_id="session-agent",
            metadata_grade="A",
        )


def _base_analysis_summary(
    *,
    row_count: int,
    time_mode: str,
    time_applied: bool,
) -> AnalysisSummary:
    return AnalysisSummary(
        stages=tuple(
            StageSummary(
                name=name,
                status=StageStatus.NOT_RUN,
                enabled=True,
                output_available=False,
            )
            for name in StageName
        ),
        profile=ProfileSummary(
            row_count=row_count,
            feature_count=4,
            numeric_feature_count=3,
            positive_rate=0.2,
            segment_count=2,
            segment_min_size=20,
            segment_max_size=max(20, row_count),
            metadata_grade="A",
        ),
        partition=PartitionSummary(
            requested_time_validation=time_mode != "disabled",
            applied_time_validation=time_applied,
            mode=time_mode,
            strategy="time_ordered" if time_applied else "stratified_fallback",
            train_rows=max(0, row_count - 2),
            test_rows=min(1, row_count),
            holdout_rows=min(1, max(0, row_count - 1)),
            excluded_null_snapshot_rows=0,
            fallback_reason_code=(
                None if time_applied or time_mode == "disabled" else "time_split_unavailable"
            ),
        ),
    )


def test_agent_analysis_summary_preserves_authoritative_diagnosis_stages() -> None:
    summary = _base_analysis_summary(
        row_count=100,
        time_mode="strict",
        time_applied=True,
    )
    authoritative = {
        StageName.DIAGNOSE_QUALITY: StageSummary(
            name=StageName.DIAGNOSE_QUALITY,
            status=StageStatus.FAILED,
            enabled=True,
            output_available=False,
            reason_code="diagnose_quality_failed",
        ),
        StageName.DIAGNOSE_FEATURE_DRIFT: StageSummary(
            name=StageName.DIAGNOSE_FEATURE_DRIFT,
            status=StageStatus.REUSED,
            enabled=True,
            output_available=True,
            duration_ms=7,
        ),
        StageName.DIAGNOSE_POPULATION_SHIFT: StageSummary(
            name=StageName.DIAGNOSE_POPULATION_SHIFT,
            status=StageStatus.UNAVAILABLE,
            enabled=True,
            output_available=False,
            reason_code="population_reference_unavailable",
        ),
        StageName.DIAGNOSE_TARGET_SHIFT: StageSummary(
            name=StageName.DIAGNOSE_TARGET_SHIFT,
            status=StageStatus.SKIPPED,
            enabled=False,
            output_available=False,
            reason_code="target_shift_disabled",
        ),
        StageName.DIAGNOSE_SEGMENT_RISK: StageSummary(
            name=StageName.DIAGNOSE_SEGMENT_RISK,
            status=StageStatus.SUCCEEDED,
            enabled=True,
            output_available=True,
            limitations=("aggregate_only",),
        ),
        StageName.DIAGNOSE_TIME_STABILITY: StageSummary(
            name=StageName.DIAGNOSE_TIME_STABILITY,
            status=StageStatus.SKIPPED,
            enabled=False,
            output_available=False,
            reason_code="time_validation_disabled",
        ),
        StageName.DIAGNOSE_RULE_EVIDENCE: StageSummary(
            name=StageName.DIAGNOSE_RULE_EVIDENCE,
            status=StageStatus.UNAVAILABLE,
            enabled=True,
            output_available=False,
            reason_code="rule_evidence_unavailable",
        ),
    }
    payload = summary.model_dump(mode="python")
    payload["stages"] = tuple(
        authoritative.get(stage.name, stage) for stage in summary.stages
    )
    summary = AnalysisSummary.model_validate(payload)

    projected = orchestrator_module._agent_analysis_summary(
        summary,
        inspect_succeeded=True,
        diagnose_succeeded=True,
        discover_succeeded=True,
    )

    assert projected is not None
    stages = {stage.name: stage for stage in projected.stages}
    for name, expected in authoritative.items():
        assert stages[name] == expected


@pytest.mark.parametrize(
    (
        "row_count",
        "time_mode",
        "time_applied",
        "drift_status",
        "drift_reason",
        "time_status",
        "time_reason",
    ),
    (
        (
            100,
            "strict",
            True,
            StageStatus.SUCCEEDED,
            None,
            StageStatus.SUCCEEDED,
            None,
        ),
        (
            1,
            "strict",
            True,
            StageStatus.UNAVAILABLE,
            "insufficient_drift_reference",
            StageStatus.SUCCEEDED,
            None,
        ),
        (
            100,
            "disabled",
            False,
            StageStatus.SUCCEEDED,
            None,
            StageStatus.SKIPPED,
            "time_validation_disabled",
        ),
        (
            100,
            "auto",
            False,
            StageStatus.SUCCEEDED,
            None,
            StageStatus.UNAVAILABLE,
            "time_validation_unavailable",
        ),
    ),
)
def test_agent_analysis_summary_uses_executed_diagnosis_facts(
    row_count: int,
    time_mode: str,
    time_applied: bool,
    drift_status: StageStatus,
    drift_reason: str | None,
    time_status: StageStatus,
    time_reason: str | None,
) -> None:
    projector = getattr(orchestrator_module, "_agent_analysis_summary", None)
    assert callable(projector)

    projected = projector(
        _base_analysis_summary(
            row_count=row_count,
            time_mode=time_mode,
            time_applied=time_applied,
        ),
        inspect_succeeded=True,
        diagnose_succeeded=True,
        discover_succeeded=True,
    )

    assert projected is not None
    assert len(projected.stages) == 24
    assert len({stage.name for stage in projected.stages}) == 24
    stages = {stage.name: stage for stage in projected.stages}
    assert stages[StageName.INSPECT].status is StageStatus.SUCCEEDED
    assert stages[StageName.DIAGNOSE_QUALITY].status is StageStatus.SUCCEEDED
    assert stages[StageName.DIAGNOSE_SEGMENT_RISK].status is StageStatus.SUCCEEDED
    for name in (
        StageName.DIAGNOSE_FEATURE_DRIFT,
        StageName.DIAGNOSE_POPULATION_SHIFT,
        StageName.DIAGNOSE_TARGET_SHIFT,
    ):
        assert stages[name].status is drift_status
        assert stages[name].reason_code == drift_reason
    assert stages[StageName.DIAGNOSE_TIME_STABILITY].status is time_status
    assert stages[StageName.DIAGNOSE_TIME_STABILITY].reason_code == time_reason
    assert stages[StageName.DIAGNOSE_RULE_EVIDENCE].status is StageStatus.NOT_RUN
    assert (
        stages[StageName.DIAGNOSE_RULE_EVIDENCE].reason_code
        == "rule_evidence_not_run"
    )
    assert stages[StageName.DISCOVER_RESTORE].status is StageStatus.SUCCEEDED


def test_agent_analysis_summary_does_not_infer_diagnosis_from_the_plan() -> None:
    projector = getattr(orchestrator_module, "_agent_analysis_summary", None)
    assert callable(projector)

    projected = projector(
        _base_analysis_summary(
            row_count=100,
            time_mode="strict",
            time_applied=True,
        ),
        inspect_succeeded=True,
        diagnose_succeeded=False,
        discover_succeeded=False,
    )

    assert projected is not None
    stages = {stage.name: stage for stage in projected.stages}
    for name in (
        StageName.DIAGNOSE_QUALITY,
        StageName.DIAGNOSE_FEATURE_DRIFT,
        StageName.DIAGNOSE_POPULATION_SHIFT,
        StageName.DIAGNOSE_TARGET_SHIFT,
        StageName.DIAGNOSE_SEGMENT_RISK,
        StageName.DIAGNOSE_TIME_STABILITY,
        StageName.DIAGNOSE_RULE_EVIDENCE,
    ):
        assert stages[name].status is StageStatus.NOT_RUN
    assert stages[StageName.DISCOVER_RESTORE].status is StageStatus.NOT_RUN
