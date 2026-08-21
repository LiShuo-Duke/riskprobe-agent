"""Typed offline planner-to-gateway-to-evidence-to-reviewer state machine."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from riskprobe.agents.contracts import (
    AgentResult,
    AgentState,
    AgentStatus,
    ExecutionPlan,
    ReviewDecision,
)
from riskprobe.agents.decision_contracts import (
    DecisionContext,
    DecisionProposal,
    DecisionSource,
    DecisionStatus,
)
from riskprobe.agents.decision_controller import (
    DecisionController,
    DecisionSubmission,
    _DecisionUnavailableOutcome,
    _DecisionUnavailableReason,
)
from riskprobe.agents.decision_providers import (
    DecisionDisposition,
    DecisionProvider,
    DecisionProviderMode,
    DecisionProviderResolution,
    DeterministicDecisionProvider,
    _DecisionProviderBinding,
    _DecisionProviderIdentity,
    _DecisionProviderRole,
    default_decision_provider,
)
from riskprobe.agents.planner import Planner, PlanningError
from riskprobe.agents.reviewer import Reviewer
from riskprobe.agents.sessions import (
    SessionNode,
    SessionNodeKind,
    SessionStore,
    SessionToolCall,
)
from riskprobe.evidence import (
    EvidenceRecord,
    EvidenceStore,
    PrivacyClass,
    assert_safe_payload as assert_safe_evidence_payload,
)
from riskprobe.policy import Budget, Principal
from riskprobe.privacy import assert_safe_payload
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
    ToolGateway,
    ToolRequest,
    ToolResponse,
)
from riskprobe.tools.models import _ControlledRecommendRequest

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DIAGNOSTIC_KIND = "diagnostic.finding"
_RECOMMENDATION_KIND = "recommendation"
_DECISION_AUDIT_TOOL = "decision"
_DECISION_AUDIT_SUMMARY = "controlled decision audited"
_DECISION_UNAVAILABLE_AUDIT_SUMMARY = "controlled decision unavailable audited"
_DECISION_AUDIT_FIELDS = frozenset(
    {
        "context_evidence_id",
        "proposal_evidence_id",
        "result_evidence_id",
    }
)
_DECISION_UNAVAILABLE_AUDIT_FIELDS = frozenset(
    {"context_evidence_id", "outcome_evidence_id"}
)
_TOOL_FAILURE_SUMMARIES = frozenset(
    {
        "tool permission denied",
        "tool invocation failed",
        "tool response contract failed",
        "unsafe tool payload rejected",
    }
)
_DISCOVER_SUCCESS_SUMMARY = re.compile(r"^discovery rule count [0-9]+$")


@runtime_checkable
class EvidenceResolver(Protocol):
    def get(self, evidence_id: str) -> EvidenceRecord | None: ...


class AgentOrchestrator:
    """Execute immutable plans and persist only redacted append-only session nodes."""

    def __init__(
        self,
        *,
        planner: Planner,
        reviewer: Reviewer,
        gateway: ToolGateway | object,
        sessions: SessionStore,
        evidence_resolver: EvidenceResolver,
        decision_controller: DecisionController | None = None,
        decision_provider: DecisionProvider | object | None = None,
        decision_fallback: DecisionProvider | object | None = None,
        version: str = "orchestrator-v1",
    ) -> None:
        if not isinstance(planner, Planner):
            raise TypeError("planner must be a Planner")
        if not isinstance(reviewer, Reviewer):
            raise TypeError("reviewer must be a Reviewer")
        if not isinstance(sessions, SessionStore):
            raise TypeError("sessions must be a SessionStore")
        if not callable(getattr(gateway, "invoke", None)):
            raise TypeError("gateway must implement invoke")
        if not callable(getattr(evidence_resolver, "get", None)):
            raise TypeError("evidence_resolver must implement get")
        if decision_controller is not None and type(decision_controller) is not DecisionController:
            raise TypeError("decision_controller must be a DecisionController")
        selected_provider = (
            default_decision_provider()
            if decision_provider is None
            else decision_provider
        )
        selected_fallback = (
            DeterministicDecisionProvider()
            if decision_fallback is None
            else decision_fallback
        )
        _require_provider(selected_provider, "decision_provider")
        _require_provider(selected_fallback, "decision_fallback")
        if selected_fallback.mode is not DecisionProviderMode.DETERMINISTIC:
            raise TypeError("decision_fallback must be deterministic")
        self._planner = planner
        self._reviewer = reviewer
        self._gateway = gateway
        self._sessions = sessions
        self._evidence_resolver = evidence_resolver
        self._decision_controller = decision_controller
        self._decision_provider = selected_provider
        self._decision_fallback = selected_fallback
        self.version = version

    def run(
        self,
        *,
        objective: str,
        dataset_id: str,
        principal: Principal,
        budget: Budget,
        session_id: str | None = None,
    ) -> AgentResult:
        states: list[AgentState] = [AgentState.PLANNING]
        plan = self._planner.plan(objective=objective, dataset_id=dataset_id)
        root = self._sessions.create_session(
            session_id=session_id,
            goal=objective,
            redacted_summary="safe objective accepted",
            component_versions={
                "orchestrator": self.version,
                "planner": self._planner.version,
                "reviewer": self._reviewer.version,
            },
        )
        leaf = root
        retry_count = 0
        final_decision: ReviewDecision | None = None
        final_evidence: tuple[str, ...] = ()
        final_diagnosis: tuple[str, ...] = ()

        while True:
            if retry_count:
                states.append(AgentState.RETRYING)
                leaf = self._sessions.retry(
                    leaf.node_id,
                    redacted_summary="single child retry started",
                    component_versions={"orchestrator": self.version},
                )

            outcome = self._execute_once(
                plan=plan,
                principal=principal,
                budget=budget,
                parent=leaf,
                retry_count=retry_count,
                states=states,
                expected_run_id=root.session_id,
            )
            leaf = outcome.leaf
            final_decision = outcome.decision
            final_evidence = outcome.evidence_ids
            final_diagnosis = outcome.diagnosis_evidence_ids
            if final_decision.approved:
                states.append(AgentState.COMPLETED)
                status = AgentStatus.SUCCEEDED
                summary = "comprehensive objective approved from aggregate evidence"
                break
            if final_decision.retry_allowed and retry_count == 0:
                try:
                    plan = self._planner.repair(
                        objective=objective,
                        dataset_id=dataset_id,
                        decision=final_decision,
                        retry_count=retry_count,
                    )
                except PlanningError:
                    pass
                else:
                    retry_count = 1
                    continue
            states.append(AgentState.REJECTED)
            status = AgentStatus.REJECTED
            summary = "comprehensive objective rejected by deterministic review"
            break

        return AgentResult(
            session_id=root.session_id,
            status=status,
            plan=plan,
            review=final_decision,
            tool_sequence=plan.tool_sequence,
            evidence_ids=final_evidence,
            diagnosis_evidence_ids=final_diagnosis,
            retry_count=retry_count,
            state_history=tuple(states),
            leaf_node_id=leaf.node_id,
            redacted_summary=summary,
        )

    execute = run

    def validate_terminal_result(
        self,
        result: AgentResult,
        *,
        objective: str,
        dataset_id: str,
        session_id: str,
        metadata_grade: str,
    ) -> AgentResult:
        """Revalidate a persisted terminal result against deterministic sidecars."""

        try:
            if type(result) is not AgentResult or metadata_grade not in {"A", "B"}:
                raise ValueError("invalid terminal result")
            expected_plan = self._planner.plan(
                objective=objective,
                dataset_id=dataset_id,
            )
            terminal_state = (
                AgentState.COMPLETED
                if result.status is AgentStatus.SUCCEEDED
                else AgentState.REJECTED
            )
            expected_states = [
                AgentState.PLANNING,
                AgentState.EXECUTING,
                AgentState.COLLECTING_EVIDENCE,
                AgentState.REVIEWING,
            ]
            if result.retry_count:
                expected_states.extend(
                    (
                        AgentState.RETRYING,
                        AgentState.EXECUTING,
                        AgentState.COLLECTING_EVIDENCE,
                        AgentState.REVIEWING,
                    )
                )
            expected_states.append(terminal_state)
            if (
                result.session_id != session_id
                or result.plan != expected_plan
                or result.tool_sequence != expected_plan.tool_sequence
                or result.review.evidence_ids != result.evidence_ids
                or result.review.retry_allowed
                or result.state_history != tuple(expected_states)
                or result.redacted_summary
                != (
                    "comprehensive objective approved from aggregate evidence"
                    if result.status is AgentStatus.SUCCEEDED
                    else "comprehensive objective rejected by deterministic review"
                )
            ):
                raise ValueError("terminal result binding is invalid")

            nodes = self._sessions.replay(session_id)
            branch = self._sessions.replay(
                session_id,
                leaf_node_id=result.leaf_node_id,
            )
            expected_root_versions = {
                "orchestrator": self.version,
                "planner": self._planner.version,
                "reviewer": self._reviewer.version,
            }
            if (
                not nodes
                or tuple(node.node_id for node in nodes)
                != tuple(node.node_id for node in branch)
                or nodes[0].kind is not SessionNodeKind.ROOT
                or nodes[0].component_versions != expected_root_versions
                or nodes[0].goal != objective
                or nodes[0].session_id != session_id
                or nodes[-1].node_id != result.leaf_node_id
                or any(
                    node.session_id != session_id or node.goal != objective
                    for node in nodes
                )
                or any(
                    child.parent_node_id != parent.node_id
                    for parent, child in zip(
                        nodes,
                        nodes[1:],
                        strict=False,
                    )
                )
                or tuple(node.sequence for node in nodes)
                != tuple(range(1, len(nodes) + 1))
            ):
                raise ValueError("terminal session binding is invalid")

            attempts = _session_attempts(nodes)
            if len(attempts) != result.retry_count + 1:
                raise ValueError("terminal attempt count is invalid")
            expected_tools = expected_plan.tool_sequence[:-1]
            expected_requests = {
                step.tool_name: step.request
                for step in expected_plan.steps
                if step.request is not None
            }
            terminal_submission: DecisionSubmission | None = None
            decision_evidence_ids: set[str] = set()
            final_diagnosis_ids: tuple[str, ...] = ()
            final_recommendation_ids: tuple[str, ...] = ()
            for attempt_index, attempt in enumerate(attempts):
                names = tuple(_node_tool_name(node) for node in attempt)
                public_names = tuple(
                    name for name in names if name != _DECISION_AUDIT_TOOL
                )
                if (
                    not public_names
                    or public_names[-1] != "review"
                    or public_names[:-1]
                    != expected_tools[: len(public_names) - 1]
                    or names[-1] != "review"
                    or (
                        attempt_index == len(attempts) - 1
                        and result.status is AgentStatus.SUCCEEDED
                        and public_names != expected_plan.tool_sequence
                    )
                ):
                    raise ValueError("terminal attempt sequence is invalid")

                business_evidence: set[str] = set()
                diagnosis_ids: tuple[str, ...] = ()
                recommendation_ids: set[str] = set()
                discover_node: SessionNode | None = None
                discover_succeeded = False
                audit_count = 0
                audit_outcome: (
                    DecisionSubmission | _DecisionUnavailableOutcome | None
                ) = None
                permission_denied = False
                unsafe_payload_detected = False
                tool_failed = False

                for node_index, node in enumerate(attempt[:-1]):
                    name = names[node_index]
                    if name == _DECISION_AUDIT_TOOL:
                        audit_count += 1
                        if (
                            audit_count != 1
                            or discover_node is None
                            or not discover_succeeded
                            or not diagnosis_ids
                            or node_index == 0
                            or attempt[node_index - 1].node_id
                            != discover_node.node_id
                        ):
                            raise ValueError(
                                "terminal decision audit position is invalid"
                            )
                        audit_outcome = self._validate_decision_audit(
                            node,
                            expected_run_id=session_id,
                            expected_dataset_id=dataset_id,
                            expected_attempt=attempt_index,
                            expected_anchor_node_id=discover_node.node_id,
                            expected_diagnosis_ids=diagnosis_ids,
                            expected_metadata_grade=metadata_grade,
                        )
                        decision_evidence_ids.update(node.evidence_ids)
                        next_name = names[node_index + 1]
                        if isinstance(audit_outcome, DecisionSubmission):
                            if (
                                audit_outcome.result.status
                                is DecisionStatus.ACCEPTED
                            ):
                                if next_name != "recommend":
                                    raise ValueError(
                                        "accepted decision did not bind recommend"
                                    )
                            elif next_name != "review":
                                raise ValueError(
                                    "rejected decision did not fail closed"
                                )
                            else:
                                tool_failed = True
                            if attempt_index == len(attempts) - 1:
                                terminal_submission = audit_outcome
                        elif next_name != "review":
                            raise ValueError(
                                "unavailable decision did not fail closed"
                            )
                        else:
                            tool_failed = True
                        continue

                    if (
                        node.component_versions
                        != {"orchestrator": self.version}
                        or node.tool_call is None
                        or name not in expected_requests
                    ):
                        raise ValueError("terminal tool node is invalid")
                    expected_arguments = expected_requests[name].model_dump(
                        mode="python"
                    )
                    if name == "recommend":
                        expected_arguments = RecommendRequest(
                            dataset_id=dataset_id,
                            evidence_ids=diagnosis_ids,
                        ).model_dump(mode="python")
                    if dict(node.tool_call.arguments) != expected_arguments:
                        raise ValueError("terminal tool arguments are invalid")

                    summary = node.redacted_summary
                    if summary in _TOOL_FAILURE_SUMMARIES:
                        if (
                            node.evidence_ids
                            != tuple(sorted(business_evidence))
                            or node_index != len(attempt) - 2
                        ):
                            raise ValueError(
                                "terminal failed tool evidence is invalid"
                            )
                        if name == "discover":
                            discover_node = node
                        permission_denied = (
                            permission_denied
                            or summary == "tool permission denied"
                        )
                        unsafe_payload_detected = (
                            unsafe_payload_detected
                            or summary == "unsafe tool payload rejected"
                        )
                        tool_failed = tool_failed or summary in {
                            "tool invocation failed",
                            "tool response contract failed",
                        }
                        continue

                    if name == "inspect":
                        if (
                            summary != "inspection aggregates recorded"
                            or node.evidence_ids
                            != tuple(sorted(business_evidence))
                        ):
                            raise ValueError(
                                "terminal inspection evidence is invalid"
                            )
                    elif name == "diagnose":
                        if summary != (
                            f"diagnosis evidence count {len(node.evidence_ids)}"
                        ):
                            raise ValueError(
                                "terminal diagnosis summary is invalid"
                            )
                        diagnosis_ids = node.evidence_ids
                        business_evidence = set(diagnosis_ids)
                    elif name == "discover":
                        discover_node = node
                        discover_succeeded = (
                            isinstance(summary, str)
                            and _DISCOVER_SUCCESS_SUMMARY.fullmatch(summary)
                            is not None
                        )
                        if (
                            not discover_succeeded
                            or node.evidence_ids
                            != tuple(sorted(business_evidence))
                        ):
                            raise ValueError(
                                "terminal discovery evidence is invalid"
                            )
                    elif name == "recommend":
                        node_evidence = set(node.evidence_ids)
                        if not business_evidence.issubset(node_evidence):
                            raise ValueError(
                                "terminal recommendation evidence is invalid"
                            )
                        new_ids = node_evidence - business_evidence
                        if summary != (
                            f"recommendation evidence count {len(new_ids)}"
                        ):
                            raise ValueError(
                                "terminal recommendation summary is invalid"
                            )
                        recommendation_ids.update(new_ids)
                        business_evidence = node_evidence
                    else:
                        raise ValueError("terminal tool sequence is invalid")

                if (
                    self._decision_controller is not None
                    and diagnosis_ids
                    and discover_succeeded
                ):
                    if audit_count != 1:
                        raise ValueError("terminal decision audit is missing")
                elif audit_count:
                    raise ValueError("terminal decision audit count is invalid")

                review = attempt[-1]
                if review.tool_call is None:
                    raise ValueError("terminal review is unavailable")
                expected_actions = (
                    audit_outcome.result.action_codes
                    if isinstance(audit_outcome, DecisionSubmission)
                    else None
                )
                (
                    _,
                    attempt_evidence,
                    resolver_unsafe,
                    resolver_failed,
                ) = self._resolve_evidence(
                    diagnosis_ids=diagnosis_ids,
                    recommendation_ids=tuple(sorted(recommendation_ids)),
                    expected_run_id=session_id,
                    metadata_grade=metadata_grade,
                    expected_action_codes=expected_actions,
                )
                unsafe_payload_detected = (
                    unsafe_payload_detected or resolver_unsafe
                )
                tool_failed = tool_failed or resolver_failed
                review_diagnosis = set(diagnosis_ids).intersection(
                    attempt_evidence
                )
                expected_review = self._reviewer.review(
                    expected_plan,
                    evidence_ids=tuple(sorted(attempt_evidence)),
                    diagnosis_evidence_ids=tuple(sorted(review_diagnosis)),
                    claimed_evidence_ids=diagnosis_ids,
                    metadata_grade=metadata_grade,
                    payloads=(),
                    permission_denied=permission_denied,
                    unsafe_payload_detected=unsafe_payload_detected,
                    tool_failed=tool_failed,
                    retry_count=attempt_index,
                )
                expected_review_arguments = {
                    "approved": expected_review.approved,
                    "reason_codes": tuple(
                        reason.value for reason in expected_review.reason_codes
                    ),
                    "retry_count": attempt_index,
                }
                review_arguments = dict(review.tool_call.arguments)
                if (
                    review_arguments != expected_review_arguments
                    or review.evidence_ids != expected_review.evidence_ids
                    or review.component_versions
                    != {"reviewer": self._reviewer.version}
                    or review.redacted_summary
                    != (
                        "deterministic review approved"
                        if expected_review.approved
                        else "deterministic review rejected"
                    )
                    or decision_evidence_ids.intersection(review.evidence_ids)
                ):
                    raise ValueError("terminal review binding is invalid")

                if attempt_index < len(attempts) - 1:
                    review_position = nodes.index(review)
                    retry = nodes[review_position + 1]
                    if (
                        expected_review.approved
                        or not expected_review.retry_allowed
                        or retry.kind is not SessionNodeKind.RETRY
                        or retry.parent_node_id != review.node_id
                        or retry.retry_of_node_id != review.node_id
                        or retry.tool_call is not None
                    ):
                        raise ValueError("terminal retry review is invalid")
                else:
                    if (
                        review.node_id != result.leaf_node_id
                        or expected_review != result.review
                        or review.evidence_ids != result.evidence_ids
                    ):
                        raise ValueError("terminal review binding is invalid")
                    final_diagnosis_ids = diagnosis_ids
                    final_recommendation_ids = tuple(
                        sorted(recommendation_ids)
                    )

            if (
                result.status is AgentStatus.SUCCEEDED
                and result.diagnosis_evidence_ids != final_diagnosis_ids
            ) or (
                result.status is AgentStatus.REJECTED
                and not set(result.diagnosis_evidence_ids).issubset(
                    final_diagnosis_ids
                )
            ):
                raise ValueError("terminal diagnosis binding is invalid")

            verify_chain = getattr(self._evidence_resolver, "verify_chain", None)
            if not callable(verify_chain) or verify_chain(session_id) is not True:
                raise ValueError("terminal evidence chain is invalid")
            expected_actions = (
                None
                if terminal_submission is None
                else terminal_submission.result.action_codes
            )
            _, resolved_ids, unsafe, failed = self._resolve_evidence(
                diagnosis_ids=final_diagnosis_ids,
                recommendation_ids=final_recommendation_ids,
                expected_run_id=session_id,
                metadata_grade=metadata_grade,
                expected_action_codes=expected_actions,
            )
            if unsafe or failed or resolved_ids != set(result.evidence_ids):
                raise ValueError("terminal evidence binding is invalid")
            return result
        except Exception as error:
            raise RuntimeError("agent result is unavailable") from error

    def _execute_once(
        self,
        *,
        plan: ExecutionPlan,
        principal: Principal,
        budget: Budget,
        parent: SessionNode,
        retry_count: int,
        states: list[AgentState],
        expected_run_id: str,
    ) -> _AttemptOutcome:
        states.append(AgentState.EXECUTING)
        leaf = parent
        evidence: set[str] = set()
        diagnosis_evidence: set[str] = set()
        recommendation_evidence: set[str] = set()
        payloads: list[object] = []
        metadata_grade: str = "A"
        inspect_response: InspectResponse | None = None
        decision_submission: DecisionSubmission | None = None
        permission_denied = False
        unsafe_payload = False
        tool_failed = False

        for step in plan.steps:
            if step.tool_name == "review":
                break
            request = self._request_for_step(
                step.request,
                diagnosis_evidence,
                decision_submission,
            )
            try:
                response = self._gateway.invoke(principal, request, budget)
            except PermissionError:
                permission_denied = True
                leaf = self._append_tool_node(
                    leaf,
                    request,
                    evidence_ids=tuple(sorted(evidence)),
                    summary="tool permission denied",
                )
                break
            except Exception:
                tool_failed = True
                leaf = self._append_tool_node(
                    leaf,
                    request,
                    evidence_ids=tuple(sorted(evidence)),
                    summary="tool invocation failed",
                )
                break

            if not _response_matches_request(request, response):
                tool_failed = True
                leaf = self._append_tool_node(
                    leaf,
                    request,
                    evidence_ids=tuple(sorted(evidence)),
                    summary="tool response contract failed",
                )
                break
            if not isinstance(response, BaseModel) or not self._response_is_safe(response):
                unsafe_payload = True
                leaf = self._append_tool_node(
                    leaf,
                    request,
                    evidence_ids=tuple(sorted(evidence)),
                    summary="unsafe tool payload rejected",
                )
                break
            payloads.append(response.model_dump(mode="json"))
            try:
                if isinstance(response, DiagnoseResponse):
                    finding_ids = _validated_sha_ids(response.finding_ids)
                    diagnosis_evidence.update(finding_ids)
                    evidence.update(finding_ids)
                elif isinstance(response, RecommendResponse):
                    recommendation_ids = _validated_sha_ids(
                        response.recommendation_ids
                    )
                    recommendation_evidence.update(recommendation_ids)
                    evidence.update(recommendation_ids)
            except (TypeError, ValueError):
                tool_failed = True
                leaf = self._append_tool_node(
                    leaf,
                    request,
                    evidence_ids=tuple(sorted(evidence)),
                    summary="tool response contract failed",
                )
                break

            if isinstance(response, InspectResponse):
                inspect_response = response
                metadata_grade = response.metadata_grade
            leaf = self._append_tool_node(
                leaf,
                request,
                evidence_ids=tuple(sorted(evidence)),
                summary=_response_summary(response),
            )

            if (
                isinstance(response, DiscoverResponse)
                and diagnosis_evidence
                and self._decision_controller is not None
            ):
                try:
                    if inspect_response is None:
                        raise ValueError("inspection response is unavailable")
                    preparation = self._decision_controller.prepare(
                        session_id=expected_run_id,
                        attempt=retry_count,
                        anchor_node_id=leaf.node_id,
                        diagnosis_evidence_ids=tuple(sorted(diagnosis_evidence)),
                        inspect_response=inspect_response,
                        discover_response=response,
                        orchestrator_version=self.version,
                        planner_version=self._planner.version,
                    )
                except Exception:
                    tool_failed = True
                    break
                try:
                    proposal, provider_binding = self._resolve_decision_proposal(
                        preparation.context
                    )
                except _DecisionProviderUnavailable as unavailable:
                    try:
                        outcome = self._decision_controller.record_unavailable(
                            context_evidence_id=(
                                preparation.context_evidence_id
                            ),
                            reason=unavailable.reason,
                            provider_binding=unavailable.provider_binding,
                        )
                        leaf = self._append_decision_node(leaf, outcome)
                    except Exception:
                        tool_failed = True
                        break
                    tool_failed = True
                    break
                try:
                    decision_submission = self._decision_controller.submit(
                        context_evidence_id=preparation.context_evidence_id,
                        proposal=proposal,
                        provider_binding=provider_binding,
                    )
                    leaf = self._append_decision_node(leaf, decision_submission)
                    if (
                        decision_submission.result.status
                        is not DecisionStatus.ACCEPTED
                    ):
                        tool_failed = True
                        break
                except Exception:
                    tool_failed = True
                    break

        states.extend((AgentState.COLLECTING_EVIDENCE, AgentState.REVIEWING))
        expected_actions = (
            None
            if decision_submission is None
            else decision_submission.result.action_codes
        )
        resolved_payloads, resolved_ids, resolver_unsafe, resolver_failed = (
            self._resolve_evidence(
                diagnosis_ids=tuple(sorted(diagnosis_evidence)),
                recommendation_ids=tuple(sorted(recommendation_evidence)),
                expected_run_id=expected_run_id,
                metadata_grade="B" if metadata_grade == "B" else "A",
                expected_action_codes=expected_actions,
            )
        )
        payloads.extend(resolved_payloads)
        review_diagnosis = diagnosis_evidence.intersection(resolved_ids)
        unsafe_payload = unsafe_payload or resolver_unsafe
        tool_failed = tool_failed or resolver_failed
        decision = self._reviewer.review(
            plan,
            evidence_ids=tuple(sorted(resolved_ids)),
            diagnosis_evidence_ids=tuple(sorted(review_diagnosis)),
            claimed_evidence_ids=tuple(sorted(diagnosis_evidence)),
            metadata_grade="B" if metadata_grade == "B" else "A",
            payloads=tuple(payloads),
            permission_denied=permission_denied,
            unsafe_payload_detected=unsafe_payload,
            tool_failed=tool_failed,
            retry_count=retry_count,
        )
        review_call = SessionToolCall(
            tool_name="review",
            arguments={
                "approved": decision.approved,
                "reason_codes": tuple(
                    reason.value for reason in decision.reason_codes
                ),
                "retry_count": retry_count,
            },
        )
        leaf = self._sessions.append_child(
            leaf.node_id,
            tool_call=review_call,
            evidence_ids=decision.evidence_ids,
            redacted_summary=(
                "deterministic review approved"
                if decision.approved
                else "deterministic review rejected"
            ),
            component_versions={"reviewer": self._reviewer.version},
        )
        return _AttemptOutcome(
            decision=decision,
            diagnosis_evidence_ids=tuple(sorted(review_diagnosis)),
            evidence_ids=tuple(sorted(resolved_ids)),
            leaf=leaf,
        )

    @staticmethod
    def _request_for_step(
        request: ToolRequest | None,
        diagnosis_evidence: set[str],
        decision_submission: DecisionSubmission | None,
    ) -> ToolRequest:
        if request is None:
            raise TypeError("tool step requires a typed request")
        if isinstance(request, RecommendRequest):
            evidence_ids = tuple(sorted(diagnosis_evidence))
            if decision_submission is None:
                return RecommendRequest(
                    dataset_id=request.dataset_id,
                    evidence_ids=evidence_ids,
                )
            if (
                decision_submission.result.status is not DecisionStatus.ACCEPTED
                or decision_submission.result.diagnosis_evidence_ids != evidence_ids
            ):
                raise RuntimeError("decision is unavailable")
            return _ControlledRecommendRequest(
                dataset_id=request.dataset_id,
                evidence_ids=evidence_ids,
                decision_result_evidence_id=(
                    decision_submission.result_evidence_id
                ),
            )
        return request

    def _resolve_decision_proposal(
        self,
        context: DecisionContext,
    ) -> tuple[DecisionProposal, _DecisionProviderBinding]:
        provider = self._decision_provider
        selected_role = _DecisionProviderRole.PRIMARY
        try:
            resolution = _provider_resolution(provider, context)
        except Exception:
            raise _DecisionProviderUnavailable(
                reason=_DecisionUnavailableReason.PROVIDER_ERROR,
                provider_binding=self._provider_binding(selected_role),
            ) from None
        if resolution.disposition is DecisionDisposition.FALLBACK:
            provider = self._decision_fallback
            selected_role = _DecisionProviderRole.FALLBACK
            try:
                resolution = _provider_resolution(provider, context)
            except Exception:
                raise _DecisionProviderUnavailable(
                    reason=_DecisionUnavailableReason.PROVIDER_ERROR,
                    provider_binding=self._provider_binding(selected_role),
                ) from None
        if resolution.disposition is DecisionDisposition.PENDING:
            raise _DecisionProviderUnavailable(
                reason=_DecisionUnavailableReason.PROVIDER_PENDING,
                provider_binding=self._provider_binding(selected_role),
            )
        if (
            resolution.disposition is not DecisionDisposition.PROPOSAL
            or resolution.proposal is None
        ):
            raise _DecisionProviderUnavailable(
                reason=_DecisionUnavailableReason.PROVIDER_ERROR,
                provider_binding=self._provider_binding(selected_role),
            )
        proposal = resolution.proposal
        expected_source = (
            DecisionSource.DETERMINISTIC
            if provider.mode is DecisionProviderMode.DETERMINISTIC
            else DecisionSource.EXTERNAL_HOST
        )
        if (
            provider.mode is DecisionProviderMode.DISABLED
            or proposal.source is not expected_source
            or proposal.source_version != provider.version
        ):
            raise _DecisionProviderUnavailable(
                reason=_DecisionUnavailableReason.PROVIDER_ERROR,
                provider_binding=self._provider_binding(selected_role),
            )
        return proposal, self._provider_binding(selected_role)

    def _provider_binding(
        self,
        selected_role: _DecisionProviderRole,
    ) -> _DecisionProviderBinding:
        primary = _provider_identity(self._decision_provider)
        fallback = _provider_identity(self._decision_fallback)
        return _DecisionProviderBinding(
            primary=primary,
            fallback=fallback,
            selected=(
                primary
                if selected_role is _DecisionProviderRole.PRIMARY
                else fallback
            ),
            selected_role=selected_role,
        )

    def _append_decision_node(
        self,
        parent: SessionNode,
        outcome: DecisionSubmission | _DecisionUnavailableOutcome,
    ) -> SessionNode:
        if self._decision_controller is None:
            raise RuntimeError("decision is unavailable")
        if isinstance(outcome, DecisionSubmission):
            arguments = {
                "context_evidence_id": outcome.context_evidence_id,
                "proposal_evidence_id": outcome.proposal_evidence_id,
                "result_evidence_id": outcome.result_evidence_id,
            }
            summary = _DECISION_AUDIT_SUMMARY
        else:
            arguments = {
                "context_evidence_id": outcome.context_evidence_id,
                "outcome_evidence_id": outcome.outcome_evidence_id,
            }
            summary = _DECISION_UNAVAILABLE_AUDIT_SUMMARY
        return self._sessions.append_child(
            parent.node_id,
            tool_call=SessionToolCall(
                tool_name=_DECISION_AUDIT_TOOL,
                arguments=arguments,
            ),
            evidence_ids=tuple(arguments.values()),
            redacted_summary=summary,
            component_versions={
                "decision_controller": self._decision_controller.version,
                "proposal_validator": (
                    self._decision_controller.validator_version
                ),
            },
        )

    def _validate_decision_audit(
        self,
        audit: SessionNode,
        *,
        expected_run_id: str,
        expected_dataset_id: str,
        expected_attempt: int,
        expected_anchor_node_id: str,
        expected_diagnosis_ids: Sequence[str],
        expected_metadata_grade: str,
    ) -> DecisionSubmission | _DecisionUnavailableOutcome:
        if self._decision_controller is None or audit.tool_call is None:
            raise ValueError("decision audit controller is unavailable")
        arguments = dict(audit.tool_call.arguments)
        argument_fields = set(arguments)
        expected_summary = (
            _DECISION_AUDIT_SUMMARY
            if argument_fields == _DECISION_AUDIT_FIELDS
            else _DECISION_UNAVAILABLE_AUDIT_SUMMARY
        )
        if (
            argument_fields
            not in {
                _DECISION_AUDIT_FIELDS,
                _DECISION_UNAVAILABLE_AUDIT_FIELDS,
            }
            or any(
                not isinstance(value, str) or _SHA256.fullmatch(value) is None
                for value in arguments.values()
            )
            or len(set(arguments.values())) != len(arguments)
            or audit.evidence_ids != tuple(sorted(arguments.values()))
            or audit.redacted_summary != expected_summary
            or audit.component_versions
            != {
                "decision_controller": self._decision_controller.version,
                "proposal_validator": (
                    self._decision_controller.validator_version
                ),
            }
        ):
            raise ValueError("decision audit fields are invalid")
        if argument_fields == _DECISION_AUDIT_FIELDS:
            outcome: DecisionSubmission | _DecisionUnavailableOutcome = (
                self._decision_controller.replay(
                    result_evidence_id=str(arguments["result_evidence_id"]),
                    expected_run_id=expected_run_id,
                )
            )
            if outcome.proposal_evidence_id != arguments["proposal_evidence_id"]:
                raise ValueError("decision audit binding is invalid")
        else:
            outcome = self._decision_controller.replay_unavailable(
                outcome_evidence_id=str(arguments["outcome_evidence_id"]),
                expected_run_id=expected_run_id,
            )
        context = outcome.context
        if (
            outcome.context_evidence_id
            != arguments["context_evidence_id"]
            or context.session_id != expected_run_id
            or context.dataset_id != expected_dataset_id
            or context.attempt != expected_attempt
            or context.anchor_node_id != expected_anchor_node_id
            or context.diagnosis_evidence_ids
            != tuple(sorted(expected_diagnosis_ids))
            or context.metadata_grade != expected_metadata_grade
            or context.component_versions["orchestrator"] != self.version
            or context.component_versions["planner"] != self._planner.version
            or not self._provider_binding_matches_current(
                outcome.provider_binding
            )
        ):
            raise ValueError("decision audit binding is invalid")
        return outcome

    def _provider_binding_matches_current(
        self,
        binding: _DecisionProviderBinding,
    ) -> bool:
        try:
            expected = self._provider_binding(binding.selected_role)
        except Exception:
            return False
        return binding == expected

    def _append_tool_node(
        self,
        parent: SessionNode,
        request: ToolRequest,
        *,
        evidence_ids: Sequence[str],
        summary: str,
    ) -> SessionNode:
        public_request = (
            RecommendRequest(
                dataset_id=request.dataset_id,
                evidence_ids=request.evidence_ids,
            )
            if isinstance(request, _ControlledRecommendRequest)
            else request
        )
        return self._sessions.append_child(
            parent.node_id,
            tool_call=SessionToolCall(
                tool_name=_tool_name(request),
                arguments=public_request.model_dump(mode="json"),
            ),
            evidence_ids=evidence_ids,
            redacted_summary=summary,
            component_versions={"orchestrator": self.version},
        )

    @staticmethod
    def _response_is_safe(response: BaseModel) -> bool:
        try:
            assert_safe_payload(response.model_dump(mode="json"))
        except Exception:
            return False
        return True

    def _resolve_evidence(
        self,
        *,
        diagnosis_ids: Sequence[str],
        recommendation_ids: Sequence[str],
        expected_run_id: str,
        metadata_grade: str,
        expected_action_codes: Sequence[ActionCode] | None = None,
    ) -> tuple[list[object], set[str], bool, bool]:
        diagnosis = set(diagnosis_ids)
        recommendations = set(recommendation_ids)
        if diagnosis.intersection(recommendations):
            return [], set(), False, True

        expected_kinds = {
            **{evidence_id: _DIAGNOSTIC_KIND for evidence_id in diagnosis},
            **{
                evidence_id: _RECOMMENDATION_KIND
                for evidence_id in recommendations
            },
        }
        records: dict[str, EvidenceRecord] = {}
        safe_payloads: dict[str, object] = {}
        visiting: set[str] = set()

        def resolve(evidence_id: str) -> None:
            if _SHA256.fullmatch(evidence_id) is None:
                raise _InvalidResolvedEvidence
            if evidence_id in records:
                return
            if evidence_id in visiting:
                raise _InvalidResolvedEvidence
            try:
                record = self._evidence_resolver.get(evidence_id)
            except Exception as error:
                raise _InvalidResolvedEvidence from error
            if not isinstance(record, EvidenceRecord):
                raise _InvalidResolvedEvidence
            payload = record.model_dump(mode="json")["payload"]
            try:
                assert_safe_evidence_payload(payload)
                assert_safe_payload(payload)
            except Exception as error:
                raise _UnsafeResolvedEvidence from error
            if (
                EvidenceStore.content_id(record) != evidence_id
                or record.run_id != expected_run_id
                or record.privacy_class is not PrivacyClass.AGGREGATE
                or (
                    evidence_id in expected_kinds
                    and record.kind != expected_kinds[evidence_id]
                )
            ):
                raise _InvalidResolvedEvidence

            visiting.add(evidence_id)
            try:
                for parent_id in record.parent_ids:
                    resolve(parent_id)
            finally:
                visiting.discard(evidence_id)
            records[evidence_id] = record
            safe_payloads[evidence_id] = payload

        try:
            for evidence_id in sorted(expected_kinds):
                resolve(evidence_id)

            finding_evidence: dict[str, str] = {}
            for evidence_id in sorted(diagnosis):
                if records[evidence_id].parent_ids:
                    raise _InvalidResolvedEvidence
                finding_id = _payload_sha_id(
                    records[evidence_id].payload,
                    "finding_id",
                )
                if finding_id in finding_evidence:
                    raise _InvalidResolvedEvidence
                finding_evidence[finding_id] = evidence_id

            actual_actions: set[ActionCode] = set()
            for evidence_id in sorted(recommendations):
                record = records[evidence_id]
                finding_ids = _payload_sha_ids(record.payload, "finding_ids")
                try:
                    expected_parents = tuple(
                        sorted(
                            finding_evidence[finding_id]
                            for finding_id in finding_ids
                        )
                    )
                except KeyError as error:
                    raise _InvalidResolvedEvidence from error
                if record.parent_ids != expected_parents:
                    raise _InvalidResolvedEvidence
                if record.payload.get("human_approval_required") is not True:
                    raise _InvalidResolvedEvidence
                if (
                    metadata_grade == "B"
                    and record.payload.get("decision_eligibility")
                    != "analysis_only"
                ):
                    raise _InvalidResolvedEvidence
                if expected_action_codes is not None:
                    try:
                        action = ActionCode(record.payload.get("action_code"))
                    except (TypeError, ValueError) as error:
                        raise _InvalidResolvedEvidence from error
                    if action in actual_actions:
                        raise _InvalidResolvedEvidence
                    actual_actions.add(action)
            if expected_action_codes is not None and tuple(
                sorted(actual_actions, key=lambda action: action.value)
            ) != tuple(
                sorted(expected_action_codes, key=lambda action: action.value)
            ):
                raise _InvalidResolvedEvidence
        except _UnsafeResolvedEvidence:
            return [], set(), True, False
        except _InvalidResolvedEvidence:
            return [], set(), False, True

        return (
            [safe_payloads[evidence_id] for evidence_id in sorted(records)],
            set(records),
            False,
            False,
        )


class _DecisionProviderUnavailable(RuntimeError):
    __slots__ = ("provider_binding", "reason")

    def __init__(
        self,
        *,
        reason: _DecisionUnavailableReason,
        provider_binding: _DecisionProviderBinding,
    ) -> None:
        super().__init__("decision provider is unavailable")
        self.reason = reason
        self.provider_binding = provider_binding


class _AttemptOutcome:
    __slots__ = ("decision", "diagnosis_evidence_ids", "evidence_ids", "leaf")

    def __init__(
        self,
        *,
        decision: ReviewDecision,
        diagnosis_evidence_ids: tuple[str, ...],
        evidence_ids: tuple[str, ...],
        leaf: SessionNode,
    ) -> None:
        self.decision = decision
        self.diagnosis_evidence_ids = diagnosis_evidence_ids
        self.evidence_ids = evidence_ids
        self.leaf = leaf


class _InvalidResolvedEvidence(ValueError):
    pass


class _UnsafeResolvedEvidence(ValueError):
    pass


def _provider_identity(provider: object) -> _DecisionProviderIdentity:
    return _DecisionProviderIdentity(
        provider_id=getattr(provider, "provider_id", None),
        mode=getattr(provider, "mode", None),
        version=getattr(provider, "version", None),
    )


def _require_provider(provider: object, field_name: str) -> None:
    if not callable(getattr(provider, "resolve", None)):
        raise TypeError(f"{field_name} must implement DecisionProvider")
    try:
        _provider_identity(provider)
    except Exception as error:
        raise TypeError(
            f"{field_name} must implement DecisionProvider"
        ) from error


def _provider_resolution(
    provider: object,
    context: DecisionContext,
) -> DecisionProviderResolution:
    resolution = provider.resolve(context=context)
    if type(resolution) is not DecisionProviderResolution:
        raise TypeError("decision provider returned an invalid resolution")
    return DecisionProviderResolution.model_validate(
        resolution.model_dump(mode="python")
    )


def _session_attempts(
    nodes: Sequence[SessionNode],
) -> tuple[tuple[SessionNode, ...], ...]:
    attempts: list[tuple[SessionNode, ...]] = []
    current: list[SessionNode] = []
    retry_nodes = 0
    for node in nodes[1:]:
        if node.kind is SessionNodeKind.RETRY:
            if node.tool_call is not None or not current:
                raise ValueError("terminal retry structure is invalid")
            attempts.append(tuple(current))
            current = []
            retry_nodes += 1
            continue
        if node.kind is not SessionNodeKind.CHILD or node.tool_call is None:
            raise ValueError("terminal session structure is invalid")
        current.append(node)
    if current:
        attempts.append(tuple(current))
    if retry_nodes != len(attempts) - 1:
        raise ValueError("terminal retry count is invalid")
    return tuple(attempts)


def _node_tool_name(node: SessionNode | None) -> str:
    if node is None or node.tool_call is None:
        raise ValueError("session tool call is unavailable")
    return node.tool_call.tool_name


def _validated_sha_ids(values: Sequence[str]) -> tuple[str, ...]:
    if type(values) is not tuple:
        raise TypeError("evidence IDs must be a tuple")
    if len(values) != len(set(values)) or any(
        not isinstance(value, str) or _SHA256.fullmatch(value) is None
        for value in values
    ):
        raise ValueError(
            "evidence IDs must be unique lowercase SHA-256 identifiers"
        )
    return tuple(values)


def _payload_sha_id(payload: Mapping[str, object], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise _InvalidResolvedEvidence
    return value


def _payload_sha_ids(
    payload: Mapping[str, object],
    field_name: str,
) -> tuple[str, ...]:
    value = payload.get(field_name)
    if not isinstance(value, (list, tuple)) or not value:
        raise _InvalidResolvedEvidence
    normalized = tuple(value)
    if len(normalized) != len(set(normalized)) or any(
        not isinstance(item, str) or _SHA256.fullmatch(item) is None
        for item in normalized
    ):
        raise _InvalidResolvedEvidence
    return normalized


def _response_matches_request(request: ToolRequest, response: object) -> bool:
    if type(request) is InspectRequest:
        expected = InspectResponse
    elif type(request) is DiagnoseRequest:
        expected = DiagnoseResponse
    elif type(request) is DiscoverRequest:
        expected = DiscoverResponse
    elif isinstance(request, RecommendRequest):
        expected = RecommendResponse
    else:
        return False
    return type(response) is expected and getattr(response, "dataset_id", None) == getattr(
        request,
        "dataset_id",
        None,
    )


def _tool_name(request: ToolRequest) -> str:
    if isinstance(request, _ControlledRecommendRequest):
        return "recommend"
    name = type(request).__name__
    return name.removesuffix("Request").lower()


def _response_summary(response: ToolResponse | BaseModel) -> str:
    if isinstance(response, InspectResponse):
        return "inspection aggregates recorded"
    if isinstance(response, DiagnoseResponse):
        return f"diagnosis evidence count {len(response.finding_ids)}"
    if isinstance(response, DiscoverResponse):
        return f"discovery rule count {len(response.rule_ids)}"
    if isinstance(response, RecommendResponse):
        return (
            f"recommendation evidence count {len(response.recommendation_ids)}"
        )
    return "typed tool response recorded"


Orchestrator = AgentOrchestrator

__all__ = ["AgentOrchestrator", "EvidenceResolver", "Orchestrator"]
