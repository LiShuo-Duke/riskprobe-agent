from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from riskprobe.agents.decision_contracts import (
    DecisionContext,
    DecisionFinding,
    DecisionProposal,
    DecisionSource,
    DecisionStatus,
    ProposalValidator,
    default_decision_policy,
)
from riskprobe.monitoring.models import FindingKind, FindingSeverity, RiskFinding
from riskprobe.recommendations.policy import ActionCode


def _finding() -> RiskFinding:
    return RiskFinding(
        kind=FindingKind.FEATURE_DRIFT,
        severity=FindingSeverity.WARNING,
        code="feature_psi",
        feature="order_count",
        metrics={"psi": 0.25},
    )


def _context(*, metadata_grade: str = "A") -> DecisionContext:
    issued_at = datetime(2026, 1, 1, tzinfo=UTC)
    finding = _finding()
    evidence_id = "a" * 64
    return DecisionContext(
        session_id="0123456789abcdef",
        attempt=0,
        anchor_node_id="b" * 64,
        dataset_id="synthetic_demo",
        objective="comprehensive",
        metadata_grade=metadata_grade,
        row_count=100,
        feature_count=8,
        issue_codes=("LABEL_PERFORMANCE_WINDOW_UNKNOWN",),
        rule_ids=("rule-b", "rule-a"),
        diagnosis_evidence_ids=(evidence_id,),
        findings=(DecisionFinding(evidence_id=evidence_id, finding=finding),),
        policy=default_decision_policy(),
        issued_at=issued_at,
        expires_at=issued_at + timedelta(minutes=5),
        component_versions={
            "diagnostics": "diagnostics-v1",
            "orchestrator": "orchestrator-v1",
            "planner": "planner-v1",
            "recommendations": "recommendations-v1",
        },
    )


def test_decision_contract_accepts_only_bound_complete_allowlisted_proposal() -> None:
    context = _context()
    proposal = DecisionProposal(
        context_id=context.context_id,
        diagnosis_evidence_ids=context.diagnosis_evidence_ids,
        action_codes=(ActionCode.INVESTIGATE_FEATURE_DRIFT,),
        source=DecisionSource.EXTERNAL_HOST,
        source_version="kiro-host-v1",
    )

    result = ProposalValidator().validate(
        context,
        proposal,
        now=context.issued_at + timedelta(seconds=1),
    )

    assert result.status is DecisionStatus.ACCEPTED
    assert result.reason_codes == ()
    assert result.action_codes == (ActionCode.INVESTIGATE_FEATURE_DRIFT,)
    assert result.diagnosis_evidence_ids == context.diagnosis_evidence_ids
    assert len(context.context_id) == len(proposal.proposal_id) == len(result.decision_id) == 64
    assert context.rule_ids == ("rule-a", "rule-b")


@pytest.mark.parametrize(
    "forbidden",
    ("path", "tool_sequence", "role", "budget", "payload", "approved"),
)
def test_decision_proposal_forbids_control_plane_fields(forbidden: str) -> None:
    context = _context()
    payload = {
        "context_id": context.context_id,
        "diagnosis_evidence_ids": context.diagnosis_evidence_ids,
        "action_codes": (ActionCode.INVESTIGATE_FEATURE_DRIFT,),
        "source": DecisionSource.EXTERNAL_HOST,
        "source_version": "kiro-host-v1",
        forbidden: "forbidden",
    }

    with pytest.raises(ValidationError):
        DecisionProposal.model_validate(payload)


@pytest.mark.parametrize(
    "diagnosis_ids",
    ((), ("a" * 64, "c" * 64)),
    ids=("missing", "extra"),
)
def test_proposal_validator_rejects_incomplete_or_extra_diagnosis_set(
    diagnosis_ids: tuple[str, ...],
) -> None:
    from riskprobe.agents.decision_contracts import DecisionReason

    context = _context()
    proposal = DecisionProposal(
        context_id=context.context_id,
        diagnosis_evidence_ids=diagnosis_ids,
        action_codes=(ActionCode.INVESTIGATE_FEATURE_DRIFT,),
        source=DecisionSource.EXTERNAL_HOST,
        source_version="kiro-host-v1",
    )

    result = ProposalValidator().validate(context, proposal, now=context.issued_at)

    assert result.status is DecisionStatus.REJECTED
    assert result.reason_codes == (DecisionReason.EVIDENCE_MISMATCH,)
    assert result.action_codes == ()
    assert result.diagnosis_evidence_ids == context.diagnosis_evidence_ids


def test_decision_proposal_rejects_duplicate_diagnosis_ids() -> None:
    context = _context()

    with pytest.raises(ValidationError, match="unique SHA-256"):
        DecisionProposal(
            context_id=context.context_id,
            diagnosis_evidence_ids=("a" * 64, "a" * 64),
            action_codes=(ActionCode.INVESTIGATE_FEATURE_DRIFT,),
            source=DecisionSource.EXTERNAL_HOST,
            source_version="kiro-host-v1",
        )


def test_proposal_validator_treats_exact_expiry_as_expired() -> None:
    from riskprobe.agents.decision_contracts import DecisionReason

    context = _context()
    proposal = DecisionProposal(
        context_id=context.context_id,
        diagnosis_evidence_ids=context.diagnosis_evidence_ids,
        action_codes=(ActionCode.INVESTIGATE_FEATURE_DRIFT,),
        source=DecisionSource.EXTERNAL_HOST,
        source_version="kiro-host-v1",
    )

    result = ProposalValidator().validate(context, proposal, now=context.expires_at)

    assert result.status is DecisionStatus.REJECTED
    assert result.reason_codes == (DecisionReason.CONTEXT_EXPIRED,)


def test_proposal_validator_rejects_action_not_applicable_to_findings() -> None:
    from riskprobe.agents.decision_contracts import DecisionReason

    context = _context()
    proposal = DecisionProposal(
        context_id=context.context_id,
        diagnosis_evidence_ids=context.diagnosis_evidence_ids,
        action_codes=(ActionCode.REMEDIATE_DATA_QUALITY,),
        source=DecisionSource.EXTERNAL_HOST,
        source_version="kiro-host-v1",
    )

    result = ProposalValidator().validate(context, proposal, now=context.issued_at)

    assert result.status is DecisionStatus.REJECTED
    assert result.reason_codes == (DecisionReason.ACTION_NOT_APPLICABLE,)


def test_proposal_validator_rejects_action_outside_policy_allowlist() -> None:
    from riskprobe.agents.decision_contracts import DecisionPolicy, DecisionReason

    policy = DecisionPolicy(
        allowed_action_codes=(ActionCode.REMEDIATE_DATA_QUALITY,),
        grade_b_allowed_action_codes=(ActionCode.REMEDIATE_DATA_QUALITY,),
        max_action_count=1,
    )
    base = _context()
    payload = base.model_dump(mode="python", exclude={"context_id"})
    payload["policy"] = policy
    context = DecisionContext.model_validate(payload)
    proposal = DecisionProposal(
        context_id=context.context_id,
        diagnosis_evidence_ids=context.diagnosis_evidence_ids,
        action_codes=(ActionCode.INVESTIGATE_FEATURE_DRIFT,),
        source=DecisionSource.EXTERNAL_HOST,
        source_version="kiro-host-v1",
    )

    result = ProposalValidator().validate(context, proposal, now=context.issued_at)

    assert result.status is DecisionStatus.REJECTED
    assert result.reason_codes == (DecisionReason.ACTION_NOT_ALLOWED,)


def test_proposal_validator_enforces_grade_b_action_subset() -> None:
    from riskprobe.agents.decision_contracts import DecisionPolicy, DecisionReason

    policy = DecisionPolicy(
        allowed_action_codes=(
            ActionCode.INVESTIGATE_FEATURE_DRIFT,
            ActionCode.REMEDIATE_DATA_QUALITY,
        ),
        grade_b_allowed_action_codes=(ActionCode.REMEDIATE_DATA_QUALITY,),
        max_action_count=2,
    )
    base = _context(metadata_grade="B")
    payload = base.model_dump(mode="python", exclude={"context_id"})
    payload["policy"] = policy
    context = DecisionContext.model_validate(payload)
    proposal = DecisionProposal(
        context_id=context.context_id,
        diagnosis_evidence_ids=context.diagnosis_evidence_ids,
        action_codes=(ActionCode.INVESTIGATE_FEATURE_DRIFT,),
        source=DecisionSource.EXTERNAL_HOST,
        source_version="kiro-host-v1",
    )

    result = ProposalValidator().validate(context, proposal, now=context.issued_at)

    assert result.status is DecisionStatus.REJECTED
    assert result.reason_codes == (DecisionReason.GRADE_B_ACTION_NOT_ALLOWED,)


def test_decision_proposal_has_canonical_ordering_and_rejects_hash_tamper() -> None:
    context = _context()
    unordered = DecisionProposal(
        context_id=context.context_id,
        diagnosis_evidence_ids=("c" * 64, "a" * 64),
        action_codes=(
            ActionCode.REMEDIATE_DATA_QUALITY,
            ActionCode.INVESTIGATE_FEATURE_DRIFT,
        ),
        source=DecisionSource.EXTERNAL_HOST,
        source_version="kiro-host-v1",
    )
    canonical = DecisionProposal(
        context_id=context.context_id,
        diagnosis_evidence_ids=("a" * 64, "c" * 64),
        action_codes=(
            ActionCode.INVESTIGATE_FEATURE_DRIFT,
            ActionCode.REMEDIATE_DATA_QUALITY,
        ),
        source=DecisionSource.EXTERNAL_HOST,
        source_version="kiro-host-v1",
    )

    assert unordered == canonical
    assert unordered.proposal_id == canonical.proposal_id
    tampered = unordered.model_dump(mode="python")
    tampered["proposal_id"] = "0" * 64
    with pytest.raises(ValidationError, match="canonical payload"):
        DecisionProposal.model_validate(tampered)


def test_controlled_decision_and_policy_symbols_are_publicly_exported() -> None:
    import riskprobe.agents as agents_api
    import riskprobe.recommendations as recommendations_api
    from riskprobe.agents.decision_contracts import (
        DecisionPolicy,
        DecisionReason,
        DecisionResult,
    )
    from riskprobe.recommendations.policy import (
        ACTION_TEMPLATE_BY_FINDING_KIND_V1,
        ALL_ACTION_CODES,
        RECOMMENDATION_POLICY_VERSION,
        applicable_action_codes,
    )

    expected_agent_symbols = {
        "DecisionContext": DecisionContext,
        "DecisionFinding": DecisionFinding,
        "DecisionPolicy": DecisionPolicy,
        "DecisionProposal": DecisionProposal,
        "DecisionReason": DecisionReason,
        "DecisionResult": DecisionResult,
        "DecisionSource": DecisionSource,
        "DecisionStatus": DecisionStatus,
        "ProposalValidator": ProposalValidator,
        "default_decision_policy": default_decision_policy,
    }
    for name, symbol in expected_agent_symbols.items():
        assert getattr(agents_api, name) is symbol

    expected_recommendation_symbols = {
        "ACTION_TEMPLATE_BY_FINDING_KIND_V1": ACTION_TEMPLATE_BY_FINDING_KIND_V1,
        "ALL_ACTION_CODES": ALL_ACTION_CODES,
        "ActionCode": ActionCode,
        "RECOMMENDATION_POLICY_VERSION": RECOMMENDATION_POLICY_VERSION,
        "applicable_action_codes": applicable_action_codes,
    }
    for name, symbol in expected_recommendation_symbols.items():
        assert getattr(recommendations_api, name) is symbol
