from pathlib import Path

from riskprobe.agents.decision_contracts import (
    DecisionProposal,
    DecisionSource,
)
from riskprobe.agents.decision_providers import (
    DecisionProviderConfig,
    DecisionProviderMode,
    DeterministicDecisionProvider,
)
from riskprobe.host_decision import HostDecisionCoordinator
from riskprobe.policy import Budget, Principal, Role
from riskprobe.service import RiskProbeService


def test_host_decision_pauses_for_context_then_resumes_fixed_flow(
    tmp_path: Path,
    synthetic_config: object,
) -> None:
    coordinator = HostDecisionCoordinator(
        provider_id="kiro",
        version="gpt5.6sol",
    )
    runs_dir = tmp_path / "runs"
    state_dir = tmp_path / "state"
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
