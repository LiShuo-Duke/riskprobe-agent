"""RiskProbe stdio MCP server for bounded Host decisions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from mcp.server import MCPServer

from riskprobe.agents.decision_contracts import DecisionProposal, DecisionSource
from riskprobe.agents.decision_providers import (
    DecisionProviderConfig,
    DecisionProviderMode,
)
from riskprobe.config import ProjectConfig
from riskprobe.host_decision import (
    HostDecisionContext,
    HostDecisionCoordinator,
    HostDecisionError,
    HostDecisionFailure,
    HostDecisionNoActionOutcome,
    _LegacyHostDecisionNoActionOutcome,
)
from riskprobe.policy import Budget, Principal, Role
from riskprobe.recommendations.policy import ActionCode
from riskprobe.service import RiskProbeService
from riskprobe.terminal_reports import TerminalReportError


def create_mcp_server(
    *,
    service: RiskProbeService,
    coordinator: HostDecisionCoordinator,
    dataset_id: str,
    principal: Principal,
    max_queries: int,
) -> MCPServer:
    """Expose only the two high-level operations in the Host decision protocol."""

    if type(service) is not RiskProbeService:
        raise TypeError("service must be a RiskProbeService")
    if type(coordinator) is not HostDecisionCoordinator:
        raise TypeError("coordinator must be a HostDecisionCoordinator")
    if coordinator._store.directory is None:
        raise ValueError("coordinator state_dir is required")
    if coordinator._store.directory.resolve() != service.state_dir.expanduser().resolve():
        raise ValueError("service and coordinator state_dir must match")
    if type(principal) is not Principal:
        raise TypeError("principal must be a Principal")
    if not isinstance(dataset_id, str):
        raise TypeError("dataset_id must be a string")
    Budget(max_queries=max_queries)

    server = MCPServer(
        "RiskProbe",
        version="0.1.0",
        instructions=(
            "Request bounded aggregate decision context, choose only applicable action codes, "
            "then submit one proposal with the exact context and diagnosis evidence IDs."
        ),
    )

    def _report_failure(idempotency_key: str) -> HostDecisionFailure | None:
        try:
            subject = coordinator.report_subject(idempotency_key=idempotency_key)
        except Exception:
            return HostDecisionFailure(error_code="session_state_unavailable")
        if subject is None:
            return None
        try:
            service.ensure_terminal_report(subject)
        except TerminalReportError:
            return HostDecisionFailure(error_code="agent_orchestration_failed")
        except Exception:
            return HostDecisionFailure(error_code="agent_orchestration_failed")
        return None

    @server.tool()
    def riskprobe_get_decision_context(
        idempotency_key: str,
    ) -> (
        HostDecisionContext
        | HostDecisionNoActionOutcome
        | _LegacyHostDecisionNoActionOutcome
        | HostDecisionFailure
    ):
        """Run the fixed pipeline against the startup configuration."""

        result = coordinator.get_context(
            idempotency_key=idempotency_key,
            runner=lambda: service.orchestrate(
                dataset_id=dataset_id,
                principal=principal,
                budget=Budget(max_queries=max_queries),
            ),
        )
        report_failure = _report_failure(idempotency_key)
        return report_failure if report_failure is not None else result

    @server.tool()
    def riskprobe_submit_decision_proposal(
        idempotency_key: str,
        context_id: str,
        diagnosis_evidence_ids: list[str],
        action_codes: list[str],
    ) -> dict[str, object]:
        """Submit one bounded Host proposal and return the validated terminal result."""

        failure = coordinator.get_failure(idempotency_key=idempotency_key)
        if failure is not None:
            report_failure = _report_failure(idempotency_key)
            return (report_failure or failure).model_dump(mode="json")
        try:
            proposal = DecisionProposal(
                context_id=context_id,
                diagnosis_evidence_ids=tuple(diagnosis_evidence_ids),
                action_codes=tuple(ActionCode(code) for code in action_codes),
                source=DecisionSource.EXTERNAL_HOST,
                source_version=coordinator.version,
            )
        except Exception as error:
            raise HostDecisionError("host decision is unavailable") from error
        try:
            outcome = coordinator.submit_proposal(
                idempotency_key=idempotency_key,
                proposal=proposal,
            )
        except HostDecisionError:
            failure = coordinator.get_failure(idempotency_key=idempotency_key)
            if failure is None:
                return HostDecisionFailure(
                    error_code="session_state_unavailable"
                ).model_dump(mode="json")
            report_failure = _report_failure(idempotency_key)
            return (report_failure or failure).model_dump(mode="json")
        report_failure = _report_failure(idempotency_key)
        return (report_failure or outcome).model_dump(mode="json")

    return server


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="riskprobe-mcp",
        description="Run the local RiskProbe Host-decision MCP server over stdio.",
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--provider-id", default="kiro")
    parser.add_argument("--provider-version", default="gpt5.6sol")
    parser.add_argument("--principal-id", default="kiro-host")
    parser.add_argument(
        "--role",
        choices=tuple(role.value for role in Role),
        default=Role.ANALYST.value,
    )
    parser.add_argument("--max-queries", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Build the fixed local runtime and serve MCP on stdin/stdout."""

    args = _parser().parse_args(argv)
    config = ProjectConfig.from_yaml(args.config)
    coordinator = HostDecisionCoordinator(
        provider_id=args.provider_id,
        version=args.provider_version,
        state_dir=args.state_dir,
    )
    service = RiskProbeService(
        config=config,
        runs_dir=args.runs_dir,
        state_dir=args.state_dir,
        decision_provider_config=DecisionProviderConfig(
            mode=DecisionProviderMode.EXTERNAL_HOST,
            provider_id=coordinator.provider_id,
            provider_version=coordinator.version,
        ),
        decision_provider=coordinator,
    )
    server = create_mcp_server(
        service=service,
        coordinator=coordinator,
        dataset_id=config.dataset.id,
        principal=Principal(
            principal_id=args.principal_id,
            role=Role(args.role),
        ),
        max_queries=args.max_queries,
    )
    server.run()


if __name__ == "__main__":
    main()


__all__ = ["create_mcp_server", "main"]
