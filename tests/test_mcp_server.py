import asyncio
import json
from pathlib import Path

from mcp import Client

from riskprobe.agents.decision_contracts import DecisionContext
from riskprobe.agents.decision_providers import (
    DecisionProviderConfig,
    DecisionProviderMode,
    DeterministicDecisionProvider,
)
from riskprobe.host_decision import HostDecisionCoordinator
from riskprobe.mcp_server import create_mcp_server
from riskprobe.policy import Principal, Role
from riskprobe.service import RiskProbeService


def _server(
    *,
    tmp_path: Path,
    synthetic_config: object,
) -> tuple[object, Path]:
    coordinator = HostDecisionCoordinator(provider_id="kiro", version="gpt5.6sol")
    runs_dir = tmp_path / "runs"
    service = RiskProbeService(
        config=synthetic_config,
        runs_dir=runs_dir,
        state_dir=tmp_path / "state",
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
        assert set(
            schemas["riskprobe_get_decision_context"]["properties"]
        ) == {"idempotency_key"}
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
            context_payload = pending.structured_content["context"]
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
    }
