import json
from pathlib import Path


def test_agent_exposes_only_riskprobe_decision_tools() -> None:
    config = json.loads(Path(".kiro/agents/riskprobe.json").read_text())
    assert config["tools"] == ["@riskprobe"]
    rules = config["permissions"]["rules"]
    allowed_mcp = {
        match
        for rule in rules
        if rule["capability"] == "mcp" and rule["effect"] == "allow"
        for match in rule["match"]
    }
    assert allowed_mcp == {
        "riskprobe/riskprobe_get_decision_context",
        "riskprobe/riskprobe_submit_decision_proposal",
    }
    assert all(
        rule["capability"] != "fs_read" or rule["effect"] == "deny"
        for rule in rules
    )
    for capability in {"shell", "fs_write", "web_fetch", "web_search"}:
        assert any(
            rule["capability"] == capability and rule["effect"] == "deny"
            for rule in rules
        )


def test_workspace_mcp_starts_only_the_two_phase_local_server() -> None:
    config = json.loads(Path(".kiro/settings/mcp.json").read_text())
    server = config["mcpServers"]["riskprobe"]
    assert set(config["mcpServers"]) == {"riskprobe"}
    assert server["command"] == ".venv/bin/python"
    assert server["args"] == [
        "-m",
        "riskprobe.mcp_server",
        "--config",
        "configs/synthetic.example.yaml",
        "--runs-dir",
        "runs",
        "--state-dir",
        "runs/state",
        "--provider-id",
        "kiro",
        "--provider-version",
        "gpt5.6sol",
        "--principal-id",
        "kiro-host",
        "--role",
        "analyst",
        "--max-queries",
        "16",
    ]
    assert "env" not in server
    assert "/Users/" not in json.dumps(server)


def test_agent_skill_requires_exact_two_phase_host_decision_flow() -> None:
    skill = Path(".kiro/skills/riskprobe/SKILL.md").read_text()
    context_tool = "riskprobe_get_decision_context"
    proposal_tool = "riskprobe_submit_decision_proposal"

    assert skill.index(context_tool) < skill.index(proposal_tool)
    assert "inspect → diagnose → discover → recommend → review" in skill
    assert "policy.allowed_action_codes" in skill
    assert "policy.grade_b_allowed_action_codes" in skill
    assert "context_id" in skill
    assert "diagnosis_evidence_ids" in skill
    assert "idempotency_key" in skill
    for legacy_tool in (
        "register_local_dataset",
        "register_local_parquet",
        "inspect_local_parquet_schema",
        "preview_local_parquet_features",
        "inspect_dataset",
        "discover_rules",
        "validate_rules",
        "detect_anomalies",
        "diagnose_anomaly",
        "build_report",
    ):
        assert legacy_tool not in skill
