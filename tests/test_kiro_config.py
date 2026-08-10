import json
from pathlib import Path


def test_agent_exposes_riskprobe_and_only_allowlisted_parquet_reads() -> None:
    config = json.loads(Path(".kiro/agents/riskprobe.json").read_text())
    assert config["tools"] == ["@riskprobe"]
    rules = config["permissions"]["rules"]
    assert {rule["capability"] for rule in rules if rule["effect"] == "allow"} >= {
        "mcp",
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


def test_workspace_mcp_starts_only_the_local_riskprobe_server() -> None:
    config = json.loads(Path(".kiro/settings/mcp.json").read_text())
    server = config["mcpServers"]["riskprobe"]
    assert set(config["mcpServers"]) == {"riskprobe"}
    assert server["command"] == ".venv/bin/python"
    assert server["args"] == ["-m", "riskprobe.mcp_server"]
    assert server["env"]["RISKPROBE_REGISTRY"] == "configs/datasets.example.yaml"
    allowed_roots = server["env"]["RISKPROBE_ALLOWED_DATA_ROOTS"]
    assert allowed_roots == "${RISKPROBE_ALLOWED_DATA_ROOTS}"
    assert "/Users/" not in allowed_roots


def test_agent_skill_requires_schema_roles_preview_and_confirmation_before_registration() -> None:
    skill = Path(".kiro/skills/riskprobe/SKILL.md").read_text()
    steps = [
        "inspect_local_parquet_schema",
        "preview_local_parquet_features",
        "register_local_parquet",
    ]
    positions = [skill.index(step) for step in steps]
    assert positions == sorted(positions)
    assert "Do not call `register_local_parquet` until the user confirms" in skill
    assert "candidate numeric feature columns" in skill


def test_agent_skill_requires_three_concrete_reports_and_metric_explanations() -> None:
    skill = Path(".kiro/skills/riskprobe/SKILL.md").read_text()
    required = [
        "discovery_report",
        "Train Lift",
        "validation_report",
        "Test-Lift TOP5",
        "Global Stable",
        "Local",
        "institution_rule_report",
        "Institution TOP5",
        "monitoring_report",
        "Global Alert",
        "Institution Alert",
        "diagnosis_report",
        "root-cause TOP3",
    ]
    section = skill.split("多机构分析必须遵守以下报告顺序：", 1)[1]
    positions = [section.index(term) for term in required]
    assert positions == sorted(positions)
    for grade in ("Stable", "Local", "Unstable", "Suspicious"):
        assert grade in skill


def test_agent_skill_uses_only_supported_discovery_arguments() -> None:
    skill = Path(".kiro/skills/riskprobe/SKILL.md").read_text()

    assert "discover_rules(dataset_id, objective)" in skill
    assert "Do not pass `constraints`" in skill
    assert "non-empty `constraints`" in skill
