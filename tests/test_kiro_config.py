import json
from pathlib import Path


def test_agent_exposes_only_riskprobe_and_denies_builtin_capabilities() -> None:
    config = json.loads(Path(".kiro/agents/riskprobe.json").read_text())
    assert config["tools"] == ["@riskprobe"]
    rules = {(rule["capability"], rule["effect"]) for rule in config["permissions"]["rules"]}
    assert ("mcp", "allow") in rules
    for capability in {"shell", "fs_read", "fs_write", "web_fetch", "web_search"}:
        assert (capability, "deny") in rules


def test_workspace_mcp_starts_only_the_local_riskprobe_server() -> None:
    config = json.loads(Path(".kiro/settings/mcp.json").read_text())
    assert config == {
        "mcpServers": {
            "riskprobe": {
                "command": ".venv/bin/python",
                "args": ["-m", "riskprobe.mcp_server"],
            }
        }
    }
