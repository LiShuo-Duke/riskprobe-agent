import yaml
import pytest

from riskprobe.artifacts import RunStore
from riskprobe.mcp_server import RiskProbeTools
from riskprobe.registry import DatasetNotRegisteredError, DatasetRegistry


def make_tools(tmp_path, synthetic_config) -> RiskProbeTools:
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(yaml.safe_dump(synthetic_config.model_dump(mode="json")), encoding="utf-8")
    registry_path = tmp_path / "datasets.yaml"
    registry_path.write_text(
        yaml.safe_dump({"datasets": {"synthetic_demo": {"config": str(config_path)}}}),
        encoding="utf-8",
    )
    return RiskProbeTools(DatasetRegistry.from_yaml(registry_path), RunStore(tmp_path / "runs"))


def test_tools_accept_dataset_id_but_reject_path(tmp_path, synthetic_config) -> None:
    tools = make_tools(tmp_path, synthetic_config)
    with pytest.raises(DatasetNotRegisteredError):
        tools.inspect_dataset("/tmp/private.parquet")


def test_inspect_output_has_no_entity_or_path(tmp_path, synthetic_config) -> None:
    tools = make_tools(tmp_path, synthetic_config)
    payload = tools.inspect_dataset("synthetic_demo")
    serialized = str(payload)
    assert "entity_id" not in serialized
    assert ".parquet" not in serialized


def test_tool_class_exposes_the_fixed_six_operations(tmp_path, synthetic_config) -> None:
    tools = make_tools(tmp_path, synthetic_config)
    assert all(
        callable(getattr(tools, name))
        for name in (
            "inspect_dataset",
            "discover_rules",
            "validate_rules",
            "detect_anomalies",
            "diagnose_anomaly",
            "build_report",
        )
    )


def test_mcp_payloads_tokenize_values_and_omit_rule_details(tmp_path, synthetic_config) -> None:
    tools = make_tools(tmp_path, synthetic_config)
    payload = tools.inspect_dataset("synthetic_demo")
    serialized = str(payload)
    assert "synthetic_demo" not in serialized
    assert "institution" not in serialized
    assert "private-segment" not in serialized
    assert ".parquet" not in serialized

    import inspect

    assert list(inspect.signature(tools.discover_rules).parameters) == [
        "dataset_id", "objective", "constraints"
    ]
    assert list(inspect.signature(tools.validate_rules).parameters) == [
        "dataset_id", "rule_ids", "split_config"
    ]
    assert list(inspect.signature(tools.diagnose_anomaly).parameters) == ["alert_ids"]
    assert list(inspect.signature(tools.build_report).parameters) == ["run_id", "report_type"]




def test_mcp_rejects_constraints_and_split_config_it_cannot_apply(
    tmp_path, synthetic_config
) -> None:
    tools = make_tools(tmp_path, synthetic_config)
    tools.inspect_dataset("synthetic_demo")

    with pytest.raises(ValueError, match="constraints"):
        tools.discover_rules("synthetic_demo", "risk", {"max_rules": 1})

    tools.discover_rules("synthetic_demo", "risk", {})
    with pytest.raises(ValueError, match="split_config"):
        tools.validate_rules("synthetic_demo", [], {"time_validation": False})


def test_mcp_rejects_unknown_objective_instead_of_ignoring_it(tmp_path, synthetic_config) -> None:
    tools = make_tools(tmp_path, synthetic_config)
    tools.inspect_dataset("synthetic_demo")

    with pytest.raises(ValueError, match="objective"):
        tools.discover_rules("synthetic_demo", "other-objective", {})
    tools = make_tools(tmp_path, synthetic_config)
    with pytest.raises(ValueError, match="inspect"):
        tools.discover_rules("synthetic_demo", "risk", {})


def test_module_mcp_tools_are_session_scoped(monkeypatch) -> None:
    import riskprobe.mcp_server as server

    original = server._SERVER_TOOLS
    monkeypatch.setenv("RISKPROBE_REGISTRY", "configs/datasets.example.yaml")
    server._SERVER_TOOLS = None
    try:
        assert server.get_tools() is server.get_tools()
    finally:
        server._SERVER_TOOLS = original
