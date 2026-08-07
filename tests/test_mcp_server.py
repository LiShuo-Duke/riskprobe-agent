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
