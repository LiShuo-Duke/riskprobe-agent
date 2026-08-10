import inspect
import json
from types import SimpleNamespace

import pytest
import yaml

from riskprobe.artifacts import RunStore
from riskprobe.config import PrivacyConfig
from riskprobe.mcp_server import RiskProbeTools
from riskprobe.models import Condition, EvidenceCard, RiskRule, RuleMetrics, SliceMetrics
from riskprobe.privacy import stable_token
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


def test_tool_class_exposes_the_fixed_ten_operations(tmp_path, synthetic_config) -> None:
    tools = make_tools(tmp_path, synthetic_config)
    assert all(
        callable(getattr(tools, name))
        for name in (
            "inspect_dataset",
            "register_local_dataset",
            "register_local_parquet",
            "inspect_local_parquet_schema",
            "preview_local_parquet_features",
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


def test_discover_rules_accepts_omitted_and_null_constraints(tmp_path, synthetic_config) -> None:
    for constraints in (None, {}):
        case_dir = tmp_path / ("null" if constraints is None else "empty")
        case_dir.mkdir()
        tools = make_tools(case_dir, synthetic_config)
        tools.inspect_dataset("synthetic_demo")
        if constraints is None:
            payload = tools.discover_rules("synthetic_demo", "risk")
        else:
            payload = tools.discover_rules("synthetic_demo", "risk", constraints)
        assert set(payload) == {
            "dataset_id", "objective", "rule_count", "rules", "discovery_report"
        }
        assert isinstance(payload["rule_count"], int)


def test_discover_rules_exposes_optional_constraints_contract() -> None:
    import riskprobe.mcp_server as server

    for function in (server.discover_rules, server.RiskProbeTools.discover_rules):
        parameters = inspect.signature(function).parameters
        assert parameters["objective"].default is inspect.Parameter.empty
        assert parameters["constraints"].default is None


def test_validate_rules_accepts_omitted_and_null_split_config(tmp_path, synthetic_config) -> None:
    for split_config in (None, {}):
        case_dir = tmp_path / ("null" if split_config is None else "empty")
        case_dir.mkdir()
        tools = make_tools(case_dir, synthetic_config)
        tools.inspect_dataset("synthetic_demo")
        tools.discover_rules("synthetic_demo", "risk", {})
        if split_config is None:
            payload = tools.validate_rules("synthetic_demo", [])
        else:
            payload = tools.validate_rules("synthetic_demo", [], split_config)
        assert set(payload) == {
            "dataset_id", "run_id", "reference_run_id", "evidence_card_count",
            "retry_count", "grade_counts", "validation_report",
        }
        assert isinstance(payload["evidence_card_count"], int)
        assert payload["reference_run_id"].startswith("tok_")


def test_validate_report_shows_real_institution_name_by_default_and_hides_when_disabled(
    tmp_path, synthetic_config
) -> None:
    def card() -> EvidenceCard:
        metrics = RuleMetrics(
            support_count=40,
            coverage=0.2,
            base_bad_rate=0.2,
            hit_bad_rate=0.4,
            non_hit_bad_rate=0.1,
            lift=2.0,
            precision=0.4,
            recall=0.2,
            p_value=0.01,
        )
        return EvidenceCard(
            rule=RiskRule(
                rule_id="rule-a",
                conditions=(Condition(feature="order_cnt_7d", operator=">", value=1.0),),
                origin="test",
            ),
            train=metrics,
            test=metrics,
            slices=(
                SliceMetrics(
                    slice_type="segment",
                    slice_value="bank_north",
                    metrics=metrics,
                ),
            ),
            lift_ci=(1.2, 2.8),
            adjusted_p_value=0.01,
            segment_consistency=1.0,
            max_time_decay=0.0,
            grade="Stable",
        )

    def run_validation(tools: RiskProbeTools, run_dir, *, include_name: bool):
        run_dir.mkdir()
        (run_dir / "evidence_cards.json").write_text(
            json.dumps([card().model_dump(mode="json")]), encoding="utf-8"
        )
        report = {
            "institution_analysis": {
                "institution_reports": [
                    {
                        "institution_token": stable_token(
                            "bank_north", namespace="institution"
                        ),
                        **({"institution_name": "bank_north"} if include_name else {}),
                        "status": "completed",
                    }
                ]
            }
        }
        (run_dir / "metadata_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        tools._discovered["synthetic_demo"] = ()
        tools._monitoring_snapshot = lambda _dataset_id: (
            SimpleNamespace(run_id="0123456789abcdef", run_dir=run_dir),
            None,
        )
        return tools.validate_rules("synthetic_demo", [])

    default_dir = tmp_path / "default"
    default_dir.mkdir()
    default_tools = make_tools(default_dir, synthetic_config)
    default = run_validation(default_tools, default_dir / "run", include_name=True)
    validation_report = default["validation_report"]
    assert validation_report["top_rules"][0]["institution_results"][0]["institution_name"] == "bank_north"
    assert validation_report["institution_rule_report"]["institution_reports"][0]["institution_name"] == "bank_north"

    hidden_config = synthetic_config.model_copy(
        update={"privacy": PrivacyConfig(expose_segment_values=False)}
    )
    hidden_dir = tmp_path / "hidden"
    hidden_dir.mkdir()
    hidden_tools = make_tools(hidden_dir, hidden_config)
    hidden = run_validation(hidden_tools, hidden_dir / "run", include_name=True)
    assert "bank_north" not in str(hidden["validation_report"])
    assert stable_token("bank_north", namespace="institution") in str(
        hidden["validation_report"]
    )


def test_validate_reference_token_completes_the_mcp_monitoring_workflow(
    tmp_path, synthetic_config
) -> None:
    tools = make_tools(tmp_path, synthetic_config)
    tools.inspect_dataset("synthetic_demo")
    tools.discover_rules("synthetic_demo", "risk", {})

    validation = tools.validate_rules("synthetic_demo", [])
    detection = tools.detect_anomalies(
        validation["reference_run_id"], "synthetic_demo"
    )
    diagnosis = tools.diagnose_anomaly()
    report = tools.build_report(detection["run_id"], "monitoring")

    assert detection["reference_run_id"] == validation["reference_run_id"]
    assert diagnosis["run_id"] == detection["run_id"]
    assert report["report_id"] == detection["run_id"]
    assert isinstance(validation["evidence_card_count"], int)
    assert validation["evidence_card_count"] >= 0
    assert isinstance(validation["grade_counts"], dict)
    assert all(
        isinstance(grade, str) and isinstance(count, int)
        for grade, count in validation["grade_counts"].items()
    )
    assert isinstance(detection["alert_count"], int)
    assert isinstance(detection["alert_ids"], list)
    assert detection["alert_count"] == len(detection["alert_ids"])
    assert isinstance(detection["severity_counts"], dict)
    assert set(detection["severity_counts"]) == {"warning", "critical"}
    assert sum(detection["severity_counts"].values()) == detection["alert_count"]
    assert diagnosis["diagnosis_count"] == 0
    assert isinstance(diagnosis["diagnosis_count"], int)
    assert isinstance(report["available"], bool)
    assert report["available"] is True
    serialized = str((validation, detection, diagnosis, report))
    assert str(tmp_path) not in serialized
    assert ".parquet" not in serialized


def test_validate_rules_exposes_optional_split_config_contract() -> None:
    import riskprobe.mcp_server as server

    for function in (server.validate_rules, server.RiskProbeTools.validate_rules):
        parameters = inspect.signature(function).parameters
        assert parameters["split_config"].default is None


def test_diagnose_anomaly_accepts_omitted_null_and_empty_alerts(tmp_path, synthetic_config) -> None:
    for alert_ids in (None, []):
        case_dir = tmp_path / ("null" if alert_ids is None else "empty")
        case_dir.mkdir()
        tools = make_tools(case_dir, synthetic_config)
        tools._last_detected_run_id = "0123456789abcdef"
        if alert_ids is None:
            payload = tools.diagnose_anomaly()
        else:
            payload = tools.diagnose_anomaly(alert_ids)
        assert payload["diagnosis_count"] == 0
        assert payload["root_cause_count"] == 0


def test_diagnose_anomaly_exposes_optional_alert_ids_contract() -> None:
    import riskprobe.mcp_server as server

    for function in (server.diagnose_anomaly, server.RiskProbeTools.diagnose_anomaly):
        parameters = inspect.signature(function).parameters
        assert parameters["alert_ids"].default is None


def test_register_local_dataset_is_available_and_returns_aggregate_only(
    tmp_path, synthetic_config, monkeypatch
) -> None:
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(
        yaml.safe_dump(synthetic_config.model_dump(mode="json")), encoding="utf-8"
    )
    tools = make_tools(tmp_path, synthetic_config)
    monkeypatch.setenv("RISKPROBE_ALLOWED_DATA_ROOTS", str(tmp_path))

    payload = tools.register_local_dataset("local_demo", str(config_path))

    assert payload["registered"] is True
    assert payload["read_only"] is True
    assert "config_path" not in payload
    assert str(tmp_path) not in str(payload)
    assert tools.registry.get_config("local_demo").dataset.read_only is True


def test_register_local_dataset_rejects_missing_allowed_roots(
    tmp_path, synthetic_config, monkeypatch
) -> None:
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(
        yaml.safe_dump(synthetic_config.model_dump(mode="json")), encoding="utf-8"
    )
    tools = make_tools(tmp_path, synthetic_config)
    monkeypatch.delenv("RISKPROBE_ALLOWED_DATA_ROOTS", raising=False)

    with pytest.raises(ValueError, match="allowed local data roots"):
        tools.register_local_dataset("local_demo", str(config_path))


def test_register_local_dataset_exposes_config_path_parameter() -> None:
    import inspect
    import riskprobe.mcp_server as server

    assert list(inspect.signature(server.register_local_dataset).parameters) == [
        "dataset_id", "config_path"
    ]
    assert list(inspect.signature(server.RiskProbeTools.register_local_dataset).parameters)[1:] == [
        "dataset_id", "config_path"
    ]


def test_module_mcp_tools_are_session_scoped(monkeypatch) -> None:
    import riskprobe.mcp_server as server

    original = server._SERVER_TOOLS
    monkeypatch.setenv("RISKPROBE_REGISTRY", "configs/datasets.example.yaml")
    server._SERVER_TOOLS = None
    try:
        assert server.get_tools() is server.get_tools()
    finally:
        server._SERVER_TOOLS = original


def test_register_local_parquet_is_available_and_disables_time_validation_without_snapshot(
    tmp_path, synthetic_config, monkeypatch
) -> None:
    tools = make_tools(tmp_path, synthetic_config)
    monkeypatch.setenv("RISKPROBE_ALLOWED_DATA_ROOTS", str(tmp_path))
    tools.preview_local_parquet_features(
        str(synthetic_config.dataset.path),
        entity_column="entity_id",
        target_column="target",
        segment_column="institution",
        snapshot_column=None,
    )

    payload = tools.register_local_parquet(
        "local_parquet",
        str(synthetic_config.dataset.path),
        entity_column="entity_id",
        target_column="target",
        segment_column="institution",
        snapshot_column=None,
        feature_columns=["order_cnt_7d", "order_amount_30d"],
    )

    assert payload["dataset_id"].startswith("tok_")
    assert payload["registered"] is True
    assert payload["read_only"] is True
    assert payload["time_validation_enabled"] is False
    assert set(payload) == {"dataset_id", "registered", "read_only", "time_validation_enabled"}
    config = tools.registry.get_config("local_parquet")
    assert config.columns.snapshot == "entity_id"
    assert config.time_validation_enabled is False
    assert config.features.exact_columns == (
        "order_cnt_7d",
        "order_amount_30d",
    )


def test_register_local_parquet_rejects_invalid_features_without_registration(
    tmp_path, synthetic_config, monkeypatch
) -> None:
    tools = make_tools(tmp_path, synthetic_config)
    monkeypatch.setenv("RISKPROBE_ALLOWED_DATA_ROOTS", str(tmp_path))

    with pytest.raises(ValueError, match="feature columns"):
        tools.register_local_parquet(
            "invalid_local_parquet",
            str(synthetic_config.dataset.path),
            entity_column="entity_id",
            target_column="target",
            segment_column="institution",
            snapshot_column=None,
            feature_columns=["missing_feature"],
        )

    with pytest.raises(DatasetNotRegisteredError):
        tools.registry.get_config("invalid_local_parquet")


def test_register_local_parquet_exposes_explicit_role_parameters() -> None:
    import inspect
    import riskprobe.mcp_server as server

    expected = [
        "dataset_id",
        "parquet_path",
        "entity_column",
        "target_column",
        "segment_column",
        "snapshot_column",
        "feature_columns",
    ]
    assert list(inspect.signature(server.register_local_parquet).parameters) == expected
    assert list(inspect.signature(server.RiskProbeTools.register_local_parquet).parameters)[1:] == expected


def test_local_parquet_schema_preview_returns_columns_without_path(
    tmp_path, synthetic_config, monkeypatch
) -> None:
    tools = make_tools(tmp_path, synthetic_config)
    monkeypatch.setenv("RISKPROBE_ALLOWED_DATA_ROOTS", str(tmp_path))

    payload = tools.inspect_local_parquet_schema(str(synthetic_config.dataset.path))

    assert "parquet_path" not in payload
    assert str(tmp_path) not in str(payload)
    assert {column["name"] for column in payload["columns"]} >= {
        "entity_id",
        "snapshot_date",
        "institution",
        "target",
        "order_cnt_7d",
    }
    assert all(set(column) == {"name", "dtype"} for column in payload["columns"])


def test_preview_local_parquet_features_returns_candidates_after_roles(
    tmp_path, synthetic_config, monkeypatch
) -> None:
    tools = make_tools(tmp_path, synthetic_config)
    monkeypatch.setenv("RISKPROBE_ALLOWED_DATA_ROOTS", str(tmp_path))

    payload = tools.preview_local_parquet_features(
        str(synthetic_config.dataset.path),
        entity_column="entity_id",
        target_column="target",
        segment_column="institution",
        snapshot_column=None,
    )

    assert payload["candidate_feature_columns"]
    assert set(payload["excluded_role_columns"]) == {
        "entity_id", "target", "institution"
    }
    assert set(payload["candidate_feature_columns"]).isdisjoint(
        set(payload["excluded_role_columns"])
    )
    assert "order_cnt_7d" in payload["candidate_feature_columns"]
    assert str(tmp_path) not in str(payload)


@pytest.mark.parametrize(
    "roles",
    [
        {"entity_column": "missing_entity"},
        {"target_column": "institution"},
        {"segment_column": "entity_id"},
    ],
)
def test_preview_local_parquet_features_rejects_invalid_roles(
    tmp_path, synthetic_config, monkeypatch, roles
) -> None:
    tools = make_tools(tmp_path, synthetic_config)
    monkeypatch.setenv("RISKPROBE_ALLOWED_DATA_ROOTS", str(tmp_path))
    values = {
        "entity_column": "entity_id",
        "target_column": "target",
        "segment_column": "institution",
        "snapshot_column": None,
    }
    values.update(roles)

    with pytest.raises(ValueError, match="role"):
        tools.preview_local_parquet_features(str(synthetic_config.dataset.path), **values)


def test_local_parquet_schema_tools_are_exposed() -> None:
    import inspect
    import riskprobe.mcp_server as server

    assert list(inspect.signature(server.inspect_local_parquet_schema).parameters) == [
        "parquet_path"
    ]
    assert list(inspect.signature(server.preview_local_parquet_features).parameters) == [
        "parquet_path",
        "entity_column",
        "target_column",
        "segment_column",
        "snapshot_column",
    ]


def test_register_local_parquet_uses_the_latest_role_preview_as_feature_boundary(
    tmp_path, synthetic_config, monkeypatch
) -> None:
    tools = make_tools(tmp_path, synthetic_config)
    monkeypatch.setenv("RISKPROBE_ALLOWED_DATA_ROOTS", str(tmp_path))

    tools.preview_local_parquet_features(
        str(synthetic_config.dataset.path),
        entity_column="entity_id",
        target_column="order_cnt_7d",
        segment_column="institution",
        snapshot_column=None,
    )

    with pytest.raises(ValueError, match="confirmed preview"):
        tools.register_local_parquet(
            "preview_boundary",
            str(synthetic_config.dataset.path),
            entity_column="entity_id",
            target_column="order_cnt_7d",
            segment_column="institution",
            snapshot_column=None,
            feature_columns=["order_cnt_7d"],
        )

    with pytest.raises(DatasetNotRegisteredError):
        tools.registry.get_config("preview_boundary")


def test_mcp_returns_discovery_validation_monitoring_and_diagnosis_reports(
    tmp_path, synthetic_config
) -> None:
    tools = make_tools(tmp_path, synthetic_config)
    tools.inspect_dataset("synthetic_demo")

    discovered = tools.discover_rules("synthetic_demo", "risk", {})
    discovery_report = discovered["discovery_report"]
    assert discovery_report["candidate_rule_count"] == discovered["rule_count"]
    assert set(discovery_report) >= {
        "candidate_rule_count",
        "single_rule_count",
        "two_condition_rule_count",
        "top_rules",
        "top_two_condition_rules",
    }
    assert len(discovery_report["top_rules"]) <= 5

    validated = tools.validate_rules("synthetic_demo", [])
    validation_report = validated["validation_report"]
    assert set(validation_report) >= {
        "grade_counts",
        "top_rules",
        "top_two_condition_rules",
        "stable_top_rules",
    }
    assert len(validation_report["top_rules"]) <= 5
    assert all("train" in item and "test" in item for item in validation_report["top_rules"])

    detected = tools.detect_anomalies(
        validated["reference_run_id"], "synthetic_demo"
    )
    monitoring_report = detected["monitoring_report"]
    assert set(monitoring_report) >= {"alert_counts", "alerts", "overview"}
    assert set(monitoring_report["alert_counts"]) == {
        "schema", "missingness", "distribution", "population", "label", "rule_decay"
    }

    diagnosed = tools.diagnose_anomaly(detected["alert_ids"])
    diagnosis_report = diagnosed["diagnosis_report"]
    assert set(diagnosis_report) >= {"alert_count", "diagnoses", "empty"}
    assert diagnosis_report["empty"] is True



def test_validation_and_monitoring_reports_separate_institution_layer(
    tmp_path, synthetic_config
) -> None:
    tools = make_tools(tmp_path, synthetic_config)
    tools.inspect_dataset("synthetic_demo")
    tools.discover_rules("synthetic_demo", "risk", {})

    validation = tools.validate_rules("synthetic_demo", [])
    validation_report = validation["validation_report"]
    assert "institution_summary" in validation_report
    assert "institution_rule_report" in validation_report
    assert "interpretation" in validation_report["institution_summary"]

    detection = tools.detect_anomalies(validation["reference_run_id"], "synthetic_demo")
    monitoring_report = detection["monitoring_report"]
    assert "global_alerts" in monitoring_report
    assert "institution_alerts" in monitoring_report
    assert "interpretation" in monitoring_report


def test_register_local_parquet_requires_preview_and_explicit_features(
    tmp_path, synthetic_config, monkeypatch
) -> None:
    tools = make_tools(tmp_path, synthetic_config)
    monkeypatch.setenv("RISKPROBE_ALLOWED_DATA_ROOTS", str(tmp_path))

    with pytest.raises(ValueError, match="confirmed preview"):
        tools.register_local_parquet(
            "without_preview",
            str(synthetic_config.dataset.path),
            entity_column="entity_id",
            target_column="target",
            segment_column="institution",
            snapshot_column=None,
            feature_columns=["order_cnt_7d"],
        )
    with pytest.raises(ValueError, match="explicit non-empty list"):
        tools.register_local_parquet(
            "without_features",
            str(synthetic_config.dataset.path),
            entity_column="entity_id",
            target_column="target",
            segment_column="institution",
            snapshot_column=None,
            feature_columns=None,
        )
