"""Local stdio MCP server with a fixed, aggregate-only RiskProbe tool surface."""

import json
import os
import re
from pathlib import Path
from typing import Any

import polars as pl
from mcp.server.fastmcp import FastMCP

from riskprobe.artifacts import RunStore
from riskprobe.features.catalog import FeatureCatalog
from riskprobe.explainability import (
    summarize_alerts,
    summarize_candidate_rules,
    summarize_diagnoses,
    summarize_evidence_cards,
)
from riskprobe.io.parquet import ParquetDataset
from riskprobe.models import EvidenceCard
from riskprobe.monitoring.detection import detect_anomalies as detect_alerts
from riskprobe.monitoring.diagnosis import diagnose_alerts
from riskprobe.monitoring.models import ReferenceSnapshot
from riskprobe.privacy import assert_safe_payload, redact_payload, stable_token, suppress_small_groups
from riskprobe.profiling import profile_dataset
from riskprobe.registry import (
    DatasetRegistry,
    _allowed_roots,
    _resolve_under_roots,
)
from riskprobe.service import RiskProbeService, _restore_json_tuples

_RUN_ID = re.compile(r"^[0-9a-f]{16}$")
mcp = FastMCP("riskprobe")
_SERVER_TOOLS: Any = None
_ALLOWED_ROOTS_ENV = "RISKPROBE_ALLOWED_DATA_ROOTS"


def _allowlisted_schema(parquet_path: str) -> pl.Schema:
    if not isinstance(parquet_path, str) or not parquet_path.strip():
        raise ValueError("parquet_path is required")
    roots = _allowed_roots(_allowed_data_roots())
    safe_path = _resolve_under_roots(Path(parquet_path), roots, "parquet path")
    if safe_path.suffix.lower() != ".parquet":
        raise ValueError("parquet path must have a .parquet extension")
    try:
        return ParquetDataset(safe_path).schema()
    except (OSError, ValueError, pl.exceptions.PolarsError) as error:
        raise ValueError("local dataset Parquet file is not readable") from error


def _allowed_data_roots() -> tuple[Path, ...]:
    raw = os.environ.get(_ALLOWED_ROOTS_ENV, "")
    if not raw.strip():
        raise ValueError("allowed local data roots are required")
    values = tuple(item.strip() for item in raw.split(","))
    if any(not value or not Path(value).is_absolute() for value in values):
        raise ValueError("allowed local data roots must be absolute directories")
    return tuple(Path(value) for value in values)


class RiskProbeTools:
    """Aggregate-only operations with an explicit in-process workflow state."""

    def __init__(self, registry: DatasetRegistry, store: RunStore) -> None:
        self.registry = registry
        self.store = store
        self._inspected: set[str] = set()
        self._discovered: dict[str, tuple[str, ...]] = {}
        self._discovery_results: dict[str, Any] = {}
        self._validated: dict[str, str] = {}
        self._detected: dict[str, dict[str, Any]] = {}
        self._diagnosed: set[str] = set()
        self._retry_counts: dict[str, int] = {}
        self._alert_handles: dict[str, tuple[str, str]] = {}
        self._run_handles: dict[str, str] = {}
        self._last_detected_run_id: str | None = None
        self._parquet_previews: dict[tuple[str, tuple[str, ...]], frozenset[str]] = {}

    def inspect_dataset(self, dataset_id: str) -> dict[str, Any]:
        config = self.registry.get_config(dataset_id)
        profile = profile_dataset(ParquetDataset(config.dataset.path), config)
        self._inspected.add(dataset_id)
        return self._safe(
            {
                "dataset_id": dataset_id,
                "metadata_grade": profile.metadata_grade,
                "row_count": profile.row_count,
                "feature_count": profile.feature_count,
                "positive_rate": profile.positive_rate,
                "segment_counts": suppress_small_groups(
                    ({"segment": segment, "count": count} for segment, count in profile.segment_counts.items()),
                    "count",
                    config.validation.min_group_size,
                ),
                "limitations": sorted(issue.code for issue in profile.issues),
            }
        )

    def inspect_local_parquet_schema(self, parquet_path: str) -> dict[str, Any]:
        schema = _allowlisted_schema(parquet_path)
        payload = {
            "columns": [
                {"name": name, "dtype": str(dtype)}
                for name, dtype in schema.items()
            ]
        }
        # Schema names are metadata requested for role confirmation; no data values or paths are returned.
        return payload

    def preview_local_parquet_features(
        self,
        parquet_path: str,
        entity_column: str,
        target_column: str,
        segment_column: str,
        snapshot_column: str | None = None,
    ) -> dict[str, Any]:
        values = (entity_column, target_column, segment_column)
        if any(not isinstance(column, str) or not column.strip() for column in values):
            raise ValueError("role columns must be non-empty strings")
        normalized_roles = tuple(column.strip() for column in values)
        normalized_snapshot = (
            snapshot_column.strip()
            if isinstance(snapshot_column, str) and snapshot_column.strip()
            else None
        )
        if snapshot_column is not None and normalized_snapshot is None:
            raise ValueError("snapshot column must be a non-empty string or null")
        role_columns = normalized_roles + ((normalized_snapshot,) if normalized_snapshot else ())
        if len(role_columns) != len(set(role_columns)):
            raise ValueError("role columns must be distinct")

        schema = _allowlisted_schema(parquet_path)
        schema_names = set(schema.names())
        if not set(role_columns).issubset(schema_names):
            raise ValueError("role columns must exist in the local dataset schema")
        excluded = set(role_columns)
        candidate_feature_columns = [
            name for name, dtype in schema.items()
            if name not in excluded and dtype.is_numeric()
        ]
        non_numeric_columns = [
            name for name, dtype in schema.items()
            if name not in excluded and not dtype.is_numeric()
        ]
        payload = {
            "excluded_role_columns": list(role_columns),
            "candidate_feature_columns": candidate_feature_columns,
            "non_numeric_columns": non_numeric_columns,
        }
        preview_key = (str(Path(parquet_path).resolve()), role_columns)
        self._parquet_previews[preview_key] = frozenset(candidate_feature_columns)
        # Candidate names are schema metadata for the explicit user confirmation step.
        return payload

    def register_local_dataset(self, dataset_id: str, config_path: str) -> dict[str, Any]:
        if not isinstance(config_path, str) or not config_path.strip():
            raise ValueError("config_path is required")
        self.registry = self.registry.register_local_config(
            dataset_id,
            Path(config_path),
            _allowed_data_roots(),
        )
        return self._safe(
            {
                "dataset_id": dataset_id,
                "registered": True,
                "read_only": True,
            }
        )

    def register_local_parquet(
        self,
        dataset_id: str,
        parquet_path: str,
        entity_column: str,
        target_column: str,
        segment_column: str,
        snapshot_column: str | None = None,
        feature_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(parquet_path, str) or not parquet_path.strip():
            raise ValueError("parquet_path is required")
        if not isinstance(feature_columns, list) or not feature_columns:
            raise ValueError(
                "feature columns must be an explicit non-empty list confirmed by preview"
            )
        if not all(isinstance(feature, str) and feature.strip() for feature in feature_columns):
            raise ValueError("feature columns must be non-empty strings")
        role_columns = (entity_column, target_column, segment_column, snapshot_column)
        if not all(isinstance(column, str) and column.strip() for column in role_columns[:3]):
            raise ValueError("role columns must be non-empty strings")
        normalized_roles = tuple(column.strip() for column in role_columns[:3])
        if snapshot_column is not None:
            if not isinstance(snapshot_column, str) or not snapshot_column.strip():
                raise ValueError("snapshot column must be a non-empty string or null")
            normalized_roles += (snapshot_column.strip(),)
        if len(normalized_roles) != len(set(normalized_roles)):
            raise ValueError("role columns must be distinct")
        preview_key = (str(Path(parquet_path).resolve()), normalized_roles)
        preview_candidates = self._parquet_previews.get(preview_key)
        if preview_candidates is None:
            raise ValueError("feature columns must come from a confirmed preview")
        requested_features = set(feature_columns)
        if len(requested_features) != len(feature_columns):
            raise ValueError("feature columns must be unique")
        if not requested_features.issubset(preview_candidates):
            raise ValueError("feature columns must come from the confirmed preview")
        self.registry = self.registry.register_local_parquet(
            dataset_id,
            Path(parquet_path),
            entity_column=entity_column,
            target_column=target_column,
            segment_column=segment_column,
            snapshot_column=snapshot_column,
            feature_columns=feature_columns,
            allowed_roots=_allowed_data_roots(),
        )
        config = self.registry.get_config(dataset_id)
        return self._safe(
            {
                "dataset_id": dataset_id,
                "registered": True,
                "read_only": True,
                "time_validation_enabled": config.time_validation_enabled,
            }
        )

    def discover_rules(
        self,
        dataset_id: str,
        objective: str,
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Discover risk rules using the registered project configuration.

        ``constraints`` is retained for protocol compatibility, but this
        version accepts only omitted or empty values and never applies per-call
        discovery overrides.
        """
        self._require(dataset_id in self._inspected, "inspect must complete before discover")
        if not isinstance(objective, str) or objective.strip() != "risk":
            raise ValueError("objective must be the supported 'risk' objective")
        normalized_constraints = {} if constraints is None else constraints
        if not isinstance(normalized_constraints, dict):
            raise ValueError("constraints must be an object")
        if normalized_constraints:
            raise ValueError("non-empty constraints are unsupported and must be rejected")
        service = self._service(dataset_id)
        result = service.discover_with_metrics()
        rules = list(result.rules)
        rule_ids = tuple(rule.rule_id for rule in rules)
        self._discovered[dataset_id] = rule_ids
        self._discovery_results[dataset_id] = result
        operational = self._safe(
            {
                "dataset_id": dataset_id,
                "objective": objective,
                "rule_count": len(rule_ids),
                "rules": [
                    {"rule_id": rule.rule_id, "condition_count": len(rule.conditions), "origin": rule.origin}
                    for rule in rules
                ],
            }
        )
        operational["discovery_report"] = self._safe_explainable(
            summarize_candidate_rules(result, self._confirmed_features(dataset_id))
        )
        return operational

    def validate_rules(
        self,
        dataset_id: str,
        rule_ids: list[str],
        split_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._require(dataset_id in self._discovered, "discover must complete before validate")
        normalized_split_config = {} if split_config is None else split_config
        if not isinstance(rule_ids, list) or not isinstance(normalized_split_config, dict):
            raise ValueError("rule_ids must be a list and split_config must be an object")
        if normalized_split_config:
            raise ValueError("non-empty split_config is unsupported and must be rejected")
        discovered = set(self._discovered[dataset_id])
        rule_handles = {stable_token(rule_id): rule_id for rule_id in discovered}
        resolved_rule_ids = {
            rule_handles.get(rule_id, rule_id)
            for rule_id in rule_ids
        }
        if not resolved_rule_ids.issubset(discovered):
            raise ValueError("rule_ids must come from discover")
        context, _ = self._monitoring_snapshot(dataset_id)
        self._validated[dataset_id] = context.run_id
        self._run_handles[stable_token(context.run_id)] = context.run_id
        cards = self._read_cards(context.run_dir / "evidence_cards.json")
        config = self.registry.get_config(dataset_id)
        report = summarize_evidence_cards(
            cards,
            self._confirmed_features(dataset_id),
            config.validation,
            time_validation_enabled=config.time_validation_enabled,
            expose_segment_values=config.privacy.expose_segment_values,
        )
        metadata_payload = json.loads(
            (context.run_dir / "metadata_report.json").read_text(encoding="utf-8")
        )
        institution_analysis = metadata_payload.get(
            "institution_analysis",
            {
                "analysis_mode": "global_first_conditional_local",
                "eligible_institution_count": 0,
                "triggered_institution_count": 0,
                "blocked_institution_count": 0,
                "institution_reports": [],
                "interpretation": "未发现需要机构内规则发现的 Local 机构。",
            },
        )
        report["institution_rule_report"] = self._filter_institution_names(
            institution_analysis,
            expose_segment_values=config.privacy.expose_segment_values,
        )
        operational = self._safe(
            {
                "dataset_id": dataset_id,
                "run_id": context.run_id,
                "reference_run_id": context.run_id,
                "evidence_card_count": len(cards),
                "retry_count": self._retry_counts.get(dataset_id, 0),
                "grade_counts": report["grade_counts"],
            }
        )
        operational["validation_report"] = self._safe_explainable(report)
        return operational

    def detect_anomalies(self, reference_run_id: str, current_dataset_id: str) -> dict[str, Any]:
        reference_run_id = self._run_handles.get(reference_run_id, reference_run_id)
        self._require(reference_run_id and _RUN_ID.fullmatch(reference_run_id) is not None, "reference run ID is invalid")
        self._require(current_dataset_id in self._validated, "validate must complete before detect")
        reference = self._reference_snapshot(reference_run_id)
        config = self.registry.get_config(current_dataset_id)
        current_context = self._run_dataset(current_dataset_id)
        frame, profile, catalog = self._frame_profile_catalog(config)
        cards = self._read_cards(current_context.run_dir / "evidence_cards.json")
        alerts = detect_alerts(reference, frame, cards, catalog)
        self._detected[current_context.run_id] = {
            "reference": reference, "frame": frame, "catalog": catalog, "alerts": tuple(alerts),
            "dataset_id": current_dataset_id,
            "run_dir": current_context.run_dir,
        }
        self._run_handles[stable_token(current_context.run_id)] = current_context.run_id
        self._last_detected_run_id = current_context.run_id
        alert_handles = []
        for alert in alerts:
            handle = stable_token(f"{current_context.run_id}:{alert.alert_id}", namespace="alert")
            self._alert_handles[handle] = (current_context.run_id, alert.alert_id)
            alert_handles.append(handle)
        operational = self._safe(
            {
                "reference_run_id": reference_run_id,
                "current_dataset_id": current_dataset_id,
                "run_id": current_context.run_id,
                "alert_count": len(alerts),
                "alert_ids": alert_handles,
                "severity_counts": {
                    severity: sum(alert.severity == severity for alert in alerts)
                    for severity in ("warning", "critical")
                },
                "metadata_grade": profile.metadata_grade,
                "retry_count": self._retry_counts.get(current_dataset_id, 0),
            }
        )
        operational["monitoring_report"] = self._safe_explainable(
            summarize_alerts(
                alerts,
                reference_row_count=reference.row_count,
                reference_positive_rate=reference.positive_rate,
                reference_feature_count=len(reference.features),
                current_row_count=profile.row_count,
                current_positive_rate=profile.positive_rate,
                current_feature_count=profile.feature_count,
                expose_segment_values=config.privacy.expose_segment_values,
            )
        )
        return operational

    def diagnose_anomaly(self, alert_ids: list[str] | None = None) -> dict[str, Any]:
        normalized_alert_ids = [] if alert_ids is None else alert_ids
        self._require(isinstance(normalized_alert_ids, list), "alert_ids must be a list")
        if not normalized_alert_ids:
            self._require(self._last_detected_run_id is not None, "detect must complete before diagnose")
            run_id = self._last_detected_run_id
            diagnoses = ()
        else:
            handles = set(normalized_alert_ids)
            if not handles.issubset(self._alert_handles):
                raise ValueError("alert_ids must come from detect")
            run_ids = {self._alert_handles[handle][0] for handle in handles}
            self._require(len(run_ids) == 1, "alert_ids must belong to one detect run")
            run_id = next(iter(run_ids))
            context = self._detected[run_id]
            raw_ids = {self._alert_handles[handle][1] for handle in handles}
            alerts = tuple(alert for alert in context["alerts"] if alert.alert_id in raw_ids)
            diagnoses = diagnose_alerts(
                alerts, context["reference"], context["frame"], context["catalog"], top_k=3
            )
        self._diagnosed.add(run_id)
        operational = self._safe(
            {
                "run_id": run_id,
                "diagnosis_count": len(diagnoses),
                "root_cause_count": sum(len(diagnosis.root_causes) for diagnosis in diagnoses),
            }
        )
        operational["diagnosis_report"] = self._safe_explainable(
            summarize_diagnoses(diagnoses)
        )
        return operational

    def build_report(self, run_id: str, report_type: str) -> dict[str, Any]:
        run_id = self._run_handles.get(run_id, run_id)
        self._require(run_id in self._diagnosed, "diagnose must complete before report")
        if report_type not in {"summary", "monitoring"}:
            raise ValueError("unsupported report_type")
        context = self._detected[run_id]
        report = (context["run_dir"] / "risk_report.md").read_text(encoding="utf-8")
        return self._safe(
            {
                "report_id": run_id,
                "report_type": report_type,
                "section_count": report.count("\n## "),
                "available": bool(report),
            }
        )

    def _service(self, dataset_id: str) -> RiskProbeService:
        return RiskProbeService(config=self.registry.get_config(dataset_id), runs_dir=self.store.runs_dir)

    def _monitoring_snapshot(self, dataset_id: str) -> Any:
        service = self._service(dataset_id)
        try:
            return service.monitoring_snapshot()
        except Exception:
            attempts = self._retry_counts.get(dataset_id, 0)
            if attempts >= 1:
                raise
            self._retry_counts[dataset_id] = attempts + 1
            return service.monitoring_snapshot()

    def _run_dataset(self, dataset_id: str) -> Any:
        service = self._service(dataset_id)
        try:
            return service.run()
        except Exception:
            attempts = self._retry_counts.get(dataset_id, 0)
            if attempts >= 1:
                raise
            self._retry_counts[dataset_id] = attempts + 1
            return service.run()

    def _reference_snapshot(self, run_id: str) -> ReferenceSnapshot:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("reference run ID is invalid")
        path = self.store.runs_dir / "monitoring" / run_id / "reference_snapshot.json"
        try:
            return ReferenceSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError("reference run does not contain a monitoring snapshot") from error

    @staticmethod
    def _read_cards(path: Path) -> tuple[EvidenceCard, ...]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return tuple(EvidenceCard.model_validate(_restore_json_tuples(item)) for item in payload)

    @staticmethod
    def _frame_profile_catalog(config: Any) -> tuple[Any, Any, FeatureCatalog]:
        dataset = ParquetDataset(config.dataset.path)
        profile = profile_dataset(dataset, config)
        roles = (config.columns.entity, config.columns.snapshot, config.columns.segment, config.columns.target)
        features = config.features.select_columns(dataset.schema().names(), roles)
        frame = dataset.collect([config.columns.segment, config.columns.target, *features])
        catalog = FeatureCatalog.from_columns(features, config.features.families)
        return frame, profile, catalog

    def _confirmed_features(self, dataset_id: str) -> frozenset[str]:
        config = self.registry.get_config(dataset_id)
        dataset = ParquetDataset(config.dataset.path)
        role_columns = (
            config.columns.entity,
            config.columns.snapshot,
            config.columns.segment,
            config.columns.target,
        )
        return frozenset(
            config.features.select_columns(dataset.schema().names(), role_columns)
        )

    @staticmethod
    def _filter_institution_names(
        payload: object, *, expose_segment_values: bool
    ) -> object:
        if isinstance(payload, dict):
            filtered: dict[str, Any] = {}
            for key, value in payload.items():
                if not expose_segment_values and key in {
                    "institution_name",
                    "institution_names",
                }:
                    continue
                filtered[key] = RiskProbeTools._filter_institution_names(
                    value, expose_segment_values=expose_segment_values
                )
            return filtered
        if isinstance(payload, list):
            return [
                RiskProbeTools._filter_institution_names(
                    item, expose_segment_values=expose_segment_values
                )
                for item in payload
            ]
        return payload

    @staticmethod
    def _safe_explainable(payload: dict[str, Any]) -> dict[str, Any]:
        def mask_rule_features(value: object, key: str | None = None) -> object:
            if isinstance(value, dict):
                return {
                    item_key: mask_rule_features(item_value, str(item_key))
                    for item_key, item_value in value.items()
                }
            if isinstance(value, list):
                return [mask_rule_features(item) for item in value]
            if key == "feature" and isinstance(value, str):
                return stable_token(value, namespace="feature")
            return value

        assert_safe_payload(mask_rule_features(payload))
        return payload

    @staticmethod
    def _safe(payload: dict[str, Any]) -> dict[str, Any]:
        redacted = redact_payload(payload)
        assert_safe_payload(redacted)
        return redacted

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)


def get_tools() -> RiskProbeTools:
    global _SERVER_TOOLS
    if _SERVER_TOOLS is None:
        registry_path = Path(os.environ.get("RISKPROBE_REGISTRY", "configs/datasets.example.yaml"))
        _SERVER_TOOLS = RiskProbeTools(DatasetRegistry.from_yaml(registry_path), RunStore("runs"))
    return _SERVER_TOOLS


@mcp.tool()
def register_local_dataset(dataset_id: str, config_path: str) -> dict[str, Any]:
    return get_tools().register_local_dataset(dataset_id, config_path)


@mcp.tool()
def register_local_parquet(
    dataset_id: str,
    parquet_path: str,
    entity_column: str,
    target_column: str,
    segment_column: str,
    snapshot_column: str | None = None,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    return get_tools().register_local_parquet(
        dataset_id,
        parquet_path,
        entity_column,
        target_column,
        segment_column,
        snapshot_column,
        feature_columns,
    )


@mcp.tool()
def inspect_local_parquet_schema(parquet_path: str) -> dict[str, Any]:
    return get_tools().inspect_local_parquet_schema(parquet_path)


@mcp.tool()
def preview_local_parquet_features(
    parquet_path: str,
    entity_column: str,
    target_column: str,
    segment_column: str,
    snapshot_column: str | None = None,
) -> dict[str, Any]:
    return get_tools().preview_local_parquet_features(
        parquet_path,
        entity_column,
        target_column,
        segment_column,
        snapshot_column,
    )


@mcp.tool()
def inspect_dataset(dataset_id: str) -> dict[str, Any]:
    return get_tools().inspect_dataset(dataset_id)


@mcp.tool()
def discover_rules(
    dataset_id: str,
    objective: str,
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return get_tools().discover_rules(dataset_id, objective, constraints)


@mcp.tool()
def validate_rules(
    dataset_id: str,
    rule_ids: list[str],
    split_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return get_tools().validate_rules(dataset_id, rule_ids, split_config)


@mcp.tool()
def detect_anomalies(reference_run_id: str, current_dataset_id: str) -> dict[str, Any]:
    return get_tools().detect_anomalies(reference_run_id, current_dataset_id)


@mcp.tool()
def diagnose_anomaly(alert_ids: list[str] | None = None) -> dict[str, Any]:
    return get_tools().diagnose_anomaly(alert_ids)


@mcp.tool()
def build_report(run_id: str, report_type: str) -> dict[str, Any]:
    return get_tools().build_report(run_id, report_type)


if __name__ == "__main__":
    mcp.run(transport="stdio")
