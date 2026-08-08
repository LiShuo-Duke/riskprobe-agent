"""Local stdio MCP server with a fixed, aggregate-only RiskProbe tool surface."""

import json
import os
import re
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from riskprobe.artifacts import RunStore
from riskprobe.features.catalog import FeatureCatalog
from riskprobe.io.parquet import ParquetDataset
from riskprobe.models import EvidenceCard
from riskprobe.monitoring.detection import detect_anomalies as detect_alerts
from riskprobe.monitoring.diagnosis import diagnose_alerts
from riskprobe.monitoring.models import ReferenceSnapshot
from riskprobe.privacy import assert_safe_payload, redact_payload, stable_token, suppress_small_groups
from riskprobe.profiling import profile_dataset
from riskprobe.registry import DatasetRegistry
from riskprobe.service import RiskProbeService

_RUN_ID = re.compile(r"^[0-9a-f]{16}$")
mcp = FastMCP("riskprobe")
_SERVER_TOOLS: Any = None


class RiskProbeTools:
    """Aggregate-only operations with an explicit in-process workflow state."""

    def __init__(self, registry: DatasetRegistry, store: RunStore) -> None:
        self.registry = registry
        self.store = store
        self._inspected: set[str] = set()
        self._discovered: dict[str, tuple[str, ...]] = {}
        self._validated: dict[str, str] = {}
        self._detected: dict[str, dict[str, Any]] = {}
        self._diagnosed: set[str] = set()
        self._retry_counts: dict[str, int] = {}
        self._alert_handles: dict[str, tuple[str, str]] = {}

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

    def discover_rules(
        self, dataset_id: str, objective: str, constraints: dict[str, Any]
    ) -> dict[str, Any]:
        self._require(dataset_id in self._inspected, "inspect must complete before discover")
        if not isinstance(objective, str) or objective.strip() != "risk":
            raise ValueError("objective must be the supported 'risk' objective")
        if not isinstance(constraints, dict):
            raise ValueError("constraints must be an object")
        if constraints:
            raise ValueError("non-empty constraints are unsupported and must be rejected")
        service = self._service(dataset_id)
        rules = service.discover()
        rule_ids = tuple(rule.rule_id for rule in rules)
        self._discovered[dataset_id] = rule_ids
        return self._safe(
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

    def validate_rules(
        self, dataset_id: str, rule_ids: list[str], split_config: dict[str, Any]
    ) -> dict[str, Any]:
        self._require(dataset_id in self._discovered, "discover must complete before validate")
        if not isinstance(rule_ids, list) or not isinstance(split_config, dict):
            raise ValueError("rule_ids and split_config are required")
        if split_config:
            raise ValueError("non-empty split_config is unsupported and must be rejected")
        discovered = set(self._discovered[dataset_id])
        if not set(rule_ids).issubset(discovered):
            raise ValueError("rule_ids must come from discover")
        context = self._run_dataset(dataset_id)
        self._validated[dataset_id] = context.run_id
        cards = self._read_cards(context.run_dir / "evidence_cards.json")
        grade_counts: dict[str, int] = {}
        for card in cards:
            grade_counts[card.grade] = grade_counts.get(card.grade, 0) + 1
        return self._safe(
            {
                "dataset_id": dataset_id,
                "run_id": context.run_id,
                "evidence_card_count": len(cards),
                "retry_count": self._retry_counts.get(dataset_id, 0),
                "grade_counts": grade_counts,
            }
        )

    def detect_anomalies(self, reference_run_id: str, current_dataset_id: str) -> dict[str, Any]:
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
        alert_handles = []
        for alert in alerts:
            handle = stable_token(f"{current_context.run_id}:{alert.alert_id}", namespace="alert")
            self._alert_handles[handle] = (current_context.run_id, alert.alert_id)
            alert_handles.append(handle)
        return self._safe(
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

    def diagnose_anomaly(self, alert_ids: list[str]) -> dict[str, Any]:
        self._require(isinstance(alert_ids, list) and alert_ids, "alert_ids are required")
        handles = set(alert_ids)
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
        return self._safe(
            {
                "run_id": run_id,
                "diagnosis_count": len(diagnoses),
                "root_cause_count": sum(len(diagnosis.root_causes) for diagnosis in diagnoses),
            }
        )

    def build_report(self, run_id: str, report_type: str) -> dict[str, Any]:
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
        return tuple(EvidenceCard.model_validate(item) for item in payload)

    @staticmethod
    def _frame_profile_catalog(config: Any) -> tuple[Any, Any, FeatureCatalog]:
        dataset = ParquetDataset(config.dataset.path)
        profile = profile_dataset(dataset, config)
        roles = (config.columns.entity, config.columns.snapshot, config.columns.segment, config.columns.target)
        features = config.features.select_columns(dataset.schema().names(), roles)
        frame = dataset.collect([config.columns.segment, config.columns.target, *features])
        catalog = FeatureCatalog.from_columns(features, config.features.families)
        return frame, profile, catalog

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
def inspect_dataset(dataset_id: str) -> dict[str, Any]:
    return get_tools().inspect_dataset(dataset_id)


@mcp.tool()
def discover_rules(dataset_id: str, objective: str, constraints: dict[str, Any]) -> dict[str, Any]:
    return get_tools().discover_rules(dataset_id, objective, constraints)


@mcp.tool()
def validate_rules(dataset_id: str, rule_ids: list[str], split_config: dict[str, Any]) -> dict[str, Any]:
    return get_tools().validate_rules(dataset_id, rule_ids, split_config)


@mcp.tool()
def detect_anomalies(reference_run_id: str, current_dataset_id: str) -> dict[str, Any]:
    return get_tools().detect_anomalies(reference_run_id, current_dataset_id)


@mcp.tool()
def diagnose_anomaly(alert_ids: list[str]) -> dict[str, Any]:
    return get_tools().diagnose_anomaly(alert_ids)


@mcp.tool()
def build_report(run_id: str, report_type: str) -> dict[str, Any]:
    return get_tools().build_report(run_id, report_type)


if __name__ == "__main__":
    mcp.run(transport="stdio")
