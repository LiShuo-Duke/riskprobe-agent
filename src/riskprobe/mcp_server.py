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
from riskprobe.privacy import assert_safe_payload, suppress_small_groups
from riskprobe.profiling import profile_dataset
from riskprobe.registry import DatasetRegistry
from riskprobe.service import RiskProbeService

_RUN_ID = re.compile(r"^[0-9a-f]{16}$")
mcp = FastMCP("riskprobe")


class RiskProbeTools:
    """Synchronous operations that accept only registered IDs and return safe aggregates."""

    def __init__(self, registry: DatasetRegistry, store: RunStore) -> None:
        self.registry = registry
        self.store = store

    def inspect_dataset(self, dataset_id: str) -> dict[str, Any]:
        config = self.registry.get_config(dataset_id)
        profile = profile_dataset(ParquetDataset(config.dataset.path), config)
        payload = {
            "dataset_id": dataset_id,
            "metadata_grade": profile.metadata_grade,
            "row_count": profile.row_count,
            "feature_count": profile.feature_count,
            "positive_rate": profile.positive_rate,
            "segment_counts": suppress_small_groups(
                (
                    {"segment": segment, "count": count}
                    for segment, count in profile.segment_counts.items()
                ),
                "count",
                config.validation.min_group_size,
            ),
            "limitations": sorted(issue.code for issue in profile.issues),
        }
        return self._safe(payload)

    def discover_rules(self, dataset_id: str) -> dict[str, Any]:
        service = self._service(dataset_id)
        payload = {
            "dataset_id": dataset_id,
            "rules": [
                {
                    "rule_id": rule.rule_id,
                    "origin": rule.origin,
                    "expression": " AND ".join(
                        f"{condition.feature} {condition.operator} {condition.value}"
                        for condition in rule.conditions
                    ),
                }
                for rule in service.discover()
            ],
        }
        return self._safe(payload)

    def validate_rules(self, dataset_id: str) -> dict[str, Any]:
        context = self._service(dataset_id).run()
        cards = self._read_cards(context.run_dir / "evidence_cards.json")
        return self._safe(
            {
                "dataset_id": dataset_id,
                "run_id": context.run_id,
                "evidence_cards": [card.model_dump(mode="json") for card in cards],
            }
        )

    def detect_anomalies(self, reference_run_id: str, current_dataset_id: str) -> dict[str, Any]:
        reference = self._reference_snapshot(reference_run_id)
        config = self.registry.get_config(current_dataset_id)
        service = self._service(current_dataset_id)
        current_context = service.run()
        frame, profile, catalog = self._frame_profile_catalog(config)
        cards = self._read_cards(current_context.run_dir / "evidence_cards.json")
        alerts = detect_alerts(reference, frame, cards, catalog)
        return self._safe(
            {
                "reference_run_id": reference_run_id,
                "current_dataset_id": current_dataset_id,
                "alerts": [alert.model_dump(mode="json") for alert in alerts],
                "metadata_grade": profile.metadata_grade,
            }
        )

    def diagnose_anomaly(self, reference_run_id: str, current_dataset_id: str) -> dict[str, Any]:
        reference = self._reference_snapshot(reference_run_id)
        config = self.registry.get_config(current_dataset_id)
        current_context = self._service(current_dataset_id).run()
        frame, _, catalog = self._frame_profile_catalog(config)
        cards = self._read_cards(current_context.run_dir / "evidence_cards.json")
        alerts = detect_alerts(reference, frame, cards, catalog)
        diagnoses = diagnose_alerts(alerts, reference, frame, catalog, top_k=3)
        return self._safe(
            {
                "reference_run_id": reference_run_id,
                "current_dataset_id": current_dataset_id,
                "diagnoses": [diagnosis.model_dump(mode="json") for diagnosis in diagnoses],
            }
        )

    def build_report(self, dataset_id: str) -> dict[str, Any]:
        context = self._service(dataset_id).run()
        markdown = (context.run_dir / "risk_report.md").read_text(encoding="utf-8")
        return self._safe({"report_id": context.run_id, "markdown": markdown})

    def _service(self, dataset_id: str) -> RiskProbeService:
        return RiskProbeService(config=self.registry.get_config(dataset_id), runs_dir=self.store.runs_dir)

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
        assert_safe_payload(payload)
        return payload


def get_tools() -> RiskProbeTools:
    registry_path = Path(os.environ.get("RISKPROBE_REGISTRY", "configs/datasets.example.yaml"))
    return RiskProbeTools(DatasetRegistry.from_yaml(registry_path), RunStore("runs"))


@mcp.tool()
def inspect_dataset(dataset_id: str) -> dict[str, Any]:
    return get_tools().inspect_dataset(dataset_id)


@mcp.tool()
def discover_rules(dataset_id: str) -> dict[str, Any]:
    return get_tools().discover_rules(dataset_id)


@mcp.tool()
def validate_rules(dataset_id: str) -> dict[str, Any]:
    return get_tools().validate_rules(dataset_id)


@mcp.tool()
def detect_anomalies(reference_run_id: str, current_dataset_id: str) -> dict[str, Any]:
    return get_tools().detect_anomalies(reference_run_id, current_dataset_id)


@mcp.tool()
def diagnose_anomaly(reference_run_id: str, current_dataset_id: str) -> dict[str, Any]:
    return get_tools().diagnose_anomaly(reference_run_id, current_dataset_id)


@mcp.tool()
def build_report(dataset_id: str) -> dict[str, Any]:
    return get_tools().build_report(dataset_id)


if __name__ == "__main__":
    mcp.run(transport="stdio")
