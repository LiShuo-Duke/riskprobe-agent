"""Production local implementation of the typed RiskProbe tool handler."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from riskprobe.artifacts import RunContext
from riskprobe.evidence import EvidenceRecord, EvidenceStore, PrivacyClass
from riskprobe.monitoring.models import SafeProfile
from riskprobe.registry import DatasetHandle
from riskprobe.runtime import RunRuntime
from riskprobe.service import RiskProbeService
from riskprobe.tools.models import (
    DiagnoseRequest,
    DiagnoseResponse,
    DiscoverRequest,
    DiscoverResponse,
    EvidenceLookupRequest,
    EvidenceLookupResponse,
    InspectRequest,
    InspectResponse,
    RecommendRequest,
    RecommendResponse,
    RunRequest,
    RunResponse,
    StatusRequest,
    StatusResponse,
    ToolRequest,
    ToolResponse,
    TraceEvent,
    TraceRequest,
    TraceResponse,
    _ControlledRecommendRequest,
)

if TYPE_CHECKING:
    from riskprobe.agents.decision_controller import DecisionController

_PUBLIC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_STATUS_MAP = {
    "pending": "pending",
    "running": "running",
    "succeeded": "succeeded",
    "failed": "failed",
    "interrupted": "failed",
    "invalidated": "failed",
    "cancelled": "cancelled",
}


class LocalRiskProbeToolHandler:
    """Execute local services while binding all evidence and runtime access to one run."""

    def __init__(
        self,
        *,
        run_id: str,
        runs_dir: Path,
        evidence_store: EvidenceStore,
        run_context: RunContext | None = None,
        decision_controller: DecisionController | None = None,
        version: str = "local-tool-handler-v1",
    ) -> None:
        if not isinstance(run_id, str) or _PUBLIC_ID.fullmatch(run_id) is None:
            raise ValueError("run_id must be a public identifier")
        if not isinstance(evidence_store, EvidenceStore):
            raise TypeError("evidence_store must be an EvidenceStore")
        if run_context is not None and type(run_context) is not RunContext:
            raise TypeError("run_context must be a RunContext")
        if decision_controller is not None:
            from riskprobe.agents.decision_controller import DecisionController

            if type(decision_controller) is not DecisionController:
                raise TypeError("decision_controller must be a DecisionController")
        normalized_runs_dir = Path(runs_dir).resolve()
        if run_context is not None and (
            run_context.run_id != run_id
            or run_context.run_dir != normalized_runs_dir / run_id
        ):
            raise ValueError("run_context is unavailable")
        if not version or len(version) > 128 or any(character.isspace() for character in version):
            raise ValueError("version must be a non-empty token")
        self._run_id = run_id
        self._runs_dir = normalized_runs_dir
        self._evidence_store = evidence_store
        self._run_context = run_context
        self._decision_controller = decision_controller
        self.version = version

    def handle(
        self,
        request: ToolRequest,
        dataset: DatasetHandle | None,
    ) -> ToolResponse:
        if isinstance(request, InspectRequest):
            return self._inspect(request, self._require_dataset(request.dataset_id, dataset))
        if isinstance(request, DiscoverRequest):
            return self._discover(request, self._require_dataset(request.dataset_id, dataset))
        if isinstance(request, DiagnoseRequest):
            return self._diagnose(request, self._require_dataset(request.dataset_id, dataset))
        if isinstance(request, RecommendRequest):
            return self._recommend(request, self._require_dataset(request.dataset_id, dataset))
        if isinstance(request, RunRequest):
            return self._run(request, self._require_dataset(request.dataset_id, dataset))
        if isinstance(request, StatusRequest):
            return self._status(request)
        if isinstance(request, TraceRequest):
            return self._trace(request)
        if isinstance(request, EvidenceLookupRequest):
            return self._lookup(request)
        raise TypeError("unsupported tool request")

    @staticmethod
    def _require_dataset(
        dataset_id: str,
        dataset: DatasetHandle | None,
    ) -> DatasetHandle:
        if dataset is None or dataset.dataset_id != dataset_id:
            raise RuntimeError("dataset is unavailable")
        return dataset

    def _service(self, dataset: DatasetHandle) -> RiskProbeService:
        return RiskProbeService(config=dataset.config, runs_dir=self._runs_dir)

    def _inspect(
        self,
        request: InspectRequest,
        dataset: DatasetHandle,
    ) -> InspectResponse:
        service = self._service(dataset)
        profile = (
            SafeProfile.from_profile(service.inspect())
            if self._run_context is None
            else service._profile_from_run(self._run_context)
        )
        return InspectResponse(
            dataset_id=request.dataset_id,
            row_count=profile.row_count,
            feature_count=profile.feature_count,
            metadata_grade=profile.metadata_grade,
            issue_codes=profile.issue_codes,
        )

    def _discover(
        self,
        request: DiscoverRequest,
        dataset: DatasetHandle,
    ) -> DiscoverResponse:
        service = self._service(dataset)
        rules = (
            service.discover()
            if self._run_context is None
            else service._rules_from_run(self._run_context)
        )
        return DiscoverResponse(
            dataset_id=request.dataset_id,
            rule_ids=tuple(sorted(rule.rule_id for rule in rules)),
        )

    def _diagnose(
        self,
        request: DiagnoseRequest,
        dataset: DatasetHandle,
    ) -> DiagnoseResponse:
        return self._service(dataset)._diagnose_with_store(
            self._run_id,
            self._evidence_store,
            dataset_id=request.dataset_id,
            run_context=self._run_context,
        )

    def _recommend(
        self,
        request: RecommendRequest,
        dataset: DatasetHandle,
    ) -> RecommendResponse:
        service = self._service(dataset)
        decision_result_evidence_id = (
            request.decision_result_evidence_id
            if isinstance(request, _ControlledRecommendRequest)
            else None
        )
        if decision_result_evidence_id is not None and self._decision_controller is None:
            raise RuntimeError("decision is unavailable")
        if self._run_context is None:
            return service._recommend_with_store(
                self._run_id,
                request.evidence_ids,
                self._evidence_store,
                dataset_id=request.dataset_id,
                decision_controller=self._decision_controller,
                decision_result_evidence_id=decision_result_evidence_id,
            )
        return service._recommend_with_store(
            self._run_id,
            request.evidence_ids,
            self._evidence_store,
            dataset_id=request.dataset_id,
            safe_profile=service._profile_from_run(self._run_context),
            decision_controller=self._decision_controller,
            decision_result_evidence_id=decision_result_evidence_id,
        )

    def _run(self, request: RunRequest, dataset: DatasetHandle) -> RunResponse:
        context = self._service(dataset).run()
        if context.run_id != self._run_id:
            raise RuntimeError("run is unavailable")
        return RunResponse(
            dataset_id=request.dataset_id,
            run_id=context.run_id,
            reused=context.is_existing,
            metadata_grade=dataset.config.metadata_grade,
            artifact_count=6,
        )

    def _status(self, request: StatusRequest) -> StatusResponse:
        self._require_run(request.run_id)
        status = self._tool_status(RunRuntime(self._runs_dir, self._run_id).run_status().value)
        return StatusResponse(run_id=self._run_id, status=status)

    def _trace(self, request: TraceRequest) -> TraceResponse:
        self._require_run(request.run_id)
        events: list[TraceEvent] = []
        for event in RunRuntime(self._runs_dir, self._run_id).trace():
            node_id = event.get("node_id") or "run"
            attempt = event.get("attempt")
            events.append(
                TraceEvent(
                    sequence=event["sequence"],
                    node_id=node_id,
                    event_type=event["event_type"],
                    status=self._tool_status(event["status"]),
                    attempt=max(1, attempt),
                    error_class=event.get("error_class"),
                )
            )
        return TraceResponse(run_id=self._run_id, events=tuple(events))

    def _lookup(self, request: EvidenceLookupRequest) -> EvidenceLookupResponse:
        record = self._current_record(request.evidence_id)
        return EvidenceLookupResponse(
            evidence_id=request.evidence_id,
            run_id=record.run_id,
            kind=record.kind,
            payload=record.payload,
            parent_ids=record.parent_ids,
            artifact_hashes=record.artifact_hashes,
            privacy_class=record.privacy_class.value,
            producer_version=record.producer_version,
        )

    def _current_record(self, evidence_id: str) -> EvidenceRecord:
        record = self._evidence_store.get(evidence_id)
        if (
            record is None
            or record.run_id != self._run_id
            or EvidenceStore.content_id(record) != evidence_id
            or record.privacy_class is not PrivacyClass.AGGREGATE
        ):
            raise RuntimeError("evidence is unavailable")
        return record

    def _require_run(self, run_id: str) -> None:
        if run_id != self._run_id:
            raise RuntimeError("run is unavailable")

    @staticmethod
    def _tool_status(status: object) -> str:
        if not isinstance(status, str) or status not in _STATUS_MAP:
            raise RuntimeError("runtime status is unavailable")
        return _STATUS_MAP[status]


__all__ = ["LocalRiskProbeToolHandler"]
