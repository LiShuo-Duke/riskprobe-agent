"""Typed, path-free RiskProbe tool contracts and injectable gateway."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from riskprobe.tools.local import LocalRiskProbeToolHandler

from riskprobe.tools.models import (
    DiagnoseRequest,
    DiagnoseResponse,
    DiscoverRequest,
    DiscoverResponse,
    EvidenceLookupRequest,
    EvidenceLookupResponse,
    EvidenceRequest,
    EvidenceResponse,
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
    ToolStatus,
    TraceEvent,
    TraceRequest,
    TraceResponse,
)
from riskprobe.tools.service import (
    HandlerCallable,
    HandlerToolGateway,
    ToolContractError,
    ToolGateway,
    ToolHandler,
    ToolService,
)


def __getattr__(name: str) -> object:
    if name == "LocalRiskProbeToolHandler":
        from riskprobe.tools.local import LocalRiskProbeToolHandler

        globals()[name] = LocalRiskProbeToolHandler
        return LocalRiskProbeToolHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DiagnoseRequest",
    "DiagnoseResponse",
    "DiscoverRequest",
    "DiscoverResponse",
    "EvidenceLookupRequest",
    "EvidenceLookupResponse",
    "EvidenceRequest",
    "EvidenceResponse",
    "HandlerCallable",
    "HandlerToolGateway",
    "InspectRequest",
    "InspectResponse",
    "LocalRiskProbeToolHandler",
    "RecommendRequest",
    "RecommendResponse",
    "RunRequest",
    "RunResponse",
    "StatusRequest",
    "StatusResponse",
    "ToolContractError",
    "ToolGateway",
    "ToolHandler",
    "ToolRequest",
    "ToolResponse",
    "ToolService",
    "ToolStatus",
    "TraceEvent",
    "TraceRequest",
    "TraceResponse",
]
