"""Policy-enforcing gateway over an injected, implementation-agnostic tool handler."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast, runtime_checkable

from riskprobe.evidence import assert_safe_payload
from riskprobe.policy import Budget, Capability, PolicyEngine, Principal, ToolCall
from riskprobe.registry import DatasetHandle, DatasetRegistry
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
    TraceRequest,
    TraceResponse,
    _ControlledRecommendRequest,
)


class ToolContractError(RuntimeError):
    """Raised with a fixed message when a handler violates the public contract."""


@runtime_checkable
class ToolHandler(Protocol):
    def handle(
        self,
        request: ToolRequest,
        dataset: DatasetHandle | None,
    ) -> ToolResponse: ...


@runtime_checkable
class ToolGateway(Protocol):
    def invoke(
        self,
        principal: Principal,
        request: ToolRequest,
        budget: Budget,
    ) -> ToolResponse: ...


HandlerCallable = Callable[[ToolRequest, DatasetHandle | None], ToolResponse]
SafePayloadHook = Callable[[object], None]

_CONTRACTS: dict[type[ToolRequest], tuple[Capability, type[ToolResponse], bool]] = {
    InspectRequest: (Capability.INSPECT, InspectResponse, True),
    DiscoverRequest: (Capability.DISCOVER, DiscoverResponse, True),
    DiagnoseRequest: (Capability.DIAGNOSE, DiagnoseResponse, True),
    RecommendRequest: (Capability.RECOMMEND, RecommendResponse, True),
    _ControlledRecommendRequest: (
        Capability.RECOMMEND,
        RecommendResponse,
        True,
    ),
    RunRequest: (Capability.RUN, RunResponse, True),
    StatusRequest: (Capability.STATUS, StatusResponse, False),
    TraceRequest: (Capability.TRACE, TraceResponse, False),
    EvidenceLookupRequest: (
        Capability.EVIDENCE_LOOKUP,
        EvidenceLookupResponse,
        False,
    ),
}


class HandlerToolGateway:
    """Resolve IDs privately, authorize, invoke a handler, and validate safe output."""

    def __init__(
        self,
        *,
        registry: DatasetRegistry,
        policy: PolicyEngine,
        handler: ToolHandler | HandlerCallable | object,
        safe_payload_hook: SafePayloadHook = assert_safe_payload,
    ) -> None:
        if not isinstance(registry, DatasetRegistry):
            raise TypeError("registry must be a DatasetRegistry")
        if not isinstance(policy, PolicyEngine):
            raise TypeError("policy must be a PolicyEngine")
        if not callable(safe_payload_hook):
            raise TypeError("safe_payload_hook must be callable")
        self._registry = registry
        self._policy = policy
        self._handler = handler
        self._safe_payload_hook = safe_payload_hook

    def invoke(
        self,
        principal: Principal,
        request: ToolRequest,
        budget: Budget,
    ) -> ToolResponse:
        contract = _CONTRACTS.get(type(request))
        if contract is None:
            raise ToolContractError("unsupported tool request")
        capability, response_type, needs_dataset = contract
        call = ToolCall(
            capability=capability,
            dataset_id=getattr(request, "dataset_id", None),
            run_id=getattr(request, "run_id", None),
            evidence_id=getattr(request, "evidence_id", None),
        )
        self._policy.authorize(principal, call, budget)
        dataset = (
            self._registry.resolve(cast(str, getattr(request, "dataset_id")))
            if needs_dataset
            else None
        )
        response = self._invoke_handler(request, dataset, capability)
        if not isinstance(response, response_type):
            raise ToolContractError("tool handler returned an unexpected response type")
        self._require_matching_resource(request, response)
        try:
            result = self._safe_payload_hook(response.model_dump(mode="python"))
            if result is False:
                raise ValueError("payload hook rejected response")
        except Exception as error:
            raise ToolContractError("tool response is unsafe") from error
        return response

    def _invoke_handler(
        self,
        request: ToolRequest,
        dataset: DatasetHandle | None,
        capability: Capability,
    ) -> ToolResponse:
        try:
            handle = getattr(self._handler, "handle", None)
            if callable(handle):
                return cast(ToolResponse, handle(request, dataset))
            if callable(self._handler):
                return cast(HandlerCallable, self._handler)(request, dataset)
            method = getattr(self._handler, capability.value, None)
            if callable(method):
                return cast(ToolResponse, method(request, dataset))
            raise TypeError("handler has no compatible entry point")
        except Exception as error:
            raise ToolContractError("tool handler failed") from error

    @staticmethod
    def _require_matching_resource(request: ToolRequest, response: ToolResponse) -> None:
        if hasattr(request, "dataset_id") and getattr(response, "dataset_id", None) != getattr(
            request, "dataset_id"
        ):
            raise ToolContractError("tool response resource does not match request")
        if isinstance(request, (StatusRequest, TraceRequest)) and getattr(
            response, "run_id", None
        ) != getattr(request, "run_id"):
            raise ToolContractError("tool response resource does not match request")
        if isinstance(request, EvidenceLookupRequest) and getattr(
            response, "evidence_id", None
        ) != request.evidence_id:
            raise ToolContractError("tool response resource does not match request")

    def inspect(
        self,
        principal: Principal,
        request: InspectRequest,
        budget: Budget,
    ) -> InspectResponse:
        return cast(InspectResponse, self.invoke(principal, request, budget))

    def discover(
        self,
        principal: Principal,
        request: DiscoverRequest,
        budget: Budget,
    ) -> DiscoverResponse:
        return cast(DiscoverResponse, self.invoke(principal, request, budget))

    def diagnose(
        self,
        principal: Principal,
        request: DiagnoseRequest,
        budget: Budget,
    ) -> DiagnoseResponse:
        return cast(DiagnoseResponse, self.invoke(principal, request, budget))

    def recommend(
        self,
        principal: Principal,
        request: RecommendRequest,
        budget: Budget,
    ) -> RecommendResponse:
        return cast(RecommendResponse, self.invoke(principal, request, budget))

    def run(
        self,
        principal: Principal,
        request: RunRequest,
        budget: Budget,
    ) -> RunResponse:
        return cast(RunResponse, self.invoke(principal, request, budget))

    def status(
        self,
        principal: Principal,
        request: StatusRequest,
        budget: Budget,
    ) -> StatusResponse:
        return cast(StatusResponse, self.invoke(principal, request, budget))

    def trace(
        self,
        principal: Principal,
        request: TraceRequest,
        budget: Budget,
    ) -> TraceResponse:
        return cast(TraceResponse, self.invoke(principal, request, budget))

    def evidence_lookup(
        self,
        principal: Principal,
        request: EvidenceLookupRequest,
        budget: Budget,
    ) -> EvidenceLookupResponse:
        return cast(EvidenceLookupResponse, self.invoke(principal, request, budget))


ToolService = HandlerToolGateway
