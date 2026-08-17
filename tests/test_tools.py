from pathlib import Path

import pytest
from pydantic import ValidationError

from riskprobe.config import ProjectConfig
from riskprobe.policy import (
    Budget,
    PolicyDeniedError,
    PolicyEngine,
    Principal,
    Role,
)
from riskprobe.registry import DatasetHandle, DatasetNotRegisteredError, DatasetRegistry
from riskprobe.tools import (
    DiagnoseRequest,
    DiagnoseResponse,
    DiscoverRequest,
    DiscoverResponse,
    EvidenceLookupRequest,
    EvidenceLookupResponse,
    HandlerToolGateway,
    InspectRequest,
    InspectResponse,
    RecommendRequest,
    RecommendResponse,
    RunRequest,
    RunResponse,
    StatusRequest,
    StatusResponse,
    ToolContractError,
    ToolGateway,
    TraceEvent,
    TraceRequest,
    TraceResponse,
)


class RecordingHandler:
    def __init__(self) -> None:
        self.calls: list[tuple[object, DatasetHandle | None]] = []

    def handle(self, request: object, dataset: DatasetHandle | None) -> object:
        self.calls.append((request, dataset))
        if isinstance(request, InspectRequest):
            return InspectResponse(
                dataset_id=request.dataset_id,
                row_count=100,
                feature_count=8,
                metadata_grade="A",
            )
        if isinstance(request, DiscoverRequest):
            return DiscoverResponse(dataset_id=request.dataset_id, rule_ids=("rule-1",))
        if isinstance(request, DiagnoseRequest):
            return DiagnoseResponse(
                dataset_id=request.dataset_id,
                finding_ids=("a" * 64,),
            )
        if isinstance(request, RecommendRequest):
            return RecommendResponse(
                dataset_id=request.dataset_id,
                recommendation_ids=("b" * 64,),
            )
        if isinstance(request, RunRequest):
            return RunResponse(
                dataset_id=request.dataset_id,
                run_id="run-001",
                reused=False,
                metadata_grade="A",
            )
        if isinstance(request, StatusRequest):
            return StatusResponse(run_id=request.run_id, status="succeeded")
        if isinstance(request, TraceRequest):
            return TraceResponse(
                run_id=request.run_id,
                events=(
                    TraceEvent(
                        sequence=1,
                        node_id="profile",
                        event_type="node_succeeded",
                        status="succeeded",
                        attempt=1,
                    ),
                ),
            )
        if isinstance(request, EvidenceLookupRequest):
            return EvidenceLookupResponse(
                evidence_id=request.evidence_id,
                run_id="run-001",
                kind="finding",
                payload={"finding_count": 1},
            )
        raise AssertionError("unexpected request")


def _gateway(
    synthetic_config: ProjectConfig,
    handler: object,
) -> HandlerToolGateway:
    return HandlerToolGateway(
        registry=DatasetRegistry.from_mapping({"synthetic_demo": synthetic_config}),
        policy=PolicyEngine(),
        handler=handler,
    )


def _principal(role: Role) -> Principal:
    return Principal(principal_id=f"tool-{role.value}", role=role)


def test_gateway_is_protocol_compatible_and_passes_private_handle_only_to_handler(
    synthetic_config: ProjectConfig,
) -> None:
    handler = RecordingHandler()
    gateway = _gateway(synthetic_config, handler)

    response = gateway.inspect(
        _principal(Role.ANALYST),
        InspectRequest(dataset_id="synthetic_demo"),
        Budget(max_queries=1),
    )

    assert isinstance(gateway, ToolGateway)
    assert response == InspectResponse(
        dataset_id="synthetic_demo",
        row_count=100,
        feature_count=8,
        metadata_grade="A",
    )
    request, handle = handler.calls[0]
    assert isinstance(request, InspectRequest)
    assert handle is not None
    assert handle.config is synthetic_config
    assert str(synthetic_config.dataset.path) not in response.model_dump_json()


@pytest.mark.parametrize(
    ("tool_request", "response_type"),
    [
        (DiscoverRequest(dataset_id="synthetic_demo"), DiscoverResponse),
        (DiagnoseRequest(dataset_id="synthetic_demo"), DiagnoseResponse),
        (RecommendRequest(dataset_id="synthetic_demo"), RecommendResponse),
        (RunRequest(dataset_id="synthetic_demo"), RunResponse),
        (StatusRequest(run_id="run-001"), StatusResponse),
        (TraceRequest(run_id="run-001"), TraceResponse),
        (EvidenceLookupRequest(evidence_id="a" * 64), EvidenceLookupResponse),
    ],
)
def test_gateway_dispatches_every_typed_contract(
    synthetic_config: ProjectConfig,
    tool_request: object,
    response_type: type[object],
) -> None:
    gateway = _gateway(synthetic_config, RecordingHandler())

    response = gateway.invoke(
        _principal(Role.OPERATOR),
        tool_request,
        Budget(max_queries=1),
    )

    assert isinstance(response, response_type)


def test_gateway_authorizes_before_invoking_handler(
    synthetic_config: ProjectConfig,
) -> None:
    handler = RecordingHandler()
    gateway = _gateway(synthetic_config, handler)
    budget = Budget(max_queries=1)

    with pytest.raises(PolicyDeniedError):
        gateway.run(
            _principal(Role.ANALYST),
            RunRequest(dataset_id="synthetic_demo"),
            budget,
        )

    assert handler.calls == []
    assert budget.used_queries == 0


def test_gateway_resolves_dataset_id_and_never_accepts_a_path(
    synthetic_config: ProjectConfig,
) -> None:
    handler = RecordingHandler()
    gateway = _gateway(synthetic_config, handler)

    with pytest.raises(ValidationError):
        InspectRequest.model_validate(
            {"dataset_id": "synthetic_demo", "path": "/private/company.parquet"}
        )
    with pytest.raises(ValidationError):
        InspectRequest(dataset_id=Path("/private/company.parquet"))
    with pytest.raises(DatasetNotRegisteredError):
        gateway.inspect(
            _principal(Role.ANALYST),
            InspectRequest(dataset_id="unknown_demo"),
            Budget(max_queries=1),
        )

    assert handler.calls == []


def test_gateway_rejects_wrong_response_type_and_unsafe_payload_without_leakage(
    synthetic_config: ProjectConfig,
) -> None:
    private_path = "/private/company.parquet"

    def wrong_type(request: object, dataset: DatasetHandle | None) -> object:
        del request, dataset
        return StatusResponse(run_id="run-001", status="succeeded")

    with pytest.raises(ToolContractError, match="unexpected response type"):
        _gateway(synthetic_config, wrong_type).inspect(
            _principal(Role.ANALYST),
            InspectRequest(dataset_id="synthetic_demo"),
            Budget(max_queries=1),
        )

    def unsafe(request: object, dataset: DatasetHandle | None) -> object:
        del dataset
        assert isinstance(request, EvidenceLookupRequest)
        return EvidenceLookupResponse(
            evidence_id=request.evidence_id,
            run_id="run-001",
            kind="finding",
            payload={"config_path": private_path},
        )

    with pytest.raises(ToolContractError) as exc_info:
        _gateway(synthetic_config, unsafe).invoke(
            _principal(Role.OPERATOR),
            EvidenceLookupRequest(evidence_id="b" * 64),
            Budget(max_queries=1),
        )

    assert str(exc_info.value) == "tool response is unsafe"
    assert private_path not in str(exc_info.value)


def test_handler_failure_is_sanitized(
    synthetic_config: ProjectConfig,
) -> None:
    private_path = "/private/company.parquet"

    def fail(request: object, dataset: DatasetHandle | None) -> object:
        del request, dataset
        raise RuntimeError(f"could not read {private_path}")

    with pytest.raises(ToolContractError) as exc_info:
        _gateway(synthetic_config, fail).inspect(
            _principal(Role.ANALYST),
            InspectRequest(dataset_id="synthetic_demo"),
            Budget(max_queries=1),
        )

    assert str(exc_info.value) == "tool handler failed"
    assert private_path not in str(exc_info.value)


def test_tool_dtos_are_strict_and_aggregate_only() -> None:
    with pytest.raises(ValidationError):
        InspectResponse(
            dataset_id="synthetic_demo",
            row_count="100",
            feature_count=8,
            metadata_grade="A",
        )
    with pytest.raises(ValidationError):
        TraceResponse(
            run_id="run-001",
            events=[
                TraceEvent(
                    sequence=1,
                    node_id="profile",
                    event_type="node_succeeded",
                    status="succeeded",
                    attempt=1,
                )
            ],
        )


@pytest.mark.parametrize(
    ("response_type", "field_name"),
    [
        (DiagnoseResponse, "finding_ids"),
        (RecommendResponse, "recommendation_ids"),
    ],
)
@pytest.mark.parametrize(
    "invalid_id",
    ["finding-1", "A" * 64, "a" * 63, "a" * 65, "g" * 64],
)
def test_evidence_response_ids_require_strict_lowercase_sha256(
    response_type: type[DiagnoseResponse] | type[RecommendResponse],
    field_name: str,
    invalid_id: str,
) -> None:
    with pytest.raises(ValidationError, match="SHA-256"):
        response_type.model_validate(
            {
                "dataset_id": "synthetic_demo",
                field_name: (invalid_id,),
            }
        )
