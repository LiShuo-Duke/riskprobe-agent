"""Vendor-neutral external integration protocols with disabled defaults.

Direct ``IntegrationRequest`` Pydantic construction is trusted-code only. External
values must enter through ``IntegrationRequest.safe_validate`` or an adapter, both
of which copy and fully revalidate model instances as well as mappings.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import NoReturn, Protocol, Self, cast, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from riskprobe.privacy import assert_safe_payload

_LEGACY_REQUEST_CODE = re.compile(r"^request-[0-9]{3}$")
_OPAQUE_REQUEST_ID = re.compile(r"^request-[0-9a-f]{24}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STATUS_CODES = frozenset({"error", "ok", "stable", "unknown", "warning"})
_MAX_OPERATION_LENGTH = 32
_MAX_STATUS_LENGTH = 16
_SHA256_LENGTH = 64
_LEGACY_REQUEST_ID_LENGTH = len("request-000")
_OPAQUE_REQUEST_ID_LENGTH = len("request-") + 24
_OPERATION_SPECS: dict[str, dict[str, str]] = {
    "embed_summary": {"document_ids": "digest_tuple"},
    "emit_summary_metrics": {"finding_count": "count", "status": "status"},
    "enqueue_summary": {"finding_count": "count", "summary_id": "digest"},
    "generate_summary": {"finding_count": "count", "status": "status"},
    "get_summary_cache": {"cache_id": "digest"},
    "publish_summary": {"finding_count": "count", "status": "status"},
    "set_summary_cache": {"cache_id": "digest", "summary_hash": "digest"},
}
_REQUEST_KEYS = frozenset({"operation", "payload", "request_id"})
_REQUEST_KEYS_WITH_HASH = frozenset({*_REQUEST_KEYS, "content_hash"})
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))
_TRUSTED_REQUEST_REFS: dict[
    int,
    tuple[
        weakref.ReferenceType[object],
        object,
        str,
        str,
        str,
    ],
] = {}
_TRUSTED_REQUEST_REFS_LOCK = threading.Lock()
_SECRET_VALUE = re.compile(
    r"(?:\bsk-(?:live|test)-?[A-Za-z0-9_-]{6,}\b|"
    r"\bgh[pousr]_[A-Za-z0-9]{8,}\b|\bAKIA[A-Z0-9]{12,}\b|"
    r"[\"']?\b(?:password|passwd|secret|api[ _-]?key|access[ _-]?token|"
    r"auth[ _-]?token|bearer)\b[\"']?"
    r"\s*(?::|=|\bis\b|\bwas\b|\bof\b|\s)\s*[\"']?\S+)",
    re.IGNORECASE,
)
_SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "auth_token",
        "bearer_token",
        "credential",
        "credentials",
        "password",
        "passwd",
        "private_key",
        "secret",
    }
)
_PAYLOAD_ERROR = "integration payload is not safe"
_DISABLED_ERROR = "integration is disabled"


class IntegrationDisabledError(RuntimeError):
    """Raised by every safe invocation of a disabled external integration."""


class IntegrationPayloadError(ValueError):
    """Raised when an integration request is not aggregate-safe."""


def _raise_unlinked(error_type: type[Exception], message: str) -> NoReturn:
    """Raise a fixed exception without retaining a cause/context object graph."""

    error = error_type(message)
    try:
        raise error from None
    finally:
        error.__cause__ = None
        error.__context__ = None


def _passive_exact_string_mapping(
    value: object,
    *,
    expected_type: type[object],
) -> tuple[Mapping[str, object], frozenset[str]]:
    if type(value) is not expected_type:
        raise TypeError
    mapping = cast(Mapping[object, object], value)
    keys: list[str] = []
    for key in mapping:
        if type(key) is not str:
            raise TypeError
        keys.append(key)
    return cast(Mapping[str, object], mapping), frozenset(keys)


def _preflight_request_id(value: object) -> str:
    if type(value) is not str:
        raise TypeError
    length = len(value)
    if length == _LEGACY_REQUEST_ID_LENGTH:
        if _LEGACY_REQUEST_CODE.fullmatch(value) is None:
            raise ValueError
    elif length == _OPAQUE_REQUEST_ID_LENGTH:
        if _OPAQUE_REQUEST_ID.fullmatch(value) is None:
            raise ValueError
    else:
        raise ValueError
    return value


def _preflight_content_hash(value: object) -> str:
    if type(value) is not str:
        raise TypeError
    length = len(value)
    if length == 0:
        return value
    if length != _SHA256_LENGTH or _SHA256.fullmatch(value) is None:
        raise ValueError
    return value


def _preflight_operation(operation: object) -> tuple[str, dict[str, str]]:
    if type(operation) is not str:
        raise TypeError
    length = len(operation)
    if not 1 <= length <= _MAX_OPERATION_LENGTH:
        raise ValueError
    specification = _OPERATION_SPECS.get(operation)
    if specification is None:
        raise ValueError
    return operation, specification


def _preflight_payload_scalar(field_type: str, field_value: object) -> None:
    if field_type == "count":
        if type(field_value) is not int or not 0 <= field_value <= 1_000_000:
            raise ValueError
        return
    if field_type == "status":
        if type(field_value) is not str:
            raise TypeError
        length = len(field_value)
        if length > _MAX_STATUS_LENGTH or field_value not in _STATUS_CODES:
            raise ValueError
        return
    if field_type == "digest":
        if type(field_value) is not str:
            raise TypeError
        length = len(field_value)
        if length != _SHA256_LENGTH or _SHA256.fullmatch(field_value) is None:
            raise ValueError
        return
    if field_type == "digest_tuple":
        if type(field_value) not in {list, tuple}:
            raise TypeError
        if not 1 <= len(field_value) <= 256:
            raise ValueError
        for item in field_value:
            if type(item) is not str:
                raise TypeError
            length = len(item)
            if length != _SHA256_LENGTH or _SHA256.fullmatch(item) is None:
                raise ValueError
        return
    raise RuntimeError


def _register_trusted_request(value: IntegrationRequest) -> None:
    identifier = id(value)

    def remove(dead_reference: weakref.ReferenceType[object]) -> None:
        with _TRUSTED_REQUEST_REFS_LOCK:
            state = _TRUSTED_REQUEST_REFS.get(identifier)
            if state is not None and state[0] is dead_reference:
                _TRUSTED_REQUEST_REFS.pop(identifier, None)

    reference = weakref.ref(value, remove)
    state = (
        reference,
        value.payload,
        value.request_id,
        value.operation,
        value.content_hash,
    )
    with _TRUSTED_REQUEST_REFS_LOCK:
        _TRUSTED_REQUEST_REFS[identifier] = state


def _is_registered_trusted_request(value: IntegrationRequest) -> bool:
    payload = value.payload
    request_id = value.request_id
    operation = value.operation
    content_hash = value.content_hash
    if (
        type(request_id) is not str
        or type(operation) is not str
        or type(content_hash) is not str
    ):
        return False
    request_id_length = len(request_id)
    operation_length = len(operation)
    content_hash_length = len(content_hash)
    if (
        request_id_length not in {
            _LEGACY_REQUEST_ID_LENGTH,
            _OPAQUE_REQUEST_ID_LENGTH,
        }
        or not 1 <= operation_length <= _MAX_OPERATION_LENGTH
        or content_hash_length not in {0, _SHA256_LENGTH}
    ):
        return False
    with _TRUSTED_REQUEST_REFS_LOCK:
        state = _TRUSTED_REQUEST_REFS.get(id(value))
        if state is None or state[0]() is not value:
            return False
        return (
            payload is state[1]
            and request_id == state[2]
            and operation == state[3]
            and content_hash == state[4]
        )


class IntegrationRequest(BaseModel):
    """Strict content-addressed request; direct construction is trusted-only."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )

    request_id: str
    operation: str
    payload: Mapping[str, object]
    content_hash: str = ""

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        _register_trusted_request(self)

    @classmethod
    def safe_validate(cls, value: object) -> Self:
        """Snapshot and revalidate only exact passive boundary inputs."""

        try:
            return _request_from_boundary(cls, value)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            value = None
            _raise_unlinked(IntegrationPayloadError, _PAYLOAD_ERROR)

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        try:
            return _preflight_request_id(value)
        except Exception:
            raise ValueError("request_id must be content-addressed") from None

    @field_validator("operation")
    @classmethod
    def validate_operation(cls, value: str) -> str:
        try:
            operation, _ = _preflight_operation(value)
            return operation
        except Exception:
            raise ValueError("operation is not allowed") from None

    @field_validator("payload")
    @classmethod
    def validate_payload(
        cls,
        value: Mapping[str, object],
        info: ValidationInfo,
    ) -> Mapping[str, object]:
        try:
            operation = info.data.get("operation")
            if not isinstance(operation, str):
                raise TypeError
            normalized = _validate_operation_payload(operation, value)
            assert_safe_payload(normalized)
            _assert_no_secret_material(normalized)
            frozen = _freeze_value(normalized)
            if not isinstance(frozen, Mapping):
                raise TypeError
        except Exception:
            raise ValueError(_PAYLOAD_ERROR) from None
        return frozen

    @field_serializer("payload")
    def serialize_payload(self, value: Mapping[str, object]) -> dict[str, object]:
        serialized = _jsonable_value(value)
        if not isinstance(serialized, dict):
            raise ValueError(_PAYLOAD_ERROR)
        return serialized

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        try:
            return _preflight_content_hash(value)
        except Exception:
            raise ValueError("content_hash must be a SHA-256 identifier") from None

    @model_validator(mode="after")
    def derive_content_hash(self) -> IntegrationRequest:
        typed_content = {
            "operation": self.operation,
            "payload": _jsonable_value(self.payload),
        }
        try:
            assert_safe_payload(typed_content)
            _assert_no_secret_material(typed_content)
            request_digest = hashlib.sha256(
                b"integration-request\0" + _canonical_bytes(typed_content)
            ).hexdigest()[:24]
            expected_request_id = f"request-{request_digest}"
            if (
                _OPAQUE_REQUEST_ID.fullmatch(self.request_id) is not None
                and self.request_id != expected_request_id
            ):
                raise ValueError
            object.__setattr__(self, "request_id", expected_request_id)
            addressed = {**typed_content, "request_id": expected_request_id}
            expected = hashlib.sha256(_canonical_bytes(addressed)).hexdigest()
        except Exception:
            raise ValueError(_PAYLOAD_ERROR) from None
        if self.content_hash and self.content_hash != expected:
            raise ValueError("content_hash does not match integration request")
        object.__setattr__(self, "content_hash", expected)
        return self


def _request_from_boundary(
    model_type: type[IntegrationRequest],
    value: object,
) -> IntegrationRequest:
    candidate = _snapshot_request_input(value)
    validated = model_type.model_validate(candidate)
    dumped = validated.model_dump(mode="json")
    assert_safe_payload(dumped)
    _assert_no_secret_material(dumped)
    return validated


def _snapshot_request_input(value: object) -> dict[str, object]:
    if type(value) is IntegrationRequest:
        if not _is_registered_trusted_request(value):
            raise TypeError
        request_id = value.request_id
        operation = value.operation
        payload = value.payload
        content_hash = value.content_hash
        _preflight_request_id(request_id)
        _preflight_content_hash(content_hash)
        _preflight_operation_payload(
            operation,
            payload,
            expected_payload_type=_MAPPING_PROXY_TYPE,
        )
        return value.model_dump(mode="python", warnings=False)

    if type(value) is not dict or len(value) not in {3, 4}:
        raise TypeError
    mapping, keys = _passive_exact_string_mapping(value, expected_type=dict)
    if keys != _REQUEST_KEYS and keys != _REQUEST_KEYS_WITH_HASH:
        raise ValueError
    request_id = _preflight_request_id(mapping["request_id"])
    operation = mapping["operation"]
    payload = mapping["payload"]
    _preflight_operation_payload(
        operation,
        payload,
        expected_payload_type=dict,
    )
    if "content_hash" in keys:
        _preflight_content_hash(mapping["content_hash"])
    candidate = dict(mapping)
    candidate["request_id"] = request_id
    candidate["payload"] = dict(payload)
    return candidate


def _preflight_operation_payload(
    operation: object,
    payload: object,
    *,
    expected_payload_type: type[object],
) -> None:
    _, specification = _preflight_operation(operation)
    if type(payload) is not expected_payload_type:
        raise TypeError
    if len(payload) != len(specification):
        raise ValueError
    mapping, keys = _passive_exact_string_mapping(
        payload,
        expected_type=expected_payload_type,
    )
    if keys != frozenset(specification):
        raise ValueError
    for field_name, field_type in specification.items():
        _preflight_payload_scalar(field_type, mapping[field_name])


RequestLike = IntegrationRequest | Mapping[str, object]


@runtime_checkable
class GitHubAdapter(Protocol):
    """Publish only aggregate-safe, content-addressed requests."""

    def publish(self, *, request: RequestLike) -> str: ...


@runtime_checkable
class RedisAdapter(Protocol):
    """Queue and cache only aggregate-safe, content-addressed requests."""

    def enqueue(self, *, request: RequestLike) -> str: ...

    def cache_get(self, *, request: RequestLike) -> Mapping[str, object] | None: ...

    def cache_set(self, *, request: RequestLike) -> None: ...


@runtime_checkable
class ModelAdapter(Protocol):
    """Generate from aggregate-safe requests without exposing provider SDK types."""

    def generate(self, *, request: RequestLike) -> str: ...


@runtime_checkable
class EmbeddingAdapter(Protocol):
    """Embed aggregate-safe requests without exposing provider SDK types."""

    def embed(self, *, request: RequestLike) -> tuple[float, ...]: ...


@runtime_checkable
class OTelAdapter(Protocol):
    """Emit aggregate-safe telemetry without exposing OTel SDK types."""

    def emit(self, *, request: RequestLike) -> None: ...


def _disabled(request: RequestLike, *, expected_operation: str) -> NoReturn:
    try:
        _validated_request(request, expected_operation=expected_operation)
        _raise_unlinked(IntegrationDisabledError, _DISABLED_ERROR)
    finally:
        del request


class DisabledGitHubAdapter:
    """GitHub integration that validates and then performs no external operation."""

    def publish(self, *, request: RequestLike) -> str:
        try:
            _disabled(request, expected_operation="publish_summary")
        finally:
            del request


class DisabledRedisAdapter:
    """Redis queue/cache integration that never imports or connects to Redis."""

    def enqueue(self, *, request: RequestLike) -> str:
        try:
            _disabled(request, expected_operation="enqueue_summary")
        finally:
            del request

    def cache_get(self, *, request: RequestLike) -> Mapping[str, object] | None:
        try:
            _disabled(request, expected_operation="get_summary_cache")
        finally:
            del request

    def cache_set(self, *, request: RequestLike) -> None:
        try:
            _disabled(request, expected_operation="set_summary_cache")
        finally:
            del request


class DisabledModelAdapter:
    """Model integration that never imports or invokes a provider SDK."""

    def generate(self, *, request: RequestLike) -> str:
        try:
            _disabled(request, expected_operation="generate_summary")
        finally:
            del request


class DisabledEmbeddingAdapter:
    """Embedding integration that never imports or invokes a provider SDK."""

    def embed(self, *, request: RequestLike) -> tuple[float, ...]:
        try:
            _disabled(request, expected_operation="embed_summary")
        finally:
            del request


class DisabledOTelAdapter:
    """Telemetry integration that never imports or invokes an OTel SDK."""

    def emit(self, *, request: RequestLike) -> None:
        try:
            _disabled(request, expected_operation="emit_summary_metrics")
        finally:
            del request


@dataclass(frozen=True, slots=True)
class IntegrationBundle:
    """All optional external capabilities, disabled unless replaced explicitly."""

    github: GitHubAdapter
    redis: RedisAdapter
    model: ModelAdapter
    embedding: EmbeddingAdapter
    telemetry: OTelAdapter


def default_integrations(*, enabled: bool = False) -> IntegrationBundle:
    """Return disabled adapters even when unavailable enablement is requested."""

    if not isinstance(enabled, bool):
        _raise_unlinked(IntegrationPayloadError, _PAYLOAD_ERROR)
    return IntegrationBundle(
        github=DisabledGitHubAdapter(),
        redis=DisabledRedisAdapter(),
        model=DisabledModelAdapter(),
        embedding=DisabledEmbeddingAdapter(),
        telemetry=DisabledOTelAdapter(),
    )


def create_integration_bundle(*, enabled: bool = False) -> IntegrationBundle:
    """Explicit factory alias that preserves disabled/offline behavior."""

    return default_integrations(enabled=enabled)


def _validated_request(
    request: RequestLike,
    *,
    expected_operation: str,
) -> IntegrationRequest:
    validated: IntegrationRequest | None = None
    try:
        validated = IntegrationRequest.safe_validate(request)
        if validated.operation != expected_operation:
            _raise_unlinked(IntegrationPayloadError, _PAYLOAD_ERROR)
        return validated
    finally:
        del request, validated


def _validate_operation_payload(
    operation: str,
    value: Mapping[str, object],
) -> dict[str, object]:
    _preflight_operation_payload(operation, value, expected_payload_type=dict)
    _, specification = _preflight_operation(operation)
    mapping = cast(Mapping[str, object], value)
    normalized: dict[str, object] = {}
    for field_name, field_type in specification.items():
        field_value = mapping[field_name]
        normalized[field_name] = (
            tuple(cast(list[object] | tuple[object, ...], field_value))
            if field_type == "digest_tuple"
            else field_value
        )
    return normalized


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _freeze_value(item)
                for key, item in sorted(value.items(), key=lambda pair: pair[0])
            }
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_value(item) for item in value)
    return value


def _jsonable_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _jsonable_value(item)
            for key, item in sorted(value.items(), key=lambda pair: pair[0])
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable_value(item) for item in value]
    return value


def _assert_no_secret_material(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            if normalized in _SECRET_KEYS:
                raise ValueError
            _assert_no_secret_material(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _assert_no_secret_material(item)
        return
    if isinstance(value, str) and _SECRET_VALUE.search(value):
        raise ValueError


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


LLMAdapter = ModelAdapter
TelemetryAdapter = OTelAdapter
DisabledLLMAdapter = DisabledModelAdapter
DisabledTelemetryAdapter = DisabledOTelAdapter

__all__ = [
    "DisabledEmbeddingAdapter",
    "DisabledGitHubAdapter",
    "DisabledLLMAdapter",
    "DisabledModelAdapter",
    "DisabledOTelAdapter",
    "DisabledRedisAdapter",
    "DisabledTelemetryAdapter",
    "EmbeddingAdapter",
    "GitHubAdapter",
    "IntegrationBundle",
    "IntegrationDisabledError",
    "IntegrationPayloadError",
    "IntegrationRequest",
    "LLMAdapter",
    "ModelAdapter",
    "OTelAdapter",
    "RedisAdapter",
    "RequestLike",
    "TelemetryAdapter",
    "create_integration_bundle",
    "default_integrations",
]
