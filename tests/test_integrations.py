from __future__ import annotations

import ast
import inspect
import socket
import sys
import threading
from collections.abc import Callable, Mapping

import pytest
from pydantic import ValidationError

import riskprobe.integrations.protocols as protocols_module
from riskprobe.integrations import (
    DisabledEmbeddingAdapter,
    DisabledGitHubAdapter,
    DisabledModelAdapter,
    DisabledOTelAdapter,
    DisabledRedisAdapter,
    EmbeddingAdapter,
    GitHubAdapter,
    IntegrationDisabledError,
    IntegrationPayloadError,
    IntegrationRequest,
    ModelAdapter,
    OTelAdapter,
    RedisAdapter,
    default_integrations,
)


def _request() -> IntegrationRequest:
    return IntegrationRequest(
        request_id="request-001",
        operation="publish_summary",
        payload={"finding_count": 2, "status": "stable"},
    )


def _disabled_calls() -> tuple[Callable[[], object], ...]:
    bundle = default_integrations()
    digest_a = "a" * 64
    digest_b = "b" * 64
    requests = {
        "embed": IntegrationRequest(
            request_id="request-001",
            operation="embed_summary",
            payload={"document_ids": (digest_a, digest_b)},
        ),
        "emit": IntegrationRequest(
            request_id="request-002",
            operation="emit_summary_metrics",
            payload={"finding_count": 2, "status": "stable"},
        ),
        "enqueue": IntegrationRequest(
            request_id="request-003",
            operation="enqueue_summary",
            payload={"finding_count": 2, "summary_id": digest_a},
        ),
        "generate": IntegrationRequest(
            request_id="request-004",
            operation="generate_summary",
            payload={"finding_count": 2, "status": "stable"},
        ),
        "get": IntegrationRequest(
            request_id="request-005",
            operation="get_summary_cache",
            payload={"cache_id": digest_a},
        ),
        "publish": _request(),
        "set": IntegrationRequest(
            request_id="request-006",
            operation="set_summary_cache",
            payload={"cache_id": digest_a, "summary_hash": digest_b},
        ),
    }
    return (
        lambda: bundle.github.publish(request=requests["publish"]),
        lambda: bundle.redis.enqueue(request=requests["enqueue"]),
        lambda: bundle.redis.cache_get(request=requests["get"]),
        lambda: bundle.redis.cache_set(request=requests["set"]),
        lambda: bundle.model.generate(request=requests["generate"]),
        lambda: bundle.embedding.embed(request=requests["embed"]),
        lambda: bundle.telemetry.emit(request=requests["emit"]),
    )


def test_integration_request_is_strict_frozen_extra_forbid_and_content_addressed() -> None:
    first = _request()
    second = IntegrationRequest(
        request_id="request-001",
        operation="publish_summary",
        payload={"status": "stable", "finding_count": 2},
    )

    assert first == second
    assert len(first.content_hash) == 64

    with pytest.raises(ValidationError):
        IntegrationRequest.model_validate(
            {
                "request_id": "request-001",
                "operation": "publish_summary",
                "payload": {"finding_count": 2},
                "extra": True,
            }
        )
    with pytest.raises(ValidationError):
        IntegrationRequest.model_validate(
            {
                "request_id": 1,
                "operation": "publish_summary",
                "payload": {"finding_count": 2},
            }
        )
    with pytest.raises(ValidationError):
        first.operation = "other"  # type: ignore[misc]


def test_default_bundle_uses_runtime_checkable_disabled_adapters() -> None:
    bundle = default_integrations()

    assert isinstance(bundle.github, DisabledGitHubAdapter)
    assert isinstance(bundle.redis, DisabledRedisAdapter)
    assert isinstance(bundle.model, DisabledModelAdapter)
    assert isinstance(bundle.embedding, DisabledEmbeddingAdapter)
    assert isinstance(bundle.telemetry, DisabledOTelAdapter)
    assert isinstance(bundle.github, GitHubAdapter)
    assert isinstance(bundle.redis, RedisAdapter)
    assert isinstance(bundle.model, ModelAdapter)
    assert isinstance(bundle.embedding, EmbeddingAdapter)
    assert isinstance(bundle.telemetry, OTelAdapter)


def test_every_disabled_operation_raises_the_same_fixed_error_without_cause() -> None:
    for call in _disabled_calls():
        with pytest.raises(IntegrationDisabledError) as exc_info:
            call()
        assert str(exc_info.value) == "integration is disabled"
        assert exc_info.value.__cause__ is None


@pytest.mark.parametrize(
    "unsafe_request",
    [
        {
            "request_id": "request-001",
            "operation": "publish_summary",
            "payload": {"source_path": "/private/company.csv"},
        },
        {
            "request_id": "request-001",
            "operation": "publish_summary",
            "payload": {"raw_rows": [{"customer": "private"}]},
        },
        {
            "request_id": "request-001",
            "operation": "publish_summary",
            "payload": {"segment_label": "north"},
        },
        {
            "request_id": "request-001",
            "operation": "publish_summary",
            "payload": {"api_key": "sk-live-private-marker"},
        },
    ],
)
def test_unsafe_adapter_requests_raise_fixed_non_leaking_payload_error(
    unsafe_request: dict[str, object],
) -> None:
    adapter = DisabledGitHubAdapter()

    with pytest.raises(IntegrationPayloadError) as exc_info:
        adapter.publish(request=unsafe_request)

    assert str(exc_info.value) == "integration payload is not safe"
    assert "private" not in str(exc_info.value).lower()
    assert exc_info.value.__cause__ is None


def test_all_disabled_adapters_validate_before_reporting_disabled() -> None:
    unsafe = {
        "request_id": "request-001",
        "operation": "publish_summary",
        "payload": {"password": "private-marker"},
    }
    bundle = default_integrations()
    calls = (
        lambda: bundle.github.publish(request=unsafe),
        lambda: bundle.redis.enqueue(request=unsafe),
        lambda: bundle.redis.cache_get(request=unsafe),
        lambda: bundle.redis.cache_set(request=unsafe),
        lambda: bundle.model.generate(request=unsafe),
        lambda: bundle.embedding.embed(request=unsafe),
        lambda: bundle.telemetry.emit(request=unsafe),
    )

    for call in calls:
        with pytest.raises(IntegrationPayloadError) as exc_info:
            call()
        assert str(exc_info.value) == "integration payload is not safe"
        assert exc_info.value.__cause__ is None


def test_explicit_enabled_flag_still_never_creates_real_integrations() -> None:
    bundle = default_integrations(enabled=True)

    assert isinstance(bundle.github, DisabledGitHubAdapter)
    assert isinstance(bundle.redis, DisabledRedisAdapter)
    assert isinstance(bundle.model, DisabledModelAdapter)
    assert isinstance(bundle.embedding, DisabledEmbeddingAdapter)
    assert isinstance(bundle.telemetry, DisabledOTelAdapter)


def test_protocol_module_imports_no_external_sdk() -> None:
    tree = ast.parse(inspect.getsource(protocols_module))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    assert imported_roots <= {
        "__future__",
        "collections",
        "dataclasses",
        "hashlib",
        "json",
        "re",
        "threading",
        "types",
        "typing",
        "weakref",
        "pydantic",
        "riskprobe",
    }
    forbidden_prefixes = ("github", "redis", "openai", "anthropic", "opentelemetry")
    before = set(sys.modules)
    default_integrations()
    newly_imported = set(sys.modules) - before
    assert not any(name.startswith(forbidden_prefixes) for name in newly_imported)


def test_factory_and_disabled_calls_are_offline_and_create_no_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    before = {thread.ident for thread in threading.enumerate()}

    default_integrations()
    for call in _disabled_calls():
        with pytest.raises(IntegrationDisabledError):
            call()

    after = {thread.ident for thread in threading.enumerate()}
    assert after == before


def test_integration_request_snapshots_bounded_tuple_payload() -> None:
    document_ids = ["a" * 64, "b" * 64]
    request = IntegrationRequest(
        request_id="request-001",
        operation="embed_summary",
        payload={"document_ids": document_ids},
    )

    document_ids.append("c" * 64)

    assert request.model_dump(mode="json")["payload"] == {
        "document_ids": ["a" * 64, "b" * 64]
    }


def _exception_graph(error: BaseException) -> tuple[BaseException, ...]:
    pending = [error]
    seen: set[int] = set()
    result: list[BaseException] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        result.append(current)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return tuple(result)


def test_integration_safe_validate_rechecks_mapping_and_model_construct() -> None:
    safe = IntegrationRequest.safe_validate(
        {
            "request_id": "request-001",
            "operation": "publish_summary",
            "payload": {"finding_count": 2, "status": "stable"},
        }
    )
    assert safe.payload["finding_count"] == 2
    assert safe.payload["status"] == "stable"

    marker = "sk-live-private-marker"
    forged = IntegrationRequest.model_construct(
        request_id="request-001",
        operation="publish_summary",
        payload={"status": marker},
        content_hash="f" * 64,
    )
    for request in (
        {
            "request_id": "request-001",
            "operation": "publish_summary",
            "payload": {"status": marker},
        },
        forged,
    ):
        with pytest.raises(IntegrationPayloadError) as exc_info:
            DisabledGitHubAdapter().publish(request=request)
        assert str(exc_info.value) == "integration payload is not safe"
        assert _exception_graph(exc_info.value) == (exc_info.value,)
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        assert marker not in str(exc_info.value)


def test_disabled_error_has_no_recursive_exception_chain() -> None:
    with pytest.raises(IntegrationDisabledError) as exc_info:
        DisabledGitHubAdapter().publish(request=_request())

    assert _exception_graph(exc_info.value) == (exc_info.value,)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_integration_payload_is_deeply_immutable() -> None:
    request = IntegrationRequest.safe_validate(
        {
            "request_id": "request-001",
            "operation": "embed_summary",
            "payload": {
                "document_ids": ["a" * 64, "b" * 64],
            },
        }
    )

    with pytest.raises(TypeError):
        request.payload["new"] = "value"  # type: ignore[index]
    document_ids = request.payload["document_ids"]
    assert document_ids == ("a" * 64, "b" * 64)
    with pytest.raises(TypeError):
        document_ids[0] = "c" * 64  # type: ignore[index]


def test_create_factory_enabled_true_is_offline_thread_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from riskprobe.integrations import create_integration_bundle

    def blocked(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    before = {thread.ident for thread in threading.enumerate()}

    bundle = create_integration_bundle(enabled=True)

    assert isinstance(bundle.github, DisabledGitHubAdapter)
    assert isinstance(bundle.redis, DisabledRedisAdapter)
    assert isinstance(bundle.model, DisabledModelAdapter)
    assert isinstance(bundle.embedding, DisabledEmbeddingAdapter)
    assert isinstance(bundle.telemetry, DisabledOTelAdapter)
    assert {thread.ident for thread in threading.enumerate()} == before


@pytest.mark.parametrize(
    "secret_text",
    [
        "The API key ordinarysecret rotates quarterly.",
        'Use the quoted key "api_key": "ordinarysecret".',
    ],
)
def test_integration_rejects_natural_language_and_quoted_secret_material(
    secret_text: str,
) -> None:
    request = {
        "request_id": "request-001",
        "operation": "publish_summary",
        "payload": {"status": secret_text},
    }

    with pytest.raises(IntegrationPayloadError) as exc_info:
        DisabledGitHubAdapter().publish(request=request)

    assert str(exc_info.value) == "integration payload is not safe"
    assert _exception_graph(exc_info.value) == (exc_info.value,)
    assert exc_info.value.__context__ is None
    assert "ordinarysecret" not in str(exc_info.value)


@pytest.mark.parametrize(
    "request_case",
    [
        {
            "request_id": "request-001",
            "operation": "unknown_safe_operation",
            "payload": {"finding_count": 2, "status": "stable"},
        },
        {
            "request_id": "request-001",
            "operation": "publish_summary",
            "payload": {"client_secret": "ordinary-private-marker"},
        },
        {
            "request_id": "request-001",
            "operation": "publish_summary",
            "payload": {"status": "Loaded from /private/company.csv"},
        },
        {
            "request_id": "request-001",
            "operation": "publish_summary",
            "payload": {"row_snapshot": {"name": "alice", "count": 1}},
        },
        {
            "request_id": "request-001",
            "operation": "publish_summary",
            "payload": {"cohort_label": "north_region"},
        },
        {
            "request_id": "request-001",
            "operation": "publish_summary",
            "payload": {"status": "Review with calculate_risk(value)"},
        },
        {
            "request_id": "request-001",
            "operation": "publish_summary",
            "payload": {"finding_count": True, "status": "stable"},
        },
        {
            "request_id": "request-001",
            "operation": "publish_summary",
            "payload": {"finding_count": 2, "status": "stable", "extra": 1},
        },
    ],
)
def test_review_regression_integration_request_uses_operation_specific_schema(
    request_case: dict[str, object],
) -> None:
    with pytest.raises(IntegrationPayloadError) as exc_info:
        IntegrationRequest.safe_validate(request_case)

    assert str(exc_info.value) == "integration payload is not safe"
    assert _exception_graph(exc_info.value) == (exc_info.value,)
    assert "private-marker" not in str(exc_info.value)


def test_review_regression_integration_allows_only_approved_typed_operations() -> None:
    digest_a = "a" * 64
    digest_b = "b" * 64
    requests = (
        {
            "request_id": "request-001",
            "operation": "publish_summary",
            "payload": {"finding_count": 2, "status": "stable"},
        },
        {
            "request_id": "request-002",
            "operation": "enqueue_summary",
            "payload": {"summary_id": digest_a, "finding_count": 2},
        },
        {
            "request_id": "request-003",
            "operation": "get_summary_cache",
            "payload": {"cache_id": digest_a},
        },
        {
            "request_id": "request-004",
            "operation": "set_summary_cache",
            "payload": {"cache_id": digest_a, "summary_hash": digest_b},
        },
        {
            "request_id": "request-005",
            "operation": "generate_summary",
            "payload": {"finding_count": 2, "status": "stable"},
        },
        {
            "request_id": "request-006",
            "operation": "embed_summary",
            "payload": {"document_ids": (digest_a, digest_b)},
        },
        {
            "request_id": "request-007",
            "operation": "emit_summary_metrics",
            "payload": {"finding_count": 2, "status": "stable"},
        },
    )

    for request in requests:
        validated = IntegrationRequest.safe_validate(request)
        assert len(validated.content_hash) == 64


def test_review_regression_custom_mapping_is_rejected_without_iteration() -> None:
    class ActiveMapping(Mapping[str, object]):
        def __init__(self) -> None:
            self.iterations = 0

        def __getitem__(self, key: str) -> object:
            raise KeyError(key)

        def __iter__(self):  # type: ignore[no-untyped-def]
            self.iterations += 1
            raise AssertionError("custom mapping executed")

        def __len__(self) -> int:
            return 3

    request = ActiveMapping()

    with pytest.raises(IntegrationPayloadError, match="integration payload is not safe"):
        IntegrationRequest.safe_validate(request)

    assert request.iterations == 0


def test_review_regression_fresh_import_has_no_sdk_network_or_thread_side_effects() -> None:
    import os
    import subprocess
    import textwrap
    from pathlib import Path

    project = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import builtins
        import socket
        import threading

        forbidden = {"anthropic", "github", "openai", "opentelemetry", "redis"}
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in forbidden:
                raise AssertionError(f"vendor SDK import attempted: {name}")
            return original_import(name, *args, **kwargs)

        def blocked(*args, **kwargs):
            raise AssertionError("network or thread side effect attempted")

        builtins.__import__ = guarded_import
        socket.getaddrinfo = blocked
        socket.create_connection = blocked
        socket.socket.connect = blocked
        socket.socket.connect_ex = blocked
        socket.socket.sendto = blocked
        threading.Thread.start = blocked

        import riskprobe.integrations as integrations

        first = integrations.default_integrations(enabled=True)
        second = integrations.create_integration_bundle(enabled=True)
        assert type(first.github).__name__ == "DisabledGitHubAdapter"
        assert type(second.telemetry).__name__ == "DisabledOTelAdapter"
        """
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project / "src")

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "request_id",
    [
        "request-customer-123456",
        "request-client-secret-swordfish",
        "request-north-region",
    ],
)
def test_review_followup_integration_request_id_must_be_opaque(
    request_id: str,
) -> None:
    with pytest.raises(IntegrationPayloadError, match="integration payload is not safe"):
        IntegrationRequest.safe_validate(
            {
                "request_id": request_id,
                "operation": "publish_summary",
                "payload": {"finding_count": 2, "status": "stable"},
            }
        )


def test_review_followup_adapter_rejects_operation_mismatch_before_disabled() -> None:
    request = {
        "request_id": "request-001",
        "operation": "publish_summary",
        "payload": {"finding_count": 2, "status": "stable"},
    }

    with pytest.raises(IntegrationPayloadError) as exc_info:
        default_integrations().embedding.embed(request=request)

    assert str(exc_info.value) == "integration payload is not safe"
    assert _exception_graph(exc_info.value) == (exc_info.value,)


def test_review_followup_request_id_is_derived_from_typed_content() -> None:
    payload = {"finding_count": 2, "status": "stable"}
    first = IntegrationRequest(
        request_id="request-001",
        operation="publish_summary",
        payload=payload,
    )
    second = IntegrationRequest(
        request_id="request-999",
        operation="publish_summary",
        payload=payload,
    )

    assert first.request_id == second.request_id
    assert first.content_hash == second.content_hash

    with pytest.raises(IntegrationPayloadError, match="integration payload is not safe"):
        IntegrationRequest.safe_validate(
            {
                "request_id": f"request-{'a' * 24}",
                "operation": "publish_summary",
                "payload": payload,
            }
        )
    with pytest.raises(IntegrationPayloadError, match="integration payload is not safe"):
        IntegrationRequest.safe_validate(
            {
                "request_id": "request-123456789",
                "operation": "publish_summary",
                "payload": payload,
            }
        )


def test_review_important_integration_preflight_runs_before_validation_and_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_calls = 0
    tuple_calls = 0

    def active_validation(
        cls: type[IntegrationRequest],
        value: object,
    ) -> IntegrationRequest:
        nonlocal validation_calls
        del cls, value
        validation_calls += 1
        raise AssertionError("model validation executed")

    def active_tuple(value: object) -> tuple[object, ...]:
        nonlocal tuple_calls
        del value
        tuple_calls += 1
        raise AssertionError("tuple snapshot executed")

    monkeypatch.setattr(
        IntegrationRequest,
        "model_validate",
        classmethod(active_validation),
    )
    with pytest.raises(IntegrationPayloadError, match="integration payload is not safe"):
        IntegrationRequest.safe_validate(
            {
                "request_id": "request-001",
                "operation": "publish_summary",
                "payload": {"finding_count": 2, "status": "stable"},
                **{f"extra-{index}": index for index in range(32)},
            }
        )
    assert validation_calls == 0

    monkeypatch.undo()
    monkeypatch.setattr(protocols_module, "tuple", active_tuple, raising=False)
    with pytest.raises(IntegrationPayloadError, match="integration payload is not safe"):
        IntegrationRequest.safe_validate(
            {
                "request_id": "request-001",
                "operation": "embed_summary",
                "payload": {"document_ids": ["a" * 64] * 257},
            }
        )
    assert tuple_calls == 0


def test_review_important_integration_capability_rejects_construct_and_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import MappingProxyType

    valid = _request()
    forged = IntegrationRequest.model_construct(
        request_id=valid.request_id,
        operation=valid.operation,
        payload=MappingProxyType({"finding_count": 2, "status": "stable"}),
        content_hash=valid.content_hash,
    )
    copied = valid.model_copy()
    dump_calls = 0
    original_dump = IntegrationRequest.model_dump

    def counted_dump(self: IntegrationRequest, *args: object, **kwargs: object) -> object:
        nonlocal dump_calls
        dump_calls += 1
        return original_dump(self, *args, **kwargs)

    monkeypatch.setattr(IntegrationRequest, "model_dump", counted_dump)

    for request in (forged, copied):
        with pytest.raises(
            IntegrationPayloadError,
            match="integration payload is not safe",
        ):
            DisabledGitHubAdapter().publish(request=request)

    assert dump_calls == 0
    with pytest.raises(IntegrationDisabledError, match="integration is disabled"):
        DisabledGitHubAdapter().publish(request=valid)
    assert dump_calls == 2


def test_review_important_forged_active_model_payload_is_never_iterated() -> None:
    class ActiveMapping(Mapping[str, object]):
        def __init__(self) -> None:
            self.iterations = 0
            self.getitems = 0

        def __getitem__(self, key: str) -> object:
            self.getitems += 1
            raise KeyError(key)

        def __iter__(self):  # type: ignore[no-untyped-def]
            self.iterations += 1
            raise AssertionError("active payload iterated")

        def __len__(self) -> int:
            return 2

    active = ActiveMapping()
    valid = _request()
    forged = IntegrationRequest.model_construct(
        request_id=valid.request_id,
        operation=valid.operation,
        payload=active,
        content_hash=valid.content_hash,
    )
    copied = valid.model_copy(update={"payload": active})

    for request in (forged, copied):
        with pytest.raises(
            IntegrationPayloadError,
            match="integration payload is not safe",
        ):
            DisabledGitHubAdapter().publish(request=request)

    assert active.iterations == 0
    assert active.getitems == 0


def _integration_traceback_value_contains_rejected(
    value: object,
    *,
    rejected_ids: set[int],
    markers: tuple[str, ...],
    seen: set[int],
    depth: int = 0,
) -> bool:
    if id(value) in rejected_ids:
        return True
    if type(value) is str:
        return any(marker in value for marker in markers)
    if isinstance(value, BaseException):
        return any(marker in str(value) for marker in markers) or (
            _integration_traceback_value_contains_rejected(
                value.args,
                rejected_ids=rejected_ids,
                markers=markers,
                seen=seen,
                depth=depth + 1,
            )
        )
    if depth >= 5 or id(value) in seen:
        return False
    seen.add(id(value))
    if type(value) is dict:
        items = tuple(value.items())[:512]
        return any(
            _integration_traceback_value_contains_rejected(
                item,
                rejected_ids=rejected_ids,
                markers=markers,
                seen=seen,
                depth=depth + 1,
            )
            for pair in items
            for item in pair
        )
    if type(value) in {list, tuple, set, frozenset}:
        return any(
            _integration_traceback_value_contains_rejected(
                item,
                rejected_ids=rejected_ids,
                markers=markers,
                seen=seen,
                depth=depth + 1,
            )
            for item in tuple(value)[:512]
        )
    return False


def _assert_integration_traceback_locals_clean(
    error: BaseException,
    *,
    rejected: tuple[object, ...],
    markers: tuple[str, ...],
) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None
    target_frames = []
    traceback = error.__traceback__
    while traceback is not None:
        module_name = traceback.tb_frame.f_globals.get("__name__", "")
        if module_name == "riskprobe.integrations" or module_name.startswith(
            "riskprobe.integrations."
        ):
            target_frames.append(traceback.tb_frame)
        traceback = traceback.tb_next
    assert target_frames
    rejected_ids = {id(value) for value in rejected}
    for frame in target_frames:
        for local_name, value in frame.f_locals.items():
            assert not _integration_traceback_value_contains_rejected(
                value,
                rejected_ids=rejected_ids,
                markers=markers,
                seen=set(),
            ), f"{frame.f_code.co_name}.{local_name} retained rejected input"


def _capture_integration_error(call: Callable[[], object]) -> BaseException:
    try:
        call()
    except BaseException as error:
        return error
    raise AssertionError("expected integration operation to fail")


def test_review_important_integration_fixed_errors_scrub_target_traceback_locals() -> None:
    marker = "integration-private-trace-marker"
    request = {
        "request_id": "request-001",
        "operation": "publish_summary",
        "payload": {"finding_count": 2, "status": marker},
    }

    direct_error = _capture_integration_error(
        lambda: IntegrationRequest.safe_validate(request)
    )
    assert isinstance(direct_error, IntegrationPayloadError)
    _assert_integration_traceback_locals_clean(
        direct_error,
        rejected=(request,),
        markers=(marker,),
    )

    adapter_error = _capture_integration_error(
        lambda: DisabledGitHubAdapter().publish(request=request)
    )
    assert isinstance(adapter_error, IntegrationPayloadError)
    _assert_integration_traceback_locals_clean(
        adapter_error,
        rejected=(request,),
        markers=(marker,),
    )

    class ActiveMapping(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise KeyError(key)

        def __iter__(self):  # type: ignore[no-untyped-def]
            raise AssertionError("active integration mapping executed")

        def __len__(self) -> int:
            return 2

    active = ActiveMapping()
    valid = _request()
    forged = valid.model_copy(update={"payload": active})
    forged_error = _capture_integration_error(
        lambda: DisabledGitHubAdapter().publish(request=forged)
    )
    assert isinstance(forged_error, IntegrationPayloadError)
    _assert_integration_traceback_locals_clean(
        forged_error,
        rejected=(forged, active),
        markers=(marker,),
    )


def test_review_followup_integration_dict_keys_are_exact_before_lookup() -> None:
    class ActiveCollisionKey:
        def __init__(self, target: str) -> None:
            self.target = target
            self.hashes = 0
            self.comparisons = 0

        def __hash__(self) -> int:
            self.hashes += 1
            return hash(self.target)

        def __eq__(self, other: object) -> bool:
            del other
            self.comparisons += 1
            return False

        def reset(self) -> None:
            self.hashes = 0
            self.comparisons = 0

    outer_key = ActiveCollisionKey("operation")
    outer_request = {
        "request_id": "request-001",
        outer_key: "publish_summary",
        "payload": {"finding_count": 2, "status": "stable"},
    }
    outer_key.reset()

    payload_key = ActiveCollisionKey("status")
    payload = {
        "finding_count": 2,
        payload_key: "stable",
    }
    payload_request = {
        "request_id": "request-001",
        "operation": "publish_summary",
        "payload": payload,
    }
    payload_key.reset()

    for request in (outer_request, payload_request):
        with pytest.raises(
            IntegrationPayloadError,
            match="integration payload is not safe",
        ):
            IntegrationRequest.safe_validate(request)

    for key in (outer_key, payload_key):
        assert key.hashes == 0
        assert key.comparisons == 0


class _ActiveIntegrationString(str):
    def __new__(cls, value: str) -> _ActiveIntegrationString:
        instance = super().__new__(cls, value)
        instance.hashes = 0
        instance.comparisons = 0
        return instance

    def __hash__(self) -> int:
        self.hashes += 1
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        self.comparisons += 1
        return str(self) == other

    def reset(self) -> None:
        self.hashes = 0
        self.comparisons = 0


def test_review_followup_payload_strings_are_exact_before_allowlist_lookup() -> None:
    status = _ActiveIntegrationString("stable")

    with pytest.raises(
        IntegrationPayloadError,
        match="integration payload is not safe",
    ):
        IntegrationRequest.safe_validate(
            {
                "request_id": "request-001",
                "operation": "publish_summary",
                "payload": {"finding_count": 2, "status": status},
            }
        )

    assert status.hashes == 0
    assert status.comparisons == 0


def test_review_followup_registered_request_fields_are_exact_before_comparison() -> None:
    request = _request()
    operation = _ActiveIntegrationString(request.operation)
    object.__setattr__(request, "operation", operation)
    operation.reset()

    with pytest.raises(
        IntegrationPayloadError,
        match="integration payload is not safe",
    ):
        IntegrationRequest.safe_validate(request)

    assert operation.hashes == 0
    assert operation.comparisons == 0


@pytest.mark.parametrize(
    "candidate",
    [
        pytest.param(
            {
                "request_id": "r" * 33,
                "operation": "publish_summary",
                "payload": {"finding_count": 2, "status": "stable"},
            },
            id="request-id",
        ),
        pytest.param(
            {
                "request_id": "request-001",
                "operation": "o" * 33,
                "payload": {},
            },
            id="operation",
        ),
        pytest.param(
            {
                "request_id": "request-001",
                "operation": "publish_summary",
                "payload": {"finding_count": 2, "status": "s" * 17},
            },
            id="status",
        ),
        pytest.param(
            {
                "request_id": "request-001",
                "operation": "enqueue_summary",
                "payload": {"finding_count": 2, "summary_id": "d" * 65},
            },
            id="digest",
        ),
        pytest.param(
            {
                "request_id": "request-001",
                "operation": "embed_summary",
                "payload": {"document_ids": ("d" * 65,)},
            },
            id="digest-tuple",
        ),
        pytest.param(
            {
                "request_id": "request-001",
                "operation": "publish_summary",
                "payload": {"finding_count": True, "status": "stable"},
            },
            id="count",
        ),
        pytest.param(
            {
                "request_id": "request-001",
                "operation": "publish_summary",
                "payload": {"finding_count": 2, "status": "stable"},
                "content_hash": "h" * 65,
            },
            id="content-hash",
        ),
    ],
)
def test_task1_scalar_integration_preflight_precedes_model_validation(
    monkeypatch: pytest.MonkeyPatch,
    candidate: dict[str, object],
) -> None:
    validation_calls = 0

    def active_validation(
        cls: type[IntegrationRequest],
        value: object,
    ) -> IntegrationRequest:
        nonlocal validation_calls
        del cls, value
        validation_calls += 1
        raise AssertionError("model validation executed")

    monkeypatch.setattr(
        IntegrationRequest,
        "model_validate",
        classmethod(active_validation),
    )
    with pytest.raises(IntegrationPayloadError, match="integration payload is not safe"):
        IntegrationRequest.safe_validate(candidate)
    assert validation_calls == 0


def test_task1_scalar_integration_lengths_precede_lookup_and_digest_regex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ActiveSpecs(dict[str, dict[str, str]]):
        def __init__(self) -> None:
            super().__init__(protocols_module._OPERATION_SPECS)
            self.get_calls = 0

        def get(
            self,
            key: str,
            default: object = None,
        ) -> dict[str, str] | object:
            self.get_calls += 1
            return super().get(key, default)

    class ActivePattern:
        def __init__(self) -> None:
            self.calls = 0

        def fullmatch(self, value: str) -> None:
            del value
            self.calls += 1
            raise AssertionError("digest regex executed")

    specifications = ActiveSpecs()
    monkeypatch.setattr(protocols_module, "_OPERATION_SPECS", specifications)
    with pytest.raises(IntegrationPayloadError, match="integration payload is not safe"):
        IntegrationRequest.safe_validate(
            {
                "request_id": "request-001",
                "operation": "o" * 33,
                "payload": {},
            }
        )
    assert specifications.get_calls == 0

    digest_pattern = ActivePattern()
    monkeypatch.setattr(protocols_module, "_SHA256", digest_pattern)
    with pytest.raises(IntegrationPayloadError, match="integration payload is not safe"):
        IntegrationRequest.safe_validate(
            {
                "request_id": "request-001",
                "operation": "enqueue_summary",
                "payload": {"finding_count": 2, "summary_id": "d" * 65},
            }
        )
    assert digest_pattern.calls == 0
