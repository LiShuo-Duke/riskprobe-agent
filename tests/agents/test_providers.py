import socket

import pytest

from riskprobe.agents.providers import (
    DeterministicProvider,
    DisabledProvider,
    ModelProvider,
    ProviderDisabledError,
    UnsafeProviderPayloadError,
    default_provider,
)


def test_disabled_provider_is_default_protocol_implementation() -> None:
    provider = default_provider()

    assert isinstance(provider, DisabledProvider)
    assert isinstance(provider, ModelProvider)
    with pytest.raises(ProviderDisabledError, match="model provider is disabled"):
        provider.summarize(objective="comprehensive", evidence=())


def test_deterministic_provider_summarizes_only_safe_aggregate_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def network_forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "getaddrinfo", network_forbidden)
    monkeypatch.setattr(socket.socket, "connect", network_forbidden)
    provider = DeterministicProvider()
    evidence = (
        {
            "evidence_id": "b" * 64,
            "kind": "recommendation",
            "privacy_class": "aggregate",
            "payload": {"recommendation_count": 1, "safe_marker": "do-not-copy-me"},
        },
        {
            "evidence_id": "a" * 64,
            "kind": "diagnosis",
            "privacy_class": "aggregate",
            "payload": {"finding_count": 2},
        },
    )

    first = provider.summarize(objective="comprehensive", evidence=evidence)
    second = provider.summarize(objective="comprehensive", evidence=tuple(reversed(evidence)))

    assert first == second
    assert "aggregate_evidence=2" in first
    assert "diagnosis,recommendation" in first
    assert "do-not-copy-me" not in first


def test_deterministic_provider_rejects_unsafe_or_nonaggregate_evidence() -> None:
    provider = DeterministicProvider()

    with pytest.raises(UnsafeProviderPayloadError, match="provider evidence is not safe"):
        provider.summarize(
            objective="comprehensive",
            evidence=({"privacy_class": "aggregate", "payload": {"path": "/secret"}},),
        )
    with pytest.raises(UnsafeProviderPayloadError, match="provider evidence is not safe"):
        provider.summarize(
            objective="comprehensive",
            evidence=({"privacy_class": "restricted", "payload": {"count": 1}},),
        )
