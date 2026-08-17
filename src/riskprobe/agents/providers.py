"""Offline model-provider protocol with disabled and deterministic implementations."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from riskprobe.privacy import assert_safe_payload

_CODE = re.compile(r"^[a-z][a-z0-9_-]{0,127}$")


class ProviderDisabledError(RuntimeError):
    """Raised when the default disabled provider is invoked."""


class UnsafeProviderPayloadError(ValueError):
    """Raised without echoing evidence when provider input is not aggregate-safe."""


@runtime_checkable
class ModelProvider(Protocol):
    """Minimal provider boundary; implementations receive aggregate evidence only."""

    def summarize(
        self,
        *,
        objective: str,
        evidence: Sequence[Mapping[str, object] | BaseModel],
    ) -> str: ...


class DisabledProvider:
    """Default provider that imports no SDK and performs no external operation."""

    def summarize(
        self,
        *,
        objective: str,
        evidence: Sequence[Mapping[str, object] | BaseModel],
    ) -> str:
        del objective, evidence
        raise ProviderDisabledError("model provider is disabled")


class DeterministicProvider:
    """Summarize only evidence counts and kinds after the privacy gate approves input."""

    def summarize(
        self,
        *,
        objective: str,
        evidence: Sequence[Mapping[str, object] | BaseModel],
    ) -> str:
        try:
            if _CODE.fullmatch(objective) is None:
                raise ValueError("unsafe objective")
            assert_safe_payload({"objective": objective})
            kinds: set[str] = set()
            for item in evidence:
                payload = item.model_dump(mode="json") if isinstance(item, BaseModel) else item
                if not isinstance(payload, Mapping):
                    raise TypeError("evidence must be a mapping")
                privacy_class = payload.get("privacy_class", "aggregate")
                if isinstance(privacy_class, Enum):
                    privacy_class = privacy_class.value
                if privacy_class != "aggregate":
                    raise ValueError("non-aggregate evidence")
                assert_safe_payload(payload)
                kind = payload.get("kind", "aggregate")
                if not isinstance(kind, str) or _CODE.fullmatch(kind) is None:
                    raise ValueError("unsafe evidence kind")
                kinds.add(kind)
        except Exception as error:
            raise UnsafeProviderPayloadError("provider evidence is not safe") from error

        kind_summary = ",".join(sorted(kinds)) if kinds else "none"
        return (
            f"objective={objective}; aggregate_evidence={len(evidence)}; "
            f"kinds={kind_summary}"
        )


def default_provider() -> ModelProvider:
    """Return a fresh disabled provider so external access is always opt-in."""

    return DisabledProvider()


__all__ = [
    "DeterministicProvider",
    "DisabledProvider",
    "ModelProvider",
    "ProviderDisabledError",
    "UnsafeProviderPayloadError",
    "default_provider",
]
