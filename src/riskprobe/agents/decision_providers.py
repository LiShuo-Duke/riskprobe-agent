"""Offline provider seam for bounded decision proposals."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    model_validator,
)

from riskprobe.agents.decision_contracts import (
    DecisionContext,
    DecisionProposal,
    DecisionSource,
)
from riskprobe.recommendations.policy import applicable_action_codes

_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}$")


class DecisionProviderMode(StrEnum):
    DISABLED = "disabled"
    DETERMINISTIC = "deterministic"
    EXTERNAL_HOST = "external_host"


class DecisionProviderConfig(BaseModel):
    """Strict runtime-only provider composition; never part of run identity."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )

    mode: DecisionProviderMode = DecisionProviderMode.DISABLED
    provider_id: str | None = None
    provider_version: str | None = None

    @field_validator("provider_id", "provider_version")
    @classmethod
    def validate_optional_public_token(cls, value: str | None) -> str | None:
        if value is not None and _VERSION.fullmatch(value) is None:
            raise ValueError("external host identity must use public tokens")
        return value

    @model_validator(mode="after")
    def validate_external_host_identity(self) -> DecisionProviderConfig:
        if self.mode is DecisionProviderMode.EXTERNAL_HOST:
            if self.provider_id is None or self.provider_version is None:
                raise ValueError("external host identity is invalid")
        elif self.provider_id is not None or self.provider_version is not None:
            raise ValueError("external host identity is invalid")
        return self


class DecisionDisposition(StrEnum):
    FALLBACK = "fallback"
    PROPOSAL = "proposal"
    PENDING = "pending"


class DecisionProviderError(RuntimeError):
    """Raised with a fixed message when a provider cannot safely resolve context."""


class _DecisionProviderRole(StrEnum):
    PRIMARY = "primary"
    FALLBACK = "fallback"


class _DecisionProviderIdentity(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )

    provider_id: str
    mode: DecisionProviderMode
    version: str

    @field_validator("provider_id", "version")
    @classmethod
    def validate_public_token(cls, value: str) -> str:
        if _VERSION.fullmatch(value) is None:
            raise ValueError("provider identity must use public tokens")
        return value


class _DecisionProviderBinding(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )

    schema_version: Literal["riskprobe.decision-provider-binding.v1"] = (
        "riskprobe.decision-provider-binding.v1"
    )
    primary: _DecisionProviderIdentity
    fallback: _DecisionProviderIdentity
    selected: _DecisionProviderIdentity
    selected_role: _DecisionProviderRole

    @model_validator(mode="after")
    def validate_selected_provider(self) -> _DecisionProviderBinding:
        expected = (
            self.primary
            if self.selected_role is _DecisionProviderRole.PRIMARY
            else self.fallback
        )
        if (
            self.selected != expected
            or self.fallback.mode is not DecisionProviderMode.DETERMINISTIC
        ):
            raise ValueError("selected provider binding is invalid")
        return self


class DecisionProviderResolution(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )

    disposition: DecisionDisposition
    proposal: DecisionProposal | None = None

    @model_validator(mode="after")
    def validate_resolution(self) -> DecisionProviderResolution:
        if (self.disposition is DecisionDisposition.PROPOSAL) != (
            self.proposal is not None
        ):
            raise ValueError("proposal disposition must match proposal presence")
        return self


@runtime_checkable
class DecisionProvider(Protocol):
    mode: DecisionProviderMode
    provider_id: str
    version: str

    def resolve(self, *, context: DecisionContext) -> DecisionProviderResolution: ...


class DisabledDecisionProvider:
    mode = DecisionProviderMode.DISABLED
    provider_id = "disabled"
    version = "disabled-decision-provider-v1"

    def resolve(self, *, context: DecisionContext) -> DecisionProviderResolution:
        if type(context) is not DecisionContext:
            raise TypeError("context must be a DecisionContext")
        return DecisionProviderResolution(disposition=DecisionDisposition.FALLBACK)


class DeterministicDecisionProvider:
    mode = DecisionProviderMode.DETERMINISTIC
    provider_id = "deterministic"

    def __init__(self, *, version: str = "deterministic-decision-provider-v1") -> None:
        if _VERSION.fullmatch(version) is None:
            raise ValueError("version must be a public version token")
        self.version = version

    def resolve(self, *, context: DecisionContext) -> DecisionProviderResolution:
        if type(context) is not DecisionContext:
            raise TypeError("context must be a DecisionContext")
        try:
            policy = context.policy
            allowed = set(policy.allowed_action_codes)
            if context.metadata_grade == "B":
                allowed.intersection_update(policy.grade_b_allowed_action_codes)
            actions = tuple(
                action
                for action in applicable_action_codes(
                    item.finding for item in context.findings
                )
                if action in allowed
            )[: policy.max_action_count]
            if len(actions) < policy.min_action_count:
                raise ValueError("insufficient applicable actions")
            proposal = DecisionProposal(
                context_id=context.context_id,
                diagnosis_evidence_ids=context.diagnosis_evidence_ids,
                action_codes=actions,
                source=DecisionSource.DETERMINISTIC,
                source_version=self.version,
            )
        except Exception as error:
            raise DecisionProviderError("decision provider could not resolve context") from error
        return DecisionProviderResolution(
            disposition=DecisionDisposition.PROPOSAL,
            proposal=proposal,
        )


def bind_decision_providers(
    config: DecisionProviderConfig | None = None,
    *,
    external_provider: DecisionProvider | None = None,
) -> tuple[DecisionProvider, DecisionProvider]:
    """Bind a runtime config to built-in providers or one injected host adapter."""

    try:
        if config is None:
            canonical_config = DecisionProviderConfig()
        else:
            if type(config) is not DecisionProviderConfig:
                raise TypeError("config must be a DecisionProviderConfig")
            canonical_config = DecisionProviderConfig.model_validate(
                config.model_dump(mode="python")
            )
        fallback = DeterministicDecisionProvider()
        if canonical_config.mode is DecisionProviderMode.DISABLED:
            if external_provider is not None:
                raise ValueError("external provider is not allowed")
            return DisabledDecisionProvider(), fallback
        if canonical_config.mode is DecisionProviderMode.DETERMINISTIC:
            if external_provider is not None:
                raise ValueError("external provider is not allowed")
            return DeterministicDecisionProvider(), fallback
        if external_provider is None or not callable(
            getattr(external_provider, "resolve", None)
        ):
            raise ValueError("external provider is unavailable")
        identity = _DecisionProviderIdentity(
            provider_id=getattr(external_provider, "provider_id", None),
            mode=getattr(external_provider, "mode", None),
            version=getattr(external_provider, "version", None),
        )
        if (
            identity.mode is not DecisionProviderMode.EXTERNAL_HOST
            or identity.provider_id != canonical_config.provider_id
            or identity.version != canonical_config.provider_version
        ):
            raise ValueError("external provider identity does not match")
        return external_provider, fallback
    except Exception as error:
        raise DecisionProviderError(
            "decision provider configuration is invalid"
        ) from error


def default_decision_provider() -> DecisionProvider:
    """Default to no external decision; the orchestrator owns deterministic fallback."""

    return DisabledDecisionProvider()


__all__ = [
    "DecisionDisposition",
    "DecisionProvider",
    "DecisionProviderConfig",
    "DecisionProviderError",
    "DecisionProviderMode",
    "DecisionProviderResolution",
    "DeterministicDecisionProvider",
    "DisabledDecisionProvider",
    "bind_decision_providers",
    "default_decision_provider",
]
