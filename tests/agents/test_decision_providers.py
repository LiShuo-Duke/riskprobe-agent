import socket
from datetime import UTC, datetime, timedelta

import pytest

from riskprobe.agents.decision_contracts import (
    DecisionContext,
    DecisionFinding,
    DecisionSource,
    default_decision_policy,
)
from riskprobe.agents.decision_providers import (
    DecisionDisposition,
    DecisionProvider,
    DecisionProviderConfig,
    DecisionProviderError,
    DecisionProviderMode,
    DeterministicDecisionProvider,
    DisabledDecisionProvider,
    bind_decision_providers,
    default_decision_provider,
)
from riskprobe.monitoring.models import FindingKind, FindingSeverity, RiskFinding
from riskprobe.recommendations.policy import ActionCode


def _context() -> DecisionContext:
    issued_at = datetime(2026, 1, 1, tzinfo=UTC)
    findings = (
        DecisionFinding(
            evidence_id="b" * 64,
            finding=RiskFinding(
                kind=FindingKind.DATA_QUALITY,
                severity=FindingSeverity.WARNING,
                code="missing_values",
                metrics={"affected_rate": 0.1},
            ),
        ),
        DecisionFinding(
            evidence_id="a" * 64,
            finding=RiskFinding(
                kind=FindingKind.FEATURE_DRIFT,
                severity=FindingSeverity.WARNING,
                code="feature_psi",
                metrics={"psi": 0.25},
            ),
        ),
    )
    return DecisionContext(
        session_id="0123456789abcdef",
        attempt=0,
        anchor_node_id="c" * 64,
        dataset_id="synthetic_demo",
        metadata_grade="B",
        row_count=100,
        feature_count=8,
        diagnosis_evidence_ids=tuple(item.evidence_id for item in findings),
        findings=findings,
        policy=default_decision_policy(),
        issued_at=issued_at,
        expires_at=issued_at + timedelta(minutes=5),
        component_versions={
            "diagnostics": "diagnostics-v1",
            "orchestrator": "orchestrator-v1",
            "planner": "planner-v1",
            "recommendations": "recommendations-v1",
        },
    )


def test_disabled_decision_provider_is_default_fallback() -> None:
    provider = default_decision_provider()
    resolution = provider.resolve(context=_context())

    assert isinstance(provider, DisabledDecisionProvider)
    assert isinstance(provider, DecisionProvider)
    assert resolution.disposition is DecisionDisposition.FALLBACK
    assert resolution.proposal is None


def test_deterministic_decision_provider_is_offline_complete_and_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def network_forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "getaddrinfo", network_forbidden)
    monkeypatch.setattr(socket.socket, "connect", network_forbidden)
    provider = DeterministicDecisionProvider()
    context = _context()

    first = provider.resolve(context=context)
    second = provider.resolve(context=context)

    assert first == second
    assert first.disposition is DecisionDisposition.PROPOSAL
    assert first.proposal is not None
    assert first.proposal.source is DecisionSource.DETERMINISTIC
    assert first.proposal.context_id == context.context_id
    assert first.proposal.diagnosis_evidence_ids == context.diagnosis_evidence_ids
    assert first.proposal.action_codes == (
        ActionCode.INVESTIGATE_FEATURE_DRIFT,
        ActionCode.REMEDIATE_DATA_QUALITY,
    )


def test_deterministic_provider_uses_fixed_fail_closed_error() -> None:
    from riskprobe.agents.decision_contracts import DecisionPolicy
    from riskprobe.agents.decision_providers import DecisionProviderError

    policy = DecisionPolicy(
        allowed_action_codes=(ActionCode.REVIEW_SEGMENT_RISK,),
        grade_b_allowed_action_codes=(ActionCode.REVIEW_SEGMENT_RISK,),
        max_action_count=1,
    )
    base = _context()
    payload = base.model_dump(mode="python", exclude={"context_id"})
    payload["policy"] = policy
    context = DecisionContext.model_validate(payload)

    with pytest.raises(
        DecisionProviderError,
        match="^decision provider could not resolve context$",
    ) as error:
        DeterministicDecisionProvider().resolve(context=context)

    assert isinstance(error.value.__cause__, ValueError)


def test_decision_provider_resolution_rejects_disposition_payload_mismatch() -> None:
    from pydantic import ValidationError

    from riskprobe.agents.decision_providers import DecisionProviderResolution

    proposal = DeterministicDecisionProvider().resolve(context=_context()).proposal
    assert proposal is not None

    with pytest.raises(ValidationError, match="proposal disposition"):
        DecisionProviderResolution(
            disposition=DecisionDisposition.FALLBACK,
            proposal=proposal,
        )
    with pytest.raises(ValidationError, match="proposal disposition"):
        DecisionProviderResolution(disposition=DecisionDisposition.PROPOSAL)


def test_runtime_provider_config_defaults_to_offline_fallback() -> None:
    primary, fallback = bind_decision_providers()

    assert isinstance(primary, DisabledDecisionProvider)
    assert isinstance(fallback, DeterministicDecisionProvider)


def test_runtime_provider_config_binds_deterministic_primary() -> None:
    primary, fallback = bind_decision_providers(
        DecisionProviderConfig(mode=DecisionProviderMode.DETERMINISTIC)
    )

    assert isinstance(primary, DeterministicDecisionProvider)
    assert isinstance(fallback, DeterministicDecisionProvider)
    assert primary is not fallback


def test_runtime_provider_config_requires_matching_external_host() -> None:
    class ExternalHost:
        mode = DecisionProviderMode.EXTERNAL_HOST
        provider_id = "configured-host"
        version = "configured-host-v1"

        def resolve(self, *, context: DecisionContext):
            del context
            raise AssertionError("binding must not invoke the external host")

    provider = ExternalHost()
    config = DecisionProviderConfig(
        mode=DecisionProviderMode.EXTERNAL_HOST,
        provider_id=provider.provider_id,
        provider_version=provider.version,
    )

    primary, fallback = bind_decision_providers(
        config,
        external_provider=provider,
    )

    assert primary is provider
    assert isinstance(fallback, DeterministicDecisionProvider)
    with pytest.raises(
        DecisionProviderError,
        match="^decision provider configuration is invalid$",
    ):
        bind_decision_providers(config)
    with pytest.raises(
        DecisionProviderError,
        match="^decision provider configuration is invalid$",
    ):
        bind_decision_providers(
            config.model_copy(update={"provider_version": "other-host-v1"}),
            external_provider=provider,
        )


def test_runtime_provider_config_rejects_identity_outside_external_mode() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="external host identity"):
        DecisionProviderConfig(
            provider_id="configured-host",
            provider_version="configured-host-v1",
        )
    with pytest.raises(ValidationError, match="external host identity"):
        DecisionProviderConfig(mode=DecisionProviderMode.EXTERNAL_HOST)


def test_decision_provider_symbols_are_publicly_exported() -> None:
    import riskprobe.agents as agents_api
    from riskprobe.agents.decision_providers import DecisionProviderResolution

    expected = {
        "DecisionDisposition": DecisionDisposition,
        "DecisionProvider": DecisionProvider,
        "DecisionProviderConfig": DecisionProviderConfig,
        "DecisionProviderError": DecisionProviderError,
        "DecisionProviderMode": DecisionProviderMode,
        "DecisionProviderResolution": DecisionProviderResolution,
        "DeterministicDecisionProvider": DeterministicDecisionProvider,
        "DisabledDecisionProvider": DisabledDecisionProvider,
        "bind_decision_providers": bind_decision_providers,
        "default_decision_provider": default_decision_provider,
    }
    for name, symbol in expected.items():
        assert getattr(agents_api, name) is symbol
