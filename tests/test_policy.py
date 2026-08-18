import pytest
from pydantic import ValidationError

from riskprobe.policy import (
    Budget,
    Capability,
    PolicyDeniedError,
    PolicyEngine,
    Principal,
    QueryBudgetExceededError,
    Role,
    ToolCall,
)


def _principal(role: Role) -> Principal:
    return Principal(principal_id=f"test-{role.value}", role=role)


@pytest.mark.parametrize(
    ("role", "capability"),
    [
        (Role.ANALYST, Capability.INSPECT),
        (Role.ANALYST, Capability.DISCOVER),
        (Role.ANALYST, Capability.DIAGNOSE),
        (Role.ANALYST, Capability.RECOMMEND),
        (Role.ANALYST, Capability.EVIDENCE_LOOKUP),
        (Role.REVIEWER, Capability.INSPECT),
        (Role.REVIEWER, Capability.STATUS),
        (Role.REVIEWER, Capability.TRACE),
        (Role.REVIEWER, Capability.EVIDENCE_LOOKUP),
        (Role.OPERATOR, Capability.RUN),
        (Role.OPERATOR, Capability.TRACE),
    ],
)
def test_default_profiles_allow_only_declared_capabilities(
    role: Role,
    capability: Capability,
) -> None:
    budget = Budget(max_queries=1)

    PolicyEngine().authorize(
        _principal(role),
        ToolCall(capability=capability),
        budget,
    )

    assert budget.used_queries == 1
    assert budget.remaining_queries == 0


@pytest.mark.parametrize(
    ("role", "capability"),
    [
        (Role.ANALYST, Capability.RUN),
        (Role.ANALYST, Capability.STATUS),
        (Role.REVIEWER, Capability.DISCOVER),
        (Role.REVIEWER, Capability.DIAGNOSE),
        (Role.REVIEWER, Capability.RECOMMEND),
        (Role.REVIEWER, Capability.RUN),
    ],
)
def test_default_profiles_deny_undeclared_capabilities_without_spending_budget(
    role: Role,
    capability: Capability,
) -> None:
    budget = Budget(max_queries=1)

    with pytest.raises(PolicyDeniedError) as exc_info:
        PolicyEngine().authorize(
            _principal(role),
            ToolCall(capability=capability),
            budget,
        )

    assert str(exc_info.value) == "capability is not authorized"
    assert budget.used_queries == 0


def test_custom_profiles_are_default_deny_for_missing_roles_and_capabilities() -> None:
    engine = PolicyEngine(
        profiles={Role.ANALYST: frozenset({Capability.INSPECT})}
    )

    engine.authorize(
        _principal(Role.ANALYST),
        ToolCall(capability=Capability.INSPECT),
        Budget(max_queries=1),
    )
    with pytest.raises(PolicyDeniedError):
        engine.authorize(
            _principal(Role.ANALYST),
            ToolCall(capability=Capability.DISCOVER),
            Budget(max_queries=1),
        )
    with pytest.raises(PolicyDeniedError):
        engine.authorize(
            _principal(Role.OPERATOR),
            ToolCall(capability=Capability.INSPECT),
            Budget(max_queries=1),
        )


def test_query_budget_is_shared_and_exhaustion_fails_closed() -> None:
    engine = PolicyEngine()
    principal = _principal(Role.OPERATOR)
    budget = Budget(max_queries=2)

    engine.authorize(principal, ToolCall(capability=Capability.INSPECT), budget)
    engine.authorize(principal, ToolCall(capability=Capability.RUN), budget)

    with pytest.raises(QueryBudgetExceededError) as exc_info:
        engine.authorize(principal, ToolCall(capability=Capability.STATUS), budget)

    assert str(exc_info.value) == "query budget exceeded"
    assert budget.used_queries == 2
    assert budget.remaining_queries == 0


def test_tool_call_contains_public_ids_but_has_no_path_escape_hatch() -> None:
    call = ToolCall(
        capability=Capability.INSPECT,
        dataset_id="public_demo",
    )

    assert call.dataset_id == "public_demo"
    with pytest.raises(ValidationError):
        ToolCall.model_validate(
            {
                "capability": Capability.INSPECT,
                "dataset_id": "public_demo",
                "path": "/private/company.parquet",
            }
        )
    with pytest.raises(ValidationError):
        ToolCall(capability=Capability.INSPECT, dataset_id="/private/company.parquet")


def test_policy_models_are_strict() -> None:
    with pytest.raises(ValidationError):
        Principal(principal_id=123, role=Role.ANALYST)
    with pytest.raises(ValidationError):
        Principal.model_validate(
            {"principal_id": "analyst-1", "role": Role.ANALYST, "extra": True}
        )
    with pytest.raises(ValidationError):
        ToolCall(capability="inspect")
    with pytest.raises(ValidationError):
        Budget(max_queries="1")
    with pytest.raises(ValidationError):
        Budget(max_queries=1, used_queries=2)
