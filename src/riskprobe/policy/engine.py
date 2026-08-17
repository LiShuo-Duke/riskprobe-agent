"""Default-deny capability profiles and budget enforcement."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from riskprobe.policy.models import (
    Budget,
    Capability,
    PolicyDeniedError,
    Principal,
    QueryBudgetExceededError,
    Role,
    ToolCall,
)


_DEFAULT_PROFILES: Mapping[Role, frozenset[Capability]] = MappingProxyType(
    {
        Role.ANALYST: frozenset(
            {
                Capability.INSPECT,
                Capability.DISCOVER,
                Capability.DIAGNOSE,
                Capability.RECOMMEND,
                Capability.EVIDENCE_LOOKUP,
            }
        ),
        Role.REVIEWER: frozenset(
            {
                Capability.INSPECT,
                Capability.STATUS,
                Capability.TRACE,
                Capability.EVIDENCE_LOOKUP,
            }
        ),
        Role.OPERATOR: frozenset(Capability),
    }
)


class PolicyEngine:
    """Authorize declared capabilities only, then charge the shared query budget."""

    def __init__(
        self,
        *,
        profiles: Mapping[Role, Iterable[Capability]] | None = None,
        query_costs: Mapping[Capability, int] | None = None,
    ) -> None:
        selected_profiles = _DEFAULT_PROFILES if profiles is None else profiles
        normalized_profiles: dict[Role, frozenset[Capability]] = {}
        for role, capabilities in selected_profiles.items():
            if not isinstance(role, Role):
                raise ValueError("policy profiles are invalid")
            frozen_capabilities = frozenset(capabilities)
            if any(not isinstance(capability, Capability) for capability in frozen_capabilities):
                raise ValueError("policy profiles are invalid")
            normalized_profiles[role] = frozen_capabilities
        self._profiles = MappingProxyType(normalized_profiles)

        selected_costs: Mapping[Capability, int]
        selected_costs = (
            {capability: 1 for capability in Capability}
            if query_costs is None
            else query_costs
        )
        normalized_costs: dict[Capability, int] = {}
        for capability, cost in selected_costs.items():
            if (
                not isinstance(capability, Capability)
                or isinstance(cost, bool)
                or not isinstance(cost, int)
                or cost < 0
            ):
                raise ValueError("policy query costs are invalid")
            normalized_costs[capability] = cost
        self._query_costs = MappingProxyType(normalized_costs)

    def capabilities_for(self, role: Role) -> frozenset[Capability]:
        if not isinstance(role, Role):
            return frozenset()
        return self._profiles.get(role, frozenset())

    def authorize(
        self,
        principal: Principal,
        call: ToolCall,
        budget: Budget,
    ) -> None:
        if (
            not isinstance(principal, Principal)
            or not isinstance(call, ToolCall)
            or not isinstance(budget, Budget)
        ):
            raise PolicyDeniedError("capability is not authorized")
        allowed = self._profiles.get(principal.role, frozenset())
        cost = self._query_costs.get(call.capability)
        if call.capability not in allowed or cost is None:
            raise PolicyDeniedError("capability is not authorized")
        if not budget.consume(cost):
            raise QueryBudgetExceededError("query budget exceeded")
