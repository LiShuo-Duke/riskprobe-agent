"""Deterministic planner restricted to injected typed tool contracts."""

from __future__ import annotations

from collections.abc import Mapping

from riskprobe.agents.contracts import (
    ExecutionPlan,
    PlanStep,
    ReviewDecision,
    ReviewReason,
)
from riskprobe.agents.providers import ModelProvider, default_provider
from riskprobe.tools.models import (
    DiagnoseRequest,
    DiscoverRequest,
    InspectRequest,
    RecommendRequest,
    ToolRequest,
)

_REQUIRED_TOOLS: tuple[tuple[str, type[ToolRequest]], ...] = (
    ("inspect", InspectRequest),
    ("diagnose", DiagnoseRequest),
    ("discover", DiscoverRequest),
    ("recommend", RecommendRequest),
)
_REPAIRABLE_REASONS = frozenset(
    {
        ReviewReason.EVIDENCE_MISMATCH,
        ReviewReason.MISSING_DIAGNOSIS,
        ReviewReason.MISSING_EVIDENCE,
    }
)


class PlanningError(ValueError):
    """Raised with a fixed safe message when a plan cannot be selected."""


class Planner:
    """Select the fixed comprehensive plan only from an injected typed allowlist."""

    def __init__(
        self,
        *,
        allowed_tools: Mapping[str, type[ToolRequest]],
        provider: ModelProvider | None = None,
        version: str = "planner-v1",
    ) -> None:
        if not isinstance(allowed_tools, Mapping):
            raise TypeError("allowed_tools must be a typed mapping")
        self._allowed_tools = dict(allowed_tools)
        self.provider = provider if provider is not None else default_provider()
        self.version = version

    @property
    def allowed_tools(self) -> Mapping[str, type[ToolRequest]]:
        return dict(self._allowed_tools)

    def repair(
        self,
        *,
        objective: str,
        dataset_id: str,
        decision: ReviewDecision,
        retry_count: int,
    ) -> ExecutionPlan:
        """Authorize one deterministic full-plan retry from reviewer reason codes."""

        if not isinstance(decision, ReviewDecision):
            raise TypeError("decision must be a ReviewDecision")
        if type(retry_count) is not int or retry_count != 0:
            raise PlanningError("repair retry limit exceeded")
        if decision.approved:
            raise PlanningError("review does not authorize repair")
        if not decision.reason_codes or not set(decision.reason_codes).issubset(
            _REPAIRABLE_REASONS
        ):
            raise PlanningError("review reason is not repairable")
        if not decision.retry_allowed:
            raise PlanningError("review does not authorize repair")
        return self.plan(objective=objective, dataset_id=dataset_id)

    def plan(self, *, objective: str, dataset_id: str) -> ExecutionPlan:
        if objective != "comprehensive":
            raise PlanningError("unsupported safe objective")
        for name, request_type in _REQUIRED_TOOLS:
            if self._allowed_tools.get(name) is not request_type:
                raise PlanningError("required typed tool is unavailable")

        requests: tuple[ToolRequest, ...] = (
            InspectRequest(dataset_id=dataset_id),
            DiagnoseRequest(dataset_id=dataset_id),
            DiscoverRequest(dataset_id=dataset_id),
            RecommendRequest(dataset_id=dataset_id),
        )
        steps = tuple(
            PlanStep(
                step_id=name,
                tool_name=name,
                request=request,
                requires_evidence=name == "recommend",
            )
            for (name, _), request in zip(_REQUIRED_TOOLS, requests, strict=True)
        )
        return ExecutionPlan(
            objective=objective,
            dataset_id=dataset_id,
            steps=(*steps, PlanStep(step_id="review", tool_name="review")),
            component_versions={"planner": self.version},
        )


WhitelistPlanner = Planner

__all__ = ["Planner", "PlanningError", "WhitelistPlanner"]
