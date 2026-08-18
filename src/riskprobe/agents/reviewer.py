"""Deterministic policy, evidence, grade, privacy, and retry reviewer."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from riskprobe.agents.contracts import (
    ExecutionPlan,
    ReviewDecision,
    ReviewReason,
)
from riskprobe.privacy import assert_safe_payload

SafePayloadHook = Callable[[object], None]
_RECOVERABLE_REASONS = frozenset(
    {
        ReviewReason.EVIDENCE_MISMATCH,
        ReviewReason.MISSING_DIAGNOSIS,
        ReviewReason.MISSING_EVIDENCE,
    }
)


class Reviewer:
    """Apply deterministic fail-closed gates; no model provider participates."""

    def __init__(
        self,
        *,
        safe_payload_hook: SafePayloadHook = assert_safe_payload,
        version: str = "reviewer-v1",
    ) -> None:
        if not callable(safe_payload_hook):
            raise TypeError("safe_payload_hook must be callable")
        self._safe_payload_hook = safe_payload_hook
        self.version = version

    def review(
        self,
        plan: ExecutionPlan,
        *,
        evidence_ids: Sequence[str] = (),
        diagnosis_evidence_ids: Sequence[str] = (),
        claimed_evidence_ids: Sequence[str] = (),
        metadata_grade: Literal["A", "B"] = "A",
        payloads: Sequence[object] = (),
        permission_denied: bool = False,
        unsafe_payload_detected: bool = False,
        tool_failed: bool = False,
        retry_count: int = 0,
    ) -> ReviewDecision:
        if not isinstance(plan, ExecutionPlan):
            raise TypeError("plan must be an ExecutionPlan")
        if metadata_grade not in {"A", "B"}:
            raise ValueError("metadata_grade must be A or B")
        if isinstance(retry_count, bool) or not isinstance(retry_count, int) or retry_count < 0:
            raise ValueError("retry_count must be a non-negative integer")

        evidence = tuple(sorted(set(evidence_ids)))
        diagnosis = tuple(sorted(set(diagnosis_evidence_ids)))
        claims = tuple(sorted(set(claimed_evidence_ids)))
        reasons: list[ReviewReason] = []
        if not evidence:
            reasons.append(ReviewReason.MISSING_EVIDENCE)
        if permission_denied:
            reasons.append(ReviewReason.PERMISSION_DENIED)
        if unsafe_payload_detected or not self._payloads_are_safe(payloads):
            reasons.append(ReviewReason.UNSAFE_PAYLOAD)
        if metadata_grade == "B" and any(step.production_action for step in plan.steps):
            reasons.append(ReviewReason.GRADE_B_PRODUCTION_ACTION)
        if retry_count > 1:
            reasons.append(ReviewReason.RETRY_LIMIT_EXCEEDED)
        if not diagnosis:
            reasons.append(ReviewReason.MISSING_DIAGNOSIS)
        if (diagnosis and not set(diagnosis).issubset(evidence)) or (
            claims and not set(claims).issubset(evidence)
        ):
            reasons.append(ReviewReason.EVIDENCE_MISMATCH)
        if tool_failed:
            reasons.append(ReviewReason.TOOL_FAILURE)

        normalized_reasons = tuple(dict.fromkeys(reasons))
        approved = not normalized_reasons
        retry_allowed = (
            not approved
            and retry_count < 1
            and set(normalized_reasons).issubset(_RECOVERABLE_REASONS)
        )
        return ReviewDecision(
            approved=approved,
            reason_codes=normalized_reasons,
            evidence_ids=evidence,
            retry_allowed=retry_allowed,
        )

    def _payloads_are_safe(self, payloads: Sequence[object]) -> bool:
        try:
            for payload in payloads:
                result = self._safe_payload_hook(payload)
                if result is False:
                    return False
        except Exception:
            return False
        return True


DeterministicReviewer = Reviewer

__all__ = ["DeterministicReviewer", "Reviewer", "SafePayloadHook"]
