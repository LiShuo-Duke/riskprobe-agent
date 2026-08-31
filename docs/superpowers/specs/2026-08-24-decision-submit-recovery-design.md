# Decision submit recovery

## Problem

After an external Host proposal is accepted, `AgentOrchestrator._execute_once` catches any `DecisionController.submit` exception, marks `tool_failed`, and continues to review without a decision audit node. Terminal validation requires that audit, so the otherwise safe rejected result cannot be published and the MCP response collapses to `agent_orchestration_failed`. The incomplete journal then prevents recovery.

## Constraints

- Keep the public MCP schema and error codes unchanged.
- Preserve fail-closed behavior and evidence-backed audit validation.
- Do not relax terminal validation.
- Do not expose exception text or payload details.

## Design

Add the internal decision-unavailable reason `submission_error`. If `DecisionController.submit` fails, call the existing `record_unavailable` path with that reason and the already validated provider binding, append the resulting evidence-backed decision audit node, then continue to deterministic review as `TOOL_FAILURE`.

If recording the unavailable outcome also fails, propagate the failure rather than creating an unverifiable terminal result.

Normal accepted proposals and existing provider pending/error behavior remain unchanged.

## Tests

1. Force `DecisionController.submit` to fail and assert the orchestrator returns a rejected `TOOL_FAILURE` result whose journal passes `validate_terminal_result` and `recover_terminal_result`.
2. Cover a proposal selecting both `review_segment_risk` and `monitor_time_stability`; assert two recommendation records and an approved terminal result.
3. Keep the existing MCP response shape assertions unchanged.
