# Decision Submit Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep failed decision submission journals valid and recoverable while preserving the public MCP protocol.

**Architecture:** Reuse the existing evidence-backed decision-unavailable path. A submit exception becomes an internal `submission_error` outcome, an auditable decision node, and a deterministic `TOOL_FAILURE` terminal result; normal accepted submissions remain unchanged.

**Tech Stack:** Python, Pydantic, pytest, RiskProbe evidence/session stores.

## Global Constraints

- Public MCP schemas and error codes remain unchanged.
- Do not relax terminal validation or expose exception details.
- No new dependency.
- Do not create a git commit unless explicitly requested.

---

### Task 1: Make submit failures auditable and recoverable

**Files:**
- Modify: `src/riskprobe/agents/decision_controller.py:67-70`
- Modify: `src/riskprobe/agents/orchestrator.py:912-927`
- Test: `tests/agents/test_orchestrator.py`

**Interfaces:**
- Consumes: `DecisionController.record_unavailable(...)`, `AgentOrchestrator._append_decision_node(...)`.
- Produces: `_DecisionUnavailableReason.SUBMISSION_ERROR` and a replayable rejected `AgentResult`.

- [ ] **Step 1: Write the failing recovery test**

Add a test that monkeypatches `DecisionController.submit` to raise, runs `_controlled_orchestrator`, and asserts:

```python
assert result.status is AgentStatus.REJECTED
assert result.review.reason_codes == (ReviewReason.TOOL_FAILURE,)
assert outcome_record.payload["reason"] == "submission_error"
assert orchestrator.validate_terminal_result(
    result,
    objective="comprehensive",
    dataset_id="synthetic_demo",
    session_id="session-agent",
    metadata_grade="A",
) == result
assert orchestrator.recover_terminal_result(
    objective="comprehensive",
    dataset_id="synthetic_demo",
    session_id="session-agent",
    metadata_grade="A",
) == result
```

- [ ] **Step 2: Verify the test fails before implementation**

Run: `pytest tests/agents/test_orchestrator.py::test_submit_failure_records_replayable_unavailable_decision -q`

Expected: failure because no unavailable audit is written and terminal validation rejects the journal.

- [ ] **Step 3: Add the internal reason**

Extend `_DecisionUnavailableReason`:

```python
class _DecisionUnavailableReason(StrEnum):
    PROVIDER_PENDING = "provider_pending"
    PROVIDER_ERROR = "provider_error"
    SUBMISSION_ERROR = "submission_error"
```

- [ ] **Step 4: Record submit failure through the existing audit path**

Replace the broad submit catch with:

```python
except Exception:
    try:
        outcome = self._decision_controller.record_unavailable(
            context_evidence_id=preparation.context_evidence_id,
            reason=_DecisionUnavailableReason.SUBMISSION_ERROR,
            provider_binding=provider_binding,
        )
        leaf = self._append_decision_node(leaf, outcome)
    except Exception:
        raise RuntimeError("decision submission audit is unavailable") from None
    tool_failed = True
    break
```

- [ ] **Step 5: Verify focused recovery behavior**

Run: `pytest tests/agents/test_orchestrator.py::test_submit_failure_records_replayable_unavailable_decision -q`

Expected: PASS.

### Task 2: Regress the real two-action payload shape

**Files:**
- Modify: `tests/agents/test_orchestrator.py`

**Interfaces:**
- Consumes: `_DecisionGateway`, `_ExternalProposalProvider`, `tokenize_segment`.
- Produces: coverage for `review_segment_risk` plus `monitor_time_stability` through submit, recommend, review, terminal validation, and recovery.

- [ ] **Step 1: Generalize only the test helpers**

Allow `_DecisionGateway` to receive an optional action-to-finding mapping and `_ExternalProposalProvider` to receive `*action_codes`; preserve their current defaults so existing tests are unchanged.

- [ ] **Step 2: Add the exact two-action test**

Use two findings:

```python
RiskFinding(
    kind=FindingKind.SEGMENT_RISK,
    severity=FindingSeverity.INFO,
    code="segment_target_rate",
    segment_token=tokenize_segment("private-segment", namespace="test"),
    metrics={"sample_count": 100, "target_rate": 0.2},
)
RiskFinding(
    kind=FindingKind.TIME_INSTABILITY,
    severity=FindingSeverity.CRITICAL,
    code="monthly_sample_drop",
    period="2026-08",
    metrics={"current_count": 154, "previous_count": 4468, "drop_rate": 0.96},
)
```

Submit both matching actions and assert an approved result, exactly two recommendation records, successful `validate_terminal_result`, and successful `recover_terminal_result`.

- [ ] **Step 3: Run focused tests**

Run: `pytest tests/agents/test_orchestrator.py -q`

Expected: PASS.

- [ ] **Step 4: Run affected contract tests**

Run: `pytest tests/agents/test_decision_controller.py tests/test_mcp_server.py -q`

Expected: PASS with unchanged MCP response shape.
