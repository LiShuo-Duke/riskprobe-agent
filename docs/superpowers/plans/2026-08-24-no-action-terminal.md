# No-Action Terminal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the fixed RiskProbe analysis safely when diagnosis is explicitly empty, returning a replayable aggregate terminal outcome without fabricating a context, evidence, proposal, or action.

**Architecture:** The orchestrator records a constrained no-action review only after every non-review tool completes safely and diagnosis is explicitly empty. The Host coordinator recognizes only this approved result without a context as a terminal no-action outcome, persists and replays it. MCP returns that terminal from the first tool and replays it from the proposal tool; ordinary findings retain the current context → proposal → terminal flow.

**Tech Stack:** Python 3.13, Pydantic v2 strict DTOs, pytest, stdio MCP.

## Global Constraints

- Do not read or modify private startup configs, Parquet, runs, state, logs, or MCP configuration.
- Do not add dependencies, tools, network calls, raw values, or path-bearing payloads.
- No-action is legal only for a clean, explicit empty `DiagnoseResponse`; all error/unsafe/incomplete paths remain fail-closed.
- Preserve the two existing MCP tool names and the normal non-empty diagnosis proposal contract.
- Do not create commits unless the user explicitly requests one.

---

### Task 1: Encode the clean no-action review state

**Files:**
- Modify: `src/riskprobe/agents/contracts.py:140-190`
- Modify: `src/riskprobe/agents/reviewer.py:35-100`
- Test: `tests/agents/test_planner_reviewer.py:100-210`

**Interfaces:**
- Consumes: `Reviewer.review(..., evidence_ids, diagnosis_evidence_ids, tool_failed, ...)`.
- Produces: `ReviewDecision.no_action_required: bool = False` and `Reviewer.review(..., no_action_required: bool = False)`.

- [ ] **Step 1: Write failing Reviewer tests**

```python
def test_reviewer_approves_clean_no_action_terminal() -> None:
    decision = Reviewer().review(_plan(), no_action_required=True)
    assert decision.approved is True
    assert decision.no_action_required is True
    assert decision.reason_codes == ()
    assert decision.evidence_ids == ()
    assert decision.retry_allowed is False

@pytest.mark.parametrize("kwargs", ({"tool_failed": True}, {"permission_denied": True}))
def test_reviewer_rejects_no_action_when_a_safety_gate_fails(kwargs: dict[str, bool]) -> None:
    decision = Reviewer().review(_plan(), no_action_required=True, **kwargs)
    assert decision.approved is False
    assert decision.no_action_required is False
```

- [ ] **Step 2: Run the targeted test to verify RED**

Run: `PYTHONPATH=src pytest tests/agents/test_planner_reviewer.py -q`

Expected: failure because `no_action_required` is not an accepted argument/field.

- [ ] **Step 3: Add the strict DTO field and validation**

Add `no_action_required: bool = False` to `ReviewDecision`. Its validator must reject it unless `approved is True`, `reason_codes == ()`, `evidence_ids == ()`, and `retry_allowed is False`; retain all prior review validation.

- [ ] **Step 4: Add minimal Reviewer behavior**

Add keyword-only `no_action_required: bool = False`. When true, do not add `MISSING_EVIDENCE` or `MISSING_DIAGNOSIS`; evaluate all remaining permission, safety, Grade-B, retry and tool-failure gates. Pass `no_action_required=approved and no_action_required` to `ReviewDecision`.

- [ ] **Step 5: Run targeted tests to verify GREEN**

Run: `PYTHONPATH=src pytest tests/agents/test_planner_reviewer.py -q`

Expected: all pass.

### Task 2: Let the orchestrator complete only explicit empty diagnoses

**Files:**
- Modify: `src/riskprobe/agents/orchestrator.py:660-810`
- Test: `tests/agents/test_orchestrator.py:1248-1290`

**Interfaces:**
- Consumes: validated `DiagnoseResponse`, tool safety flags, `ReviewDecision.no_action_required` from Task 1.
- Produces: succeeded `AgentResult` with `review.no_action_required=True`, no decision-provider call and no `decision.*` evidence only for clean empty diagnosis.

- [ ] **Step 1: Write failing orchestrator regression**

Replace the current permanent-empty-diagnosis expectation with a clean empty-diagnosis test that asserts one attempt, a succeeded result, `result.review.no_action_required is True`, fixed five-tool sequence, zero provider calls, and no decision evidence/node. Add a separate gateway failure assertion preserving rejected/fail-closed behavior.

- [ ] **Step 2: Run the focused test to verify RED**

Run: `PYTHONPATH=src pytest tests/agents/test_orchestrator.py -q`

Expected: clean empty diagnosis remains rejected with `MISSING_DIAGNOSIS`.

- [ ] **Step 3: Track explicit empty diagnosis in `_execute_once`**

Initialize `clean_empty_diagnosis = False`. Set it true only after a validated `DiagnoseResponse` has `finding_ids == ()`; clear/leave false for nonempty diagnosis, tool errors, type mismatches, unsafe payloads, and invalid finding IDs. Before `Reviewer.review`, pass `no_action_required=(clean_empty_diagnosis and not diagnosis_evidence and not tool_failed and not permission_denied and not unsafe_payload)`.

- [ ] **Step 4: Preserve normal decision behavior**

Do not enter `DecisionController.prepare` or provider resolution for empty diagnosis. Continue fixed discover/recommend calls, then complete deterministic review. Do not add synthetic evidence or recommendations.

- [ ] **Step 5: Run focused tests to verify GREEN**

Run: `PYTHONPATH=src pytest tests/agents/test_orchestrator.py -q`

Expected: empty clean diagnosis succeeds without decision artifacts; ordinary and unsafe paths retain their existing results.

### Task 3: Add persistent Host no-action terminal projection

**Files:**
- Modify: `src/riskprobe/host_decision.py:90-160, 360-760`
- Test: `tests/test_host_decision.py:300-450`

**Interfaces:**
- Consumes: `AgentResult` where `status is AgentStatus.SUCCEEDED` and `review.no_action_required is True`.
- Produces: `HostDecisionNoActionOutcome` with `phase="terminal"`, `terminal_reason="no_actionable_diagnosis"`, `action_codes=()`, `agent_result`; coordinator `get_context()` returns it and `submit_proposal()` replays it.

- [ ] **Step 1: Write failing coordinator tests**

```python
def test_clean_no_action_result_returns_and_replays_terminal(tmp_path: Path) -> None:
    coordinator = HostDecisionCoordinator(provider_id="kiro", version="gpt5.6sol", state_dir=tmp_path)
    result = _no_action_result("no-action-run")
    outcome = coordinator.get_context(idempotency_key="no-action-key", runner=lambda: result)
    assert outcome.phase == "terminal"
    assert outcome.terminal_reason == "no_actionable_diagnosis"
    assert outcome.action_codes == ()
    assert coordinator.get_context(idempotency_key="no-action-key", runner=lambda: result) == outcome
```

Include restart replay and assert a normal rejected/no-context `AgentResult` still returns `agent_state_incomplete`.

- [ ] **Step 2: Run focused tests to verify RED**

Run: `PYTHONPATH=src pytest tests/test_host_decision.py -q`

Expected: no-action result is currently projected as `agent_state_incomplete`.

- [ ] **Step 3: Define strict outcome and state union**

Create `HostDecisionNoActionOutcome(_StrictDTO)` with fixed protocol version, `phase: Literal["terminal"]`, `terminal_reason: Literal["no_actionable_diagnosis"]`, `action_codes: Literal[()]`, and `agent_result: AgentResult`. Validate agent success plus `review.no_action_required`. Widen `_SessionState.outcome`, `get_context`, `submit_proposal`, persisted outcome parsing and `__all__` to accept `HostDecisionOutcome | HostDecisionNoActionOutcome`.

- [ ] **Step 4: Persist and replay no-action safely**

In `_execute`, when `session.context is None`, build the no-action outcome only if the result satisfies the new DTO. Otherwise retain `agent_state_incomplete`. Add a discriminating `terminal_reason` key to `_outcome_from_payload`; retain strict exact-key validation for both normal and no-action serialized payloads. Persist lifecycle `terminal` as existing code does.

- [ ] **Step 5: Make proposal submission replay no-action terminal**

Before requiring a context in `submit_proposal`, if `session.outcome` is `HostDecisionNoActionOutcome`, return it without validating caller supplied action fields. This remains idempotent and does not create a proposal/evidence record.

- [ ] **Step 6: Run focused tests to verify GREEN**

Run: `PYTHONPATH=src pytest tests/test_host_decision.py -q`

Expected: normal proposal flow, failures and no-action persistence/replay pass.

### Task 4: Project no-action terminal through MCP and user contracts

**Files:**
- Modify: `src/riskprobe/mcp_server.py:50-110`
- Modify: `.kiro/skills/riskprobe/SKILL.md`
- Modify: `docs/agent-system-prompt.md`
- Test: `tests/test_mcp_server.py:60-150`
- Test: `tests/test_kiro_config.py`

**Interfaces:**
- Consumes: coordinator union returned by `get_context` and the no-action terminal type from Task 3.
- Produces: first MCP call can return a terminal no-action aggregate result; proposal call replays that terminal; Host instructions forbid actions/proposals for it.

- [ ] **Step 1: Write failing MCP contract test**

Monkeypatch `RiskProbeService.orchestrate` to return a strict no-action `AgentResult`. Assert `riskprobe_get_decision_context` returns a structured terminal result with `terminal_reason="no_actionable_diagnosis"`, empty action codes, succeeded five-step agent result, and no context/evidence fields. Assert calling submit returns the same terminal payload.

- [ ] **Step 2: Run focused test to verify RED**

Run: `PYTHONPATH=src pytest tests/test_mcp_server.py -q`

Expected: current tool schema/serialization cannot accept no-action terminal.

- [ ] **Step 3: Widen the first tool return annotation and second tool return projection**

Import the new DTO. Make `riskprobe_get_decision_context()` return `HostDecisionContext | HostDecisionNoActionOutcome | HostDecisionFailure`. In the submit tool, serialize `HostDecisionNoActionOutcome` with `model_dump(mode="json")` exactly as normal terminal responses.

- [ ] **Step 4: Update restricted Host instructions**

In the Kiro skill and agent system prompt, document: a first-call `phase="terminal"` with `terminal_reason="no_actionable_diagnosis"` is final; do not call proposal; report the completed five-step aggregate result, absence of actions and manual-review limitation. Retain all existing non-empty context evidence/action rules unchanged.

- [ ] **Step 5: Run MCP and contract tests to verify GREEN**

Run: `PYTHONPATH=src pytest tests/test_mcp_server.py tests/test_kiro_config.py -q`

Expected: existing two-tool schema and happy path remain unchanged; no-action terminal is accepted only as the defined aggregate outcome.

### Task 5: Run regression suite and real fixed-profile protocol

**Files:**
- No product-file changes expected.

**Interfaces:**
- Consumes: completed no-action terminal or ordinary `awaiting_proposal` from `riskprobe_get_decision_context`.
- Produces: only a protocol-valid terminal response and a Chinese aggregate final report.

- [ ] **Step 1: Run full automated verification**

Run:

`PYTHONPATH=src pytest tests/agents/test_planner_reviewer.py tests/agents/test_orchestrator.py tests/test_config.py tests/test_profiling.py tests/test_scorecard.py tests/test_service.py tests/test_host_decision.py tests/test_mcp_server.py tests/test_kiro_config.py -q`

Then run:

`ruff check src tests && git diff --check`

Expected: all tests and checks pass; only a pre-existing numexpr environment warning may remain.

- [ ] **Step 2: Reconnect MCP after source change**

Use Kiro MCP Servers to reconnect `riskprobe`; do not inspect or change MCP configuration.

- [ ] **Step 3: Execute the real bounded protocol with a fresh public key**

Call `riskprobe_get_decision_context` once using `xinyongka-fixed-profile-full-analysis-v10`.

- [ ] **Step 4: Complete exactly one valid terminal path**

- If it returns `phase="terminal"` and `terminal_reason="no_actionable_diagnosis"`, do not submit a proposal; emit the no-action Chinese aggregate conclusion.
- If it returns `phase="awaiting_proposal"`, choose only allowed actions from all returned findings, then submit the exact context/evidence once with the same key; accept only `phase="terminal"`.
- If it returns a context failure, stop without private artifact inspection or bypass; report only its safe code.

- [ ] **Step 5: Do not commit**

Leave the verified working-tree diff uncommitted unless the user explicitly requests a commit.
