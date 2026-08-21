# RiskProbe v0.2.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 RiskProbe 的 Host 决策协议从单进程、单 key 的内存协调升级为可恢复、多会话、结构化终态的 v0.2.0，同时补齐无发现结果和 Grade-B 策略语义。

**Architecture:** 继续使用现有 `EvidenceStore`、`DecisionController` 和两工具 MCP 边界。新增轻量的 Host session sidecar，以幂等 key 绑定已持久化的 context/proposal/result evidence；协调器只负责等待和恢复，不复制决策校验逻辑。`ProposalValidator` 仍是唯一 server-owned 决策门，Host 终态只增加聚合的结构化字段。

**Tech Stack:** Python 3.11+, Pydantic 2, SQLite/现有 EvidenceStore sidecar, stdio MCP, pytest, ruff。

## Global Constraints

- 版本目标为 `0.2.0`；本地当前 `0.1.0`。
- 保持 `riskprobe_get_decision_context` 和 `riskprobe_submit_decision_proposal` 两个高层 MCP 工具，不暴露底层数据工具。
- 不接入在线 LLM、网络数据源、生产动作执行或热更新 policy。
- 所有新增持久化状态写入启动时指定的 `state-dir`，权限保持用户私有和 `0600`。
- 所有新协议 DTO 必须使用 strict/frozen Pydantic 模式并拒绝额外字段。
- 生产代码必须先有一个会失败的回归测试，再实现最小修复。

## 文件边界

- Modify `src/riskprobe/host_decision.py`: Host session 状态、持久化恢复、多 key 协调、TTL-aware waiting、结构化 terminal outcome。
- Modify `src/riskprobe/mcp_server.py`: 将 state-dir/session store 注入 Host coordinator；继续只注册两个 MCP 工具。
- Modify `src/riskprobe/agents/decision_contracts.py`: no-action 语义、空诊断安全契约、Grade-B action policy 校验。
- Modify `src/riskprobe/agents/decision_controller.py` and/or `src/riskprobe/service.py`: 允许空诊断生成明确的 no-action terminal path；不改变已有 evidence chain 规则。
- Modify `tests/test_host_decision.py`, `tests/test_mcp_server.py`, `tests/agents/test_decision_contracts.py`（若实际目录结构不同，以现有文件为准）: RED/GREEN 回归测试。
- Modify `pyproject.toml`, `README.md`, create `CHANGELOG.md` only in the implementation worktree: v0.2.0 标记与变更说明。

### Task 1: Host session persistence and multi-key protocol

**Files:**
- Modify: `src/riskprobe/host_decision.py`
- Modify: `src/riskprobe/mcp_server.py`
- Test: `tests/test_host_decision.py`
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- `HostDecisionCoordinator(..., state_dir: Path | None = None)` keeps `state_dir` optional for direct unit tests; MCP runtime passes the configured state directory.
- `HostDecisionOutcome` retains `context_id` and `agent_result`, and adds only aggregate fields: `decision_status`, `reason_codes`, `action_codes`, `context_evidence_id`, `proposal_evidence_id`, `result_evidence_id`, `expires_at`.
- A stored session is keyed by validated `idempotency_key` and contains only public key, context identity, proposal identity, result identity, lifecycle status, and serialized safe DTOs/evidence references.

- [ ] Step 1: Add a failing test proving two different idempotency keys can obtain independent contexts and terminal outcomes.

```python
def test_coordinator_supports_independent_idempotency_keys(tmp_path):
    first = make_coordinator(tmp_path)
    second = make_coordinator(tmp_path)
    first_context = get_context(first, "first-key")
    second_context = get_context(second, "second-key")
    assert first_context.context.context_id != second_context.context.context_id
```

- [ ] Step 2: Run `python3 -m pytest tests/test_host_decision.py -q`; confirm the test fails because the current coordinator rejects the second key.
- [ ] Step 3: Add a per-key session map and per-session condition/state. Do not use a global single `_idempotency_key`, `_context`, or `_proposal`.
- [ ] Step 4: Add a failing test proving a second coordinator instance can replay a completed terminal result from the same `state_dir` and key.
- [ ] Step 5: Run the focused test and confirm it fails because state is currently process-local.
- [ ] Step 6: Implement a minimal JSON/SQLite-compatible `HostSessionStore` in the existing Host module using an atomic temp-file replace, `0600` mode, and canonical Pydantic serialization. Reuse evidence IDs rather than copying private payloads.
- [ ] Step 7: Add a failing test proving a conflicting proposal for the same key is rejected after replay, while the identical proposal returns the same terminal outcome.
- [ ] Step 8: Implement exact proposal fingerprint binding and idempotent replay under the per-session lock.
- [ ] Step 9: Add a failing test proving wait duration is bounded by `context.expires_at - now`, not a fixed 300-second value.
- [ ] Step 10: Implement TTL-aware context and terminal waits; preserve fail-closed behavior when the deadline expires.
- [ ] Step 11: Add a failing test asserting the terminal response exposes structured decision status, reason codes, action codes, and evidence IDs without raw data.
- [ ] Step 12: Implement the extended frozen DTO and populate it from the existing `AgentResult` and persisted decision evidence. Keep the protocol version stable only if fields are backward-compatible; otherwise use `riskprobe.host-decision.v2` and update both MCP tools together.
- [ ] Step 13: Run `python3 -m pytest tests/test_host_decision.py tests/test_mcp_server.py -q`; expected: all focused Host/MCP tests pass.

### Task 2: No-action and Grade-B decision contracts

**Files:**
- Modify: `src/riskprobe/agents/decision_contracts.py`
- Modify: `src/riskprobe/agents/decision_controller.py`
- Modify: `src/riskprobe/service.py`
- Modify: `tests/agents/test_decision_contracts.py` or `tests/test_service.py`

**Interfaces:**
- Add an explicit `DecisionStatus.NO_ACTION` and `DecisionReason.NO_APPLICABLE_FINDINGS` (or equivalent names matching existing enum conventions).
- Empty diagnosis is represented by a valid terminal decision, not by a fabricated `RiskFinding`.
- Grade-B policy rejects actions categorized as operationally effectful; analysis-only actions remain accepted. Existing action codes are mapped conservatively without adding production execution semantics.

- [ ] Step 1: Add a failing contract test constructing an empty-diagnosis context and expecting a valid no-action result.
- [ ] Step 2: Run the focused contract test and confirm the current non-empty validators reject it.
- [ ] Step 3: Extend the contract and validator so no findings produces a canonical no-action result with no action codes and a reason code; ensure normal accepted/rejected proposal behavior is unchanged.
- [ ] Step 4: Add a failing Grade-B test using an effectful action and assert rejection, plus a permitted analysis/manual-review action and assert acceptance.
- [ ] Step 5: Implement a small explicit action-effect mapping in `src/riskprobe/recommendations/policy.py` or `decision_contracts.py`; default Grade-B allowlist must no longer equal all actions when `grade_b_analysis_only` is true.
- [ ] Step 6: Add a service/orchestrator regression test for an empty diagnostic report and make it return the no-action terminal path without entering controlled recommendation.
- [ ] Step 7: Run `python3 -m pytest tests/test_host_decision.py tests/test_service.py tests/test_policy.py -q` and confirm all related tests pass.

### Task 3: Version marker and documentation

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Create: `CHANGELOG.md`
- Modify: `docs/superpowers/plans/2026-08-21-riskprobe-v0.2.0.md` only if implementation decisions differ

- [ ] Step 1: Add the failing version/documentation assertions to the existing package/CLI test location, asserting the package version and README current version are `0.2.0`.
- [ ] Step 2: Run the focused version tests and confirm they fail against `0.1.0`.
- [ ] Step 3: Change only the package version and current-version documentation; add a concise `v0.2.0` changelog entry listing persistence, multi-key Host sessions, structured terminal results, no-action semantics, and Grade-B policy tightening.
- [ ] Step 4: Run the version tests and confirm they pass.

### Task 4: Full verification and review gate

**Files:**
- No new production files unless a focused test exposes a necessary correction.

- [ ] Step 1: Run `python3 -m pytest tests/test_host_decision.py tests/test_mcp_server.py tests/test_service.py tests/test_policy.py tests/agents -q`.
- [ ] Step 2: Run `python3 -m ruff check src tests`.
- [ ] Step 3: Run `python3 -m pytest -q` and record the full pass count and any pre-existing environment warnings.
- [ ] Step 4: Inspect `git diff --check` and `git status --short`; verify no files outside the v0.2.0 worktree were changed.
- [ ] Step 5: Dispatch a code-review subagent against the worktree diff; fix all Critical/Important findings and rerun affected tests.
- [ ] Step 6: Prepare a local change summary and stop before commit/tag/push, waiting for the user to confirm the v0.2.0 result.
