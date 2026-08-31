# RiskProbe 全分析输出与 Host 状态安全加固实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 一次性修复 RiskProbe 24-stage 摘要、评分卡、隐私、terminal 恢复和 Host Reconnect，使两个 MCP 工具完成可验证的端到端流程。

**Architecture:** 保留两工具和现有 Host 门控。服务在 AgentResultStore 锁内按 virgin/cached/terminal-journal/prefix 分类；Host 只恢复可验证终态或重启同一未完成 runner，不重新决策。所有公开输出继续使用 bounded DTO 和固定安全错误码。

**Tech Stack:** Python、Pydantic、SQLite session journal、stdio MCP、现有 RiskProbe privacy/artifact contracts。

## Global Constraints

- 单个 RiskProbe MCP 进程；共享 state-dir 的多进程并发不支持并必须 fail-fast。
- MCP 只暴露 `riskprobe_get_decision_context` 和 `riskprobe_submit_decision_proposal`。
- 24 stages 各出现一次；不返回实体值、行级数据、真实分群、路径、日志或 traceback。
- 仅完整且可验证 terminal journal 可恢复；非 terminal prefix、证据不完整和绑定不一致必须 fail-closed。
- 不删除状态、不切换 state-dir、不增加依赖、不新增 MCP 工具、不提交 Git。
- 按 workspace 规则不执行 Shell、任意 Python/SQL、网络或 pytest；仅静态检查和真实 MCP 工具验收。

---

### Task 1: 统一公开分析摘要、评分卡和隐私契约

**Files:**
- Modify: `src/riskprobe/analysis_contracts.py`
- Modify: `src/riskprobe/service.py`
- Modify: `src/riskprobe/privacy.py`
- Modify: `src/riskprobe/artifacts.py`
- Modify: `src/riskprobe/tools/local.py`
- Modify: `src/riskprobe/tools/models.py`
- Test: `tests/test_analysis_contracts.py`
- Test: `tests/test_service.py`
- Test: `tests/test_privacy.py`
- Test: `tests/test_artifacts.py`

**Interfaces:**
- Keep `AnalysisSummary`, `ScorecardSummary`, `FeatureRef` and `StageSummary` public shapes.
- Keep `_analysis_summary_payload(...)` bounded and return safe dataset IDs.
- Keep `_analysis_summary_from_run(...)` returning `None` only for verified legacy manifests without `analysis_summary.json`.

- [ ] **Step 1: Add failing contract assertions**

```python
def test_succeeded_scorecard_requires_aligned_features_terms_and_splits():
    payload = valid_scorecard_payload()
    payload["terms"] = []
    with pytest.raises(ValidationError):
        ScorecardSummary.model_validate(payload)


def test_legacy_manifest_without_summary_is_read_as_none():
    result = inspect_verified_legacy_run()
    assert result.analysis_summary is None
```

- [ ] **Step 2: Implement the minimum contract and projection changes**

```python
# succeeded scorecard validation must enforce:
# input == included | excluded; included.isdisjoint(excluded)
# feature summary names == included names
# feature-kind terms names == included names
# split names == {"train", "test", "holdout"} when holdout exists
# every risk-level count sum == sample_count
```

Use the already computed safe dataset identifier when building `InputSummary`; replace unavailable exception text with fixed codes such as `scorecard_fit_unavailable`; derive `artifact_count` from the verified manifest; treat legacy manifests without a declared summary as `None` and reject declared-but-missing summaries.

- [ ] **Step 3: Add the public privacy boundary**

Use one explicit safe-payload entry point for gateway responses and public JSON/text artifacts. Preserve exact allowlisted shapes for `FeatureRef`, trace events and aggregate summaries; reject paths, bytes, entity-like strings, raw row arrays and real segment values. Force institution/segment output to stable tokens regardless of the expose configuration.

- [ ] **Step 4: Static-check the changed producers**

Verify every `ScorecardSummary` success path includes model parameters, class counts, input/included/excluded features, terms, feature summaries, intercept and split metrics. Verify every public report/metadata/manifest path receives tokenized segments and fixed error codes.

---

### Task 2: Make artifact manifest state accurate and backward compatible

**Files:**
- Modify: `src/riskprobe/service.py`
- Modify: `src/riskprobe/artifacts.py`
- Test: `tests/test_service.py`
- Test: `tests/test_artifacts.py`

**Interfaces:**
- Preserve all existing legacy artifact sets.
- Add a fixed schema identifier for new manifests without accepting unknown schema/field combinations.

- [ ] **Step 1: Add failing manifest assertions**

```python
def test_new_manifest_reports_schema_and_finalized_summary():
    summary = run_and_read_analysis_summary()
    assert summary["stages"][10]["name"] == "finalize"
    assert summary["stages"][10]["status"] == "succeeded"
    assert summary["artifacts"]["published"] is True
    assert summary["artifacts"]["integrity_verified"] is True
```

- [ ] **Step 2: Implement schema/version and finalization projection**

Write `schema_version: "riskprobe.manifest.v1"` in new manifests; validate it together with the v1 field set while keeping legacy sets on their existing decoder branch. Build the immutable summary only with final publication facts, or pass the finalized facts into the summary builder before writing it.

- [ ] **Step 3: Verify legacy behavior statically**

Ensure a legacy manifest without `analysis_summary.json` reaches local inspect with `analysis_summary=None`, while a v1 manifest that declares the file but lacks/invalidates it raises the existing fixed safe failure.

---

### Task 3: Recover complete terminal journals safely

**Files:**
- Modify: `src/riskprobe/agents/orchestrator.py`
- Modify: `src/riskprobe/service.py`
- Modify: `src/riskprobe/agents/results.py`
- Test: `tests/test_service.py`
- Test: `tests/test_host_decision.py`

**Interfaces:**
- Add one internal recovery method, for example `recover_terminal_result(...) -> AgentResult | None`, that only projects a complete terminal journal and then calls the existing `validate_terminal_result(...)`.
- Do not change gateway, provider, policy or evidence semantics.

- [ ] **Step 1: Add failing recovery cases**

```python
def test_missing_result_sidecar_is_rebuilt_from_complete_terminal_journal():
    expected = run_once_to_terminal()
    remove_result_sidecar_without_touching_journal()
    actual = run_again()
    assert actual == expected


def test_nonterminal_prefix_stays_fail_closed_without_tool_calls():
    create_prefix_journal()
    with pytest.raises(HostSafeStageError, match="agent_state_incomplete"):
        run_again()
    assert tool_call_count() == 0
```

- [ ] **Step 2: Move state inspection under the result lock**

Acquire the existing result lock before reading/creating the session store and before deciding whether the state is virgin, cached, recoverable terminal, or incomplete. Remove the lock-external existence XOR check.

- [ ] **Step 3: Implement deterministic terminal projection**

Read and verify the append-only session chain, evidence chain, provider binding and final review node. Construct the same bounded `AgentResult`; validate it through the existing terminal validator; publish atomically. Return fixed `agent_state_incomplete` for prefixes and `agent_state_unavailable` for tampering or inconsistent bindings.

- [ ] **Step 4: Static-check recovery safety**

Confirm recovery performs no gateway call, no provider call, no fallback and no proposal creation. Confirm cached replay does not overwrite a valid result and all writes remain owner-only and atomic.

---

### Task 4: Make Host Reconnect and terminal evidence fail closed

**Files:**
- Modify: `src/riskprobe/host_decision.py`
- Modify: `src/riskprobe/agents/decision_contracts.py`
- Modify: `src/riskprobe/agents/decision_controller.py`
- Modify: `src/riskprobe/mcp_server.py`
- Test: `tests/test_host_decision.py`
- Test: `tests/test_mcp_server.py`
- Test: `tests/agents/test_decision_contracts.py`

**Interfaces:**
- Keep current MCP tool names and request/response schemas.
- Current terminal outcomes require complete evidence IDs and replayable bindings; legacy decoding remains isolated.

- [ ] **Step 1: Add failing restart/evidence cases**

```python
def test_reconnect_restarts_persisted_awaiting_runner_once():
    persist_context_then_recreate_coordinator()
    assert get_context_again().phase == "awaiting_proposal"
    submit_original_proposal()
    assert wait_for_terminal().phase == "terminal"


def test_missing_terminal_evidence_never_returns_success():
    remove_one_evidence_record()
    result = submit_original_proposal()
    assert result["phase"] == "context"
    assert result["error_code"] == "session_state_unavailable"
```

- [ ] **Step 2: Restore live-runner state after deserialize**

Set `runner_started=False` for persisted non-terminal sessions. On `get_context` and `submit`, start exactly one reconciliation runner for any unfinished session, reusing the persisted context/proposal and never creating a new proposal.

- [ ] **Step 3: Make current outcome evidence mandatory**

Remove broad exception swallowing in current outcome construction. Require context/proposal/result evidence IDs to be present, mutually bound and replayable. On failure persist a fixed session-state error; keep nullable fields only in explicit legacy decoders.

- [ ] **Step 4: Correct stage and summary projection**

Project actual journal nodes and decision/recommend/review outcomes into all 24 stages. Ensure accepted, rejected and no-action terminal outcomes carry bounded `analysis_summary` and `decision_summary`; map rejected no-context results to orchestration failure, not incomplete.

- [ ] **Step 5: Protect Host persistence and single-instance startup**

Reject oversized serialized Host state before replacement, preserving the previous valid file. Add the smallest existing platform-appropriate process lock/fail-fast guard at MCP startup; do not add cross-process CAS because shared state-dir is unsupported.

---

### Task 5: Static verification and one live MCP acceptance

**Files:**
- Modify: `.kiro/skills/riskprobe/SKILL.md` only if the final summary fields need wording alignment.
- Test: all changed tests listed above.

**Interfaces:**
- Live validation uses only `mcp_riskprobe_riskprobe_get_decision_context` and `mcp_riskprobe_riskprobe_submit_decision_proposal`.

- [ ] **Step 1: Re-read all changed code and contracts**

Check the 24-stage set, safe error allowlist, manifest sets, scorecard strategy field, terminal evidence requirements and legacy branches. Do not inspect run artifacts, state files or logs.

- [ ] **Step 2: Perform available static validation**

Use editor/static inspection only. Do not run Shell, arbitrary Python/SQL, network access or pytest under the workspace rule.

- [ ] **Step 3: Reconnect exactly once**

After all modifications are complete, reconnect RiskProbe once. Use one new stable key and verify `awaiting_proposal`, exactly 24 unique stages, scorecard aggregate fields, policy/evidence IDs and no forbidden data.

- [ ] **Step 4: Submit and replay exactly once**

Submit the unchanged allowed proposal with the original context ID and complete diagnosis evidence IDs. Verify terminal acceptance, review approval, both summaries, 24 stages, actual tool sequence and evidence completeness. Submit the identical proposal again and verify an equivalent terminal response.
