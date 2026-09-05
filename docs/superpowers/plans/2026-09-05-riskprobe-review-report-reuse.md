# RiskProbe Review and Report Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove duplicate Reviewer computation for newly produced terminal results and reuse an already-built terminal report model on idempotent replay.

**Architecture:** Keep the mandatory `inspect → diagnose → discover → recommend → review` gate unchanged. Trust only the fresh in-process `AgentOrchestrator.run()` result for its initial publish; persisted results and journals continue through full deterministic validation. Cache one successfully published `(TerminalReportSubject, ReportModel)` pair in `RiskProbeService`, while `TerminalReportStore` continues verifying the on-disk bundle on every response.

**Tech Stack:** Python 3.11+, pytest, Pydantic, existing RiskProbe stores.

## Global Constraints

- Keep exactly the two existing MCP tools and all public payloads unchanged.
- Every terminal result must still contain the mandatory deterministic review.
- Persisted result and journal recovery must remain fail-closed and revalidate Reviewer output.
- Every report response must still verify file permissions, manifest identity, size, and SHA-256 hashes.
- Add no dependency, schema, or report file.

---

### Task 1: Publish a fresh reviewed result without recomputing Reviewer

**Files:**
- Modify: `src/riskprobe/service.py:2260-2275`
- Test: `tests/test_service.py`

**Interfaces:**
- Consumes: `AgentOrchestrator.run(...) -> AgentResult`
- Produces: the same canonical `AgentResultStore` sidecar and public `AgentResult`

- [ ] **Step 1: Write the failing test**

Add a test that wraps `Reviewer.review`, executes one normal `RiskProbeService.orchestrate()` call, and asserts exactly one review call and a persisted terminal result containing `review`.

```python
def test_orchestrate_reviews_fresh_terminal_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from riskprobe.agents import Reviewer
    from riskprobe.agents.results import AgentResultStore
    from riskprobe.policy import Budget, Principal, Role

    calls = 0
    original = Reviewer.review

    def count_review(self: Reviewer, *args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Reviewer, "review", count_review)
    config = _small_config(tmp_path)
    state_dir = tmp_path / "state"
    service = RiskProbeService(
        config=config,
        runs_dir=tmp_path / "runs",
        state_dir=state_dir,
    )

    result = service.orchestrate(
        dataset_id=config.dataset.id,
        principal=Principal(principal_id="single-review", role=Role.ANALYST),
        budget=Budget(max_queries=16),
    )

    assert calls == 1
    assert result.review.approved is True
    assert AgentResultStore(
        state_dir / f".{result.session_id}.agent-result.json"
    ).load() == result
```

- [ ] **Step 2: Run the test and verify RED**

Run: `PYTHONPATH=src python3 -m pytest tests/test_service.py::test_orchestrate_reviews_fresh_terminal_once -q`

Expected: FAIL because the fresh path currently invokes `Reviewer.review` once in `_execute_once()` and once in `validate_terminal_result()`.

- [ ] **Step 3: Implement the minimum change**

Replace the immediate fresh-result revalidation with direct immutable publish:

```python
result = orchestrator.run(...)
result_store.publish(result)
return result
```

Do not change cached-result validation or journal recovery.

- [ ] **Step 4: Verify GREEN and nearby cache tests**

Run:

```bash
PYTHONPATH=src python3 -m pytest \
  tests/test_service.py::test_orchestrate_reviews_fresh_terminal_once \
  tests/test_service.py::test_orchestrate_reuses_verified_terminal_result_without_tool_calls \
  tests/test_service.py::test_orchestrate_recovers_terminal_result_when_result_sidecar_is_missing -q
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-09-05-riskprobe-review-report-reuse.md src/riskprobe/service.py tests/test_service.py
git commit -m "perf: avoid duplicate fresh terminal review"
```

### Task 2: Reuse the last successfully built terminal report model

**Files:**
- Modify: `src/riskprobe/service.py:45-65,1303-1330,1612-1660`
- Test: `tests/test_service.py`

**Interfaces:**
- Consumes: exact `TerminalReportSubject` equality and existing `TerminalReportStore.ensure_published(...)`
- Produces: unchanged `TerminalReportManifest`; disk bundle verification still runs for every call

- [ ] **Step 1: Write the failing test**

Create a verified run, publish one failed terminal report, replace `_load_verified_report_inputs` with a function that fails, then request the exact same report again and assert the manifest is reused.

```python
def test_terminal_report_reuses_built_model_for_same_subject(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from riskprobe.terminal_reports import TerminalReportSubject

    config = _small_config(tmp_path)
    service = RiskProbeService(
        config=config,
        runs_dir=tmp_path / "runs",
        state_dir=tmp_path / "state",
    )
    run = service.run()
    subject = TerminalReportSubject(
        idempotency_key="report-model-reuse",
        run_id=run.run_id,
        context_id=None,
        findings=(),
        proposal_action_codes=(),
        diagnosis_evidence_ids=(),
        agent_result=None,
        analysis_summary=None,
        decision_summary=None,
        terminal_status="failed",
        error_code="agent_orchestration_failed",
    )

    first = service.ensure_terminal_report(subject)

    def reject_rebuild(_context: object) -> object:
        raise AssertionError("terminal report model must be reused")

    monkeypatch.setattr(service, "_load_verified_report_inputs", reject_rebuild)

    assert service.ensure_terminal_report(subject) == first
```

- [ ] **Step 2: Run the test and verify RED**

Run: `PYTHONPATH=src python3 -m pytest tests/test_service.py::test_terminal_report_reuses_built_model_for_same_subject -q`

Expected: FAIL with `terminal report model must be reused`.

- [ ] **Step 3: Implement the bounded cache**

Import `ReportModel`, add one optional cache entry in `RiskProbeService.__init__`, and reuse it only when the complete subject compares equal. Populate the cache only after `TerminalReportStore.ensure_published()` succeeds.

```python
cached = self._cached_terminal_report
if cached is not None and cached[0] == subject:
    model = cached[1]
else:
    context = self.store.open_verified(subject.run_id)
    model = build_final_report_model(
        subject=subject,
        **self._load_verified_report_inputs(context),
    )
manifest = self._terminal_report_store.ensure_published(subject=subject, model=model)
self._cached_terminal_report = (subject, model)
return manifest
```

- [ ] **Step 4: Verify GREEN and report integrity behavior**

Run:

```bash
PYTHONPATH=src python3 -m pytest \
  tests/test_service.py::test_terminal_report_reuses_built_model_for_same_subject \
  tests/test_terminal_reporting.py::test_terminal_report_store_reuses_repairs_and_rejects_tampering \
  tests/test_mcp_server.py::test_mcp_no_action_unknown_report_gate_failure_is_safe_and_retryable \
  tests/test_mcp_server.py::test_mcp_actionable_report_gate_failure_is_safe_and_retryable -q
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/riskprobe/service.py tests/test_service.py
git commit -m "perf: reuse terminal report models"
```

### Task 3: Verify and publish the branch

**Files:**
- Verify all modified files.

**Interfaces:**
- Consumes: Tasks 1-2 commits
- Produces: pushed branch `perf/reuse-reviewed-terminal-results`

- [ ] **Step 1: Run formatting and lint checks**

```bash
python3 -m ruff check src/riskprobe/service.py tests/test_service.py
python3 -m ruff format --check src/riskprobe/service.py tests/test_service.py
```

Expected: both commands exit 0.

- [ ] **Step 2: Run the full suite**

Run: `PYTHONPATH=src python3 -m pytest -q`

Expected: all tests pass; the existing environment-only pandas/numexpr warning may remain.

- [ ] **Step 3: Inspect the final diff and commit graph**

```bash
git status --short --branch
git diff origin/main...HEAD --check
git log --oneline origin/main..HEAD
```

Expected: only the plan, service, and service tests changed; no whitespace errors.

- [ ] **Step 4: Push the feature branch**

Run: `git push -u origin perf/reuse-reviewed-terminal-results`

Expected: GitHub accepts the new branch without force-push.
