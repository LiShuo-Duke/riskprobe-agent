# RiskProbe Full Analysis Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute task-by-task. Do not create commits unless the user explicitly requests one.

**Goal:** Return every safe analysis-stage status and aggregate result through the existing two-tool Host protocol, including complete scorecard model terms and AUC/KS/Gini, then require Kiro to summarize only after terminal.

**Architecture:** Add strict safe summary DTOs and one immutable `analysis_summary.json` artifact. Carry that summary through `InspectResponse` into `DecisionContext`; enrich it with diagnostic/agent state; attach a terminal `DecisionSummary` built from actual recommendation and review outputs. Preserve the two MCP tools, privacy gate, evidence binding, proposal validation, and fixed tool sequence.

**Tech Stack:** Python, Pydantic, Polars, NumPy, scikit-learn, existing RiskProbe artifact/evidence/MCP layers.

**Design:** `docs/superpowers/specs/2026-08-24-riskprobe-full-analysis-output-design.md`

## Global constraints

- Never return entity values, sample rows, real segment labels, local paths, raw logs/tracebacks, row-level predictions, or unsuppressed small-bin details.
- Every new public DTO must be strict, bounded, deterministic, JSON-safe, and pass `assert_safe_payload`.
- Preserve `riskprobe_get_decision_context` and `riskprobe_submit_decision_proposal` as the only tools and preserve their input schemas.
- Do not weaken privacy, evidence, policy, expiry, idempotency, or review checks.
- Use installed dependencies only. Reuse `discover_with_metrics()` and `compute_score_ks()`.
- Old persisted sessions remain readable; new sessions use the extended schema.
- Workspace rules prohibit the agent from running Shell/arbitrary Python. Implement tests, perform static review, and use the real MCP protocol for runtime acceptance after restart; explicitly report pytest as not executed.

---

### Task 1: Define bounded analysis and decision contracts

**Files:**
- Create: `src/riskprobe/analysis_contracts.py`
- Create: `tests/test_analysis_contracts.py`
- Modify: `src/riskprobe/__init__.py` only if public exports are already maintained there

**Implementation:**

- Add `StageStatus` enum values `not_run`, `succeeded`, `skipped`, `unavailable`, `failed`, `reused`.
- Add strict DTOs for `StageSummary`, `InputSummary`, `ProfileSummary`, `PartitionSummary`, `MetricDistribution`, `DiscoverySummary`, `FeatureRef`, `ScorecardFeatureSummary`, `ScorecardTerm`, `ScorecardSplitMetrics`, `ScorecardSummary`, `ValidationSummary`, `ArtifactSummary`, `DiagnosticsSummary`, `AnalysisSummary`, `RecommendationSummary`, and `DecisionSummary`.
- Fix the stage-name allowlist to the approved 24 stages. Reject duplicate/missing stage names in new full summaries.
- Add deterministic feature references: preserve a safe public feature name; otherwise return a stable `model-feature-<hex>` token. Never drop a model term.
- Bound top-rule/top-evidence arrays; require complete model term arrays to remain under a hard maximum derived from configured feature/rule caps. Exceeding it produces `scorecard.status=unavailable` and `reason_code=output_limit_exceeded`, not truncation.
- Validators must enforce finite numbers, unique IDs, status/reason consistency, score relationships (`gini == 2*auc-1` within tolerance), and `assert_safe_payload(model_dump(mode="json"))`.
- Keep LLM prose out of DTOs.

**Tests:**

- Validate complete happy-path DTO roundtrip and stable JSON serialization.
- Reject forbidden paths, entity-like values, duplicate stages, non-finite metrics, inconsistent statuses, too many complete model terms, and invalid AUC/Gini relationships.
- Verify unsafe feature names become deterministic public refs while ordinary feature names remain unchanged.

### Task 2: Produce complete artifact-pipeline summary and score metrics

**Files:**
- Modify: `src/riskprobe/service.py`
- Modify: `src/riskprobe/rules/discovery.py` only if an additional safe counter is unavailable
- Modify: `src/riskprobe/metrics.py`
- Modify: `tests/test_metrics.py`
- Modify: `tests/test_scorecard.py`
- Modify: `tests/test_service.py`
- Modify: artifact-name assertions in `tests/test_host_decision.py` and `tests/test_mcp_server.py`

**Implementation:**

- Reuse `discover_with_metrics()` in `RiskProbeService.run()` so persisted rules are unchanged while candidate/selected single/pair counts and selected train metrics remain available.
- Extend `_scorecard_payload()` with safe input/included/excluded features, exclusion reasons, rule-hit terms, intercept, every coefficient, WOE IV/bin count/monotonic/missing-bin presence, and per-split aggregate metrics.
- Add `compute_score_auc()` in `metrics.py` using installed `sklearn.metrics.roc_auc_score`; reuse `compute_score_ks()` and calculate Gini as `2*AUC-1`. Empty/single-class splits return fixed unavailable reason codes.
- Generate `analysis_summary.json` after report and before manifest. It must project config/snapshot/profile/partition/discovery/woe/scorecard/validation/institution/report statuses and outputs from in-memory trusted objects; never parse report markdown.
- Add `analysis_summary.json` to expected immutable artifacts and manifest verification. The on-disk summary records finalize as `not_run`; the public projection upgrades finalize/artifact integrity after manifest verification.
- Explicitly represent scorecard disabled as skipped, no usable model as unavailable, empty rule discovery as succeeded with zero rules, and holdout absence as unavailable rather than failed.
- Include all safe feature/model terms; apply small-group suppression to segment/bin aggregates.

**Tests:**

- Fixed arrays for AUC/KS/Gini and empty/single-class unavailable behavior.
- Scorecard fitted/disabled/unavailable projections; all model coefficients and included/excluded feature refs preserved.
- Partition strict/auto fallback and split counts.
- Discovery counts equal `discover_with_metrics()` results without changing rules.
- Validation grade/limitation distributions and institution aggregate only.
- Manifest contains the new artifact and summary passes privacy validation.

### Task 3: Carry the summary into DecisionContext and diagnostics

**Files:**
- Modify: `src/riskprobe/tools/models.py`
- Modify: `src/riskprobe/tools/local.py`
- Modify: `src/riskprobe/agents/decision_contracts.py`
- Modify: `src/riskprobe/agents/decision_controller.py`
- Modify: `src/riskprobe/agents/orchestrator.py`
- Modify: `tests/test_local_tool_handler.py`
- Modify: `tests/agents/test_orchestrator.py`
- Modify: `tests/test_host_decision.py`

**Implementation:**

- Add optional `analysis_summary` to `InspectResponse` for legacy compatibility. Local inspect loads only the verified summary artifact and overlays finalize/integrity status; it must not expose paths.
- Extend `DecisionContext` schema to accept legacy v1 without summary and emit v2 with required summary for new runs.
- In `DecisionController.prepare()`, combine the artifact summary with actual inspect/discover state and the complete resolved findings. Produce deterministic diagnostic stage statuses and counts by kind/severity/code; distinguish run time-validation-applied from diagnostic-time-enabled.
- Mark inspect, discover_restore, and decision_context accurately; keep recommend/review/terminal as not_run at awaiting-proposal.
- Persist the complete context as aggregate evidence exactly as before and include the summary in context identity.
- Preserve policy, TTL, complete evidence, and context replay checks.

**Tests:**

- New context has v2 summary and every approved stage exactly once.
- Legacy v1 context/session remains readable.
- Context privacy gate passes with complete scorecard terms.
- Diagnostic counts match complete findings and skipped diagnostics have fixed reason codes.
- Context replay remains deterministic.

### Task 4: Return actual recommendation, review, terminal, and failure progress

**Files:**
- Modify: `src/riskprobe/tools/models.py`
- Modify: `src/riskprobe/service.py`
- Modify: `src/riskprobe/agents/contracts.py`
- Modify: `src/riskprobe/agents/orchestrator.py`
- Modify: `src/riskprobe/host_decision.py`
- Modify: `tests/agents/test_orchestrator.py`
- Modify: `tests/test_host_decision.py`
- Modify: `tests/test_mcp_server.py`

**Implementation:**

- Extend `RecommendResponse` with bounded safe `RecommendationSummary` values: action code, parent finding IDs, human-approval flag, analysis-only flag, limitations, and evidence ID. Populate these from the actual recommendation objects written by service; do not regenerate recommendations in Host code.
- Capture the real recommend response and review result in `AgentResult.decision_summary` for new runs; retain optional field for legacy result stores.
- Extend `HostDecisionOutcome` and no-action terminal with `analysis_summary` and `decision_summary`; build from authoritative context/result/proposal and reject mismatches.
- Extend `HostDecisionFailure` with optional bounded `analysis_progress`. Attach the latest safe stage projection to host-safe exceptions; never include exception text.
- Ensure terminal stages mark recommend/review/terminal succeeded or failed and evidence completeness is explicit.
- Ensure exact idempotent replay returns equal summaries and old session files still validate.

**Tests:**

- Proposal terminal includes action-to-finding recommendation bindings and approved review.
- No-action terminal has zero recommendations but complete stage status.
- Rejected review and context failures expose safe progress only.
- Same key/proposal replay is equal; mismatched proposal remains rejected.
- Tool input schemas and exactly-two-tool assertion remain unchanged.

### Task 5: Require Kiro terminal-only LLM summary

**Files:**
- Modify: `.kiro/skills/riskprobe/SKILL.md`
- Modify: `.kiro/agents/riskprobe.json` only if description needs clarification; permissions/tools must remain unchanged
- Modify: `tests/test_kiro_config.py`
- Modify: `docs/agent-system-prompt.md` if other MCP clients share the reporting contract

**Implementation:**

- Require Kiro to wait for terminal, then summarize execution completeness, data/profile, partition, discovery/WOE, scorecard parameters and all model terms, AUC/KS/Gini, validation, diagnostics, recommendations/review, limitations, and human approval.
- Require key metrics and statuses to be retained; do not replace structured facts with prose or infer absent data.
- For failure terminal/context failure, summarize only returned progress and fixed reason codes.
- Keep all denied permissions and the two-tool workflow unchanged.

**Tests:**

- Skill contains terminal-only summary requirements and every major analysis section.
- Agent still exposes only `@riskprobe`, denies filesystem/shell/network, and allows only two RiskProbe MCP methods.

### Task 6: Integration, compatibility, and real MCP acceptance

**Files:**
- Modify: `tests/test_mcp_server.py`
- Modify: `tests/test_service.py`
- Modify: `tests/test_version_contract.py` if schema snapshots require updates
- Modify: docs/examples only when existing contract fixtures require it

**Implementation and verification:**

- Add an end-to-end synthetic scorecard-enabled test asserting: all stages present; model terms complete; AUC/KS/Gini finite or explicitly unavailable; findings complete; proposal accepted; recommendation bindings present; review approved; terminal summary complete; replay equal.
- Add privacy assertions over both awaiting-proposal and terminal payloads.
- Update exact artifact sets and schema-version expectations.
- Perform independent spec and quality review with no Critical/Important findings.
- Because agent-run pytest is prohibited by workspace policy, provide the exact external command for the user/CI:

```bash
/opt/anaconda3/bin/python -m pytest tests/test_analysis_contracts.py tests/test_metrics.py tests/test_scorecard.py tests/test_service.py tests/test_host_decision.py tests/test_mcp_server.py tests/test_kiro_config.py -q
```

- Restart RiskProbe MCP and use a new stable idempotency key. Verify awaiting-proposal contains all stages and scorecard fields, submit policy-valid actions with exact evidence IDs, require terminal accepted/review approved, then replay the same proposal and require an equal terminal result.
- Kiro produces the final Chinese LLM summary from terminal only.

## Completion criteria

- Existing two-tool protocol and all safety gates remain intact.
- Every approved analysis stage has explicit status in both success and failure paths.
- Full safe scorecard model terms and AUC/KS/Gini are available before Host proposal.
- Actual recommendation/review results are returned at terminal.
- Legacy persisted sessions parse; new runs emit the new schema.
- Real MCP analysis reaches terminal and Kiro summarizes all returned sections without prohibited data.
