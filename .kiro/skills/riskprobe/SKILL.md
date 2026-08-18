---
name: riskprobe
description: Use when a configured local RiskProbe MCP server needs a bounded Host decision.
---

# RiskProbe Host decision workflow

RiskProbe exposes exactly two MCP tools. The deterministic service owns data access and the fixed pipeline; the Host only chooses bounded recommendation action codes from aggregate evidence.

1. Create one public, stable `idempotency_key`, then call `riskprobe_get_decision_context(idempotency_key)` once. This runs the fixed flow through `inspect → diagnose → discover` and returns an `awaiting_proposal` context.
2. Read only the returned aggregate `context`. Ground the decision in every item in `context.findings`; do not request files, rows, paths, logs, Shell, network access, or additional MCP tools.
3. Choose between `policy.min_action_count` and `policy.max_action_count` unique values from `policy.allowed_action_codes`. For metadata grade B, use only `policy.grade_b_allowed_action_codes` and describe the result as analysis-only, never production-ready or strict OOT.
4. Call `riskprobe_submit_decision_proposal` exactly once with the same `idempotency_key`, the exact returned `context_id`, the complete unchanged `diagnosis_evidence_ids`, and the selected `action_codes`. Never invent, omit, reorder semantically, or substitute evidence IDs.
5. Accept only a `terminal` outcome. Report the review status, reason codes, evidence IDs, retry count, and fixed completed sequence `inspect → diagnose → discover → recommend → review`. Recommendations require human review and never modify or deploy policy automatically.

If either tool rejects the context, evidence set, action allowlist, expiry, or idempotency key, stop and report that the Host decision is unavailable. Do not bypass the gate with CLI, filesystem, or alternate tools.
