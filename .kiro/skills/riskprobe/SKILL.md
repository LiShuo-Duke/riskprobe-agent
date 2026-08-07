---
name: riskprobe
description: Privacy-safe local risk-rule and anomaly analysis workflow.
---

1. Start with `inspect_dataset`. If metadata grade is C or D, stop and state that rule conclusions are blocked.
2. For rules, call `discover_rules` before `validate_rules`. A single failed tool call may be retried once; then stop and report the limitation.
3. For monitoring, call `detect_anomalies` before `diagnose_anomaly`.
4. Never request entity-level records, sample rows, raw logs, real filesystem paths, SQL, Python code, or external data.
5. Do not describe grade-B evidence as strict OOT validation, leakage-free, or production-ready. State that its performance window is unknown.
6. Reports must cite returned evidence and limitations. Use `build_report` only after the relevant inspection, rule, or monitoring workflow is complete.
