# Changelog

## v0.2.0 — 2026-08-21

### Added

- 持久化 Host session sidecar、per-key idempotency session、completed replay、冲突拒绝、TTL-aware wait 和结构化 terminal outcome。
- Train-only 数值 WOE/IV 分箱：缺失箱、平滑、最小箱合并、单调坏率合并和冻结 edges transform。
- 规则—评分卡融合：WOE 特征与 `RiskRule` 命中特征训练 LogisticRegression，输出坏账概率、0–1000 风险分、风险等级和 reason codes。
- 监控闭环：Alert、Diagnosis/RootCause、RiskFinding、Recommendation 和 before/after remediation retest；`monitor` CLI 输出 `recommendations.json`。
- 有界 Agent repair：仅对 evidence mismatch、missing diagnosis、missing evidence 允许一次全量 typed-plan 重跑。
- 本地 sealed RAG citation 接入点：AgentResult 完成后独立返回 metadata-only citations，不改变 MCP 两工具协议。

### Safety boundaries

- 保留 `riskprobe_get_decision_context` 和 `riskprobe_submit_decision_proposal` 两个 MCP 工具。
- 不接入在线 LLM、远程数据源、动态工具选择、真实整改执行或自动策略上线。
- citation 不能替代业务 evidence、改变 action proposal 或绕过 `ProposalValidator`。

## v0.1.0

首个公开版本，提供确定性规则发现、统计验证、聚合监控、CLI 和 stdio MCP 决策边界。
