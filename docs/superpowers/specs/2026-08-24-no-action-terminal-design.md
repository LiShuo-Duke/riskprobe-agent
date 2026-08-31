# 无可行动诊断终态设计

## 目标

当固定五步流程 `inspect → diagnose → discover → recommend → review` 全部安全完成，但诊断明确为空时，RiskProbe 以受限聚合 `terminal` 结果完成分析，而不是以 `agent_state_incomplete` 失败。正常有诊断时的 Host context/proposal/review 协议保持不变。

## 触发条件

仅当下列条件同时成立时允许无 action 终态：

- `DiagnoseResponse` 已成功通过类型和隐私校验，且 `finding_ids` 为空；
- 所有非 review 工具都成功完成；
- 不存在权限拒绝、工具失败、不安全 payload、证据不一致或 Grade-B 生产行为；
- review 以显式 `no_action_required=true` 批准，agent result 为 `succeeded`。

任何其他无 context 情形继续 fail-closed，保留已有安全错误码。

## 协议

新增严格 DTO `HostDecisionNoActionOutcome`，字段为固定 Host 协议版本、`phase: "terminal"`、`terminal_reason: "no_actionable_diagnosis"`、空 action 集和完整 `AgentResult`。不包含 context ID、proposal、diagnosis evidence、原始数据或工件路径。

`riskprobe_get_decision_context` 的结果联合加入该终态。它一经生成即持久化并用同一 idempotency key 重放。`riskprobe_submit_decision_proposal` 对此终态不接收 action，而是重放同一终态，防止绕过无 action 约束。

## Agent 与 review

`ReviewDecision` 增加默认 false 的 `no_action_required`。该标记只能与 approved、无 reason codes、无 retry、空 evidence 组合使用。Reviewer 仅在调用方证明干净空诊断时跳过 missing-evidence/missing-diagnosis 拒绝；其余 gate 保持不变。Orchestrator 记录该确定性事实并正常返回 succeeded `AgentResult`。

## 输出与隐私

Host 输出中文受限聚合结论：完整五步序列已完成、无可行动诊断、未创建 action/proposal、无需自动修改或上线，结果仍供人工审阅。不得输出样本、实体、列值、文件路径、原始日志、原始报告或伪造 evidence。

## 测试

- Reviewer：干净空诊断批准为 no-action；任一失败条件仍拒绝。
- Orchestrator：空诊断执行一次，不创建 decision evidence/node/provider 调用，返回 succeeded no-action。
- Coordinator：无 action agent result 返回、重放并持久化 no-action terminal；无该标记仍是 `agent_state_incomplete`。
- MCP：第一工具返回 terminal no-action，第二工具重放；既有 context/proposal 五步 happy path 与故障安全投影不变。
- 全量相关 pytest、ruff 和 diff check。
