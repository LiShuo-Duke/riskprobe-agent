# RiskProbe rule_id 隐私误报修复设计

## 问题

规则发现使用 SHA-256 摘要前 12 位生成公开 `rule_id`。摘要偶然全为数字时，通用隐私检查会将其识别为长数字敏感标识，导致 `discover` 响应被拒绝。Agent 最终得到 `unsafe_payload`，Host 因未收到 `DecisionContext` 而返回 `agent_state_incomplete`。

## 约束

- 不放宽长数字隐私检查。
- 不读取或输出实体值、样本行或数据明细。
- 不修改旧的 immutable sidecar。
- 非纯数字的既有 `rule_id` 必须保持不变。

## 设计

在 `src/riskprobe/rules/discovery.py` 的 ID 生成源修复：

1. 继续计算现有 SHA-256 摘要前 12 位。
2. 仅当该值 `isdigit()` 时添加固定 `rule-` 前缀。
3. 其他 ID 原样返回。

示例：`123456789012` 变为 `rule-123456789012`；`12ab567890cd` 保持不变。

该方案避免调整隐私边界，且仅改变原本无法通过安全门的 ID。前缀形式不会与现有 12 位十六进制 ID 冲突。

## 数据流

`discover` 生成安全 `rule_id` → gateway/privacy 校验通过 → Orchestrator 发布 `DecisionContext` → Host 提交 allowlist 内 action proposal → `recommend` 和 `review` 完成 → MCP 返回 `terminal`。

## 错误处理

隐私检查继续 fail closed。任何其他不安全 payload 仍按原逻辑拒绝；本修复不改变 `agent_state_incomplete`、`unsafe_payload` 或 Host 状态机语义。

## 验证

- 定向检查纯数字摘要会获得 `rule-` 前缀。
- 检查含字母摘要保持原值。
- 运行相关规则发现、隐私与 agent 测试。
- 重启 MCP，使用新的稳定 `idempotency_key` 获取 context。
- 按返回 policy 选择 action code，使用同一 key 和原样证据 ID 提交，确认结果为 `terminal`。
