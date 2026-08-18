# RiskProbe Agent 工作规则

本仓库提供本地只读风险分析引擎、CLI，以及一个由 Host 主动决策的两阶段 stdio MCP。MCP 只暴露 `riskprobe_get_decision_context` 和 `riskprobe_submit_decision_proposal`；数据配置、运行目录和状态目录必须在服务启动时由用户提供，不能作为工具参数传入。

## 固定流程

1. 使用一个公开且稳定的 `idempotency_key` 调用 `riskprobe_get_decision_context`。服务在本地执行 `inspect → diagnose → discover`，只返回受限聚合上下文。
2. Host 必须基于完整 `findings` 选择 action code；数量满足 `policy.min_action_count`/`max_action_count`，且只能来自 `policy.allowed_action_codes`。B 级数据还必须限制在 `policy.grade_b_allowed_action_codes`，结论仅用于分析。
3. 使用同一个 `idempotency_key`、原样 `context_id`、完整且未修改的 `diagnosis_evidence_ids` 和选定的 `action_codes` 调用 `riskprobe_submit_decision_proposal`。
4. 只接受 `terminal` 结果。完整顺序固定为 `inspect → diagnose → discover → recommend → review`；不得跳步、调用隐藏工具或自行构造证据。
5. 任一上下文、证据、action allowlist、过期时间或幂等校验失败时停止，不得通过 CLI、文件系统或其他工具绕过。

## 输出和隐私

- 不请求、读取或输出实体值、样本行、原始日志、真实文件路径或 Parquet 明细。
- 不执行 Shell、网络访问、任意 Python/SQL，也不自动修改或上线规则。
- 机构级规则只用于稳定性验证和人工复核，不自动升级为全局规则。
- B 级数据的表现窗口未知；`Stable` 不等于严格 OOT、生产就绪或自动上线。
- 所有 action 都是受控建议，最终结果必须经过 review，业务执行仍需人工批准。

## 客户端使用

Kiro 使用 `.kiro/agents/riskprobe.json` 与 `.kiro/skills/riskprobe/SKILL.md`。其他 MCP 客户端使用 `configs/mcp/` 中的模板，并加载 `docs/agent-system-prompt.md`。不支持 MCP 的客户端可使用 CLI，但 CLI 不是绕过 Host 决策门控的替代通道。
