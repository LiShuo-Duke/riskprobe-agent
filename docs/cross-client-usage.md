# 跨客户端使用指南

RiskProbe 的核心是本地 Python 包、CLI 和标准 stdio MCP 服务。MCP 采用两阶段 Host 决策协议，只暴露 `riskprobe_get_decision_context` 与 `riskprobe_submit_decision_proposal`；Kiro、Codex、Trae 和其他 MCP 客户端使用同一契约。

## 共同准备

```bash
git clone https://github.com/LiShuo-Duke/riskprobe-agent.git
cd riskprobe-agent
python3 -m venv .venv
./.venv/bin/python -m pip install -e .
```

准备本地 `ProjectConfig` YAML，在启动时固定 Parquet 路径、列角色、特征、运行目录与状态目录。真实数据、本地配置和运行产物不得提交到 Git；第一 MCP 工具只接收 `idempotency_key`，不能动态传入数据路径或角色。

```bash
./.venv/bin/python -m riskprobe.mcp_server \
  --config /absolute/path/project.yaml \
  --runs-dir /absolute/private/riskprobe-runs \
  --state-dir /absolute/private/riskprobe-state \
  --provider-id local-host \
  --provider-version host-model-version \
  --principal-id local-analyst \
  --role analyst \
  --max-queries 16
```

## Kiro

仓库内 `.kiro/settings/mcp.json` 指向被 Git 忽略的本地启动档案。Agent 仅允许两个 RiskProbe MCP tools，并拒绝 Shell、文件读写和网络。Host 依据受限聚合上下文生成中文报告；服务本身产出完整聚合工件，Host 不读取这些工件。

## Codex、Trae 与其他客户端

- Codex：复制 `configs/mcp/codex.example.toml`，替换全部绝对路径和 Host model version，并加载 `AGENTS.md`。
- Trae：导入 `configs/mcp/trae.example.json`，替换占位值，并加载 `docs/agent-system-prompt.md`。
- 其他客户端：复制 `configs/mcp/mcp.example.json`，保持 `python -m riskprobe.mcp_server` 及全部必需启动参数。

客户端配置只决定本地进程如何启动；不会扩大 RiskProbe 的工具面或数据权限。

## 统一工作流

```text
riskprobe_get_decision_context(idempotency_key)
  └─ 本地固定执行 inspect → diagnose → discover
Host 根据完整 findings 和 policy allowlist 选择 action_codes
riskprobe_submit_decision_proposal(
  同一 idempotency_key,
  原样 context_id,
  完整 diagnosis_evidence_ids,
  action_codes,
)
  └─ 本地固定执行 recommend → review，返回 terminal outcome
```

所有客户端都必须保持完整诊断证据集合，只能选择策略允许的 action code，并把 B 级结果限定为分析建议。中文报告按返回顺序覆盖 findings 和 evidence ID，并包含受控 action 与依据、terminal review/reason codes/retry、隐私限制和人工审批要求。禁止实体值、样本行、原始日志、真实路径、明细读取、Shell、网络访问和自动上线；不支持 MCP 的客户端可使用 CLI 做本地确定性分析，但 CLI 不是外部 Host 决策门控的绕过方式。
