# 跨客户端使用指南

RiskProbe 的核心是本地 Python 包、CLI 和标准 stdio MCP 服务。Kiro 的 Agent/Skill 是原生增强层；Trae、Codex 和其他支持 MCP 的客户端通过 MCP 配置和通用规则使用同一套核心能力。

## 共同准备

```bash
git clone https://github.com/LiShuo-Duke/riskprobe-agent.git
cd riskprobe-agent
python3 -m venv .venv
./.venv/bin/python -m pip install -e .
mkdir -p "$HOME/riskprobe-data"
export RISKPROBE_ALLOWED_DATA_ROOTS="$HOME/riskprobe-data"
```

将用户自己的 Parquet 放入 allowlist 目录。不要把数据、运行目录、密钥或本地配置提交到 Git。Windows 使用 `.venv\\Scripts\\python.exe`，并将环境变量设置为绝对目录。

先确认 MCP 服务器可启动：

```bash
./.venv/bin/python -m riskprobe.mcp_server
```

这是 stdio 服务，启动后等待客户端连接，不是 HTTP 服务；直接在终端运行时可用 `Ctrl-C` 停止。

## Kiro

在 Kiro 中打开仓库并信任 workspace，选择 `riskprobe` Agent。Kiro 会读取：

- `.kiro/agents/riskprobe.json`；
- `.kiro/skills/riskprobe/SKILL.md`；
- `.kiro/settings/mcp.json`。

设置 `RISKPROBE_ALLOWED_DATA_ROOTS` 后，在 Agent 对话中使用 `@riskprobe`。Kiro 会按 schema → 角色确认 → 特征确认 → 注册 → 分析的流程运行。

## Codex

将 `configs/mcp/codex.example.toml` 的内容复制到受信任的 project-scoped Codex 配置中，并替换绝对路径。将仓库根目录的 `AGENTS.md` 作为项目规则加载。Codex 的 MCP 配置通常位于项目 `.codex/config.toml` 或用户配置中，具体以客户端版本为准。

Codex 不会自动解析 Kiro 的 `.kiro/skills`，因此必须保留并遵守 `AGENTS.md`；需要更强的提示约束时，将 `docs/agent-system-prompt.md` 复制到 Codex 的项目指令中。

## Trae

在 Trae 的 MCP 管理界面中手动导入或复制 `configs/mcp/trae.example.json`，替换绝对路径。将 `docs/agent-system-prompt.md` 复制到 Trae 的 Agent/system prompt 配置中。

Trae 不一定自动加载 Kiro 的 `.kiro/agents` 和 `.kiro/skills`。如果当前版本的 MCP 配置字段不同，以 Trae 界面生成的 schema 为准，但服务命令必须保持：

```text
<project>/.venv/bin/python -m riskprobe.mcp_server
```

## 其他 MCP 客户端

只要客户端支持本地 stdio MCP，就配置同样的三项内容：

```text
command: <项目绝对路径>/.venv/bin/python
args: -m riskprobe.mcp_server
RISKPROBE_REGISTRY: <项目>/configs/datasets.example.yaml
RISKPROBE_ALLOWED_DATA_ROOTS: <用户自己的绝对数据目录>
```

再加载 `AGENTS.md` 或 `docs/agent-system-prompt.md`。如果客户端不支持 MCP，只能使用 Python CLI。

## 统一工作流

```text
inspect_local_parquet_schema
→ 用户确认实体/时间/机构/目标
→ preview_local_parquet_features
→ 用户确认精确特征列
→ register_local_parquet
→ inspect_dataset
→ discover_rules(objective="risk")
→ validate_rules
→ detect_anomalies
→ diagnose_anomaly
→ build_report
```

`discover_rules` 不接受非空 `constraints`；无时间列时只能做随机 Train/Test 验证。默认受限字段展示真实机构名，显式 `privacy.expose_segment_values=false` 时改为机构 token。所有客户端都禁止实体值、样本行、原始日志、真实路径、明细读取、Shell、网络和自动上线。
