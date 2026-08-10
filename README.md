# RiskProbe Agent

RiskProbe 是一个本地只读的风控规则发现、证据验证、机构稳定性分析和漂移监控工具。它提供 Python 包、CLI 和标准 stdio MCP 服务，可被 Kiro、Codex、Trae 及其他 MCP 客户端使用。

## 核心能力

- 在 Train 数据中发现可解释风险规则；
- 在 Test/Holdout/时间切片中验证 Lift、覆盖率、置信区间和稳定性；
- 先做全局规则，再做机构级稳定性和条件式局部发现；
- 检测 Schema、缺失率、分布、分层占比、标签率和规则衰减告警；
- 输出聚合报告，不读取或输出实体明细和样本行；
- 默认在受限聚合字段展示已确认的真实机构名，显式关闭时使用稳定 token。

RiskProbe 不上传 Parquet，不连接远程数据库，不执行任意 Shell，不访问网络，也不自动上线风控策略。

## 安装

需要 Python 3.11 或更高版本。macOS/Linux：

```bash
git clone https://github.com/LiShuo-Duke/riskprobe-agent.git
cd riskprobe-agent
python3 -m venv .venv
./.venv/bin/python -m pip install -e .
```

Windows PowerShell：

```powershell
git clone https://github.com/LiShuo-Duke/riskprobe-agent.git
Set-Location riskprobe-agent
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
```

开发依赖：

```bash
./.venv/bin/python -m pip install -e '.[dev]'
```

## 本地数据目录

RiskProbe 只访问用户明确配置的本地 allowlist 目录。macOS/Linux 示例：

```bash
mkdir -p "$HOME/riskprobe-data"
export RISKPROBE_ALLOWED_DATA_ROOTS="$HOME/riskprobe-data"
```

将 Parquet 放入该目录。不要把真实数据、个人配置、运行产物或密钥提交到 GitHub。

配置文件中的：

```yaml
privacy:
  expose_segment_values: false
```

可以关闭受限报告中的真实机构名展示；默认值为 `true`。实体值、样本行、原始日志、真实路径和 Parquet 明细读取边界不受影响。

## CLI

生成合成数据并运行本地分析：

```bash
./.venv/bin/riskprobe synthetic --output ./behavior.parquet --rows 5000 --seed 42
./.venv/bin/riskprobe inspect --config ./configs/synthetic.example.yaml --runs-dir ./runs
./.venv/bin/riskprobe run --config ./configs/synthetic.example.yaml --runs-dir ./runs
```

CLI 结果写入用户指定的本地目录；`runs/` 已被 Git 忽略。

## MCP

RiskProbe MCP 是本地 stdio 服务，不是 HTTP 服务。直接启动：

```bash
./.venv/bin/python -m riskprobe.mcp_server
```

通用 MCP 配置模板：

```text
configs/mcp/mcp.example.json
```

Codex 和 Trae 的配置示例：

```text
configs/mcp/codex.example.toml
configs/mcp/trae.example.json
```

模板中的 `/ABSOLUTE/PATH/...` 必须替换为用户自己的项目路径和数据目录。MCP 的工作流是：

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

`discover_rules` 不接受非空 `constraints`。没有时间列时只做随机 Train/Test 验证，不得声称严格 OOT。

## 各客户端使用

- **Kiro**：打开仓库并选择 `riskprobe` Agent；自动读取 `.kiro/agents`、`.kiro/skills` 和 `.kiro/settings/mcp.json`。
- **Codex**：配置 `configs/mcp/codex.example.toml`，并使用仓库根目录的 `AGENTS.md`。
- **Trae**：在 MCP 管理界面导入 `configs/mcp/trae.example.json`，再加载 `docs/agent-system-prompt.md`。
- **其他 MCP 客户端**：配置同一个本地 stdio MCP 命令，并加载 `AGENTS.md` 或通用提示词。
- **不支持 MCP 的客户端**：可以使用 Python 包和 CLI，但无法获得工具调用式 Agent 流程。

完整说明见：

- `docs/cross-client-usage.md`
- `docs/agent-system-prompt.md`
- `AGENTS.md`
- `docs/riskprobe-agent-technical-guide.md`

## 安全边界

这是本地分析工具，不是在线数据服务：

- GitHub 只分发代码和脱敏示例配置；
- 数据只从用户配置的本地 allowlist 目录读取；
- 源 Parquet 以只读方式使用，不会被修改；
- 不请求实体明细、样本行、原始日志或真实路径；
- 不执行 Shell、网络访问、任意 SQL/Python 或自动策略上线；
- 机构级规则仅供稳定性验证和人工复核。

## 开发验证

```bash
./.venv/bin/python -m pytest --disable-warnings --maxfail=1
./.venv/bin/ruff check src tests
```

公开发布前还应检查 Git tracked 文件中没有真实数据、个人路径、密钥、虚拟环境或运行产物。
