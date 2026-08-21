# RiskProbe Agent

**面向风控规则发现、模型监控、PSI 漂移检测和机构稳定性分析的本地隐私安全 Agent。**

RiskProbe 是一个 **local-first** 的 Python 风控分析工具包，提供确定性规则引擎、CLI 和标准 stdio MCP 服务。它可以被 Kiro、Codex、Trae 以及其他 MCP 客户端调用。

> **当前版本：** `v0.2.0`。RiskProbe 只在本地读取用户明确允许的 Parquet 数据，不上传数据，不暴露实体级明细。v0.2.0 增加了 WOE/IV 规则分箱、规则—评分卡融合、告警到复测闭环，以及有界 Agent 重试和本地 citation 接入。

## RiskProbe 解决什么问题？

在消费金融、信用风险和模型监控场景中，仅有一个模型分数通常不够。分析人员还需要知道：

- 哪些风险规则可以稳定复现；
- 规则在不同机构或分群中是否一致；
- 当前漂移是全局问题还是局部机构问题；
- Test/Holdout/时间切片证据是否支持继续人工复核；
- 如何让 Agent 使用聚合证据，而不是读取用户明细。

RiskProbe 将规则发现、统计验证和监控计算固定在 Python 确定性引擎中，Agent 只负责受限编排和解释。

## 核心功能

### 1. 可解释规则发现（Explainable Rule Discovery）

- 基于分位点、浅层决策树和 LightGBM 辅助生成候选阈值；
- 支持一条件规则和二条件组合规则；
- 规则 ID 稳定、排序确定、随机种子固定；
- 规则条件只使用用户确认过的建模特征。

### 2. 统计验证与稳定性分级

支持 Train/Test 以及可选的时间切片验证，计算：

- Support、Coverage、Bad Rate、Lift；
- Precision、Recall；
- Fisher 精确检验 p-value；
- Benjamini-Hochberg / FDR 校正；
- Bootstrap Lift 置信区间；
- 机构或分群一致性；
- 时间 Lift 衰减。

规则分为：

```text
Stable / Local / Unstable / Suspicious
```

### 3. 规则—评分卡融合

v0.2.0 提供 train-only WOE/IV 分箱和冻结边界变换，可将 WOE 特征与 `RiskRule` 命中特征一起训练 LogisticRegression 评分卡，输出：

- 坏账概率和 0–1000 风险分；
- `low / medium / high / critical` 风险等级；
- 基于特征贡献度的 top reason codes。

评分卡预测不会重新拟合分箱，缺失值使用训练期缺失箱，规则命中特征只由确定性规则表达式计算。

### 4. 多机构稳定性分析

RiskProbe 采用“全局优先”的顺序：

```text
全机构合并 Train 发现全局规则
→ Test/Holdout 按机构验证
→ 判断 Stable / Local / Unstable / Suspicious
→ 只对满足门槛的 Local 机构做条件式局部发现
```

机构内规则只用于稳定性验证和人工复核，不会自动升级为全局规则或上线策略。

### 5. PSI 和聚合漂移监控

支持检测：

- Schema 变化；
- Missingness 缺失率变化；
- PSI 分布漂移；
- Population / 机构占比变化；
- Label 正类率变化；
- Rule Lift 衰减。

### 6. 根因诊断与整改复测闭环

监控链路固定为：

```text
Alert → Diagnosis / RootCause → RiskFinding → Recommendation → before/after retest
```

`monitor` CLI 额外写出 `recommendations.json`。整改记录是不可变状态，复测按稳定 finding 语义键比较，输出 `verified / remaining / inconclusive`；建议仍需要人工审批，不执行真实数据修改或策略上线。


对告警生成聚合根因 TOP3，支持以下维度：

```text
feature / family / segment / label / rule / schema
```

输出贡献度、排名和数值证据，不返回用户实体、样本行或原始明细。

### 7. 启动时只读数据配置

本地 Parquet、列角色、特征族和隐私策略由 `ProjectConfig` YAML 明确声明。MCP 服务启动时接收配置路径、运行目录和状态目录；两个 MCP 工具都不接受文件路径、任意代码、身份、预算或数据注册参数。源 Parquet 始终只读，运行产物写入用户指定的本地目录。

### 8. 两阶段 Host 决策 MCP

标准本地 stdio MCP 只暴露：

```text
riskprobe_get_decision_context
riskprobe_submit_decision_proposal
```

第一阶段固定执行 `inspect → diagnose → discover` 并返回聚合决策上下文；Kiro、Codex、Trae 或其他 Host 从 policy allowlist 选择 action code 后提交原样上下文和完整诊断证据。第二阶段固定执行 `recommend → review` 并返回 terminal result。Host 不拥有底层数据工具，不能跳步或自动上线策略。

### 当前没有实现的功能

`v0.2.0` 仍不宣称已经实现以下能力：

- ADASYN/SMOTE 过采样；
- KS 检验；
- 在线模型服务；
- 数据库连接器；
- 远程数据上传；
- 自动策略上线。

这些属于后续规划，不是当前版本的已交付功能。

## 技术栈

- Python 3.11+
- Polars、PyArrow：本地列式数据处理；
- Pydantic：配置和数据契约；
- LightGBM、scikit-learn、SciPy、statsmodels：规则发现和统计计算；
- Typer：CLI；
- 官方 MCP Python SDK：本地 stdio 两阶段 Host 决策服务；
- PyYAML：项目配置。

## 快速开始

### 1. 安装

macOS/Linux：

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

开发和测试依赖：

```bash
./.venv/bin/python -m pip install -e '.[dev]'
```

### 2. 运行公开合成数据示例

该示例是确定性的，不需要私有数据：

```bash
mkdir -p data/synthetic
./.venv/bin/riskprobe synthetic \
  --output data/synthetic/behavior.parquet \
  --rows 5000 \
  --seed 42
./.venv/bin/riskprobe inspect \
  --config configs/synthetic.example.yaml \
  --runs-dir runs
./.venv/bin/riskprobe run \
  --config configs/synthetic.example.yaml \
  --runs-dir runs
```

结果写入本地 `runs/` 目录，该目录已被 Git 忽略。

### 3. 使用自己的本地 Parquet

复制公开示例并创建一个不提交到 Git 的本地 `ProjectConfig` YAML，在其中明确填写只读 Parquet 路径、实体/时间/分层/目标角色和特征配置。CLI 使用 `--config` 读取该文件；MCP 则在进程启动时通过 `--config` 接收它。数据路径不会作为 MCP tool 参数暴露，RiskProbe 也不会上传或修改源 Parquet。

## MCP 和 Agent 使用

RiskProbe MCP 是本地 stdio 服务，不是 HTTP 服务。启动时必须固定本地配置、运行目录、状态目录和 Host 身份：

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

配置模板：

```text
configs/mcp/mcp.example.json       # 通用 MCP JSON
configs/mcp/codex.example.toml     # Codex 配置
configs/mcp/trae.example.json      # Trae 配置
```

将模板中的占位路径替换为用户自己的项目配置和私有运行目录。标准工作流为：

```text
riskprobe_get_decision_context(idempotency_key)
→ Host 基于完整 findings 和 policy allowlist 选择 action_codes
→ riskprobe_submit_decision_proposal(
    同一 idempotency_key,
    原样 context_id,
    完整 diagnosis_evidence_ids,
    action_codes
  )
→ terminal agent_result
```

服务端完整顺序固定为 `inspect → diagnose → discover → recommend → review`。Host 不能调用底层分析函数、传入路径、删减诊断证据或绕过 review。

### 有界 Agent 编排与本地知识引用

Agent 使用固定 typed 工具顺序和确定性 Reviewer。仅当审核原因属于 `missing_evidence`、`missing_diagnosis` 或 `evidence_mismatch` 时允许一次受控全量重跑；权限、隐私、工具失败和 Grade-B 生产动作不会自动重试。

本地 RAG 使用 sealed citation index，返回文档 ID、内容 hash、标题和相关性分数等 metadata-only citation。`orchestrate_with_citations` 在 AgentResult 完成后单独查询 citation；知识引用不能替代业务 evidence、改变 action proposal 或绕过 ProposalValidator。

### 客户端对照

| 客户端 | 接入方式 | RiskProbe 核心能力 | 原生 RiskProbe Agent |
|---|---|---:|---:|
| Kiro | `.kiro/agents`、`.kiro/skills`、workspace MCP | 完整 | 支持 |
| Codex | MCP TOML + `AGENTS.md` | 完整 | 使用 Codex 指令 |
| Trae | 手动 MCP JSON + System Prompt | 完整 | 使用 Trae 指令 |
| 其他 MCP 客户端 | 本地 stdio MCP + 项目提示词 | 完整 | 取决于客户端 |
| 不支持 MCP 的客户端 | Python 包和 CLI | 仅 CLI | 不支持 |

详细说明：

- [`docs/cross-client-usage.md`](docs/cross-client-usage.md)
- [`docs/agent-system-prompt.md`](docs/agent-system-prompt.md)
- [`AGENTS.md`](AGENTS.md)
- [`docs/riskprobe-agent-technical-guide.md`](docs/riskprobe-agent-technical-guide.md)

## 隐私和安全边界

- Parquet 只读，路径只由用户在本地 `ProjectConfig` 中声明，并在 MCP 启动时固定；
- 输出只包含聚合指标，不输出实体值、样本行、原始日志、真实路径或明细行；
- 工作流不执行 Shell、任意 SQL/Python、网络访问或自动策略上线；
- 默认在受限聚合字段中展示已确认分层列的真实机构名，并保留 `institution_token`；
- 如需隐藏真实机构名，可配置：

```yaml
privacy:
  expose_segment_values: false
```

- B 级数据表现窗口未知，`Stable` 不等于严格 OOT 或生产就绪；
- 机构内规则只作为验证和人工复核证据，不自动推广到全局。

## 仓库结构

```text
src/riskprobe/             确定性引擎、CLI、MCP 服务
configs/                   示例配置和 MCP 模板
.kiro/                     Kiro Agent、Skill 和 workspace MCP 配置
AGENTS.md                  跨客户端通用项目规则
docs/                      跨客户端说明和技术指南
tests/                     单元、集成和安全边界测试
```

## 后续规划

- 在明确数据契约、防泄漏和可复现性后，增加可选的类别不平衡处理策略；
- 在定义统计解释和输入契约后，补充 KS 等互补检验；
- 继续完善 Python 打包、CLI 和多客户端适配；
- 只有在完成认证、租户隔离、审计和数据治理设计后，才考虑远程部署。

## 开发验证

```bash
./.venv/bin/python -m pytest --disable-warnings --maxfail=1
./.venv/bin/ruff check src tests
```

贡献代码时请保持：本地优先、确定性随机种子、显式角色确认、聚合输出、无自动策略上线。
