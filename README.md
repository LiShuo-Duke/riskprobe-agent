# RiskProbe Agent

**面向风控规则发现、模型监控、PSI 漂移检测和机构稳定性分析的本地隐私安全 Agent。**

RiskProbe 是一个 **local-first** 的 Python 风控分析工具包，提供确定性规则引擎、CLI 和标准 stdio MCP 服务。它可以被 Kiro、Codex、Trae 以及其他 MCP 客户端调用。

> **当前版本：** `v0.1.0` 首个公开版本。RiskProbe 只在本地读取用户明确允许的 Parquet 数据，不上传数据，不暴露实体级明细。

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

### 3. 多机构稳定性分析

RiskProbe 采用“全局优先”的顺序：

```text
全机构合并 Train 发现全局规则
→ Test/Holdout 按机构验证
→ 判断 Stable / Local / Unstable / Suspicious
→ 只对满足门槛的 Local 机构做条件式局部发现
```

机构内规则只用于稳定性验证和人工复核，不会自动升级为全局规则或上线策略。

### 4. PSI 和聚合漂移监控

支持检测：

- Schema 变化；
- Missingness 缺失率变化；
- PSI 分布漂移；
- Population / 机构占比变化；
- Label 正类率变化；
- Rule Lift 衰减。

### 5. 根因诊断（Root-Cause Diagnosis）

对告警生成聚合根因 TOP3，支持以下维度：

```text
feature / family / segment / label / rule / schema
```

输出贡献度、排名和数值证据，不返回用户实体、样本行或原始明细。

### 6. 只读 Parquet 数据接入

直接使用本地 Parquet 时，必须经过：

```text
schema 预览
→ 用户确认实体、时间、机构/分群、目标列
→ 候选特征预览
→ 用户确认精确特征清单
→ 只读注册
```

RiskProbe 不根据列名猜测角色，不自动补充未确认特征，也不修改源 Parquet。

### 7. 跨客户端 Agent

核心 MCP 服务使用标准本地 stdio 协议：

- Kiro：原生 Agent + Skill；
- Codex：MCP 配置 + `AGENTS.md`；
- Trae：手动 MCP 配置 + 通用 System Prompt；
- 其他 MCP 客户端：配置同一个 MCP 服务即可；
- 不支持 MCP 的客户端：仍可使用 Python 包和 CLI。

### 当前没有实现的功能

`v0.1.0` 不宣称已经实现以下能力：

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
- FastMCP：本地 stdio MCP；
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

先设置本地数据白名单：

```bash
mkdir -p "$HOME/riskprobe-data"
export RISKPROBE_ALLOWED_DATA_ROOTS="$HOME/riskprobe-data"
```

将 Parquet 放入该目录。RiskProbe 不会上传、修改或返回实体级数据。

## MCP 和 Agent 使用

RiskProbe MCP 是本地 stdio 服务，不是 HTTP 服务：

```bash
./.venv/bin/python -m riskprobe.mcp_server
```

配置模板：

```text
configs/mcp/mcp.example.json       # 通用 MCP JSON
configs/mcp/codex.example.toml     # Codex 配置
configs/mcp/trae.example.json      # Trae 配置
```

将模板中的占位路径替换为用户自己的项目路径和数据目录。标准工作流为：

```text
inspect_local_parquet_schema
→ 确认实体 / 时间 / 机构 / 目标角色
→ preview_local_parquet_features
→ 确认精确特征列
→ register_local_parquet
→ inspect_dataset
→ discover_rules(objective="risk")
→ validate_rules
→ detect_anomalies
→ diagnose_anomaly
→ build_report
```

`discover_rules` 不接受非空 `constraints`，发现阈值来自已注册项目配置。没有真实时间列时只能进行随机 Train/Test 验证，不得称为严格 OOT。

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

- Parquet 只读，并且必须位于 `RISKPROBE_ALLOWED_DATA_ROOTS` 白名单目录；
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
