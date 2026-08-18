# RiskProbe Agent 完整流程技术说明

> 用途：项目介绍、面试准备与本地复现。本文区分**当前已实现能力**与**真实数据运行边界**，不把未执行的真实公司运行或业务效果描述为已交付结果。

## 1. 定位、边界与架构

RiskProbe Agent 是一个面向风控规则发现、证据验证与异常感知的本地可审计系统。它解决的重点不是让大模型随意“猜规则”，而是在多重搜索、标签窗口不完整、客群差异、时间变化与数据质量约束下，用可复核证据筛掉偶然规则。

首期明确**不做** SQL/Spark SQL 或数据仓库连接、Web 前端、在线流处理、自动策略上线、任意 Python 执行、多 Agent 角色扮演、强化学习、外部 LLM API 依赖，也不替代既有评分模型。系统不会自动修改、删除或上线任何风控策略，最终业务决策始终由人工完成。

```text
External Host（Kiro / Codex / Trae）
  只读取聚合 DecisionContext，并选择 policy allowlist 中的 action_codes
                         │ 两阶段、仅两个 MCP tools
Local stdio MCP
  riskprobe_get_decision_context / riskprobe_submit_decision_proposal
                         │ 固定 inspect → diagnose → discover → recommend → review
确定性 Risk Engine（当前核心引擎已实现）
  数据契约、特征目录、规则发现、证据验证、监控、不可变运行产物
                         │
本地只读 Parquet / DataFrame
```

职责边界必须严格保持：**确定性引擎**计算规则阈值、覆盖率、Lift、置信区间、统计校正、漂移、根因和受控建议；**Host**只基于完整聚合证据选择允许的 action code。MCP 工具不接受路径、任意代码、身份或预算参数，Host 不能调用底层分析函数、删减诊断证据或跳过 review。即使没有 MCP Host，CLI 仍可运行本地确定性分析，但不能作为绕过 Host 决策门控的通道。

## 2. 数据安全、Git 隔离与输入契约

输入是已经在进入 RiskProbe 前完成脱敏的本地 Parquet 宽表。`DatasetConfig` 要求：路径为本地路径，后缀必须是 `.parquet`，并且 `read_only` 固定为 `true`；拒绝网络 UNC 路径、URL 和非 Parquet 文件。服务运行时为输入建立私有临时快照，源 Parquet 不被修改；运行结束清理快照与失败运行的部分产物。

公开仓库与公司环境必须隔离：不得提交公司数据、真实字段名、真实配置、规则阈值、运行结果、密钥、原始日志、样本行或实际公司路径。Kiro 上下文也不应挂载数据目录。公开项目只能使用公开数据与合成数据，且不得把公司实验的效果、覆盖机构数或提效比例写成公开结论。

面向 MCP 的数据边界在服务启动时固定。操作者通过 `python -m riskprobe.mcp_server --config ... --runs-dir ... --state-dir ...` 提供一个本地 `ProjectConfig`、私有运行目录和状态目录；服务使用 `ProjectConfig.from_yaml()` 完成数据契约校验，并把该配置中的唯一 `dataset_id` 绑定到当前 stdio 进程。路径、字段映射、身份、角色和查询预算都只能作为受控启动参数提供，不能通过 MCP 工具动态传入或改写。

底层 `DatasetRegistry`、数据适配器和 CLI 仍可供本地确定性流程使用，但它们不构成 MCP 工具，也不能由 Host 调用来新增数据集、检查路径、预览列或绕过决策门控。MCP 工具面固定为两个高层 Host-decision 操作；重连或切换数据集必须由本地操作者停止服务、验证新的 `ProjectConfig` 后重新启动。

### 2.1 启动时固定配置门控

本地操作者必须在启动 MCP 前完成实体、截面、分层、目标和建模特征配置；Host 不读取 Parquet schema，也不猜测或修改字段角色。没有真实时间列时，配置必须显式设置 `time_validation_enabled=false`，系统只执行固定随机分层验证，不得声称严格 OOT、时间外推或无时间穿越。

配置或数据契约校验失败时服务启动或固定流水线直接失败，不向 Host 返回可供猜测修补的文件信息。服务启动后，第一阶段只接受公开稳定的 `idempotency_key`，在本地固定执行 `inspect → diagnose → discover`；第二阶段只接受原样上下文、完整诊断证据 ID 和策略允许的 action codes，并固定执行 `recommend → review`。Host 不能传入路径、数据集 ID、任意代码、用户级筛选、身份或预算，也不能跳步或删减诊断证据。

所有返回均为聚合 DecisionContext 或终态 AgentResult：递归拒绝实体标识、样本行、原始日志、真实路径和未授权明细，并抑制低于最小分组阈值的结果。元数据 C/D 或其他 fail-closed 条件会在上下文生成前阻断；B 级上下文仅允许分析建议，不能解释为生产就绪。

## 3. 配置、角色列和元数据等级

项目配置的四个角色列为：

| 角色 | 配置字段 | 含义 |
|---|---|---|
| 实体 | `columns.entity` | 已脱敏实体标识；用于数据契约，不进入监控快照或公开输出 |
| 截面 | `columns.snapshot` | 客户指定特征截面/回溯基准日 |
| 分层 | `columns.segment` | 机构或客户分群；`segment_display_name` 只允许 `institution` 或 `customer_segment` |
| 标签 | `columns.target` | 坏账标签；正类固定 `positive_value=1`、`positive_meaning=bad_debt` |

`target.performance_window_days` 为正整数或空；`snapshot.meaning` 只允许 `customer_specified_feature_cutoff` 或 `public_relative_reference`。特征族由前缀映射或显式 YAML 目录定义，角色列会从特征候选中排除。

元数据等级是结论的使用边界：

| 等级 | 语义 | 处理 |
|---|---|---|
| A | 标签含义、截面日期、表现窗口、特征截止边界完整 | 可做严格时间验证 |
| B | 可以发现规则并按时间切片验证，但仍有标签成熟度等缺口 | 输出限制；不得称为严格 OOT、无时间穿越或可直接上线 |
| C | 疑似标签穿越或样本构造异常 | 仅允许数据诊断，不输出规则结论 |
| D | 数据契约不通过 | 停止分析 |

当前 `ProjectConfig.metadata_grade` 会在已知 `performance_window_days` 时给出 A，否则给出 B；已实现的 Agent/MCP 编排在低于 B 时阻断稳定规则结论，并对 C/D 元数据状态执行完整阻断。B 级结果会保留标签成熟度限制，不得称为严格 OOT、无时间穿越或可直接上线。

## 4. 从 inspect 到分区：当前核心流程

1. **配置与 inspect。** `riskprobe inspect` 读取本地 YAML 后执行数据概况检查：角色列、标签取值与正类率、日期解析、重复样本、常量/高缺失列、异常值、单一标签切片及元数据缺口。输出是安全的聚合 profile，不输出实体明细。
2. **建立特征目录。** 角色列之外的列优先按 `features.explicit_catalog` 选择；没有显式目录时按已配置的族前缀选择。设计层的行为目录可记录 `feature_name | family | window_days | aggregation | source | value_type`，并检查累计窗口单调性（`7d ≤ 30d ≤ 90d ≤ 180d ≤ 365d`）、比例范围 `[0,1]`、非负间隔、零次数约束、无行为与未匹配的区别、特征包覆盖和相邻窗口高相关。无法可靠归类的设计语义是 `unknown`，不臆测业务含义。
3. **读取与划分。** Parquet 处理使用列裁剪；发现阶段最多抽取 **50,000** 条 Train 行生成候选，最终验证仍在未改变分布的数据上完成。时间验证打开时，日期归一化、排序并在日期组边界附近划为 Train/Test/Holdout，目标比例为 **60% / 20% / 20%**；空截面日期行不参与时间分区并被记录。若只有两个日期组，划分为 Train/其余 Test，Holdout 为空；少于两个时相应分区为空。
4. **非时间划分。** `time_validation_enabled=false` 时，固定 `random_state=42` 的分层随机划分为 **70% Train / 30% Test**，没有 Holdout。规则阈值只可由 Train 产生，Test、Holdout 和时间切片只能验证，不能参与阈值搜索。

## 5. 规则发现：从阈值到二阶组合

当前发现器仅处理数值特征；缺列会报错，空数据、单一标签或没有正类时返回空候选。默认 `min_support=0.05`、随机种子固定 **42**。

### 5.1 单变量候选

每个数值特征汇集三类阈值后去重排序：

- 经验分位点：**0.10、0.25、0.50、0.75、0.90**；
- 浅层 `DecisionTreeClassifier`：`max_depth=2`，`min_samples_leaf=ceil(0.05 × Train 样本数)`，`random_state=42`；
- 单特征 LightGBM：`n_estimators=30`、`max_depth=2`、`num_leaves=4`、`learning_rate=0.05`、`deterministic=true`、`force_col_wise=true`、`random_state=42`、`n_jobs=1`。

每个阈值生成 `feature <= threshold` 与 `feature > threshold` 两种规则。单变量规则支持度必须满足：

```text
ceil(0.05 × N_train) ≤ 命中数 ≤ floor(0.95 × N_train)
```

候选按 Lift、支持样本数和规范化表达式稳定排序，最多保留 **100** 条单变量规则（`max_single_rules=100`）。规则 ID 是规范化条件 JSON 的 SHA-256 前 **12** 位，因此相同条件顺序不会产生不同 ID。

### 5.2 Beam 二阶组合

二阶搜索使用所有单变量候选形成宽度 **20** 的 Beam（`beam_width=20`）：按特征轮转挑选，避免单一特征占满 Beam。Beam 内仅将**不同特征**的两个条件做 AND 组合；组合规则只要求命中数不少于 `ceil(0.05 × N_train)`，不再施加 95% 上限。若组合掩码与任一单条件掩码完全相同，则丢弃，避免伪组合。

二阶候选先确保每个特征对至少有一个代表，再按同一排序规则补齐，最多 **50** 条（`max_pair_rules=50`）。当前实现的二阶预筛是 Lift/支持度排序与 Beam 多样化，并**没有**实现设计文字中“高 IV 或高增益”作为额外硬门槛；面试或介绍时应如实说明这一点。单变量与组合规则的 `origin` 分别为 `discovery_single`、`discovery_pair`。

## 6. 规则验证、公式和分级

对规则命中掩码 `M` 与正类标签 `Y=1`，当前指标定义为：

```text
coverage      = |M| / N
base_bad_rate = |Y=1| / N
hit_bad_rate  = |M ∩ (Y=1)| / |M|                 （未命中时为 0）
non_hit_rate  = |¬M ∩ (Y=1)| / |¬M|               （未命中组为空时为 0）
Lift          = hit_bad_rate / base_bad_rate
Precision     = hit_bad_rate
Recall        = |M ∩ (Y=1)| / |Y=1|
```

同时以命中/未命中的 2×2 列联表计算双侧 Fisher 精确检验 p 值。所有规则的 **Train p 值**统一进行 Benjamini–Hochberg FDR 校正；默认显著性水平 `alpha=0.05`。

Test Lift 的 Bootstrap CI 使用正、负类分层有放回重抽样：每轮保留各自原样本数，固定 `seed=42`，默认 **500** 轮，取 **95%** 区间的 2.5% 与 97.5% 分位数。最小分组样本量默认 **100**；小于该值的分层不计算效果。分层和时间切片均在 Test 上评估：时间切片使用 `%Y-%m` 月桶。

时间衰减定义为：

```text
max_time_decay = max(0, (Train Lift - min(月度 Lift)) / Train Lift)
```

默认最大可接受衰减为 **0.30**，分层方向一致率为 `Lift > 1` 的可用分层占比，默认下限 **0.60**。若 Train/Test 小于 100、没有可用分层、存在单标签分层，或时间验证启用但没有可用时间切片/存在时间切片限制，则标记样本不足。

分级顺序如下：

- **Suspicious**：BH 校正后 p 值 `>0.05`，或 CI 下界 `≤1`，或样本不足；
- **Unstable**：不属于 Suspicious，且 `max_time_decay>0.30`；
- **Local**：不属于以上两类，存在至少一个 `Lift>1` 分层且分层一致率 `<0.60`；
- **Stable**：其余情形，可进入人工复核而不是自动上线。

B 级数据会在证据卡和运行限制中写入“标签表现窗口未知”；这代表跨时间切片验证，而非严格生产 OOT 验证。

## 7. 参考快照：脱敏聚合与确定性 ID（Task 1 已完成）

参考快照只保留聚合信息：行数、整体正类率、达到最小样本量的分层计数、数值特征的缺失率/零值率/分箱边界/直方图计数，以及规则的 Test 覆盖率、命中坏账率、Lift。实体列既不选择也不序列化。创建时间固定为 `1970-01-01T00:00:00Z`，保证相同输入不会因时钟不同而改变结果。

标识符假定已经是稳定脱敏业务编码，但仍执行路径型拒绝：循环 percent 解码直到稳定，所以重复编码（例如 `%252F`）也会暴露为 `/` 并被拒绝；`file:` URL、`/`、`\\` 以及形如 `C:` 的盘符路径一律拒绝。规则 ID 还要求唯一。分层小组仅在 `count >= min_group_size` 时保存，默认阈值为 **100**。

数值特征以 **0%、25%、50%、75%、100%** 分位点建固定分箱，重复边界会去重；常量特征形成一个计数桶。`snapshot_id` 是对以下规范化、键排序、紧凑 JSON 的 **完整 SHA-256**：`dataset_id`、行数、正类率、经抑制的分层计数、特征聚合、规则聚合与固定创建时间。因此它可复现且不依赖样本行。

## 8. 监控检测器：Task 2 已实现并完成统一审查修复

Task 2 的 `detect_anomalies` 已实现固定的聚合检测器，并已修复统一审查发现的角色列、分层并集和最小样本边界问题。检测器使用 `ReferenceSnapshot` 保存的真实 `target_column`、`segment_column` 与 `min_group_size`，不再猜测自定义角色列；缺失 target/segment 会产生 critical Schema 告警，新增或消失且达到最小样本量的分层会参与 population 告警，低样本分组仍被抑制。

现有检测规则和固定阈值如下。`ReferenceSnapshot` 当前没有时间戳、月份或时间桶字段，因此监控 detect/diagnose 只在 dataset、segment、feature、family、rule 维度闭合；规则验证阶段独立的时间分区/月度验证不能被表述为监控时间漂移或月度根因结论。

| 检测 | 当前行为 | 阈值 |
|---|---|---|
| Schema | 参考特征缺失、或 dtype 家族改变 | 直接 `critical` |
| 缺失率 | 仅对“增加”的缺失率差值告警 | warning `0.10`，critical `0.25` |
| 分布漂移 | 用参考快照固定分箱计算 PSI | warning `0.20`，critical `0.30` |
| 标签率 | 当前整体正类率与参考正类率的绝对差 | warning `0.03`，critical `0.08` |
| 规则衰减 | `(reference_lift-current_lift)/reference_lift` 的正向衰减 | warning `0.20`，critical `0.40` |
| 分层占比 | 已有分层的当前占比减参考占比的绝对值 | `≥0.10` 时 warning |

PSI 的精确公式为：

```text
PSI = Σ_i (p_current,i - p_reference,i)
            × ln(p_current,i / p_reference,i)
```

其中参考与当前桶概率均下限截断为 `1e-6`，当前数据始终沿用参考快照的边界。两个及以上同一特征族的缺失率告警会额外生成族级告警；严重等级取其中最高等级。告警 ID 为 `SHA-256("alert_type|scope|scope_value|metric")` 的前 **12** 位，输出排序固定。

## 9. Plan 2 Task 3–8：已实现的本地监控、诊断、安全工具与 Agent 编排

以下能力已在本地实现并通过统一审查后的全量测试；所有能力仍保持本地-only，不提供网络服务、不上传数据、不使用 SQL 或外部模型 API。由于当前 `ReferenceSnapshot` 不保存时间/月度聚合字段，监控诊断闭合在 dataset、segment、feature、family、rule 维度；时间/月度监控不是当前已交付能力。

### Task 3：六类有真值漂移注入与评分（已实现）

已实现接口为 `inject_drift(frame, scenario, seed) -> InjectedDrift` 与 `evaluate_alerts(alerts, truth, top_k=3, diagnoses=()) -> DetectionScore`。六类漂移严格限定为：

1. `missingness`：在目标切片按 `magnitude` 置空；
2. `numeric_shift`：数值增加 `magnitude × reference_std`；
3. `population_shift`：对目标机构按 `magnitude` 过采样；
4. `label_shift`：对目标切片按 `magnitude` 将 `0→1` 翻转；
5. `schema`：删除目标列；
6. `rule_decay`：对命中目标规则的正类按 `magnitude` 将 `1→0` 翻转。

注入返回新 DataFrame，原始 DataFrame 不变；同一 seed 必须可复现，并随结果写出机器可读真值。检测评分以 `(alert_type, scope_value)` 匹配真值，输出 precision、recall、`false_positive_rate`（存在 TN 时）或 `None`（无 TN 时）、`false_discovery_rate` 与根因 Top-**3** 命中率。

### Task 4：根因贡献排序（已实现）

已实现接口为 `diagnose_alerts(alerts, reference, current_frame, catalog, top_k)`。对特征缺失率或 PSI 告警，按分层计算：

```text
contribution = abs(current_metric - reference_metric) × current_share
```

诊断会显式保留 feature 根因，并将同族特征贡献聚合为 family 根因；segment/family/feature/target/rule/schema 的比较维度与漂移真值保持一致。当前参考快照没有时间/月度字段，因此不伪造月份根因；若未来实现时间诊断，必须先在模型中加入明确时间聚合字段和定义。

### Task 5：底层注册表与安全输出门控（已实现，不暴露为 MCP）

已实现 `DatasetRegistry.from_yaml()`、`get_config(dataset_id)`、`assert_safe_payload` 与 `suppress_small_groups`。这些能力供确定性服务、CLI 和本地适配流程复用；外部 Host 不能直接调用注册表，也不能通过 MCP 新增数据集或传入路径。MCP 进程只使用启动时已验证的单个 `ProjectConfig`，返回前继续执行值级隐私检查和最小样本抑制。

### Task 6：本地 MCP 两阶段两工具（已实现）

本地 stdio `MCPServer` 只注册以下两个工具：

```text
riskprobe_get_decision_context(
  idempotency_key,
)

riskprobe_submit_decision_proposal(
  idempotency_key,
  context_id,
  diagnosis_evidence_ids,
  action_codes,
)
```

`riskprobe_get_decision_context` 启动一次固定服务编排，在本地完成 `inspect → diagnose → discover`，返回 `phase="awaiting_proposal"` 的聚合 `DecisionContext`。上下文包含完整 `findings`、`diagnosis_evidence_ids`、元数据等级、规则/问题公开 ID、有效期、组件版本和 `policy`；其 `context_id` 由规范化内容确定性派生。相同会话只接受同一个公开 `idempotency_key`。

Host 必须覆盖全部 findings，并从 `policy.allowed_action_codes` 中选择满足 `min_action_count`/`max_action_count` 的不重复 action codes；B 级只能使用 `grade_b_allowed_action_codes`。`riskprobe_submit_decision_proposal` 要求同一个幂等键、原样 `context_id`、完整未修改的 `diagnosis_evidence_ids` 和所选 action codes。校验通过后服务固定完成 `recommend → review`，只返回 `phase="terminal"` 的 `AgentResult`。上下文过期、ID/证据不一致、action 越权、重复提交不一致或预算/状态失败均统一 fail-closed。

配置路径、运行/状态目录、provider 身份、principal、role 和查询预算只存在于服务启动参数中，不属于 MCP 调用参数。底层 registry、schema preview、规则发现、监控、报告和 CLI 能力都没有注册为额外 MCP tools，因此 Host 不能跳过固定状态机或自行拼接分析链路。

### Task 7：Kiro Agent 最小权限（已实现）

workspace Agent 名为 `riskprobe`，只暴露 `@riskprobe`，且 MCP allowlist 精确限定为 `riskprobe_get_decision_context` 与 `riskprobe_submit_decision_proposal`；明确 deny `shell`、`fs_read`、`fs_write`、`web_fetch` 和 `web_search`。Agent 不直接读取 Parquet、配置、运行产物或真实路径。

SOP 为：生成一个公开稳定的 `idempotency_key` 并调用第一阶段工具；读取完整聚合上下文和全部 findings；按 policy 与元数据等级选择允许的 action codes；用同一幂等键、原样上下文 ID 和完整证据集合调用第二阶段工具一次；仅接受 terminal 结果并解释 review、证据、重试次数与限制。任一门控失败即停止，不更换 ID 重试，也不通过 CLI、文件系统或其他工具绕过。建议始终进入人工复核，不能自动修改或上线策略。

### Task 8：监控 CLI 与端到端评估（已实现）

已实现三条命令：

```bash
riskprobe snapshot --config PATH --runs-dir PATH
riskprobe monitor --reference-run-id ID --current-config PATH --runs-dir PATH
riskprobe evaluate-drift --config PATH --runs-dir PATH --seed INTEGER
```

`evaluate-drift` 会逐场景执行六种注入、检测与诊断，输出总体和分场景 Precision、Recall、FDR 及根因 Top-3 命中率；由于没有 TN，`false_positive_rate` 明确为 `null`。它写入 `anomaly_alerts.json`、`diagnoses.json`、`drift_evaluation.json`，且输出目录要求为仓库外 owner-private 路径。

## 9.1 Plan 3：数据适配、公开/公司运行与简历证据闭环（代码已实现；真实公司运行未执行）

Plan 3 已实现公开多表信用数据和公司本地脱敏 Parquet 宽表到 Plan 1 统一契约的代码、CLI、虚构示例配置及运行手册；新增模块的基础测试已通过。真实公司 Parquet、真实本地配置、人工基线、运行产物与量化简历证据均**未执行且不得视为已交付结果**。Plan 1 完成是数据适配前提；Plan 2 的底层注册表、输出门控、两工具 MCP 与 Kiro Agent 已完成，可用于合成漂移和受限本地 Host 决策验证，其中 registry 和 CLI 能力不暴露为 MCP tools。

### 9.1.1 公开 Home Credit：本地多表聚合为统一行为宽表

实现模块和接口为：

```python
HomeCreditPaths.from_directory(path) -> HomeCreditPaths
prepare_home_credit(paths, output_path, seed=42) -> HomeCreditPreparationResult
```

输入目录必须存在 `application_train.csv`，并至少存在一张历史表：`previous_application.csv`、`installments_payments.csv`、`POS_CASH_balance.csv`、`credit_card_balance.csv`、`bureau.csv`。适配器不得读取 `application_test.csv`、结果型预测文件或未白名单字段；使用 Polars `LazyFrame` 进行列裁剪并按 `SK_ID_CURR` 本地聚合。输出满足 Plan 1 契约，列顺序起始为 `entity_id`、`target`、`customer_segment`，追加行为特征和固定 `snapshot_date="public_relative_reference"`；不保留原始 `TARGET` 名称及未白名单列。

| 历史源 | 列白名单 | 相对历史窗口与聚合 |
|---|---|---|
| previous application | `SK_ID_CURR`、`DAYS_DECISION`、`AMT_APPLICATION`、`AMT_CREDIT`、`NAME_CONTRACT_STATUS` | 近 **30/90/365 天**申请次数、申请/授信金额、拒绝率 |
| installments | `SK_ID_CURR`、`DAYS_INSTALMENT`、`DAYS_ENTRY_PAYMENT`、`AMT_INSTALMENT`、`AMT_PAYMENT` | 近 **30/90/365 天**记录数、逾期天数均值、少还比例 |
| POS cash | `SK_ID_CURR`、`MONTHS_BALANCE`、`SK_DPD`、`SK_DPD_DEF` | 近 **3/6/12 月**活跃月数、DPD 均值和最大值 |
| credit card | `SK_ID_CURR`、`MONTHS_BALANCE`、`AMT_BALANCE`、`AMT_PAYMENT_CURRENT`、`SK_DPD` | 近 **3/6/12 月**余额均值、支付/余额比、DPD 最大值 |
| bureau | `SK_ID_CURR`、`DAYS_CREDIT`、`AMT_CREDIT_SUM`、`AMT_CREDIT_SUM_DEBT`、`CREDIT_ACTIVE` | 近 **90/365 天**查询数、活跃授信数、债务/授信比 |

其中 `application_train.csv` 仅读取 `SK_ID_CURR`、`TARGET`、`NAME_INCOME_TYPE`、`DAYS_BIRTH`、`AMT_INCOME_TOTAL`。Home Credit 的 `DAYS_*` 和 `MONTHS_BALANCE` 仅用于相对历史窗口，绝不伪造绝对申请日期、观察日期或表现日期。公开配置将固定如下，因此公开实验只能做固定种子 Train/Test 与跨客群验证，不得声称严格 OOT、时间切片、跨机构或线上效果：

```yaml
dataset:
  id: home_credit_public
columns:
  entity: entity_id
  snapshot: snapshot_date
  segment: customer_segment
  target: target
target:
  positive_value: 1
  positive_meaning: bad_debt
  performance_window_days: null
snapshot:
  meaning: public_relative_reference
segment_display_name: customer_segment
time_validation_enabled: false
```

CLI 是：

```bash
riskprobe prepare-home-credit --input-dir PATH --output PATH
```

它只输出表数、行列数、特征族与输出路径，不输出客户样本；用户自行下载 CSV，CSV 和生成的 Parquet 均不随 Git 分发。

### 9.1.2 公司 Parquet：只读预检与 64 列特征批次

实现接口为：

```python
preflight_company_dataset(config) -> CompanyPreflight
plan_feature_batches(schema, catalog, batch_size=64) -> tuple[FeatureBatch, ...]
```

`preflight_company_dataset` 先通过 `scan_parquet().collect_schema()` 验证实体、截面、分层、标签四个角色列和特征族，再仅列裁剪扫描角色列，以计算行数、标签率、日期范围、分层计数、特征族计数和元数据等级。它不读取全部宽表、不导出样本、不写回源文件；验收时源 Parquet 的 `mtime_ns` 前后必须相同。

`FeatureBatch` 保存 `index`、`features`、`required_columns`。`required_columns` 固定附加 entity/snapshot/segment/target，`features` 不含角色列。默认 `batch_size=64`，**977** 个特征需切成 **16** 个顺序稳定、无遗漏、无重复的批次。

仓库只提供虚构示例：`anonymous_id`、`cutoff_date`、`org_code`、`bad_label`，以及 `ord_x_`、`brw_x_`、`multi_x_`、`emb_x_` 特征前缀。真实配置固定为被忽略的 `configs/company.local.yaml`，报告的分层显示名固定为 `institution`。`cutoff_date` 是客户特征截止日，不是申请日或坏账日；表现窗口未知时仍为 B 级，不能称为严格 OOT。

CLI：

```bash
riskprobe preflight-company --config configs/company.local.yaml
```

它只输出聚合结果（行数、特征数、特征族计数、批次数、标签率、分层数、元数据等级和限制），不显示真实机构、实体、样本行、规则或真实路径。公司 Parquet、真实字段映射、真实配置、运行产物和内部报告均必须在 Git 之外；运行前后用 `git check-ignore` 与 `git ls-files` 验证隔离。

### 9.1.3 基准记录：实测计时、原始计数和可追溯性

已实现模块 `benchmarking.py` 以 `time.perf_counter` 记录 `inspect`、`discover`、`validate`、`monitor`、`report` 阶段。每个 `BenchmarkRecord` 至少保存：run/task/dataset ID、测量时间、代码版本、配置哈希、数据指纹、人工分钟、Agent 分钟、阶段耗时、候选规则数、证据通过数、人工复核/接受数、异常 TP/FP/FN、根因 Top-3 命中数和根因案例数。所有计数和分钟必须非负，接受数不超过复核数。

关键指标必须由原始计数推导，而非人工填写舍入比例：

```text
precision                 = TP / (TP + FP)
recall                    = TP / (TP + FN)
false_positive_rate       = FP / (FP + TN)               （TN 可用时）
root_cause_top3_hit_rate  = top3_hit_count / root_cause_case_count
rule_review_acceptance    = accepted_rule_count / reviewed_rule_count
```

没有实测 `manual_minutes` 时，程序必须拒绝计算“提效”。基准 CLI 读取人工预先填写、且不会被程序改写的本地 baseline：

```bash
riskprobe benchmark --config PATH --runs-dir PATH --baseline-record PATH
```

记录只写入被忽略的运行目录，终端不打印公司规则表达式。

### 9.1.4 简历证据：三次真实任务是量化表述的最小门槛

已实现模块 `resume_evidence.py` 只聚合本地 `BenchmarkRecord`。生成量化公司经历前，必须有至少 **3** 个不同 `task_id` 的完整任务，每条均含人工基线、Agent 耗时和原始计数；不足三次或字段缺失即失败，不得使用默认值、示例数字、估算或编造。总提效为总分钟加权而不是简单平均：

```text
efficiency_rate = (Σ manual_minutes - Σ agent_minutes) / Σ manual_minutes
```

规则接受率与异常/根因指标按跨任务原始分子分母汇总。`ResumeEvidence` 必须保留来源 `run_id` 列表；简历只可引用其中存在且可追溯的数字，缺一项就省略该句。CLI：

```bash
riskprobe resume-evidence --records-dir PATH --output reports/internal/resume_evidence.md
```

输出可含公开项目与公司实习两段草稿，但公司证据只可写入 Git 忽略的内部目录。

### 9.1.5 端到端运行顺序与验收边界

1. **公开闭环：** 用户在仓库外下载 Home Credit；按列白名单与相对窗口聚合为 Parquet；以 `public_relative_reference` 及非时间验证运行 Plan 1；Plan 2 完成后，再以合成漂移验证监控、根因和受限 Agent。公开结果只能报告 Train/Test 与跨客群结论。
2. **公司闭环：** 在仓库外保留脱敏 Parquet 和 `company.local.yaml`；只读预检后按 64 列分批运行确定性分析；人工复核并记录真实基线与阶段耗时；完成至少三次任务后才生成内部量化证据。
3. **Agent 闭环：** 仅在 Plan 2 的启动配置、隐私门控、两工具 MCP 和 Kiro 最小权限均完成后，`@riskprobe` 才可执行两阶段协议。第一阶段由服务固定完成 `inspect → diagnose → discover` 并返回完整聚合上下文；Host 依据全部 findings 与 policy allowlist 选择 action codes；第二阶段由服务固定完成 `recommend → review`。Agent 不能读取明细、任意路径、Shell、网络或写入数据，也不能把底层 registry/CLI 能力重新暴露为 MCP 或用来绕过 Host 决策门控。

Plan 3 的完整端到端门槛是：公开适配不伪造时间验证；公司预检不改写源文件；977 维为 16 个批次；公开/公司显示分别为 `customer_segment`/`institution`；三次真实任务前拒绝量化简历；私有数据、配置、结果和内部证据均不在 Git 中。当前通用实现、虚构示例、运行手册和基础测试均已具备；但真实公司 Parquet、人工基线、三至五次真实任务、运行产物及量化简历证据均不存在。故真实公司闭环与任何公司效果结论尚未完成。

## 10. 运行方式、产物、失败处理与可复现性

当前可用 CLI：

```bash
# 生成确定性的公开合成数据
riskprobe synthetic --output ./behavior.parquet --rows 5000 --seed 42

# 查看安全的聚合 profile
riskprobe inspect --config ./project.yaml --runs-dir ./runs

# 在 Train 上发现候选规则
riskprobe discover --config ./project.yaml --runs-dir ./runs

# 在本地创建或复用完整的不可变分析运行
riskprobe run --config ./project.yaml --runs-dir ./runs
```

Plan 3 的以下通用命令也已实现，但它们不代表任何真实公司运行已经发生：

```bash
# 将用户自行取得的公开 CSV 在本地聚合为 Parquet
riskprobe prepare-home-credit --input-dir /local/home-credit --output /local/home-credit-riskprobe.parquet

# 对本地只读公司 Parquet 做聚合预检
riskprobe preflight-company --config configs/company.local.yaml

# 以本地人工基线测量工作流；运行产物必须保持在 Git 忽略目录
riskprobe benchmark --config configs/company.local.yaml --runs-dir runs --baseline-record /local/baseline.json

# 仅在至少三项完整真实测量后生成 Git 忽略的内部草稿
riskprobe resume-evidence --records-dir runs --output reports/internal/resume_evidence.md
```

当前基础 `run` 运行有六项核心产物：

```text
runs/<run_id>/
├── manifest.json
├── metadata_report.json
├── data_profile.json
├── candidate_rules.parquet
├── evidence_cards.json
└── risk_report.md
```

`manifest.json` 记录数据指纹、配置指纹、代码版本、数据集 ID、时间验证开关、产物列表与完整性哈希。相同数据、配置和代码版本会命中同一运行 ID 并复用已有运行；规则排序、Bootstrap 种子和分层随机划分也固定。基础 `run` 运行保留上述六项核心产物；监控与 `evaluate-drift` 运行另外写入 `anomaly_alerts.json`、`diagnoses.json` 和 `drift_evaluation.json`。

CLI 对参数错误输出结构化 JSON（`argument_error`）；配置读取失败为 `configuration_error`；运行目录不可写为 `runs_directory_error`；数据检查、发现或完整运行失败分别输出 `inspection_error`、`discovery_error`、`run_error`。底层还会在缺少角色列、单标签切片、命中支持度不足、无正类、常量/全空特征、无效日期或不可用 Holdout 时停止、跳过相应指标或写入限制。发生异常时，未完成运行目录会清理；报告渲染失败时，设计目标是保留结构化中间产物以便重建。

## 11. 已人工核验的公开 UCI 基准

本地已运行并人工核验的公开基准来自 [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)。以下是可公开陈述的全部事实：

- 数据为 **30,000** 行、**23** 个特征，含 **7** 个教育分层；总体违约率 **22.12%**；
- 使用 **70/30** 随机分层划分，Train/Test 为 **21,000 / 9,000**；
- 产生 **150** 条候选规则及其证据卡；Top 测试规则的 Lift 为 **3.2837**、覆盖率 **6.011%**、命中违约率 **72.64%**、规则证据等级为 **Stable**；数据元数据等级为 **B**；
- 未进行时间验证，表现窗口未知；运行 ID 为 `acc8ea38446954df`；六项核心产物均存在，重跑会复用该运行。

这不是严格 OOT 验证，也不是线上收益证明：切分是随机分层而非按时间外推，未做时间验证，且标签表现窗口未知。因此不能由上述结果推断无时间穿越、上线可用性、坏账挽回金额、AUC/KS 改善或实际业务提效。

## 12. 当前状态矩阵

| 范围 | 当前状态 | 可对外准确表述 |
|---|---|---|
| Plan 1：本地确定性核心引擎 | **完成** | 已完成数据契约、inspect、特征目录、规则发现/验证、六项不可变运行产物和本地 CLI |
| Plan 2 Task 1：监控模型与参考快照 | **完成且审查通过** | 已具备不含实体明细、可确定性复现、包含特征/分层/规则聚合指标的参考快照 |
| Plan 2 Task 2：Schema、缺失、PSI、标签、分层与规则衰减检测 | **完成且审查修复通过** | 已实现角色元数据驱动的 Schema/缺失率/PSI/标签率/分层占比/规则 Lift 衰减检测；支持新增/消失分层与最小样本抑制 |
| Plan 2 Task 3–4：漂移注入、评分与根因 | **完成且审查修复通过** | 已实现六类可复现真值注入、逐场景评分、FDR/FPR 边界和确定性根因排序；当前不提供月份根因 |
| Plan 2 Task 5–8：注册表、隐私门控、MCP、Kiro Agent 与 CLI | **完成** | 已实现底层本地白名单、值级隐私门控、固定启动配置、两阶段两工具 MCP、最小权限 Kiro Agent 和监控 CLI；registry/CLI 不暴露为 MCP，仅本地 stdio，不提供网络服务 |
| Plan 3：公开/公司数据适配与证据 | **通用代码、CLI、虚构示例与运行手册已实现；真实公司执行待完成** | 已可在本地准备用户自行下载的 Home Credit CSV、预检只读公司 Parquet、规划 64 列批次、记录本地基准并仅从完整记录生成内部草稿；尚无真实公司 Parquet、人工基线、三次真实任务或量化提效，故不得称为公司闭环或业务结果已交付 |
| Kiro/MCP Agent | **本地实现** | 已提供本地 stdio MCP、`@riskprobe` Agent、精确两工具 allowlist 与权限 deny 规则；Host 只选择允许的 action codes，不连接外部服务 |
| 公开 UCI 基准 | **已运行** | 仅按第 11 节的已核验事实介绍：Top 规则为 Stable、数据为 B 级，且不是严格 OOT 或线上收益 |

面试或项目介绍应优先说明“确定性引擎输出证据，Agent 只做受限编排和解释”的架构取舍，并在每个效果数字旁保留数据划分、元数据等级、表现窗口与人工复核边界。

## 13. 两阶段 Host 决策协议（当前已实现）

MCP 服务在启动时读取一个固定 `ProjectConfig`，并把该数据集、运行目录、状态目录、provider 版本、principal、role 和查询预算绑定到当前本地 stdio 进程。外部 Host 不能通过工具传入路径、注册数据集、预览 schema、改变字段映射或覆盖预算。底层 registry、数据适配、监控和报告 API 仍是 Python/CLI 能力，但不注册为 MCP tools。

服务启动示例：

```bash
python -m riskprobe.mcp_server \
  --config /absolute/private/project.yaml \
  --runs-dir /absolute/private/runs \
  --state-dir /absolute/private/state \
  --provider-id kiro \
  --provider-version gpt5.6sol \
  --principal-id kiro-host \
  --role analyst \
  --max-queries 16
```

真实数据、真实配置、运行/状态目录和内部报告必须留在 Git 之外。更换数据集或启动边界时，由本地操作者停止服务、验证新配置并重启；不提供运行时 MCP 注册通道。

### 13.1 第一阶段：取得完整聚合决策上下文

Host 生成一个公开、稳定的幂等键，并调用：

```text
riskprobe_get_decision_context(idempotency_key)
```

服务在后台执行固定流程的 `inspect → diagnose → discover`，成功后返回：

```text
protocol_version = "riskprobe.host-decision.v1"
phase = "awaiting_proposal"
provider_id / provider_version
context = DecisionContext
```

`DecisionContext` 的 `privacy_class` 固定为 `aggregate`，并包含确定性 `context_id`、session/attempt、锚点节点、启动时绑定的 dataset ID、A/B 元数据等级、聚合行数与特征数、公开 issue/rule IDs、完整 `diagnosis_evidence_ids`、非空 `findings`、`policy`、签发/过期时间和组件版本。每个 finding 与一个诊断 evidence ID 对应，证据集合必须完整且唯一；`context_id` 由规范化上下文确定性派生，Host 不得自行构造或修改。

C/D 元数据、隐私检查、预算、状态恢复或流水线前置条件失败时，不返回可供 Host 猜测修补的部分上下文。第一阶段工具不接受 dataset ID、路径、角色列、目标、约束、身份或预算参数。

### 13.2 第二阶段：提交受策略约束的 Host proposal

Host 必须阅读全部 findings，再按 `context.policy` 做唯一一次受控选择：

- 只选择 `allowed_action_codes` 中的值，数量位于 `min_action_count` 与 `max_action_count` 之间，且不得重复；
- 元数据等级为 B 时，只能选择 `grade_b_allowed_action_codes`，并把结果限定为分析建议；
- 原样保留 `context_id` 和完整 `diagnosis_evidence_ids`，不得遗漏、替换或虚构证据；
- 在 `expires_at` 前使用与第一阶段相同的 `idempotency_key`。

提交接口为：

```text
riskprobe_submit_decision_proposal(
  idempotency_key,
  context_id,
  diagnosis_evidence_ids,
  action_codes,
)
```

服务端把载荷构造成来源固定为 `external_host`、版本固定为当前 provider version 的 `DecisionProposal`，重新校验幂等键、上下文、证据全集、action allowlist、数量、元数据等级、TTL 和状态。通过后确定性完成 `recommend → review`，只返回 `phase="terminal"`、原上下文 ID 和最终 `AgentResult`；完整完成序列为 `inspect → diagnose → discover → recommend → review`。

任何上下文、证据、action、有效期或幂等冲突都以 `host decision is unavailable` fail-closed。Host 不应更换 ID、删减证据、调用额外工具或通过 CLI/文件系统绕过；建议只供人工复核，永不自动修改或上线策略。

### 13.3 底层报告、机构分析与隐私边界

确定性服务和 CLI 仍可在本地生成规则发现/验证、漂移监控、根因诊断、全局与机构分析及六项不可变核心产物。这些能力用于构造聚合 findings、证据和审查结果，或供本地人工复核；它们不是可由 Host 单独调用的 MCP 报告工具。Host 不能指定 report type、rule IDs、alert IDs 或 reference run，也不能把局部机构证据自动提升为全局结论。

多机构分析仍遵循“全局发现 → 机构稳定性 → 条件式机构规则 → Global/Institution 监控与根因”的确定性顺序。机构列只参与切分、验证和监控，不作为建模特征；小样本和单标签机构被阻断或抑制。只有机构告警时不能称为全局规则失效，Local 规则也不能自动推广。

`privacy.expose_segment_values` 是启动时 `ProjectConfig` 的本地报告策略，不是 MCP 参数。受限本地报告可按配置展示已确认的机构 metadata，但 DecisionContext 始终保持聚合边界，并在返回前执行隐私检查。实体值、样本行、原始日志、真实文件路径、Parquet 明细、Shell、网络访问和自动策略上线始终禁止。
