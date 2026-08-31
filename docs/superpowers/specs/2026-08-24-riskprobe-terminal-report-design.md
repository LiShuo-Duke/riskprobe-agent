# RiskProbe 完整终态报告设计

## 目的

在不改变两个公开 MCP 工具、请求/响应 schema、error code、Host 决策顺序和隐私门的前提下，让每次可验证运行自动生成正式中文报告：

- 分析阶段生成增强版 `risk_report.md`；
- 完整 MCP 终态生成内容一致的 `final_risk_report.md` 与 `final_risk_report.docx`；
- 覆盖全部 24 个阶段、TOP10 规则及关键指标、评分卡、机构稳定性、诊断、建议和 review；
- 同一幂等请求复用同一份不可变报告。

本设计是 `2026-08-24-riskprobe-full-analysis-output-design.md` 的报告交付扩展。原设计中“最终自然语言仅由 Kiro 生成”和“不新增依赖”两点，由本设计分别调整为“服务端生成确定性正式报告”和“允许精确固定 `python-docx` 版本”；其余安全聚合、DTO 和 MCP 约束保持不变。

## 不变约束

- MCP 仍只暴露 `riskprobe_get_decision_context` 与 `riskprobe_submit_decision_proposal`。
- 固定顺序仍为 `inspect → diagnose → discover → recommend → review`。
- 不返回或写入实体值、样本行、原始日志、真实机构名、真实输入路径、traceback 或 Parquet 明细。
- 报告只使用已验证的安全聚合 DTO、`EvidenceCard` 投影、diagnostics、recommendation 和 review 结果。
- 已 finalize 的 run 目录及其 manifest/hash 不得在 Host 终态后修改。
- 不自动执行规则或业务动作；报告中的建议仍需人工批准。

## 选定方案：两阶段统一报告模型

### 阶段一：分析报告

现有分析流水线继续在 finalize 前生成 run artifact `risk_report.md`。增强内容包括：

- 数据画像、质量、切分和时间验证摘要；
- 规则发现与筛选统计；
- 按确定性口径选出的 TOP10 规则及完整安全指标；
- 评分卡、机构和稳定性摘要；
- 当前已执行分析阶段的状态、原因和限制。

该文件继续使用现有 artifact 名称和公开 schema，不新增 run artifact，不改变 legacy 6/7 项组合。现有 `analysis_summary.json` 和 `evidence_cards.json` 是渲染输入；不得读取 `candidate_rules.parquet`。

### 阶段二：完整终态报告

AgentResult 通过 terminal validator 后，服务构建私有 `FinalReportModel`。模型合并：

- 已验证的 `AnalysisSummary`；
- 经过安全投影的 TOP10 `EvidenceCard`；
- diagnostics 与 evidence IDs；
- proposal、recommendations、review 和 terminal status；
- 固定 reason codes 与 allowlisted limitations。

模型先经过现有递归安全校验，再由两个 renderer 分别生成 Markdown 和 DOCX。两个 renderer 不互相解析，避免 Markdown 与 Word 内容漂移。

终态报告存入独立的 terminal report store，根目录位于既有 Host `state_dir` 下，而不是已发布 run 目录。目录使用稳定 report ID，不包含原始幂等键、数据集路径或实体值。每份报告包含：

- `final_risk_report.md`
- `final_risk_report.docx`
- `final_report_manifest.json`

manifest 绑定 report ID、context/session/run 身份、terminal result 摘要哈希，以及两个报告文件的 SHA-256 和大小。公开 MCP payload 不返回真实路径、文件 bytes、hash 或新增字段。

## 幂等与发布

- report ID 由既有协调器身份稳定派生；相同幂等请求和相同 proposal 必须命中同一 report ID。
- 写入采用临时文件、校验、原子发布；只把 manifest 完整且 hash/size 匹配的目录视为已发布。
- 幂等重放先验证现有报告；验证通过直接复用，不重新渲染。
- 报告缺失或中断时，只能从已验证的安全 terminal state 确定性重建，不重新执行 inspect、diagnose、discover、recommend 或 review。
- accepted、rejected、no-action 等可验证终态均生成报告。失败时若已有安全 `analysis_progress`/summary，则生成失败报告并标明失败阶段；在任何安全摘要产生前的启动级失败不伪造报告。

## 报告结构

### 1. 封面与执行结论

- 安全 dataset ID、Run ID、Session ID；
- pipeline status、decision status、Metadata Grade；
- 总体风险结论、关键限制和人工审批提示。

### 2. 管理摘要

- 行数、特征数、正样本率和安全时间范围；
- 候选/入选/Stable 规则数；
- 评分卡 Train/Test/Holdout 的 AUC、KS、Gini；
- 主要诊断、最终建议和 review 结论。

### 3. 全流程阶段摘要

按固定顺序展示全部 24 个阶段。每阶段包含 name、status、enabled、output_available、核心聚合结果、reason code 和 limitations。只有已有真实数据时才显示 duration；不得推算或填充虚假耗时。

### 4. 数据画像、质量和切分

展示样本与特征聚合、目标分布、Metadata Grade、issue codes，以及 Train/Test/Holdout 行数、比例、时间范围、请求/实际切分策略和 fallback 原因。

### 5. 规则发现与筛选

展示单变量/组合候选与入选数、特征跳过原因、Lift/Coverage/Precision 分布及 Stable/Local/Unstable/Suspicious 数量。

### 6. TOP10 规则

排序固定为：

1. Grade：Stable → Local → Unstable → Suspicious；
2. 同 Grade 按 Test Lift 降序；
3. 再按 Rule ID 升序，保证确定性。

主表与逐规则摘要至少包含：

- 排名、Rule ID、规则来源和安全可读条件；
- Grade、Support、Test Coverage、Hit Bad Rate；
- Test Lift、Holdout Lift、signed KS、Adjusted p-value；
- Lift CI、Segment Consistency、Time Decay；
- Train/Test/Holdout 对比、时间/分群稳定性及 limitations。

数值条件允许展示聚合阈值；分类条件只展示稳定 token，不展示真实类别值。Holdout 或时间验证未应用时显示 `N/A` 并附固定原因，不把缺失值解释为 0。

### 7. 评分卡

启用且可用时展示特征、WOE/IV、系数、类别平衡策略，以及 Train/Test/Holdout 的 AUC、KS、Gini。未启用、跳过或不可用时展示真实状态和固定原因。

### 8. 机构与稳定性

只展示脱敏机构 token、机构内 TOP 风险、跨机构一致性、时间衰减和限制。机构级发现不得表述为全局生产规则。

### 9. 诊断、建议与审核

展示各诊断子阶段状态、结论、evidence IDs，proposal actions，每条 recommendation 的受控 action、父 finding 和审批要求，以及 review 的 approved/reasons/retry count 和最终 decision status。

### 10. 限制与附录

展示数据、验证、诊断和决策限制，关键指标定义，以及全部入选规则的安全简表。详细聚合事实继续保留在 JSON artifacts，不把原始数据复制到报告。

## Word 渲染

- 新增精确版本固定的 `python-docx` 依赖，不使用系统 Pandoc 或网络服务。
- 使用 A4、统一中文字体回退、标题层级、页眉页脚、页码、重复表头和正式表格样式；宽 TOP10 表可使用横向 section。
- core properties 不写用户名、主机、输入路径或当前墙钟时间。
- DOCX 生成后按固定成员顺序、固定 ZIP timestamp、固定压缩参数重新打包，保证相同模型字节稳定。
- 报告时间只使用已持久化且与该运行绑定的时间，不在重放时取当前时间。

## 失败处理

- report model 安全校验失败时不得渲染或写盘。
- Markdown 或 DOCX 任一失败时不得发布 manifest，也不得留下被视为成功的半份报告。
- 不新增公开 error code。首次报告发布失败映射为既有安全编排失败；同 proposal 重试仅修复报告，不重复执行 Agent 工具或建议动作。
- 生成失败不得把异常文本、路径、traceback 或第三方库细节返回给 MCP 客户端。
- 已验证 terminal 决策事实保持可恢复，报告失败不能导致重复业务动作。

## 兼容性

- 公开 MCP schema、工具数、请求字段、响应字段及固定 error code 不变。
- run 内仍使用现有 artifact 集；旧 6/7 项及 scorecard 组合继续可读。
- `artifact_count` 从已验证 manifest 动态派生，清理当前硬编码 6/7 的不一致，不因新增 terminal sidecar 增加 run artifact count。
- terminal report store 是内部交付层，不纳入 run manifest，也不允许通过 MCP 参数选择路径。

## 最小改动边界

- 复用现有 `AnalysisSummary`、`DecisionSummary`、`EvidenceCard`、安全校验、排序和原子存储模式。
- 仅新增私有 report model、DOCX renderer 和 terminal report store；不引入通用模板系统、配置开关、HTML/PDF 或新 MCP 工具。
- Markdown 与 Word 始终同时生成，不增加格式组合矩阵。

## 验收标准

1. 新分析 run 的 `risk_report.md` 包含阶段摘要和恰好 `min(10, evidence_count)` 条 TOP 规则，且展示安全规则条件和约定指标。
2. accepted、rejected、no-action 及有安全摘要的失败终态均生成内容一致的 Markdown 与 DOCX。
3. 同一幂等 proposal 重放复用同一 report ID，报告 hash 不变，且不新增工具调用。
4. DOCX 可由 Word 正常打开，具有正式标题、表格和分页；Markdown 与 DOCX 的章节、Rule ID 和关键指标一致。
5. 报告及解包后的 DOCX XML 不含输入路径、真实机构/分类值、样本行、原始日志或 traceback。
6. run manifest/完整性和 legacy 读取不回退；公开 MCP schema/error code 保持不变。
7. xinyongka 新 run 完成全流程后，TOP10、评分卡指标、诊断、建议和 review 均出现在最终报告中；完全相同 proposal 重放结果等价。

## 验证方式

遵守仓库限制，不执行 Shell、Python、pytest、SQL、网络、SQLite，不读取原始日志或数据。实施完成后集中进行一次源码级契约核对和一次语义审查；随后由已加载 MCP 使用全新稳定幂等键完成一轮真实上下文、proposal、terminal 与幂等重放验证。