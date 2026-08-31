# RiskProbe 全阶段安全分析输出设计

## 目标

扩展现有两阶段 stdio MCP，使 Host 在提交 action proposal 前获得完整、安全、结构化的分析结果，并在 terminal 后获得推荐与 review 摘要。RiskProbe 保持本地确定性，不调用外部 LLM；Kiro 仅基于通过隐私门的 terminal 数据生成最终自然语言总结。

## 不变约束

- MCP 仍只暴露 `riskprobe_get_decision_context` 和 `riskprobe_submit_decision_proposal`。
- 固定执行顺序仍为 `inspect → diagnose → discover → recommend → review`。
- Host 必须原样提交 context、完整诊断证据及 allowlist 内 actions。
- 不返回实体值、样本行、真实分群名、逐行概率/分数、路径、原始日志、traceback、可反推小样本或无限制大型 payload。
- 不新增外部模型、网络调用或依赖。

## 架构

新增独立的安全 DTO 层，生成 `AnalysisSummary` 和 `DecisionSummary`：

1. artifact pipeline 在本地可信进程内从内存对象生成 `analysis_summary.json`，不透传现有 artifact 原文。
2. summary 在写入和 MCP 返回前均通过 `assert_safe_payload`。
3. `DecisionContext` 携带 immutable artifact summary，并补充 diagnostics 与 agent 阶段状态。
4. terminal 携带同一 analysis summary 的身份绑定及 `DecisionSummary`，涵盖 proposal、recommend、review 和最终状态。
5. Kiro skill/agent 指令要求 terminal 后输出 LLM 总结；MCP 服务本身不生成自然语言。

`analysis_summary.json` 不包含 finalize 的最终事实。构造 DecisionContext 时，在验证 manifest 后将 finalize/integrity 状态投影为 succeeded。这样避免 artifact 自引用，同时保留不可变分析事实。

## 统一阶段状态

所有阶段使用：

- `not_run`
- `succeeded`
- `skipped`
- `unavailable`
- `failed`
- `reused`

`StageSummary` 包含 `name`、`status`、`enabled`、`reason_code`、`output_available`、可选 `duration_ms` 和 allowlisted `limitations`。失败只返回固定错误码。空规则、单类标签、无 holdout、scorecard 无可用特征必须区分 unavailable/skipped，而不能伪装成失败。

阶段列表固定为：config、snapshot、profile、partition、discovery、woe、scorecard、validation、institution_analysis、report、finalize、inspect、diagnose_quality、diagnose_feature_drift、diagnose_population_shift、diagnose_target_shift、diagnose_segment_risk、diagnose_time_stability、diagnose_rule_evidence、discover_restore、decision_context、recommend、review、terminal。

## AnalysisSummary

### Input/Profile

返回 dataset ID、read-only、行数、总/数值/入选特征数、正样本率、segment 数与安全规模区间、metadata grade、issue codes、安全日期范围、目标窗口是否已知。不得返回 dataset path、实体列、配置原文或 segment label。

### Partition

返回 requested/applied time validation、strict/auto/disabled、实际 split strategy、train/test/holdout 行数、排除空 snapshot 数和固定 fallback reason code。不得返回索引或分区行。

### Discovery

返回配置参数、抽样状态与规模、输入/有效/跳过特征和 reason counts、候选/选中/单规则/组合规则数、全部安全 rule IDs、指标分布及有界 top rules。完整 conditions、全部阈值和大型 rule metrics 不返回。

若现有 `discover_rules()` 不保留所需计数，服务改用同一算法入口 `discover_with_metrics()`，确保规则输出不变，同时持久化安全聚合指标。

### Scorecard

返回 enabled、status、model type、calibrated、所有配置参数、类别平衡策略、class counts/rates、全部 input/included/excluded feature refs、排除原因、全部 rule-hit terms、intercept 和全部模型系数。feature ref 若不能通过隐私门则使用稳定公开 token，不删除模型项。

每个普通特征返回 IV、bin count、monotonic direction、missing-bin presence；不返回全部 edges、逐 bin 小组计数或 bad rate。

每个 train/test/holdout split 返回 sample count、positive rate、probability min/mean/max、mean score、risk-level counts，并新增：

- ROC AUC
- KS
- Gini = 2 × AUC - 1

单类或空 split 返回 `unavailable` 和固定 reason code。KS 使用预测概率排序后的正负类累计分布最大绝对差。不得返回逐行 prediction。

所有模型项必须完整返回；schema 根据已有 discovery/scorecard 配置上限设置硬性最大数量。超过安全上限时 scorecard summary 标记 `unavailable/output_limit_exceeded`，不得静默截断模型项。

### Validation/Institution Analysis

返回 evidence 数、grade counts、holdout 状态、limitation code counts、lift/CI/adjusted p-value/segment consistency/time decay 分布，以及有界 top evidence。segment 只允许显式 token；统一应用 min-group suppression。机构分析仅返回计数、等级和 limitations，不返回原始 label。

### Artifacts/Report

返回 logical artifact 名称、schema version、产物数量、published/reused 和 integrity_verified。不得返回本地路径、原始 markdown、文件 hash/size 或数据/config fingerprint。

### Diagnostics

保留现有完整安全 findings，并增加各子诊断的 enabled/status/reason、按 kind/severity/code 的 counts。明确区分 run 的 `time_validation_applied` 与 diagnose 的 `diagnostic_time_enabled`。

## DecisionSummary

terminal 返回：context ID、完整 selected actions、action-to-finding bindings、recommendation count/status、human approval、Grade-B analysis-only、review approved/reasons/no-action/retry count、tool sequence、evidence completeness、decision/final status。

recommendation 只返回 action code、parent finding IDs、审批要求及限制，不返回任意自然语言或内部审计 payload。

## 失败与重放

`HostDecisionFailure` 增加安全 `analysis_progress`：已成功/复用/跳过/失败/未运行阶段及失败码。Service 将当前安全 summary 附着到 host-safe exception；Coordinator 只持久化已通过 DTO 和隐私校验的数据。

幂等重放必须返回字节等价的 analysis/decision summary。旧持久化会话允许读取为 legacy，但新运行必须生成完整 summary。schema 版本升级，工具名称保持不变。

## Kiro LLM 总结

更新 RiskProbe Kiro skill/agent 指令：只在收到 terminal 后，基于完整结构化结果总结执行完整性、数据质量、切分、规则发现、评分卡参数/入模特征/效果、验证、诊断、recommend/review、限制与人工复核事项。必须同时保留关键指标，不能用自然语言替代结构化结果，也不能推断未返回的数据。

## 测试与验收

- DTO 严格校验、数量上限、隐私门和序列化稳定性。
- artifact summary 覆盖 scorecard fitted/unavailable/skipped、partition fallback、empty rules、holdout unavailable。
- AUC/KS/Gini 使用固定小数组验证，空/单类返回固定 unavailable。
- DecisionContext、failure progress、no-action、proposal、terminal 和幂等重放测试。
- 旧 session legacy 读取测试。
- MCP 端到端验证所有阶段均有状态、模型字段完整、proposal 被接受、terminal review approved。
- Kiro 输出最终 LLM 总结且不包含禁止内容。

## 成功标准

对启用 scorecard 的配置执行一次新运行后，Host 在 awaiting_proposal 阶段能看到 artifact pipeline 和 diagnostics 的完整安全摘要；terminal 能看到 recommendations/review/final 状态；全部阶段都有显式状态；评分卡包含全部入模模型项与 AUC/KS/Gini；Kiro 最后生成基于 terminal 的中文总结；现有隐私、幂等、证据和 action 门控不回退。
