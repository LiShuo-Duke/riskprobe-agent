# Xinyongka MCP 完整分析设计

## 目标

让 Host 在一次 `riskprobe_get_decision_context` 请求中，使用用户指定的本地 Parquet、四个角色列及精确特征列，完成既有的 `inspect → diagnose → discover` 分析，并持久化规则、WOE 分箱、训练集拟合评分卡和完整聚合报告；随后仍由 Host 按固定门控完成 `recommend → review`。

该设计不修改源 Parquet、不输出实体值或样本行、不自动部署规则或评分卡。

## 输入契约

MCP 仍只暴露两个工具。第一工具改为：

```python
riskprobe_get_decision_context(
    idempotency_key: str,
    parquet_path: str,
    column_roles: dict[str, str],
    feature_columns: list[str] | None = None,
) -> HostDecisionContext
```

`column_roles` 只接受 `entity`、`snapshot`、`segment`、`target`。`feature_columns` 未提供时，保留启动 YAML 的特征选择策略；提供时，运行时配置以该顺序覆盖为精确特征列。

用户的 Xinyongka 调用应传入：

```json
{
  "parquet_path": "/absolute/private/xinyongka.parquet",
  "column_roles": {
    "entity": "ID",
    "snapshot": "date",
    "segment": "SEX",
    "target": "default payment next month"
  },
  "feature_columns": [
    "LIMIT_BAL", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2",
    "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2",
    "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1",
    "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
  ]
}
```

Kiro 的自然语言层负责把“实体列”等标签映射到该工具契约；MCP 服务不解析自然语言。

## 运行时配置和数据契约

`ProjectConfig.with_runtime_dataset()` 接受可选 `feature_columns`，并创建仅本次请求有效的 `FeatureFamilyConfig(exact_columns=...)`。传入的列名必须是非空、唯一字符串。

`FeatureFamilyConfig.select_columns()` 在 `exact_columns` 模式下必须验证全部声明列存在于 Parquet schema，任一缺失即抛出确定性、安全的配置错误。它不再静默舍弃缺失精确特征。角色列仍在 profile 阶段验证。

对 Xinyongka，`default payment next month` 必须无空值、且只含 0/1；`date` 必须能解析且存在至少三个有效时间点，才能获得 Train/Test/Holdout 时间切分。若前置条件不满足，第一工具返回阶段化但不泄漏记录或文件路径的错误；不得把可预检的输入契约错误折叠成 `host decision is unavailable`。

## 评分卡与分箱阶段

新增 `ScorecardConfig`，由 YAML 控制，默认关闭以保持既有运行结果和工件集合兼容：

```yaml
scorecard:
  enabled: true
  max_bins: 10
  min_bin_fraction: 0.05
  smoothing: 0.5
  monotonic: auto
  min_iv: 0.0
  C: 1.0
  max_iter: 1000
```

启用时，`RiskProbeService.run()` 在规则发现之后、报告之前：

1. 从已选精确特征中保留训练集实际为数值类型的列；其余列不参与 WOE/评分卡，并被记录为排除项。
2. 以 Train 分区和既有 `fit_scorecard()` 拟合 WOE 分箱、IV、逻辑回归系数及可选规则命中项；不在 Test/Holdout 重新拟合。
3. 对 Train、Test 及可用 Holdout 仅使用冻结模型评分，写入聚合风险概率/分数统计和可用的二分类指标。
4. 无法拟合时，不使规则分析失败；输出受控的评分卡状态、原因和空模型，不伪称评分卡已生成。

`ScorecardConfig.enabled=false` 时不执行该阶段，也不新增评分卡工件。

## 工件和报告

启用评分卡且成功或受控降级时，运行目录新增 `scorecard.json`。内容仅包括：版本、状态、训练行数/类别计数、已入模和排除特征、每个 WOE bin 的边界/计数/坏账率/WOE/IV、缺失箱统计、规则 ID、系数、截距、未校准标志，以及分区级别聚合指标。不得写入每行预测、实体 ID 或原始特征值。

`manifest.json` 和运行时工件 schema 将把 `scorecard.json` 纳入完整性校验。`risk_report.md` 增加“WOE 分箱与评分卡”章节，内容来自该 JSON：模型状态、训练边界、IV、系数、规则贡献、分区指标、排除项和限制（尤其是 `calibrated=false` 与 B 级数据只能分析）。原有候选规则和证据卡保持不变。

MCP 的 `HostDecisionContext` 和 terminal outcome 不携带完整报告正文，仍保持受限聚合和 Host 复核契约。第一阶段写完不可变完整报告；终端输出仅陈述报告已生成、评分卡状态、固定序列和人工复核结果。

## 错误处理

- 工具签名/Pydantic 参数错误：由 MCP schema 直接拒绝。
- 绝对路径、四角色、精确特征列、目标/时间前置条件错误：在进入 `HostDecisionCoordinator` 前以稳定的输入/分析错误返回。
- 数据处理或工件失败：coordinator 保持拒绝决策，不允许伪造 context、证据或 proposal；对外仅返回安全的阶段状态，详细异常仅保留在本地受控状态。
- 任何失败均不启动第二阶段，不通过 CLI 或额外 MCP 工具绕过。

## Xinyongka 模板与 Kiro 配置

新增 `configs/xinyongka.example.yaml`，使用示例占位 Parquet 路径、以上 22 个 `features.exact_columns`、`woe_binning_enabled: true` 与 `scorecard.enabled: true`。用户复制为未提交的私有 YAML，并将 `.kiro/settings/mcp.json` 的 `--config` 改为私有文件路径。公开仓库不写入 `/absolute/private/xinyongka.parquet`。

## 测试

1. 配置单元测试：运行时精确特征覆盖、非空/唯一校验、缺列失败。
2. MCP 集成测试：工具 schema 含可选 `feature_columns`；22 列透传；错误发生在 Host session 前；两阶段序列仍不变。
3. 服务集成测试：评分卡启用时产生带完整性记录的 `scorecard.json`，报告包含评分卡章节，分箱只由训练集拟合；关闭时保持旧工件集合。
4. 回归测试：现有 synthetic MCP 和无评分卡运行保持通过。

## 不在范围内

- 不新增第三个“读取报告”MCP 工具。
- 不输出行级评分、预测或原始数据。
- 不把 `SEX` 等编码变量自动转换为类别 WOE；当前评分卡仅训练数值特征。
- 不自动部署模型、规则或改变人工 review 机制。
