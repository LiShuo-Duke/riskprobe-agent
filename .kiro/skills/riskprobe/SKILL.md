---
name: riskprobe
description: Privacy-safe local risk-rule and anomaly analysis workflow.
---

1. If the user provides a local dataset config path, call `register_local_dataset(dataset_id, config_path)` first. If the user provides a local Parquet path, first call `inspect_local_parquet_schema(parquet_path)` to read only column names and dtypes. Present that schema metadata to the user, then ask for: entity identifier column, time column (or none), institution/segment column, and target column. Next call `preview_local_parquet_features(parquet_path, entity_column, target_column, segment_column, snapshot_column)` and present its candidate numeric feature columns plus any non-numeric columns excluded from modeling. Ask the user to confirm which candidate columns are modeling features. Do not call `register_local_parquet` until the user confirms the roles and exact feature list; never infer roles or automatically add features. Then call `register_local_parquet(dataset_id, parquet_path, entity_column, target_column, segment_column, snapshot_column, feature_columns)` with exactly the confirmed list. If MCP reports a missing field, incompatible type, role conflict, or duplicate feature, ask only about the corresponding field; execute the complete workflow only after validation succeeds. The MCP server may register only a read-only Parquet file under its configured `RISKPROBE_ALLOWED_DATA_ROOTS`; never request Shell, writes, arbitrary filesystem access, or a path outside the allowlist. When `snapshot_column` is null, use random validation only and do not claim OOT or time validation.
2. Start with `inspect_dataset`. If metadata grade is C or D, stop and state that rule conclusions are blocked.
3. For rules, call `discover_rules(dataset_id, objective)` before `validate_rules`; use the exact supported value `objective="risk"` and omit `constraints` entirely. Do not pass `constraints` to this tool unless it is omitted or empty; do not invent or forward `min_support`, `max_conditions`, or any other per-call constraint: the current MCP interface rejects non-empty `constraints`, and discovery thresholds come from the registered project configuration. If a user asks for a custom threshold, explain that it must be changed in the project configuration before registration/rerun rather than sent to this tool. After `discover_rules`, present the `discovery_report`: candidate counts, one-condition and two-condition counts, overall TOP5 and two-condition TOP5. Explain that discovery ranking uses Train Lift, then support. A successful `validate_rules` response returns a tokenized `reference_run_id` and a `validation_report`; present grade counts for Stable, Local, Unstable, and Suspicious, overall Test-Lift TOP5, two-condition TOP5, Stable TOP5, every rule's conditions, Train/Test metrics, Lift CI, adjusted p-value, segment consistency, time decay, grade, and safe reason codes. Pass the reference token to `detect_anomalies`. A single failed tool call may be retried once; then stop and report the limitation.
4. For monitoring, call `detect_anomalies` before `diagnose_anomaly`. Present `monitoring_report` with reference/current aggregate overview, all six alert-category counts, each alert's severity and numeric evidence; by default, confirmed institution names may be shown only in the authorized institution fields while raw scope values remain protected, and when `privacy.expose_segment_values` is `false`, use tokenized scope values and do not reveal institution names. Present `diagnosis_report` with the root-cause TOP3 for each alert. An empty alert result is valid; call `diagnose_anomaly` with no alert IDs and continue to `build_report`. Explain that `Stable` requires adjusted p-value <= alpha, Lift CI lower bound > 1, sufficient samples/slices, acceptable time decay, and no Local condition. When no snapshot exists, Stable does not mean OOT stability or production readiness.
5. Never request entity-level records, sample rows, raw logs, or external data. Do not additionally request, echo, or disclose real filesystem paths; if the user has already provided a path, pass it only as `parquet_path` to MCP's `register_local_parquet` and do not read or output the path.
6. Do not describe grade-B evidence as strict OOT validation, leakage-free, or production-ready. State that its performance window is unknown.
7. Reports must cite returned evidence and limitations. Use `build_report` only after the relevant inspection, rule, or monitoring workflow is complete.
8. Runtime registration is session-scoped and is lost after MCP reconnect; durable registration must be maintained by the user in the local registry outside Git.


多机构分析必须遵守以下报告顺序：

1. **全局规则发现报告**：展示 `discovery_report`，说明总体规则来自全机构合并 Train 数据，排名使用 Train Lift、support 和稳定 rule ID。
2. **机构稳定性验证报告**：展示 `validation_report` 中的总体 Test-Lift TOP5、二维 TOP5、Stable TOP5，以及 `institution_summary`。每条规则解释机构 token、Support、Coverage、Hit Bad Rate、Lift、方向和 CI 可用性。`Global Stable` 表示满足样本门槛的机构方向总体一致；`Local` 表示效果集中于少数机构；`Unstable` 表示时间衰减超过阈值；`Suspicious` 表示统计或样本证据不足。
3. **条件式机构内规则报告**：展示 `institution_rule_report` 中的 `Institution TOP5`。只有 Global `Local` 证据且机构 Train/Test 样本达到 `min_group_size`、标签至少两类时才会运行；小样本或单标签机构必须展示 blocked 原因，不得调用或暗示机构内规则发现成功。机构特异规则必须与 Global Rule 分开治理，不得自动上线或升级为全局规则。
4. **双层监控与根因报告**：展示 `monitoring_report`，区分 `Global Alert` 和 `Institution Alert`；随后展示 `diagnosis_report` 的 root-cause TOP3。机构告警不能直接表述为全局规则失效，必须结合影响机构和聚合数值解释。

没有机构差异或只有单机构时，明确说明未形成跨机构比较；没有触发 Local 时，明确说明未执行机构内规则发现。默认在受限机构字段中展示已确认的真实机构名，并保留稳定 token；显式配置 `privacy.expose_segment_values: false` 时，改为只使用稳定 token。不请求或输出实体、样本、路径和原始明细。无 snapshot 时，Stable、机构稳定性和局部规则均不等于严格 OOT、生产稳定或自动上线。


安全门控补充：Agent 不直接读取 Parquet，`fs_read` 必须保持 deny；所有 schema、角色预览和数据分析都必须通过本地 MCP 工具完成。直接注册 Parquet 时，`feature_columns` 必须是用户确认后的非空显式列表，且必须存在同一路径/同一角色组合的 preview 记录；缺少 preview 或省略特征列表必须停止并追问，不得回退到自动数值列选择。


真实分群名现在默认展示：默认 `privacy.expose_segment_values` 为 `true`，机构 token、告警 `scope_value` 和其他聚合标识仍保留以兼容既有消费者；本地 `risk_report.md` 及 Agent/MCP 的 `institution_summary`、`institution_results`、`institution_rule_report` 和机构告警可以在受限字段中增加真实 `institution_name`。如需收紧输出，用户可在本地配置中明确设置：

```yaml
privacy:
  expose_segment_values: false
```

该配置只控制受限的分群业务 metadata，不授权实体值、样本行、原始日志、真实路径、Parquet 读取、Shell、网络或自动上线。配置为 false 时，Agent 不得从本地报告、机构摘要或告警解释中复述真实分群名；两种模式都须保留 token 和聚合指标，并继续通过安全 payload 检查。