---
name: riskprobe
description: Use for Chinese credit-risk, rule-mining, binning, scorecard, or aggregate-report requests through the configured local RiskProbe MCP.
---

# RiskProbe Host 决策工作流

服务在启动 YAML 中固定只读数据档案、列角色和特征；Host 不传路径、角色或特征，不调用外部 LLM API。

1. 创建一个公开、稳定的 `idempotency_key`，调用且只调用一次 `riskprobe_get_decision_context(idempotency_key)`。服务执行 `inspect → diagnose → discover`。
2. 若首次结果是 `phase="terminal"` 且 `terminal_reason="no_actionable_diagnosis"`，该结果即为最终结果：不得调用 proposal 工具；报告完整 `inspect → diagnose → discover → recommend → review` 序列已完成、没有 diagnosis/action/proposal，且建议仍须人工审批、未自动修改或上线规则。
3. 若首次结果是 `awaiting_proposal`，只读取返回的 `context`，按返回顺序覆盖每项 `findings` 和 evidence ID；不得读取 runs 工件、路径、实体、样本、原始日志或行级预测，也不得使用 Shell、文件、网络或额外 MCP 工具。
4. 从 `policy.allowed_action_codes` 选择 `policy.min_action_count` 至 `policy.max_action_count` 个唯一 action code。B 级仅限 `policy.grade_b_allowed_action_codes`，报告必须说明只用于分析，不等于严格 OOT 或生产就绪。
5. 用同一个 `idempotency_key` 调用一次 `riskprobe_submit_decision_proposal`，传入原样 `context_id`、完整且未修改的 `diagnosis_evidence_ids` 和选定 `action_codes`。不得伪造、遗漏或替换证据。
6. 正常流程只接受 `terminal` 结果。上下文、证据、allowlist、有效期或幂等校验失败即停止，不能通过 CLI、文件系统或其他工具绕过。

## Terminal 后的中文 LLM 总结

仅在 terminal 后基于返回的 `analysis_summary` 与 `decision_summary` 生成中文用户可读完整总结；结构化字段是事实来源，不得推断缺失数据或补写未返回指标。

按以下顺序覆盖可用内容：

1. 执行完整性：24 个 stage 的用户可读完整摘要、未运行/不可用/失败原因、完整工具序列与最终 decision status。
2. 输入与 profile：只读、数据等级、样本/特征聚合、目标含义、性能窗口限制、问题码。
3. partition：请求与实际时间验证、策略、各分区聚合行数及回退原因。
4. discovery/WOE：候选与入选规则计数、公开 rule ID、指标分布、WOE 的特征/IV/分箱数量/单调性/缺失箱摘要；逐条覆盖已返回的 TOP10 安全规则与指标，包括 Rank、Rule ID、安全条件、Origin、Grade、Support、Test/Holdout Lift、Coverage、Hit Bad Rate、signed KS、Adjusted p-value、Lift CI、Segment Consistency、Time Decay；安全条件只使用已返回的安全投影，不还原原始条件、分箱边界或小样本统计。
5. scorecard：启用状态、模型类型、校准、参数、真实不平衡策略、所有入模/排除特征及原因、全部系数和截距、各分区的 AUC、KS、Gini、概率与风险等级聚合；指标不可用时报告返回的固定原因码。
6. validation 与 diagnostics：证据数、等级和限制统计、各项诊断类型/严重度/代码计数及时间稳定性限制。
7. recommend/review：受控 action、recommendation evidence、review 结果和原因码、retry、人工审批要求。

成功 terminal 后可声明本地逻辑文件名 `final_risk_report.md` 和 `final_risk_report.docx`，但不得输出 `state_dir` 或任何真实路径。若报告 gate 返回 `agent_orchestration_failed`，只报告该固定错误码；允许使用完全相同的 `idempotency_key`、`context_id`、`diagnosis_evidence_ids` 和 `action_codes` 原样安全重试，不重选 action，不从未返回字段推断。

禁止输出或推断实体值、样本行、真实分群值、路径、日志、traceback、原始条件、分箱边界或行级预测。B 级数据必须说明仅用于分析，不等于严格 OOT、生产就绪或自动上线。结尾固定说明：建议仍须人工审批，未自动修改或上线规则。
