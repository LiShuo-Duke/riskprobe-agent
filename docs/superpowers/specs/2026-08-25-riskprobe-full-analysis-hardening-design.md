# RiskProbe 全分析输出与 Host 状态安全加固设计

## 目标

在不增加 MCP 工具、不绕过 Host 门控、不暴露实体/行级/路径/日志/traceback 的前提下，让 RiskProbe 的 24-stage 分析、评分卡摘要、DecisionContext、recommend/review/terminal 和幂等恢复一次跑通。

## 范围与不变约束

- 部署假设为单个 RiskProbe MCP 进程；共享 state-dir 的多进程并发不作为支持场景，启动时单实例 fail-fast。
- MCP 仍只暴露 `riskprobe_get_decision_context` 与 `riskprobe_submit_decision_proposal`。
- 24 个 stage 必须各出现一次；状态按真实执行结果投影。
- 仅完整、可验证 terminal journal 允许恢复；非 terminal prefix、证据不完整、hash/provider/context 不一致继续 fail-closed。
- 不删除状态、不切换 state-dir、不用新幂等键掩盖错误、不自动生成或修改 action proposal。
- 不新增依赖，不提交 Git。

## 设计

### 1. 分析摘要、评分卡与公开隐私

- 所有公开 JSON/文本输出使用统一强隐私门；合法的聚合 DTO 使用显式白名单，不放宽行数组检测。
- 机构/分群值始终公开为稳定 token；配置不能让公开 manifest、report、metadata 或 summary 返回真实值。
- 评分卡成功契约强制校验 input/included/excluded 集合关系、特征 summary 与 terms 对齐、train/test/holdout 分区、class counts、risk counts 与 sample count 一致。
- scorecard unavailable 只返回固定 reason code，不返回底层异常文本。
- summary 使用安全 dataset ID；完成发布的 summary 标记 finalize succeeded、published 和 integrity verified。legacy manifest 未声明 summary 时 inspect 返回 `None`；声明后缺失或损坏则 fail-closed。
- artifact_count 从已验证 manifest 派生；新 manifest 带固定 schema version，旧 manifest 仍按既有 legacy 集兼容。

### 2. Agent terminal 恢复

- 在同一 `AgentResultStore` 单飞锁内读取当前 session/result 状态，移除锁外存在性快照，避免并发误判。
- `virgin` 状态正常执行；已有 cached result 且 journal/evidence 完整时严格重放；result 缺失但 terminal journal 完整时确定性重建并原子发布；非 terminal prefix 或任何不一致返回固定安全错误。
- 恢复不调用工具、provider 或 fallback；重建结果必须经过现有 terminal validator。
- orphan 临时文件不作为结果，允许在安全重建时被新原子发布覆盖。

### 3. Host 状态与终态投影

- 单进程内的持久化 awaiting session 在 Reconnect 后重新启动一次 runner，并复用原 context、proposal、evidence 和 action；不得重新决策。
- current terminal 要求 context、proposal/result evidence 完整且可 replay；读取或 replay 失败立即 fail-closed，不静默生成成功终态。
- accepted、rejected、no-action 都携带 bounded `analysis_summary` 与 `decision_summary`；legacy payload 仅由专用兼容解码器接收缺字段。
- 无 context 的 rejected AgentResult 映射为编排失败；只有成功但缺 context 的 actionable result 才映射为 state incomplete。
- 24-stage terminal 投影依据实际 journal 和 decision/recommend/review 节点，不依据计划序列猜测。
- Host session 写入前检查大小；超限保留旧文件并返回结构化安全失败。

## 数据流

1. `get_decision_context` 启动本地 immutable pipeline 和 agent graph。
2. pipeline 生成受限 `analysis_summary`，agent 按 inspect → diagnose → discover 执行并发布完整 context。
3. Host 基于完整 findings 选择 policy 允许的 action，并用原 context/evidence 提交 proposal。
4. agent 只执行受控 recommend，完成 deterministic review，写入 terminal journal 和 bounded summaries。
5. `submit_decision_proposal` 只返回 terminal 或固定 failure；相同 proposal 重放等价。
6. Kiro 仅基于 `analysis_summary` 与 `decision_summary` 生成中文总结。

## 错误处理

- 对外只返回 allowlisted phase/error code；不返回异常字符串、路径、日志或 traceback。
- 终态恢复失败保持 fail-closed；不得通过重复 Reconnect、删除 sidecar、改 state-dir 或绕过 Host 工具恢复。
- 旧数据仅在字段、hash、manifest、evidence 和 provider 绑定均可验证时兼容；新字段缺失只允许走明确 legacy decoder。

## 验证

- 静态核对所有修改链路及现有相关测试断言；按 workspace 规则不执行 Shell、任意 Python/SQL、网络或 pytest。
- 集中修改完成后只通知一次 Reconnect。
- 使用一个新的稳定幂等键调用 get；核验 awaiting_proposal、24 stages、评分卡全聚合项、隐私和完整 evidence。
- 原样提交允许 proposal；核验 accepted terminal、review.approved、两个 summaries、24 stages、工具顺序和 evidence。
- 完全相同 proposal 再提交一次，核验幂等结果等价。
