# Host 第一阶段安全失败诊断设计

## 目标

固定启动档案的第一阶段失败时，返回稳定阶段错误码；不暴露路径、列名、数据值、行数、异常类型、异常消息、日志或堆栈。

## MCP 契约

`riskprobe_get_decision_context(idempotency_key)` 成功时返回既有 `HostDecisionContext`；失败时返回：

```json
{"phase":"context","error_code":"profile_contract_failed"}
```

仅允许：`profile_contract_failed`、`partition_failed`、`discover_failed`、`scorecard_failed`、`artifact_runtime_failed`、`agent_session_failed`、`context_timeout`、`session_state_unavailable`。失败响应没有 context、证据或 policy，Host 必须停止且不能提交 proposal。

## 内部边界

服务在固定节点边界将任意底层异常包装为仅含白名单错误码的内部异常：profile、partition、discover、scorecard、artifact/runtime，以及 agent session。Coordinator 持久化并重放该代码，但不保留或返回原始异常。MCP 显式返回结构化失败投影，成功与既有 proposal/terminal 契约不变。

## 验证

测试覆盖每个主要阶段的错误投影、敏感异常文本不泄露、失败 session 重启重放、失败后 proposal 不可继续，以及既有完整两阶段成功路径。