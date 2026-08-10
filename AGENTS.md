# RiskProbe Agent 工作规则

本仓库提供本地只读风险规则发现、验证和监控工具。所有客户端都必须遵守以下规则；核心工具通过 `riskprobe.mcp_server` 暴露。

## 工具调用顺序

1. 直接使用 Parquet 时，先调用 schema 预览，再向用户确认实体、时间（或无时间）、机构/分群、目标和精确特征列。
2. 用户确认前不得注册 Parquet，不得根据列名猜测角色或自动补充特征。
3. 注册后按顺序执行：`inspect_dataset` → `discover_rules` → `validate_rules` → `detect_anomalies` → `diagnose_anomaly` → `build_report`。
4. `discover_rules` 必须传 `objective="risk"`；不要传非空 `constraints`，当前接口不支持每次调用覆盖发现阈值。
5. 没有时间列时只做固定随机 Train/Test 验证，不得声称严格 OOT、时间外推或无时间穿越。
6. 发现失败最多按客户端策略重试一次；仍失败时报告限制并停止，不用猜测参数继续调用。

## 输出和隐私

- 默认在受限聚合字段中展示已确认分层列的真实机构名，同时保留 `institution_token`；显式配置 `privacy.expose_segment_values=false` 时只展示 token。
- 不请求、读取或输出实体值、样本行、原始日志、真实文件路径或 Parquet 明细。
- 不执行 Shell、网络访问、任意 Python/SQL，也不自动上线规则。
- 机构级规则只用于稳定性验证和人工复核，不自动升级为全局规则。
- B 级数据的表现窗口未知；`Stable` 不等于 OOT 稳定、生产就绪或自动上线。

## 客户端使用

不支持 Kiro 原生 Agent 文件的客户端，应读取本文并加载 `docs/agent-system-prompt.md`。所有客户端都必须通过 MCP allowlist 访问本地数据，并由用户配置自己的 `RISKPROBE_ALLOWED_DATA_ROOTS`。
