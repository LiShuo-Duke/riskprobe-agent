# Xinyongka MCP Scorecard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 RiskProbe MCP 接受运行时精确特征列，并在启用时输出训练集拟合的 WOE 分箱、评分卡和聚合报告，同时保持两阶段 Host 决策门控。

**Architecture:** 运行时 `ProjectConfig` 以可选 `feature_columns` 覆盖 YAML 特征选择；精确列必须全部存在。服务在既有规则发现后执行一个可恢复的 scorecard 节点，写入受限聚合 `scorecard.json`，再由现有报告节点渲染该工件。MCP 仍只保留两工具，终端决策行为不变。

**Tech Stack:** Python 3.13、Pydantic、Polars、NumPy、scikit-learn、FastMCP、pytest。

## Global Constraints

- 保留现有两个 MCP 工具和 `inspect → diagnose → discover → recommend → review` 顺序。
- 不输出或持久化实体 ID、样本行、原始特征值或行级预测。
- 评分卡只以 Train 分区拟合；Test/Holdout 只能用冻结模型评分。
- 不覆盖现有未提交的运行时路径/角色列实现；在其上扩展。
- `scorecard.enabled` 默认 `false`，关闭时旧工件集合保持不变。
- 不提交 git；用户尚未要求提交。

---

### Task 1: 运行时精确特征列与严格数据契约

**Files:**
- Modify: `src/riskprobe/config.py:57-116, 208-225`
- Modify: `src/riskprobe/mcp_server.py:58-87`
- Modify: `tests/test_config.py:1-80`
- Modify: `tests/test_mcp_server.py:42-111`

**Interfaces:**
- Consumes: `riskprobe_get_decision_context(idempotency_key, parquet_path, column_roles)`。
- Produces: `riskprobe_get_decision_context(idempotency_key, parquet_path, column_roles, feature_columns: list[str] | None = None)`；`ProjectConfig.with_runtime_dataset(..., feature_columns: Sequence[str] | None = None)`。

- [ ] **Step 1: 写入失败测试：精确列必须完整存在且运行时输入覆盖模板。**

```python
def test_runtime_feature_columns_override_template_and_require_each_schema_column() -> None:
    config = _template_config().with_runtime_dataset(
        parquet_path=Path("/tmp/runtime.parquet"),
        column_roles=_roles(),
        feature_columns=["feature_b", "feature_a"],
    )
    assert config.features.exact_columns == ("feature_b", "feature_a")
    with pytest.raises(ValueError, match="missing required exact feature columns: feature_b"):
        config.features.select_columns(
            ["entity", "snapshot", "segment", "target", "feature_a"],
            ["entity", "snapshot", "segment", "target"],
        )
```

Add MCP-schema and client-call assertions for `feature_columns`.

- [ ] **Step 2: 运行目标测试并确认失败原因是尚未支持运行时列覆盖/缺列拒绝。**

Run: `pytest tests/test_config.py tests/test_mcp_server.py -q`

Expected: FAIL because `with_runtime_dataset()` has no `feature_columns` parameter, the MCP schema lacks it, and exact-column selection silently intersects schema.

- [ ] **Step 3: 实现最小输入覆盖与严格列验证。**

In `FeatureFamilyConfig.select_columns()`, preserve declared `exact_columns` order and, before returning, raise `ValueError("missing required exact feature columns: ...")` when any declared name is absent from `columns`. In `ProjectConfig.with_runtime_dataset()`, validate a non-string sequence of non-empty unique strings through `FeatureFamilyConfig` and update only `exact_columns`, retaining template `families` and catalog. In MCP, add optional `feature_columns: list[str] | None = None` and pass it only to the runtime config derivation.

- [ ] **Step 4: 运行目标测试确认通过。**

Run: `pytest tests/test_config.py tests/test_mcp_server.py -q`

Expected: PASS; tool schema includes `feature_columns`; all requested columns are retained in caller order; an omitted exact feature is rejected before a Host context is published.

### Task 2: 评分卡配置和受限聚合序列化

**Files:**
- Modify: `src/riskprobe/config.py:117-168`
- Modify: `src/riskprobe/service.py:80-99, 551-650`
- Modify: `tests/test_config.py`
- Modify: `tests/test_service.py:132-160`

**Interfaces:**
- Consumes: `ScorecardModel` from `riskprobe.rules.scorecard` and Train/Test/Holdout frames.
- Produces: `ScorecardConfig`; private service helper `_scorecard_payload(model, partitions, excluded_features, status, limitation) -> dict[str, Any]`.

- [ ] **Step 1: 写入失败测试：配置默认关闭，序列化只含聚合字段。**

```python
def test_scorecard_config_defaults_disabled() -> None:
    assert ProjectConfig.model_validate(_config_payload()).scorecard.enabled is False


def test_scorecard_payload_excludes_row_level_data(service: RiskProbeService) -> None:
    payload = service._scorecard_payload_for_test()
    assert payload["status"] == "fitted"
    assert "predictions" not in json.dumps(payload)
    assert payload["model"]["calibrated"] is False
    assert payload["model"]["binning_models"][0]["feature"] == "feature_a"
```

Use a deterministic small fixture with two numeric features and target classes 0/1. Assert bin edges, counts, WOE, IV, coefficients, rule IDs, class counts and partition aggregates are serializable primitives.

- [ ] **Step 2: 运行目标测试并确认失败原因是配置/序列化接口尚不存在。**

Run: `pytest tests/test_config.py tests/test_service.py -q`

Expected: FAIL because `ScorecardConfig` and the scorecard payload helper do not exist.

- [ ] **Step 3: 实现最小配置与聚合 payload。**

Add strict `ScorecardConfig` fields: `enabled: bool = False`, `max_bins: int = Field(10, ge=2, le=100)`, `min_bin_fraction: float = Field(0.05, gt=0, le=1)`, `smoothing: float = Field(0.5, gt=0, le=100)`, `monotonic: Literal["none", "increasing", "decreasing", "auto"] = "auto"`, `min_iv: float = Field(0.0, ge=0)`, `C: float = Field(1.0, gt=0)`, `max_iter: int = Field(1000, ge=1)`. Add it to `ProjectConfig`.

In `service.py`, serialize only the `ScorecardModel` public model fields and `WOEBinningModel` bin metadata. For each non-empty partition, compute frozen prediction summary (`row_count`, `mean_bad_probability`, `min_bad_probability`, `max_bad_probability`, `mean_risk_score`, risk-level counts); include no per-row data. Use a status object for `fitted` and `unavailable`, including an aggregate limitation string when fitting cannot proceed.

- [ ] **Step 4: 运行目标测试确认通过。**

Run: `pytest tests/test_config.py tests/test_service.py -q`

Expected: PASS with deterministic JSON-compatible payloads and no row-level information.

### Task 3: 服务评分卡节点、完整性工件与报告章节

**Files:**
- Modify: `src/riskprobe/service.py:80-99, 838-856, 1865-2318`
- Modify: `tests/test_service.py:132-160, 407-418, 1533-1559`

**Interfaces:**
- Consumes: Train/Test/Holdout partitions, selected `feature_names`, discovered `rules`, and `ProjectConfig.scorecard`.
- Produces: optional `scorecard.json`; scorecard checkpoint node; `_render_service_report(..., scorecard_payload: Mapping[str, Any] | None = None) -> str`.

- [ ] **Step 1: 写入失败服务测试：启用评分卡会写出完整性保护的工件和报告章节。**

```python
def test_enabled_scorecard_writes_aggregate_artifact_and_report(tmp_path, synthetic_config):
    config = synthetic_config.model_copy(
        update={"scorecard": ScorecardConfig(enabled=True)}
    )
    context = RiskProbeService(config=config, runs_dir=tmp_path / "runs").run()
    payload = json.loads((context.run_dir / "scorecard.json").read_text())
    assert payload["status"] in {"fitted", "unavailable"}
    assert "WOE Binning and Scorecard" in (context.run_dir / "risk_report.md").read_text()
    assert "scorecard.json" in json.loads((context.run_dir / "manifest.json").read_text())["artifacts"]
```

Add a successful fitted-fixture assertion that verifies bins are trained from Train only by comparing stored edges with `fit_scorecard(train, ...)`, not Test/Holdout. Add disabled-mode assertion retaining the six historical artifacts exactly.

- [ ] **Step 2: 运行目标测试并确认失败原因是工件、节点和报告章节尚不存在。**

Run: `pytest tests/test_service.py -q`

Expected: FAIL because `scorecard.json` is absent and report output contains no scorecard section.

- [ ] **Step 3: 接入可恢复 scorecard 节点。**

Add `scorecard.json` to `_ARTIFACT_NAMES` and `_ARTIFACT_SCHEMAS`, but construct node-specific expected artifacts conditionally only when `scorecard.enabled`. Insert `scorecard` between `discover` and `validate` in `_NODE_ORDER`. Its action must:

1. choose `feature_names` whose Train schema is numeric;
2. record all non-numeric selected names as exclusions;
3. call `fit_scorecard(train, feature_names=numeric_names, target_col=..., rules=rules, ...)` using only `ScorecardConfig` and existing imbalance config;
4. score frozen model on Train/Test/available Holdout and write canonical `scorecard.json`;
5. catch expected data/model eligibility errors and write `status="unavailable"` with an aggregate limitation, then continue to validation/report.

The restore path reads and type-checks the artifact. Pass the payload to report rendering. Add a deterministic Markdown section headed `## WOE Binning and Scorecard`, with status, limitation, selected/excluded features, bins, IV, coefficients, rules, partition summaries, and `calibrated: false` limitation. Keep existing report content untouched when disabled.

- [ ] **Step 4: 运行目标测试确认通过。**

Run: `pytest tests/test_service.py -q`

Expected: PASS; enabled scorecards are train-only and integrity-protected, unavailable models remain auditable, disabled runs remain byte-compatible.

### Task 4: Xinyongka 模板、Kiro MCP 默认配置与文档

**Files:**
- Create: `configs/xinyongka.example.yaml`
- Modify: `.kiro/settings/mcp.json:1-24`
- Modify: `README.md:192-244`
- Modify: `tests/test_mcp_server.py:42-136`

**Interfaces:**
- Consumes: Xinyongka roles and the exact 22 provided feature names.
- Produces: an MCP server started with `configs/xinyongka.example.yaml`, whose first tool accepts the supplied runtime request.

- [ ] **Step 1: 写入失败测试：MCP 客户端传精确列后仍完成固定两阶段。**

```python
assert set(schemas["riskprobe_get_decision_context"]["properties"]) == {
    "idempotency_key", "parquet_path", "column_roles", "feature_columns"
}
# use the synthetic fixture's names to keep the test self-contained
pending = await client.call_tool(
    "riskprobe_get_decision_context",
    {**request, "feature_columns": ["feature_a", "feature_b"]},
)
assert pending.is_error is False
```

Assert resulting run config selects only these features and terminal tool sequence remains unchanged.

- [ ] **Step 2: 运行 MCP 测试并确认失败原因是 schema/调用或模板尚未完成。**

Run: `pytest tests/test_mcp_server.py -q`

Expected: FAIL until the previous interface is fully wired and test artifact expectations account for enabled/disabled scorecard mode.

- [ ] **Step 3: 写入最小用户可用配置和文档。**

Create `configs/xinyongka.example.yaml` with a placeholder `dataset.path`, `ID/date/SEX/default payment next month` role template, all 22 exact feature names, WOE discovery enabled, scorecard enabled, and explicit target/snapshot/validation/privacy/imbalance defaults copied from the existing synthetic template. Point workspace `.kiro/settings/mcp.json` at this template while retaining runtime `parquet_path`. Update README’s MCP call example so roles stay inside `column_roles` and the exact list goes to `feature_columns`; state that the Host needs to submit the returned evidence and policy-allowed actions in phase two.

- [ ] **Step 4: 运行 MCP 测试确认通过。**

Run: `pytest tests/test_mcp_server.py -q`

Expected: PASS; runtime exact features work without a third MCP tool and terminal review still enforces the fixed workflow.

### Task 5: 全量质量验证与真实调用前置检查

**Files:**
- Modify: files only when preceding tests expose defects.
- Test: `tests/test_config.py`, `tests/test_scorecard.py`, `tests/test_service.py`, `tests/test_mcp_server.py`.

**Interfaces:**
- Consumes: all completed changes.
- Produces: validated package, formatter/linter-clean source, and documented runtime invocation.

- [ ] **Step 1: 运行所有相关测试。**

Run: `pytest tests/test_config.py tests/test_scorecard.py tests/test_service.py tests/test_mcp_server.py -q`

Expected: PASS.

- [ ] **Step 2: 运行静态质量检查。**

Run: `ruff check src tests && ruff format --check src tests`

Expected: PASS without modifying files.

- [ ] **Step 3: 构建包并确认示例配置可解析。**

Run: `.venv/bin/python -m build`

Expected: PASS and creates only ignored build artifacts. Then run:

```bash
.venv/bin/python -c 'from pathlib import Path; from riskprobe.config import ProjectConfig; assert ProjectConfig.from_yaml(Path("configs/xinyongka.example.yaml")).scorecard.enabled'
```

Expected: exit 0.

- [ ] **Step 4: 执行真实数据 MCP 完整流程（仅在用户当前已启动配置指向新模板并确认数据前置条件后）。**

Use the first MCP tool once with a fresh public idempotency key, exact runtime path, four roles, and all 22 `feature_columns`. Select only policy-allowed action codes after receiving the bounded context; submit the exact returned context ID and full diagnosis evidence IDs once. Verify terminal sequence and report/scorecard aggregate status. If preconditions fail, report safe failure stage and do not bypass through CLI or filesystem.

## 2026-08-24 决策修订：固定启动档案

用户确认采用档案 A。第一 MCP 工具恢复为仅接收 `idempotency_key`；数据路径、四个角色列和精确特征列仅由启动 YAML 提供。所有计划中 `parquet_path`、`column_roles`、`feature_columns` 的 MCP 输入要求作废。保留两阶段协议，并在 Kiro Skill 中定义基于受限聚合上下文的中文完整报告格式；不增加外部 LLM API 调用或读取运行工件的工具。
