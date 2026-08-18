# RiskProbe Monitoring and Kiro Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在核心规则引擎之上增加可量化的行为漂移注入、异常检测、根因排序、安全数据集注册表、本地 MCP 白名单工具和 Kiro Custom Agent。

**Architecture:** 监控层将参考数据固化为不含用户明细的 `ReferenceSnapshot`，用同一批固定算法比较当前数据和历史规则。MCP 层只接受 `dataset_id`/`run_id`，通过注册表解析本地配置；Kiro Agent 仅可见 `@riskprobe` 工具，显式拒绝 Shell、文件和网络能力。

**Tech Stack:** Plan 1 全部依赖，新增 MCP Python SDK 1.13.0；Kiro workspace custom agent、Agent Skill 和 stdio MCP。

## Global Constraints

- 必须先完成 `2026-08-05-riskprobe-core-engine.md`。
- MCP 不接受任意文件路径、SQL、Python 代码或用户级筛选条件。
- 所有输入在进入 RiskProbe 前已完成脱敏；MCP 输出只能来自固定 Pydantic 模型，可返回稳定的脱敏 dataset/segment/rule 编码用于聚合比较，但不返回 entity ID、样本行、未脱敏字段、原始日志、文件路径或低于最小样本量的分组。
- Kiro Agent 只暴露 `@riskprobe`，并 deny `shell`、`fs_read`、`fs_write`、`web_fetch`、`web_search`。
- 异常检测算法与根因贡献由本地 Python 计算；模型只能总结计算结果。
- 自动重试最多一次；C/D 元数据等级必须阻断规则结论。
- 每个任务包含提交命令，但只有用户明确授权提交后才能执行。

---

## File Map

```text
pyproject.toml                          # 增加 mcp==1.13.0
src/riskprobe/monitoring/models.py      # ReferenceSnapshot、Alert、Diagnosis、评估指标
src/riskprobe/monitoring/reference.py   # 无明细参考快照
src/riskprobe/monitoring/detection.py   # Schema/缺失/PSI/标签/规则衰减检测
src/riskprobe/monitoring/diagnosis.py   # 机构、时间、特征族根因贡献排序
src/riskprobe/monitoring/injection.py   # 六类有真值漂移注入
src/riskprobe/privacy.py                # 最小分组和 MCP 输出安全检查
src/riskprobe/registry.py               # dataset_id 到本地配置的白名单映射
src/riskprobe/mcp_server.py             # 六个 FastMCP 工具
src/riskprobe/cli.py                    # 新增 snapshot、monitor、evaluate-drift
configs/datasets.example.yaml           # 公开注册表示例
.kiro/settings/mcp.json                 # workspace stdio MCP
.kiro/agents/riskprobe.json             # 最小权限 custom agent
.kiro/skills/riskprobe/SKILL.md          # 风控分析 SOP
```

## Task 1: 监控模型与参考快照

**Files:**
- Modify: `pyproject.toml`
- Create: `src/riskprobe/monitoring/__init__.py`
- Create: `src/riskprobe/monitoring/models.py`
- Create: `src/riskprobe/monitoring/reference.py`
- Create: `tests/monitoring/test_reference.py`
- Create: `tests/monitoring/conftest.py`

**Interfaces:**
- Produces: `FeatureReference`, `RuleReference`, `ReferenceSnapshot`, `Alert`, `RootCause`, `Diagnosis`
- Produces: `build_reference_snapshot(frame, profile, evidence_cards, catalog, config) -> ReferenceSnapshot`
- Consumes: Plan 1 `DatasetProfile`, `EvidenceCard`, `FeatureCatalog`, `ValidationConfig`

- [ ] **Step 1: 增加 MCP 精确依赖**

在 `pyproject.toml` 核心依赖加入：

```toml
"mcp==1.13.0",
```

Run: `.venv/bin/python -m pip install -e ".[dev]"`

Expected: 安装成功，`.venv/bin/python -c 'import mcp; print("mcp-ok")'` 输出 `mcp-ok`。

- [ ] **Step 2: 编写参考快照不含明细测试**

```python
# tests/monitoring/test_reference.py
from riskprobe.monitoring.reference import build_reference_snapshot


def test_reference_snapshot_contains_aggregates_not_entities(reference_fixture) -> None:
    snapshot = build_reference_snapshot(**reference_fixture)
    payload = snapshot.model_dump_json()

    assert "entity_id" not in payload
    assert "user_0001" not in payload
    assert snapshot.row_count > 0
    assert snapshot.features[0].histogram_counts
```

`tests/monitoring/conftest.py` 提供后续监控测试共用的聚合 fixture：

```python
import pytest

from riskprobe.features.catalog import FeatureCatalog
from riskprobe.io.parquet import ParquetDataset
from riskprobe.profiling import profile_dataset
from riskprobe.synthetic import generate_behavior_dataset


@pytest.fixture
def reference_fixture(tmp_path, synthetic_config):
    frame, _ = generate_behavior_dataset(5_000, seed=42)
    frame.write_parquet(synthetic_config.dataset.path)
    catalog = FeatureCatalog.from_columns(
        frame.columns,
        synthetic_config.features.families,
    )
    profile = profile_dataset(ParquetDataset(synthetic_config.dataset.path), synthetic_config)
    return {
        "frame": frame,
        "profile": profile,
        "evidence_cards": (),
        "catalog": catalog,
        "config": synthetic_config,
    }


@pytest.fixture
def catalog(reference_fixture):
    return reference_fixture["catalog"]
```

- [ ] **Step 3: 运行测试并确认失败**

Run: `.venv/bin/python -m pytest tests/monitoring/test_reference.py -v`

Expected: FAIL，监控模块不存在。

- [ ] **Step 4: 实现固定监控模型**

核心字段必须为：

```python
class FeatureReference(FrozenModel):
    feature: str
    family: str
    dtype: str
    missing_rate: float
    zero_rate: float
    quantile_edges: tuple[float, ...]
    histogram_counts: tuple[int, ...]

class RuleReference(FrozenModel):
    rule_id: str
    coverage: float
    bad_rate: float
    lift: float

class ReferenceSnapshot(FrozenModel):
    snapshot_id: str
    dataset_id: str
    row_count: int
    positive_rate: float
    segment_counts: dict[str, int]
    features: tuple[FeatureReference, ...]
    rules: tuple[RuleReference, ...]
    created_at: str

class Alert(FrozenModel):
    alert_id: str
    alert_type: Literal["schema", "missingness", "distribution", "population", "label", "rule_decay"]
    severity: Literal["warning", "critical"]
    scope: Literal["dataset", "institution", "feature", "family", "rule"]
    scope_value: str
    metric: str
    reference_value: float | str | None
    current_value: float | str | None
    delta: float | None
    evidence: dict[str, float | int | str]
```

`build_reference_snapshot` 只保存直方图区间和计数、比例、segment 计数和规则聚合指标；禁止接收或保存 entity 列值。

- [ ] **Step 5: 运行测试**

Run: `.venv/bin/python -m pytest tests/monitoring/test_reference.py -v`

Expected: PASS。

- [ ] **Step 6: 用户授权后提交检查点**

```bash
git add pyproject.toml src/riskprobe/monitoring tests/monitoring/test_reference.py
git commit -m "feat: add privacy-safe monitoring snapshots"
```

## Task 2: Schema、缺失、PSI、标签和规则衰减检测

**Files:**
- Create: `src/riskprobe/monitoring/detection.py`
- Create: `tests/monitoring/test_detection.py`
- Modify: `tests/monitoring/conftest.py`

**Interfaces:**
- Produces: `detect_anomalies(reference, current_frame, current_rule_cards, catalog, config) -> list[Alert]`
- Consumes: Task 1 models，Plan 1 metrics/evidence

在 `tests/monitoring/conftest.py` 增加：

```python
from riskprobe.monitoring.reference import build_reference_snapshot


@pytest.fixture
def reference_snapshot(reference_fixture):
    return build_reference_snapshot(**reference_fixture)
```

- [ ] **Step 1: 编写缺失突增和规则衰减测试**

```python
# tests/monitoring/test_detection.py
import polars as pl

from riskprobe.monitoring.detection import detect_anomalies


def test_missingness_jump_creates_feature_and_family_alert(reference_snapshot, catalog) -> None:
    current = pl.DataFrame({
        "target": [0, 1] * 100,
        "institution": ["A"] * 200,
        "browse_pv_30d": [None] * 80 + [1.0] * 120,
    })

    alerts = detect_anomalies(reference_snapshot, current, (), catalog)

    assert any(a.alert_type == "missingness" and a.scope_value == "browse_pv_30d" for a in alerts)
    assert any(a.scope == "family" and a.scope_value == "browse" for a in alerts)
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/python -m pytest tests/monitoring/test_detection.py -v`

Expected: FAIL，检测器不存在。

- [ ] **Step 3: 实现固定阈值和告警 ID**

默认阈值：

```text
missing_rate_delta_warning = 0.10
missing_rate_delta_critical = 0.25
psi_warning = 0.20
psi_critical = 0.30
positive_rate_delta_warning = 0.03
positive_rate_delta_critical = 0.08
rule_lift_decay_warning = 0.20
rule_lift_decay_critical = 0.40
institution_share_delta_warning = 0.10
```

PSI 使用参考快照的固定分箱，桶概率最小截断为 `1e-6`。告警 ID 为 `alert_type|scope|scope_value|metric` 的 SHA-256 前 12 位。Schema 缺列或类型族变化直接 critical；多个同族特征在当前数据中异常时追加 family 告警，`evidence` 写入异常特征数和特征族总数。

- [ ] **Step 4: 运行检测测试**

Run: `.venv/bin/python -m pytest tests/monitoring/test_detection.py -v`

Expected: PASS，缺失率特征告警和浏览特征族告警同时存在。

- [ ] **Step 5: 用户授权后提交检查点**

```bash
git add src/riskprobe/monitoring/detection.py tests/monitoring/test_detection.py
git commit -m "feat: detect risk data and rule drift"
```

## Task 3: 六类有真值漂移注入与检测评估

**Files:**
- Create: `src/riskprobe/monitoring/injection.py`
- Create: `tests/monitoring/test_injection.py`

**Interfaces:**
- Produces: `DriftScenario`, `InjectedDrift`, `inject_drift(frame, scenario, seed) -> InjectedDrift`
- Produces: `evaluate_alerts(alerts, truth, top_k=3) -> DetectionScore`
- Consumes: Task 1 `Alert`，Plan 1 synthetic frame

- [ ] **Step 1: 编写注入可复现性和真值评分测试**

```python
# tests/monitoring/test_injection.py
from riskprobe.monitoring.injection import DriftScenario, inject_drift
from riskprobe.synthetic import generate_behavior_dataset


def test_missingness_injection_is_reproducible_and_records_truth() -> None:
    frame, _ = generate_behavior_dataset(5_000, seed=42)
    scenario = DriftScenario(
        scenario_id="missing-browse",
        drift_type="missingness",
        target="browse_pv_30d",
        magnitude=0.30,
        institution="B",
    )

    first = inject_drift(frame, scenario, seed=7)
    second = inject_drift(frame, scenario, seed=7)

    assert first.frame.equals(second.frame)
    assert first.truth.expected_alert_type == "missingness"
    assert first.truth.expected_scope_value == "browse_pv_30d"
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/python -m pytest tests/monitoring/test_injection.py -v`

Expected: FAIL，注入器不存在。

- [ ] **Step 3: 实现六类场景**

`DriftScenario.drift_type` 只允许：

- `missingness`：目标切片按 magnitude 置 null；
- `numeric_shift`：数值增加 `magnitude * reference_std`；
- `population_shift`：对目标机构按 magnitude 过采样；
- `label_shift`：对目标切片按 magnitude 翻转 0→1；
- `schema`：删除目标列；
- `rule_decay`：对命中目标规则的正类按 magnitude 翻转 1→0。

每次注入返回修改后的新 DataFrame 和 `DriftTruth`，原 DataFrame 不变。`evaluate_alerts` 用 `(alert_type, scope_value)` 匹配真值，输出 precision、recall、false_positive_rate 和 top_k_root_cause_hit。

- [ ] **Step 4: 运行测试和六场景参数化验证**

Run: `.venv/bin/python -m pytest tests/monitoring/test_injection.py -v`

Expected: 六种场景均可构造且相同种子结果一致。

- [ ] **Step 5: 用户授权后提交检查点**

```bash
git add src/riskprobe/monitoring/injection.py tests/monitoring/test_injection.py
git commit -m "feat: add measurable drift injection"
```

## Task 4: 根因贡献排序

**Files:**
- Create: `src/riskprobe/monitoring/diagnosis.py`
- Create: `tests/monitoring/test_diagnosis.py`

**Interfaces:**
- Produces: `diagnose_alerts(alerts, reference, current_frame, catalog, top_k) -> list[Diagnosis]`
- Consumes: Tasks 1–2 models，Plan 1 catalog

- [ ] **Step 1: 编写机构和特征族根因测试**

```python
# tests/monitoring/test_diagnosis.py
from riskprobe.monitoring.detection import detect_anomalies
from riskprobe.monitoring.diagnosis import diagnose_alerts
from riskprobe.monitoring.injection import DriftScenario, inject_drift
from riskprobe.synthetic import generate_behavior_dataset


def test_browser_missingness_is_attributed_to_institution_b(reference_snapshot, catalog) -> None:
    frame, _ = generate_behavior_dataset(5_000, seed=42)
    injected = inject_drift(
        frame,
        DriftScenario(
            scenario_id="browse-b",
            drift_type="missingness",
            target="browse_pv_30d",
            magnitude=0.60,
            institution="B",
        ),
        seed=7,
    )
    alerts = detect_anomalies(reference_snapshot, injected.frame, (), catalog)
    diagnoses = diagnose_alerts(
        alerts,
        reference_snapshot,
        injected.frame,
        catalog,
        top_k=3,
    )

    causes = diagnoses[0].root_causes
    assert causes[0].dimension == "segment"
    assert causes[0].value == "B"
    assert any(c.dimension == "family" and c.value == "browse" for c in causes)
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/python -m pytest tests/monitoring/test_diagnosis.py -v`

Expected: FAIL，诊断器不存在。

- [ ] **Step 3: 实现可解释贡献分数**

对 feature missingness/PSI 告警，按 segment 计算 `abs(current_metric - reference_metric) * current_share`；对 label/rule 告警按 segment 和月份计算坏账率或 Lift 变化的绝对贡献；按 feature catalog 聚合 family。`RootCause` 固定字段为 `dimension`, `value`, `contribution`, `rank`, `evidence`。排序键为贡献降序、dimension、value，保证确定性。

- [ ] **Step 4: 运行根因测试**

Run: `.venv/bin/python -m pytest tests/monitoring/test_diagnosis.py -v`

Expected: 机构 B 排名第一，browse family 出现在 Top 3。

- [ ] **Step 5: 用户授权后提交检查点**

```bash
git add src/riskprobe/monitoring/diagnosis.py tests/monitoring/test_diagnosis.py
git commit -m "feat: rank anomaly root causes"
```

## Task 5: 数据集白名单注册表与隐私输出门控

**Files:**
- Create: `src/riskprobe/registry.py`
- Create: `src/riskprobe/privacy.py`
- Create: `configs/datasets.example.yaml`
- Create: `tests/test_registry.py`
- Create: `tests/test_privacy.py`

**Interfaces:**
- Produces: `DatasetRegistry.from_yaml(path)`, `get_config(dataset_id) -> ProjectConfig`
- Produces: `assert_safe_payload(payload, forbidden_fields) -> None`
- Produces: `suppress_small_groups(records, count_key, min_group_size)`
- Consumes: Plan 1 `ProjectConfig`

- [ ] **Step 1: 编写路径注入拒绝和小样本抑制测试**

```python
# tests/test_registry.py
import yaml
import pytest

from riskprobe.registry import DatasetNotRegisteredError, DatasetRegistry


def test_registry_rejects_path_instead_of_dataset_id(tmp_path, synthetic_config) -> None:
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(
        yaml.safe_dump(synthetic_config.model_dump(mode="json")),
        encoding="utf-8",
    )
    registry_path = tmp_path / "datasets.yaml"
    registry_path.write_text(
        yaml.safe_dump({"datasets": {"synthetic_demo": {"config": str(config_path)}}}),
        encoding="utf-8",
    )
    registry = DatasetRegistry.from_yaml(registry_path)

    with pytest.raises(DatasetNotRegisteredError):
        registry.get_config("/tmp/company.parquet")
```

```python
# tests/test_privacy.py
from riskprobe.privacy import suppress_small_groups


def test_small_group_is_removed_from_tool_output() -> None:
    records = [{"institution": "A", "count": 99}, {"institution": "B", "count": 101}]
    assert suppress_small_groups(records, "count", 100) == [
        {"institution": "B", "count": 101}
    ]
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/python -m pytest tests/test_registry.py tests/test_privacy.py -v`

Expected: FAIL，模块不存在。

- [ ] **Step 3: 实现白名单注册表**

注册表 YAML：

```yaml
datasets:
  synthetic_demo:
    config: configs/synthetic.example.yaml
```

只允许 ID 匹配 `^[a-z][a-z0-9_-]{2,63}$`；配置路径在服务启动时加载，不接受 MCP 参数覆盖。`assert_safe_payload` 递归检查 key，默认禁止 `entity_id`, `md5_phone`, `rows`, `records`, `raw_data`, `file_path`，检测到即抛出 `UnsafePayloadError`。

- [ ] **Step 4: 运行安全测试**

Run: `.venv/bin/python -m pytest tests/test_registry.py tests/test_privacy.py -v`

Expected: PASS。

- [ ] **Step 5: 用户授权后提交检查点**

```bash
git add src/riskprobe/registry.py src/riskprobe/privacy.py configs/datasets.example.yaml tests/test_registry.py tests/test_privacy.py
git commit -m "feat: enforce dataset and output allowlists"
```

## Task 6: 本地 MCP 六工具服务

**Files:**
- Create: `src/riskprobe/mcp_server.py`
- Create: `tests/test_mcp_server.py`

**Interfaces:**
- Produces MCP tools: `inspect_dataset`, `discover_rules`, `validate_rules`, `detect_anomalies`, `diagnose_anomaly`, `build_report`
- Consumes: Plan 1 `RiskProbeService`，Tasks 1–5

- [ ] **Step 1: 编写工具拒绝路径和输出安全测试**

```python
# tests/test_mcp_server.py
import yaml
import pytest

from riskprobe.artifacts import RunStore
from riskprobe.mcp_server import RiskProbeTools
from riskprobe.registry import DatasetNotRegisteredError, DatasetRegistry


def make_tools(tmp_path, synthetic_config) -> RiskProbeTools:
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(
        yaml.safe_dump(synthetic_config.model_dump(mode="json")),
        encoding="utf-8",
    )
    registry_path = tmp_path / "datasets.yaml"
    registry_path.write_text(
        yaml.safe_dump({"datasets": {"synthetic_demo": {"config": str(config_path)}}}),
        encoding="utf-8",
    )
    return RiskProbeTools(
        DatasetRegistry.from_yaml(registry_path),
        RunStore(tmp_path / "runs"),
    )


def test_tools_accept_dataset_id_but_reject_path(tmp_path, synthetic_config) -> None:
    tools = make_tools(tmp_path, synthetic_config)
    with pytest.raises(DatasetNotRegisteredError):
        tools.inspect_dataset("/tmp/private.parquet")


def test_inspect_output_has_no_entity_or_path(tmp_path, synthetic_config) -> None:
    tools = make_tools(tmp_path, synthetic_config)
    payload = tools.inspect_dataset("synthetic_demo")
    serialized = str(payload)
    assert "entity_id" not in serialized
    assert ".parquet" not in serialized
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/python -m pytest tests/test_mcp_server.py -v`

Expected: FAIL，MCP 服务不存在。

- [ ] **Step 3: 实现可直接测试的工具类**

`RiskProbeTools` 中六个同步方法返回 Pydantic 模型的 `model_dump(mode="json")`；每次返回前调用 `assert_safe_payload`。`discover_rules` 只返回 rule ID、脱敏表达式、origin 和聚合 Train 指标；`validate_rules` 返回证据卡；`detect_anomalies` 使用 reference run ID 和 current dataset ID；`build_report` 返回 report path 的逻辑 ID 和 Markdown 内容，不返回真实磁盘路径。

- [ ] **Step 4: 用 FastMCP 暴露同名函数**

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("riskprobe")

@mcp.tool()
def inspect_dataset(dataset_id: str) -> dict:
    return get_tools().inspect_dataset(dataset_id)

# 其余五个工具使用同一模式委托给 RiskProbeTools。

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

注册表路径只从 `RISKPROBE_REGISTRY` 环境变量读取；未设置时使用 `configs/datasets.example.yaml`。环境变量只接受注册表文件路径，不接受数据文件路径。

- [ ] **Step 5: 运行 MCP 测试和进程导入冒烟**

Run:

```bash
.venv/bin/python -m pytest tests/test_mcp_server.py -v
.venv/bin/python -c 'from riskprobe.mcp_server import mcp; print(mcp.name)'
```

Expected: 测试通过，打印 `riskprobe`。

- [ ] **Step 6: 用户授权后提交检查点**

```bash
git add src/riskprobe/mcp_server.py tests/test_mcp_server.py
git commit -m "feat: expose allowlisted risk mcp tools"
```

## Task 7: Kiro Custom Agent、MCP 配置与分析 SOP

**Files:**
- Create: `.kiro/settings/mcp.json`
- Create: `.kiro/agents/riskprobe.json`
- Create: `.kiro/skills/riskprobe/SKILL.md`
- Create: `tests/test_kiro_config.py`

**Interfaces:**
- Produces: Kiro workspace agent `riskprobe`
- Consumes: Task 6 stdio MCP server

- [ ] **Step 1: 编写静态权限测试**

```python
# tests/test_kiro_config.py
import json
from pathlib import Path


def test_agent_exposes_only_riskprobe_and_denies_builtin_capabilities() -> None:
    config = json.loads(Path(".kiro/agents/riskprobe.json").read_text())
    assert config["tools"] == ["@riskprobe"]
    rules = {(r["capability"], r["effect"]) for r in config["permissions"]["rules"]}
    assert ("mcp", "allow") in rules
    for capability in {"shell", "fs_read", "fs_write", "web_fetch", "web_search"}:
        assert (capability, "deny") in rules
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/python -m pytest tests/test_kiro_config.py -v`

Expected: FAIL，Agent 配置不存在。

- [ ] **Step 3: 创建 workspace MCP 配置**

```json
{
  "mcpServers": {
    "riskprobe": {
      "command": ".venv/bin/python",
      "args": ["-m", "riskprobe.mcp_server"]
    }
  }
}
```

- [ ] **Step 4: 创建最小权限 Agent 配置**

`.kiro/agents/riskprobe.json` 必须精确包含：

```json
{
  "name": "riskprobe",
  "description": "Analyze risk rules and anomalies through privacy-safe local tools.",
  "tools": ["@riskprobe"],
  "includeMcpJson": true,
  "includePowers": false,
  "resources": ["skill://.kiro/skills/riskprobe/SKILL.md"],
  "permissions": {
    "rules": [
      {"capability": "mcp", "match": ["riskprobe/*"], "effect": "allow"},
      {"capability": "shell", "effect": "deny"},
      {"capability": "fs_read", "effect": "deny"},
      {"capability": "fs_write", "effect": "deny"},
      {"capability": "web_fetch", "effect": "deny"},
      {"capability": "web_search", "effect": "deny"}
    ]
  }
}
```

- [ ] **Step 5: 编写 Agent Skill SOP**

Skill frontmatter 使用 `name: riskprobe`。正文规定：先 inspect；C/D 停止；规则必须 discover 后 validate；异常必须 detect 后 diagnose；同一失败最多重试一次；禁止把 B 级结果称为严格 OOT、无穿越或可上线；报告必须列证据和限制；不得请求用户明细或真实路径。

- [ ] **Step 6: 运行配置测试**

Run:

```bash
.venv/bin/python -m pytest tests/test_kiro_config.py -v
.venv/bin/python -m json.tool .kiro/settings/mcp.json >/dev/null
.venv/bin/python -m json.tool .kiro/agents/riskprobe.json >/dev/null
```

Expected: 测试和 JSON 校验全部通过。

- [ ] **Step 7: 用户授权后提交检查点**

```bash
git add .kiro/settings/mcp.json .kiro/agents/riskprobe.json .kiro/skills/riskprobe/SKILL.md tests/test_kiro_config.py
git commit -m "feat: add minimum-permission kiro risk agent"
```

## Task 8: 监控 CLI 与端到端漂移评估

**Files:**
- Modify: `src/riskprobe/cli.py`
- Modify: `src/riskprobe/service.py`
- Create: `tests/test_monitoring_cli.py`

**Interfaces:**
- Produces CLI: `riskprobe snapshot`, `riskprobe monitor`, `riskprobe evaluate-drift`
- Consumes: Tasks 1–7

- [ ] **Step 1: 编写监控命令集成测试**

```python
# tests/test_monitoring_cli.py
from typer.testing import CliRunner
from riskprobe.cli import app

runner = CliRunner()


def test_evaluate_drift_reports_recall(tmp_path) -> None:
    data_path = tmp_path / "behavior.parquet"
    generated = runner.invoke(
        app,
        ["synthetic", "--output", str(data_path), "--rows", "5000", "--seed", "42"],
    )
    assert generated.exit_code == 0
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
dataset: {{id: synthetic_demo, path: {data_path}}}
columns: {{entity: entity_id, snapshot: snapshot_date, segment: institution, target: target}}
target: {{positive_value: 1, positive_meaning: bad_debt, performance_window_days: null}}
snapshot: {{meaning: customer_specified_feature_cutoff}}
features: {{families: {{order: [order_], browse: [browse_], multi: [multi_], embedding: [emb_]}}}}
segment_display_name: institution
time_validation_enabled: true
""".strip(),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "evaluate-drift",
            "--config", str(config_path),
            "--runs-dir", str(tmp_path / "runs"),
            "--seed", "42",
        ],
    )
    assert result.exit_code == 0
    assert "recall" in result.stdout.lower()
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/python -m pytest tests/test_monitoring_cli.py -v`

Expected: FAIL，命令不存在。

- [ ] **Step 3: 实现三个命令**

```text
riskprobe snapshot --config PATH --runs-dir PATH
riskprobe monitor --reference-run-id ID --current-config PATH --runs-dir PATH
riskprobe evaluate-drift --config PATH --runs-dir PATH --seed INTEGER
```

`evaluate-drift` 对六类场景逐一注入、检测和诊断，输出总体 Precision、Recall、false positive rate、Top-3 hit rate，并写 `anomaly_alerts.json`、`diagnoses.json`、`drift_evaluation.json`。

- [ ] **Step 4: 运行端到端验证**

Run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
.venv/bin/riskprobe synthetic --output /tmp/riskprobe-monitor.parquet --rows 20000 --seed 42
cp configs/synthetic.example.yaml /tmp/riskprobe-monitor.yaml
.venv/bin/python -c 'from pathlib import Path; p=Path("/tmp/riskprobe-monitor.yaml"); p.write_text(p.read_text().replace("data/synthetic/behavior.parquet", "/tmp/riskprobe-monitor.parquet"), encoding="utf-8")'
.venv/bin/riskprobe evaluate-drift --config /tmp/riskprobe-monitor.yaml --runs-dir /tmp/riskprobe-monitor-runs --seed 42
```

Expected: 测试和 Ruff 通过；CLI 输出四项指标；运行目录中三类监控产物存在。

- [ ] **Step 5: Kiro 手工冒烟**

在 Kiro Agent 选择器中启用 `riskprobe`，输入：

```text
检查 synthetic_demo 的元数据等级，并说明哪些结论被限制。不要读取文件或执行 Shell。
```

Expected: Agent 只调用 `riskprobe/inspect_dataset`；回答包含 B 级和表现窗口未知限制；工具记录中没有 Shell、文件或网络调用。若 MCP 无法连接，先在 MCP Server 视图重连，不扩大 Agent 权限。

- [ ] **Step 6: 用户授权后提交检查点**

```bash
git add src/riskprobe/cli.py src/riskprobe/service.py tests/test_monitoring_cli.py
git commit -m "feat: add anomaly monitoring workflow"
```

## Plan 2 Completion Gate

Run:

```bash
.venv/bin/python -m pytest --cov=riskprobe --cov-report=term-missing
.venv/bin/ruff check src tests
.venv/bin/python -m json.tool .kiro/settings/mcp.json >/dev/null
.venv/bin/python -m json.tool .kiro/agents/riskprobe.json >/dev/null
```

验收条件：

- 六类漂移都带机器可读真值并产生检测评分；
- 告警 ID、根因排序和评估结果可复现；
- MCP 拒绝文件路径和未登记 dataset ID；
- MCP payload 不含 entity、真实路径、样本行和小样本组；
- Kiro Agent 仅暴露 `@riskprobe`，五类内置能力均为 deny；
- Agent 遵循 inspect → discover/monitor → validate/diagnose → report 状态机；
- 无外部模型 API 依赖，原始数据始终在本地 Python 进程中。

## Official Kiro References

- Agent config: https://kiro.dev/docs/cli/v3/agent-config.md
- Custom agent configuration: https://kiro.dev/docs/custom-agents/configuration-reference.md
- Permissions: https://kiro.dev/docs/permissions.md
- MCP configuration: https://kiro.dev/docs/mcp/configuration.md
- Agent Skills: https://kiro.dev/docs/skills.md
