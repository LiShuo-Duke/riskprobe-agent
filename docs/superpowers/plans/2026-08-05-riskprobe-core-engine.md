# RiskProbe Core Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建无需模型 API 即可运行的 RiskProbe 核心引擎，完成本地 Parquet 检查、行为特征规则发现、证据门控、合成数据演示和 CLI 端到端产物。

**Architecture:** 使用 `src` 布局，将配置契约、Parquet 访问、行为特征目录、指标、规则发现、规则验证和运行产物分离。所有阈值只从 Train 产生，Test/时间切片只做验证；CLI 与后续 MCP 共用 `RiskProbeService`，避免两套计算逻辑。

**Tech Stack:** Python 3.11+（本机 3.13.5）、Polars 1.33.1、PyArrow 21.0.0、NumPy 2.3.2、SciPy 1.16.1、scikit-learn 1.7.1、LightGBM 4.6.0、statsmodels 0.14.5、Pydantic 2.11.7、PyYAML 6.0.2、Typer 0.16.0、Rich 14.1.0、pytest 8.4.1、Ruff 0.12.8。

## Global Constraints

- 公司输入仅为本地只读 Parquet；不实现 SQL、Spark SQL 或数据仓库连接。
- `target=1` 表示坏账；`huisu_date` 只作为客户指定样本截面日期，不解释为申请日或坏账发生日。
- 标签表现窗口未知时元数据最高为 B 级，并使用“跨时间切片稳定性”措辞。
- 原始数据、用户标识、真实公司配置、内部结果不得进入 Git；遵守现有 `.gitignore`。
- 规则只从 Train 生成，Test 和时间切片不得参与阈值搜索。
- 抽样仅生成候选规则，最终证据必须在未改变分布的数据上重算。
- 所有随机过程显式接收 `random_seed=42`。
- 每个任务包含提交命令，但只有用户明确授权提交后才能执行。

---

## File Map

```text
pyproject.toml                         # 包元数据、精确依赖、CLI 入口、pytest/ruff 配置
src/riskprobe/__init__.py              # 版本号
src/riskprobe/config.py                # YAML 配置和 Pydantic 数据契约
src/riskprobe/models.py                # Profile、规则、指标、证据卡模型
src/riskprobe/io/parquet.py            # 只读 Parquet 惰性扫描和列裁剪
src/riskprobe/features/catalog.py      # 特征族、窗口元数据和行为约束
src/riskprobe/profiling.py             # 数据契约检查和聚合 Profile
src/riskprobe/metrics.py               # 坏账率、Lift、KS、PSI、Bootstrap、BH 校正
src/riskprobe/rules/expression.py      # 规则表达式及 Polars/Pandas 求值
src/riskprobe/rules/discovery.py       # 分位点、树分裂、LightGBM 分裂及二阶 Beam Search
src/riskprobe/rules/validation.py      # 全量验证、机构/时间稳定性和证据分级
src/riskprobe/synthetic.py             # 可复现订单/浏览行为宽表生成器
src/riskprobe/artifacts.py             # run_id、Manifest 和不可变运行目录
src/riskprobe/reporting.py             # 确定性 Markdown 报告
src/riskprobe/service.py               # CLI/MCP 共用应用服务
src/riskprobe/cli.py                   # Typer CLI
configs/synthetic.example.yaml         # 可公开的合成数据配置
tests/                                 # 与模块同名的单元和集成验证
```

## Task 1: 工程骨架与配置契约

**Files:**
- Create: `pyproject.toml`
- Create: `src/riskprobe/__init__.py`
- Create: `src/riskprobe/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `ProjectConfig.from_yaml(path: Path) -> ProjectConfig`
- Produces: `DatasetConfig`, `ColumnRoles`, `TargetConfig`, `SnapshotConfig`, `FeatureFamilyConfig`, `DiscoveryConfig`, `ValidationConfig`
- Consumes: 无

- [ ] **Step 1: 编写配置解析失败用例**

```python
# tests/test_config.py
from pathlib import Path

import pytest
from pydantic import ValidationError

from riskprobe.config import ProjectConfig


def test_company_metadata_without_performance_window_is_grade_b(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  id: demo
  path: /tmp/demo.parquet
columns:
  entity: entity_id
  snapshot: snapshot_date
  segment: institution
  target: target
target:
  positive_value: 1
  positive_meaning: bad_debt
  performance_window_days: null
snapshot:
  meaning: customer_specified_feature_cutoff
features:
  families:
    order: [order_]
    browse: [browse_]
""".strip(),
        encoding="utf-8",
    )

    config = ProjectConfig.from_yaml(config_path)

    assert config.dataset.id == "demo"
    assert config.metadata_grade == "B"


def test_unknown_snapshot_semantics_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
dataset: {id: demo, path: /tmp/demo.parquet}
columns: {entity: id, snapshot: dt, segment: org, target: y}
target: {positive_value: 1, positive_meaning: bad_debt}
snapshot: {meaning: bad_debt_date}
features: {families: {order: [order_]}}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        ProjectConfig.from_yaml(config_path)
```

- [ ] **Step 2: 创建虚拟环境并确认测试失败**

Run:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest tests/test_config.py -v
```

Expected: 在实现文件创建前，测试因 `riskprobe.config` 不存在而失败。

- [ ] **Step 3: 创建精确依赖和配置模型**

`pyproject.toml` 使用以下核心内容：

```toml
[project]
name = "riskprobe-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "lightgbm==4.6.0",
  "numpy==2.3.2",
  "polars==1.33.1",
  "pyarrow==21.0.0",
  "pydantic==2.11.7",
  "PyYAML==6.0.2",
  "rich==14.1.0",
  "scikit-learn==1.7.1",
  "scipy==1.16.1",
  "statsmodels==0.14.5",
  "typer==0.16.0",
]

[project.optional-dependencies]
dev = ["pytest==8.4.1", "pytest-cov==6.2.1", "ruff==0.12.8"]

[project.scripts]
riskprobe = "riskprobe.cli:app"

[build-system]
requires = ["hatchling==1.27.0"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/riskprobe"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
target-version = "py311"
line-length = 100
```

`src/riskprobe/config.py` 定义严格模型：

```python
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetConfig(StrictModel):
    id: str
    path: Path
    read_only: bool = True


class ColumnRoles(StrictModel):
    entity: str
    snapshot: str
    segment: str
    target: str


class TargetConfig(StrictModel):
    positive_value: int = 1
    positive_meaning: Literal["bad_debt"]
    performance_window_days: int | None = Field(default=None, gt=0)


class SnapshotConfig(StrictModel):
    meaning: Literal[
        "customer_specified_feature_cutoff",
        "public_relative_reference",
    ]


class FeatureFamilyConfig(StrictModel):
    families: dict[str, list[str]]
    explicit_catalog: Path | None = None


class DiscoveryConfig(StrictModel):
    min_support: float = Field(default=0.05, gt=0, lt=1)
    max_single_rules: int = Field(default=100, ge=1)
    beam_width: int = Field(default=20, ge=1)
    max_pair_rules: int = Field(default=50, ge=0)
    random_seed: int = 42


class ValidationConfig(StrictModel):
    alpha: float = Field(default=0.05, gt=0, lt=1)
    min_segment_consistency: float = Field(default=0.6, ge=0, le=1)
    max_lift_decay: float = Field(default=0.3, ge=0)
    bootstrap_rounds: int = Field(default=500, ge=100)
    min_group_size: int = Field(default=100, ge=20)


class ProjectConfig(StrictModel):
    dataset: DatasetConfig
    columns: ColumnRoles
    target: TargetConfig
    snapshot: SnapshotConfig
    features: FeatureFamilyConfig
    segment_display_name: Literal["institution", "customer_segment"] = "institution"
    time_validation_enabled: bool = True
    discovery: DiscoveryConfig = DiscoveryConfig()
    validation: ValidationConfig = ValidationConfig()

    @property
    def metadata_grade(self) -> Literal["A", "B"]:
        return "A" if self.target.performance_window_days is not None else "B"

    @classmethod
    def from_yaml(cls, path: Path) -> "ProjectConfig":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(payload)
```

- [ ] **Step 4: 运行配置测试和 Ruff**

Run:

```bash
.venv/bin/python -m pytest tests/test_config.py -v
.venv/bin/ruff check src tests
```

Expected: 2 tests pass，Ruff 无错误。

- [ ] **Step 5: 用户授权后提交检查点**

```bash
git add pyproject.toml src/riskprobe/__init__.py src/riskprobe/config.py tests/test_config.py
git commit -m "feat: add project configuration contract"
```

## Task 2: 规则模型、指标与表达式

**Files:**
- Create: `src/riskprobe/models.py`
- Create: `src/riskprobe/metrics.py`
- Create: `src/riskprobe/rules/__init__.py`
- Create: `src/riskprobe/rules/expression.py`
- Create: `tests/test_metrics.py`
- Create: `tests/rules/test_expression.py`

**Interfaces:**
- Produces: `Condition`, `RiskRule`, `RuleMetrics`, `SliceMetrics`, `EvidenceCard`
- Produces: `evaluate_rule(frame: pl.DataFrame, rule: RiskRule) -> pl.Series`
- Produces: `compute_rule_metrics(mask, target, positive_value) -> RuleMetrics`
- Produces: `bootstrap_lift_ci(...) -> tuple[float, float]`, `adjust_pvalues(...) -> list[float]`
- Consumes: Task 1 `ValidationConfig`

- [ ] **Step 1: 编写规则和指标手算测试**

```python
# tests/test_metrics.py
import numpy as np

from riskprobe.metrics import compute_rule_metrics


def test_rule_metrics_match_hand_calculation() -> None:
    target = np.array([1, 1, 0, 0, 0, 0])
    mask = np.array([True, True, True, False, False, False])

    metrics = compute_rule_metrics(mask, target, positive_value=1)

    assert metrics.support_count == 3
    assert metrics.coverage == 0.5
    assert metrics.hit_bad_rate == 2 / 3
    assert metrics.base_bad_rate == 1 / 3
    assert metrics.lift == 2.0
    assert metrics.precision == 2 / 3
    assert metrics.recall == 1.0
```

```python
# tests/rules/test_expression.py
import polars as pl

from riskprobe.models import Condition, RiskRule
from riskprobe.rules.expression import evaluate_rule


def test_two_condition_rule_handles_null_as_false() -> None:
    frame = pl.DataFrame({"a": [1.0, 4.0, None], "b": [0.1, 0.3, 0.2]})
    rule = RiskRule(
        rule_id="R_test",
        conditions=(
            Condition(feature="a", operator=">", value=3.0),
            Condition(feature="b", operator="<=", value=0.3),
        ),
        origin="quantile_pair",
    )

    assert evaluate_rule(frame, rule).to_list() == [False, True, False]
```

- [ ] **Step 2: 运行测试并确认缺少模型/函数**

Run: `.venv/bin/python -m pytest tests/test_metrics.py tests/rules/test_expression.py -v`

Expected: FAIL，提示 `riskprobe.models` 或目标函数不存在。

- [ ] **Step 3: 实现不可变模型和规则求值**

核心类型必须固定为：

```python
# src/riskprobe/models.py
from typing import Literal
from pydantic import BaseModel, ConfigDict

Operator = Literal[">", ">=", "<", "<=", "==", "!=", "is_null"]
EvidenceGrade = Literal["Stable", "Local", "Unstable", "Suspicious"]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Condition(FrozenModel):
    feature: str
    operator: Operator
    value: float | int | str | None = None


class RiskRule(FrozenModel):
    rule_id: str
    conditions: tuple[Condition, ...]
    origin: str


class RuleMetrics(FrozenModel):
    support_count: int
    coverage: float
    base_bad_rate: float
    hit_bad_rate: float
    non_hit_bad_rate: float
    lift: float
    precision: float
    recall: float
    p_value: float


class SliceMetrics(FrozenModel):
    slice_type: Literal["dataset", "institution", "time"]
    slice_value: str
    metrics: RuleMetrics


class EvidenceCard(FrozenModel):
    rule: RiskRule
    train: RuleMetrics
    test: RuleMetrics
    slices: tuple[SliceMetrics, ...]
    lift_ci: tuple[float, float]
    adjusted_p_value: float
    segment_consistency: float
    max_time_decay: float
    grade: EvidenceGrade
    limitations: tuple[str, ...] = ()
```

`evaluate_rule` 将每个条件转换为 Polars 表达式，所有条件使用 AND，null 默认填充为 `False`。`is_null` 是唯一把 null 判为命中的操作符。

- [ ] **Step 4: 实现指标、Fisher 检验、Bootstrap 和 BH 校正**

`compute_rule_metrics` 使用 2×2 列联表计算 Fisher 精确检验；当总体正类率为零时显式抛出 `ValueError("target has no positive samples")`。`bootstrap_lift_ci` 使用 `np.random.default_rng(random_seed)`，每轮对行索引有放回采样；`adjust_pvalues` 调用 `statsmodels.stats.multitest.multipletests(method="fdr_bh")`。

- [ ] **Step 5: 运行测试**

Run:

```bash
.venv/bin/python -m pytest tests/test_metrics.py tests/rules/test_expression.py -v
.venv/bin/ruff check src tests
```

Expected: 全部通过。

- [ ] **Step 6: 用户授权后提交检查点**

```bash
git add src/riskprobe/models.py src/riskprobe/metrics.py src/riskprobe/rules tests/test_metrics.py tests/rules/test_expression.py
git commit -m "feat: add auditable rule metrics"
```

## Task 3: 合成订单/浏览宽表与只读 Parquet 访问

**Files:**
- Create: `src/riskprobe/io/__init__.py`
- Create: `src/riskprobe/io/parquet.py`
- Create: `src/riskprobe/synthetic.py`
- Create: `tests/io/test_parquet.py`
- Create: `tests/test_synthetic.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Produces: `ParquetDataset(path: Path)` with `schema()`, `scan(columns)`, `collect(columns)`
- Produces: `generate_behavior_dataset(rows: int, seed: int) -> tuple[pl.DataFrame, SyntheticTruth]`
- Produces: `SyntheticTruth.hidden_rules: tuple[RiskRule, ...]`
- Consumes: Task 2 `RiskRule`

- [ ] **Step 1: 编写列裁剪和合成真值测试**

```python
# tests/io/test_parquet.py
from pathlib import Path
import polars as pl

from riskprobe.io.parquet import ParquetDataset


def test_collect_reads_only_requested_columns(tmp_path: Path) -> None:
    path = tmp_path / "wide.parquet"
    pl.DataFrame({"id": [1, 2], "target": [0, 1], "unused": [9, 9]}).write_parquet(path)
    dataset = ParquetDataset(path)

    result = dataset.collect(["id", "target"])

    assert result.columns == ["id", "target"]
```

```python
# tests/test_synthetic.py
from riskprobe.synthetic import generate_behavior_dataset


def test_synthetic_behavior_is_reproducible_and_contains_hidden_rules() -> None:
    first, truth = generate_behavior_dataset(rows=2_000, seed=42)
    second, _ = generate_behavior_dataset(rows=2_000, seed=42)

    assert first.equals(second)
    assert 0.05 < first["target"].mean() < 0.35
    assert {"order_cancel_rate_30d", "browse_night_ratio_30d"}.issubset(first.columns)
    assert len(truth.hidden_rules) == 3
```

- [ ] **Step 2: 确认测试失败**

Run: `.venv/bin/python -m pytest tests/io/test_parquet.py tests/test_synthetic.py -v`

Expected: FAIL，模块尚不存在。

- [ ] **Step 3: 实现安全 Parquet 访问**

`ParquetDataset` 在构造时检查路径后缀为 `.parquet` 且存在；`scan(columns)` 先根据 `pl.scan_parquet(path).collect_schema()` 检查列，不存在时抛出 `MissingColumnsError(missing)`；`collect` 只能调用 `scan(columns).collect()`，不提供写回方法。

- [ ] **Step 4: 实现可复现行为数据和共享配置 fixture**

生成列至少包括：

```text
entity_id, snapshot_date, institution, target,
order_cnt_7d, order_cnt_30d, order_amount_30d,
order_cancel_rate_30d, browse_pv_7d, browse_pv_30d,
browse_days_30d, browse_night_ratio_30d,
browse_to_order_ratio_30d, multi_platform_cnt_30d, emb_00, emb_01
```

生成规则固定为：

1. `order_cancel_rate_30d > 0.45`；
2. `browse_night_ratio_30d > 0.55 AND browse_to_order_ratio_30d > 8`；
3. `multi_platform_cnt_30d >= 4 AND order_cnt_30d <= 1`。

使用逻辑函数构造坏账概率，加入机构和月份弱偏移，再用种子采样标签。窗口计数必须满足 `7d <= 30d`，比例截断到 `[0, 1]`。

`tests/conftest.py` 在本任务只定义不依赖后续服务的配置 fixture：

```python
from pathlib import Path
import pytest
from riskprobe.config import ProjectConfig
from riskprobe.synthetic import generate_behavior_dataset


@pytest.fixture
def synthetic_config(tmp_path: Path) -> ProjectConfig:
    data_path = tmp_path / "synthetic.parquet"
    frame, _ = generate_behavior_dataset(5_000, seed=42)
    frame.write_parquet(data_path)
    return ProjectConfig.model_validate({
        "dataset": {"id": "synthetic_test", "path": data_path},
        "columns": {
            "entity": "entity_id",
            "snapshot": "snapshot_date",
            "segment": "institution",
            "target": "target",
        },
        "target": {"positive_value": 1, "positive_meaning": "bad_debt"},
        "snapshot": {"meaning": "customer_specified_feature_cutoff"},
        "features": {"families": {"order": ["order_"], "browse": ["browse_"]}},
        "segment_display_name": "institution",
        "time_validation_enabled": True,
    })
```

- [ ] **Step 5: 运行测试并写入临时 Parquet 冒烟**

Run:

```bash
.venv/bin/python -m pytest tests/io/test_parquet.py tests/test_synthetic.py -v
.venv/bin/python -c 'from pathlib import Path; from riskprobe.synthetic import generate_behavior_dataset; df,_=generate_behavior_dataset(10000,42); df.write_parquet("/tmp/riskprobe-synthetic.parquet"); print(df.shape)'
```

Expected: tests pass，打印 `(10000, 16)` 或实现中固定的更多列数，且首维为 10000。

- [ ] **Step 6: 用户授权后提交检查点**

```bash
git add src/riskprobe/io src/riskprobe/synthetic.py tests/conftest.py tests/io tests/test_synthetic.py
git commit -m "feat: add synthetic behavior dataset"
```

## Task 4: 行为特征目录与数据 Profile

**Files:**
- Create: `src/riskprobe/features/__init__.py`
- Create: `src/riskprobe/features/catalog.py`
- Create: `src/riskprobe/profiling.py`
- Create: `tests/features/test_catalog.py`
- Create: `tests/test_profiling.py`

**Interfaces:**
- Produces: `FeatureSpec`, `FeatureCatalog.from_columns(columns, family_prefixes)`
- Produces: `check_window_invariants(frame, catalog) -> tuple[QualityIssue, ...]`
- Produces: `profile_dataset(dataset, config) -> DatasetProfile`
- Consumes: Task 1 config，Task 3 `ParquetDataset`

- [ ] **Step 1: 编写窗口异常和元数据等级测试**

```python
# tests/features/test_catalog.py
import polars as pl

from riskprobe.features.catalog import FeatureCatalog, check_window_invariants


def test_window_inversion_is_reported_at_feature_family_level() -> None:
    frame = pl.DataFrame({"order_cnt_7d": [5, 1], "order_cnt_30d": [3, 4]})
    catalog = FeatureCatalog.from_columns(
        frame.columns,
        {"order": ["order_"], "browse": ["browse_"]},
    )

    issues = check_window_invariants(frame, catalog)

    assert issues[0].code == "WINDOW_INVERSION"
    assert issues[0].family == "order"
    assert issues[0].affected_rows == 1
```

```python
# tests/test_profiling.py
from pathlib import Path
import polars as pl

from riskprobe.config import ProjectConfig
from riskprobe.io.parquet import ParquetDataset
from riskprobe.profiling import profile_dataset


def test_profile_is_grade_b_when_performance_window_is_unknown(
    tmp_path: Path, synthetic_config: ProjectConfig
) -> None:
    path = tmp_path / "sample.parquet"
    pl.DataFrame({
        "entity_id": ["a", "b", "c", "d"],
        "snapshot_date": ["2026-01-01"] * 4,
        "institution": ["A", "A", "B", "B"],
        "target": [0, 1, 0, 1],
        "order_cnt_7d": [0, 1, 2, 3],
        "order_cnt_30d": [1, 2, 3, 4],
    }).write_parquet(path)

    profile = profile_dataset(ParquetDataset(path), synthetic_config)

    assert profile.metadata_grade == "B"
    assert profile.row_count == 4
    assert "LABEL_PERFORMANCE_WINDOW_UNKNOWN" in {issue.code for issue in profile.issues}
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/python -m pytest tests/features/test_catalog.py tests/test_profiling.py -v`

Expected: FAIL，缺少目标模块。

- [ ] **Step 3: 实现目录解析和质量模型**

`FeatureSpec` 字段固定为 `name`, `family`, `window_days`, `aggregation`, `value_type`。用正则 `r"_(7|30|90|180|365)d(?:_|$)"` 解析窗口；无法识别前缀时 family 为 `unknown`。`QualityIssue` 固定包含 `code`, `severity`, `family`, `features`, `affected_rows`, `message`。

- [ ] **Step 4: 实现聚合 Profile**

`DatasetProfile` 包含 `dataset_id`, `row_count`, `feature_count`, `positive_rate`, `segment_counts`, `snapshot_min`, `snapshot_max`, `metadata_grade`, `issues`。Profile 不保存 entity 值或样本行；若缺少角色列则抛出 `DataContractError`；若某分组单一标签，生成 `SINGLE_CLASS_SLICE` 警告而非阻断全局分析。`time_validation_enabled=true` 时必须解析截面日期并填充起止范围；为 false 时只检查 snapshot 列存在且非全空，`snapshot_min`/`snapshot_max` 允许为 `None`，不得生成时间切片。

- [ ] **Step 5: 运行测试**

Run: `.venv/bin/python -m pytest tests/features/test_catalog.py tests/test_profiling.py -v`

Expected: 全部通过。

- [ ] **Step 6: 用户授权后提交检查点**

```bash
git add src/riskprobe/features src/riskprobe/profiling.py tests/features tests/test_profiling.py
git commit -m "feat: add behavior feature quality profiling"
```

## Task 5: 单变量与二阶候选规则发现

**Files:**
- Create: `src/riskprobe/rules/discovery.py`
- Create: `tests/rules/test_discovery.py`

**Interfaces:**
- Produces: `discover_rules(train: pl.DataFrame, feature_names: list[str], target_col: str, config: DiscoveryConfig) -> list[RiskRule]`
- Consumes: Task 1 `DiscoveryConfig`，Task 2 `RiskRule/evaluate_rule/compute_rule_metrics`

- [ ] **Step 1: 编写只从 Train 找回隐藏规则的测试**

```python
# tests/rules/test_discovery.py
from riskprobe.config import DiscoveryConfig
from riskprobe.rules.discovery import discover_rules
from riskprobe.synthetic import generate_behavior_dataset


def test_discovery_finds_cancel_rate_signal_deterministically() -> None:
    frame, _ = generate_behavior_dataset(rows=20_000, seed=42)
    train = frame.sort("snapshot_date").head(int(frame.height * 0.7))
    config = DiscoveryConfig(min_support=0.03, max_single_rules=40, beam_width=10)

    first = discover_rules(train, ["order_cancel_rate_30d"], "target", config)
    second = discover_rules(train, ["order_cancel_rate_30d"], "target", config)

    assert [rule.model_dump() for rule in first] == [rule.model_dump() for rule in second]
    assert any(
        c.feature == "order_cancel_rate_30d" and c.operator == ">"
        for rule in first
        for c in rule.conditions
    )
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/python -m pytest tests/rules/test_discovery.py -v`

Expected: FAIL，`discover_rules` 不存在。

- [ ] **Step 3: 实现三类单变量阈值**

对每个数值特征生成：

- 非空值的 10%、25%、50%、75%、90% 分位点；
- `DecisionTreeClassifier(max_depth=2, min_samples_leaf=ceil(min_support*n), random_state=seed)` 的内部节点阈值；
- `LGBMClassifier(n_estimators=30, max_depth=2, num_leaves=4, learning_rate=0.05, deterministic=True, force_col_wise=True, random_state=seed, verbosity=-1)` 的分裂阈值。

LightGBM 阈值从 `booster_.dump_model()["tree_info"]` 递归提取 `split_feature` 和 `threshold`。每个阈值同时生成 `<=` 与 `>` 条件；过滤重复阈值、覆盖率小于 `min_support` 或大于 `1-min_support` 的候选。

- [ ] **Step 4: 实现候选排序与二阶 Beam Search**

单变量按 `(lift, support_count, canonical_expression)` 降序排序，取 `max_single_rules`。二阶候选只组合不同特征的 Top `beam_width` 单规则，使用 AND，过滤覆盖不足、完全包含和重复表达式，取 `max_pair_rules`。规则 ID 为规范化 JSON 的 SHA-256 前 12 位，保证同一规则跨运行 ID 稳定。

- [ ] **Step 5: 运行发现测试并检查确定性**

Run:

```bash
.venv/bin/python -m pytest tests/rules/test_discovery.py -v
.venv/bin/ruff check src tests
```

Expected: 测试通过，两次规则列表完全一致。

- [ ] **Step 6: 用户授权后提交检查点**

```bash
git add src/riskprobe/rules/discovery.py tests/rules/test_discovery.py
git commit -m "feat: discover deterministic risk rules"
```

## Task 6: 全量验证、机构/时间稳定性与证据分级

**Files:**
- Create: `src/riskprobe/rules/validation.py`
- Create: `tests/rules/test_validation.py`

**Interfaces:**
- Produces: `validate_rules(train, test, rules, target_col, segment_col, snapshot_col, segment_display_name, time_validation_enabled, config, metadata_grade) -> list[EvidenceCard]`
- Consumes: Task 1 `ValidationConfig`，Task 2 metrics/models，Task 5 rules

- [ ] **Step 1: 编写多重检验和不稳定规则测试**

```python
# tests/rules/test_validation.py
import polars as pl

from riskprobe.config import ValidationConfig
from riskprobe.models import Condition, RiskRule
from riskprobe.rules.validation import validate_rules


def test_rule_with_test_lift_decay_is_not_stable() -> None:
    train = pl.DataFrame({
        "f": [1] * 100 + [0] * 100,
        "target": [1] * 50 + [0] * 50 + [1] * 10 + [0] * 90,
        "institution": ["A"] * 200,
        "snapshot_date": ["2026-01-01"] * 200,
    })
    test = pl.DataFrame({
        "f": [1] * 100 + [0] * 100,
        "target": [1] * 20 + [0] * 80 + [1] * 20 + [0] * 80,
        "institution": ["A"] * 200,
        "snapshot_date": ["2026-02-01"] * 200,
    })
    rule = RiskRule(
        rule_id="R1",
        conditions=(Condition(feature="f", operator=">", value=0.5),),
        origin="test",
    )

    cards = validate_rules(
        train,
        test,
        [rule],
        target_col="target",
        segment_col="institution",
        snapshot_col="snapshot_date",
        segment_display_name="institution",
        time_validation_enabled=True,
        config=ValidationConfig(bootstrap_rounds=100, min_group_size=20),
        metadata_grade="B",
    )

    assert cards[0].grade in {"Unstable", "Suspicious"}
    assert "label performance window unknown" in cards[0].limitations
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/python -m pytest tests/rules/test_validation.py -v`

Expected: FAIL，验证模块不存在。

- [ ] **Step 3: 实现切片验证和证据门控**

对每条规则：

1. 计算 Train/Test 指标和 Test Bootstrap Lift 区间；
2. 按机构计算满足 `min_group_size` 的切片，方向一致定义为切片 `lift > 1`；
3. `time_validation_enabled=true` 时将截面日期转成 `YYYY-MM`，计算满足最小样本量的时间切片；为 false 时不创建时间切片；
4. 启用时间验证时最大衰减定义为 `max(0, (train_lift - min_valid_time_lift) / train_lift)`，未启用时固定为 0 且报告不展示该指标；
5. 对所有规则的 Train p-value 做 BH 校正；
6. Grade 规则：调整后 p-value 超阈值、Test 下界不大于 1 或样本不足为 `Suspicious`；衰减超限为 `Unstable`；机构一致率不足但至少一个机构稳定为 `Local`；其余为 `Stable`；
7. metadata grade B 时添加 `label performance window unknown`，但不强制降级；C/D 由上游阻断。

- [ ] **Step 4: 运行验证测试**

Run: `.venv/bin/python -m pytest tests/rules/test_validation.py tests/test_metrics.py -v`

Expected: 全部通过。

- [ ] **Step 5: 用户授权后提交检查点**

```bash
git add src/riskprobe/rules/validation.py tests/rules/test_validation.py
git commit -m "feat: validate rule stability evidence"
```

## Task 7: 不可变产物、确定性报告与应用服务

**Files:**
- Create: `src/riskprobe/artifacts.py`
- Create: `src/riskprobe/reporting.py`
- Create: `src/riskprobe/service.py`
- Create: `tests/test_artifacts.py`
- Create: `tests/test_service.py`

**Interfaces:**
- Produces: `RunStore.create(config, data_fingerprint, code_version) -> RunContext`
- Produces: `RiskProbeService.inspect()`, `discover()`, `run()`
- Produces: `render_risk_report(profile, evidence_cards) -> str`
- Consumes: Tasks 1–6

- [ ] **Step 1: 编写不可变运行目录和端到端服务测试**

```python
# tests/test_artifacts.py
from riskprobe.artifacts import RunStore


def test_same_inputs_produce_same_run_id(tmp_path) -> None:
    store = RunStore(tmp_path)
    first = store.compute_run_id("cfg", "data", "0.1.0")
    second = store.compute_run_id("cfg", "data", "0.1.0")
    assert first == second
```

```python
# tests/test_service.py

def test_service_run_writes_required_artifacts(tmp_path, synthetic_config) -> None:
    from riskprobe.service import RiskProbeService

    service = RiskProbeService(config=synthetic_config, runs_dir=tmp_path / "runs")
    result = service.run()
    names = {path.name for path in result.run_dir.iterdir()}
    assert {
        "manifest.json",
        "metadata_report.json",
        "data_profile.json",
        "candidate_rules.parquet",
        "evidence_cards.json",
        "risk_report.md",
    } <= names
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/python -m pytest tests/test_artifacts.py tests/test_service.py -v`

Expected: FAIL，目标模块不存在。

- [ ] **Step 3: 实现运行标识和原子写入**

`run_id` 为配置规范 JSON、Parquet 元数据指纹和包版本拼接后的 SHA-256 前 16 位。所有 JSON 先写入同目录临时文件，再用 `Path.replace` 原子替换。若目标 `runs/<run_id>` 已完整存在，服务只读返回，不覆盖；若存在 `.incomplete` 标记，则允许清理该运行目录后重建。

- [ ] **Step 4: 实现服务编排**

`RiskProbeService.run()` 顺序固定为：加载配置 → Profile → 根据 `time_validation_enabled` 划分数据 → 仅在 Train 抽样发现规则 → 使用完整 Train/Test 及可用的 Holdout 涉及列验证 → 写产物 → 渲染报告。时间验证开启时按截面日期排序划分 60% Train/20% Test/20% Holdout；关闭时使用固定种子分层划分 70% Train/30% Test，且不生成时间切片指标。公司 B 级元数据在报告首页显示限制，不出现“严格 OOT”或“可上线”措辞。

- [ ] **Step 5: 实现确定性 Markdown 报告**

报告包含元数据等级、样本概况、质量问题、Stable/Local/Unstable/Suspicious 数量、Top 规则证据表和限制。表格按 grade、Test Lift、规则 ID 固定排序；数值统一保留 4 位小数。

- [ ] **Step 6: 运行服务测试**

Run: `.venv/bin/python -m pytest tests/test_artifacts.py tests/test_service.py -v`

Expected: 全部通过，测试临时目录包含六个规定产物。

- [ ] **Step 7: 用户授权后提交检查点**

```bash
git add src/riskprobe/artifacts.py src/riskprobe/reporting.py src/riskprobe/service.py tests/test_artifacts.py tests/test_service.py
git commit -m "feat: add reproducible risk analysis runs"
```

## Task 8: CLI 与公开合成演示

**Files:**
- Create: `src/riskprobe/cli.py`
- Create: `configs/synthetic.example.yaml`
- Create: `tests/test_cli.py`

**Interfaces:**
- Produces CLI: `riskprobe synthetic`, `riskprobe inspect`, `riskprobe discover`, `riskprobe run`
- Consumes: Task 3 generator，Task 7 service

- [ ] **Step 1: 编写 CLI 集成测试**

```python
# tests/test_cli.py
from typer.testing import CliRunner

from riskprobe.cli import app

runner = CliRunner()


def test_synthetic_then_run(tmp_path) -> None:
    data_path = tmp_path / "demo.parquet"
    result = runner.invoke(
        app,
        ["synthetic", "--output", str(data_path), "--rows", "5000", "--seed", "42"],
    )
    assert result.exit_code == 0
    assert data_path.exists()
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`

Expected: FAIL，CLI 尚不存在。

- [ ] **Step 3: 实现 Typer 命令**

命令签名固定为：

```text
riskprobe synthetic --output PATH --rows INTEGER --seed INTEGER
riskprobe inspect --config PATH --runs-dir PATH
riskprobe discover --config PATH --runs-dir PATH
riskprobe run --config PATH --runs-dir PATH
```

所有命令失败时输出结构化错误类别和可行动消息，退出码 2；不得打印样本行。`synthetic` 输出行数、列数和真值规则 ID，不打印用户级数据。

- [ ] **Step 4: 创建公开配置示例**

`configs/synthetic.example.yaml` 使用 `data/synthetic/behavior.parquet`、`entity_id`、`snapshot_date`、`institution`、`target` 和四个公开特征前缀；`performance_window_days: null`，因此演示 B 级限制。

- [ ] **Step 5: 运行全部核心验证和真实 CLI 冒烟**

Run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
.venv/bin/riskprobe synthetic --output /tmp/riskprobe-demo.parquet --rows 20000 --seed 42
cp configs/synthetic.example.yaml /tmp/riskprobe-demo.yaml
.venv/bin/python -c 'from pathlib import Path; p=Path("/tmp/riskprobe-demo.yaml"); p.write_text(p.read_text().replace("data/synthetic/behavior.parquet", "/tmp/riskprobe-demo.parquet"), encoding="utf-8")'
.venv/bin/riskprobe run --config /tmp/riskprobe-demo.yaml --runs-dir /tmp/riskprobe-runs
```

Expected: 测试和 Ruff 通过，CLI 输出 run ID，运行目录包含规定产物，报告明确标注 metadata grade B。

- [ ] **Step 6: 检查 Git 不跟踪演示数据**

Run:

```bash
git status --short
git check-ignore -v data/synthetic/behavior.parquet runs/demo/manifest.json
```

Expected: `.parquet` 和 `runs/` 命中 `.gitignore`，没有数据文件进入待提交列表。

- [ ] **Step 7: 用户授权后提交检查点**

```bash
git add configs/synthetic.example.yaml src/riskprobe/cli.py tests/test_cli.py
git commit -m "feat: add riskprobe command line workflow"
```

## Plan 1 Completion Gate

Run:

```bash
.venv/bin/python -m pytest --cov=riskprobe --cov-report=term-missing
.venv/bin/ruff check src tests
.venv/bin/riskprobe --help
```

验收条件：

- 所有验证通过；
- 合成数据同种子完全一致；
- Train 以外数据未参与候选阈值生成；
- 每条规则都有证据卡和 BH 调整后 p-value；
- 报告不包含用户级明细，不使用“严格 OOT”措辞；
- CLI 无 Kiro、无外部模型 API 也能完整运行；
- Git 状态中不出现 Parquet、运行目录或真实公司配置。
