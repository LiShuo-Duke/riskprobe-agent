# RiskProbe Public and Company Data Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 RiskProbe 接入 Home Credit 公开多表信用行为数据和公司本地 Parquet 宽表，完成公开演示、公司预检、真实任务计时、证据留存与基于实测数据的简历表述。

**Architecture:** 公开适配器将用户自行下载的 Home Credit CSV 在本地聚合为统一 Parquet；公司适配器不复制数据，只验证本地宽表字段、特征族和批处理计划。两个适配器都输出 Plan 1 的统一契约；公司试运行产物写入 Git 忽略目录，简历生成器只读取完整且可审计的测量记录。

**Tech Stack:** Plan 1–2 全部依赖；Polars LazyFrame 完成多表聚合，不使用 SQL；Python `time.perf_counter` 记录耗时。

## Global Constraints

- 必须先完成 Plan 1；Agent 演示和公司白名单工具还要求 Plan 2 完成。
- Home Credit 没有可用于严格时间切分的绝对申请时间，公开真实数据只做 Train/Test 和跨客群验证；时间漂移评估使用合成行为数据。
- Home Credit 原始 CSV 和生成 Parquet 不进入 Git，不在仓库中再分发。
- 公司真实字段映射、Parquet、机构编码、规则阈值、运行产物和报告不进入 Git。
- 公司数据只读；不实现 SQL、数据拉取、文件覆盖或样本级导出。
- 公开报告使用“customer segment”；公司报告使用“institution”，由配置中的 `segment_display_name` 控制。
- 没有实测数据时，简历生成器必须失败，不得用默认值或示例数字生成表述。
- 每个任务包含提交命令，但只有用户明确授权提交后才能执行。

---

## File Map

```text
.gitignore                                 # 允许 company.example.yaml，继续忽略 company.local.yaml
src/riskprobe/adapters/home_credit.py      # Home Credit 多表本地聚合
src/riskprobe/adapters/company.py          # 公司 Parquet 契约和特征族预检
src/riskprobe/batching.py                  # 宽表列批次计划
src/riskprobe/benchmarking.py              # 真实任务阶段计时和测量记录
src/riskprobe/resume_evidence.py           # 只基于完整实测记录生成表述
src/riskprobe/cli.py                       # prepare-home-credit、preflight-company、benchmark、resume-evidence
configs/home_credit.example.yaml           # 公开数据配置
configs/company.example.yaml               # 虚构字段的公司配置模板
schemas/benchmark-record.schema.json       # 测量记录约束
examples/company_schema.json               # 虚构宽表 schema，不含公司资产
docs/company-runbook.md                    # 公司环境实际运行步骤
```

## Task 1: Home Credit 本地多表聚合

**Files:**
- Create: `src/riskprobe/adapters/__init__.py`
- Create: `src/riskprobe/adapters/home_credit.py`
- Create: `tests/adapters/test_home_credit.py`

**Interfaces:**
- Produces: `HomeCreditPaths.from_directory(path) -> HomeCreditPaths`
- Produces: `prepare_home_credit(paths, output_path, seed=42) -> HomeCreditPreparationResult`
- Output columns: `entity_id`, `target`, `customer_segment`, behavior features
- Consumes: Plan 1 Parquet contract

- [ ] **Step 1: 编写最小多表聚合测试**

```python
# tests/adapters/test_home_credit.py
import polars as pl

from riskprobe.adapters.home_credit import HomeCreditPaths, prepare_home_credit


def test_prepare_home_credit_aggregates_history_without_post_target_fields(tmp_path) -> None:
    pl.DataFrame({
        "SK_ID_CURR": [1, 2],
        "TARGET": [1, 0],
        "NAME_INCOME_TYPE": ["Working", "Pensioner"],
        "DAYS_BIRTH": [-12000, -20000],
        "AMT_INCOME_TOTAL": [100000.0, 80000.0],
    }).write_csv(tmp_path / "application_train.csv")
    pl.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "DAYS_DECISION": [-10, -100, -20],
        "AMT_APPLICATION": [1000.0, 2000.0, 500.0],
        "AMT_CREDIT": [900.0, 1800.0, 500.0],
        "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved"],
    }).write_csv(tmp_path / "previous_application.csv")
    output = tmp_path / "home_credit.parquet"

    result = prepare_home_credit(HomeCreditPaths.from_directory(tmp_path), output)
    frame = pl.read_parquet(output)

    assert result.rows == 2
    assert "prev_application_cnt_30d" in frame.columns
    assert "prev_refused_rate_all" in frame.columns
    assert "TARGET" not in frame.columns
    assert frame.columns[:3] == ["entity_id", "target", "customer_segment"]
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/python -m pytest tests/adapters/test_home_credit.py -v`

Expected: FAIL，公开适配器不存在。

- [ ] **Step 3: 实现文件发现和列白名单**

`application_train.csv` 必需；`previous_application.csv`、`installments_payments.csv`、`POS_CASH_balance.csv`、`credit_card_balance.csv`、`bureau.csv` 可选，但至少存在一个历史表。只读取以下字段：

```text
application_train: SK_ID_CURR, TARGET, NAME_INCOME_TYPE, DAYS_BIRTH, AMT_INCOME_TOTAL
previous_application: SK_ID_CURR, DAYS_DECISION, AMT_APPLICATION, AMT_CREDIT, NAME_CONTRACT_STATUS
installments_payments: SK_ID_CURR, DAYS_INSTALMENT, DAYS_ENTRY_PAYMENT, AMT_INSTALMENT, AMT_PAYMENT
POS_CASH_balance: SK_ID_CURR, MONTHS_BALANCE, SK_DPD, SK_DPD_DEF
credit_card_balance: SK_ID_CURR, MONTHS_BALANCE, AMT_BALANCE, AMT_PAYMENT_CURRENT, SK_DPD
bureau: SK_ID_CURR, DAYS_CREDIT, AMT_CREDIT_SUM, AMT_CREDIT_SUM_DEBT, CREDIT_ACTIVE
```

适配器不得读取 `application_test.csv` 或结果型预测文件。

- [ ] **Step 4: 实现行为聚合**

按 `SK_ID_CURR` 生成：

- previous application 近 30/90/365 天次数、总金额、拒绝率；
- installment 近 30/90/365 天记录数、逾期天数均值、少还比例；
- POS 近 3/6/12 月活跃月数、DPD 均值和最大值；
- credit card 近 3/6/12 月余额均值、支付余额比、DPD 最大值；
- bureau 近 90/365 天查询数、活跃信用数、债务授信比。

Home Credit 的相对时间字段只用于历史窗口聚合，不构造虚假的绝对申请日期。输出增加固定 `snapshot_date="public_relative_reference"`；公开配置设置 `snapshot.meaning: public_relative_reference` 和 `time_validation_enabled: false`，服务不解析该列为日期，也不生成时间切片证据。

- [ ] **Step 5: 运行适配器测试和惰性扫描检查**

Run:

```bash
.venv/bin/python -m pytest tests/adapters/test_home_credit.py -v
.venv/bin/ruff check src tests
```

Expected: PASS，输出不含原始 `TARGET` 名称和未白名单字段。

- [ ] **Step 6: 用户授权后提交检查点**

```bash
git add src/riskprobe/adapters tests/adapters/test_home_credit.py
git commit -m "feat: prepare home credit behavior features"
```

## Task 2: Home Credit CLI、公开配置与真实数据验证

**Files:**
- Modify: `src/riskprobe/cli.py`
- Create: `configs/home_credit.example.yaml`
- Create: `tests/test_home_credit_cli.py`
- Create: `docs/home-credit-runbook.md`

**Interfaces:**
- Produces CLI: `riskprobe prepare-home-credit --input-dir PATH --output PATH`
- Produces: public config with `time_validation_enabled: false`
- Consumes: Task 1 adapter，Plan 1 service

- [ ] **Step 1: 编写 CLI 测试**

```python
# tests/test_home_credit_cli.py
from typer.testing import CliRunner
from riskprobe.cli import app

runner = CliRunner()


def test_prepare_home_credit_rejects_missing_application_table(tmp_path) -> None:
    result = runner.invoke(
        app,
        ["prepare-home-credit", "--input-dir", str(tmp_path), "--output", str(tmp_path / "x.parquet")],
    )
    assert result.exit_code == 2
    assert "application_train.csv" in result.stdout
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/python -m pytest tests/test_home_credit_cli.py -v`

Expected: FAIL，命令不存在。

- [ ] **Step 3: 实现 CLI 和配置**

`prepare-home-credit` 只输出表数、行列数、生成特征族和目标路径，不打印样本。`configs/home_credit.example.yaml` 使用：

```yaml
dataset:
  id: home_credit_public
  path: data/public/home_credit.parquet
columns:
  entity: entity_id
  snapshot: snapshot_date
  segment: customer_segment
  target: target
target:
  positive_value: 1
  positive_meaning: bad_debt
  performance_window_days: null
snapshot:
  meaning: public_relative_reference
features:
  families:
    previous_application: [prev_]
    installment: [inst_]
    pos: [pos_]
    credit_card: [cc_]
    bureau: [bureau_]
segment_display_name: customer_segment
time_validation_enabled: false
```

该模式使用固定种子的分层 Train/Test 切分和跨客群验证，不创建 OOT 或时间切片证据。

- [ ] **Step 4: 编写本地运行说明**

`docs/home-credit-runbook.md` 明确：用户自行从数据提供方下载；数据不随仓库分发；将 CSV 放在仓库外目录；执行 `prepare-home-credit`；生成 Parquet 受 `.gitignore` 保护；公开报告只能称 Train/Test 和跨客群验证。

- [ ] **Step 5: 在最小 fixture 上运行回归**

Run:

```bash
.venv/bin/python -m pytest tests/test_home_credit_cli.py tests/adapters/test_home_credit.py -v
.venv/bin/ruff check src tests
```

Expected: PASS。

- [ ] **Step 6: 获得真实 Home Credit 路径后运行公开实验**

Run:

```bash
.venv/bin/riskprobe prepare-home-credit --input-dir "/absolute/path/to/home-credit" --output "/tmp/home-credit-riskprobe.parquet"
cp configs/home_credit.example.yaml /tmp/home-credit-riskprobe.yaml
.venv/bin/python -c 'from pathlib import Path; p=Path("/tmp/home-credit-riskprobe.yaml"); p.write_text(p.read_text().replace("data/public/home_credit.parquet", "/tmp/home-credit-riskprobe.parquet"), encoding="utf-8")'
.venv/bin/riskprobe run --config /tmp/home-credit-riskprobe.yaml --runs-dir /tmp/riskprobe-home-credit-runs
```

Expected: 生成公开数据 Profile、候选规则、证据卡和报告；报告中不出现 OOT、跨机构或线上可用声明。

- [ ] **Step 7: 用户授权后提交检查点**

```bash
git add src/riskprobe/cli.py configs/home_credit.example.yaml tests/test_home_credit_cli.py docs/home-credit-runbook.md
git commit -m "feat: add home credit public workflow"
```

## Task 3: 公司 Parquet 预检和特征批次计划

**Files:**
- Modify: `.gitignore`
- Create: `src/riskprobe/batching.py`
- Create: `src/riskprobe/adapters/company.py`
- Create: `configs/company.example.yaml`
- Create: `examples/company_schema.json`
- Create: `tests/adapters/test_company.py`
- Create: `tests/test_batching.py`

**Interfaces:**
- Produces: `preflight_company_dataset(config) -> CompanyPreflight`
- Produces: `plan_feature_batches(schema, catalog, batch_size=64) -> tuple[FeatureBatch, ...]`
- Consumes: Plan 1 config/parquet/catalog

- [ ] **Step 1: 允许仅跟踪虚构公司配置示例**

在 `.gitignore` 的公司配置规则后加入：

```gitignore
!/configs/company.example.yaml
```

真实配置继续命名为 `configs/company.local.yaml`，并被 `/configs/company*.yaml` 忽略；执行 `git check-ignore -v configs/company.local.yaml` 必须命中忽略规则，`git check-ignore configs/company.example.yaml` 必须返回非零状态。

- [ ] **Step 2: 编写 977 维批次和只读预检测试**

```python
# tests/test_batching.py
from riskprobe.batching import plan_feature_batches


def test_977_features_are_split_without_role_columns() -> None:
    features = [f"order_f_{i:04d}" for i in range(977)]
    batches = plan_feature_batches(features, batch_size=64)
    flattened = [name for batch in batches for name in batch.features]
    assert len(batches) == 16
    assert flattened == features
```

```python
# tests/adapters/test_company.py
from datetime import date
import polars as pl

from riskprobe.adapters.company import preflight_company_dataset
from riskprobe.config import ProjectConfig


def test_preflight_does_not_modify_source(tmp_path) -> None:
    path = tmp_path / "company.parquet"
    pl.DataFrame({
        "anonymous_id": [f"u{i}" for i in range(200)],
        "cutoff_date": [date(2026, 1, 1)] * 200,
        "org_code": ["A"] * 100 + ["B"] * 100,
        "bad_label": [0, 1] * 100,
        "ord_x_cnt_30d": list(range(200)),
        "brw_x_pv_30d": list(range(200, 400)),
    }).write_parquet(path)
    config = ProjectConfig.model_validate({
        "dataset": {"id": "company_test", "path": path},
        "columns": {
            "entity": "anonymous_id",
            "snapshot": "cutoff_date",
            "segment": "org_code",
            "target": "bad_label",
        },
        "target": {"positive_value": 1, "positive_meaning": "bad_debt"},
        "snapshot": {"meaning": "customer_specified_feature_cutoff"},
        "features": {"families": {"order": ["ord_x_"], "browse": ["brw_x_"]}},
        "segment_display_name": "institution",
    })
    before = path.stat().st_mtime_ns
    result = preflight_company_dataset(config)
    after = path.stat().st_mtime_ns
    assert before == after
    assert result.feature_family_counts["order"] > 0
    assert result.metadata_grade == "B"
```

- [ ] **Step 3: 运行测试并确认失败**

Run: `.venv/bin/python -m pytest tests/test_batching.py tests/adapters/test_company.py -v`

Expected: FAIL，批次和公司适配器不存在。

- [ ] **Step 4: 实现批处理和预检**

`FeatureBatch` 包含 `index`, `features`, `required_columns`；每批 required columns 固定附加 entity、snapshot、segment、target，但 `features` 不含角色列。`preflight_company_dataset` 使用 `scan_parquet().collect_schema()` 检查字段和类型，再仅扫描角色列计算行数、标签率、日期范围和 segment 计数；不读取全部 977 维。

- [ ] **Step 5: 创建完全虚构的公司示例**

`configs/company.example.yaml` 使用 `/private/company.local.parquet`、`anonymous_id`、`cutoff_date`、`org_code`、`bad_label`，特征前缀使用 `ord_x_`, `brw_x_`, `multi_x_`, `emb_x_`。`examples/company_schema.json` 只列 12 个虚构字段及类型，不含真实机构、阈值或结果。

- [ ] **Step 6: 运行测试和 Git 隔离验证**

Run:

```bash
.venv/bin/python -m pytest tests/test_batching.py tests/adapters/test_company.py -v
git check-ignore -v configs/company.local.yaml sample.parquet runs/demo/result.json
git status --short
```

Expected: 测试通过；三类私有文件均被忽略；`company.example.yaml` 可跟踪。

- [ ] **Step 7: 用户授权后提交检查点**

```bash
git add .gitignore src/riskprobe/batching.py src/riskprobe/adapters/company.py configs/company.example.yaml examples/company_schema.json tests/test_batching.py tests/adapters/test_company.py
git commit -m "feat: add private parquet preflight"
```

## Task 4: 公司试运行计时与结构化证据

**Files:**
- Create: `src/riskprobe/benchmarking.py`
- Create: `schemas/benchmark-record.schema.json`
- Modify: `src/riskprobe/cli.py`
- Create: `tests/test_benchmarking.py`

**Interfaces:**
- Produces: `BenchmarkRecord`, `StageTiming`, `RuleReviewSummary`, `AnomalyEvaluationSummary`
- Produces CLI: `riskprobe benchmark --config PATH --runs-dir PATH --baseline-record PATH`
- Consumes: Plan 1 service，Plan 2 drift evaluation

- [ ] **Step 1: 编写无人工基线拒绝生成提效测试**

```python
# tests/test_benchmarking.py
import pytest

from riskprobe.benchmarking import BenchmarkRecord, StageTiming, calculate_efficiency


def test_efficiency_requires_measured_manual_baseline() -> None:
    record = BenchmarkRecord(
        run_id="run001",
        task_id="joint-validation-001",
        dataset_id="company_current",
        measured_at="2026-08-05T10:00:00Z",
        code_version="0.1.0",
        config_hash="cfg123",
        data_fingerprint="data123",
        manual_minutes=None,
        agent_minutes=32.0,
        stage_timings=(StageTiming(stage="inspect", seconds=12.0),),
        candidate_rule_count=30,
        evidence_passed_count=15,
        reviewed_rule_count=15,
        accepted_rule_count=6,
        anomaly_true_positive_count=9,
        anomaly_false_positive_count=2,
        anomaly_false_negative_count=1,
        root_cause_top3_hit_count=7,
        root_cause_case_count=10,
    )
    with pytest.raises(ValueError, match="manual baseline"):
        calculate_efficiency(record)
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/python -m pytest tests/test_benchmarking.py -v`

Expected: FAIL，计时模型不存在。

- [ ] **Step 3: 实现测量记录**

`BenchmarkRecord` 必须记录：run ID、task ID、dataset ID、时间、代码版本、配置哈希、数据指纹、人工分钟数、Agent 分钟数、各阶段耗时、候选规则数、证据门控通过数、人工复核数、人工接受数、异常 TP/FP/FN 数、根因 Top-3 命中数和根因案例数。Precision、Recall、false-positive-rate 与 Top-3 hit rate 均由这些计数派生，不允许只存舍入后的比率。所有数字必须非负；接受数不得超过复核数；没有人工基线时不计算提效。

- [ ] **Step 4: 实现 benchmark CLI**

CLI 读取人工事先填写的本地 `baseline-record`，用 `time.perf_counter` 测量 inspect、discover、validate、monitor、report 阶段，输出到 `runs/<run_id>/benchmark_record.json`。CLI 不修改 baseline 文件，不向终端打印公司规则表达式。

- [ ] **Step 5: 运行测试**

Run: `.venv/bin/python -m pytest tests/test_benchmarking.py -v`

Expected: PASS。

- [ ] **Step 6: 用户授权后提交检查点**

```bash
git add src/riskprobe/benchmarking.py src/riskprobe/cli.py schemas/benchmark-record.schema.json tests/test_benchmarking.py
git commit -m "feat: capture measured workflow evidence"
```

## Task 5: 只基于实测记录生成简历证据

**Files:**
- Create: `src/riskprobe/resume_evidence.py`
- Create: `tests/test_resume_evidence.py`
- Modify: `src/riskprobe/cli.py`

**Interfaces:**
- Produces: `aggregate_benchmarks(records: list[BenchmarkRecord]) -> ResumeEvidence`
- Produces: `render_resume_bullets(evidence: ResumeEvidence) -> ResumeDraft`
- Produces CLI: `riskprobe resume-evidence --records-dir PATH --output PATH`
- Consumes: Task 4 records

- [ ] **Step 1: 编写少于三次任务拒绝量化表述测试**

```python
# tests/test_resume_evidence.py
import pytest

from riskprobe.benchmarking import BenchmarkRecord, StageTiming
from riskprobe.resume_evidence import aggregate_benchmarks


def make_record(task_id: str, manual: float = 60.0, agent: float = 30.0) -> BenchmarkRecord:
    return BenchmarkRecord(
        run_id=f"run-{task_id}",
        task_id=task_id,
        dataset_id="company_current",
        measured_at="2026-08-05T10:00:00Z",
        code_version="0.1.0",
        config_hash="cfg123",
        data_fingerprint=f"data-{task_id}",
        manual_minutes=manual,
        agent_minutes=agent,
        stage_timings=(StageTiming(stage="inspect", seconds=12.0),),
        candidate_rule_count=30,
        evidence_passed_count=15,
        reviewed_rule_count=10,
        accepted_rule_count=4,
        anomaly_true_positive_count=9,
        anomaly_false_positive_count=2,
        anomaly_false_negative_count=1,
        root_cause_top3_hit_count=7,
        root_cause_case_count=10,
    )


def test_resume_evidence_requires_three_completed_tasks() -> None:
    with pytest.raises(ValueError, match="at least 3 completed tasks"):
        aggregate_benchmarks([make_record("001"), make_record("002")])
```

- [ ] **Step 2: 编写聚合公式测试**

```python

def test_resume_metrics_use_recorded_values() -> None:
    records = [make_record("001"), make_record("002"), make_record("003")]
    evidence = aggregate_benchmarks(records)
    assert evidence.task_count == 3
    assert evidence.total_reviewed_rules == 30
    assert evidence.total_accepted_rules == 12
    assert evidence.efficiency_rate == 0.5
    assert evidence.anomaly_recall == 27 / 30
    assert evidence.root_cause_top3_hit_rate == 21 / 30
```

- [ ] **Step 3: 运行测试并确认失败**

Run: `.venv/bin/python -m pytest tests/test_resume_evidence.py -v`

Expected: FAIL，简历证据模块不存在。

- [ ] **Step 4: 实现审计型聚合**

至少需要三条不同 task ID、均含人工基线和 Agent 耗时的记录。提效使用总分钟加权公式：

```text
(sum(manual_minutes) - sum(agent_minutes)) / sum(manual_minutes)
```

规则复核通过率为总接受数/总复核数；异常指标按各次真值数量加权，不做简单平均。`ResumeEvidence` 保存来源 run ID 列表，便于面试前回查。

- [ ] **Step 5: 实现两段简历草稿输出**

输出一段公开项目和一段实习经历。公司段只能引用 `ResumeEvidence` 中存在的 task_count、efficiency、reviewed/accepted rules、recall 和 Top-3 hit；缺少任一指标时省略对应句，不填 0、不使用估算。文件写入用户指定的内部输出路径，推荐 `reports/internal/resume_evidence.md`，该路径已被 Git 忽略。

- [ ] **Step 6: 运行测试和 Git 隔离检查**

Run:

```bash
.venv/bin/python -m pytest tests/test_resume_evidence.py -v
git check-ignore -v reports/internal/resume_evidence.md
```

Expected: 测试通过，简历内部证据文件命中 `.gitignore`。

- [ ] **Step 7: 用户授权后提交检查点**

```bash
git add src/riskprobe/resume_evidence.py src/riskprobe/cli.py tests/test_resume_evidence.py
git commit -m "feat: generate resume claims from measured evidence"
```

## Task 6: 公司环境三至五次真实试运行

**Files:**
- Create: `docs/company-runbook.md`
- Runtime only, ignored: `configs/company.local.yaml`
- Runtime only, ignored: `runs/<run_id>/benchmark_record.json`
- Runtime only, ignored: `reports/internal/resume_evidence.md`

**Interfaces:**
- Produces: 至少三条 `BenchmarkRecord` 和最终 `ResumeEvidence`
- Consumes: Tasks 3–5，用户在公司环境提供的本地 Parquet 路径

- [ ] **Step 1: 编写公司运行手册**

手册规定：复制 `company.example.yaml` 为 `company.local.yaml`；只填写本地路径和脱敏字段角色；运行 `git check-ignore` 确认配置与数据被忽略；记录同类历史人工任务耗时；执行 preflight；确认 metadata grade 和表现窗口限制；再运行 benchmark；报告仅供人工复核。

- [ ] **Step 2: 在每个真实任务前执行泄漏防护检查**

Run:

```bash
git check-ignore -v configs/company.local.yaml "/absolute/path/to/company/modeling_sample.parquet"
git status --short
```

若公司 Parquet 在仓库外，第一条对绝对路径可能无匹配，这是允许的；关键是它不位于仓库。Expected: company.local.yaml 被忽略，Git 状态不显示任何数据、结果或内部报告。

- [ ] **Step 3: 执行预检**

Run:

```bash
.venv/bin/riskprobe preflight-company --config configs/company.local.yaml
```

Expected: 输出行数、特征数、特征族计数、批次数、标签率、segment 数和 metadata grade；不输出 entity、样本行、真实机构名称或规则。

- [ ] **Step 4: 对三至五个任务执行 benchmark**

Run for each task:

```bash
.venv/bin/riskprobe benchmark --config configs/company.local.yaml --runs-dir runs --baseline-record "/private/path/to/baseline-record.json"
```

Expected: 每个 task ID 生成独立 benchmark record，包含阶段耗时、代码版本、配置哈希和数据指纹。

- [ ] **Step 5: 人工复核规则并回填测量记录**

复核只记录候选数、复核数、接受数和拒绝原因分类，不把规则表达式写入个人 Git。修改后的记录必须通过 `benchmark-record.schema.json` 校验。

- [ ] **Step 6: 生成内部简历证据**

Run:

```bash
.venv/bin/riskprobe resume-evidence --records-dir runs --output reports/internal/resume_evidence.md
```

Expected: 至少三次完整任务后生成两段草稿和来源 run ID；不足三次或缺人工基线时退出码 2 并说明缺失项。

- [ ] **Step 7: 最终安全检查**

Run:

```bash
git status --short
git ls-files "*.parquet" "*.csv" "configs/company.local.yaml" "runs/*" "reports/internal/*"
```

Expected: 第二条命令无输出；Git 状态仅含通用代码、公开示例和文档。

- [ ] **Step 8: 用户授权后提交运行手册，不提交任何运行产物**

```bash
git add docs/company-runbook.md
git commit -m "docs: add private company validation runbook"
```

## Plan 3 Completion Gate

Run:

```bash
.venv/bin/python -m pytest --cov=riskprobe --cov-report=term-missing
.venv/bin/python -m pytest tests/adapters/test_company.py tests/test_batching.py -v
.venv/bin/ruff check src tests

git check-ignore -v configs/company.local.yaml sample.parquet runs/demo/result.json reports/internal/resume_evidence.md
git ls-files "*.parquet" "*.csv" "configs/company.local.yaml" "runs/*" "reports/internal/*"
```

验收条件：

- Home Credit 多表在本地聚合为统一 Parquet，公开报告不伪造时间验证；
- 公司适配器仅读 schema 和必要列，原文件 mtime 不变；
- 977 维特征按 64 列形成 16 个无重复批次；
- 公开和公司分组显示分别为 customer segment 与 institution；
- 至少三次真实公司任务完成前，简历生成器拒绝量化输出；
- 公司数据、真实配置、运行结果和内部简历证据均不在 Git 索引中；
- 最终简历数字可追溯到具体 run ID 和测量记录。
