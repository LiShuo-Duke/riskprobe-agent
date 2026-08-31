# RiskProbe Complete Terminal Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not create commits unless the user explicitly requests one.

**Goal:** 每次可验证 RiskProbe 运行自动生成覆盖全部分析与 Host 决策阶段的中文 Markdown 和 Word 正式报告，并保持现有 MCP schema、error code、隐私、幂等和不可变 run 契约。

**Architecture:** 先把安全聚合事实投影为唯一的私有 `ReportModel`，由 Markdown 与 DOCX renderer 分别消费；现有 run 在 finalize 前继续生成增强版 `risk_report.md`。Host 终态持久化后，MCP 返回前通过独立 terminal report store 原子发布 `final_risk_report.md`、`final_risk_report.docx` 和 manifest；重放只校验或修复报告，不重新运行 Agent 工具。

**Tech Stack:** Python 3.11、Pydantic 2.13.4、python-docx 1.2.0、标准库 `zipfile/hashlib/json/tempfile/fcntl`、现有 RiskProbe artifact/privacy/Host/MCP 层。

**Design:** `docs/superpowers/specs/2026-08-24-riskprobe-terminal-report-design.md`

## Global Constraints

- MCP 仍只暴露 `riskprobe_get_decision_context` 与 `riskprobe_submit_decision_proposal`，请求和响应字段不变。
- 不新增公开 error code；报告发布失败复用 `agent_orchestration_failed`。
- 不读取 `candidate_rules.parquet`、原始数据、SQLite、日志或 traceback；只读已验证安全聚合 artifacts 和 terminal DTO。
- 不输出真实路径、实体值、真实机构/分类值、样本行、文件 bytes 或 hash。
- run artifact 集、manifest 和已 finalize 目录不变；终态报告只写入 `state_dir` 私有 sidecar。
- Markdown 与 DOCX 始终同时生成；同一幂等 proposal 复用同一 report ID 和相同内容 hash。
- TOP10 固定按 `Stable → Local → Unstable → Suspicious`、Test Lift 降序、Rule ID 升序排序。
- 规则数值阈值可展示；字符串条件必须稳定 token 化；缺失 Holdout/时间验证显示 `N/A`，不得当作 0。
- 精确固定 `python-docx==1.2.0`，不使用 Pandoc、外部服务或额外模板依赖。
- Workspace 禁止 Shell、任意 Python/pytest、SQL 和网络；实现后只做源码级核对、一次语义审查和真实 MCP 协议验收。
- 只保留一个小型 runnable check 文件，不引入 fixtures 或测试框架抽象。
- 不创建 Git commit，除非用户另行明确要求。

---

### Task 1: 建立唯一安全报告模型并增强分析 Markdown

**Files:**
- Create: `src/riskprobe/report_models.py`
- Modify: `src/riskprobe/reporting.py`
- Modify: `src/riskprobe/service.py:920-943, 2690-2800`
- Create: `tests/test_terminal_reporting.py`

**Interfaces:**
- Consumes: `DatasetProfile`、`EvidenceCard`、`AnalysisSummary`、可选机构聚合和 run limitations。
- Produces:
  - `build_analysis_report_model(...) -> ReportModel`
  - `build_final_report_model(...) -> ReportModel`
  - `render_report_markdown(model: ReportModel) -> str`
  - 兼容入口 `render_risk_report(...) -> str` 保留。

- [ ] **Step 1: 写一个无 fixture 的最小 runnable check**

在 `tests/test_terminal_reporting.py` 使用现有严格模型直接构造 11 张卡片，检查 TOP10、稳定排序、条件 token 和 `N/A` 语义。测试文件只覆盖本功能最容易回退的纯逻辑：

```python
from riskprobe.models import Condition, EvidenceCard, RiskRule, RuleMetrics
from riskprobe.report_models import build_analysis_report_model


def _metrics(lift: float) -> RuleMetrics:
    return RuleMetrics(
        support_count=20,
        coverage=0.2,
        base_bad_rate=0.1,
        hit_bad_rate=0.3,
        non_hit_bad_rate=0.05,
        lift=lift,
        precision=0.3,
        recall=0.6,
        p_value=0.01,
        hit_good_rate=0.1,
        ks_signed=0.2,
        ks_stat=0.2,
    )


def _card(index: int, *, grade: str, lift: float) -> EvidenceCard:
    return EvidenceCard(
        rule=RiskRule(
            rule_id=f"rule-{index:02d}",
            origin="single",
            conditions=(Condition(feature="amount", operator=">", value=float(index)),),
        ),
        train=_metrics(lift - 0.1),
        test=_metrics(lift),
        slices=(),
        lift_ci=(lift - 0.2, lift + 0.2),
        adjusted_p_value=0.02,
        segment_consistency=1.0,
        max_time_decay=None,
        grade=grade,
        limitations=(),
    )


def test_report_model_limits_and_orders_top_rules() -> None:
    cards = tuple(
        _card(index, grade="Stable" if index < 2 else "Unstable", lift=float(index))
        for index in range(11)
    )
    model = build_analysis_report_model(evidence_cards=cards)
    assert len(model.top_rules) == 10
    assert [rule.rule_id for rule in model.top_rules[:2]] == ["rule-01", "rule-00"]
    assert model.top_rules[0].holdout is None
```

再加一个字符串条件：`Condition(feature="merchant_type", operator="==", value="private-category")`。断言 model dump 不含 `private-category`，但两次构建得到相同 token。

- [ ] **Step 2: 新建 canonical report DTO**

在 `report_models.py` 复用 `FrozenModel`、`FeatureRef`、`StageSummary` 和现有 safe validators，定义包内 DTO：

```python
class ReportCondition(FrozenModel):
    feature: FeatureRef
    operator: Operator
    numeric_value: int | float | None = None
    value_token: str | None = None

    @model_validator(mode="after")
    def validate_value_shape(self) -> "ReportCondition":
        has_number = self.numeric_value is not None
        has_token = self.value_token is not None
        if self.operator == "is_null":
            if has_number or has_token:
                raise ValueError("is_null must not carry a value")
            return self
        if has_number == has_token:
            raise ValueError("condition must carry exactly one safe value")
        return self


class ReportRuleMetrics(FrozenModel):
    split: Literal["Train", "Test", "Holdout"]
    support_count: int
    coverage: float
    hit_bad_rate: float
    lift: float
    precision: float
    recall: float
    signed_ks: float | None = None


class ReportRule(FrozenModel):
    rank: int
    rule_id: str
    origin: str
    conditions: tuple[ReportCondition, ...]
    grade: EvidenceGrade
    train: ReportRuleMetrics
    test: ReportRuleMetrics
    holdout: ReportRuleMetrics | None = None
    lift_ci: tuple[float, float]
    adjusted_p_value: float
    segment_consistency: float
    time_decay: float | None = None
    limitations: tuple[str, ...] = ()
```

`ReportModel` 保存 `analysis_summary`、TOP10、全量规则安全简表、可选 diagnostics/decision/review 投影、总体 limitations 和固定 schema version。模型 validator 对 `model_dump(mode="json")` 调用 `assert_safe_payload`。

- [ ] **Step 3: 实现唯一规则投影和 TOP10**

实现以下固定行为：

```python
def _top_cards(cards: Sequence[EvidenceCard]) -> tuple[EvidenceCard, ...]:
    return tuple(sorted(cards, key=evidence_sort_key)[:10])


def _project_condition(condition: Condition) -> ReportCondition:
    feature = FeatureRef.from_name(condition.feature)
    if condition.operator == "is_null":
        if condition.value is not None:
            raise ValueError("is_null condition contains a value")
        return ReportCondition(feature=feature, operator=condition.operator)
    if isinstance(condition.value, bool) or condition.value is None:
        raise ValueError("condition contains an unsupported value")
    if isinstance(condition.value, (int, float)):
        return ReportCondition(
            feature=feature,
            operator=condition.operator,
            numeric_value=condition.value,
        )
    token = stable_token(
        condition.value,
        namespace=f"rule-condition:{feature.value}",
    )
    return ReportCondition(
        feature=feature,
        operator=condition.operator,
        value_token=token,
    )
```

Holdout 只从 `slice_type="dataset"` 且 `slice_value="Holdout"` 的 slice 投影；没有则保持 `None`。时间验证未应用时保持 `time_decay=None`。全量 grade counts 基于所有 cards，不能基于截断后的 TOP10。

- [ ] **Step 4: 让 Markdown renderer 只消费 `ReportModel`**

在 `reporting.py` 新增 `render_report_markdown(model)`，按设计中的十章顺序输出中文正文。规则主表必须包含 Rank、Rule ID、规则条件、Origin、Grade、Support、Test Lift、Holdout Lift、Coverage、Hit Bad Rate、signed KS、Adjusted p-value、Lift CI、Segment Consistency、Time Decay。

规则条件使用：

```python
def _condition_text(condition: ReportCondition) -> str:
    if condition.operator == "is_null":
        return f"{condition.feature.value} IS NULL"
    value = (
        repr(condition.numeric_value)
        if condition.numeric_value is not None
        else condition.value_token
    )
    return f"{condition.feature.value} {condition.operator} {value}"
```

用 `AND` 连接条件。Markdown 单元格必须转义 `|`、换行和反斜杠。所有 `None` 输出 `N/A`，四位小数仅用于指标展示，规则阈值用 round-trip `repr`，不能改变规则边界。

- [ ] **Step 5: 保留旧 renderer 兼容入口并消除原始条件泄露**

`render_risk_report(...)` 保持原签名，内部先调用 `build_analysis_report_model(...)` 再调用 `render_report_markdown(...)`。正式 service 路径不允许启用真实 segment/category 展示。删除 institution TOP5 对 `condition.get("value")` 的直接格式化，全部经过 `ReportCondition` 投影。

- [ ] **Step 6: 把现有 `AnalysisSummary` 传入分析报告**

调整 `service.report_action()`：先构造 `analysis_summary_payload`，再构造 `ReportModel` 并渲染；评分卡由 `AnalysisSummary.scorecard` 进入统一模型，不再由 `_render_scorecard_section()` 事后拼接。仍只写现有 `risk_report.md`、`metadata_report.json`、`analysis_summary.json`，不改变 run artifact 集或 checkpoint 引用。

- [ ] **Step 7: 源码级核对 Task 1**

用 `read_code`/`grep_search` 检查：只有 builder 执行 TOP10；全量 counts 未被截断；正式路径没有 raw condition value；现有 `render_risk_report` patch 点仍存在，避免破坏 checkpoint recovery 测试。

---

### Task 2: 生成正式且确定性的 DOCX

**Files:**
- Modify: `pyproject.toml`
- Create: `src/riskprobe/reporting_docx.py`
- Modify: `tests/test_terminal_reporting.py`

**Interfaces:**
- Consumes: Task 1 的 `ReportModel`。
- Produces:
  - `render_report_docx(model: ReportModel) -> bytes`
  - `_canonicalize_docx_package(content: bytes) -> bytes`

- [ ] **Step 1: 固定运行时依赖**

在 `[project].dependencies` 按字母顺序加入：

```toml
"python-docx==1.2.0",
```

不增加 Pandoc、HTML renderer、字体包或锁文件生成步骤。

- [ ] **Step 2: 实现 Word 文档结构**

`render_report_docx()` 创建 A4 文档，设置中文字体回退、标题层级、页眉页脚和页码。使用与 Markdown 相同的章节和 model 顺序；TOP10 宽表放入横向 section，设置 `Table Grid`、重复表头和固定列标题。

核心入口必须只接受 `ReportModel`：

```python
def render_report_docx(model: ReportModel) -> bytes:
    document = Document()
    _configure_document(document)
    _add_cover(document, model)
    _add_executive_summary(document, model)
    _add_stage_summary(document, model)
    _add_profile_partition(document, model)
    _add_discovery(document, model)
    _add_top_rules(document, model)
    _add_scorecard(document, model)
    _add_institution_stability(document, model)
    _add_decision_review(document, model)
    _add_limitations_appendix(document, model)
    buffer = BytesIO()
    document.save(buffer)
    return _canonicalize_docx_package(buffer.getvalue())
```

每个 helper 只读 model，不重新排序、token 化或计算指标。

- [ ] **Step 3: 清理 DOCX 元数据**

写入前把 core properties 固定为空字符串或运行绑定值：author、last_modified_by、keywords、comments 为空；created/modified 使用固定 UTC 时间 `1980-01-01T00:00:00Z`；title 固定为报告标题。正文不得加入当前墙钟时间、用户名、主机或真实路径。

- [ ] **Step 4: 规范化 ZIP 包以保证字节稳定**

实现确定性重打包：

```python
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def _canonicalize_docx_package(content: bytes) -> bytes:
    source = ZipFile(BytesIO(content), "r")
    target_buffer = BytesIO()
    with source, ZipFile(
        target_buffer,
        "w",
        compression=ZIP_DEFLATED,
        compresslevel=9,
    ) as target:
        for name in sorted(source.namelist()):
            info = ZipInfo(filename=name, date_time=_FIXED_ZIP_TIME)
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            info.create_system = 3
            target.writestr(info, source.read(name), compress_type=ZIP_DEFLATED, compresslevel=9)
    return target_buffer.getvalue()
```

拒绝 duplicate ZIP member；不保留原 ZIP comment、extra、创建系统和时间戳。

- [ ] **Step 5: 扩展单一 runnable check**

在同一个 `tests/test_terminal_reporting.py` 中：对同一 model 连续渲染两次，断言 bytes 相等；用标准库 `zipfile` 读取 `word/document.xml` 和 `docProps/core.xml`，断言包含 TOP10 Rule ID，不含分类原值、路径片段、用户名或当前年份；用 `Document(BytesIO(docx_bytes))` 验证 Word 包可打开。

- [ ] **Step 6: 源码级核对 Task 2**

确认 `reporting_docx.py` 不导入 Markdown renderer，不访问文件系统，不使用 `datetime.now()`，不从 model 之外计算排序或业务结论。

---

### Task 3: 原子发布和验证 terminal report bundle

**Files:**
- Create: `src/riskprobe/terminal_reports.py`
- Modify: `tests/test_terminal_reporting.py`

**Interfaces:**
- Consumes: `ReportModel`、Markdown `str`、DOCX `bytes`、私有 terminal subject。
- Produces:
  - `TerminalReportSubject`
  - `TerminalReportManifest`
  - `TerminalReportStore.ensure_published(...) -> TerminalReportManifest`
  - `TerminalReportError`

- [ ] **Step 1: 定义不依赖 Host outcome 类型的私有 subject**

把 shared internal subject 放在 `terminal_reports.py`，避免 `host_decision` 循环依赖：

```python
@dataclass(frozen=True, slots=True)
class TerminalReportSubject:
    idempotency_key: str
    run_id: str
    context_id: str | None
    findings: tuple[DecisionFinding, ...]
    proposal_action_codes: tuple[str, ...]
    diagnosis_evidence_ids: tuple[str, ...]
    agent_result: AgentResult | None
    analysis_summary: AnalysisSummary | None
    decision_summary: DecisionSummary | None
    terminal_status: Literal["accepted", "rejected", "no_action", "failed"]
    error_code: str | None = None
```

`idempotency_key` 仅用于派生 report ID，不进入 model、manifest 或正文。`run_id` 必须来自已验证 result/session binding，不接受 MCP 参数。subject 只保存报告需要的安全 context 投影，避免导入 Host outcome 类型或形成循环依赖。

- [ ] **Step 2: 定义严格 manifest**

`TerminalReportManifest` 使用 frozen/extra-forbid 模型，字段固定为 schema version、report ID、run/context/session binding、model SHA-256、两个逻辑文件名及各自 SHA-256/size。文件名只允许 `final_risk_report.md` 与 `final_risk_report.docx`。

report ID 固定派生：

```python
def derive_report_id(subject: TerminalReportSubject) -> str:
    payload = "\n".join(
        (
            subject.idempotency_key,
            subject.run_id,
            subject.context_id or "no-context",
            subject.agent_result.leaf_node_id if subject.agent_result is not None else "no-result",
            subject.terminal_status,
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
```

- [ ] **Step 3: 实现私有目录和精确权限**

store 根目录固定为 `state_dir/.riskprobe-terminal-reports`，0700；每个 report 目录 0700；lock、Markdown、DOCX、manifest 均 0600。所有 open 都使用 `O_NOFOLLOW`（平台存在时）并核对 owner、regular-file、link count 和 mode。

- [ ] **Step 4: 实现 manifest-last 发布**

`ensure_published()` 在 per-report lock 内执行：

1. 计算 canonical model JSON 与 SHA-256；
2. 若 manifest 存在，验证 binding、model hash、文件名、hash/size、owner/mode；完整则直接返回；
3. manifest 存在但任何 binding/hash 不同，抛 `TerminalReportError("terminal_report_integrity_failed")`，禁止覆盖；
4. manifest 不存在时渲染 Markdown/DOCX；
5. 分别 tempfile → fsync → `os.replace` 两个内容文件；
6. 最后原子发布 manifest 并 fsync report 目录。

Markdown 和 DOCX 任一失败时不发布 manifest。残留无 manifest 内容文件不算成功，下次同 subject 可确定性覆盖修复。

- [ ] **Step 5: 在单一 check 文件覆盖 store 不变量**

用 `tempfile.TemporaryDirectory()` 而非 fixture：第一次发布后断言三文件存在且 manifest hash/size 匹配；第二次发布返回同 manifest；删除 manifest 后重发得到相同文件 hash；篡改已发布 DOCX 后必须抛 `TerminalReportError`，不得静默覆盖。

- [ ] **Step 6: 源码级核对 Task 3**

确认 raw idempotency key 不进入任何文件内容/目录名；manifest 是成功标志且最后发布；完整性冲突 fail-closed；没有 SQLite、网络、run-dir 写入或公开路径返回。

---

### Task 4: 从已验证 run 构建完整终态报告

**Files:**
- Modify: `src/riskprobe/artifacts.py:318-513`
- Modify: `src/riskprobe/service.py:695-767, 181-190, 1951-2135`
- Modify: `src/riskprobe/terminal_reports.py`

**Interfaces:**
- Consumes: verified run ID、`analysis_summary.json`、`evidence_cards.json`、`metadata_report.json`、Task 3 subject。
- Produces:
  - `RunStore.open_verified(run_id: str) -> RunContext`
  - `RiskProbeService.ensure_terminal_report(subject) -> TerminalReportManifest | None`

- [ ] **Step 1: 增加只读 verified run reopen**

在 `RunStore` 新增 `open_verified(run_id)`：验证 run ID 是普通安全组件；打开已有目录，不创建文件；复用现有 manifest、artifact set、hash/size、symlink 和 integrity-anchor 校验；返回只允许 `read_verified_artifact()` 的 context。不存在、incomplete、anchor 不匹配或多余文件继续 fail-closed。

- [ ] **Step 2: 从 verified bytes 恢复安全聚合模型**

复用现有 JSON 恢复函数，把 `analysis_summary.json` 严格解析为 `AnalysisSummary`，把 `evidence_cards.json` 恢复为 `tuple[EvidenceCard, ...]`，metadata 只读取 allowlisted 聚合字段。禁止直接 `Path.read_*` 或读取 candidate Parquet。

- [ ] **Step 3: 完成 final model builder**

`build_final_report_model(...)` 合并：

- terminal `AnalysisSummary`，确保 24 stages 各一次；
- verified TOP10 cards；
- `subject.findings` 的 evidence ID、kind、severity、summary、limitations；
- `DecisionSummary.recommendations`；
- `AgentResult.review`、tool sequence、state history 和 terminal status；
- safe failure 的 stage progress/error code。

accepted/rejected/no-action/failed 各自使用真实 status，不把 no-action 或 failed 伪装为 rejected。构建后调用 `assert_safe_payload`。

- [ ] **Step 4: 在 service 中拥有 terminal report service**

`RiskProbeService.__init__` 用现有 `state_dir` 和 `runs_dir` 创建 `TerminalReportStore`。新增：

```python
def ensure_terminal_report(
    self,
    subject: TerminalReportSubject,
) -> TerminalReportManifest | None:
    context = self._run_store.open_verified(subject.run_id)
    inputs = self._load_verified_report_inputs(context)
    model = build_final_report_model(subject=subject, **inputs)
    return self._terminal_report_store.ensure_published(model=model, subject=subject)
```

只有 subject 有安全 run binding 时调用；启动级 failure 返回 `None`。

- [ ] **Step 5: 为安全失败保留内部 run binding**

给内部 `HostSafeStageError` 增加可选 `report_run_id`，默认 `None`。仅当 `self.run()` 已返回 verified run 后，后续 Host/Agent 异常才绑定 run ID；error code 和公开异常映射不变，异常文本不持久化。

- [ ] **Step 6: 源码级核对 Task 4**

确认终态报告只读取三个 verified JSON/text 聚合 artifact；report model 无 raw path/idempotency key；24 stages、TOP10 和 scorecard 来自权威 summary/cards；run context 没有任何 write 方法被调用。

---

### Task 5: 在 Host 终态持久化后、MCP 返回前发布报告

**Files:**
- Modify: `src/riskprobe/host_decision.py:233-293, 480-752, 856-1046`
- Modify: `src/riskprobe/mcp_server.py:26-116, 135-166`
- Modify: `src/riskprobe/service.py`

**Interfaces:**
- Consumes: persisted Host session、Task 3 subject、Task 4 `ensure_terminal_report()`。
- Produces:
  - `HostDecisionCoordinator.report_subject(idempotency_key: str) -> TerminalReportSubject | None`
  - 不变的两个 MCP tool schemas。

- [ ] **Step 1: 在私有 Host session 保存可恢复 report binding**

私有 `_SessionState` 增加 `report_run_id: str | None`；legacy session 缺字段时读为 `None`。成功/no-action 从 `AgentResult.session_id` 绑定；有 context 的失败从 `HostSafeStageError.report_run_id` 绑定。公开 outcome/failure 模型不增加字段。

- [ ] **Step 2: 实现只读 `report_subject()`**

该方法先 refresh/persisted-state validate，只在以下情况返回 subject：

- accepted/rejected/no-action terminal 已完整持久化；
- failed session 具有 allowlisted failure 和安全 run binding。

awaiting proposal、context timeout、proposal identity/source/version/evidence 前置校验失败返回 `None`。subject 中 outcome 拆为现有 `AnalysisSummary`、`DecisionSummary`、`AgentResult`；context/proposal 仅投影同一 session 已验证的 `context_id`、findings、action codes 和 diagnosis evidence IDs。

- [ ] **Step 3: 新增 MCP 内部 report gate**

在 `create_mcp_server()` closure 中新增：

```python
def _report_failure(idempotency_key: str) -> HostDecisionFailure | None:
    subject = coordinator.report_subject(idempotency_key=idempotency_key)
    if subject is None:
        return None
    try:
        service.ensure_terminal_report(subject)
    except TerminalReportError:
        return HostDecisionFailure(error_code="agent_orchestration_failed")
    return None
```

get 工具在 `coordinator.get_context()` 返回后调用；awaiting context 不生成，no-action/safe failure 生成。submit 工具在 `coordinator.submit_proposal()` 返回后调用；accepted/rejected 生成。report failure 替换本次返回值，但不覆盖已持久化 terminal。

- [ ] **Step 4: 保证重放只修复报告**

同 key/proposal 再调用时，coordinator 直接返回 persisted terminal；report gate 调用同一 `ensure_published()`。manifest 完整则零渲染复用；无 manifest 则只从 verified terminal/run 重建报告，不启动 runner、gateway、provider、recommend 或 review。

- [ ] **Step 5: 冻结公开协议**

逐项核对：

- get 仍只有 `idempotency_key`；
- submit 仍只有 `idempotency_key/context_id/diagnosis_evidence_ids/action_codes`；
- `HostDecisionContext/Outcome/NoActionOutcome/Failure` 字段集合不变；
- `AgentResult/AnalysisSummary/DecisionSummary` 字段集合不变；
- error allowlist 不新增 report code；
- payload 中没有 report ID、逻辑文件以外路径、hash 或 bytes。

- [ ] **Step 6: 源码级核对四分支**

accepted、rejected、no-action、有安全 run binding 的 failed 均到达 report gate；pre-context failure、timeout、非法 proposal 不生成。报告失败后的重试不会重复 Agent 工具或业务建议。

---

### Task 6: 清理计数漂移、更新交付说明并完成一次验收

**Files:**
- Modify: `src/riskprobe/cli.py:276-301`
- Modify: `src/riskprobe/tools/models.py:180-193` only if a stale default remains
- Modify: `.kiro/skills/riskprobe/SKILL.md`
- Modify: `CHANGELOG.md`
- Modify: stale exact-count assertions in existing test files only when they contradict verified manifest behavior
- Review: all files changed by Tasks 1-5

**Interfaces:**
- Run artifact count remains derived only from verified run manifest。
- Kiro may state logical report filenames after terminal, but must not expose actual state-dir path。

- [ ] **Step 1: 从 verified manifest 派生 artifact count**

删除 CLI/response 中硬编码 6 或 7 的默认事实；复用 manifest 的 artifact list 长度。`final_risk_report.md/.docx` 是 terminal sidecar，不计入 run artifact count。

- [ ] **Step 2: 更新 Kiro 交付说明**

在 RiskProbe skill 中要求 terminal 后：

- 输出用户可读的完整阶段摘要和 TOP10 指标；
- 明确本地逻辑文件名为 `final_risk_report.md`、`final_risk_report.docx`；
- 不输出真实路径；
- 若 MCP 返回 report gate failure，只报告固定错误码并允许同 proposal 安全重试；
- 不从未返回的数据推断结论。

- [ ] **Step 3: 更新 changelog**

记录：中文完整终态双格式报告、TOP10 安全规则条件、确定性 DOCX、terminal sidecar 幂等修复、公开 MCP 契约不变。不得写入真实运行路径或数据值。

- [ ] **Step 4: 对照设计做一次集中源码审查**

逐项检查：十章内容、24 stages、TOP10 指标、评分卡 AUC/KS/Gini、机构 token、diagnostics/recommend/review、accepted/rejected/no-action/safe failure、manifest-last、DOCX 稳定化、legacy run/session、两工具 schema 和 error code。

- [ ] **Step 5: 执行一次语义审查**

仅调用一次 `semantic_reviewer`，重点检查：是否有 raw condition/segment/path 泄露；terminal 决策是否可能因报告失败重复执行；DOCX 是否确定性；是否修改公开 schema/error code；是否破坏 run manifest。修复 Critical/Important 后不追加无关评审。

- [ ] **Step 6: 重新加载 MCP 后做真实全流程验收**

使用全新稳定幂等键调用 `riskprobe_get_decision_context`；核对 awaiting context 的 24 stages 和完整 findings。按 policy 提交原样 context/evidence 和允许 actions；要求返回 terminal、review 完成而非 report failure。由于 report gate 是 terminal 返回前置条件，成功 terminal 即证明双格式 bundle 已完整发布，无需读取真实路径。

- [ ] **Step 7: 验证幂等重放**

完全相同 proposal 再提交一次，要求返回相同 terminal；工具顺序仍为 `inspect → diagnose → discover → recommend → review`，没有新增 Agent 工具调用，也没有新的公开字段。报告 store 应命中已有 manifest，不重复渲染。

- [ ] **Step 8: 明确未执行项**

最终说明中明确：因 workspace 规则未运行 Shell、Python、pytest、SQL、网络或 SQLite；未读取本地报告文件、原始数据或日志；未创建 commit。报告内容通过源码契约、单次语义审查和 MCP terminal gate 间接验证。

## Completion Criteria

- 每个新分析 run 的 `risk_report.md` 包含中文阶段摘要和恰好 `min(10, evidence_count)` 条安全规则。
- 每个可验证 accepted、rejected、no-action 或有安全摘要的 failed 终态都原子发布 Markdown、DOCX 和 manifest。
- 最终报告包含 24 stages、规则条件与全套指标、评分卡、机构稳定性、diagnostics、recommendations 和 review。
- DOCX 包无墙钟时间、用户名、真实路径或分类原值；同 model 字节稳定。
- 同 key/proposal 重放不重复 Agent 工具，报告完整则直接复用，部分发布可安全修复。
- run manifest、legacy 6/7 artifact 读取、MCP 两工具 schema 和固定 error code 不变。
- 未引入 HTML/PDF、配置开关、新 MCP 工具、外部服务、自动业务动作或未经授权的 Git commit。
