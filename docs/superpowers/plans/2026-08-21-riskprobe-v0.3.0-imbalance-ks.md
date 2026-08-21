# RiskProbe v0.3.0 Imbalance and KS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不引入新依赖、不改变默认结果语义的前提下，为 RiskProbe 增加可选的 train-only 类别不平衡处理和可解释的 KS 互补检验，并发布 v0.3.0。

**Architecture:** 使用严格的 `ImbalanceConfig` 作为 ProjectConfig 的显式策略边界；只在 Train 拟合树、LightGBM 和 LogisticRegression 时使用 balanced class/sample weights，规则 Test/Holdout 证据继续使用未加权真实人群指标。KS 分成两层：`RuleMetrics` 记录二值规则的 bad/good 分离度，`compute_score_ks` 对冻结连续 score 做 finite-filtered 两样本检验；KS 不替代 Fisher p-value，也不改变现有 grade 默认门槛。

**Tech Stack:** Python 3.11+, Pydantic 2, NumPy, SciPy, scikit-learn, Polars, LightGBM, pytest, Ruff。

## Global Constraints

- 默认 `imbalance.enabled` 为 `false`，关闭时现有规则和评分卡结果保持不变。
- 只从 Train 的 0/1 target 派生 balanced 权重；Test/Holdout、全量 profile positive rate 和任何未来标签不得参与权重。
- v0.3.0 不实现 SMOTE/ADASYN/随机过采样，不新增依赖，不接受任意 weight 列。
- WOE/IV edges、规则阈值、LogisticRegression 系数继续只从 Train 拟合；Test/Holdout 只调用冻结变换和验证。
- 保留现有 Fisher `p_value`/`adjusted_p_value` 语义；KS 是新增互补指标，不自动改变 Stable/Local/Unstable/Suspicious 分级。
- KS 空类或没有 finite score 时返回 `None` 统计量并写 limitation，不用 0 混淆“不可用”和“无分离”。
- 不修改现有两工具 MCP 边界；版本为 `0.3.0`，变更需在隔离 worktree 完成后合并到远程 `main`。

---

### Task 1: Strict imbalance contract and train-only weight helpers

**Files:**
- Modify: `src/riskprobe/config.py`
- Modify: `src/riskprobe/metrics.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_metrics.py`

**Interfaces:**
- `ImbalanceConfig(enabled=False, strategy="class_weight", weighting="balanced")` is strict/frozen and rejects unknown fields or unsupported strategies.
- `balanced_class_weights(target) -> dict[int, float]` and `balanced_sample_weights(target) -> numpy.ndarray` accept only a non-empty one-dimensional binary target containing both classes; both use `n / (2 * class_count)` and return finite positive weights.

- [ ] **Step 1: Write the failing tests.**

```python
def test_imbalance_config_defaults_off_and_rejects_unsupported_strategy():
    config = ImbalanceConfig()
    assert config.enabled is False
    assert config.strategy == "class_weight"
    with pytest.raises(ValidationError):
        ImbalanceConfig.model_validate({"enabled": True, "strategy": "smote"})


def test_balanced_weights_use_train_class_counts_only():
    target = np.array([0, 0, 0, 1], dtype=np.int8)
    assert balanced_class_weights(target) == {0: 2 / 3, 1: 2.0}
    np.testing.assert_allclose(
        balanced_sample_weights(target),
        np.array([2 / 3, 2 / 3, 2 / 3, 2.0]),
    )


@pytest.mark.parametrize("target", [[], [0, 0], [1, 1], [0, 2], [0, np.nan]])
def test_balanced_weights_fail_closed_for_invalid_or_single_class_target(target):
    with pytest.raises(ValueError, match="binary target"):
        balanced_sample_weights(np.asarray(target, dtype=object))
```

- [ ] **Step 2: Run the focused tests and verify the expected RED failure.**

Run:

```bash
PYTHONPATH=src python3 -m pytest -q tests/test_config.py -k imbalance tests/test_metrics.py -k balanced_weights
```

Expected: import/attribute failures because `ImbalanceConfig` and the weight helpers do not exist.

- [ ] **Step 3: Implement the strict config and helpers.** Add `ImbalanceConfig` beside the other strict config blocks and add it to `ProjectConfig` as `imbalance: ImbalanceConfig = ImbalanceConfig()`. In `metrics.py`, validate shape, finite numeric values, exact `{0, 1}` classes, and compute the same deterministic class weight for every row of that class. Do not read any DataFrame column other than the passed Train target.

- [ ] **Step 4: Run the focused tests and preserve existing metric behavior.**

```bash
PYTHONPATH=src python3 -m pytest -q tests/test_config.py tests/test_metrics.py
```

Expected: all existing config/metric tests and new imbalance tests pass.

---

### Task 2: Connect optional imbalance strategies to discovery and scorecard

**Files:**
- Modify: `src/riskprobe/rules/discovery.py`
- Modify: `src/riskprobe/rules/scorecard.py`
- Modify: `src/riskprobe/rules/__init__.py`
- Modify: `src/riskprobe/service.py`
- Modify: `tests/rules/test_discovery.py`
- Modify: `tests/test_scorecard.py`
- Modify: `tests/test_service.py`

**Interfaces:**
- `discover_with_metrics(..., *, imbalance: ImbalanceConfig | None = None)` and `discover_rules(..., *, imbalance: ImbalanceConfig | None = None)` preserve existing positional calls.
- `fit_scorecard(..., imbalance: ImbalanceConfig | None = None, random_seed: int = 42)` preserves the existing API and stores `imbalance_strategy`, `random_seed`, `calibrated=False`, and Train `class_counts` on `ScorecardModel`.
- `RiskProbeService` passes `self.config.imbalance` to discovery and uses `_snapshot_dataset()` in `discover_with_metrics()` so inspect and discovery use one immutable input snapshot.

- [ ] **Step 1: Write RED tests for disabled, class-weight, and sample-weight paths.**

```python
def test_discovery_imbalance_is_default_off_and_train_only():
    baseline = discover_with_metrics(train, FEATURES, "target", DiscoveryConfig())
    explicit_off = discover_with_metrics(
        train, FEATURES, "target", DiscoveryConfig(), imbalance=ImbalanceConfig()
    )
    assert baseline == explicit_off


def test_scorecard_records_weight_strategy_without_calibration():
    model = fit_scorecard(
        train,
        feature_names=("amount",),
        target_col="target",
        imbalance=ImbalanceConfig(enabled=True, strategy="sample_weight"),
    )
    assert model.imbalance_strategy == "sample_weight"
    assert model.calibrated is False
    assert model.class_counts == (int((train["target"] == 0).sum()), int((train["target"] == 1).sum()))
```

Add a small estimator-spy test or monkeypatch around the existing tree/LightGBM/LogisticRegression constructors to assert that sample weights have exactly the filtered Train length and that no Test/Holdout frame is accepted by the fit API.

- [ ] **Step 2: Run the new tests and verify RED failures.**

```bash
PYTHONPATH=src python3 -m pytest -q tests/rules/test_discovery.py -k imbalance tests/test_scorecard.py -k weight
```

Expected: unexpected keyword/attribute failures because the new optional parameters and model provenance fields do not exist.

- [ ] **Step 3: Implement the minimum weighted fit behavior.** Thread `ImbalanceConfig` through discovery only to `_tree_thresholds` and `_lightgbm_thresholds`; use class weights on the estimator or `balanced_sample_weights` on the exact finite feature mask. Leave `_candidate` and every `RuleMetrics` calculation unweighted. In `fit_scorecard`, keep `fit_woe_binning` unchanged and pass either `class_weight` to LogisticRegression or `sample_weight` to `.fit()`. Reject an invalid config type and record strategy/class counts; do not claim calibrated probabilities.

- [ ] **Step 4: Connect ProjectConfig to service discovery and repair snapshot reproducibility.** Pass `imbalance=self.config.imbalance` from both `_discover_from_train` and `_discovery_result_from_train`. Change `RiskProbeService.discover_with_metrics()` to perform profile, feature selection, and partition inside one `_snapshot_dataset()` context, matching `discover()`.

- [ ] **Step 5: Run discovery, scorecard, and service tests.**

```bash
PYTHONPATH=src python3 -m pytest -q tests/rules/test_discovery.py tests/test_scorecard.py tests/test_service.py -k 'discovery or scorecard or imbalance or snapshot'
```

Expected: existing deterministic behavior remains green and both enabled strategies are reproducible.

---

### Task 3: Add KS metrics with strict interpretation and unavailable states

**Files:**
- Modify: `src/riskprobe/models.py`
- Modify: `src/riskprobe/metrics.py`
- Modify: `src/riskprobe/explainability.py`
- Modify: `src/riskprobe/reporting.py` only if the current metric table requires a new column
- Modify: `tests/test_metrics.py`
- Modify: `tests/rules/test_validation.py`
- Modify: `tests/test_scorecard.py`

**Interfaces:**
- `RuleMetrics` gains backward-compatible fields `hit_good_rate: float = 0.0`, `ks_signed: float | None = None`, and `ks_stat: float | None = None`; `p_value` remains Fisher exact-test p-value.
- `ScoreSeparation` is a strict result DTO with `statistic`, `signed_statistic`, `p_value`, `bad_count`, `good_count`, `excluded_count`, `method`, and `limitation`.
- `compute_score_ks(scores, target, *, positive_value=1, direction="higher_is_bad") -> ScoreSeparation` filters only non-finite scores, rejects malformed target input, computes SciPy two-sample KS p-value and a direction-aware signed statistic, and never fits or selects a score.

- [ ] **Step 1: Write RED tests for binary-rule and continuous-score KS.**

```python
def test_rule_metrics_expose_bad_good_ks_separation():
    metrics = compute_rule_metrics(
        np.array([True, True, False, False]),
        np.array([1, 0, 0, 0]),
        positive_value=1,
    )
    assert metrics.hit_good_rate == 1 / 3
    assert metrics.ks_signed == 1.0
    assert metrics.ks_stat == 1.0
    assert metrics.p_value != metrics.ks_stat


def test_score_ks_filters_nonfinite_scores_and_preserves_direction():
    result = compute_score_ks(
        np.array([0.9, 0.8, np.nan, 0.1, 0.2]),
        np.array([1, 1, 0, 0, 0]),
    )
    assert result.statistic == pytest.approx(1.0)
    assert result.signed_statistic == pytest.approx(1.0)
    assert result.bad_count == 2
    assert result.good_count == 2
    assert result.excluded_count == 1
    assert result.p_value is not None


def test_score_ks_returns_unavailable_for_single_class_after_filtering():
    result = compute_score_ks([0.1, float("nan")], [1, 1])
    assert result.statistic is None
    assert result.p_value is None
    assert result.limitation == "single_class_or_no_finite_scores"
```

- [ ] **Step 2: Run the metric RED tests.**

```bash
PYTHONPATH=src python3 -m pytest -q tests/test_metrics.py -k 'ks or separation'
```

Expected: missing fields/function failures.

- [ ] **Step 3: Implement KS without changing Fisher semantics.** Compute binary-rule KS as `hit_bad_rate - hit_good_rate` and absolute value when both classes exist. For continuous scores use finite-score masks, `scipy.stats.ks_2samp`, and a ROC-style signed separation for the requested direction. Return a limitation instead of a fake zero if either class has no finite scores.

- [ ] **Step 4: Add serialization and validation coverage.** Include the new RuleMetrics fields in explainability metric allowlists and any report metric table. Keep `validate_rules` grade logic unchanged; it automatically carries KS through train/test/slices via `compute_rule_metrics`. Add tests proving Test/Holdout changes affect KS output but do not alter Train-fitted scorecard edges/coefficients.

- [ ] **Step 5: Run metrics, validation, scorecard, and explainability tests.**

```bash
PYTHONPATH=src python3 -m pytest -q tests/test_metrics.py tests/rules/test_validation.py tests/test_scorecard.py
```

---

### Task 4: Version documentation and public contracts

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/superpowers/plans/2026-08-21-riskprobe-v0.3.0-imbalance-ks.md`
- Add or update: focused config/metric/service tests for version and provenance

- [ ] **Step 1: Write version assertions before changing metadata.** Assert package version `0.3.0`, README current version `v0.3.0`, and changelog heading `## v0.3.0`.
- [ ] **Step 2: Update documentation.** Document `imbalance.enabled`, `class_weight`/`sample_weight`, Train-only weighting, uncalibrated scorecard probabilities, KS interpretation (`ks_signed`, `ks_stat`, score direction), unavailable limitations, and the fact that Fisher p-values remain separate.
- [ ] **Step 3: Update package version and changelog.** Change only the current package version and prepend a v0.3.0 entry; retain v0.2.0 history.
- [ ] **Step 4: Run documentation and strict-config checks.**

```bash
python3 -c 'import tomllib; from pathlib import Path; p=tomllib.loads(Path("pyproject.toml").read_text()); assert p["project"]["version"] == "0.3.0"; assert "`v0.3.0`" in Path("README.md").read_text(); assert "## v0.3.0" in Path("CHANGELOG.md").read_text()'
```

---

### Task 5: Verification and GitHub release

**Files:**
- No production files beyond the tasks above.

- [ ] **Step 1: Run targeted tests for config, metrics, discovery, scorecard, validation, and service.**
- [ ] **Step 2: Run `ruff check src tests` and `git diff --check`.**
- [ ] **Step 3: Run the complete `PYTHONPATH=src python3 -m pytest -q` suite and record the result.**
- [ ] **Step 4: Review `git status --short`, ensure only the v0.3.0 worktree changed, and confirm no MCP tool boundary changed.**
- [ ] **Step 5: Commit `feat: release riskprobe v0.3.0`, create annotated tag `v0.3.0`, and push the feature branch/tag.**
- [ ] **Step 6: Fetch latest remote `main`, merge it without force push, run the merge verification, and push `main`.**
- [ ] **Step 7: Confirm remote `main`, feature branch, and tag refs and report URLs.
