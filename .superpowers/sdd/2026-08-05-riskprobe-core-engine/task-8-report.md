# Task 8 实现报告

## 状态

完成。仅实现 Task 8 的本地 CLI 与公开合成演示；未扩展 Task 9/Plan 2，未使用 Kiro 或外部模型 API，未上传任何数据。

## Commit

- 本地检查点：`feat: add riskprobe command line workflow`（本报告与实现一并提交；提交 hash 在提交完成后由 Git 分配）。
- 基线：`5f61999`。

## RED / GREEN

### RED

1. 首先创建 `tests/test_cli.py` 后运行 `.venv/bin/python -m pytest tests/test_cli.py -v`：收集阶段因 `riskprobe.cli` 不存在报错，符合 brief 的初始 RED 预期。
2. 独立审查后增加解析错误回归：`--rows NaN` 和未知选项由 Typer 输出富文本 Usage/Error，不是安全的结构化 JSON；回归测试失败。
3. 复审后增加不可用 `--runs-dir` 回归：有效配置配合普通文件路径被错误分类为 `configuration_error`；回归测试失败。

### GREEN

- 目标 CLI 测试：`8 passed`（最终全量覆盖命令包含全部 8 个 CLI 测试）。
- 全量覆盖测试：`187 passed in 42.38s`，总覆盖率 `94%`。
- Ruff：`.venv/bin/ruff check src tests` 通过。
- `git diff --check`：通过，无输出。
- `riskprobe --help`：通过，列出 `synthetic`、`inspect`、`discover`、`run`。
- 真实公开演示：`riskprobe synthetic --output /tmp/riskprobe-demo.parquet --rows 20000 --seed 42` 后使用示例配置运行 `riskprobe run` 成功，输出稳定 run ID 与 `metadata_grade=B`，第二次运行复用同一不可变 run。
- Git 忽略检查：`data/synthetic/behavior.parquet` 命中 `/data/`，`runs/demo/manifest.json` 命中 `/runs/`；没有 Parquet 或 run 产物进入待提交列表。

## 修改文件

- `src/riskprobe/cli.py`
  - 提供固定签名的 `synthetic`、`inspect`、`discover`、`run` Typer 命令。
  - 复用 Task 3 合成生成器与 Task 7 `RiskProbeService`；不复制发现、验证或报告逻辑。
  - synthetic 仅输出行数、列数和真值规则 ID；inspect/discover 输出安全 JSON 摘要；run 输出 run ID 与 metadata grade。
  - 回调、配置、运行目录、执行和解析失败均使用不含样本、路径、原始参数或底层异常的结构化错误与退出码 2。
- `configs/synthetic.example.yaml`
  - 公开 `data/synthetic/behavior.parquet`、四个固定列角色及 `order_`、`browse_`、`multi_platform_`、`emb_` 特征前缀。
  - `performance_window_days: null`，因此演示明确为 metadata Grade B。
- `tests/test_cli.py`
  - 覆盖命令解析、成功 JSON/人类输出、失败退出码 2、解析失败安全 JSON、公开合成到真实 run、确定性、覆盖写入、B 级示例配置、无样本/路径泄露和不可用运行目录。

## 自审

- 所有 CLI 文本都只包含聚合计数、规则 ID、grade、run ID 或固定消息；未打印 DataFrame、样本行、段值、配置路径、数据路径、用户输入 token 或异常详情。
- 解析错误在根 Typer group 边界统一转换，因此类型错误和未知选项不会泄露其原始参数。
- 配置 YAML 解析与 `RunStore` 初始化分开，分别提供 `configuration_error` 与 `runs_directory_error` 的可行动消息。
- 公开示例和真实 smoke 仅使用 `/tmp` 合成数据；输出目录由已有 `.gitignore` 排除。
- 审查提出的高优先级解析错误边界与低优先级 runs-dir 分类均已补 RED 回归并修复；无阻断 concern，未 push。
