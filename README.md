# RiskProbe Agent

**Privacy-safe local risk intelligence for explainable rules, PSI drift monitoring, institution stability analysis, and MCP/CLI workflows.**

RiskProbe is a local-first Python toolkit for credit-risk and model-monitoring workflows. It turns a confirmed local Parquet dataset into auditable aggregate evidence without uploading data or exposing entity-level records.

> **Status:** `v0.1.0` initial public release. The project is local-only: GitHub distributes code and examples, not user data.

## Why RiskProbe?

Risk and model-monitoring workflows often need more than a model score: analysts need to know which rules are reproducible, whether performance is stable across institutions, whether drift is global or localized, and what evidence is safe to share with an Agent. RiskProbe keeps the deterministic calculations in Python and uses Kiro, Codex, Trae, or another MCP client only as a controlled orchestration layer.

## Features

- **Explainable rule discovery:** quantile and shallow-tree thresholds, LightGBM-assisted candidate generation, deterministic one-condition and two-condition rule search, stable rule IDs.
- **Statistical validation:** Train/Test and optional time-slice validation with support, coverage, bad rate, Lift, precision, recall, Fisher p-values, BH/FDR adjustment, bootstrap Lift confidence intervals, segment consistency, and Lift decay.
- **Institution stability:** global-first discovery, institution-level metrics, `Stable`/`Local`/`Unstable`/`Suspicious` grading, and conditional local-rule discovery for sufficiently supported institutions.
- **Aggregate drift monitoring:** schema changes, missingness, PSI distribution drift, population-share changes, label-rate changes, and rule-Lift decay.
- **Root-cause diagnosis:** aggregate feature, family, segment, label, rule, and schema dimensions with deterministic contribution ranking and TOP3 explanations.
- **Read-only Parquet onboarding:** schema preview, explicit role confirmation, exact feature-list confirmation, allowlisted local registration, and no automatic role guessing.
- **Cross-client Agent access:** standard stdio MCP for Kiro, Codex, Trae, and other MCP clients, plus CLI and reusable `AGENTS.md`/system-prompt instructions.
- **Privacy controls:** aggregate-only outputs, stable tokens, suppressed small groups, default real institution names in restricted fields, and opt-out masking with `privacy.expose_segment_values=false`.

### What is not currently implemented

The public `v0.1.0` release does **not** claim ADASYN/SMOTE oversampling, KS testing, online serving, database connectors, remote data uploads, or automatic policy deployment. These are possible future directions, not current capabilities.

## Architecture

```text
Local Parquet / DataFrame
          │ read-only + allowlist
          ▼
Deterministic Risk Engine
  profiling → discovery → validation → monitoring → diagnosis
          │ aggregate JSON only
          ▼
CLI / stdio MCP
          │
Kiro Agent · Codex · Trae · other MCP clients
```

The core engine is client-independent. Kiro configuration provides the most integrated Agent experience; other MCP-capable clients use the same tools with their own configuration and project instructions.

## Tech stack

- Python 3.11+
- Polars and PyArrow for local columnar data
- Pydantic for typed configuration and contracts
- LightGBM, scikit-learn, SciPy, and statsmodels for deterministic discovery and statistics
- Typer for CLI
- FastMCP for local stdio tool integration
- PyYAML for project configuration

## Quick start

### 1. Install

macOS/Linux:

```bash
git clone https://github.com/LiShuo-Duke/riskprobe-agent.git
cd riskprobe-agent
python3 -m venv .venv
./.venv/bin/python -m pip install -e .
```

Windows PowerShell:

```powershell
git clone https://github.com/LiShuo-Duke/riskprobe-agent.git
Set-Location riskprobe-agent
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
```

For development and tests:

```bash
./.venv/bin/python -m pip install -e '.[dev]'
```

### 2. Run the public synthetic example

The example is deterministic and does not require private data:

```bash
mkdir -p data/synthetic
./.venv/bin/riskprobe synthetic \
  --output data/synthetic/behavior.parquet \
  --rows 5000 \
  --seed 42
./.venv/bin/riskprobe inspect \
  --config configs/synthetic.example.yaml \
  --runs-dir runs
./.venv/bin/riskprobe run \
  --config configs/synthetic.example.yaml \
  --runs-dir runs
```

Results are written to the local `runs/` directory, which is ignored by Git.

### 3. Use a private local Parquet dataset

Set an explicit allowlist before using the MCP server:

```bash
mkdir -p "$HOME/riskprobe-data"
export RISKPROBE_ALLOWED_DATA_ROOTS="$HOME/riskprobe-data"
```

Place the Parquet file inside that directory. RiskProbe does not upload it, modify it, or return entity-level rows.

## MCP and Agent clients

RiskProbe exposes a local stdio MCP server, not an HTTP endpoint:

```bash
./.venv/bin/python -m riskprobe.mcp_server
```

Configuration templates:

```text
configs/mcp/mcp.example.json       # generic MCP JSON
configs/mcp/codex.example.toml     # Codex project configuration
configs/mcp/trae.example.json      # Trae/manual MCP configuration
```

Replace template paths with paths on the user’s machine. The standard workflow is:

```text
inspect_local_parquet_schema
→ confirm entity / time / institution / target roles
→ preview_local_parquet_features
→ confirm exact feature columns
→ register_local_parquet
→ inspect_dataset
→ discover_rules(objective="risk")
→ validate_rules
→ detect_anomalies
→ diagnose_anomaly
→ build_report
```

`discover_rules` must not receive non-empty `constraints`; discovery thresholds come from the registered project configuration. If there is no real time column, only random Train/Test validation is allowed and results must not be described as strict OOT.

### Client matrix

| Client | Integration | Full RiskProbe core | Native RiskProbe Agent config |
|---|---|---:|---:|
| Kiro | `.kiro/agents`, `.kiro/skills`, workspace MCP | Yes | Yes |
| Codex | MCP TOML + `AGENTS.md` | Yes | Uses Codex instructions |
| Trae | Manual MCP JSON + system prompt | Yes | Uses Trae instructions |
| Other MCP client | Local stdio MCP + project prompt | Yes | Client-dependent |
| No MCP support | Python package and CLI | CLI only | No |

See [`docs/cross-client-usage.md`](docs/cross-client-usage.md) and [`docs/agent-system-prompt.md`](docs/agent-system-prompt.md).

## Privacy and safety boundaries

- Local Parquet access is read-only and restricted to `RISKPROBE_ALLOWED_DATA_ROOTS`.
- Outputs are aggregate-only; entity values, sample rows, raw logs, real filesystem paths, and Parquet detail reads are not part of the Agent surface.
- Shell, arbitrary SQL/Python, network access, and automatic policy deployment are not part of the workflow.
- Institution names are shown by default only in restricted aggregate fields and can be masked with:

```yaml
privacy:
  expose_segment_values: false
```

- Institution-local rules remain validation evidence for human review; they are not automatically promoted to global rules or production policies.
- Grade-B evidence has an unknown performance window and must not be described as strict OOT or production-ready.

## Repository layout

```text
src/riskprobe/             deterministic engine, CLI, MCP server
configs/                   public example configs and MCP templates
.kiro/                     Kiro Agent, Skill, and workspace MCP config
AGENTS.md                  client-independent project rules
docs/                      cross-client instructions and technical guide
tests/                     unit, integration, and security-boundary tests
```

## Roadmap

- Add optional, explicitly configured class-imbalance strategies after evaluating leakage and reproducibility implications.
- Add complementary statistical tests such as KS only when their data contract and interpretation are specified.
- Improve packaging and client adapters while keeping one deterministic engine.
- Consider remote deployment only after authentication, tenant isolation, audit, and data-governance requirements are designed.

## Development

```bash
./.venv/bin/python -m pytest --disable-warnings --maxfail=1
./.venv/bin/ruff check src tests
```

Contributions should preserve the local-only boundary, deterministic seeds, explicit role confirmation, aggregate-only output, and no automatic policy deployment.
