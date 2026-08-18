# Company local-validation runbook

## Scope and non-negotiable boundaries

This runbook is for a company-controlled environment only. Every input, command, artifact, and Kiro stdio MCP interaction must stay on the local machine. Use only a read-only local Parquet file, local CSV/Parquet fixtures when testing, the local RiskProbe CLI, and local output directories. Do **not** use a network service, external API, upload, remote storage, SQL, data warehouse connection, or shell command that copies data outside the approved machine.

The repository deliberately contains no company Parquet, field mapping, institution name, real threshold, manual baseline, run artifact, or resume evidence. `configs/company.example.yaml` and `examples/company_schema.json` use fictional names only.

## 1. Prepare an ignored local configuration

1. Keep the real, de-identified Parquet outside this repository. It is read-only input; RiskProbe must not overwrite it.
2. Copy `configs/company.example.yaml` to `configs/company.local.yaml` and change only the local Parquet path, de-identified role-column names, configured feature prefixes, and any metadata supplied by the data owner. Do not commit the resulting file.
3. Confirm the configuration is ignored and confirm no private artifact has entered the index:

   ```bash
   git check-ignore -v configs/company.local.yaml
   git status --short
   git ls-files '*.parquet' '*.csv' 'configs/company.local.yaml' 'runs/*' 'reports/internal/*'
   ```

   The first command must identify an ignore rule. The last command must print nothing. A company Parquet outside the repository need not match `git check-ignore`; its location outside the worktree is the required protection.

## 2. Record a real manual baseline before each task

For each of three to five distinct, comparable historical validation tasks, record the actual manual duration before running RiskProbe. Create a local baseline JSON record outside Git and retain only aggregate counts. It must conform to `schemas/benchmark-record.schema.json` and satisfy the local `BenchmarkRecord` checks: all counts and minutes are non-negative; evidence-passed count cannot exceed candidate count; accepted count cannot exceed reviewed count; and Top-3 hits cannot exceed root-cause cases.

Use a distinct `task_id` for each task. Do not invent a manual duration, candidate count, rule outcome, anomaly outcome, or root-cause result. A missing manual baseline means no efficiency claim may be calculated.

## 3. Perform read-only preflight

Run the aggregate-only schema and role check before any benchmark:

```bash
.venv/bin/riskprobe preflight-company --config configs/company.local.yaml
```

Review only the reported row count, feature and family counts, batch count, label rate, segment count, metadata grade, and limitations. The command must not print entity IDs, sample rows, institution names, source paths, or rule expressions. A B-grade result, including an unknown performance window, is not a strict OOT, no-leakage, production-readiness, or business-impact claim. Stop and resolve configuration or metadata issues before proceeding.

## 4. Measure the local workflow

For each approved task, run the benchmark with an ignored run directory and the corresponding human-authored baseline:

```bash
.venv/bin/riskprobe benchmark \
  --config configs/company.local.yaml \
  --runs-dir runs \
  --baseline-record "/private/local/baselines/task-baseline.json"
```

The command measures only local `inspect`, `discover`, `validate`, `monitor`, and `report` stages with `time.perf_counter`. It reads the baseline without rewriting it and writes `benchmark_record.json` below the ignored local `runs/<run_id>/` directory. Review candidate rules manually; retain aggregate review outcomes and reason categories only, never rule expressions, samples, real column names, or source paths in Git.

After the manual review, complete the locally retained record with the measured aggregate counts. Validate it against `schemas/benchmark-record.schema.json` and the local `BenchmarkRecord` model before treating it as evidence. Do not run this procedure against synthetic records to create company claims.

## 5. Generate internal evidence only after three real tasks

Only after at least three distinct task IDs have complete records with measured manual baselines and agent timings, generate the private draft:

```bash
.venv/bin/riskprobe resume-evidence \
  --records-dir runs \
  --output reports/internal/resume_evidence.md
```

The command must fail for fewer than three completed distinct tasks or any missing manual baseline. Its output is an internal draft with source run IDs; it is not a public claim, a completed review, or a backtest result. Do not publish or commit it.

## 6. Final local safety check

Before leaving the environment, run:

```bash
git status --short
git check-ignore -v configs/company.local.yaml sample.parquet runs/demo/result.json reports/internal/resume_evidence.md
git ls-files '*.parquet' '*.csv' 'configs/company.local.yaml' 'runs/*' 'reports/internal/*'
```

No data, local configuration, run result, baseline, or internal resume evidence may be staged or tracked. This repository currently has no real company trial; this runbook defines the prerequisite procedure rather than reporting a result.
