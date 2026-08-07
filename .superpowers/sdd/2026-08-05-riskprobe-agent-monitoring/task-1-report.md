# Plan 2 Task 1 Report

## Dependency installation

Installed the single new exact dependency using only the command-level Tsinghua index:

```bash
.venv/bin/python -m pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Result: installation succeeded and `.venv/bin/python -c 'import mcp; print("mcp-ok")'` printed `mcp-ok`. No global pip configuration was changed.

## TDD evidence

- **RED:** Added privacy, aggregate-only, fail-closed selection, deterministic snapshot ID/`created_at`, histogram/quantile, rule-reference, and all-missing/constant boundary tests before creating `riskprobe.monitoring`. `.venv/bin/python -m pytest tests/monitoring/test_reference.py -v` then failed during collection with the expected `ModuleNotFoundError: No module named 'riskprobe.monitoring'`.
- **GREEN:** Implemented the smallest monitoring model and aggregate snapshot surface. The same targeted test command passed: **7 passed**.
- **Regression:** `.venv/bin/python -m pytest -v` passed: **207 passed**. `.venv/bin/ruff check .` passed. `git diff --check` passed before the implementation checkpoint commit.

## Privacy and determinism audit

- `build_reference_snapshot` invokes Plan 1's `config.features.select_columns(...)` with configured role columns, then reads only the resulting declared feature series. It neither selects nor serializes the entity, snapshot, segment, target, undeclared text, embedding, sample-row, or dataset-path columns.
- The test payload explicitly proves the absence of `entity_id`, `user_0001`, an undeclared raw marker, `emb_00`, the configured input path, and raw `bank_north` segment text.
- Plan 1 `profile.segment_counts` is transformed to deterministic `segment_<sha256-prefix>` aggregate keys before model construction; raw segment names are not stored or serialized.
- Quantile edges and histogram counts are calculated from declared numeric feature values only. Rule references retain only rule ID, coverage, test hit bad rate, and lift.
- The reproducibility contract uses the fixed `created_at` value `1970-01-01T00:00:00Z`; no real generation time is captured. It is included in the canonical aggregate payload for a stable SHA-256 `snapshot_id`, so repeated or shuffled equivalent inputs have identical snapshots and identifiers.

## Files

- `pyproject.toml`
- `src/riskprobe/monitoring/__init__.py`
- `src/riskprobe/monitoring/models.py`
- `src/riskprobe/monitoring/reference.py`
- `tests/monitoring/conftest.py`
- `tests/monitoring/test_reference.py`
- `.superpowers/sdd/2026-08-05-riskprobe-agent-monitoring/task-1-report.md`

## Commits

- Implementation checkpoint: `f5cc07b feat: add privacy-safe monitoring snapshots`

## Fix round 1

### TDD evidence

- **RED (HEAD `2ef0287`):** Added keyed-token, no-key fail-closed, non-numeric selected-feature, identifier-boundary, duplicate-rule, non-finite, and empty-frame regressions. The focused command failed as expected: **23 failed, 1 passed**; the new key argument was unsupported and the old static segment hashes remained visible.
- **GREEN:** Added the smallest compatible `segment_token_key: bytes | None` contract. The focused Task 1 suite then passed: **24 passed**.

### Remediations

- Segment aggregation uses a caller-provided HMAC key and a domain-separated message. The key is transient, absent from models/canonical payloads, and never emitted. Without a key, `segment_counts` is `{}` so downstream detection can opt into matching keyed tokens explicitly.
- Selected features now support numeric Polars dtypes only. Boolean, String, Categorical, and Object dtypes raise the stable error `selected feature '<name>' has unsupported dtype; numeric features are required`; categorical monitoring is intentionally deferred to a dedicated model task.
- Snapshot dataset IDs require the opaque `dataset_<lowercase-hex>` form; rule IDs require the existing Plan 1 lowercase hexadecimal digest form. Path, URI, control-character, and entity-style inputs fail at the snapshot boundary. Duplicate rule IDs are rejected.
- Added executable coverage for finite values combined with `NaN`, `+inf`, and `-inf`, plus empty frames. Non-finite numeric values count as missing and never enter histogram bins.

### Verification

- `.venv/bin/python -m pytest tests/monitoring/test_reference.py -v`: **24 passed**
- `.venv/bin/python -m pytest -v`: **224 passed**
- `.venv/bin/ruff check .`: passed
- `git diff --check`: passed
