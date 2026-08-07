# Plan 2 Task 1 report

## TDD evidence

- **RED:** `HEAD` contains no `src/riskprobe/monitoring/reference.py`, so the requested snapshot-builder import is unavailable in the baseline. The staged task state already contained the new privacy, aggregation, determinism, and boundary tests together with their implementation; it was preserved rather than destructively reverting user-staged work solely to re-run the import failure.
- **GREEN:** `.venv/bin/python -m pytest tests/monitoring/test_reference.py -v` passed all 7 tests. They cover aggregate-only serialization, exclusion of entity values, paths, undeclared text, and raw segment names; deterministic IDs and timestamp; fail-closed feature selection; quantile histograms; rule aggregates; all-missing and constant features; and strict immutable alerts.

## Dependency installation

Ran `.venv/bin/python -m pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple`. The command used only the requested command-level Tsinghua mirror and made no global pip configuration changes. `.venv/bin/python -c 'import mcp; print("mcp-ok")'` printed `mcp-ok`; `pyproject.toml` adds only the exact core dependency `mcp==1.13.0`.

## Privacy and reproducibility audit

`build_reference_snapshot` selects features exclusively through Plan 1's fail-closed `config.features.select_columns` logic. It reads only selected feature series for aggregate rates, quantile edges, and histogram counts; it does not access entity values, preserve sample rows, or serialize dataset paths. Profile segment counts are converted to stable SHA-256 namespace-derived `segment_<digest>` keys before serialization, so original segment names are absent from the reference payload.

The snapshot uses the fixed `created_at` value `1970-01-01T00:00:00Z` to make the complete snapshot equality contract reproducible. Real generation time is intentionally neither serialized nor included in the snapshot ID/comparison input; the ID is a SHA-256 digest of canonical aggregate data only.

## Validation

- Target suite: 7 passed.
- Full suite: 207 passed.
- Ruff: passed.
- `git diff --check` and `git diff --cached --check`: passed.

## Files and checkpoint

- `pyproject.toml` — exact MCP dependency.
- `src/riskprobe/monitoring/__init__.py` — monitoring public exports.
- `src/riskprobe/monitoring/models.py` — immutable monitoring models.
- `src/riskprobe/monitoring/reference.py` — privacy-safe deterministic aggregate snapshot builder.
- `tests/monitoring/conftest.py` — shared synthetic aggregate fixture.
- `tests/monitoring/test_reference.py` — snapshot contracts and boundaries.
- `.superpowers/sdd/2026-08-05-riskprobe-agent-monitoring/task-1-report.md` — this record.

Local checkpoint commit: `feat: add privacy-safe monitoring snapshots`.
