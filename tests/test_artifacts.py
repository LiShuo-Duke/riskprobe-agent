import gc
import hashlib
import json
import math
from pathlib import Path

import pytest

from riskprobe.artifacts import RunStore

_ARTIFACTS = (
    "manifest.json",
    "metadata_report.json",
    "data_profile.json",
    "candidate_rules.parquet",
    "evidence_cards.json",
    "risk_report.md",
)


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"


def _write_complete_run(
    context: object,
    *,
    config: object = "cfg",
    data_fingerprint: str = "data",
    code_version: str = "0.1.0",
) -> None:
    for name in _ARTIFACTS[1:]:
        context.write_text(name, f"content for {name}\n")
    integrity = {}
    for name in _ARTIFACTS[1:]:
        content = (context.run_dir / name).read_bytes()
        integrity[name] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }
    context.write_text(
        "manifest.json",
        _canonical_json(
            {
                "artifact_integrity": integrity,
                "artifacts": list(_ARTIFACTS),
                "code_version": code_version,
                "config_fingerprint": hashlib.sha256(
                    _canonical_json(config).encode("utf-8")
                ).hexdigest(),
                "data_fingerprint": data_fingerprint,
                "dataset_id": None,
                "run_id": context.run_id,
                "time_validation_enabled": None,
            }
        ),
    )


def test_same_inputs_produce_same_run_id(tmp_path) -> None:
    store = RunStore(tmp_path)
    first = store.compute_run_id("cfg", "data", "0.1.0")
    second = store.compute_run_id("cfg", "data", "0.1.0")
    assert first == second


def test_run_id_is_canonical_and_cannot_traverse_runs_directory(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")

    first = store.compute_run_id({"b": 2, "a": 1}, "../../outside", "0.1.0")
    second = store.compute_run_id({"a": 1, "b": 2}, "../../outside", "0.1.0")
    context = store.create({"b": 2, "a": 1}, "../../outside", "0.1.0")

    assert first == second
    assert len(first) == 16
    assert all(character in "0123456789abcdef" for character in first)
    assert context.run_dir.parent == (tmp_path / "runs").resolve()
    assert context.run_dir.name == first


def test_complete_duplicate_run_is_returned_without_overwrite(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    first = store.create("cfg", "data", "0.1.0")
    _write_complete_run(first)
    first.finalize()

    second = store.create("cfg", "data", "0.1.0")

    assert second.is_existing is True
    assert {path.name for path in second.run_dir.iterdir()} == set(_ARTIFACTS)


def test_incomplete_run_is_cleaned_before_rebuild(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    first = store.create("cfg", "data", "0.1.0")
    stale = first.run_dir / "stale.txt"
    stale.write_text("partial", encoding="utf-8")
    del first
    gc.collect()

    rebuilt = store.create("cfg", "data", "0.1.0")

    assert rebuilt.is_existing is False
    assert not stale.exists()
    assert (rebuilt.run_dir / ".incomplete").is_file()


def test_atomic_json_failure_preserves_existing_target_and_removes_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = RunStore(tmp_path / "runs").create("cfg", "data", "0.1.0")
    context.write_json("manifest.json", {"status": "old"})
    original = (context.run_dir / "manifest.json").read_bytes()

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        context.write_json("manifest.json", {"status": "new"})

    assert (context.run_dir / "manifest.json").read_bytes() == original
    assert list(context.run_dir.glob(".*.tmp")) == []


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_json_writer_rejects_non_finite_numbers(tmp_path: Path, value: float) -> None:
    context = RunStore(tmp_path / "runs").create("cfg", "data", "0.1.0")

    with pytest.raises(ValueError):
        context.write_json("evidence_cards.json", {"value": value})

    assert not (context.run_dir / "evidence_cards.json").exists()


def test_finalized_context_rejects_further_writes(tmp_path: Path) -> None:
    context = RunStore(tmp_path / "runs").create("cfg", "data", "0.1.0")
    _write_complete_run(context)
    original = (context.run_dir / "manifest.json").read_bytes()
    context.finalize()

    with pytest.raises(FileExistsError, match="immutable"):
        context.write_json("manifest.json", {"status": "overwritten"})

    assert (context.run_dir / "manifest.json").read_bytes() == original


def test_active_incomplete_run_is_not_deleted_by_second_creator(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    active = store.create("cfg", "data", "0.1.0")
    partial = active.run_dir / "partial.txt"
    partial.write_text("active writer", encoding="utf-8")

    with pytest.raises(RuntimeError, match="active"):
        store.create("cfg", "data", "0.1.0")

    assert partial.read_text(encoding="utf-8") == "active writer"
    active.cleanup()


def test_unmarked_incomplete_directory_is_not_treated_as_complete(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    run_id = store.compute_run_id("cfg", "data", "0.1.0")
    run_dir = store.runs_dir / run_id
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        '{"artifacts":["manifest.json","missing.json"]}\n', encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="not complete"):
        store.create("cfg", "data", "0.1.0")


def test_finalize_rejects_manifest_that_claims_only_itself(tmp_path: Path) -> None:
    context = RunStore(tmp_path / "runs").create("cfg", "data", "0.1.0")
    context.write_text(
        "manifest.json",
        _canonical_json({"artifacts": ["manifest.json"], "artifact_integrity": {}}),
    )

    with pytest.raises(RuntimeError, match="not complete"):
        context.finalize()


def test_reuse_rejects_tampered_artifact(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    context = store.create("cfg", "data", "0.1.0")
    _write_complete_run(context)
    context.finalize()
    report = context.run_dir / "risk_report.md"
    original = report.read_bytes()
    replacement = bytes([original[0] ^ 1]) + original[1:]
    assert len(replacement) == len(original)
    report.write_bytes(replacement)

    with pytest.raises(RuntimeError, match="not complete"):
        store.create("cfg", "data", "0.1.0")


def test_reuse_rejects_manifest_missing_required_artifact(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    context = store.create("cfg", "data", "0.1.0")
    _write_complete_run(context)
    context.finalize()
    (context.run_dir / "risk_report.md").unlink()
    manifest = json.loads((context.run_dir / "manifest.json").read_text())
    manifest["artifacts"].remove("risk_report.md")
    manifest["artifact_integrity"].pop("risk_report.md")
    (context.run_dir / "manifest.json").write_text(
        _canonical_json(manifest), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="not complete"):
        store.create("cfg", "data", "0.1.0")


def test_reuse_rejects_extra_file(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    context = store.create("cfg", "data", "0.1.0")
    _write_complete_run(context)
    context.finalize()
    (context.run_dir / "extra.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(RuntimeError, match="not complete"):
        store.create("cfg", "data", "0.1.0")


def test_reuse_rejects_symlinked_artifact(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    context = store.create("cfg", "data", "0.1.0")
    _write_complete_run(context)
    context.finalize()
    report = context.run_dir / "risk_report.md"
    external = tmp_path / "external.md"
    external.write_bytes(report.read_bytes())
    report.unlink()
    report.symlink_to(external)

    with pytest.raises(RuntimeError, match="not complete"):
        store.create("cfg", "data", "0.1.0")


def test_reuse_rejects_rehashed_artifact_and_canonical_manifest_rewrite(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "runs")
    context = store.create("cfg", "data", "0.1.0")
    _write_complete_run(context)
    context.finalize()

    context.run_dir.chmod(0o755)
    report = context.run_dir / "risk_report.md"
    manifest_path = context.run_dir / "manifest.json"
    report.chmod(0o644)
    manifest_path.chmod(0o644)
    report.write_text("substituted report\n", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    content = report.read_bytes()
    manifest["artifact_integrity"]["risk_report.md"] = {
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
    }
    manifest_path.write_text(_canonical_json(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="not complete"):
        store.create("cfg", "data", "0.1.0")
