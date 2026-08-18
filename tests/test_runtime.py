import os
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from riskprobe.execution.models import ArtifactRef, RetryPolicy, RunBudget
from riskprobe.runtime import NodeStatus, RunRuntime

_RUN_ID = "0123456789abcdef"


def _artifact(tmp_path: Path, name: str = "result.json", content: bytes = b"{}\n") -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def test_runtime_persists_node_checkpoint_refs_and_events(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    reference = ArtifactRef.from_path(artifact, "result-v1")
    runtime = RunRuntime(tmp_path, _RUN_ID)

    runtime.start_node("profile", input_fingerprint="input-a")
    runtime.succeed_node(
        "profile",
        input_fingerprint="input-a",
        output={"artifact": "result.json"},
        artifact_refs=(reference,),
    )

    restored = RunRuntime(tmp_path, _RUN_ID)
    checkpoint = restored.checkpoint("profile", input_fingerprint="input-a")
    verified = restored.verified_checkpoint(
        "profile",
        input_fingerprint="input-a",
        run_dir=tmp_path,
        expected_artifacts={"result.json": "result-v1"},
    )

    assert checkpoint is not None
    assert checkpoint.status is NodeStatus.SUCCEEDED
    assert checkpoint.output == {"artifact": "result.json"}
    assert checkpoint.artifact_refs == (reference,)
    assert verified == checkpoint
    assert restored.node_status("profile") is NodeStatus.SUCCEEDED

    events = restored.events()
    assert [event["event_type"] for event in events] == [
        "node_started",
        "node_succeeded",
    ]
    assert all(event["node_id"] == "profile" for event in events)
    assert runtime.database_path.stat().st_mode & 0o777 == 0o600


def test_runtime_does_not_reuse_checkpoint_for_different_input(tmp_path: Path) -> None:
    runtime = RunRuntime(tmp_path, _RUN_ID)
    runtime.start_node("discover", input_fingerprint="input-a")
    runtime.succeed_node("discover", input_fingerprint="input-a", output={"count": 2})

    assert runtime.checkpoint("discover", input_fingerprint="input-b") is None


def test_runtime_rejects_invalid_node_transition(tmp_path: Path) -> None:
    runtime = RunRuntime(tmp_path, _RUN_ID)

    with pytest.raises(ValueError, match="must be running"):
        runtime.succeed_node("validate", input_fingerprint="input-a", output={})


def test_verified_checkpoint_rejects_tampered_or_missing_artifact(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, content=b'{"safe":true}\n')
    runtime = RunRuntime(tmp_path, _RUN_ID)
    runtime.start_node("validate", input_fingerprint="input-a")
    runtime.succeed_node(
        "validate",
        input_fingerprint="input-a",
        output={},
        artifact_refs=(ArtifactRef.from_path(artifact, "evidence-v1"),),
    )

    artifact.write_bytes(b'{"safe":fals}\n')
    assert runtime.verified_checkpoint(
        "validate",
        input_fingerprint="input-a",
        run_dir=tmp_path,
        expected_artifacts={"result.json": "evidence-v1"},
    ) is None

    artifact.unlink()
    assert runtime.verified_checkpoint(
        "validate",
        input_fingerprint="input-a",
        run_dir=tmp_path,
        expected_artifacts={"result.json": "evidence-v1"},
    ) is None


def test_verified_checkpoint_rejects_symlinked_artifact(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    runtime = RunRuntime(tmp_path, _RUN_ID)
    runtime.start_node("report", input_fingerprint="input-a")
    runtime.succeed_node(
        "report",
        input_fingerprint="input-a",
        output={},
        artifact_refs=(ArtifactRef.from_path(artifact, "report-v1"),),
    )
    external = tmp_path / "external.json"
    external.write_bytes(artifact.read_bytes())
    artifact.unlink()
    artifact.symlink_to(external)

    assert runtime.verified_checkpoint(
        "report",
        input_fingerprint="input-a",
        run_dir=tmp_path,
        expected_artifacts={"result.json": "report-v1"},
    ) is None


def test_artifact_ref_requires_plain_name_regular_file_and_schema(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    reference = ArtifactRef.from_path(artifact, "schema-v1")

    assert reference.filename == "result.json"
    assert reference.size == len(b"{}\n")
    assert len(reference.sha256) == 64

    external = tmp_path / "external"
    external.write_bytes(b"content")
    link = tmp_path / "linked"
    link.symlink_to(external)
    with pytest.raises(ValueError, match="regular non-symlink"):
        ArtifactRef.from_path(link, "schema-v1")
    with pytest.raises(ValueError, match="schema_version"):
        ArtifactRef.from_path(artifact, "")


def test_new_runtime_interrupts_stale_running_node_and_starts_next_attempt(
    tmp_path: Path,
) -> None:
    first = RunRuntime(tmp_path, _RUN_ID)
    assert first.start_node("discover", input_fingerprint="input-a") == 1

    recovered = RunRuntime(tmp_path, _RUN_ID)

    assert recovered.node_status("discover") is NodeStatus.INTERRUPTED
    assert recovered.start_node("discover", input_fingerprint="input-a") == 2
    assert [event["event_type"] for event in recovered.trace("discover")] == [
        "node_started",
        "node_interrupted",
        "node_started",
    ]


def test_transition_rolls_back_event_when_node_update_fails(tmp_path: Path) -> None:
    runtime = RunRuntime(tmp_path, _RUN_ID)
    with sqlite3.connect(runtime.database_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_node_insert
            BEFORE INSERT ON nodes
            BEGIN
                SELECT RAISE(ABORT, 'simulated node failure');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="simulated node failure"):
        runtime.start_node("profile", input_fingerprint="input-a")

    assert runtime.events() == []
    assert runtime.node_status("profile") is NodeStatus.PENDING


def test_invalidate_from_invalidates_current_and_downstream_in_one_trace(
    tmp_path: Path,
) -> None:
    runtime = RunRuntime(tmp_path, _RUN_ID)
    for node_id in ("discover", "validate", "report"):
        runtime.start_node(node_id, input_fingerprint=f"input-{node_id}")
        runtime.succeed_node(node_id, input_fingerprint=f"input-{node_id}", output={})

    runtime.invalidate_from("discover", ("validate", "report"))

    assert runtime.node_status("discover") is NodeStatus.INVALIDATED
    assert runtime.node_status("validate") is NodeStatus.INVALIDATED
    assert runtime.node_status("report") is NodeStatus.INVALIDATED
    assert runtime.checkpoint("validate", input_fingerprint="input-validate") is None
    assert runtime.start_node("discover", input_fingerprint="input-discover") == 2
    invalidated = [
        event["node_id"]
        for event in runtime.trace()
        if event["event_type"] == "node_invalidated"
    ]
    assert invalidated == ["discover", "validate", "report"]


def test_retry_policy_is_finite_and_records_backoff_without_sleep(tmp_path: Path) -> None:
    runtime = RunRuntime(
        tmp_path,
        _RUN_ID,
        retry_policy=RetryPolicy(max_attempts=2, backoff_seconds=(0.25,)),
    )
    runtime.start_node("report", input_fingerprint="input-a")
    runtime.fail_node(
        "report", input_fingerprint="input-a", error_class="RuntimeError"
    )

    assert runtime.start_node("report", input_fingerprint="input-a") == 2
    retry_start = runtime.trace("report")[-1]
    assert retry_start["metadata"] == {"backoff_seconds": 0.25}
    runtime.fail_node(
        "report", input_fingerprint="input-a", error_class="RuntimeError"
    )

    with pytest.raises(RuntimeError, match="retry attempts"):
        runtime.start_node("report", input_fingerprint="input-a")


def test_run_budget_limits_nodes_attempts_and_deadline(tmp_path: Path) -> None:
    runtime = RunRuntime(
        tmp_path,
        _RUN_ID,
        budget=RunBudget(max_nodes=1, max_attempts=2),
    )
    runtime.start_node("profile", input_fingerprint="input-a")
    runtime.fail_node(
        "profile", input_fingerprint="input-a", error_class="RuntimeError"
    )
    runtime.start_node("profile", input_fingerprint="input-a")

    with pytest.raises(RuntimeError, match="attempt budget"):
        runtime.start_node("profile", input_fingerprint="input-a")
    with pytest.raises(RuntimeError, match="node budget"):
        runtime.start_node("discover", input_fingerprint="input-b")

    expired = RunRuntime(
        tmp_path,
        "fedcba9876543210",
        budget=RunBudget(deadline=datetime.now(timezone.utc) - timedelta(seconds=1)),
    )
    with pytest.raises(RuntimeError, match="deadline"):
        expired.start_node("profile", input_fingerprint="input-a")


def test_cancel_and_trace_are_persistent(tmp_path: Path) -> None:
    runtime = RunRuntime(tmp_path, _RUN_ID)
    runtime.start_node("validate", input_fingerprint="input-a")

    runtime.cancel("validate")

    assert runtime.node_status("validate") is NodeStatus.CANCELLED
    assert runtime.trace("validate")[-1]["event_type"] == "node_cancelled"
    restored = RunRuntime(tmp_path, _RUN_ID)
    assert restored.node_status("validate") is NodeStatus.CANCELLED


def test_runtime_rejects_symlink_corrupt_owner_mismatch_and_broad_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    symlink_root = tmp_path / "symlink"
    symlink_root.mkdir()
    external = tmp_path / "external.sqlite3"
    external.write_bytes(b"not sqlite")
    (symlink_root / f".{_RUN_ID}.runtime.sqlite3").symlink_to(external)
    with pytest.raises(RuntimeError, match="secure regular file"):
        RunRuntime(symlink_root, _RUN_ID)

    corrupt_root = tmp_path / "corrupt"
    corrupt_root.mkdir()
    corrupt = corrupt_root / f".{_RUN_ID}.runtime.sqlite3"
    corrupt.write_bytes(b"not sqlite")
    corrupt.chmod(0o600)
    with pytest.raises(RuntimeError, match="invalid"):
        RunRuntime(corrupt_root, _RUN_ID)

    permission_root = tmp_path / "permission"
    runtime = RunRuntime(permission_root, _RUN_ID)
    runtime.database_path.chmod(0o644)
    with pytest.raises(RuntimeError, match="permissions"):
        RunRuntime(permission_root, _RUN_ID)

    runtime.database_path.chmod(0o600)
    actual_uid = os.geteuid()
    monkeypatch.setattr(os, "geteuid", lambda: actual_uid + 1)
    with pytest.raises(RuntimeError, match="owner"):
        RunRuntime(permission_root, _RUN_ID)


def test_runtime_database_is_bound_to_run_id(tmp_path: Path) -> None:
    original = RunRuntime(tmp_path, _RUN_ID)
    original.start_node("profile", input_fingerprint="input-a")
    other_run_id = "fedcba9876543210"
    copied = tmp_path / f".{other_run_id}.runtime.sqlite3"
    shutil.copyfile(original.database_path, copied)
    copied.chmod(0o600)

    with pytest.raises(RuntimeError, match="different run_id"):
        RunRuntime(tmp_path, other_run_id)
