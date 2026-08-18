"""Compatibility facade over the transactional RiskProbe execution store."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from riskprobe.execution.models import (
    ArtifactRef,
    NodeCheckpoint,
    NodeStatus,
    RetryPolicy,
    RunBudget,
    RunStatus,
)
from riskprobe.execution.store import ExecutionStore


class RunRuntime:
    """Persist atomic node transitions and verified resumable checkpoints."""

    def __init__(
        self,
        runs_dir: Path,
        run_id: str,
        *,
        retry_policy: RetryPolicy | None = None,
        budget: RunBudget | None = None,
    ) -> None:
        self.store = ExecutionStore(
            runs_dir,
            run_id,
            retry_policy=retry_policy,
            budget=budget,
        )
        self.runs_dir = self.store.runs_dir
        self.run_id = run_id
        self.database_path = self.store.database_path
        # Compatibility attributes for callers that only need the backing path.
        self.state_path = self.database_path
        self.events_path = self.database_path

    def start_node(self, node_id: str, *, input_fingerprint: str) -> int:
        return self.store.start_node(
            node_id, input_fingerprint=input_fingerprint
        )

    def succeed_node(
        self,
        node_id: str,
        *,
        input_fingerprint: str,
        output: Mapping[str, Any],
        artifact_refs: Iterable[ArtifactRef] = (),
        artifacts: Iterable[ArtifactRef] | None = None,
    ) -> NodeCheckpoint:
        if artifacts is not None:
            provided = tuple(artifact_refs)
            if provided:
                raise ValueError("provide artifact_refs or artifacts, not both")
            artifact_refs = artifacts
        return self.store.succeed_node(
            node_id,
            input_fingerprint=input_fingerprint,
            output=output,
            artifact_refs=artifact_refs,
        )

    def fail_node(
        self,
        node_id: str,
        *,
        input_fingerprint: str,
        error_class: str,
    ) -> None:
        self.store.fail_node(
            node_id,
            input_fingerprint=input_fingerprint,
            error_class=error_class,
        )

    def checkpoint(
        self, node_id: str, *, input_fingerprint: str
    ) -> NodeCheckpoint | None:
        return self.store.checkpoint(
            node_id, input_fingerprint=input_fingerprint
        )

    def verified_checkpoint(
        self,
        node_id: str,
        *,
        input_fingerprint: str,
        run_dir: Path,
        expected_artifacts: Mapping[str, str] | None = None,
    ) -> NodeCheckpoint | None:
        return self.store.load_verified_checkpoint(
            node_id,
            input_fingerprint=input_fingerprint,
            run_dir=run_dir,
            expected_artifacts=expected_artifacts,
        )

    load_verified_checkpoint = verified_checkpoint

    def invalidate_from(
        self, node_id: str, downstream: Iterable[str] = ()
    ) -> None:
        self.store.invalidate_from(node_id, downstream)

    def cancel(self, node_id: str | None = None) -> None:
        self.store.cancel(node_id)

    def node_status(self, node_id: str) -> NodeStatus:
        return self.store.node_status(node_id)

    def run_status(self) -> RunStatus:
        return self.store.run_status()

    def events(self) -> list[dict[str, Any]]:
        return self.store.events()

    def trace(self, node_id: str | None = None) -> list[dict[str, Any]]:
        return self.store.trace(node_id)

    def reconcile_published(
        self,
        *,
        node_id: str = "finalize",
        input_fingerprint: str,
        output: Mapping[str, Any],
        artifact_refs: Iterable[ArtifactRef] = (),
    ) -> NodeCheckpoint:
        return self.store.reconcile_published(
            node_id=node_id,
            input_fingerprint=input_fingerprint,
            output=output,
            artifact_refs=artifact_refs,
        )


__all__ = [
    "ArtifactRef",
    "NodeCheckpoint",
    "NodeStatus",
    "RetryPolicy",
    "RunBudget",
    "RunRuntime",
    "RunStatus",
]
