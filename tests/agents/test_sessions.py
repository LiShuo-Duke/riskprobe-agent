import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from riskprobe.agents.sessions import (
    SessionNode,
    SessionNodeKind,
    SessionStore,
    SessionToolCall,
)

_EVIDENCE_A = "a" * 64


def _store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "sessions.sqlite3")


def test_session_tree_supports_parent_child_fork_retry_branch_and_replay(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    root = store.create_session(
        session_id="session-001",
        goal="comprehensive",
        component_versions={"planner": "planner-v1"},
    )
    child = store.append_child(
        root.node_id,
        tool_call=SessionToolCall(
            tool_name="inspect",
            arguments={"dataset_id": "synthetic_demo"},
        ),
        redacted_summary="inspection complete",
        component_versions={"tool_gateway": "gateway-v1"},
    )
    fork = store.fork(
        root.node_id,
        goal="comprehensive",
        redacted_summary="alternate safe branch",
        component_versions={"planner": "planner-v2"},
    )
    retry = store.retry(
        child.node_id,
        evidence_ids=(_EVIDENCE_A,),
        redacted_summary="retry completed",
        component_versions={"reviewer": "reviewer-v1"},
    )

    assert root.kind is SessionNodeKind.ROOT
    assert child.parent_node_id == root.node_id
    assert retry.parent_node_id == child.node_id
    assert retry.retry_of_node_id == child.node_id
    assert retry.kind is SessionNodeKind.RETRY
    assert fork.kind is SessionNodeKind.FORK
    assert fork.branch_id != root.branch_id
    assert store.children(root.node_id) == (child, fork)
    assert store.branch(retry.node_id) == (root, child, retry)
    assert store.replay("session-001") == (root, child, fork, retry)
    assert store.get(retry.node_id) == retry


def test_reopening_store_replays_identical_strict_dtos(tmp_path: Path) -> None:
    path = tmp_path / "sessions.sqlite3"
    store = SessionStore(path)
    root = store.create_session(
        session_id="session-replay",
        goal="comprehensive",
        component_versions={"orchestrator": "orchestrator-v1"},
    )
    store.append_child(
        root.node_id,
        evidence_ids=(_EVIDENCE_A,),
        redacted_summary="aggregate evidence recorded",
        component_versions={"reviewer": "reviewer-v1"},
    )
    expected = store.replay("session-replay")

    replayed = SessionStore(path).replay("session-replay")

    assert replayed == expected
    assert all(isinstance(node, SessionNode) for node in replayed)
    assert [node.sequence for node in replayed] == [1, 2]


def test_session_history_is_append_only_even_via_direct_sql(tmp_path: Path) -> None:
    store = _store(tmp_path)
    root = store.create_session(
        session_id="session-immutable",
        goal="comprehensive",
        component_versions={"planner": "planner-v1"},
    )

    connection = sqlite3.connect(store.path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE session_nodes SET redacted_summary = 'changed' WHERE node_id = ?",
                (root.node_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM session_nodes WHERE node_id = ?", (root.node_id,))
    finally:
        connection.close()

    assert store.get(root.node_id) == root


def test_session_store_calls_privacy_gate_and_rejects_unsafe_values(tmp_path: Path) -> None:
    checked: list[object] = []

    def gate(payload: object) -> None:
        checked.append(payload)
        from riskprobe.privacy import assert_safe_payload

        assert_safe_payload(payload)

    store = SessionStore(tmp_path / "sessions.sqlite3", safe_payload_hook=gate)
    root = store.create_session(
        session_id="session-safe",
        goal="comprehensive",
        component_versions={"planner": "planner-v1"},
    )

    with pytest.raises(ValueError, match="session payload is not safe"):
        store.append_child(
            root.node_id,
            redacted_summary="/private/customer/data.parquet",
            component_versions={"planner": "planner-v1"},
        )
    with pytest.raises(ValidationError):
        SessionToolCall(
            tool_name="inspect",
            arguments={"raw_rows": [{"customer_id": "customer-123456"}]},
        )

    assert checked
    assert store.children(root.node_id) == ()


def test_concurrent_children_get_unique_monotonic_sequences(tmp_path: Path) -> None:
    store = _store(tmp_path)
    root = store.create_session(
        session_id="session-concurrent",
        goal="comprehensive",
        component_versions={"planner": "planner-v1"},
    )

    def append(index: int) -> SessionNode:
        return store.append_child(
            root.node_id,
            redacted_summary=f"safe aggregate child {index}",
            component_versions={"worker": "worker-v1"},
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        children = tuple(executor.map(append, range(8)))

    assert len({node.node_id for node in children}) == 8
    assert sorted(node.sequence for node in children) == list(range(2, 10))
    assert len(store.replay("session-concurrent")) == 9


def test_session_dtos_are_strict_and_forbid_unapproved_storage_fields() -> None:
    with pytest.raises(ValidationError):
        SessionNode.model_validate(
            {
                "session_id": "session-001",
                "node_id": "a" * 64,
                "sequence": 1,
                "kind": SessionNodeKind.ROOT,
                "branch_id": "b" * 64,
                "goal": "comprehensive",
                "component_versions": {"planner": "planner-v1"},
                "raw_data": [{"secret": 1}],
            }
        )
    with pytest.raises(ValidationError):
        SessionToolCall(tool_name="inspect", arguments={"dataset_id": 123})


def test_session_hash_normalizes_evidence_order_before_persistence(tmp_path: Path) -> None:
    store = _store(tmp_path)
    root = store.create_session(
        session_id="session-evidence-order",
        goal="comprehensive",
        component_versions={"planner": "planner-v1"},
    )
    child = store.append_child(
        root.node_id,
        evidence_ids=("b" * 64, "a" * 64),
        redacted_summary="ordered evidence references",
        component_versions={"reviewer": "reviewer-v1"},
    )

    assert child.evidence_ids == ("a" * 64, "b" * 64)
    assert SessionStore(store.path).replay("session-evidence-order")[-1] == child
