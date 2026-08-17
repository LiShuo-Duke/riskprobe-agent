import sqlite3
import stat
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path

import pytest
from pydantic import ValidationError

from riskprobe.evidence import (
    EvidenceIntegrityError,
    EvidenceParentError,
    EvidenceRecord,
    EvidenceStore,
    PrivacyClass,
    UnsafeEvidenceError,
)


def _record(
    *,
    run_id: str = "run-001",
    kind: str = "diagnostic.finding",
    payload: dict[str, object] | None = None,
    parent_ids: tuple[str, ...] = (),
) -> EvidenceRecord:
    return EvidenceRecord(
        run_id=run_id,
        kind=kind,
        payload=payload or {"finding_count": 2, "severity": "warning"},
        parent_ids=parent_ids,
        artifact_hashes={"data_profile.json": "a" * 64},
        privacy_class=PrivacyClass.AGGREGATE,
        producer_version="riskprobe-test/1",
    )


def test_append_builds_a_deterministic_idempotent_per_run_chain(tmp_path: Path) -> None:
    path = tmp_path / "evidence.sqlite3"
    store = EvidenceStore(path)
    root = _record()

    root_id = store.append(root)
    child = _record(
        kind="recommendation",
        payload={"action_count": 1, "priority": "high"},
        parent_ids=(root_id,),
    )
    child_id = store.append(child)

    assert store.append(root) == root_id
    assert root_id != child_id
    assert store.get(root_id) == root
    assert store.list_run("run-001") == (root, child)
    assert store.verify_chain("run-001") is True
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    with closing(sqlite3.connect(path)) as connection:
        rows = connection.execute(
            "SELECT sequence, previous_hash FROM evidence_records ORDER BY sequence"
        ).fetchall()
    assert rows[0] == (1, "0" * 64)
    assert rows[1][0] == 2
    assert rows[1][1] != "0" * 64


def test_append_many_allows_ordered_in_batch_parents(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    root = _record()
    root_id = store.content_id(root)
    child = _record(
        kind="recommendation",
        payload={"action_count": 1, "priority": "high"},
        parent_ids=(root_id,),
    )

    evidence_ids = store.append_many((root, child))

    assert evidence_ids == (root_id, store.content_id(child))
    assert store.list_run("run-001") == (root, child)
    assert store.verify_chain("run-001") is True


def test_append_many_rolls_back_when_a_later_parent_is_missing(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    root = _record()
    invalid_child = _record(
        kind="recommendation",
        payload={"action_count": 1, "priority": "high"},
        parent_ids=("f" * 64,),
    )

    with pytest.raises(EvidenceParentError, match="parent evidence is unavailable"):
        store.append_many((root, invalid_child))

    assert store.list_run("run-001") == ()
    assert store.verify_chain("run-001") is True


def test_append_many_rolls_back_when_a_later_payload_is_unsafe(
    tmp_path: Path,
) -> None:
    def reject_later(payload: object) -> None:
        if isinstance(payload, Mapping) and payload.get("reject") is True:
            raise ValueError("private payload detail")

    store = EvidenceStore(
        tmp_path / "evidence.sqlite3",
        safe_payload_hook=reject_later,
    )
    root = _record()
    child = _record(
        kind="recommendation",
        payload={"reject": True},
        parent_ids=(store.content_id(root),),
    )

    with pytest.raises(UnsafeEvidenceError, match="evidence payload is not safe"):
        store.append_many((root, child))

    assert store.list_run("run-001") == ()
    assert store.verify_chain("run-001") is True


def test_append_many_rolls_back_when_a_later_insert_fails(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evidence.sqlite3"
    store = EvidenceStore(path)
    root = _record()
    child = _record(
        kind="recommendation",
        payload={"action_count": 1, "priority": "high"},
        parent_ids=(store.content_id(root),),
    )
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            """
            CREATE TRIGGER reject_recommendation_insert
            BEFORE INSERT ON evidence_records
            WHEN NEW.kind = 'recommendation'
            BEGIN
                SELECT RAISE(ABORT, 'simulated insert failure');
            END;
            """
        )

    with pytest.raises(EvidenceIntegrityError, match="integrity check failed"):
        store.append_many((root, child))

    assert store.list_run("run-001") == ()
    assert store.verify_chain("run-001") is True


def test_content_id_is_public_strict_and_independent_of_mapping_order(tmp_path: Path) -> None:
    first = _record(payload={"severity": "warning", "finding_count": 2})
    second = _record(payload={"finding_count": 2, "severity": "warning"})

    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    content_id = store.content_id(first)

    assert len(content_id) == 64
    assert set(content_id) <= set("0123456789abcdef")
    assert content_id == store.content_id(second)
    assert content_id == store.append(first) == store.append(second)
    assert store.list_run("run-001") == (first,)


def test_append_rejects_missing_and_cross_run_parents(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "evidence.sqlite3")

    with pytest.raises(EvidenceParentError, match="parent evidence is unavailable"):
        store.append(_record(parent_ids=("f" * 64,)))

    parent_id = store.append(_record(run_id="run-001"))
    with pytest.raises(EvidenceParentError, match="parent evidence is unavailable"):
        store.append(_record(run_id="run-002", parent_ids=(parent_id,)))


def test_hash_chain_detects_sqlite_tampering_and_blocks_further_appends(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evidence.sqlite3"
    store = EvidenceStore(path)
    evidence_id = store.append(_record())

    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "UPDATE evidence_records SET payload_json = ? WHERE evidence_id = ?",
            ('{"finding_count":999,"severity":"warning"}', evidence_id),
        )
        connection.commit()

    assert store.verify_chain("run-001") is False
    with pytest.raises(EvidenceIntegrityError, match="integrity check failed"):
        store.get(evidence_id)
    with pytest.raises(EvidenceIntegrityError, match="integrity check failed"):
        store.append(_record(kind="another.finding"))


def test_safe_payload_hook_is_injected_and_errors_do_not_leak_payload(
    tmp_path: Path,
) -> None:
    seen: list[object] = []

    def reject(payload: object) -> None:
        seen.append(payload)
        raise ValueError("secret /private/company.parquet")

    store = EvidenceStore(
        tmp_path / "evidence.sqlite3",
        safe_payload_hook=reject,
    )

    with pytest.raises(UnsafeEvidenceError) as exc_info:
        store.append(_record(payload={"count": 1}))

    assert seen == [{"count": 1}]
    assert str(exc_info.value) == "evidence payload is not safe"
    assert "/private/company.parquet" not in str(exc_info.value)


def test_default_payload_hook_rejects_paths_raw_rows_and_non_json_values(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path / "evidence.sqlite3")

    for payload in (
        {"config_path": "/private/project.yaml"},
        {"raw_rows": [{"customer": "secret"}]},
        {"summary": Path("private.parquet")},
        {"dataset": [{"customer": "private-value"}]},
        {"api_key": "private-value"},
        {"password": "private-value"},
        {"secret": "private-value"},
    ):
        with pytest.raises(UnsafeEvidenceError, match="evidence payload is not safe"):
            store.append(_record(payload=payload))


def test_evidence_record_is_strict_and_forbids_unknown_fields() -> None:
    payload = {
        "run_id": "run-001",
        "kind": "finding",
        "payload": {"count": 1},
        "parent_ids": (),
        "artifact_hashes": {},
        "privacy_class": PrivacyClass.AGGREGATE,
        "producer_version": "test/1",
    }

    with pytest.raises(ValidationError):
        EvidenceRecord.model_validate({**payload, "parent_ids": []})
    with pytest.raises(ValidationError):
        EvidenceRecord.model_validate({**payload, "unexpected": True})
