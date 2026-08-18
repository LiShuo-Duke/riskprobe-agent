from __future__ import annotations

import hashlib
import hmac
import json
import math
import socket
import stat
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from riskprobe.rag import (
    BuildResult,
    Citation,
    IndexIntegrityError,
    LocalCitationIndex,
    ProviderSafeSummary,
    QueryResult,
    UnsafeContentError,
    UnsafeIndexRequestError,
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _rag_id(kind: str, value: str, key: bytes) -> str:
    digest = hmac.new(
        key,
        b"rag-identifier\0" + kind.encode("ascii") + b"\0" + value.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()[:24]
    return f"{kind}-{digest}"


def _forbidden_test_directory(name: str) -> bool:
    normalized = name.casefold()
    visible = normalized.lstrip("._-")
    return (
        normalized == ".git"
        or "cache" in normalized
        or visible.startswith(
            ("artifact", "data", "dataset", "output", "parquet", "raw", "run")
        )
    )


def _write_root_manifest(root: Path) -> None:
    documents: list[dict[str, str]] = []
    if root.exists():
        for path in sorted(root.rglob("*"), key=lambda candidate: candidate.as_posix()):
            if path.is_symlink() or not path.is_file() or path.suffix.casefold() not in {
                ".md",
                ".txt",
            }:
                continue
            relative = path.relative_to(root)
            if any(_forbidden_test_directory(part) for part in relative.parts[:-1]):
                continue
            documents.append(
                {
                    "content_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "path": relative.as_posix(),
                    "privacy_class": "provider_safe",
                }
            )
    (root / ".riskprobe-rag-manifest.json").write_bytes(
        _canonical_bytes({"documents": documents, "format_version": 1})
    )


class _AttestingLocalCitationIndex(LocalCitationIndex):
    """Existing behavioral tests explicitly attest their temporary safe fixtures."""

    def __init__(self, *, index_path: Path, roots: dict[str, Path]) -> None:
        self._test_roots = dict(roots)
        super().__init__(index_path=index_path, roots=roots)

    def build(
        self,
        *,
        root_id: str,
        scope_id: str,
        provider_summaries: tuple[ProviderSafeSummary | dict[str, object], ...] = (),
    ) -> BuildResult:
        root = self._test_roots.get(root_id)
        if root is not None:
            _write_root_manifest(root)
        return super().build(
            root_id=root_id,
            scope_id=scope_id,
            provider_summaries=provider_summaries,
        )


def _v4_semantic_body(payload: dict[str, object]) -> dict[str, object]:
    scopes = payload["scopes"]
    assert isinstance(scopes, dict)
    semantic_scopes: dict[str, object] = {}
    for scope_id, raw_scope in scopes.items():
        assert isinstance(scope_id, str)
        assert isinstance(raw_scope, dict)
        documents = raw_scope["documents"]
        assert isinstance(documents, list)
        semantic_documents: list[dict[str, object]] = []
        for raw_document in documents:
            assert isinstance(raw_document, dict)
            semantic_documents.append(
                {
                    "citation_id": raw_document["citation_id"],
                    "content_hash": raw_document["content_hash"],
                    "document_id": raw_document["document_id"],
                }
            )
        semantic_scopes[scope_id] = {
            "documents": semantic_documents,
            "root_id": raw_scope["root_id"],
            "scope_id": raw_scope["scope_id"],
        }
    return {"format_version": payload["format_version"], "scopes": semantic_scopes}


def _reseal_v4(payload: dict[str, object], key: bytes) -> None:
    scopes = payload["scopes"]
    assert isinstance(scopes, dict)
    for raw_scope in scopes.values():
        assert isinstance(raw_scope, dict)
        core = {
            "documents": raw_scope["documents"],
            "root_id": raw_scope["root_id"],
            "scope_id": raw_scope["scope_id"],
        }
        raw_scope["seal"] = hmac.new(
            key,
            b"scope\0" + _canonical_bytes(core),
            hashlib.sha256,
        ).hexdigest()
    payload["index_hash"] = hashlib.sha256(
        _canonical_bytes(_v4_semantic_body(payload))
    ).hexdigest()
    authenticated = {
        "format_version": payload["format_version"],
        "index_hash": payload["index_hash"],
        "key_id": payload["key_id"],
        "scopes": scopes,
    }
    payload["seal"] = hmac.new(
        key,
        b"index\0" + _canonical_bytes(authenticated),
        hashlib.sha256,
    ).hexdigest()


def _safe_index(tmp_path: Path) -> tuple[LocalCitationIndex, Path, Path]:
    root = tmp_path / "policies"
    root.mkdir()
    index_path = tmp_path / "citation-index.json"
    return (
        _AttestingLocalCitationIndex(index_path=index_path, roots={"policy-root": root}),
        root,
        index_path,
    )


def test_public_dtos_are_strict_frozen_and_extra_forbid() -> None:
    citation = Citation(
        rank=1,
        citation_id="a" * 64,
        document_id="b" * 64,
        content_hash="c" * 64,
        score=0.5,
    )
    scope_id = f"scope-{'d' * 24}"
    result = QueryResult(scope_id=scope_id, citations=(citation,))
    build = BuildResult(scope_id=scope_id, document_count=1, index_hash="d" * 64)

    assert result.citations == (citation,)
    assert build.document_count == 1

    with pytest.raises(ValidationError):
        QueryResult.model_validate(
            {
                "scope_id": "scope-a",
                "citations": [citation.model_dump(mode="json")],
            }
        )
    with pytest.raises(ValidationError):
        Citation.model_validate({**citation.model_dump(mode="json"), "extra": True})
    with pytest.raises(ValidationError):
        build.scope_id = "scope-b"  # type: ignore[misc]


def test_provider_summary_is_strict_content_addressed_and_non_leaking() -> None:
    first = ProviderSafeSummary(text="Aggregate approval rate remains stable.", aggregate_count=7)
    second = ProviderSafeSummary(text="Aggregate approval rate remains stable.", aggregate_count=7)

    assert first == second
    assert len(first.content_hash) == 64

    with pytest.raises(ValidationError):
        ProviderSafeSummary.model_validate(
            {
                "text": "Aggregate approval rate remains stable.",
                "aggregate_count": "7",
            }
        )
    with pytest.raises(ValidationError):
        ProviderSafeSummary.model_validate(
            {
                "text": "Aggregate approval rate remains stable.",
                "aggregate_count": 7,
                "unknown": True,
            }
        )

    marker = "/private/provider/customer.csv"
    with pytest.raises(ValidationError) as exc_info:
        ProviderSafeSummary(text=f"Aggregate source is {marker}.", aggregate_count=7)
    assert marker not in str(exc_info.value)


def test_build_is_persistent_content_addressed_and_deterministic(tmp_path: Path) -> None:
    index, root, index_path = _safe_index(tmp_path)
    first_text = "Liquidity risk policy requires independent controls."
    second_text = "Capital policy requires aggregate reserve monitoring."
    (root / "liquidity.md").write_text(first_text, encoding="utf-8")
    (root / "capital.txt").write_text(second_text, encoding="utf-8")

    first_build = index.build(root_id="policy-root", scope_id="scope-a")
    first_query = index.query(
        scope_id="scope-a",
        query_id="query-001",
        query_text="liquidity policy controls",
    )
    second_build = index.build(root_id="policy-root", scope_id="scope-a")
    reopened_query = _AttestingLocalCitationIndex(
        index_path=index_path,
        roots={"policy-root": root},
    ).query(
        scope_id="scope-a",
        query_id="query-001",
        query_text="liquidity policy controls",
    )

    assert first_build == second_build
    key = index_path.with_name(f"{index_path.name}.key").read_bytes()
    assert first_build.scope_id == _rag_id("scope", "scope-a", key)
    assert first_build.document_count == 2
    assert len(first_build.index_hash) == 64
    assert first_query == reopened_query
    assert first_query.citations
    assert [item.rank for item in first_query.citations] == list(
        range(1, len(first_query.citations) + 1)
    )
    assert all(math.isfinite(item.score) and item.score >= 0 for item in first_query.citations)
    assert stat.S_IMODE(index_path.stat().st_mode) == 0o600

    stored = index_path.read_bytes()
    assert str(root).encode() not in stored
    assert b"liquidity.md" not in stored
    assert b"capital.txt" not in stored
    assert first_text.encode() not in stored
    assert second_text.encode() not in stored
    assert b"query-001" not in stored
    assert b'"text"' not in _canonical_bytes(first_query.model_dump(mode="json"))
    assert b'"path"' not in _canonical_bytes(first_query.model_dump(mode="json"))


def test_scope_isolation_prevents_cross_scope_retrieval(tmp_path: Path) -> None:
    root_a = tmp_path / "root-a"
    root_b = tmp_path / "root-b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "a.txt").write_text(
        "Alpha capital reserve guidance.",
        encoding="utf-8",
    )
    (root_b / "b.txt").write_text(
        "Beta liquidity threshold guidance.",
        encoding="utf-8",
    )
    index = _AttestingLocalCitationIndex(
        index_path=tmp_path / "index.json",
        roots={"root-a": root_a, "root-b": root_b},
    )

    index.build(root_id="root-a", scope_id="scope-a")
    before = index.query(
        scope_id="scope-a",
        query_id="query-alpha",
        query_text="alpha capital",
    )
    index.build(root_id="root-b", scope_id="scope-b")
    after = index.query(
        scope_id="scope-a",
        query_id="query-alpha",
        query_text="alpha capital",
    )
    cross_scope = index.query(
        scope_id="scope-b",
        query_id="query-alpha",
        query_text="alpha capital",
    )

    assert before == after
    assert before.citations
    assert cross_scope.citations == ()


def test_citations_sort_by_score_then_stable_id_for_ties(tmp_path: Path) -> None:
    index, root, _ = _safe_index(tmp_path)
    (root / "one.txt").write_text("Shared risk signal.", encoding="utf-8")
    (root / "two.txt").write_text("Shared risk signal!", encoding="utf-8")
    index.build(root_id="policy-root", scope_id="scope-a")

    result = index.query(
        scope_id="scope-a",
        query_id="query-tie",
        query_text="shared risk signal",
        limit=10,
    )

    assert len(result.citations) == 2
    assert result.citations[0].score == result.citations[1].score
    assert [item.citation_id for item in result.citations] == sorted(
        item.citation_id for item in result.citations
    )
    assert [item.rank for item in result.citations] == [1, 2]


def test_empty_vocabulary_and_no_match_return_no_citations(tmp_path: Path) -> None:
    index, root, _ = _safe_index(tmp_path)
    (root / "punctuation.txt").write_text("... !!!", encoding="utf-8")
    index.build(root_id="policy-root", scope_id="scope-empty")

    empty = index.query(
        scope_id="scope-empty",
        query_id="query-empty",
        query_text="capital",
    )

    (root / "punctuation.txt").write_text("Aggregate policy guidance.", encoding="utf-8")
    index.build(root_id="policy-root", scope_id="scope-normal")
    no_match = index.query(
        scope_id="scope-normal",
        query_id="query-none",
        query_text="volcano astronomy",
    )

    assert empty.citations == ()
    assert no_match.citations == ()


def test_rejects_traversal_unregistered_roots_and_unsafe_request_ids(
    tmp_path: Path,
) -> None:
    index, root, _ = _safe_index(tmp_path)
    (root / "safe.txt").write_text("Aggregate risk policy.", encoding="utf-8")

    for root_id in ("../policy-root", "missing-root"):
        with pytest.raises(UnsafeIndexRequestError) as exc_info:
            index.build(root_id=root_id, scope_id="scope-a")
        assert str(exc_info.value) == "citation index request is not safe"
        assert root_id not in str(exc_info.value)

    with pytest.raises(UnsafeIndexRequestError, match="citation index request is not safe"):
        index.build(root_id="policy-root", scope_id="../scope-a")
    with pytest.raises(UnsafeIndexRequestError, match="citation index request is not safe"):
        index.query(
            scope_id="scope-a",
            query_id="../query-a",
            query_text="aggregate policy",
        )


@pytest.mark.parametrize("link_kind", ["root", "ancestor", "file", "directory"])
def test_rejects_root_ancestor_file_and_directory_symlinks(
    tmp_path: Path,
    link_kind: str,
) -> None:
    if link_kind == "root":
        real_root = tmp_path / "real-root"
        real_root.mkdir()
        (real_root / "safe.txt").write_text("Aggregate risk policy.", encoding="utf-8")
        root = tmp_path / "linked-root"
        root.symlink_to(real_root, target_is_directory=True)
    elif link_kind == "ancestor":
        real_parent = tmp_path / "real-parent"
        root_leaf = real_parent / "root"
        root_leaf.mkdir(parents=True)
        (root_leaf / "safe.txt").write_text("Aggregate risk policy.", encoding="utf-8")
        linked_parent = tmp_path / "linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        root = linked_parent / "root"
    else:
        root = tmp_path / "root"
        root.mkdir()
        (root / "safe.txt").write_text("Aggregate risk policy.", encoding="utf-8")
        if link_kind == "file":
            outside = tmp_path / "outside.txt"
            outside.write_text("Outside aggregate text.", encoding="utf-8")
            (root / "linked.txt").symlink_to(outside)
        else:
            outside = tmp_path / "outside-dir"
            outside.mkdir()
            (outside / "outside.txt").write_text("Outside aggregate text.", encoding="utf-8")
            (root / "linked-dir").symlink_to(outside, target_is_directory=True)

    index_path = tmp_path / "index.json"

    with pytest.raises(UnsafeIndexRequestError) as exc_info:
        index = _AttestingLocalCitationIndex(
            index_path=index_path,
            roots={"safe-root": root},
        )
        index.build(root_id="safe-root", scope_id="scope-a")
    assert str(exc_info.value) == "citation index request is not safe"
    assert str(root) not in str(exc_info.value)
    assert not index_path.exists()


def test_forbidden_directories_and_non_text_files_are_never_indexed(tmp_path: Path) -> None:
    index, root, index_path = _safe_index(tmp_path)
    (root / "safe.txt").write_text("Aggregate risk policy guidance.", encoding="utf-8")
    runs = root / "runs"
    artifacts = root / "artifacts"
    data = root / "data"
    runs.mkdir()
    artifacts.mkdir()
    data.mkdir()
    planted = {
        "RUNS-SECRET-4831": runs / "run.txt",
        "ARTIFACT-SECRET-4832": artifacts / "artifact.md",
        "DATA-SECRET-4833": data / "data.txt",
        "PARQUET-SECRET-4834": root / "private.parquet",
        "CSV-SECRET-4835": root / "private.csv",
        "BINARY-SECRET-4836": root / "private.bin",
    }
    for marker, path in planted.items():
        path.write_bytes(marker.encode("utf-8"))

    result = index.build(root_id="policy-root", scope_id="scope-a")

    assert result.document_count == 1
    stored = index_path.read_bytes()
    for marker, path in planted.items():
        assert marker.encode() not in stored
        assert path.name.encode() not in stored


def test_invalid_utf8_in_allowed_extension_fails_closed(tmp_path: Path) -> None:
    index, root, index_path = _safe_index(tmp_path)
    (root / "binary.txt").write_bytes(b"\xff\xfe\x00PRIVATE")

    with pytest.raises(UnsafeContentError) as exc_info:
        index.build(root_id="policy-root", scope_id="scope-a")
    assert str(exc_info.value) == "citation content is not safe"
    assert "PRIVATE" not in str(exc_info.value)
    assert not index_path.exists()


@pytest.mark.parametrize(
    ("unsafe_text", "marker"),
    [
        ("Source is /private/company/customer.csv.", "/private/company/customer.csv"),
        ("Contact owner@example.com for review.", "owner@example.com"),
        ("Review https://internal.example/risk now.", "https://internal.example/risk"),
        ("customer_123456 is above threshold.", "customer_123456"),
        ("segment_label: north_region", "north_region"),
        ("name,count\nalice,1\nbob,2", "alice"),
        ("api_key = sk-live-supersecret", "sk-live-supersecret"),
        ("```python\ndef risk(value):\n    return value\n```", "risk(value)"),
        ("def calculate_risk(value):\n    return value", "calculate_risk"),
    ],
)
def test_sensitive_documents_fail_closed_without_changing_the_index(
    tmp_path: Path,
    unsafe_text: str,
    marker: str,
) -> None:
    index, root, index_path = _safe_index(tmp_path)
    safe = root / "safe.txt"
    safe.write_text("Aggregate risk policy guidance.", encoding="utf-8")
    index.build(root_id="policy-root", scope_id="scope-a")
    original = index_path.read_bytes()
    (root / "unsafe.txt").write_text(unsafe_text, encoding="utf-8")

    with pytest.raises(UnsafeContentError) as exc_info:
        index.build(root_id="policy-root", scope_id="scope-a")

    assert str(exc_info.value) == "citation content is not safe"
    assert marker not in str(exc_info.value)
    assert index_path.read_bytes() == original
    assert marker.encode() not in index_path.read_bytes()


@pytest.mark.parametrize(
    "summary",
    [
        {
            "text": "Aggregate risk remains stable.",
            "aggregate_count": 2,
            "raw_rows": [{"name": "private"}],
        },
        {
            "text": "Aggregate risk remains stable.",
            "aggregate_count": 2,
            "segment_label": "north",
        },
        {
            "text": "Aggregate risk remains stable.",
            "aggregate_count": 2,
            "api_key": "sk-live-private",
        },
    ],
)
def test_provider_summary_mappings_reject_sensitive_fields(
    tmp_path: Path,
    summary: dict[str, object],
) -> None:
    index, _, index_path = _safe_index(tmp_path)

    with pytest.raises(UnsafeContentError) as exc_info:
        index.build(
            root_id="policy-root",
            scope_id="scope-a",
            provider_summaries=(summary,),
        )

    assert str(exc_info.value) == "citation content is not safe"
    assert "private" not in str(exc_info.value).lower()
    assert not index_path.exists()


def test_safe_provider_summary_is_searchable_without_persisting_plaintext(
    tmp_path: Path,
) -> None:
    index, _, index_path = _safe_index(tmp_path)
    summary_text = "Aggregate delinquency rate status is improved."
    summary = {
        "operation": "aggregate_status",
        "metric_code": "delinquency_rate",
        "status_code": "improved",
        "aggregate_count": 40,
    }

    build = index.build(
        root_id="policy-root",
        scope_id="scope-a",
        provider_summaries=(summary,),
    )
    result = index.query(
        scope_id="scope-a",
        query_id="query-provider",
        query_text="delinquency controls",
    )

    assert build.document_count == 1
    assert result.citations
    assert summary_text.encode() not in index_path.read_bytes()


def test_index_tampering_blocks_query_and_rebuild(tmp_path: Path) -> None:
    index, root, index_path = _safe_index(tmp_path)
    (root / "safe.txt").write_text("Aggregate risk policy guidance.", encoding="utf-8")
    index.build(root_id="policy-root", scope_id="scope-a")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    key = index_path.with_name(f"{index_path.name}.key").read_bytes()
    scope_id = _rag_id("scope", "scope-a", key)
    payload["scopes"][scope_id]["documents"][0]["content_hash"] = "f" * 64
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(IndexIntegrityError) as query_error:
        index.query(
            scope_id="scope-a",
            query_id="query-a",
            query_text="risk policy",
        )
    with pytest.raises(IndexIntegrityError) as build_error:
        index.build(root_id="policy-root", scope_id="scope-a")

    assert str(query_error.value) == "citation index integrity check failed"
    assert str(build_error.value) == "citation index integrity check failed"


def test_manifest_rejects_scope_key_mismatch_even_with_valid_v4_reseal(
    tmp_path: Path,
) -> None:
    index, root, index_path = _safe_index(tmp_path)
    (root / "safe.txt").write_text("Aggregate risk policy guidance.", encoding="utf-8")
    index.build(root_id="policy-root", scope_id="scope-a")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    key = index_path.with_name(f"{index_path.name}.key").read_bytes()
    scope_a = _rag_id("scope", "scope-a", key)
    scope_b = _rag_id("scope", "scope-b", key)
    original_index_hash = payload["index_hash"]
    original_index_seal = payload["seal"]
    scopes = payload["scopes"]
    assert isinstance(scopes, dict)
    scopes[scope_b] = scopes.pop(scope_a)
    _reseal_v4(payload, key)

    moved_scope = scopes[scope_b]
    assert isinstance(moved_scope, dict)
    scope_core = {
        "documents": moved_scope["documents"],
        "root_id": moved_scope["root_id"],
        "scope_id": moved_scope["scope_id"],
    }
    assert moved_scope["scope_id"] == scope_a
    assert moved_scope["seal"] == hmac.new(
        key,
        b"scope\0" + _canonical_bytes(scope_core),
        hashlib.sha256,
    ).hexdigest()
    assert payload["index_hash"] == hashlib.sha256(
        _canonical_bytes(_v4_semantic_body(payload))
    ).hexdigest()
    authenticated = {
        "format_version": payload["format_version"],
        "index_hash": payload["index_hash"],
        "key_id": payload["key_id"],
        "scopes": scopes,
    }
    assert payload["seal"] == hmac.new(
        key,
        b"index\0" + _canonical_bytes(authenticated),
        hashlib.sha256,
    ).hexdigest()
    assert payload["index_hash"] != original_index_hash
    assert payload["seal"] != original_index_seal
    index_path.write_bytes(_canonical_bytes(payload))

    with pytest.raises(IndexIntegrityError, match="citation index integrity check failed"):
        index.query(
            scope_id="scope-b",
            query_id="query-a",
            query_text="risk policy",
        )


def test_build_and_query_are_offline_and_do_not_create_threads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index, root, _ = _safe_index(tmp_path)
    (root / "safe.txt").write_text("Aggregate risk policy guidance.", encoding="utf-8")

    def blocked(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    before = {thread.ident for thread in threading.enumerate()}

    index.build(root_id="policy-root", scope_id="scope-a")
    result = index.query(
        scope_id="scope-a",
        query_id="query-a",
        query_text="risk policy",
    )

    after = {thread.ident for thread in threading.enumerate()}
    assert result.citations
    assert after == before


def test_entity_style_ids_are_rejected_even_when_syntactically_valid(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "safe.txt").write_text("Aggregate risk policy.", encoding="utf-8")

    with pytest.raises(UnsafeIndexRequestError, match="citation index request is not safe"):
        _AttestingLocalCitationIndex(
            index_path=tmp_path / "unsafe-index.json",
            roots={"customer_123456": root},
        )

    index = _AttestingLocalCitationIndex(
        index_path=tmp_path / "index.json",
        roots={"safe-root": root},
    )
    index.build(root_id="safe-root", scope_id="scope-a")
    with pytest.raises(UnsafeIndexRequestError, match="citation index request is not safe"):
        index.query(
            scope_id="scope-a",
            query_id="customer_123456",
            query_text="risk policy",
        )


def test_embedded_single_component_absolute_path_is_rejected(tmp_path: Path) -> None:
    index, root, index_path = _safe_index(tmp_path)
    marker = "/passwd"
    (root / "unsafe.txt").write_text(
        f"Never read {marker} during policy review.",
        encoding="utf-8",
    )

    with pytest.raises(UnsafeContentError) as exc_info:
        index.build(root_id="policy-root", scope_id="scope-a")

    assert str(exc_info.value) == "citation content is not safe"
    assert marker not in str(exc_info.value)
    assert not index_path.exists()


def test_symlink_inside_skipped_directory_still_fails_closed(tmp_path: Path) -> None:
    index, root, index_path = _safe_index(tmp_path)
    (root / "safe.txt").write_text("Aggregate risk policy.", encoding="utf-8")
    runs = root / "runs"
    runs.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("Outside aggregate text.", encoding="utf-8")
    (runs / "linked.txt").symlink_to(outside)

    with pytest.raises(UnsafeIndexRequestError) as exc_info:
        index.build(root_id="policy-root", scope_id="scope-a")

    assert str(exc_info.value) == "citation index request is not safe"
    assert not index_path.exists()
