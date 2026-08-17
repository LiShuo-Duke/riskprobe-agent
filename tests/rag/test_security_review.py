from __future__ import annotations

import hashlib
import hmac
import json
import multiprocessing
import os
import socket
import stat
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import riskprobe.rag.local as local_module
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

_MANIFEST_NAME = ".riskprobe-rag-manifest.json"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _rag_id(kind: str, value: str) -> str:
    digest = hashlib.sha256(f"rag-{kind}\0{value}".encode("ascii")).hexdigest()[:24]
    return f"{kind}-{digest}"


def _write_manifest(root: Path, paths: tuple[str, ...]) -> Path:
    documents = []
    for relative in paths:
        encoded = (root / relative).read_bytes()
        documents.append(
            {
                "content_hash": hashlib.sha256(encoded).hexdigest(),
                "path": relative,
                "privacy_class": "provider_safe",
            }
        )
    manifest = {"documents": documents, "format_version": 1}
    path = root / _MANIFEST_NAME
    path.write_bytes(_canonical_bytes(manifest))
    return path


def _index(
    tmp_path: Path,
    root: Path,
    *,
    root_id: str = "policy-root",
) -> tuple[LocalCitationIndex, Path, Path]:
    index_path = tmp_path / "citation-index.json"
    key_path = tmp_path / "citation-index.json.key"
    return (
        LocalCitationIndex(
            index_path=index_path,
            key_path=key_path,
            roots={root_id: root},
        ),
        index_path,
        key_path,
    )


def _exception_graph(error: BaseException) -> Iterator[BaseException]:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)


def _assert_clean_error(error: BaseException, marker: str) -> None:
    graph = tuple(_exception_graph(error))
    assert graph == (error,)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert marker not in " ".join(str(item) for item in graph)


def _scope_core(scope: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in scope.items() if key != "seal"}


def _semantic_body(payload: dict[str, object]) -> dict[str, object]:
    scopes = payload["scopes"]
    assert isinstance(scopes, dict)
    semantic_scopes: dict[str, object] = {}
    for scope_id, scope in scopes.items():
        assert isinstance(scope, dict)
        documents = scope["documents"]
        assert isinstance(documents, list)
        semantic_scopes[scope_id] = {
            "documents": [
                {
                    "citation_id": document["citation_id"],
                    "content_hash": document["content_hash"],
                    "document_id": document["document_id"],
                }
                for document in documents
            ],
            "root_id": scope["root_id"],
            "scope_id": scope["scope_id"],
        }
    return {"format_version": payload["format_version"], "scopes": semantic_scopes}


def _reseal(payload: dict[str, object], key: bytes) -> None:
    scopes = payload["scopes"]
    assert isinstance(scopes, dict)
    for scope in scopes.values():
        assert isinstance(scope, dict)
        scope["seal"] = hmac.new(
            key,
            b"scope\0" + _canonical_bytes(_scope_core(scope)),
            hashlib.sha256,
        ).hexdigest()
    payload["index_hash"] = hashlib.sha256(
        _canonical_bytes(_semantic_body(payload))
    ).hexdigest()
    authenticated = {
        "format_version": payload["format_version"],
        "index_hash": payload["index_hash"],
        "key_id": payload["key_id"],
        "scopes": payload["scopes"],
    }
    payload["seal"] = hmac.new(
        key,
        b"index\0" + _canonical_bytes(authenticated),
        hashlib.sha256,
    ).hexdigest()


def test_root_manifest_is_mandatory_and_only_exact_attested_files_are_indexed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    approved = root / "approved.txt"
    unlisted = root / "unlisted.txt"
    approved_text = "Approved liquidity controls are independently monitored."
    unlisted_text = "Unlisted volcano astronomy material."
    approved.write_text(approved_text, encoding="utf-8")
    unlisted.write_text(unlisted_text, encoding="utf-8")
    index, index_path, _ = _index(tmp_path, root)

    with pytest.raises(UnsafeIndexRequestError) as exc_info:
        index.build(root_id="policy-root", scope_id="scope-a")
    _assert_clean_error(exc_info.value, str(root))
    assert not index_path.exists()

    _write_manifest(root, ("approved.txt",))
    result = index.build(root_id="policy-root", scope_id="scope-a")
    absent = index.query(
        scope_id="scope-a",
        query_id="query-unlisted",
        query_text="volcano astronomy",
    )

    assert result.document_count == 1
    assert absent.citations == ()
    stored = index_path.read_bytes()
    for marker in (approved_text, unlisted_text, "approved.txt", "unlisted.txt", str(root)):
        assert marker.encode() not in stored


@pytest.mark.parametrize(
    "manifest",
    [
        {"format_version": 1.0, "documents": []},
        {
            "format_version": 1,
            "documents": [
                {
                    "path": "/absolute.txt",
                    "content_hash": "a" * 64,
                    "privacy_class": "provider_safe",
                }
            ],
        },
        {
            "format_version": 1,
            "documents": [
                {
                    "path": "../escape.txt",
                    "content_hash": "a" * 64,
                    "privacy_class": "provider_safe",
                }
            ],
        },
        {
            "format_version": 1,
            "documents": [
                {
                    "path": "nested\\policy.txt",
                    "content_hash": "a" * 64,
                    "privacy_class": "provider_safe",
                }
            ],
        },
        {
            "format_version": 1,
            "documents": [
                {
                    "path": "runs.backup/policy.txt",
                    "content_hash": "a" * 64,
                    "privacy_class": "provider_safe",
                }
            ],
        },
        {
            "format_version": 1,
            "documents": [
                {
                    "path": "raw-data/policy.txt",
                    "content_hash": "a" * 64,
                    "privacy_class": "provider_safe",
                }
            ],
        },
        {
            "format_version": 1,
            "documents": [
                {
                    "path": ".cache/policy.txt",
                    "content_hash": "a" * 64,
                    "privacy_class": "provider_safe",
                }
            ],
        },
        {
            "format_version": 1,
            "documents": [
                {
                    "path": "parquet-copy/policy.txt",
                    "content_hash": "a" * 64,
                    "privacy_class": "provider_safe",
                }
            ],
        },
        {
            "format_version": 1,
            "documents": [
                {
                    "path": "policy.txt",
                    "content_hash": "a" * 64,
                    "privacy_class": "internal",
                }
            ],
        },
        {
            "format_version": 1,
            "documents": [
                {
                    "path": "policy.txt",
                    "content_hash": "a" * 64,
                    "privacy_class": "provider_safe",
                },
                {
                    "path": "policy.txt",
                    "content_hash": "a" * 64,
                    "privacy_class": "provider_safe",
                },
            ],
        },
    ],
)
def test_manifest_schema_and_paths_fail_closed(
    tmp_path: Path,
    manifest: dict[str, object],
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    (root / _MANIFEST_NAME).write_bytes(_canonical_bytes(manifest))
    index, index_path, _ = _index(tmp_path, root)

    with pytest.raises(UnsafeIndexRequestError) as exc_info:
        index.build(root_id="policy-root", scope_id="scope-a")

    assert str(exc_info.value) == "citation index request is not safe"
    _assert_clean_error(exc_info.value, "escape.txt")
    assert not index_path.exists()


def test_manifest_hash_mismatch_fails_without_replacing_existing_index(tmp_path: Path) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    document = root / "policy.txt"
    document.write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, index_path, _ = _index(tmp_path, root)
    index.build(root_id="policy-root", scope_id="scope-a")
    original = index_path.read_bytes()
    document.write_text("Changed aggregate policy controls.", encoding="utf-8")

    with pytest.raises(UnsafeContentError) as exc_info:
        index.build(root_id="policy-root", scope_id="scope-a")

    assert str(exc_info.value) == "citation content is not safe"
    _assert_clean_error(exc_info.value, "Changed")
    assert index_path.read_bytes() == original


@pytest.mark.parametrize(
    "unsafe_text",
    [
        "name,count\nalice,1",
        '{"name":"alice","count":1}',
        '[{"name":"alice"},{"name":"bob"}]',
        r"Source is \\server\share\customers.csv.",
        "Review segment north_region performance.",
        "customer_name = alice",
        "Run `SELECT customer FROM accounts` now.",
        "The password is swordfish.",
        "The API key ordinarysecret rotates quarterly.",
        'Use the quoted key "api_key": "ordinary-secret".',
    ],
)
def test_attestation_keeps_defense_in_depth_scanner_for_reviewer_bypasses(
    tmp_path: Path,
    unsafe_text: str,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "unsafe.txt").write_text(unsafe_text, encoding="utf-8")
    _write_manifest(root, ("unsafe.txt",))
    index, index_path, _ = _index(tmp_path, root)

    with pytest.raises(UnsafeContentError) as exc_info:
        index.build(root_id="policy-root", scope_id="scope-a")

    assert str(exc_info.value) == "citation content is not safe"
    _assert_clean_error(exc_info.value, unsafe_text)
    assert not index_path.exists()


def test_terms_use_external_hmac_key_and_reopen_loads_the_same_key(tmp_path: Path) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text(
        "Liquidity policy controls remain stable.", encoding="utf-8"
    )
    _write_manifest(root, ("policy.txt",))
    index, index_path, key_path = _index(tmp_path, root)
    index.build(root_id="policy-root", scope_id="scope-a")

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    terms = payload["scopes"][_keyed_rag_id(
        "scope", "scope-a", key_path.read_bytes()
    )]["documents"][0]["terms"]
    public_token_hash = hashlib.sha256(b"liquidity").hexdigest()
    assert public_token_hash not in terms
    assert payload["key_id"] == hashlib.sha256(key_path.read_bytes()).hexdigest()
    assert str(key_path) not in index_path.read_text(encoding="utf-8")
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600

    reopened = LocalCitationIndex(
        index_path=index_path,
        key_path=key_path,
        roots={"policy-root": root},
    )
    assert reopened.query(
        scope_id="scope-a",
        query_id="query-a",
        query_text="liquidity controls",
    ).citations

    key_path.unlink()
    with pytest.raises(IndexIntegrityError) as exc_info:
        reopened.query(
            scope_id="scope-a",
            query_id="query-b",
            query_text="liquidity controls",
        )
    assert str(exc_info.value) == "citation index integrity check failed"
    _assert_clean_error(exc_info.value, str(key_path))


def test_tampered_terms_fail_even_when_all_public_hashes_are_recomputed(tmp_path: Path) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Liquidity policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, index_path, key_path = _index(tmp_path, root)
    index.build(root_id="policy-root", scope_id="scope-a")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    document = payload["scopes"][_keyed_rag_id(
        "scope", "scope-a", key_path.read_bytes()
    )]["documents"][0]
    document["terms"] = {"0" * 64: 1}
    payload["index_hash"] = hashlib.sha256(
        _canonical_bytes(_semantic_body(payload))
    ).hexdigest()
    index_path.write_bytes(_canonical_bytes(payload))

    with pytest.raises(IndexIntegrityError, match="citation index integrity check failed"):
        index.query(
            scope_id="scope-a",
            query_id="query-a",
            query_text="liquidity controls",
        )


def test_relative_roots_and_storage_paths_are_fixed_before_cwd_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = tmp_path / "base"
    base.mkdir()
    root = base / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    monkeypatch.chdir(base)
    index = LocalCitationIndex(
        index_path=Path("citation-index.json"),
        key_path=Path("citation-index.json.key"),
        roots={"policy-root": Path("policies")},
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = index.build(root_id="policy-root", scope_id="scope-a")

    assert result.document_count == 1
    assert (base / "citation-index.json").exists()
    assert (base / "citation-index.json.key").exists()
    assert not (elsewhere / "citation-index.json").exists()


def test_secure_storage_rejects_unsafe_parent_and_file_permissions(tmp_path: Path) -> None:
    unsafe_parent = tmp_path / "unsafe-parent"
    unsafe_parent.mkdir()
    unsafe_parent.chmod(0o777)
    try:
        with pytest.raises(UnsafeIndexRequestError, match="citation index request is not safe"):
            LocalCitationIndex(
                index_path=unsafe_parent / "index.json",
                roots={"policy-root": tmp_path},
            )
    finally:
        unsafe_parent.chmod(0o700)

    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, index_path, key_path = _index(tmp_path, root)
    index.build(root_id="policy-root", scope_id="scope-a")
    assert stat.S_IMODE(index_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "citation-index.json.lock").stat().st_mode) == 0o600

    key_path.chmod(0o644)
    with pytest.raises(IndexIntegrityError, match="citation index integrity check failed"):
        index.query(scope_id="scope-a", query_id="query-a", query_text="policy")


def test_symlink_and_special_files_in_forbidden_trees_still_fail_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    skipped = root / "runs.backup"
    skipped.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("Outside aggregate text.", encoding="utf-8")
    (skipped / "linked.txt").symlink_to(outside)
    index, index_path, _ = _index(tmp_path, root)

    with pytest.raises(UnsafeIndexRequestError, match="citation index request is not safe"):
        index.build(root_id="policy-root", scope_id="scope-a")
    assert not index_path.exists()


def test_provider_model_construct_is_dumped_and_fully_revalidated(tmp_path: Path) -> None:
    marker = "/private/provider/customer.csv"
    root = tmp_path / "policies"
    root.mkdir()
    _write_manifest(root, ())
    index, index_path, _ = _index(tmp_path, root)
    forged = ProviderSafeSummary.model_construct(
        text=f"Aggregate source is {marker}.",
        aggregate_count=1,
        content_hash="f" * 64,
    )

    with pytest.raises(UnsafeContentError) as exc_info:
        index.build(
            root_id="policy-root",
            scope_id="scope-a",
            provider_summaries=(forged,),
        )

    _assert_clean_error(exc_info.value, marker)
    assert not index_path.exists()


def test_all_public_rag_dtos_expose_untrusted_safe_validation() -> None:
    summary = ProviderSafeSummary.safe_validate(
        {
            "operation": "aggregate_status",
            "metric_code": "control_coverage",
            "status_code": "stable",
            "aggregate_count": 4,
        }
    )
    citation = Citation.safe_validate(
        {
            "rank": 1,
            "citation_id": "a" * 64,
            "document_id": "b" * 64,
            "content_hash": "c" * 64,
            "score": 0.5,
        }
    )
    opaque_scope = f"scope-{'e' * 24}"
    query = QueryResult.safe_validate(
        {"scope_id": opaque_scope, "citations": (citation.model_dump(mode="python"),)}
    )
    build = BuildResult.safe_validate(
        {"scope_id": opaque_scope, "document_count": 1, "index_hash": "d" * 64}
    )

    assert summary.aggregate_count == 4
    assert query.citations == (citation,)
    assert build.document_count == 1

    marker = "/private/customer.csv"
    with pytest.raises(UnsafeContentError) as exc_info:
        ProviderSafeSummary.safe_validate({"text": marker, "aggregate_count": 1})
    _assert_clean_error(exc_info.value, marker)


@pytest.mark.parametrize("field_value", [1.0, True, 1_000_001])
def test_persisted_term_counts_require_bounded_exact_integers(
    tmp_path: Path,
    field_value: object,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, index_path, key_path = _index(tmp_path, root)
    index.build(root_id="policy-root", scope_id="scope-a")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    terms = payload["scopes"][_keyed_rag_id(
        "scope", "scope-a", key_path.read_bytes()
    )]["documents"][0]["terms"]
    term = next(iter(terms))
    terms[term] = field_value
    _reseal(payload, key_path.read_bytes())
    index_path.write_bytes(_canonical_bytes(payload))

    with pytest.raises(IndexIntegrityError, match="citation index integrity check failed"):
        index.query(scope_id="scope-a", query_id="query-a", query_text="policy")


def test_persisted_format_version_rejects_float_even_with_valid_hmac(tmp_path: Path) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, index_path, key_path = _index(tmp_path, root)
    index.build(root_id="policy-root", scope_id="scope-a")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["format_version"] = 2.0
    _reseal(payload, key_path.read_bytes())
    index_path.write_bytes(_canonical_bytes(payload))

    with pytest.raises(IndexIntegrityError, match="citation index integrity check failed"):
        index.query(scope_id="scope-a", query_id="query-a", query_text="policy")


def test_oversized_encode_fails_before_replacing_old_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, index_path, _ = _index(tmp_path, root)
    index.build(root_id="policy-root", scope_id="scope-a")
    original = index_path.read_bytes()
    monkeypatch.setattr(local_module, "_MAX_INDEX_BYTES", len(original) + 1)

    with pytest.raises(IndexIntegrityError, match="citation index integrity check failed"):
        index.build(root_id="policy-root", scope_id="scope-b")

    assert index_path.read_bytes() == original


@pytest.mark.parametrize("fault_name", ["write", "replace", "fsync"])
def test_precommit_io_faults_preserve_the_previous_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_name: str,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, index_path, _ = _index(tmp_path, root)
    index.build(root_id="policy-root", scope_id="scope-a")
    original = index_path.read_bytes()

    if fault_name == "write":
        monkeypatch.setattr(local_module.os, "write", lambda *_args, **_kwargs: _io_fault())
    elif fault_name == "replace":
        monkeypatch.setattr(local_module.os, "replace", lambda *_args, **_kwargs: _io_fault())
    else:
        original_fsync = local_module.os.fsync
        calls = 0

        def fail_first_fsync(fd: int) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                _io_fault()
            original_fsync(fd)

        monkeypatch.setattr(local_module.os, "fsync", fail_first_fsync)

    expected = (
        "citation index commit may have occurred"
        if fault_name == "replace"
        else "citation index integrity check failed"
    )
    with pytest.raises(IndexIntegrityError, match=expected):
        index.build(root_id="policy-root", scope_id="scope-b")

    assert index_path.read_bytes() == original


def _io_fault() -> Any:
    raise OSError("planted-io-marker")


def test_post_replace_parent_fsync_failure_reports_uncertain_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, _, _ = _index(tmp_path, root)
    index.build(root_id="policy-root", scope_id="scope-a")
    original_fsync = local_module.os.fsync
    calls = 0

    def fail_parent_fsync(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("post-rename-marker")
        original_fsync(fd)

    monkeypatch.setattr(local_module.os, "fsync", fail_parent_fsync)

    with pytest.raises(IndexIntegrityError) as exc_info:
        index.build(root_id="policy-root", scope_id="scope-b")

    assert str(exc_info.value) == "citation index commit may have occurred"
    _assert_clean_error(exc_info.value, "post-rename-marker")


def test_close_failure_is_fixed_and_does_not_leak_or_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, index_path, _ = _index(tmp_path, root)
    index.build(root_id="policy-root", scope_id="scope-a")
    original = index_path.read_bytes()
    real_close = local_module.os.close
    failed = False

    def fail_one_close(fd: int) -> None:
        nonlocal failed
        real_close(fd)
        if not failed:
            failed = True
            raise OSError("close-private-marker")

    monkeypatch.setattr(local_module.os, "close", fail_one_close)

    with pytest.raises((UnsafeIndexRequestError, IndexIntegrityError)) as exc_info:
        index.build(root_id="policy-root", scope_id="scope-b")

    _assert_clean_error(exc_info.value, "close-private-marker")
    assert index_path.read_bytes() == original


def _thread_build(
    index_path: Path,
    key_path: Path,
    roots: dict[str, Path],
    root_id: str,
    scope_id: str,
    barrier: threading.Barrier,
    errors: list[BaseException],
) -> None:
    try:
        candidate = LocalCitationIndex(index_path=index_path, key_path=key_path, roots=roots)
        barrier.wait(timeout=10)
        candidate.build(root_id=root_id, scope_id=scope_id)
    except BaseException as error:  # test helper must return worker failures to the test thread
        errors.append(error)


def test_thread_concurrent_scope_builds_do_not_lose_updates(tmp_path: Path) -> None:
    root_a = tmp_path / "root-a"
    root_b = tmp_path / "root-b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "a.txt").write_text("Alpha capital controls.", encoding="utf-8")
    (root_b / "b.txt").write_text("Beta liquidity controls.", encoding="utf-8")
    _write_manifest(root_a, ("a.txt",))
    _write_manifest(root_b, ("b.txt",))
    roots = {"root-a": root_a, "root-b": root_b}
    index_path = tmp_path / "index.json"
    key_path = tmp_path / "index.json.key"
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []
    threads = [
        threading.Thread(
            target=_thread_build,
            args=(index_path, key_path, roots, "root-a", "scope-a", barrier, errors),
        ),
        threading.Thread(
            target=_thread_build,
            args=(index_path, key_path, roots, "root-b", "scope-b", barrier, errors),
        ),
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    reopened = LocalCitationIndex(index_path=index_path, key_path=key_path, roots=roots)
    assert reopened.query(
        scope_id="scope-a", query_id="query-a", query_text="alpha capital"
    ).citations
    assert reopened.query(
        scope_id="scope-b", query_id="query-b", query_text="beta liquidity"
    ).citations


def _process_build(
    index_path: str,
    key_path: str,
    roots: dict[str, str],
    root_id: str,
    scope_id: str,
    start: Any,
    results: Any,
) -> None:
    try:
        start.wait(10)
        index = LocalCitationIndex(
            index_path=Path(index_path),
            key_path=Path(key_path),
            roots={name: Path(path) for name, path in roots.items()},
        )
        result = index.build(root_id=root_id, scope_id=scope_id)
        results.put(("ok", result.index_hash))
    except BaseException as error:  # pragma: no cover - returned to parent process
        results.put(("error", type(error).__name__, str(error)))


def test_storage_lock_retries_transient_first_create_enoent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import errno

    root = tmp_path / "root"
    root.mkdir()
    (root / "policy.txt").write_text("Capital controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, index_path, _ = _index(tmp_path, root)
    lock_name = f"{index_path.name}.lock"
    real_open = local_module.os.open
    create_attempts = 0

    def transient_first_create(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal create_attempts
        if path == lock_name and flags & os.O_CREAT:
            create_attempts += 1
            if create_attempts == 1:
                raise FileNotFoundError(errno.ENOENT, "transient lock-create race")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    supports_dir_fd = set(local_module.os.supports_dir_fd)
    supports_dir_fd.add(transient_first_create)
    monkeypatch.setattr(local_module.os, "supports_dir_fd", supports_dir_fd)
    monkeypatch.setattr(local_module.os, "open", transient_first_create)

    result = index.build(root_id="policy-root", scope_id="scope-a")

    assert result.document_count == 1
    assert create_attempts == 2


def test_process_concurrent_scope_builds_do_not_lose_updates(tmp_path: Path) -> None:
    root_a = tmp_path / "root-a"
    root_b = tmp_path / "root-b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "a.txt").write_text("Alpha capital controls.", encoding="utf-8")
    (root_b / "b.txt").write_text("Beta liquidity controls.", encoding="utf-8")
    _write_manifest(root_a, ("a.txt",))
    _write_manifest(root_b, ("b.txt",))
    roots = {"root-a": str(root_a), "root-b": str(root_b)}
    index_path = tmp_path / "index.json"
    key_path = tmp_path / "index.json.key"
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_process_build,
            args=(
                str(index_path),
                str(key_path),
                roots,
                "root-a",
                "scope-a",
                start,
                results,
            ),
        ),
        context.Process(
            target=_process_build,
            args=(
                str(index_path),
                str(key_path),
                roots,
                "root-b",
                "scope-b",
                start,
                results,
            ),
        ),
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=20)

    assert all(process.exitcode == 0 for process in processes)
    outcomes = [results.get(timeout=5) for _ in processes]
    assert [outcome[0] for outcome in outcomes] == ["ok", "ok"], outcomes
    reopened = LocalCitationIndex(
        index_path=index_path,
        key_path=key_path,
        roots={name: Path(path) for name, path in roots.items()},
    )
    assert reopened.query(
        scope_id="scope-a", query_id="query-a", query_text="alpha capital"
    ).citations
    assert reopened.query(
        scope_id="scope-b", query_id="query-b", query_text="beta liquidity"
    ).citations


def test_enabled_factories_remain_offline_and_thread_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from riskprobe.integrations import create_integration_bundle, default_integrations

    def blocked(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    before = {thread.ident for thread in threading.enumerate()}

    default_integrations(enabled=True)
    create_integration_bundle(enabled=True)

    assert {thread.ident for thread in threading.enumerate()} == before


def test_special_files_are_opened_nonblocking_before_fstat_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    skipped = root / "runs.backup"
    skipped.mkdir()
    os.mkfifo(skipped / "special.pipe", 0o600)
    index, index_path, _ = _index(tmp_path, root)
    real_open = local_module.os.open
    observed_nonblocking = False

    def checked_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal observed_nonblocking
        if path == "special.pipe":
            observed_nonblocking = bool(flags & os.O_NONBLOCK)
            if not observed_nonblocking:
                raise OSError("blocking-special-file-open")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(local_module, "_require_fd_relative_support", lambda: None)
    monkeypatch.setattr(local_module.os, "open", checked_open)

    with pytest.raises(UnsafeIndexRequestError, match="citation index request is not safe"):
        index.build(root_id="policy-root", scope_id="scope-a")

    assert observed_nonblocking
    assert not index_path.exists()


def test_persisted_identifiers_still_apply_privacy_gate_with_valid_hmac(tmp_path: Path) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, index_path, key_path = _index(tmp_path, root)
    index.build(root_id="policy-root", scope_id="scope-a")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["scopes"][_keyed_rag_id(
        "scope", "scope-a", key_path.read_bytes()
    )]["root_id"] = "customer_123456"
    _reseal(payload, key_path.read_bytes())
    index_path.write_bytes(_canonical_bytes(payload))

    with pytest.raises(IndexIntegrityError, match="citation index integrity check failed"):
        index.query(scope_id="scope-a", query_id="query-a", query_text="policy")


def test_query_internal_failure_is_sanitized_as_index_integrity_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, _, _ = _index(tmp_path, root)
    index.build(root_id="policy-root", scope_id="scope-a")
    marker = "private-transformer-marker"

    def fail_transform(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError(marker)

    monkeypatch.setattr(local_module.TfidfTransformer, "fit_transform", fail_transform)

    with pytest.raises(IndexIntegrityError) as exc_info:
        index.query(scope_id="scope-a", query_id="query-a", query_text="policy")

    assert str(exc_info.value) == "citation index integrity check failed"
    _assert_clean_error(exc_info.value, marker)


def test_build_internal_failure_is_sanitized_and_preserves_previous_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, index_path, _ = _index(tmp_path, root)
    index.build(root_id="policy-root", scope_id="scope-a")
    original = index_path.read_bytes()
    marker = "private-seal-marker"

    def fail_seal(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError(marker)

    monkeypatch.setattr(local_module, "_sealed_index", fail_seal)

    with pytest.raises(IndexIntegrityError) as exc_info:
        index.build(root_id="policy-root", scope_id="scope-b")

    assert str(exc_info.value) == "citation index integrity check failed"
    _assert_clean_error(exc_info.value, marker)
    assert index_path.read_bytes() == original


@pytest.mark.parametrize("directory", [".cache", "__pycache__"])
def test_hidden_cache_directory_families_cannot_be_attested(
    tmp_path: Path,
    directory: str,
) -> None:
    root = tmp_path / "policies"
    nested = root / directory
    nested.mkdir(parents=True)
    relative = f"{directory}/policy.txt"
    (root / relative).write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, (relative,))
    index, index_path, _ = _index(tmp_path, root)

    with pytest.raises(UnsafeIndexRequestError, match="citation index request is not safe"):
        index.build(root_id="policy-root", scope_id="scope-a")

    assert not index_path.exists()


def test_manifest_rejects_windows_drive_absolute_path_with_forward_slashes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "policies"
    nested = root / "C:"
    nested.mkdir(parents=True)
    relative = "C:/policy.txt"
    (root / relative).write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, (relative,))
    index, index_path, _ = _index(tmp_path, root)

    with pytest.raises(UnsafeIndexRequestError, match="citation index request is not safe"):
        index.build(root_id="policy-root", scope_id="scope-a")

    assert not index_path.exists()


@pytest.mark.parametrize("unsafe_target", ["root", "manifest"])
def test_review_regression_root_and_manifest_require_trusted_permissions(
    tmp_path: Path,
    unsafe_target: str,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    manifest = _write_manifest(root, ("policy.txt",))
    index, index_path, _ = _index(tmp_path, root)
    target = root if unsafe_target == "root" else manifest
    original_mode = stat.S_IMODE(target.stat().st_mode)
    target.chmod(0o777 if unsafe_target == "root" else 0o666)
    try:
        with pytest.raises(
            UnsafeIndexRequestError,
            match="citation index request is not safe",
        ):
            index.build(root_id="policy-root", scope_id="scope-a")
    finally:
        target.chmod(original_mode)
    assert not index_path.exists()


def test_review_regression_root_open_is_bound_to_checked_ancestors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text(
        "Original aggregate quasar controls.", encoding="utf-8"
    )
    _write_manifest(root, ("policy.txt",))
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "policy.txt").write_text(
        "Replacement aggregate volcano controls.", encoding="utf-8"
    )
    _write_manifest(replacement, ("policy.txt",))
    saved = tmp_path / "opened-original"
    index, _, _ = _index(tmp_path, root)
    real_open = local_module.os.open
    swapped = False

    def swap_root() -> None:
        nonlocal swapped
        root.rename(saved)
        replacement.rename(root)
        swapped = True

    def racing_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if not swapped and dir_fd is None and os.fspath(path) == os.fspath(root):
            swap_root()
            return real_open(path, flags, mode)
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if not swapped and dir_fd is not None and path == root.name:
            swap_root()
        return descriptor

    monkeypatch.setattr(local_module, "_require_fd_relative_support", lambda: None)
    monkeypatch.setattr(local_module.os, "open", racing_open)

    index.build(root_id="policy-root", scope_id="scope-a")
    original = index.query(
        scope_id="scope-a",
        query_id="query-original",
        query_text="quasar",
    )
    replacement_result = index.query(
        scope_id="scope-a",
        query_id="query-replacement",
        query_text="volcano",
    )

    assert swapped
    assert original.citations
    assert replacement_result.citations == ()


def test_review_regression_storage_parent_open_survives_ancestor_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    storage = tmp_path / "storage-anchor"
    storage.mkdir()
    replacement = tmp_path / "storage-replacement"
    replacement.mkdir()
    saved = tmp_path / "storage-opened-original"
    index_path = storage / "index.json"
    key_path = storage / "index.json.key"
    lock_path = storage / "index.json.lock"
    index = LocalCitationIndex(
        index_path=index_path,
        key_path=key_path,
        roots={"policy-root": root},
    )
    real_open = local_module.os.open
    swapped = False

    def swap_storage() -> None:
        nonlocal swapped
        storage.rename(saved)
        replacement.rename(storage)
        swapped = True

    def racing_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if not swapped and dir_fd is None and os.fspath(path) == os.fspath(lock_path):
            swap_storage()
            return real_open(path, flags, mode)
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if not swapped and dir_fd is not None and path == storage.name:
            swap_storage()
        return descriptor

    monkeypatch.setattr(local_module, "_require_fd_relative_support", lambda: None)
    monkeypatch.setattr(local_module.os, "open", racing_open)

    index.build(root_id="policy-root", scope_id="scope-a")

    assert swapped
    assert (saved / "index.json").exists()
    assert (saved / "index.json.key").exists()
    assert not (storage / "index.json").exists()
    assert not (storage / "index.json.key").exists()


def test_review_regression_tree_walk_uses_streaming_fd_scandir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, _, _ = _index(tmp_path, root)

    def forbidden_listdir(*args: object, **kwargs: object) -> list[str]:
        del args, kwargs
        raise AssertionError("full directory materialization is forbidden")

    monkeypatch.setattr(local_module, "_require_fd_relative_support", lambda: None)
    monkeypatch.setattr(local_module.os, "listdir", forbidden_listdir)

    assert index.build(root_id="policy-root", scope_id="scope-a").document_count == 1


def test_review_regression_provider_sequence_is_bounded_before_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections.abc import Sequence

    class LengthRacingSequence(Sequence[dict[str, object]]):
        def __init__(self) -> None:
            self.iterations = 0
            self.getitems = 0
            self._items = (
                {
                    "operation": "aggregate_status",
                    "metric_code": "approval_rate",
                    "status_code": "stable",
                    "aggregate_count": 2,
                },
                {
                    "operation": "aggregate_status",
                    "metric_code": "liquidity_rate",
                    "status_code": "stable",
                    "aggregate_count": 3,
                },
            )

        def __len__(self) -> int:
            return 0

        def __iter__(self) -> Iterator[dict[str, object]]:
            self.iterations += 1
            return iter(self._items)

        def __getitem__(self, index: int) -> dict[str, object]:
            self.getitems += 1
            return self._items[index]

    root = tmp_path / "policies"
    root.mkdir()
    _write_manifest(root, ())
    index, index_path, _ = _index(tmp_path, root)
    monkeypatch.setattr(local_module, "_MAX_PROVIDER_SUMMARIES", 1)
    summaries = LengthRacingSequence()

    with pytest.raises(UnsafeContentError, match="citation content is not safe"):
        index.build(
            root_id="policy-root",
            scope_id="scope-a",
            provider_summaries=summaries,
        )

    assert summaries.iterations == 0
    assert summaries.getitems == 0
    assert not index_path.exists()


def test_review_important_rag_safe_validate_rejects_active_objects_passively() -> None:
    from collections.abc import Mapping
    from typing import ClassVar

    from pydantic import BaseModel

    class ActiveMapping(Mapping[str, object]):
        def __init__(self) -> None:
            self.iterations = 0
            self.getitems = 0

        def __getitem__(self, key: str) -> object:
            self.getitems += 1
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            self.iterations += 1
            raise AssertionError("active mapping executed")

        def __len__(self) -> int:
            return 3

    class ActiveModel(BaseModel):
        calls: ClassVar[int] = 0
        value: int = 1

        def model_dump(self, *args: object, **kwargs: object) -> dict[str, object]:
            del args, kwargs
            type(self).calls += 1
            raise AssertionError("active model_dump executed")

    mapping = ActiveMapping()
    model = ActiveModel()

    for candidate in (mapping, model):
        with pytest.raises(
            UnsafeIndexRequestError,
            match="citation index request is not safe",
        ):
            BuildResult.safe_validate(candidate)

    assert mapping.iterations == 0
    assert mapping.getitems == 0
    assert ActiveModel.calls == 0


def test_review_important_roots_are_exact_and_count_bounded_before_iteration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections.abc import Mapping

    class ActiveRoots(Mapping[str, Path]):
        def __init__(self) -> None:
            self.iterations = 0
            self.getitems = 0

        def __getitem__(self, key: str) -> Path:
            self.getitems += 1
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            self.iterations += 1
            raise AssertionError("active roots executed")

        def __len__(self) -> int:
            return 1

    active = ActiveRoots()
    with pytest.raises(UnsafeIndexRequestError, match="citation index request is not safe"):
        LocalCitationIndex(index_path=tmp_path / "active.json", roots=active)
    assert active.iterations == 0
    assert active.getitems == 0

    root_a = tmp_path / "root-a"
    root_b = tmp_path / "root-b"
    root_a.mkdir()
    root_b.mkdir()
    visited: list[Path] = []
    monkeypatch.setattr(local_module, "_MAX_REGISTERED_ROOTS", 1, raising=False)
    monkeypatch.setattr(local_module, "_validate_directory_chain", visited.append)

    with pytest.raises(UnsafeIndexRequestError, match="citation index request is not safe"):
        LocalCitationIndex(
            index_path=tmp_path / "bounded.json",
            roots={"root-a": root_a, "root-b": root_b},
        )

    assert visited == []


def test_review_important_dto_counts_have_hard_upper_bounds() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ProviderSafeSummary(
            text="Aggregate control coverage remains stable.",
            aggregate_count=1_000_001,
        )
    with pytest.raises(ValidationError):
        BuildResult(
            scope_id=f"scope-{'a' * 24}",
            document_count=8_193,
            index_hash="b" * 64,
        )
    with pytest.raises(ValidationError):
        Citation(
            rank=101,
            citation_id="a" * 64,
            document_id="b" * 64,
            content_hash="c" * 64,
            score=0.5,
        )

    citations = tuple(
        Citation.model_construct(
            rank=rank,
            citation_id=f"{rank:064x}",
            document_id=f"{rank + 200:064x}",
            content_hash=f"{rank + 400:064x}",
            score=round(1.0 - rank / 200, 6),
        )
        for rank in range(1, 102)
    )
    with pytest.raises(ValidationError):
        QueryResult(scope_id=f"scope-{'a' * 24}", citations=citations)


def test_review_important_safe_validate_preflights_dict_and_tuple_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_calls = 0
    query_calls = 0

    def active_build_validation(cls: type[BuildResult], value: object) -> BuildResult:
        nonlocal build_calls
        del cls, value
        build_calls += 1
        raise AssertionError("build model validation executed")

    def active_query_validation(cls: type[QueryResult], value: object) -> QueryResult:
        nonlocal query_calls
        del cls, value
        query_calls += 1
        raise AssertionError("query model validation executed")

    monkeypatch.setattr(BuildResult, "model_validate", classmethod(active_build_validation))
    monkeypatch.setattr(QueryResult, "model_validate", classmethod(active_query_validation))

    with pytest.raises(UnsafeIndexRequestError, match="citation index request is not safe"):
        BuildResult.safe_validate(
            {
                "scope_id": f"scope-{'a' * 24}",
                "document_count": 1,
                "index_hash": "b" * 64,
                **{f"extra-{index}": index for index in range(32)},
            }
        )
    with pytest.raises(UnsafeIndexRequestError, match="citation index request is not safe"):
        QueryResult.safe_validate(
            {
                "scope_id": f"scope-{'a' * 24}",
                "citations": tuple({} for _ in range(101)),
            }
        )

    assert build_calls == 0
    assert query_calls == 0


def test_review_regression_partial_key_and_strict_stale_temps_are_recovered(
    tmp_path: Path,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, index_path, key_path = _index(tmp_path, root)
    key_path.write_bytes(b"partial")
    key_path.chmod(0o600)
    stale_index = tmp_path / f".{index_path.name}.{'a' * 32}.index.tmp"
    stale_key = tmp_path / f".{key_path.name}.{'b' * 32}.key.tmp"
    decoy = tmp_path / f".{key_path.name}.not-hex.key.tmp"
    for path in (stale_index, stale_key, decoy):
        path.write_bytes(b"stale")
        path.chmod(0o600)

    result = index.build(root_id="policy-root", scope_id="scope-a")

    assert result.document_count == 1
    assert index_path.exists()
    assert len(key_path.read_bytes()) == 32
    assert not stale_index.exists()
    assert not stale_key.exists()
    assert decoy.exists()


def test_review_regression_stale_cleanup_never_deletes_valid_key(tmp_path: Path) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, _, key_path = _index(tmp_path, root)
    valid_key = b"k" * 32
    key_path.write_bytes(valid_key)
    key_path.chmod(0o600)
    stale_key = tmp_path / f".{key_path.name}.{'c' * 32}.key.tmp"
    stale_key.write_bytes(b"stale")
    stale_key.chmod(0o600)

    index.build(root_id="policy-root", scope_id="scope-a")

    assert key_path.read_bytes() == valid_key
    assert not stale_key.exists()


def _capture_review_regression_error(call: Any) -> BaseException:
    try:
        call()
    except BaseException as error:
        return error
    raise AssertionError("expected operation to fail")


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_review_regression_replace_then_interrupt_is_uncertain_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_type: type[BaseException],
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, _, _ = _index(tmp_path, root)
    index.build(root_id="policy-root", scope_id="scope-a")
    real_replace = local_module.os.replace

    def replace_then_interrupt(*args: object, **kwargs: object) -> None:
        real_replace(*args, **kwargs)
        raise interrupt_type("replace-interrupt-private-marker")

    monkeypatch.setattr(local_module.os, "replace", replace_then_interrupt)

    error = _capture_review_regression_error(
        lambda: index.build(root_id="policy-root", scope_id="scope-b")
    )

    assert isinstance(error, IndexIntegrityError)
    assert str(error) == "citation index commit may have occurred"
    _assert_clean_error(error, "replace-interrupt-private-marker")


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_review_regression_parent_fsync_interrupt_is_uncertain_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_type: type[BaseException],
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, _, _ = _index(tmp_path, root)
    index.build(root_id="policy-root", scope_id="scope-a")
    real_fsync = local_module.os.fsync
    calls = 0

    def interrupt_parent_fsync(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise interrupt_type("parent-fsync-private-marker")
        real_fsync(fd)

    monkeypatch.setattr(local_module.os, "fsync", interrupt_parent_fsync)

    error = _capture_review_regression_error(
        lambda: index.build(root_id="policy-root", scope_id="scope-b")
    )

    assert isinstance(error, IndexIntegrityError)
    assert str(error) == "citation index commit may have occurred"
    _assert_clean_error(error, "parent-fsync-private-marker")


def test_review_regression_precommit_interrupt_propagates_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, _, _ = _index(tmp_path, root)
    index.build(root_id="policy-root", scope_id="scope-a")
    marker = KeyboardInterrupt("precommit-private-marker")

    def interrupt_temp_fsync(_fd: int) -> None:
        raise marker

    monkeypatch.setattr(local_module.os, "fsync", interrupt_temp_fsync)

    with pytest.raises(KeyboardInterrupt) as exc_info:
        index.build(root_id="policy-root", scope_id="scope-b")

    assert exc_info.value is marker


@pytest.mark.parametrize("cleanup_stage", ["temp_close", "temp_unlink"])
@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_task1_precommit_cleanup_interrupt_propagates_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_stage: str,
    interrupt_type: type[BaseException],
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, index_path, _ = _index(tmp_path, root)
    index.build(root_id="policy-root", scope_id="scope-a")
    original = index_path.read_bytes()
    marker = interrupt_type(f"{cleanup_stage}-interrupt-private-marker")
    real_write_all = local_module._write_all
    real_close = local_module.os.close
    real_unlink = local_module.os.unlink
    temporary_fd: int | None = None
    primary_failure_seen = False
    cleanup_actions: list[str] = []

    def fail_after_temp_write(fd: int, encoded: bytes) -> None:
        nonlocal primary_failure_seen, temporary_fd
        temporary_fd = fd
        real_write_all(fd, encoded)
        primary_failure_seen = True
        raise RuntimeError("ordinary-primary-private-marker")

    def interrupt_temp_close(fd: int) -> None:
        if fd == temporary_fd:
            cleanup_actions.append("temp_close")
            real_close(fd)
            if cleanup_stage == "temp_close":
                raise marker
            return
        real_close(fd)

    def interrupt_temp_unlink(name: str, *, dir_fd: int | None = None) -> None:
        if name.endswith(".index.tmp"):
            cleanup_actions.append("temp_unlink")
            real_unlink(name, dir_fd=dir_fd)
            if cleanup_stage == "temp_unlink":
                raise marker
            return
        real_unlink(name, dir_fd=dir_fd)

    supported_dir_fd = set(local_module.os.supports_dir_fd)
    supported_dir_fd.discard(real_unlink)
    supported_dir_fd.add(interrupt_temp_unlink)
    monkeypatch.setattr(local_module, "_write_all", fail_after_temp_write)
    monkeypatch.setattr(local_module.os, "close", interrupt_temp_close)
    monkeypatch.setattr(local_module.os, "unlink", interrupt_temp_unlink)
    monkeypatch.setattr(local_module.os, "supports_dir_fd", supported_dir_fd)

    error = _capture_review_regression_error(
        lambda: index.build(root_id="policy-root", scope_id="scope-b")
    )

    assert primary_failure_seen, repr(error)
    assert cleanup_actions == ["temp_close", "temp_unlink"]
    assert error is marker
    assert not isinstance(error, IndexIntegrityError)
    assert index_path.read_bytes() == original


@pytest.mark.parametrize("cleanup_stage", ["temp_close", "temp_unlink"])
@pytest.mark.parametrize("primary_type", [RuntimeError, OSError])
def test_task1_precommit_cleanup_oserror_is_fixed_and_best_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_stage: str,
    primary_type: type[Exception],
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, index_path, _ = _index(tmp_path, root)
    index.build(root_id="policy-root", scope_id="scope-a")
    original = index_path.read_bytes()
    primary_marker = f"{primary_type.__name__}-primary-private-marker"
    cleanup_marker = f"{cleanup_stage}-oserror-private-marker"
    real_write_all = local_module._write_all
    real_close = local_module.os.close
    real_unlink = local_module.os.unlink
    temporary_fd: int | None = None
    cleanup_actions: list[str] = []

    def fail_after_temp_write(fd: int, encoded: bytes) -> None:
        nonlocal temporary_fd
        temporary_fd = fd
        real_write_all(fd, encoded)
        raise primary_type(primary_marker)

    def fail_temp_close(fd: int) -> None:
        if fd == temporary_fd:
            cleanup_actions.append("temp_close")
            real_close(fd)
            if cleanup_stage == "temp_close":
                raise OSError(cleanup_marker)
            return
        real_close(fd)

    def fail_temp_unlink(name: str, *, dir_fd: int | None = None) -> None:
        if name.endswith(".index.tmp"):
            cleanup_actions.append("temp_unlink")
            real_unlink(name, dir_fd=dir_fd)
            if cleanup_stage == "temp_unlink":
                raise OSError(cleanup_marker)
            return
        real_unlink(name, dir_fd=dir_fd)

    supported_dir_fd = set(local_module.os.supports_dir_fd)
    supported_dir_fd.discard(real_unlink)
    supported_dir_fd.add(fail_temp_unlink)
    monkeypatch.setattr(local_module, "_write_all", fail_after_temp_write)
    monkeypatch.setattr(local_module.os, "close", fail_temp_close)
    monkeypatch.setattr(local_module.os, "unlink", fail_temp_unlink)
    monkeypatch.setattr(local_module.os, "supports_dir_fd", supported_dir_fd)

    error = _capture_review_regression_error(
        lambda: index.build(root_id="policy-root", scope_id="scope-b")
    )

    assert cleanup_actions == ["temp_close", "temp_unlink"]
    assert isinstance(error, IndexIntegrityError)
    assert str(error) == "citation index integrity check failed"
    _assert_clean_error(error, primary_marker)
    _assert_clean_error(error, cleanup_marker)
    assert index_path.read_bytes() == original


def test_review_regression_cleanup_failure_cannot_override_uncertain_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, _, _ = _index(tmp_path, root)
    index.build(root_id="policy-root", scope_id="scope-a")
    real_replace = local_module.os.replace
    real_close = local_module.os.close
    replace_started = False

    def replace_then_interrupt(*args: object, **kwargs: object) -> None:
        nonlocal replace_started
        real_replace(*args, **kwargs)
        replace_started = True
        raise KeyboardInterrupt("replace-private-marker")

    def close_then_fail(fd: int) -> None:
        real_close(fd)
        if replace_started:
            raise RuntimeError("cleanup-private-marker")

    monkeypatch.setattr(local_module.os, "replace", replace_then_interrupt)
    monkeypatch.setattr(local_module.os, "close", close_then_fail)

    error = _capture_review_regression_error(
        lambda: index.build(root_id="policy-root", scope_id="scope-b")
    )

    assert isinstance(error, IndexIntegrityError)
    assert str(error) == "citation index commit may have occurred"
    _assert_clean_error(error, "private-marker")


@pytest.mark.parametrize(
    "relative",
    [
        "policy\u202etxt.md",
        "cafe\u0301.txt",
        "p\u043elicy.txt",
    ],
)
def test_review_regression_manifest_rejects_unicode_controls_and_confusables(
    tmp_path: Path,
    relative: str,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / relative).write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, (relative,))
    index, index_path, _ = _index(tmp_path, root)

    with pytest.raises(UnsafeIndexRequestError, match="citation index request is not safe"):
        index.build(root_id="policy-root", scope_id="scope-a")

    assert not index_path.exists()


@pytest.mark.parametrize(
    "unsafe_text",
    [
        "alice,1",
        "credential: ordinary-secret-value",
        "Review logic then return calculate_risk(value)",
    ],
)
def test_review_regression_attested_text_rejects_csv_secret_alias_and_inline_code(
    tmp_path: Path,
    unsafe_text: str,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "unsafe.txt").write_text(unsafe_text, encoding="utf-8")
    _write_manifest(root, ("unsafe.txt",))
    index, index_path, _ = _index(tmp_path, root)

    with pytest.raises(UnsafeContentError) as exc_info:
        index.build(root_id="policy-root", scope_id="scope-a")

    assert str(exc_info.value) == "citation content is not safe"
    _assert_clean_error(exc_info.value, unsafe_text)
    assert not index_path.exists()


def test_review_regression_provider_untrusted_boundary_is_structured(
    tmp_path: Path,
) -> None:
    structured = {
        "operation": "aggregate_status",
        "metric_code": "control_coverage",
        "status_code": "stable",
        "aggregate_count": 4,
    }
    direct = ProviderSafeSummary(
        text="Aggregate control coverage remains stable.",
        aggregate_count=4,
    )
    validated = ProviderSafeSummary.model_validate(
        {
            "text": "Aggregate control coverage remains stable.",
            "aggregate_count": 4,
        }
    )
    generated = ProviderSafeSummary.safe_validate(structured)
    assert generated.aggregate_count == 4
    assert generated.text == "Aggregate control coverage status is stable."

    for model in (direct, validated, generated):
        with pytest.raises(UnsafeContentError, match="citation content is not safe"):
            ProviderSafeSummary.safe_validate(model)

    with pytest.raises(UnsafeContentError, match="citation content is not safe"):
        ProviderSafeSummary.safe_validate(
            {
                "text": "Aggregate control coverage remains stable.",
                "aggregate_count": 4,
            }
        )

    forged = ProviderSafeSummary.model_construct(
        text="Aggregate control coverage remains stable.",
        aggregate_count=4,
        content_hash=direct.content_hash,
    )
    with pytest.raises(UnsafeContentError, match="citation content is not safe"):
        ProviderSafeSummary.safe_validate(forged)

    root = tmp_path / "policies"
    root.mkdir()
    _write_manifest(root, ())
    index, index_path, _ = _index(tmp_path, root)
    for model in (direct, validated, generated):
        with pytest.raises(UnsafeContentError, match="citation content is not safe"):
            index.build(
                root_id="policy-root",
                scope_id="scope-a",
                provider_summaries=(model,),
            )
        assert not index_path.exists()

    result = index.build(
        root_id="policy-root",
        scope_id="scope-a",
        provider_summaries=(structured,),
    )
    assert result.document_count == 1
    assert index.query(
        scope_id="scope-a",
        query_id="query-provider",
        query_text="control coverage stable",
    ).citations


def test_review_regression_rag_ids_are_opaque_and_semantic_labels_are_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, index_path, _ = _index(tmp_path, root)

    build = index.build(root_id="policy-root", scope_id="scope-a")
    query = index.query(
        scope_id="scope-a",
        query_id="query-a",
        query_text="policy controls",
    )

    assert build.scope_id == query.scope_id
    assert build.scope_id.startswith("scope-")
    assert len(build.scope_id) == len("scope-") + 24
    assert build.scope_id != "scope-a"
    stored = index_path.read_bytes()
    assert b"policy-root" not in stored
    assert b'"scope-a"' not in stored
    assert b'"query-a"' not in stored

    with pytest.raises(UnsafeIndexRequestError, match="citation index request is not safe"):
        LocalCitationIndex(
            index_path=tmp_path / "semantic-root-index.json",
            roots={"north_region": root},
        )
    with pytest.raises(UnsafeIndexRequestError, match="citation index request is not safe"):
        index.build(root_id="policy-root", scope_id="north_region")
    with pytest.raises(UnsafeIndexRequestError, match="citation index request is not safe"):
        index.query(
            scope_id="scope-a",
            query_id="north_region",
            query_text="policy controls",
        )


def test_review_followup_key_path_is_uniquely_derived_from_index_path(
    tmp_path: Path,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    _write_manifest(root, ())

    with pytest.raises(UnsafeIndexRequestError, match="citation index request is not safe"):
        LocalCitationIndex(
            index_path=tmp_path / "first-index.json",
            key_path=tmp_path / "shared.key",
            roots={"policy-root": root},
        )


def test_review_followup_provider_capability_cannot_be_laundered_by_model_copy() -> None:
    generated = ProviderSafeSummary.safe_validate(
        {
            "operation": "aggregate_status",
            "metric_code": "control_coverage",
            "status_code": "stable",
            "aggregate_count": 4,
        }
    )
    laundered = generated.model_copy(
        update={
            "text": "The credential is ordinary-private-marker.",
            "content_hash": "",
        }
    )

    with pytest.raises(UnsafeContentError) as exc_info:
        ProviderSafeSummary.safe_validate(laundered)

    assert str(exc_info.value) == "citation content is not safe"
    _assert_clean_error(exc_info.value, "ordinary-private-marker")


@pytest.mark.parametrize(
    "unsafe_text",
    [
        "The credential is ordinary-secret-value.",
        "Review calculate_risk(value) before approval.",
        "alice\t1",
        "alice | 1",
        "cohort_label north_region",
        "group label north_region",
        "The credential named ordinary-secret-value is rotated.",
        "Review [risk for risk in portfolio] before approval.",
    ],
)
def test_review_followup_trusted_scanner_rejects_alias_and_call_variants(
    tmp_path: Path,
    unsafe_text: str,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "unsafe.txt").write_text(unsafe_text, encoding="utf-8")
    _write_manifest(root, ("unsafe.txt",))
    index, index_path, _ = _index(tmp_path, root)

    with pytest.raises(UnsafeContentError, match="citation content is not safe"):
        index.build(root_id="policy-root", scope_id="scope-a")

    assert not index_path.exists()


@pytest.mark.parametrize(
    "index_name",
    [
        "first-index.json.key",
        "first-index.json.lock",
        f".first-index.json.{'a' * 32}.index.tmp",
    ],
)
def test_review_followup_index_name_cannot_overlap_storage_sidecars(
    tmp_path: Path,
    index_name: str,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    _write_manifest(root, ())

    with pytest.raises(UnsafeIndexRequestError, match="citation index request is not safe"):
        LocalCitationIndex(
            index_path=tmp_path / index_name,
            roots={"policy-root": root},
        )


def _keyed_rag_id(kind: str, value: str, key: bytes) -> str:
    digest = hmac.new(
        key,
        b"rag-identifier\0" + kind.encode("ascii") + b"\0" + value.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()[:24]
    return f"{kind}-{digest}"


def test_review_important_ids_are_per_index_keyed_and_opaque_tokens_are_idempotent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    first_path = tmp_path / "first-index.json"
    first_key_path = tmp_path / "first-index.json.key"
    first = LocalCitationIndex(index_path=first_path, roots={"policy-root": root})

    assert not first_key_path.exists()
    assert tuple(first._roots) == ("policy-root",)  # noqa: SLF001
    first_build = first.build(root_id="policy-root", scope_id="scope-a")
    first_key = first_key_path.read_bytes()
    expected_scope = _keyed_rag_id("scope", "scope-a", first_key)
    expected_root = _keyed_rag_id("root", "policy-root", first_key)
    public_scope = _rag_id("scope", "scope-a")
    payload = json.loads(first_path.read_text(encoding="utf-8"))

    assert first_build.scope_id == expected_scope
    assert first_build.scope_id != public_scope
    assert payload["format_version"] == 4
    assert payload["scopes"][expected_scope]["root_id"] == expected_root
    assert b"policy-root" not in first_path.read_bytes()
    assert b'"scope-a"' not in first_path.read_bytes()

    reopened = LocalCitationIndex(index_path=first_path, roots={"policy-root": root})
    reopened_result = reopened.query(
        scope_id="scope-a",
        query_id="query-a",
        query_text="policy controls",
    )
    assert reopened_result.scope_id == expected_scope
    assert first_key_path.read_bytes() == first_key

    second_path = tmp_path / "second-index.json"
    second_key_path = tmp_path / "second-index.json.key"
    second = LocalCitationIndex(index_path=second_path, roots={"policy-root": root})
    second_build = second.build(root_id="policy-root", scope_id="scope-a")
    assert second_build.scope_id == _keyed_rag_id(
        "scope",
        "scope-a",
        second_key_path.read_bytes(),
    )
    assert second_build.scope_id != first_build.scope_id

    opaque_scope = f"scope-{'f' * 24}"
    opaque_root = f"root-{'e' * 24}"
    opaque_index = LocalCitationIndex(
        index_path=tmp_path / "opaque-index.json",
        roots={opaque_root: root},
    )
    opaque_build = opaque_index.build(root_id=opaque_root, scope_id=opaque_scope)
    assert opaque_build.scope_id == opaque_scope
    assert opaque_index.query(
        scope_id=opaque_scope,
        query_id=f"query-{'d' * 24}",
        query_text="policy controls",
    ).scope_id == opaque_scope


def test_review_important_empty_query_creates_owner_key_and_dtos_require_opaque_scope(
    tmp_path: Path,
) -> None:
    from pydantic import ValidationError

    root = tmp_path / "policies"
    root.mkdir()
    _write_manifest(root, ())
    index_path = tmp_path / "empty-index.json"
    key_path = tmp_path / "empty-index.json.key"
    index = LocalCitationIndex(index_path=index_path, roots={"policy-root": root})

    result = index.query(
        scope_id="scope-a",
        query_id="query-a",
        query_text="",
    )

    assert not index_path.exists()
    assert key_path.exists()
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert result.scope_id == _keyed_rag_id("scope", "scope-a", key_path.read_bytes())
    with pytest.raises(ValidationError):
        BuildResult(scope_id="scope-a", document_count=0, index_hash="a" * 64)
    with pytest.raises(ValidationError):
        QueryResult(scope_id="scope-a", citations=())


def test_review_important_build_loads_key_before_id_canonicalization(
    tmp_path: Path,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    _write_manifest(root, ())
    index_path = tmp_path / "ordered-index.json"
    key_path = tmp_path / "ordered-index.json.key"
    index = LocalCitationIndex(index_path=index_path, roots={"policy-root": root})

    with pytest.raises(UnsafeIndexRequestError, match="citation index request is not safe"):
        index.build(root_id="policy-root", scope_id="north_region")

    assert key_path.exists()
    assert not index_path.exists()


def _traceback_value_contains_rejected(
    value: object,
    *,
    rejected_ids: set[int],
    markers: tuple[str, ...],
    seen: set[int],
    depth: int = 0,
) -> bool:
    if id(value) in rejected_ids:
        return True
    if type(value) is str:
        return any(marker in value for marker in markers)
    if isinstance(value, BaseException):
        return any(marker in str(value) for marker in markers) or _traceback_value_contains_rejected(
            value.args,
            rejected_ids=rejected_ids,
            markers=markers,
            seen=seen,
            depth=depth + 1,
        )
    if depth >= 5 or id(value) in seen:
        return False
    seen.add(id(value))
    if type(value) is dict:
        items = tuple(value.items())[:512]
        return any(
            _traceback_value_contains_rejected(
                item,
                rejected_ids=rejected_ids,
                markers=markers,
                seen=seen,
                depth=depth + 1,
            )
            for pair in items
            for item in pair
        )
    if type(value) in {list, tuple, set, frozenset}:
        return any(
            _traceback_value_contains_rejected(
                item,
                rejected_ids=rejected_ids,
                markers=markers,
                seen=seen,
                depth=depth + 1,
            )
            for item in tuple(value)[:512]
        )
    return False


def _assert_rag_traceback_locals_clean(
    error: BaseException,
    *,
    rejected: tuple[object, ...],
    markers: tuple[str, ...],
) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None
    target_frames = []
    traceback = error.__traceback__
    while traceback is not None:
        module_name = traceback.tb_frame.f_globals.get("__name__", "")
        if module_name == "riskprobe.rag" or module_name.startswith("riskprobe.rag."):
            target_frames.append(traceback.tb_frame)
        traceback = traceback.tb_next
    assert target_frames
    rejected_ids = {id(value) for value in rejected}
    for frame in target_frames:
        for local_name, value in frame.f_locals.items():
            assert not _traceback_value_contains_rejected(
                value,
                rejected_ids=rejected_ids,
                markers=markers,
                seen=set(),
            ), f"{frame.f_code.co_name}.{local_name} retained rejected input"


def test_review_important_rag_fixed_errors_scrub_target_traceback_locals(
    tmp_path: Path,
) -> None:
    dto_marker = "dto-private-trace-marker"
    dto_input = {
        "scope_id": f"scope-{'a' * 24}",
        "citations": tuple({"title": dto_marker} for _ in range(101)),
    }
    dto_error = _capture_review_regression_error(
        lambda: QueryResult.safe_validate(dto_input)
    )
    _assert_rag_traceback_locals_clean(
        dto_error,
        rejected=(dto_input,),
        markers=(dto_marker,),
    )

    summary_marker = "summary-private-trace-marker"
    summary_input = {
        "operation": "aggregate_status",
        "metric_code": "control_coverage",
        "status_code": summary_marker,
        "aggregate_count": 4,
    }
    summary_error = _capture_review_regression_error(
        lambda: ProviderSafeSummary.safe_validate(summary_input)
    )
    _assert_rag_traceback_locals_clean(
        summary_error,
        rejected=(summary_input,),
        markers=(summary_marker,),
    )

    root = tmp_path / "policies"
    root.mkdir()
    _write_manifest(root, ())
    constructor_marker = "customer_123456-constructor-private-marker"
    roots_input = {constructor_marker: root}
    constructor_error = _capture_review_regression_error(
        lambda: LocalCitationIndex(
            index_path=tmp_path / "constructor-index.json",
            roots=roots_input,
        )
    )
    _assert_rag_traceback_locals_clean(
        constructor_error,
        rejected=(roots_input,),
        markers=(constructor_marker,),
    )

    index = LocalCitationIndex(
        index_path=tmp_path / "trace-index.json",
        roots={"policy-root": root},
    )
    build_summary_error = _capture_review_regression_error(
        lambda: index.build(
            root_id="policy-root",
            scope_id="scope-a",
            provider_summaries=(summary_input,),
        )
    )
    _assert_rag_traceback_locals_clean(
        build_summary_error,
        rejected=(summary_input,),
        markers=(summary_marker,),
    )

    id_marker = "customer_123456-id-private-marker"
    id_error = _capture_review_regression_error(
        lambda: index.build(root_id="policy-root", scope_id=id_marker)
    )
    _assert_rag_traceback_locals_clean(
        id_error,
        rejected=(id_marker,),
        markers=(id_marker,),
    )

    query_marker = "/private/query-private-trace-marker.csv"
    query_error = _capture_review_regression_error(
        lambda: index.query(
            scope_id="scope-a",
            query_id="query-a",
            query_text=query_marker,
        )
    )
    _assert_rag_traceback_locals_clean(
        query_error,
        rejected=(query_marker,),
        markers=(query_marker,),
    )

    query_id_marker = "customer_123456-query-private-marker"
    query_id_error = _capture_review_regression_error(
        lambda: index.query(
            scope_id="scope-a",
            query_id=query_id_marker,
            query_text="aggregate policy",
        )
    )
    _assert_rag_traceback_locals_clean(
        query_id_error,
        rejected=(query_id_marker,),
        markers=(query_id_marker,),
    )


@pytest.mark.parametrize("cleanup_stage", ["unlock", "lock_close", "parent_close"])
@pytest.mark.parametrize("fault_type", [OSError, KeyboardInterrupt, SystemExit])
def test_review_important_post_commit_cleanup_fault_is_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_stage: str,
    fault_type: type[BaseException],
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, _, _ = _index(tmp_path, root)
    index.build(root_id="policy-root", scope_id="scope-a")

    marker = f"post-commit-{cleanup_stage}-{fault_type.__name__}-private-marker"
    real_fsync = local_module.os.fsync
    real_flock = local_module.fcntl.flock
    real_close = local_module.os.close
    fsync_calls = 0
    parent_fsync_completed = False
    post_sync_close_calls = 0

    def tracked_fsync(fd: int) -> None:
        nonlocal fsync_calls, parent_fsync_completed
        real_fsync(fd)
        fsync_calls += 1
        if fsync_calls == 2:
            parent_fsync_completed = True

    def faulting_flock(fd: int, operation: int) -> None:
        if (
            cleanup_stage == "unlock"
            and parent_fsync_completed
            and operation == local_module.fcntl.LOCK_UN
        ):
            raise fault_type(marker)
        real_flock(fd, operation)

    def faulting_close(fd: int) -> None:
        nonlocal post_sync_close_calls
        if parent_fsync_completed:
            post_sync_close_calls += 1
            target_call = 1 if cleanup_stage == "lock_close" else 2
            if cleanup_stage in {"lock_close", "parent_close"} and (
                post_sync_close_calls == target_call
            ):
                real_close(fd)
                raise fault_type(marker)
        real_close(fd)

    monkeypatch.setattr(local_module.os, "fsync", tracked_fsync)
    monkeypatch.setattr(local_module.fcntl, "flock", faulting_flock)
    monkeypatch.setattr(local_module.os, "close", faulting_close)

    error = _capture_review_regression_error(
        lambda: index.build(root_id="policy-root", scope_id="scope-b")
    )

    assert parent_fsync_completed
    assert isinstance(error, IndexIntegrityError)
    assert str(error) == "citation index commit may have occurred"
    _assert_clean_error(error, marker)
    _assert_rag_traceback_locals_clean(error, rejected=(), markers=(marker,))


def test_review_important_cleanup_fault_without_index_commit_is_ordinary_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    _write_manifest(root, ())
    index = LocalCitationIndex(
        index_path=tmp_path / "no-commit-index.json",
        roots={"policy-root": root},
    )
    index.query(scope_id="scope-a", query_id="query-a", query_text="")
    real_flock = local_module.fcntl.flock
    marker = "no-index-commit-cleanup-private-marker"

    def faulting_unlock(fd: int, operation: int) -> None:
        if operation == local_module.fcntl.LOCK_UN:
            raise OSError(marker)
        real_flock(fd, operation)

    monkeypatch.setattr(local_module.fcntl, "flock", faulting_unlock)

    error = _capture_review_regression_error(
        lambda: index.query(scope_id="scope-a", query_id="query-b", query_text="")
    )

    assert isinstance(error, IndexIntegrityError)
    assert str(error) == "citation index integrity check failed"
    _assert_clean_error(error, marker)


def test_review_followup_query_result_rejects_forged_nested_citation() -> None:
    forged = Citation.model_construct(
        rank=1,
        citation_id="not-a-hash",
        document_id="not-a-hash",
        content_hash="not-a-hash",
        score=0.5,
        title="/private/customer-record.txt",
    )

    with pytest.raises(
        UnsafeIndexRequestError,
        match="citation index request is not safe",
    ):
        QueryResult.safe_validate(
            {
                "scope_id": f"scope-{'a' * 24}",
                "citations": (forged,),
            }
        )


def test_review_followup_query_result_rejects_active_nested_mapping_passively() -> None:
    from collections.abc import Mapping

    class ActiveCitationMapping(Mapping[str, object]):
        def __init__(self) -> None:
            self.iterations = 0
            self.getitems = 0

        def __getitem__(self, key: str) -> object:
            del key
            self.getitems += 1
            raise AssertionError("active citation mapping accessed")

        def __iter__(self) -> Iterator[str]:
            self.iterations += 1
            raise AssertionError("active citation mapping iterated")

        def __len__(self) -> int:
            return len(Citation.model_fields)

    active = ActiveCitationMapping()

    with pytest.raises(
        UnsafeIndexRequestError,
        match="citation index request is not safe",
    ):
        QueryResult.safe_validate(
            {
                "scope_id": f"scope-{'a' * 24}",
                "citations": (active,),
            }
        )

    assert active.iterations == 0
    assert active.getitems == 0


def test_review_followup_query_result_preflights_nested_dict_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_calls = 0

    def active_validation(cls: type[QueryResult], value: object) -> QueryResult:
        nonlocal validation_calls
        del cls, value
        validation_calls += 1
        raise AssertionError("query validation executed")

    monkeypatch.setattr(
        QueryResult,
        "model_validate",
        classmethod(active_validation),
    )
    oversized = {
        f"extra-{index}": index
        for index in range(len(Citation.model_fields) + 1)
    }

    with pytest.raises(
        UnsafeIndexRequestError,
        match="citation index request is not safe",
    ):
        QueryResult.safe_validate(
            {
                "scope_id": f"scope-{'a' * 24}",
                "citations": (oversized,),
            }
        )

    assert validation_calls == 0


def test_review_followup_provider_operation_is_exact_str_before_comparison() -> None:
    class ActiveOperation:
        def __init__(self) -> None:
            self.comparisons = 0

        def __ne__(self, other: object) -> bool:
            del other
            self.comparisons += 1
            return False

    operation = ActiveOperation()

    with pytest.raises(UnsafeContentError, match="citation content is not safe"):
        ProviderSafeSummary.safe_validate(
            {
                "operation": operation,
                "metric_code": "control_coverage",
                "status_code": "stable",
                "aggregate_count": 4,
            }
        )

    assert operation.comparisons == 0


def test_review_followup_rag_dict_keys_are_exact_before_lookup() -> None:
    class ActiveCollisionKey:
        def __init__(self, target: str) -> None:
            self.target = target
            self.hashes = 0
            self.comparisons = 0

        def __hash__(self) -> int:
            self.hashes += 1
            return hash(self.target)

        def __eq__(self, other: object) -> bool:
            del other
            self.comparisons += 1
            return False

        def reset(self) -> None:
            self.hashes = 0
            self.comparisons = 0

    outer_key = ActiveCollisionKey("citations")
    outer = {
        "scope_id": f"scope-{'a' * 24}",
        outer_key: (),
    }
    outer_key.reset()

    provider_key = ActiveCollisionKey("operation")
    provider = {
        provider_key: "aggregate_status",
        "metric_code": "control_coverage",
        "status_code": "stable",
        "aggregate_count": 4,
    }
    provider_key.reset()

    citation_key = ActiveCollisionKey("rank")
    citation = {
        citation_key: 1,
        "citation_id": "a" * 64,
        "document_id": "b" * 64,
        "content_hash": "c" * 64,
        "score": 0.5,
        "title": None,
    }
    citation_key.reset()

    with pytest.raises(
        UnsafeIndexRequestError,
        match="citation index request is not safe",
    ):
        QueryResult.safe_validate(outer)
    with pytest.raises(UnsafeContentError, match="citation content is not safe"):
        ProviderSafeSummary.safe_validate(provider)
    with pytest.raises(
        UnsafeIndexRequestError,
        match="citation index request is not safe",
    ):
        QueryResult.safe_validate(
            {
                "scope_id": f"scope-{'a' * 24}",
                "citations": (citation,),
            }
        )

    for key in (outer_key, provider_key, citation_key):
        assert key.hashes == 0
        assert key.comparisons == 0


@pytest.mark.parametrize(
    "unsafe_text",
    [
        pytest.param("| alice |", id="pipe-wrapped-cell"),
        pytest.param("| alice", id="empty-leading-pipe-cell"),
        pytest.param("alice |", id="empty-trailing-pipe-cell"),
        pytest.param(
            f"[{'x' * 257} for risk in portfolio]",
            id="long-list-comprehension",
        ),
    ],
)
def test_review_followup_trusted_scanner_rejects_boundary_rows_and_long_code(
    tmp_path: Path,
    unsafe_text: str,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "unsafe.txt").write_text(unsafe_text, encoding="utf-8")
    _write_manifest(root, ("unsafe.txt",))
    index, index_path, _ = _index(tmp_path, root)

    with pytest.raises(UnsafeContentError, match="citation content is not safe"):
        index.build(root_id="policy-root", scope_id="scope-a")

    assert not index_path.exists()


@pytest.mark.parametrize(
    "unsafe_text",
    [
        pytest.param("\talice", id="leading-tab"),
        pytest.param("alice\t", id="trailing-tab"),
        pytest.param(" \talice\t ", id="space-wrapped-edge-tabs"),
        pytest.param(
            "[item[index] for item in portfolio]",
            id="nested-bracket-list-comprehension",
        ),
    ],
)
def test_review_followup_scanner_rejects_raw_edge_tabs_and_nested_code(
    tmp_path: Path,
    unsafe_text: str,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "unsafe.txt").write_text(unsafe_text, encoding="utf-8")
    _write_manifest(root, ("unsafe.txt",))
    index, index_path, _ = _index(tmp_path, root)

    with pytest.raises(UnsafeContentError, match="citation content is not safe"):
        index.build(root_id="policy-root", scope_id="scope-a")

    assert not index_path.exists()


@pytest.mark.parametrize(
    "unsafe_path",
    [
        pytest.param("x" * (local_module._MAX_PATH_LENGTH + 1), id="oversized"),
        pytest.param("safe\x00path", id="nul"),
        pytest.param(
            os.sep.join("a" for _ in range(local_module._MAX_PATH_COMPONENTS + 1)),
            id="component-count",
        ),
        pytest.param(
            "a" * (local_module._MAX_PATH_COMPONENT + 1),
            id="component-length",
        ),
    ],
)
def test_task1_invalid_path_text_stops_before_path_or_abspath(
    monkeypatch: pytest.MonkeyPatch,
    unsafe_path: str,
) -> None:
    real_path = local_module.Path
    real_abspath = local_module.os.path.abspath
    path_calls = 0
    abspath_calls = 0

    def counted_path(*args: object, **kwargs: object) -> Path:
        nonlocal path_calls
        path_calls += 1
        return real_path(*args, **kwargs)  # type: ignore[arg-type]

    def counted_abspath(value: object) -> str:
        nonlocal abspath_calls
        abspath_calls += 1
        return real_abspath(value)  # type: ignore[arg-type]

    monkeypatch.setattr(local_module, "Path", counted_path)
    monkeypatch.setattr(local_module.os.path, "abspath", counted_abspath)

    with pytest.raises(ValueError):
        local_module._lexical_absolute(unsafe_path)  # noqa: SLF001

    assert path_calls == 0
    assert abspath_calls == 0


def test_task1_pathlike_values_are_passive_before_fspath(tmp_path: Path) -> None:
    class ActivePathLike(os.PathLike[str]):
        def __init__(self, value: Path) -> None:
            self.value = value
            self.calls = 0

        def __fspath__(self) -> str:
            self.calls += 1
            raise AssertionError("active __fspath__ executed")

    root = tmp_path / "policies"
    root.mkdir()
    cases = (
        (
            ActivePathLike(tmp_path / "active-index.json"),
            {
                "index_path": None,
                "roots": {},
            },
        ),
        (
            ActivePathLike(tmp_path / "active-index.json.key"),
            {
                "index_path": tmp_path / "active-index.json",
                "key_path": None,
                "roots": {},
            },
        ),
        (
            ActivePathLike(root),
            {
                "index_path": tmp_path / "active-root-index.json",
                "roots": {"policy-root": None},
            },
        ),
    )

    for active, raw_kwargs in cases:
        kwargs = dict(raw_kwargs)
        if kwargs["index_path"] is None:
            kwargs["index_path"] = active
        elif kwargs.get("key_path") is None and "key_path" in kwargs:
            kwargs["key_path"] = active
        else:
            roots = kwargs["roots"]
            assert isinstance(roots, dict)
            roots["policy-root"] = active
        with pytest.raises(
            UnsafeIndexRequestError,
            match="citation index request is not safe",
        ):
            LocalCitationIndex(**kwargs)  # type: ignore[arg-type]
        assert active.calls == 0

    from_strings = LocalCitationIndex(
        index_path=str(tmp_path / "string-index.json"),  # type: ignore[arg-type]
        key_path=str(tmp_path / "string-index.json.key"),  # type: ignore[arg-type]
        roots={"policy-root": str(root)},  # type: ignore[dict-item]
    )
    from_paths = LocalCitationIndex(
        index_path=tmp_path / "path-index.json",
        roots={"policy-root": root},
    )
    assert from_strings._index_path.name == "string-index.json"  # noqa: SLF001
    assert from_paths._index_path.name == "path-index.json"  # noqa: SLF001


def test_task1_scalar_text_limit_precedes_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import riskprobe.rag.models as models_module

    normalized: list[str] = []

    def observed_normalize(value: str) -> str:
        normalized.append(value)
        return value.replace("\r\n", "\n").replace("\r", "\n").strip()

    monkeypatch.setattr(
        models_module,
        "_normalize_safe_text",
        observed_normalize,
        raising=False,
    )
    assert models_module._validate_safe_text(
        "Aggregate controls remain stable.",
        allow_empty=False,
    ) == "Aggregate controls remain stable."
    assert normalized == ["Aggregate controls remain stable."]

    normalized.clear()
    with pytest.raises(ValueError, match="citation content is not safe"):
        models_module._validate_safe_text("x" * 262_145, allow_empty=False)
    assert normalized == []


def test_task1_scalar_title_limit_precedes_general_scanner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import riskprobe.rag.models as models_module

    scanner_calls = 0

    def active_scanner(value: str, *, allow_empty: bool) -> str:
        nonlocal scanner_calls
        del value, allow_empty
        scanner_calls += 1
        raise AssertionError("general scanner executed")

    monkeypatch.setattr(models_module, "_validate_safe_text", active_scanner)
    with pytest.raises(
        UnsafeIndexRequestError,
        match="citation index request is not safe",
    ):
        Citation.safe_validate(
            {
                "rank": 1,
                "citation_id": "a" * 64,
                "document_id": "b" * 64,
                "content_hash": "c" * 64,
                "score": 0.5,
                "title": "t" * 257,
            }
        )
    assert scanner_calls == 0


def test_task1_scalar_provider_and_request_codes_preflight_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import riskprobe.rag.models as models_module

    class ActiveMetrics(dict[str, str]):
        def __init__(self) -> None:
            super().__init__(models_module._PROVIDER_METRICS)
            self.lookups = 0

        def __contains__(self, key: object) -> bool:
            self.lookups += 1
            return super().__contains__(key)

    class ActivePattern:
        def __init__(self) -> None:
            self.calls = 0

        def fullmatch(self, value: str) -> None:
            del value
            self.calls += 1
            return None

    metrics = ActiveMetrics()
    monkeypatch.setattr(models_module, "_PROVIDER_METRICS", metrics)
    with pytest.raises(UnsafeContentError, match="citation content is not safe"):
        ProviderSafeSummary.safe_validate(
            {
                "operation": "aggregate_status",
                "metric_code": "m" * 65,
                "status_code": "stable",
                "aggregate_count": 1,
            }
        )
    assert metrics.lookups == 0

    opaque = ActivePattern()
    root_code = ActivePattern()
    monkeypatch.setattr(local_module, "_OPAQUE_ID", opaque)
    monkeypatch.setattr(local_module, "_ROOT_CODE", root_code)
    with pytest.raises(ValueError):
        local_module._validated_request_code("r" * 133, kind="root")
    assert opaque.calls == 0
    assert root_code.calls == 0


def _task1_direct_build_index(
    tmp_path: Path,
) -> tuple[LocalCitationIndex, Path]:
    root = tmp_path / "direct-policies"
    root.mkdir()
    return (
        LocalCitationIndex(
            index_path=tmp_path / "direct-index.json",
            roots={"policy-root": root},
        ),
        root,
    )


@pytest.mark.parametrize(
    ("max_documents", "max_terms", "term_sizes"),
    [
        pytest.param(2, 10, (1, 1), id="document-limit"),
        pytest.param(10, 3, (2, 1), id="term-limit"),
        pytest.param(10, 0, (0,), id="zero-term-lower-bound"),
    ],
)
def test_task1_build_total_budget_exact_limits_succeed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_documents: int,
    max_terms: int,
    term_sizes: tuple[int, ...],
) -> None:
    index, root = _task1_direct_build_index(tmp_path)
    key = b"k" * 32
    root_id = f"root-{'a' * 24}"
    scope_id = f"scope-{'b' * 24}"
    sources = tuple(
        {
            "content_hash": f"{position + 1:064x}",
            "source_bytes": 1,
            "text": f"aggregate controls {position}",
        }
        for position in range(len(term_sizes))
    )
    real_sealed = local_module._sealed_index
    real_canonical = local_module._canonical_bytes
    record_calls = 0
    seal_calls = 0
    canonical_calls = 0
    writes: list[object] = []

    def controlled_record(
        *,
        text: str,
        content_hash: str,
        key: bytes,
    ) -> dict[str, object]:
        nonlocal record_calls
        del text, key
        position = record_calls
        record_calls += 1
        terms = {
            f"{position * 100 + term + 1:064x}": 1
            for term in range(term_sizes[position])
        }
        return {
            "content_hash": content_hash,
            "document_id": hashlib.sha256(
                b"document\0" + content_hash.encode("ascii")
            ).hexdigest(),
            "terms": terms,
        }

    def counted_seal(scopes: object, seal_key: bytes) -> dict[str, object]:
        nonlocal seal_calls
        seal_calls += 1
        return real_sealed(scopes, seal_key)  # type: ignore[arg-type]

    def counted_canonical(value: object) -> bytes:
        nonlocal canonical_calls
        canonical_calls += 1
        return real_canonical(value)

    def counted_write(_storage: object, payload: object) -> None:
        writes.append(payload)

    monkeypatch.setattr(local_module, "_MAX_TOTAL_DOCUMENTS", max_documents)
    monkeypatch.setattr(local_module, "_MAX_TOTAL_TERMS", max_terms)
    monkeypatch.setattr(local_module, "_document_record", controlled_record)
    monkeypatch.setattr(local_module, "_sealed_index", counted_seal)
    monkeypatch.setattr(local_module, "_canonical_bytes", counted_canonical)
    monkeypatch.setattr(index, "_load_index", lambda _storage, _key: {"scopes": {}})
    monkeypatch.setattr(index, "_root_sources", lambda _root: iter(sources))
    monkeypatch.setattr(index, "_provider_sources", lambda _summaries: iter(()))
    monkeypatch.setattr(index, "_write_index", counted_write)

    result = index._build_locked(  # noqa: SLF001
        storage=object(),  # type: ignore[arg-type]
        root=root,
        root_id=root_id,
        scope_id=scope_id,
        provider_summaries=(),
        key=key,
    )

    assert result.document_count == len(term_sizes)
    assert record_calls == len(term_sizes)
    assert seal_calls == 1
    assert canonical_calls > 0
    assert len(writes) == 1


@pytest.mark.parametrize(
    ("max_documents", "max_terms", "term_sizes", "expected_record_calls"),
    [
        pytest.param(2, 10, (1, 1, 1), 2, id="document-limit-plus-one"),
        pytest.param(10, 3, (2, 2, 1), 2, id="term-limit-plus-one"),
        pytest.param(10, 0, (1, 1), 1, id="zero-term-limit-plus-one"),
    ],
)
def test_task1_build_total_budget_overage_stops_before_next_pipeline_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_documents: int,
    max_terms: int,
    term_sizes: tuple[int, ...],
    expected_record_calls: int,
) -> None:
    index, root = _task1_direct_build_index(tmp_path)
    sources = tuple(
        {
            "content_hash": f"{position + 1:064x}",
            "source_bytes": 1,
            "text": f"aggregate controls {position}",
        }
        for position in range(len(term_sizes))
    )
    real_sealed = local_module._sealed_index
    real_canonical = local_module._canonical_bytes
    record_calls = 0
    seal_calls = 0
    canonical_calls = 0
    write_calls = 0

    def controlled_record(
        *,
        text: str,
        content_hash: str,
        key: bytes,
    ) -> dict[str, object]:
        nonlocal record_calls
        del text, key
        position = record_calls
        record_calls += 1
        terms = {
            f"{position * 100 + term + 1:064x}": 1
            for term in range(term_sizes[position])
        }
        return {
            "content_hash": content_hash,
            "document_id": hashlib.sha256(
                b"document\0" + content_hash.encode("ascii")
            ).hexdigest(),
            "terms": terms,
        }

    def counted_seal(scopes: object, seal_key: bytes) -> dict[str, object]:
        nonlocal seal_calls
        seal_calls += 1
        return real_sealed(scopes, seal_key)  # type: ignore[arg-type]

    def counted_canonical(value: object) -> bytes:
        nonlocal canonical_calls
        canonical_calls += 1
        return real_canonical(value)

    def counted_write(_storage: object, _payload: object) -> None:
        nonlocal write_calls
        write_calls += 1

    monkeypatch.setattr(local_module, "_MAX_TOTAL_DOCUMENTS", max_documents)
    monkeypatch.setattr(local_module, "_MAX_TOTAL_TERMS", max_terms)
    monkeypatch.setattr(local_module, "_document_record", controlled_record)
    monkeypatch.setattr(local_module, "_sealed_index", counted_seal)
    monkeypatch.setattr(local_module, "_canonical_bytes", counted_canonical)
    monkeypatch.setattr(index, "_load_index", lambda _storage, _key: {"scopes": {}})
    monkeypatch.setattr(index, "_root_sources", lambda _root: iter(sources))
    monkeypatch.setattr(index, "_provider_sources", lambda _summaries: iter(()))
    monkeypatch.setattr(index, "_write_index", counted_write)

    with pytest.raises(IndexIntegrityError, match="citation index integrity check failed"):
        index._build_locked(  # noqa: SLF001
            storage=object(),  # type: ignore[arg-type]
            root=root,
            root_id=f"root-{'a' * 24}",
            scope_id=f"scope-{'b' * 24}",
            provider_summaries=(),
            key=b"k" * 32,
        )

    assert record_calls == expected_record_calls
    assert seal_calls == 0
    assert canonical_calls == 0
    assert write_calls == 0


@pytest.mark.parametrize(
    ("max_source_bytes", "should_succeed"),
    [
        pytest.param(2, True, id="exact-limit"),
        pytest.param(1, False, id="limit-plus-one"),
    ],
)
def test_task1_build_root_then_provider_share_source_byte_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_source_bytes: int,
    should_succeed: bool,
) -> None:
    index, root = _task1_direct_build_index(tmp_path)
    root_source = {"content_hash": "1" * 64, "source_bytes": 1, "text": "root"}
    provider_source = {
        "content_hash": "2" * 64,
        "source_bytes": 1,
        "text": "provider",
    }
    events: list[str] = []
    real_record = local_module._document_record
    real_sealed = local_module._sealed_index
    real_canonical = local_module._canonical_bytes
    record_calls = 0
    seal_calls = 0
    canonical_calls = 0
    write_calls = 0

    def root_sources(_root: Path) -> Iterator[dict[str, object]]:
        events.append("root:start")
        yield root_source
        events.append("root:end")

    def provider_sources(_summaries: object) -> Iterator[dict[str, object]]:
        events.append("provider:start")
        yield provider_source
        events.append("provider:end")

    def counted_record(**kwargs: object) -> dict[str, object]:
        nonlocal record_calls
        record_calls += 1
        return real_record(**kwargs)  # type: ignore[arg-type]

    def counted_seal(scopes: object, key: bytes) -> dict[str, object]:
        nonlocal seal_calls
        seal_calls += 1
        return real_sealed(scopes, key)  # type: ignore[arg-type]

    def counted_canonical(value: object) -> bytes:
        nonlocal canonical_calls
        canonical_calls += 1
        return real_canonical(value)

    def counted_write(_storage: object, _payload: object) -> None:
        nonlocal write_calls
        write_calls += 1

    monkeypatch.setattr(local_module, "_MAX_TOTAL_SOURCE_BYTES", max_source_bytes)
    monkeypatch.setattr(local_module, "_document_record", counted_record)
    monkeypatch.setattr(local_module, "_sealed_index", counted_seal)
    monkeypatch.setattr(local_module, "_canonical_bytes", counted_canonical)
    monkeypatch.setattr(index, "_load_index", lambda _storage, _key: {"scopes": {}})
    monkeypatch.setattr(index, "_root_sources", root_sources)
    monkeypatch.setattr(index, "_provider_sources", provider_sources)
    monkeypatch.setattr(index, "_write_index", counted_write)

    call = lambda: index._build_locked(  # noqa: E731, SLF001
        storage=object(),  # type: ignore[arg-type]
        root=root,
        root_id=f"root-{'a' * 24}",
        scope_id=f"scope-{'b' * 24}",
        provider_summaries=(),
        key=b"k" * 32,
    )
    if should_succeed:
        result = call()
        assert result.document_count == 2
        assert events == ["root:start", "root:end", "provider:start", "provider:end"]
        assert record_calls == 2
        assert seal_calls == 1
        assert canonical_calls > 0
        assert write_calls == 1
    else:
        with pytest.raises(UnsafeContentError, match="citation content is not safe"):
            call()
        assert events == ["root:start", "root:end", "provider:start"]
        assert record_calls == 1
        assert seal_calls == 0
        assert canonical_calls == 0
        assert write_calls == 0


def test_task1_build_root_then_provider_share_content_dedupe_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index, root = _task1_direct_build_index(tmp_path)
    text = "Aggregate controls remain stable."
    source = {
        "content_hash": "a" * 64,
        "source_bytes": len(text.encode("utf-8")),
        "text": text,
    }
    events: list[str] = []
    real_term_counts = local_module._term_counts
    term_calls = 0
    write_calls = 0

    def root_sources(_root: Path) -> Iterator[dict[str, object]]:
        events.append("root:start")
        yield source
        events.append("root:end")

    def provider_sources(_summaries: object) -> Iterator[dict[str, object]]:
        events.append("provider:start")
        yield dict(source)
        events.append("provider:end")

    def counted_terms(*args: object, **kwargs: object) -> dict[str, int]:
        nonlocal term_calls
        term_calls += 1
        return real_term_counts(*args, **kwargs)  # type: ignore[arg-type]

    def counted_write(_storage: object, _payload: object) -> None:
        nonlocal write_calls
        write_calls += 1

    monkeypatch.setattr(local_module, "_term_counts", counted_terms)
    monkeypatch.setattr(index, "_load_index", lambda _storage, _key: {"scopes": {}})
    monkeypatch.setattr(index, "_root_sources", root_sources)
    monkeypatch.setattr(index, "_provider_sources", provider_sources)
    monkeypatch.setattr(index, "_write_index", counted_write)

    result = index._build_locked(  # noqa: SLF001
        storage=object(),  # type: ignore[arg-type]
        root=root,
        root_id=f"root-{'a' * 24}",
        scope_id=f"scope-{'b' * 24}",
        provider_summaries=(),
        key=b"k" * 32,
    )

    assert events == ["root:start", "root:end", "provider:start", "provider:end"]
    assert result.document_count == 1
    assert term_calls == 1
    assert write_calls == 1


def test_task1_build_same_content_hash_with_different_text_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index, root = _task1_direct_build_index(tmp_path)
    shared_hash = "a" * 64
    root_source = {
        "content_hash": shared_hash,
        "source_bytes": 4,
        "text": "root",
    }
    provider_source = {
        "content_hash": shared_hash,
        "source_bytes": 8,
        "text": "provider",
    }
    events: list[str] = []
    real_term_counts = local_module._term_counts
    real_sealed = local_module._sealed_index
    real_canonical = local_module._canonical_bytes
    term_calls = 0
    seal_calls = 0
    canonical_calls = 0
    write_calls = 0

    def root_sources(_root: Path) -> Iterator[dict[str, object]]:
        events.append("root:start")
        yield root_source
        events.append("root:end")

    def provider_sources(_summaries: object) -> Iterator[dict[str, object]]:
        events.append("provider:start")
        yield provider_source
        events.append("provider:end")

    def counted_terms(*args: object, **kwargs: object) -> dict[str, int]:
        nonlocal term_calls
        term_calls += 1
        return real_term_counts(*args, **kwargs)  # type: ignore[arg-type]

    def counted_seal(scopes: object, key: bytes) -> dict[str, object]:
        nonlocal seal_calls
        seal_calls += 1
        return real_sealed(scopes, key)  # type: ignore[arg-type]

    def counted_canonical(value: object) -> bytes:
        nonlocal canonical_calls
        canonical_calls += 1
        return real_canonical(value)

    def counted_write(_storage: object, _payload: object) -> None:
        nonlocal write_calls
        write_calls += 1

    monkeypatch.setattr(local_module, "_term_counts", counted_terms)
    monkeypatch.setattr(local_module, "_sealed_index", counted_seal)
    monkeypatch.setattr(local_module, "_canonical_bytes", counted_canonical)
    monkeypatch.setattr(index, "_load_index", lambda _storage, _key: {"scopes": {}})
    monkeypatch.setattr(index, "_root_sources", root_sources)
    monkeypatch.setattr(index, "_provider_sources", provider_sources)
    monkeypatch.setattr(index, "_write_index", counted_write)

    with pytest.raises(IndexIntegrityError, match="citation index integrity check failed"):
        index._build_locked(  # noqa: SLF001
            storage=object(),  # type: ignore[arg-type]
            root=root,
            root_id=f"root-{'a' * 24}",
            scope_id=f"scope-{'b' * 24}",
            provider_summaries=(),
            key=b"k" * 32,
        )

    assert events == ["root:start", "root:end", "provider:start"]
    assert term_calls == 1
    assert seal_calls == 0
    assert canonical_calls == 0
    assert write_calls == 0


def test_task1_build_source_budget_stops_before_second_record_and_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    _write_manifest(root, ())
    index, index_path, _ = _index(tmp_path, root)
    summaries = (
        {
            "operation": "aggregate_status",
            "metric_code": "approval_rate",
            "status_code": "stable",
            "aggregate_count": 1,
        },
        {
            "operation": "aggregate_status",
            "metric_code": "liquidity_rate",
            "status_code": "stable",
            "aggregate_count": 2,
        },
    )
    first_text = ProviderSafeSummary.safe_validate(summaries[0]).text
    monkeypatch.setattr(
        local_module,
        "_MAX_TOTAL_SOURCE_BYTES",
        len(first_text.encode("utf-8")),
        raising=False,
    )
    real_record = local_module._document_record
    real_seal = local_module._sealed_index
    record_calls = 0
    seal_calls = 0

    def counted_record(**kwargs: object) -> dict[str, object]:
        nonlocal record_calls
        record_calls += 1
        return real_record(**kwargs)  # type: ignore[arg-type]

    def counted_seal(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal seal_calls
        seal_calls += 1
        return real_seal(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(local_module, "_document_record", counted_record)
    monkeypatch.setattr(local_module, "_sealed_index", counted_seal)

    with pytest.raises(UnsafeContentError, match="citation content is not safe"):
        index.build(
            root_id="policy-root",
            scope_id="scope-a",
            provider_summaries=summaries,
        )

    assert record_calls == 1
    assert seal_calls == 0
    assert not index_path.exists()


def test_task1_build_full_quota_allows_replacement_but_rejects_new_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, index_path, _ = _index(tmp_path, root)
    index.build(root_id="policy-root", scope_id="scope-a")
    monkeypatch.setattr(local_module, "_MAX_TOTAL_DOCUMENTS", 1)
    record_calls = 0
    seal_calls = 0
    write_calls = 0
    real_record = local_module._document_record
    real_seal = local_module._sealed_index
    real_write = LocalCitationIndex._write_index

    def counted_record(**kwargs: object) -> dict[str, object]:
        nonlocal record_calls
        record_calls += 1
        return real_record(**kwargs)  # type: ignore[arg-type]

    def counted_seal(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal seal_calls
        seal_calls += 1
        return real_seal(*args, **kwargs)  # type: ignore[arg-type]

    def counted_write(
        self: LocalCitationIndex,
        storage: object,
        payload: object,
    ) -> None:
        nonlocal write_calls
        write_calls += 1
        real_write(self, storage, payload)  # type: ignore[arg-type]

    monkeypatch.setattr(local_module, "_document_record", counted_record)
    monkeypatch.setattr(local_module, "_sealed_index", counted_seal)
    monkeypatch.setattr(LocalCitationIndex, "_write_index", counted_write)

    replacement = index.build(root_id="policy-root", scope_id="scope-a")

    assert replacement.document_count == 1
    assert record_calls == 1
    assert seal_calls == 1
    assert write_calls == 1

    record_calls = 0
    seal_calls = 0
    write_calls = 0
    original = index_path.read_bytes()
    with pytest.raises(IndexIntegrityError, match="citation index integrity check failed"):
        index.build(root_id="policy-root", scope_id="scope-b")

    assert record_calls == 0
    assert seal_calls == 0
    assert write_calls == 0
    assert index_path.read_bytes() == original


def test_task1_build_full_term_quota_allows_replacement_but_rejects_new_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, index_path, _ = _index(tmp_path, root)
    index.build(root_id="policy-root", scope_id="scope-a")

    persisted = json.loads(index_path.read_text(encoding="utf-8"))
    scopes = persisted["scopes"]
    assert isinstance(scopes, dict)
    scope = next(iter(scopes.values()))
    assert isinstance(scope, dict)
    documents = scope["documents"]
    assert isinstance(documents, list)
    assert len(documents) == 1
    document = documents[0]
    assert isinstance(document, dict)
    terms = document["terms"]
    assert isinstance(terms, dict)
    term_limit = len(terms)
    assert term_limit > 0

    monkeypatch.setattr(local_module, "_MAX_TOTAL_TERMS", term_limit)
    record_calls = 0
    seal_calls = 0
    write_calls = 0
    real_record = local_module._document_record
    real_seal = local_module._sealed_index
    real_write = LocalCitationIndex._write_index

    def counted_record(**kwargs: object) -> dict[str, object]:
        nonlocal record_calls
        record_calls += 1
        return real_record(**kwargs)  # type: ignore[arg-type]

    def counted_seal(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal seal_calls
        seal_calls += 1
        return real_seal(*args, **kwargs)  # type: ignore[arg-type]

    def counted_write(
        self: LocalCitationIndex,
        storage: object,
        payload: object,
    ) -> None:
        nonlocal write_calls
        write_calls += 1
        real_write(self, storage, payload)  # type: ignore[arg-type]

    monkeypatch.setattr(local_module, "_document_record", counted_record)
    monkeypatch.setattr(local_module, "_sealed_index", counted_seal)
    monkeypatch.setattr(LocalCitationIndex, "_write_index", counted_write)

    replacement = index.build(root_id="policy-root", scope_id="scope-a")

    assert replacement.document_count == 1
    assert record_calls == 1
    assert seal_calls == 1
    assert write_calls == 1

    record_calls = 0
    seal_calls = 0
    write_calls = 0
    original = index_path.read_bytes()
    with pytest.raises(IndexIntegrityError) as exc_info:
        index.build(root_id="policy-root", scope_id="scope-b")

    assert str(exc_info.value) == "citation index integrity check failed"
    assert record_calls == 1
    assert seal_calls == 0
    assert write_calls == 0
    assert index_path.read_bytes() == original


def test_task1_build_duplicate_content_generates_one_term_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    _write_manifest(root, ())
    index, _, _ = _index(tmp_path, root)
    summary = {
        "operation": "aggregate_status",
        "metric_code": "control_coverage",
        "status_code": "stable",
        "aggregate_count": 4,
    }
    term_calls = 0
    real_term_counts = local_module._term_counts

    def counted_terms(*args: object, **kwargs: object) -> dict[str, int]:
        nonlocal term_calls
        term_calls += 1
        return real_term_counts(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(local_module, "_term_counts", counted_terms)

    result = index.build(
        root_id="policy-root",
        scope_id="scope-a",
        provider_summaries=(summary, summary),
    )

    assert result.document_count == 1
    assert term_calls == 1


@pytest.mark.parametrize("fault_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_task1_postcommit_body_fault_is_always_uncertain_and_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_type: type[BaseException],
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    (root / "policy.txt").write_text("Aggregate policy controls.", encoding="utf-8")
    _write_manifest(root, ("policy.txt",))
    index, _, _ = _index(tmp_path, root)
    index.build(root_id="policy-root", scope_id="scope-a")
    real_write = LocalCitationIndex._write_index
    marker = f"postcommit-body-{fault_type.__name__}-private-marker"

    def commit_then_fault(
        self: LocalCitationIndex,
        storage: object,
        payload: object,
    ) -> None:
        real_write(self, storage, payload)  # type: ignore[arg-type]
        raise fault_type(marker)

    monkeypatch.setattr(LocalCitationIndex, "_write_index", commit_then_fault)
    error = _capture_review_regression_error(
        lambda: index.build(root_id="policy-root", scope_id="scope-b")
    )

    assert isinstance(error, IndexIntegrityError)
    assert str(error) == "citation index commit may have occurred"
    _assert_clean_error(error, marker)
    _assert_rag_traceback_locals_clean(error, rejected=(), markers=(marker,))
    assert index.query(
        scope_id="scope-b",
        query_id="query-published",
        query_text="policy",
    ).citations


def test_task1_build_result_is_constructed_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "policies"
    root.mkdir()
    _write_manifest(root, ())
    index, _, _ = _index(tmp_path, root)
    events: list[str] = []
    real_result = local_module.BuildResult
    real_write = LocalCitationIndex._write_index

    def tracked_result(*args: object, **kwargs: object) -> BuildResult:
        events.append("result")
        return real_result(*args, **kwargs)  # type: ignore[arg-type]

    def tracked_write(
        self: LocalCitationIndex,
        storage: object,
        payload: object,
    ) -> None:
        events.append("write")
        real_write(self, storage, payload)  # type: ignore[arg-type]

    monkeypatch.setattr(local_module, "BuildResult", tracked_result)
    monkeypatch.setattr(LocalCitationIndex, "_write_index", tracked_write)

    index.build(root_id="policy-root", scope_id="scope-a")

    assert events == ["result", "write"]


def test_task1_shape_preflight_rejects_wrong_keys_before_hash_or_pydantic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ActiveCollisionKey:
        def __init__(self, target: str) -> None:
            self.target = target
            self.hashes = 0

        def __hash__(self) -> int:
            self.hashes += 1
            return hash(self.target)

        def __eq__(self, other: object) -> bool:
            return other == self.target

        def reset(self) -> None:
            self.hashes = 0

    key = ActiveCollisionKey("extra")
    manifest = {"format_version": 1, "documents": [], key: None}
    key.reset()
    with pytest.raises((TypeError, ValueError)):
        local_module._validate_manifest(manifest)
    assert key.hashes == 0

    persisted = {
        "format_version": 4,
        "index_hash": "a" * 64,
        "key_id": "b" * 64,
        "scopes": {},
        "seal": "c" * 64,
        key: None,
    }
    key.reset()
    with pytest.raises((TypeError, ValueError)):
        local_module._validate_persisted_index(persisted, b"k" * 32)
    assert key.hashes == 0

    scope_id = f"scope-{'a' * 24}"
    scope = {
        "documents": [],
        "root_id": f"root-{'b' * 24}",
        "scope_id": scope_id,
        "seal": "c" * 64,
        key: None,
    }
    key.reset()
    with pytest.raises((TypeError, ValueError)):
        local_module._validate_scope(scope_id, scope, b"k" * 32)
    assert key.hashes == 0

    build_calls = 0
    citation_calls = 0

    def active_build_validation(cls: type[BuildResult], value: object) -> BuildResult:
        nonlocal build_calls
        del cls, value
        build_calls += 1
        raise AssertionError("build validation executed")

    def active_citation_validation(cls: type[Citation], value: object) -> Citation:
        nonlocal citation_calls
        del cls, value
        citation_calls += 1
        raise AssertionError("citation validation executed")

    monkeypatch.setattr(BuildResult, "model_validate", classmethod(active_build_validation))
    monkeypatch.setattr(Citation, "model_validate", classmethod(active_citation_validation))
    with pytest.raises(UnsafeIndexRequestError, match="citation index request is not safe"):
        BuildResult.safe_validate(
            {
                "scope_id": scope_id,
                "document_count": 1,
                "unexpected": "a" * 64,
            }
        )
    with pytest.raises(UnsafeIndexRequestError, match="citation index request is not safe"):
        QueryResult.safe_validate(
            {
                "scope_id": scope_id,
                "citations": (
                    {
                        "unexpected": 1,
                        "citation_id": "a" * 64,
                        "document_id": "b" * 64,
                        "content_hash": "c" * 64,
                        "score": 0.5,
                        "title": None,
                    },
                ),
            }
        )
    assert build_calls == 0
    assert citation_calls == 0


def test_task1_path_lock_registry_concurrent_creation_has_one_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ControlledRegistry:
        def __init__(self) -> None:
            self.entries: dict[str, object] = {}
            self.get_calls = 0
            self.set_calls = 0
            self.first_get_entered = threading.Event()
            self.second_get_entered = threading.Event()
            self.release_first_get = threading.Event()
            self._calls_guard = threading.Lock()

        def get(self, key: str) -> object | None:
            with self._calls_guard:
                self.get_calls += 1
                call_number = self.get_calls
            if call_number == 1:
                self.first_get_entered.set()
                if not self.release_first_get.wait(timeout=2):
                    raise AssertionError("first registry get release timed out")
            else:
                self.second_get_entered.set()
            return self.entries.get(key)

        def __setitem__(self, key: str, value: object) -> None:
            self.set_calls += 1
            self.entries[key] = value

    path = tmp_path / "concurrent-registry-index.json"
    registry = ControlledRegistry()
    monkeypatch.setattr(local_module, "_PATH_LOCKS", registry)
    returned: list[object | None] = [None, None]
    errors: list[BaseException] = []
    second_started = threading.Event()
    threads: list[threading.Thread] = []

    def obtain_lock(position: int) -> None:
        try:
            if position == 1:
                second_started.set()
            returned[position] = local_module._path_lock(path)
        except BaseException as error:
            errors.append(error)

    second_entered_before_release = False
    try:
        first = threading.Thread(target=obtain_lock, args=(0,))
        first.start()
        threads.append(first)
        assert registry.first_get_entered.wait(timeout=2)

        second = threading.Thread(target=obtain_lock, args=(1,))
        second.start()
        threads.append(second)
        assert second_started.wait(timeout=2)
        second_entered_before_release = registry.second_get_entered.wait(timeout=0.2)
    finally:
        registry.release_first_get.set()
        for thread in threads:
            thread.join(timeout=2)

    assert not second_entered_before_release
    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert registry.second_get_entered.is_set()
    assert registry.get_calls == 2
    assert registry.set_calls == 1
    assert returned[0] is not None
    assert returned[0] is returned[1]


def test_task1_path_lock_registry_keeps_waited_lock_alive_until_threads_release(
    tmp_path: Path,
) -> None:
    import gc
    import weakref

    root = tmp_path / "policies"
    root.mkdir()
    index_path = tmp_path / "waiting-registry-index.json"
    index = LocalCitationIndex(index_path=index_path, roots={"policy-root": root})
    lock = index._thread_lock  # noqa: SLF001
    registry_key = os.fspath(index._index_path)  # noqa: SLF001
    lock_reference = weakref.ref(lock)
    holder_acquired = threading.Event()
    release_holder = threading.Event()
    waiter_ready = threading.Event()
    waiter_acquired = threading.Event()
    release_waiter = threading.Event()
    waiter_same_identity: list[bool] = []
    errors: list[BaseException] = []

    def hold(candidate: Any) -> None:
        try:
            with candidate:
                holder_acquired.set()
                if not release_holder.wait(timeout=2):
                    raise AssertionError("holder release timed out")
        except BaseException as error:
            errors.append(error)

    def wait_for_same_lock() -> None:
        try:
            candidate = local_module._path_lock(index_path)
            waiter_same_identity.append(candidate is lock_reference())
            waiter_ready.set()
            with candidate:
                waiter_acquired.set()
                if not release_waiter.wait(timeout=2):
                    raise AssertionError("waiter release timed out")
        except BaseException as error:
            errors.append(error)

    holder = threading.Thread(target=hold, args=(lock,))
    holder.start()
    assert holder_acquired.wait(timeout=2)
    waiter = threading.Thread(target=wait_for_same_lock)
    waiter.start()
    assert waiter_ready.wait(timeout=2)
    assert not waiter_acquired.is_set()

    del index, lock
    gc.collect()
    assert lock_reference() is not None
    assert local_module._PATH_LOCKS.get(registry_key) is lock_reference()

    release_holder.set()
    assert waiter_acquired.wait(timeout=2)
    assert waiter_same_identity == [True]
    assert local_module._PATH_LOCKS.get(registry_key) is lock_reference()
    release_waiter.set()
    holder.join(timeout=2)
    waiter.join(timeout=2)

    assert not holder.is_alive()
    assert not waiter.is_alive()
    assert errors == []

    del holder, waiter
    gc.collect()
    assert lock_reference() is None
    assert registry_key not in local_module._PATH_LOCKS


def test_task1_path_lock_registry_shares_live_locks_and_reclaims_dead_entries(
    tmp_path: Path,
) -> None:
    import gc
    import weakref

    root = tmp_path / "policies"
    root.mkdir()
    index_path = tmp_path / "registry-index.json"
    first = LocalCitationIndex(index_path=index_path, roots={"policy-root": root})
    second = LocalCitationIndex(index_path=index_path, roots={"policy-root": root})
    registry_key = os.fspath(first._index_path)  # noqa: SLF001

    assert first._thread_lock is second._thread_lock  # noqa: SLF001
    lock_reference = weakref.ref(first._thread_lock)  # noqa: SLF001
    del first
    gc.collect()
    assert local_module._PATH_LOCKS.get(registry_key) is second._thread_lock  # noqa: SLF001

    del second
    gc.collect()
    assert registry_key not in local_module._PATH_LOCKS
    assert lock_reference() is None


def test_task1_shape_nested_persisted_preflight_precedes_canonicalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ActiveCollisionKey:
        def __init__(self) -> None:
            self.hashes = 0

        def __hash__(self) -> int:
            self.hashes += 1
            return hash("extra")

        def reset(self) -> None:
            self.hashes = 0

    active_key = ActiveCollisionKey()
    malformed_document = {
        "citation_id": "a" * 64,
        "content_hash": "b" * 64,
        "document_id": "c" * 64,
        "terms": {},
        active_key: None,
    }
    active_key.reset()
    canonical_calls = 0

    def active_canonical(value: object) -> bytes:
        nonlocal canonical_calls
        del value
        canonical_calls += 1
        raise AssertionError("canonicalization executed")

    monkeypatch.setattr(local_module, "_canonical_bytes", active_canonical)
    scope_id = f"scope-{'a' * 24}"
    with pytest.raises(ValueError):
        local_module._validate_scope(
            scope_id,
            {
                "documents": [malformed_document],
                "root_id": f"root-{'b' * 24}",
                "scope_id": scope_id,
                "seal": "c" * 64,
            },
            b"k" * 32,
        )

    assert active_key.hashes == 0
    assert canonical_calls == 0


def test_task1_shape_term_length_precedes_sort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sort_calls = 0

    def active_sorted(value: object) -> list[object]:
        nonlocal sort_calls
        del value
        sort_calls += 1
        raise AssertionError("term sorting executed")

    content_hash = "a" * 64
    document_id = local_module._document_id(content_hash)
    scope_id = f"scope-{'b' * 24}"
    key = b"k" * 32
    documents = [
        {
            "citation_id": local_module._citation_id(scope_id, document_id),
            "content_hash": content_hash,
            "document_id": document_id,
            "terms": {"t" * 65: 1},
        }
    ]
    core = {
        "documents": documents,
        "root_id": f"root-{'c' * 24}",
        "scope_id": scope_id,
    }
    scope = {
        **core,
        "seal": local_module._hmac_digest(
            key,
            b"scope\0" + local_module._canonical_bytes(core),
        ),
    }
    monkeypatch.setattr(local_module, "sorted", active_sorted, raising=False)

    with pytest.raises(ValueError):
        local_module._validate_scope(scope_id, scope, key)

    assert sort_calls == 0


def test_task1_scalar_numeric_dto_preflight_precedes_pydantic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_calls = 0
    citation_calls = 0

    def active_build_validation(cls: type[BuildResult], value: object) -> BuildResult:
        nonlocal build_calls
        del cls, value
        build_calls += 1
        raise AssertionError("build validation executed")

    def active_citation_validation(cls: type[Citation], value: object) -> Citation:
        nonlocal citation_calls
        del cls, value
        citation_calls += 1
        raise AssertionError("citation validation executed")

    monkeypatch.setattr(BuildResult, "model_validate", classmethod(active_build_validation))
    monkeypatch.setattr(Citation, "model_validate", classmethod(active_citation_validation))
    scope_id = f"scope-{'a' * 24}"
    with pytest.raises(UnsafeIndexRequestError, match="citation index request is not safe"):
        BuildResult.safe_validate(
            {
                "scope_id": scope_id,
                "document_count": True,
                "index_hash": "b" * 64,
            }
        )
    with pytest.raises(UnsafeIndexRequestError, match="citation index request is not safe"):
        Citation.safe_validate(
            {
                "rank": 101,
                "citation_id": "a" * 64,
                "document_id": "b" * 64,
                "content_hash": "c" * 64,
                "score": float("nan"),
            }
        )

    assert build_calls == 0
    assert citation_calls == 0
