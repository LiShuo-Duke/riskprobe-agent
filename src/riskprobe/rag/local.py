"""Offline, deterministic, scope-isolated local citation index.

A root document is eligible only when a mandatory versioned manifest explicitly
attests its relative path, exact byte hash, and ``provider_safe`` privacy class.
That attestation is a human trust decision, not an automatic regex proof; the text
scanner remains defense in depth. Token terms and integrity seals use a separate
owner-only HMAC key. An actor able to read both that key and the index under the
same UID is outside this integrity boundary. Pure stdlib path hardening also cannot
fully defeat a same-UID actor that can rewrite arbitrary ancestor components.
"""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import threading
import unicodedata
import weakref
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer

from riskprobe.privacy import assert_safe_payload
from riskprobe.rag.models import (
    BuildResult,
    Citation,
    IndexIntegrityError,
    ProviderSafeSummary,
    QueryResult,
    UnsafeContentError,
    UnsafeIndexRequestError,
    _canonical_bytes,
    _raise_unlinked,
    _validate_safe_text,
)

_FORMAT_VERSION = 4
_MANIFEST_FORMAT_VERSION = 1
_MANIFEST_NAME = ".riskprobe-rag-manifest.json"
_ALLOWED_SUFFIXES = frozenset({".md", ".txt"})
_FORBIDDEN_DIRECTORY_PREFIXES = (
    "artifact",
    "cache",
    "data",
    "dataset",
    "output",
    "parquet",
    "raw",
    "run",
)
_ROOT_CODE = re.compile(
    r"^(?:root-[a-z0-9][a-z0-9-]{0,126}|[a-z][a-z0-9-]{0,126}-root)$"
)
_SCOPE_CODE = re.compile(r"^scope-[a-z0-9][a-z0-9-]{0,121}$")
_QUERY_CODE = re.compile(r"^query-[a-z0-9][a-z0-9-]{0,121}$")
_OPAQUE_ID = re.compile(r"^(root|scope|query)-[0-9a-f]{24}$")
_WINDOWS_MANIFEST_PATH = re.compile(r"^[A-Za-z]:/")
_INDEX_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,126}\.json$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"(?u)\b[^\W\d_][\w-]{1,63}\b")
_MAX_FILE_BYTES = 262_144
_MAX_MANIFEST_BYTES = 1_048_576
_MAX_INDEX_BYTES = 64 * 1024 * 1024
_MAX_MANIFEST_DOCUMENTS = 4_096
_MAX_PROVIDER_SUMMARIES = 4_096
_MAX_REGISTERED_ROOTS = 1_024
_MAX_DOCUMENTS_PER_SCOPE = 8_192
_MAX_TOTAL_SOURCE_BYTES = _MAX_FILE_BYTES * _MAX_DOCUMENTS_PER_SCOPE
_MAX_SCOPES = 1_024
_MAX_TOTAL_DOCUMENTS = 131_072
_MAX_TERMS_PER_DOCUMENT = 16_384
_MAX_TOTAL_TERMS = 1_048_576
_MAX_TERM_COUNT = 1_000_000
_MAX_TREE_ENTRIES = 100_000
_MAX_STALE_TEMP_ENTRIES = 100_000
_MAX_TREE_DEPTH = 64
_MAX_PATH_LENGTH = 4_096
_MAX_PATH_COMPONENT = 255
_MAX_PATH_COMPONENTS = 256
_MAX_REQUEST_CODE_LENGTH = 132
_KEY_BYTES = 32
_LOCK_SUFFIX = ".lock"
_REQUEST_ERROR = "citation index request is not safe"
_CONTENT_ERROR = "citation content is not safe"
_INTEGRITY_ERROR = "citation index integrity check failed"
_UNCERTAIN_COMMIT_ERROR = "citation index commit may have occurred"

_MANIFEST_KEYS = frozenset({"documents", "format_version"})
_MANIFEST_DOCUMENT_KEYS = frozenset({"content_hash", "path", "privacy_class"})
_INDEX_KEYS = frozenset({"format_version", "index_hash", "key_id", "scopes", "seal"})
_SCOPE_KEYS = frozenset({"documents", "root_id", "scope_id", "seal"})
_DOCUMENT_KEYS = frozenset({"citation_id", "content_hash", "document_id", "terms"})
_CONCRETE_PATH_TYPE = type(Path())

_PATH_LOCKS: weakref.WeakValueDictionary[str, threading.RLock] = (
    weakref.WeakValueDictionary()
)
_PATH_LOCKS_GUARD = threading.Lock()


@dataclass(slots=True)
class _StorageSession:
    index_parent_fd: int
    key_parent_fd: int
    index_name: str
    key_name: str
    lock_name: str
    index_commit_attempted: bool = False
    index_committed: bool = False


class _RecoverablePartialKeyError(ValueError):
    """A legacy owner-only final key shorter than a complete key."""


class LocalCitationIndex:
    """Build/query a sealed index over manifest-attested, registered safe roots.

    Relative roots and storage paths are converted to lexical absolute paths during
    construction. The manifest is the explicit human attestation boundary; direct
    Pydantic construction of provider summaries is trusted-only, while every build
    re-dumps and calls ``ProviderSafeSummary.safe_validate`` before indexing.
    """

    def __init__(
        self,
        *,
        index_path: Path,
        roots: Mapping[str, Path],
        key_path: Path | None = None,
    ) -> None:
        try:
            (
                normalized_index,
                normalized_key,
                lock_path,
                normalized_roots,
                thread_lock,
            ) = _prepare_index_configuration(
                index_path=index_path,
                roots=roots,
                key_path=key_path,
            )
            self._index_path = normalized_index
            self._key_path = normalized_key
            self._lock_path = lock_path
            self._roots = normalized_roots
            self._thread_lock = thread_lock
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            del index_path, roots, key_path
            _raise_unlinked(UnsafeIndexRequestError, _REQUEST_ERROR)

    def build(
        self,
        *,
        root_id: str,
        scope_id: str,
        provider_summaries: Sequence[ProviderSafeSummary | Mapping[str, object]] = (),
    ) -> BuildResult:
        """Atomically replace one scope after manifest and privacy validation."""

        try:
            return self._build_request(
                root_id=root_id,
                scope_id=scope_id,
                provider_summaries=provider_summaries,
            )
        except IndexIntegrityError as caught:
            message = (
                _UNCERTAIN_COMMIT_ERROR
                if str(caught) == _UNCERTAIN_COMMIT_ERROR
                else _INTEGRITY_ERROR
            )
            caught = None
            root_id = ""
            scope_id = ""
            provider_summaries = ()
            _raise_unlinked(IndexIntegrityError, message)
        except UnsafeContentError:
            root_id = ""
            scope_id = ""
            provider_summaries = ()
            _raise_unlinked(UnsafeContentError, _CONTENT_ERROR)
        except UnsafeIndexRequestError:
            root_id = ""
            scope_id = ""
            provider_summaries = ()
            _raise_unlinked(UnsafeIndexRequestError, _REQUEST_ERROR)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            root_id = ""
            scope_id = ""
            provider_summaries = ()
            _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)

    def _build_request(
        self,
        *,
        root_id: str,
        scope_id: str,
        provider_summaries: Sequence[ProviderSafeSummary | Mapping[str, object]],
    ) -> BuildResult:
        with self._storage_lock() as storage:
            key = self._load_or_create_key(storage)
            root, canonical_root_id = self._registered_root(root_id, key)
            canonical_scope_id = _canonical_request_id(
                scope_id,
                kind="scope",
                key=key,
            )
            return self._build_locked(
                storage=storage,
                root=root,
                root_id=canonical_root_id,
                scope_id=canonical_scope_id,
                provider_summaries=provider_summaries,
                key=key,
            )

    def _build_locked(
        self,
        *,
        storage: _StorageSession,
        root: Path,
        root_id: str,
        scope_id: str,
        provider_summaries: Sequence[ProviderSafeSummary | Mapping[str, object]],
        key: bytes,
    ) -> BuildResult:
        current = self._load_index(storage, key)
        current_scopes = current["scopes"]
        if type(current_scopes) is not dict:
            _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)
        scopes = dict(current_scopes)
        if scope_id not in scopes and len(scopes) >= _MAX_SCOPES:
            _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)
        existing_documents, existing_terms = _resource_totals_excluding_scope(
            scopes,
            excluded_scope_id=scope_id,
        )

        source_count = 0
        source_bytes = 0
        new_term_count = 0
        source_fingerprints: dict[str, str] = {}
        persisted_by_id: dict[str, dict[str, object]] = {}

        def consume(source: Mapping[str, object]) -> None:
            nonlocal source_bytes, source_count, new_term_count
            source_count += 1
            if source_count > _MAX_DOCUMENTS_PER_SCOPE:
                _raise_unlinked(UnsafeContentError, _CONTENT_ERROR)
            byte_count = source["source_bytes"]
            text = source["text"]
            content_hash = source["content_hash"]
            if (
                type(byte_count) is not int
                or byte_count < 0
                or type(text) is not str
                or type(content_hash) is not str
                or byte_count > _MAX_TOTAL_SOURCE_BYTES - source_bytes
            ):
                _raise_unlinked(UnsafeContentError, _CONTENT_ERROR)
            source_bytes += byte_count

            text_fingerprint = _digest(text.encode("utf-8"))
            previous_fingerprint = source_fingerprints.get(content_hash)
            if previous_fingerprint is not None:
                if not hmac.compare_digest(previous_fingerprint, text_fingerprint):
                    _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)
                return
            source_fingerprints[content_hash] = text_fingerprint

            if existing_documents + len(persisted_by_id) >= _MAX_TOTAL_DOCUMENTS:
                _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)
            document = _document_record(
                text=text,
                content_hash=content_hash,
                key=key,
            )
            terms = document["terms"]
            if not isinstance(terms, Mapping):
                _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)
            term_count = len(terms)
            if (
                new_term_count > _MAX_TOTAL_TERMS - existing_terms
                or term_count > _MAX_TOTAL_TERMS - existing_terms - new_term_count
            ):
                _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)
            new_term_count += term_count

            document_id = str(document["document_id"])
            persisted = {
                "citation_id": _citation_id(scope_id, document_id),
                "content_hash": document["content_hash"],
                "document_id": document_id,
                "terms": terms,
            }
            existing = persisted_by_id.get(document_id)
            if existing is not None and existing != persisted:
                _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)
            persisted_by_id[document_id] = persisted

        for source in self._root_sources(root):
            consume(source)
        for source in self._provider_sources(provider_summaries):
            consume(source)

        persisted_documents = [
            persisted_by_id[document_id] for document_id in sorted(persisted_by_id)
        ]
        scopes[scope_id] = {
            "documents": persisted_documents,
            "root_id": root_id,
            "scope_id": scope_id,
        }
        _validate_resource_totals(scopes)
        sealed = _sealed_index(scopes, key)
        result = BuildResult(
            scope_id=scope_id,
            document_count=len(persisted_documents),
            index_hash=str(sealed["index_hash"]),
        )
        self._write_index(storage, sealed)
        return result

    def query(
        self,
        *,
        scope_id: str,
        query_id: str,
        query_text: str,
        limit: int = 5,
    ) -> QueryResult:
        """Return deterministic TF-IDF citations from one scope, never source text."""

        try:
            return self._query_request(
                scope_id=scope_id,
                query_id=query_id,
                query_text=query_text,
                limit=limit,
            )
        except IndexIntegrityError as caught:
            message = (
                _UNCERTAIN_COMMIT_ERROR
                if str(caught) == _UNCERTAIN_COMMIT_ERROR
                else _INTEGRITY_ERROR
            )
            caught = None
            scope_id = ""
            query_id = ""
            query_text = ""
            limit = 0
            _raise_unlinked(IndexIntegrityError, message)
        except UnsafeContentError:
            scope_id = ""
            query_id = ""
            query_text = ""
            limit = 0
            _raise_unlinked(UnsafeContentError, _CONTENT_ERROR)
        except UnsafeIndexRequestError:
            scope_id = ""
            query_id = ""
            query_text = ""
            limit = 0
            _raise_unlinked(UnsafeIndexRequestError, _REQUEST_ERROR)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            scope_id = ""
            query_id = ""
            query_text = ""
            limit = 0
            _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)

    def _query_request(
        self,
        *,
        scope_id: str,
        query_id: str,
        query_text: str,
        limit: int,
    ) -> QueryResult:
        if type(limit) is not int or not 1 <= limit <= 100:
            _raise_unlinked(UnsafeIndexRequestError, _REQUEST_ERROR)
        try:
            normalized_query = _validate_safe_text(query_text, allow_empty=True)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            _raise_unlinked(UnsafeContentError, _CONTENT_ERROR)

        with self._storage_lock() as storage:
            key = self._load_or_create_key(storage)
            canonical_scope_id = _canonical_request_id(
                scope_id,
                kind="scope",
                key=key,
            )
            _canonical_request_id(query_id, kind="query", key=key)
            return self._query_locked(
                storage=storage,
                scope_id=canonical_scope_id,
                normalized_query=normalized_query,
                limit=limit,
                key=key,
            )

    def _query_locked(
        self,
        *,
        storage: _StorageSession,
        scope_id: str,
        normalized_query: str,
        limit: int,
        key: bytes,
    ) -> QueryResult:
        index = self._load_index(storage, key)
        scope = index["scopes"].get(scope_id)
        if scope is None:
            return QueryResult(scope_id=scope_id, citations=())
        documents = scope["documents"]
        if not documents:
            return QueryResult(scope_id=scope_id, citations=())

        query_terms = _term_counts(normalized_query, key)
        vocabulary = sorted({term for document in documents for term in document["terms"]})
        if not vocabulary or not query_terms or not set(vocabulary).intersection(query_terms):
            return QueryResult(scope_id=scope_id, citations=())

        positions = {term: position for position, term in enumerate(vocabulary)}
        document_counts = _count_matrix(
            [document["terms"] for document in documents],
            positions,
        )
        query_counts = _count_matrix([query_terms], positions)
        transformer = TfidfTransformer(
            norm="l2",
            smooth_idf=True,
            sublinear_tf=False,
            use_idf=True,
        )
        document_vectors = transformer.fit_transform(document_counts)
        query_vector = transformer.transform(query_counts)
        raw_scores = (document_vectors @ query_vector.T).toarray().ravel().tolist()

        ranked: list[tuple[float, dict[str, object]]] = []
        for document, raw_score in zip(documents, raw_scores, strict=True):
            score = round(max(0.0, min(1.0, float(raw_score))), 12)
            if score > 0:
                ranked.append((score, document))
        ranked.sort(key=lambda item: (-item[0], str(item[1]["citation_id"])))

        citations = tuple(
            Citation(
                rank=rank,
                citation_id=str(document["citation_id"]),
                document_id=str(document["document_id"]),
                content_hash=str(document["content_hash"]),
                score=score,
            )
            for rank, (score, document) in enumerate(ranked[:limit], start=1)
        )
        return QueryResult(scope_id=scope_id, citations=citations)

    def _registered_root(self, root_id: str, key: bytes) -> tuple[Path, str]:
        canonical_root_id = _canonical_request_id(root_id, kind="root", key=key)
        canonical_roots: dict[str, Path] = {}
        for registered_id, registered_root in self._roots.items():
            candidate_id = _canonical_request_id(registered_id, kind="root", key=key)
            if candidate_id in canonical_roots:
                _raise_unlinked(UnsafeIndexRequestError, _REQUEST_ERROR)
            canonical_roots[candidate_id] = registered_root
        root = canonical_roots.get(canonical_root_id)
        if root is None:
            _raise_unlinked(UnsafeIndexRequestError, _REQUEST_ERROR)
        return root, canonical_root_id

    def _root_sources(self, root: Path) -> Iterator[dict[str, object]]:
        try:
            _require_fd_relative_support()
            root_fd = _open_trusted_directory(root, require_current_owner=False)
            with _closing_fd(root_fd, UnsafeIndexRequestError, _REQUEST_ERROR):
                approvals = _read_manifest(root_fd)
                found: set[str] = set()
                entry_count = [0]
                yield from _walk_root(
                    root_fd,
                    prefix=(),
                    approvals=approvals,
                    found=found,
                    entry_count=entry_count,
                )
                if found != set(approvals):
                    _raise_unlinked(UnsafeIndexRequestError, _REQUEST_ERROR)
        except (UnsafeContentError, UnsafeIndexRequestError):
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            _raise_unlinked(UnsafeIndexRequestError, _REQUEST_ERROR)

    def _provider_sources(
        self,
        summaries: Sequence[ProviderSafeSummary | Mapping[str, object]],
    ) -> Iterator[dict[str, object]]:
        try:
            if type(summaries) not in {list, tuple}:
                raise TypeError
            summary_count = len(summaries)
            if summary_count > _MAX_PROVIDER_SUMMARIES:
                raise ValueError
            for position in range(summary_count):
                summary = summaries[position]
                candidate = ProviderSafeSummary.safe_validate(summary)
                dumped = candidate.model_dump(mode="json")
                assert_safe_payload(dumped)
                yield {
                    "content_hash": candidate.content_hash,
                    "source_bytes": len(candidate.text.encode("utf-8")),
                    "text": candidate.text,
                }
            if len(summaries) != summary_count:
                raise ValueError
        except UnsafeContentError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            _raise_unlinked(UnsafeContentError, _CONTENT_ERROR)

    @contextmanager
    def _storage_lock(self) -> Iterator[_StorageSession]:
        with self._thread_lock:
            index_parent_fd: int | None = None
            key_parent_fd: int | None = None
            lock_fd: int | None = None
            lock_acquired = False
            try:
                _require_fd_relative_support()
                index_parent_fd = _open_trusted_directory(
                    self._index_path.parent,
                    require_current_owner=True,
                )
                if self._key_path.parent == self._index_path.parent:
                    key_parent_fd = index_parent_fd
                else:
                    key_parent_fd = _open_trusted_directory(
                        self._key_path.parent,
                        require_current_owner=True,
                    )
                _validate_secure_entry(index_parent_fd, self._index_path.name)
                _validate_secure_entry(key_parent_fd, self._key_path.name)
                _validate_secure_entry(index_parent_fd, self._lock_path.name)
                lock_flags = (
                    os.O_RDWR
                    | os.O_CREAT
                    | os.O_NOFOLLOW
                    | _o_nonblock()
                    | _o_cloexec()
                )
                try:
                    lock_fd = os.open(
                        self._lock_path.name,
                        lock_flags,
                        0o600,
                        dir_fd=index_parent_fd,
                    )
                except FileNotFoundError:
                    lock_fd = os.open(
                        self._lock_path.name,
                        lock_flags,
                        0o600,
                        dir_fd=index_parent_fd,
                    )
                _validate_secure_file_descriptor(lock_fd)
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                lock_acquired = True
                storage = _StorageSession(
                    index_parent_fd=index_parent_fd,
                    key_parent_fd=key_parent_fd,
                    index_name=self._index_path.name,
                    key_name=self._key_path.name,
                    lock_name=self._lock_path.name,
                )
                _cleanup_stale_temps(storage)
            except BaseException as error:
                if lock_fd is not None:
                    if lock_acquired:
                        _unlock_for_cleanup(lock_fd)
                    _close_for_cleanup(lock_fd)
                for descriptor in {
                    item for item in (index_parent_fd, key_parent_fd) if item is not None
                }:
                    _close_for_cleanup(descriptor)
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)

            body_failed = False
            try:
                yield storage
            except BaseException:
                body_failed = True
                if storage.index_commit_attempted or storage.index_committed:
                    _raise_unlinked(IndexIntegrityError, _UNCERTAIN_COMMIT_ERROR)
                raise
            finally:
                cleanup_failed = not _unlock_for_cleanup(lock_fd)
                cleanup_failed = not _close_for_cleanup(lock_fd) or cleanup_failed
                for descriptor in {index_parent_fd, key_parent_fd}:
                    cleanup_failed = not _close_for_cleanup(descriptor) or cleanup_failed
                if cleanup_failed and not body_failed:
                    message = (
                        _UNCERTAIN_COMMIT_ERROR
                        if storage.index_committed
                        else _INTEGRITY_ERROR
                    )
                    _raise_unlinked(IndexIntegrityError, message)

    def _load_or_create_key(self, storage: _StorageSession) -> bytes:
        index_exists = _entry_exists(storage.index_parent_fd, storage.index_name)
        try:
            return _read_secure_key_at(storage.key_parent_fd, storage.key_name)
        except FileNotFoundError:
            if index_exists:
                _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)
        except _RecoverablePartialKeyError:
            if index_exists:
                _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)
            _remove_recoverable_partial_key(storage.key_parent_fd, storage.key_name)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)

        key = os.urandom(_KEY_BYTES)
        temporary_name = f".{storage.key_name}.{secrets.token_hex(16)}.key.tmp"
        temporary_fd: int | None = None
        link_conflict = False
        failure: BaseException | None = None
        try:
            temporary_fd = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | _o_cloexec(),
                0o600,
                dir_fd=storage.key_parent_fd,
            )
            _validate_secure_file_descriptor(temporary_fd)
            _write_all(temporary_fd, key)
            os.fsync(temporary_fd)
            os.close(temporary_fd)
            temporary_fd = None
            try:
                os.link(
                    temporary_name,
                    storage.key_name,
                    src_dir_fd=storage.key_parent_fd,
                    dst_dir_fd=storage.key_parent_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                link_conflict = True
            if not link_conflict:
                os.unlink(temporary_name, dir_fd=storage.key_parent_fd)
                temporary_name = ""
                os.fsync(storage.key_parent_fd)
        except BaseException as error:
            failure = error

        cleanup_failed = False
        if temporary_fd is not None:
            cleanup_failed = not _close_for_cleanup(temporary_fd)
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=storage.key_parent_fd)
            except FileNotFoundError:
                pass
            except BaseException:
                cleanup_failed = True

        if failure is not None:
            if isinstance(failure, (KeyboardInterrupt, SystemExit)):
                raise failure
            _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)
        if cleanup_failed:
            _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)
        if link_conflict:
            try:
                return _read_secure_key_at(storage.key_parent_fd, storage.key_name)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)
        return key

    def _load_index(self, storage: _StorageSession, key: bytes) -> dict[str, Any]:
        try:
            encoded = _read_secure_file_at(
                storage.index_parent_fd,
                storage.index_name,
                max_bytes=_MAX_INDEX_BYTES,
                missing_ok=True,
            )
            if encoded is None:
                return _finalize_index({}, key)
            payload = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_object)
            _validate_persisted_index(payload, key)
            return payload
        except IndexIntegrityError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)

    def _write_index(
        self,
        storage: _StorageSession,
        payload: Mapping[str, object],
    ) -> None:
        try:
            encoded = _canonical_bytes(payload)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)
        if len(encoded) > _MAX_INDEX_BYTES:
            _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)

        temporary_fd: int | None = None
        temporary_name = f".{storage.index_name}.{secrets.token_hex(16)}.index.tmp"
        failure: BaseException | None = None
        try:
            temporary_fd = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | _o_cloexec(),
                0o600,
                dir_fd=storage.index_parent_fd,
            )
            _validate_secure_file_descriptor(temporary_fd)
            _write_all(temporary_fd, encoded)
            os.fsync(temporary_fd)
            os.close(temporary_fd)
            temporary_fd = None
            storage.index_commit_attempted = True
            os.replace(
                temporary_name,
                storage.index_name,
                src_dir_fd=storage.index_parent_fd,
                dst_dir_fd=storage.index_parent_fd,
            )
            temporary_name = ""
            os.fsync(storage.index_parent_fd)
            storage.index_committed = True
        except BaseException as error:
            failure = error

        cleanup_failure: BaseException | None = None
        if temporary_fd is not None:
            try:
                os.close(temporary_fd)
            except BaseException as error:
                cleanup_failure = error
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=storage.index_parent_fd)
            except FileNotFoundError:
                pass
            except BaseException as error:
                if cleanup_failure is None or (
                    not isinstance(cleanup_failure, (KeyboardInterrupt, SystemExit))
                    and isinstance(error, (KeyboardInterrupt, SystemExit))
                ):
                    cleanup_failure = error

        if failure is not None or cleanup_failure is not None:
            if storage.index_commit_attempted:
                _raise_unlinked(IndexIntegrityError, _UNCERTAIN_COMMIT_ERROR)
            if isinstance(failure, (KeyboardInterrupt, SystemExit)):
                raise failure
            if isinstance(cleanup_failure, (KeyboardInterrupt, SystemExit)):
                raise cleanup_failure
            _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)


def _prepare_index_configuration(
    *,
    index_path: Path | str,
    roots: object,
    key_path: Path | str | None,
) -> tuple[Path, Path, Path, Mapping[str, Path], Any]:
    if type(roots) is not dict or len(roots) > _MAX_REGISTERED_ROOTS:
        raise TypeError
    normalized_index = _lexical_absolute(index_path)
    if _INDEX_NAME.fullmatch(normalized_index.name) is None:
        raise ValueError
    normalized_key = _lexical_absolute(
        key_path
        if key_path is not None
        else normalized_index.with_name(f"{normalized_index.name}.key")
    )
    expected_key = normalized_index.with_name(f"{normalized_index.name}.key")
    if normalized_key != expected_key:
        raise ValueError
    lock_path = normalized_index.with_name(f"{normalized_index.name}{_LOCK_SUFFIX}")
    if len({normalized_index, normalized_key, lock_path}) != 3:
        raise ValueError
    _validate_storage_path(normalized_index)
    _validate_storage_path(normalized_key)
    _validate_storage_path(lock_path)

    prepared_roots: dict[str, Path] = {}
    for root_id, root in roots.items():
        validated_root_id = _validated_request_code(root_id, kind="root")
        candidate = _lexical_absolute(root)
        _validate_directory_chain(candidate)
        if validated_root_id in prepared_roots:
            raise ValueError
        prepared_roots[validated_root_id] = candidate
    normalized_roots = MappingProxyType(dict(sorted(prepared_roots.items())))
    return (
        normalized_index,
        normalized_key,
        lock_path,
        normalized_roots,
        _path_lock(normalized_index),
    )


def _validate_path_text(value: str) -> None:
    length = len(value)
    if not 1 <= length <= _MAX_PATH_LENGTH:
        raise ValueError
    if "\x00" in value:
        raise ValueError
    normalized_separators = (
        value.replace(os.altsep, os.sep) if os.altsep is not None else value
    )
    components = normalized_separators.split(os.sep)
    if len(components) > _MAX_PATH_COMPONENTS or any(
        len(component) > _MAX_PATH_COMPONENT for component in components
    ):
        raise ValueError


def _passive_path_text(path: object) -> str:
    if type(path) not in {str, _CONCRETE_PATH_TYPE}:
        raise TypeError
    encoded = path if type(path) is str else os.fspath(path)
    if type(encoded) is not str:
        raise TypeError
    _validate_path_text(encoded)
    return encoded


def _lexical_absolute(path: object) -> Path:
    encoded = _passive_path_text(path)
    candidate = Path(encoded)
    if ".." in candidate.parts or not candidate.name:
        raise ValueError
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute_input = _passive_path_text(candidate)
    absolute = os.path.abspath(absolute_input)
    _validate_path_text(absolute)
    return Path(absolute)


def _validate_directory_chain(path: Path) -> None:
    descriptor = _open_trusted_directory(path, require_current_owner=False)
    if not _close_for_cleanup(descriptor):
        raise OSError


def _open_trusted_directory(path: Path, *, require_current_owner: bool) -> int:
    absolute = path if path.is_absolute() else _lexical_absolute(path)
    anchor = absolute.anchor
    if not anchor:
        raise ValueError
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | _o_nonblock()
        | _o_cloexec()
    )
    descriptor = os.open(anchor, flags)
    try:
        _validate_trusted_directory_descriptor(
            descriptor,
            require_current_owner=False,
        )
        for position, component in enumerate(absolute.parts[1:], start=1):
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            try:
                _validate_trusted_directory_descriptor(
                    next_descriptor,
                    require_current_owner=(
                        require_current_owner and position == len(absolute.parts) - 1
                    ),
                )
            except BaseException:
                _close_for_cleanup(next_descriptor)
                raise
            if not _close_for_cleanup(descriptor):
                _close_for_cleanup(next_descriptor)
                raise OSError
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        _close_for_cleanup(descriptor)
        raise


def _validate_trusted_directory_descriptor(
    descriptor: int,
    *,
    require_current_owner: bool,
) -> None:
    metadata = os.fstat(descriptor)
    trusted_owners = {0, os.getuid()}
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid not in trusted_owners
        or (require_current_owner and metadata.st_uid != os.getuid())
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise ValueError


def _validate_storage_path(path: Path) -> None:
    descriptor = _open_trusted_directory(path.parent, require_current_owner=True)
    try:
        _validate_secure_entry(descriptor, path.name)
    finally:
        if not _close_for_cleanup(descriptor):
            raise OSError


def _open_secure_parent(path: Path) -> int:
    descriptor = _open_trusted_directory(path.parent, require_current_owner=True)
    try:
        _validate_secure_entry(descriptor, path.name)
    except BaseException:
        _close_for_cleanup(descriptor)
        raise
    return descriptor


def _validate_secure_file_descriptor(descriptor: int) -> None:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise ValueError


def _validate_secure_entry(parent_fd: int, name: str) -> None:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | _o_nonblock() | _o_cloexec(),
            dir_fd=parent_fd,
        )
    except FileNotFoundError:
        return
    try:
        _validate_secure_file_descriptor(descriptor)
    finally:
        if not _close_for_cleanup(descriptor):
            raise OSError


def _entry_exists(parent_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _read_secure_key_at(parent_fd: int, name: str) -> bytes:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW | _o_nonblock() | _o_cloexec(),
        dir_fd=parent_fd,
    )
    try:
        _validate_secure_file_descriptor(descriptor)
        metadata = os.fstat(descriptor)
        if metadata.st_size < _KEY_BYTES:
            raise _RecoverablePartialKeyError
        if metadata.st_size != _KEY_BYTES:
            raise ValueError
        encoded = _read_bounded(descriptor, _KEY_BYTES)
        if len(encoded) < _KEY_BYTES:
            raise _RecoverablePartialKeyError
        if len(encoded) != _KEY_BYTES:
            raise ValueError
        return encoded
    finally:
        if not _close_for_cleanup(descriptor):
            raise OSError


def _remove_recoverable_partial_key(parent_fd: int, name: str) -> None:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW | _o_nonblock() | _o_cloexec(),
        dir_fd=parent_fd,
    )
    try:
        _validate_secure_file_descriptor(descriptor)
        if os.fstat(descriptor).st_size >= _KEY_BYTES:
            raise ValueError
    finally:
        if not _close_for_cleanup(descriptor):
            raise OSError
    os.unlink(name, dir_fd=parent_fd)


def _read_secure_file_at(
    parent_fd: int,
    name: str,
    *,
    max_bytes: int,
    missing_ok: bool,
) -> bytes | None:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | _o_nonblock() | _o_cloexec(),
            dir_fd=parent_fd,
        )
    except FileNotFoundError:
        if missing_ok:
            return None
        raise
    with _closing_fd(descriptor, IndexIntegrityError, _INTEGRITY_ERROR):
        _validate_secure_file_descriptor(descriptor)
        metadata = os.fstat(descriptor)
        if metadata.st_size > max_bytes:
            raise ValueError
        return _read_bounded(descriptor, max_bytes)


def _read_secure_key(path: Path) -> bytes:
    parent_fd = _open_secure_parent(path)
    try:
        return _read_secure_key_at(parent_fd, path.name)
    finally:
        if not _close_for_cleanup(parent_fd):
            raise OSError


def _read_secure_file(
    path: Path,
    *,
    max_bytes: int,
    missing_ok: bool,
) -> bytes | None:
    parent_fd = _open_secure_parent(path)
    try:
        return _read_secure_file_at(
            parent_fd,
            path.name,
            max_bytes=max_bytes,
            missing_ok=missing_ok,
        )
    finally:
        if not _close_for_cleanup(parent_fd):
            raise OSError


def _cleanup_stale_temps(storage: _StorageSession) -> None:
    parents: dict[int, list[tuple[str, str]]] = {}
    parents.setdefault(storage.index_parent_fd, []).append((storage.index_name, "index"))
    parents.setdefault(storage.key_parent_fd, []).append((storage.key_name, "key"))
    for parent_fd, specifications in parents.items():
        scanned = 0
        with os.scandir(parent_fd) as entries:
            for entry in entries:
                scanned += 1
                if scanned > _MAX_STALE_TEMP_ENTRIES:
                    raise ValueError
                name = entry.name
                if name in {storage.index_name, storage.key_name, storage.lock_name}:
                    continue
                if not isinstance(name, str) or not any(
                    re.fullmatch(
                        rf"\.{re.escape(base_name)}\.[0-9a-f]{{32}}\.{kind}\.tmp",
                        name,
                    )
                    is not None
                    for base_name, kind in specifications
                ):
                    continue
                try:
                    descriptor = os.open(
                        name,
                        os.O_RDONLY | os.O_NOFOLLOW | _o_nonblock() | _o_cloexec(),
                        dir_fd=parent_fd,
                    )
                except OSError:
                    continue
                try:
                    metadata = os.fstat(descriptor)
                    eligible = (
                        stat.S_ISREG(metadata.st_mode)
                        and metadata.st_uid == os.getuid()
                        and stat.S_IMODE(metadata.st_mode) == 0o600
                    )
                finally:
                    if not _close_for_cleanup(descriptor):
                        raise OSError
                if eligible:
                    os.unlink(name, dir_fd=parent_fd)


def _unlock_for_cleanup(descriptor: int) -> bool:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    except BaseException:
        return False
    return True


def _close_for_cleanup(descriptor: int) -> bool:
    try:
        os.close(descriptor)
    except BaseException:
        return False
    return True


def _read_manifest(root_fd: int) -> dict[str, str]:
    try:
        descriptor = os.open(
            _MANIFEST_NAME,
            os.O_RDONLY | os.O_NOFOLLOW | _o_nonblock() | _o_cloexec(),
            dir_fd=root_fd,
        )
        with _closing_fd(descriptor, UnsafeIndexRequestError, _REQUEST_ERROR):
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid not in {0, os.getuid()}
                or stat.S_IMODE(metadata.st_mode) & 0o022
                or metadata.st_size > _MAX_MANIFEST_BYTES
            ):
                raise ValueError
            encoded = _read_bounded(descriptor, _MAX_MANIFEST_BYTES)
        payload = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_object)
        return _validate_manifest(payload)
    except UnsafeIndexRequestError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _raise_unlinked(UnsafeIndexRequestError, _REQUEST_ERROR)


def _require_exact_dict_shape(
    value: object,
    expected_keys: frozenset[str],
) -> dict[str, object]:
    if type(value) is not dict or len(value) != len(expected_keys):
        raise ValueError
    keys: list[str] = []
    for key in value:
        if type(key) is not str:
            raise ValueError
        keys.append(key)
    if frozenset(keys) != expected_keys:
        raise ValueError
    return value


def _validate_manifest(value: object) -> dict[str, str]:
    manifest = _require_exact_dict_shape(value, _MANIFEST_KEYS)
    if type(manifest["format_version"]) is not int or manifest["format_version"] != 1:
        raise ValueError
    documents = manifest["documents"]
    if type(documents) is not list or len(documents) > _MAX_MANIFEST_DOCUMENTS:
        raise ValueError
    approvals: dict[str, str] = {}
    for item in documents:
        document = _require_exact_dict_shape(item, _MANIFEST_DOCUMENT_KEYS)
        path = document["path"]
        content_hash = document["content_hash"]
        privacy_class = document["privacy_class"]
        if type(path) is not str or type(content_hash) is not str:
            raise ValueError
        if type(privacy_class) is not str:
            raise ValueError
        privacy_length = len(privacy_class)
        if privacy_length != len("provider_safe") or privacy_class != "provider_safe":
            raise ValueError
        _validate_manifest_path(path)
        hash_length = len(content_hash)
        if (
            hash_length != 64
            or _SHA256.fullmatch(content_hash) is None
            or path in approvals
        ):
            raise ValueError
        approvals[path] = content_hash
    return dict(sorted(approvals.items()))


def _validate_manifest_path(path: str) -> None:
    if type(path) is not str:
        raise ValueError
    path_length = len(path)
    if (
        not path_length
        or path_length > _MAX_PATH_LENGTH
        or not path.isascii()
        or unicodedata.normalize("NFC", path) != path
        or any(unicodedata.category(character).startswith("C") for character in path)
        or path.startswith("/")
        or _WINDOWS_MANIFEST_PATH.match(path) is not None
        or "\\" in path
        or "\x00" in path
    ):
        raise ValueError
    components = path.split("/")
    if any(
        not component
        or component in {".", ".."}
        or len(component) > _MAX_PATH_COMPONENT
        for component in components
    ):
        raise ValueError
    if Path(components[-1]).suffix.casefold() not in _ALLOWED_SUFFIXES:
        raise ValueError
    if any(_forbidden_directory(component) for component in components[:-1]):
        raise ValueError


def _walk_root(
    directory_fd: int,
    *,
    prefix: tuple[str, ...],
    approvals: Mapping[str, str],
    found: set[str],
    entry_count: list[int],
) -> Iterator[dict[str, object]]:
    if len(prefix) > _MAX_TREE_DEPTH:
        _raise_unlinked(UnsafeIndexRequestError, _REQUEST_ERROR)
    try:
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                name = entry.name
                entry_count[0] += 1
                if (
                    entry_count[0] > _MAX_TREE_ENTRIES
                    or not isinstance(name, str)
                    or not name
                    or name in {".", ".."}
                    or "/" in name
                    or "\x00" in name
                ):
                    _raise_unlinked(UnsafeIndexRequestError, _REQUEST_ERROR)
                descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_NOFOLLOW | _o_nonblock() | _o_cloexec(),
                    dir_fd=directory_fd,
                )
                with _closing_fd(descriptor, UnsafeIndexRequestError, _REQUEST_ERROR):
                    metadata = os.fstat(descriptor)
                    relative_parts = (*prefix, name)
                    relative = "/".join(relative_parts)
                    if stat.S_ISDIR(metadata.st_mode):
                        _validate_trusted_directory_descriptor(
                            descriptor,
                            require_current_owner=False,
                        )
                        yield from _walk_root(
                            descriptor,
                            prefix=relative_parts,
                            approvals=approvals,
                            found=found,
                            entry_count=entry_count,
                        )
                    elif stat.S_ISREG(metadata.st_mode):
                        if not prefix and name == _MANIFEST_NAME:
                            continue
                        expected_hash = approvals.get(relative)
                        if expected_hash is not None:
                            source = _read_attested_document(
                                descriptor,
                                metadata,
                                expected_hash,
                            )
                            found.add(relative)
                            yield source
                    else:
                        _raise_unlinked(UnsafeIndexRequestError, _REQUEST_ERROR)
    except (UnsafeContentError, UnsafeIndexRequestError):
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _raise_unlinked(UnsafeIndexRequestError, _REQUEST_ERROR)


def _read_attested_document(
    descriptor: int,
    metadata: os.stat_result,
    expected_hash: str,
) -> dict[str, object]:
    if metadata.st_size > _MAX_FILE_BYTES:
        _raise_unlinked(UnsafeContentError, _CONTENT_ERROR)
    try:
        encoded = _read_bounded(descriptor, _MAX_FILE_BYTES)
        if not hmac.compare_digest(hashlib.sha256(encoded).hexdigest(), expected_hash):
            _raise_unlinked(UnsafeContentError, _CONTENT_ERROR)
        text = encoded.decode("utf-8")
        normalized = _validate_safe_text(text, allow_empty=True)
        return {
            "content_hash": expected_hash,
            "source_bytes": len(encoded),
            "text": normalized,
        }
    except UnsafeContentError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _raise_unlinked(UnsafeContentError, _CONTENT_ERROR)


def _read_bounded(descriptor: int, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = max_bytes + 1
    while remaining:
        chunk = os.read(descriptor, min(65_536, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    encoded = b"".join(chunks)
    if len(encoded) > max_bytes:
        raise ValueError
    return encoded


def _write_all(descriptor: int, encoded: bytes) -> None:
    view = memoryview(encoded)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if type(count) is not int or count <= 0:
            raise OSError
        written += count


@contextmanager
def _closing_fd(
    descriptor: int,
    error_type: type[Exception],
    message: str,
) -> Iterator[None]:
    body_failed = False
    try:
        yield
    except BaseException:
        body_failed = True
        raise
    finally:
        try:
            os.close(descriptor)
        except Exception:
            if not body_failed:
                _raise_unlinked(error_type, message)


def _close_quietly(descriptor: int) -> None:
    try:
        os.close(descriptor)
    except OSError:
        pass


def _forbidden_directory(name: str) -> bool:
    normalized = name.casefold()
    visible = normalized.lstrip("._-")
    return (
        normalized == ".git"
        or "cache" in normalized
        or visible.startswith(_FORBIDDEN_DIRECTORY_PREFIXES)
    )


def _document_record(*, text: str, content_hash: str, key: bytes) -> dict[str, object]:
    document_id = _document_id(content_hash)
    return {
        "content_hash": content_hash,
        "document_id": document_id,
        "terms": _term_counts(text, key),
    }


def _term_counts(text: str, key: bytes) -> dict[str, int]:
    raw_counts = Counter(_TOKEN.findall(text.casefold()))
    keyed: Counter[str] = Counter()
    for token, count in raw_counts.items():
        if type(count) is not int or not 1 <= count <= _MAX_TERM_COUNT:
            raise ValueError
        keyed[_hmac_digest(key, b"term\0" + token.encode("utf-8"))] += count
    if len(keyed) > _MAX_TERMS_PER_DOCUMENT:
        raise ValueError
    return dict(sorted(keyed.items()))


def _count_matrix(
    rows: Sequence[Mapping[str, object]],
    positions: Mapping[str, int],
) -> csr_matrix:
    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    for row in rows:
        if len(row) > _MAX_TERMS_PER_DOCUMENT:
            raise IndexIntegrityError(_INTEGRITY_ERROR)
        for term, count in sorted(row.items()):
            if type(count) is not int or not 1 <= count <= _MAX_TERM_COUNT:
                _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)
            position = positions.get(term)
            if position is not None:
                indices.append(position)
                data.append(float(count))
        indptr.append(len(indices))
    return csr_matrix(
        (data, indices, indptr),
        shape=(len(rows), len(positions)),
        dtype=float,
    )


def _sealed_index(scopes: Mapping[str, object], key: bytes) -> dict[str, object]:
    sealed_scopes: dict[str, object] = {}
    for scope_id, raw_scope in sorted(scopes.items()):
        if not isinstance(raw_scope, Mapping):
            raise ValueError
        core = {
            key_name: raw_scope[key_name]
            for key_name in ("documents", "root_id", "scope_id")
        }
        sealed_scopes[scope_id] = {
            **core,
            "seal": _hmac_digest(key, b"scope\0" + _canonical_bytes(core)),
        }
    return _finalize_index(sealed_scopes, key)


def _finalize_index(
    sealed_scopes: Mapping[str, object],
    key: bytes,
) -> dict[str, object]:
    semantic_body = _semantic_body(sealed_scopes)
    index_hash = _digest(_canonical_bytes(semantic_body))
    authenticated = {
        "format_version": _FORMAT_VERSION,
        "index_hash": index_hash,
        "key_id": _digest(key),
        "scopes": sealed_scopes,
    }
    return {
        **authenticated,
        "seal": _hmac_digest(key, b"index\0" + _canonical_bytes(authenticated)),
    }


def _semantic_body(scopes: Mapping[str, object]) -> dict[str, object]:
    semantic_scopes: dict[str, object] = {}
    for scope_id, raw_scope in sorted(scopes.items()):
        if not isinstance(raw_scope, Mapping):
            raise ValueError
        documents = raw_scope["documents"]
        if not isinstance(documents, list):
            raise ValueError
        semantic_scopes[scope_id] = {
            "documents": [
                {
                    "citation_id": document["citation_id"],
                    "content_hash": document["content_hash"],
                    "document_id": document["document_id"],
                }
                for document in documents
                if isinstance(document, Mapping)
            ],
            "root_id": raw_scope["root_id"],
            "scope_id": raw_scope["scope_id"],
        }
    return {"format_version": _FORMAT_VERSION, "scopes": semantic_scopes}


def _validate_persisted_index(value: object, key: bytes) -> None:
    index = _require_exact_dict_shape(value, _INDEX_KEYS)
    if type(index["format_version"]) is not int or index["format_version"] != _FORMAT_VERSION:
        raise ValueError
    for field in ("index_hash", "key_id", "seal"):
        item = index[field]
        if type(item) is not str:
            raise ValueError
        item_length = len(item)
        if item_length != 64 or _SHA256.fullmatch(item) is None:
            raise ValueError
    key_id = index["key_id"]
    if not isinstance(key_id, str) or not hmac.compare_digest(key_id, _digest(key)):
        raise ValueError
    scopes = index["scopes"]
    if type(scopes) is not dict or len(scopes) > _MAX_SCOPES:
        raise ValueError
    scope_keys: list[str] = []
    for scope_key in scopes:
        _validate_persisted_identifier(scope_key, kind="scope")
        scope_keys.append(scope_key)
    if scope_keys != sorted(scope_keys):
        raise ValueError
    total_documents = 0
    total_terms = 0
    for scope_key, scope in scopes.items():
        document_count, term_count = _validate_scope(scope_key, scope, key)
        total_documents += document_count
        total_terms += term_count
        if total_documents > _MAX_TOTAL_DOCUMENTS or total_terms > _MAX_TOTAL_TERMS:
            raise ValueError

    expected_index_hash = _digest(_canonical_bytes(_semantic_body(scopes)))
    index_hash = index["index_hash"]
    if not isinstance(index_hash, str) or not hmac.compare_digest(
        index_hash,
        expected_index_hash,
    ):
        raise ValueError
    authenticated = {
        "format_version": index["format_version"],
        "index_hash": index_hash,
        "key_id": key_id,
        "scopes": scopes,
    }
    expected_seal = _hmac_digest(key, b"index\0" + _canonical_bytes(authenticated))
    seal = index["seal"]
    if not isinstance(seal, str) or not hmac.compare_digest(seal, expected_seal):
        raise ValueError


def _validate_scope(
    scope_key: str,
    value: object,
    key: bytes,
) -> tuple[int, int]:
    scope = _require_exact_dict_shape(value, _SCOPE_KEYS)
    _validate_persisted_identifier(scope_key, kind="scope")
    scope_id = scope["scope_id"]
    _validate_persisted_identifier(scope_id, kind="scope")
    if scope_id != scope_key:
        raise ValueError
    root_id = scope["root_id"]
    _validate_persisted_identifier(root_id, kind="root")
    seal = scope["seal"]
    if type(seal) is not str:
        raise ValueError
    seal_length = len(seal)
    if seal_length != 64 or _SHA256.fullmatch(seal) is None:
        raise ValueError
    documents = scope["documents"]
    if type(documents) is not list or len(documents) > _MAX_DOCUMENTS_PER_SCOPE:
        raise ValueError

    previous_id = ""
    seen_ids: set[str] = set()
    term_total = 0
    for raw_document in documents:
        document = _require_exact_dict_shape(raw_document, _DOCUMENT_KEYS)
        content_hash = document["content_hash"]
        document_id = document["document_id"]
        citation_id = document["citation_id"]
        for item in (content_hash, document_id, citation_id):
            if type(item) is not str:
                raise ValueError
            item_length = len(item)
            if item_length != 64 or _SHA256.fullmatch(item) is None:
                raise ValueError
        if not isinstance(content_hash, str) or not isinstance(document_id, str):
            raise ValueError
        if not isinstance(citation_id, str):
            raise ValueError

        terms = document["terms"]
        if type(terms) is not dict or len(terms) > _MAX_TERMS_PER_DOCUMENT:
            raise ValueError
        term_keys: list[str] = []
        for term, count in terms.items():
            if type(term) is not str:
                raise ValueError
            term_length = len(term)
            if (
                term_length != 64
                or _SHA256.fullmatch(term) is None
                or type(count) is not int
                or not 1 <= count <= _MAX_TERM_COUNT
            ):
                raise ValueError
            term_keys.append(term)
        if term_keys != sorted(term_keys):
            raise ValueError

        if document_id != _document_id(content_hash):
            raise ValueError
        if citation_id != _citation_id(scope_key, document_id):
            raise ValueError
        if document_id in seen_ids or document_id <= previous_id:
            raise ValueError
        term_total += len(terms)
        seen_ids.add(document_id)
        previous_id = document_id

    core = {
        "documents": documents,
        "root_id": root_id,
        "scope_id": scope_key,
    }
    expected_seal = _hmac_digest(key, b"scope\0" + _canonical_bytes(core))
    if not hmac.compare_digest(seal, expected_seal):
        raise ValueError
    return len(documents), term_total


def _resource_totals_excluding_scope(
    scopes: Mapping[str, object],
    *,
    excluded_scope_id: str,
) -> tuple[int, int]:
    document_total = 0
    term_total = 0
    for scope_id, scope in scopes.items():
        if scope_id == excluded_scope_id:
            continue
        if not isinstance(scope, Mapping):
            _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)
        documents = scope.get("documents")
        if not isinstance(documents, list):
            _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)
        document_total += len(documents)
        for document in documents:
            if not isinstance(document, Mapping):
                _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)
            terms = document.get("terms")
            if not isinstance(terms, Mapping):
                _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)
            term_total += len(terms)
        if document_total > _MAX_TOTAL_DOCUMENTS or term_total > _MAX_TOTAL_TERMS:
            _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)
    return document_total, term_total


def _validate_resource_totals(scopes: object) -> None:
    if not isinstance(scopes, Mapping) or len(scopes) > _MAX_SCOPES:
        _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)
    document_total = 0
    term_total = 0
    for scope in scopes.values():
        if not isinstance(scope, Mapping):
            _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)
        documents = scope.get("documents")
        if not isinstance(documents, list) or len(documents) > _MAX_DOCUMENTS_PER_SCOPE:
            _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)
        document_total += len(documents)
        for document in documents:
            if not isinstance(document, Mapping):
                _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)
            terms = document.get("terms")
            if not isinstance(terms, Mapping):
                _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)
            term_total += len(terms)
        if document_total > _MAX_TOTAL_DOCUMENTS or term_total > _MAX_TOTAL_TERMS:
            _raise_unlinked(IndexIntegrityError, _INTEGRITY_ERROR)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _validate_persisted_identifier(value: object, *, kind: str) -> None:
    if type(value) is not str:
        raise ValueError
    expected_length = len(kind) + 1 + 24
    value_length = len(value)
    if value_length != expected_length:
        raise ValueError
    matched = _OPAQUE_ID.fullmatch(value)
    if matched is None or matched.group(1) != kind:
        raise ValueError
    assert_safe_payload({"identifier": value})


def _validated_request_code(value: object, *, kind: str) -> str:
    if type(value) is not str:
        raise ValueError
    value_length = len(value)
    if not 1 <= value_length <= _MAX_REQUEST_CODE_LENGTH:
        raise ValueError
    matched = _OPAQUE_ID.fullmatch(value)
    if matched is not None:
        if matched.group(1) != kind:
            raise ValueError
        assert_safe_payload({"identifier": value})
        return value
    patterns = {
        "query": _QUERY_CODE,
        "root": _ROOT_CODE,
        "scope": _SCOPE_CODE,
    }
    pattern = patterns.get(kind)
    if pattern is None or pattern.fullmatch(value) is None:
        raise ValueError
    assert_safe_payload({"identifier": value})
    return value


def _canonical_request_id(value: str, *, kind: str, key: bytes) -> str:
    try:
        validated = _validated_request_code(value, kind=kind)
        matched = _OPAQUE_ID.fullmatch(validated)
        if matched is not None:
            return validated
        if type(key) is not bytes or len(key) != _KEY_BYTES:
            raise ValueError
        digest = _hmac_digest(
            key,
            b"rag-identifier\0"
            + kind.encode("ascii")
            + b"\0"
            + validated.encode("ascii"),
        )[:24]
        return f"{kind}-{digest}"
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        value = ""
        _raise_unlinked(UnsafeIndexRequestError, _REQUEST_ERROR)


def _path_lock(path: Path) -> threading.RLock:
    key = os.fspath(path)
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
        return lock


def _require_fd_relative_support() -> None:
    if (
        not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_NONBLOCK")
        or not hasattr(os, "O_DIRECTORY")
        or os.open not in os.supports_dir_fd
        or os.scandir not in os.supports_fd
        or os.stat not in os.supports_dir_fd
        or os.unlink not in os.supports_dir_fd
        or os.link not in os.supports_dir_fd
    ):
        _raise_unlinked(UnsafeIndexRequestError, _REQUEST_ERROR)


def _o_nonblock() -> int:
    return int(getattr(os, "O_NONBLOCK", 0))


def _o_cloexec() -> int:
    return int(getattr(os, "O_CLOEXEC", 0))


def _document_id(content_hash: str) -> str:
    return _digest(f"document\0{content_hash}".encode("ascii"))


def _citation_id(scope_id: str, document_id: str) -> str:
    return _digest(f"citation\0{scope_id}\0{document_id}".encode("ascii"))


def _hmac_digest(key: bytes, value: bytes) -> str:
    return hmac.new(key, value, hashlib.sha256).hexdigest()


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


__all__ = ["LocalCitationIndex"]
