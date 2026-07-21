from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

from .fingerprinting import FileFingerprint, fingerprint_file, streaming_sha256
from .repositories import DocumentRepository, JobRepository
from .security import authorize_path, resolve_authorized_root

SUPPORTED_SUFFIXES = frozenset(
    {
        ".pdf",
        ".doc",
        ".docx",
        ".ppt",
        ".pptx",
        ".xlsx",
        ".xlsm",
        ".xls",
        ".zip",
        ".rar",
        ".7z",
        ".tar",
        ".gz",
        ".txt",
        ".md",
    }
)


@dataclass(frozen=True)
class ScanEvent:
    operation: str
    path: str
    sha256: str | None = None


@dataclass(frozen=True)
class ScanResult:
    events: tuple[ScanEvent, ...] = field(default_factory=tuple)
    unchanged: int = 0
    skipped_unsupported: int = 0

    @property
    def counts(self) -> dict[str, int]:
        counts = {
            "add": 0,
            "update": 0,
            "move": 0,
            "delete": 0,
            "reindex": 0,
            "unchanged": self.unchanged,
            "skipped_unsupported": self.skipped_unsupported,
        }
        for event in self.events:
            counts[event.operation] += 1
        return counts


@dataclass(frozen=True)
class FileSnapshot:
    size_bytes: int
    mtime_ns: int


class FileEventDebouncer:
    def __init__(self) -> None:
        self._snapshots: dict[str, FileSnapshot] = {}

    def is_stable(self, path: str | Path) -> bool:
        resolved = Path(path).resolve(strict=True)
        stat = resolved.stat()
        current = FileSnapshot(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        previous = self._snapshots.get(str(resolved))
        self._snapshots[str(resolved)] = current
        return previous == current


class SourceScanner:
    def __init__(
        self,
        *,
        documents: DocumentRepository,
        jobs: JobRepository,
        supported_suffixes: frozenset[str] = SUPPORTED_SUFFIXES,
    ) -> None:
        self.documents = documents
        self.jobs = jobs
        self.supported_suffixes = supported_suffixes

    def reconcile(
        self,
        *,
        knowledge_base_id: str,
        root_path: str | Path,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> ScanResult:
        root = resolve_authorized_root(root_path)
        include_patterns = include_patterns or ["*", "**/*"]
        exclude_patterns = exclude_patterns or []
        current_files = self._iter_files(
            root=root,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )

        events: list[ScanEvent] = []
        unchanged = 0
        skipped_unsupported = 0
        seen_paths: set[str] = set()

        for file_path in current_files:
            authorized = authorize_path(file_path, root=root)
            if (
                authorized.path.suffix.lower() not in self.supported_suffixes
                or authorized.path.name.startswith(("~$", "._"))
            ):
                skipped_unsupported += 1
                continue

            stat = authorized.path.stat()
            canonical_path = str(authorized.path)
            seen_paths.add(canonical_path)
            document = self.documents.get_by_path(
                knowledge_base_id=knowledge_base_id,
                canonical_path=canonical_path,
            )
            active_version = self.documents.get_active_version(document.id) if document else None
            latest_version = self.documents.get_latest_version(document.id) if document else None
            known_version = active_version or latest_version

            if (
                known_version is not None
                and known_version.size_bytes == stat.st_size
                and known_version.mtime_ns == stat.st_mtime_ns
            ):
                unchanged += 1
                continue

            fingerprint = FileFingerprint(
                canonical_path=canonical_path,
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                sha256=streaming_sha256(authorized.path),
            )

            if document is None:
                moved_document = self.documents.find_by_active_sha256(
                    knowledge_base_id=knowledge_base_id,
                    sha256=fingerprint.sha256,
                )
                if moved_document is not None:
                    self.documents.update_path(
                        document_id=moved_document.id,
                        canonical_path=fingerprint.canonical_path,
                    )
                    moved_version = self.documents.get_active_version(moved_document.id)
                    if moved_version is not None:
                        self.documents.update_version_fingerprint(
                            version_id=moved_version.id,
                            size_bytes=fingerprint.size_bytes,
                            mtime_ns=fingerprint.mtime_ns,
                        )
                    self.jobs.enqueue(
                        knowledge_base_id=knowledge_base_id,
                        operation="move",
                        path=fingerprint.canonical_path,
                        expected_sha256=fingerprint.sha256,
                    )
                    events.append(
                        ScanEvent("move", fingerprint.canonical_path, fingerprint.sha256)
                    )
                    continue

                created = self.documents.create(
                    knowledge_base_id=knowledge_base_id,
                    canonical_path=fingerprint.canonical_path,
                )
                self.documents.create_version(
                    document_id=created.id,
                    sha256=fingerprint.sha256,
                    size_bytes=fingerprint.size_bytes,
                    mtime_ns=fingerprint.mtime_ns,
                    parser_version="pending",
                )
                self.jobs.enqueue(
                    knowledge_base_id=knowledge_base_id,
                    operation="add",
                    path=fingerprint.canonical_path,
                    expected_sha256=fingerprint.sha256,
                )
                events.append(ScanEvent("add", fingerprint.canonical_path, fingerprint.sha256))
                continue

            if active_version is not None and active_version.sha256 == fingerprint.sha256:
                self.documents.update_version_fingerprint(
                    version_id=active_version.id,
                    size_bytes=fingerprint.size_bytes,
                    mtime_ns=fingerprint.mtime_ns,
                )
                unchanged += 1
                continue

            if latest_version is not None and latest_version.sha256 == fingerprint.sha256:
                if latest_version.state == "failed":
                    unchanged += 1
                    continue
                self.jobs.enqueue(
                    knowledge_base_id=knowledge_base_id,
                    operation="add" if active_version is None else "update",
                    path=fingerprint.canonical_path,
                    expected_sha256=fingerprint.sha256,
                )
                events.append(
                    ScanEvent(
                        "add" if active_version is None else "update",
                        fingerprint.canonical_path,
                        fingerprint.sha256,
                    )
                )
                continue

            self.documents.create_version(
                document_id=document.id,
                sha256=fingerprint.sha256,
                size_bytes=fingerprint.size_bytes,
                mtime_ns=fingerprint.mtime_ns,
                parser_version="pending",
            )
            self.jobs.enqueue(
                knowledge_base_id=knowledge_base_id,
                operation="update",
                path=fingerprint.canonical_path,
                expected_sha256=fingerprint.sha256,
            )
            events.append(ScanEvent("update", fingerprint.canonical_path, fingerprint.sha256))

        visible_documents = self.documents.list_visible_for_root(
            knowledge_base_id=knowledge_base_id,
            root_path=root,
        )
        for document in visible_documents:
            if document.canonical_path in seen_paths:
                continue
            active_version = self.documents.get_active_version(document.id)
            self.documents.mark_deleted(document_id=document.id)
            self.jobs.enqueue(
                knowledge_base_id=knowledge_base_id,
                operation="delete",
                path=document.canonical_path,
                expected_sha256=active_version.sha256 if active_version else None,
            )
            events.append(
                ScanEvent(
                    "delete",
                    document.canonical_path,
                    active_version.sha256 if active_version else None,
                )
            )

        return ScanResult(
            events=tuple(events),
            unchanged=unchanged,
            skipped_unsupported=skipped_unsupported,
        )

    def fingerprint_authorized_file(
        self,
        *,
        root_path: str | Path,
        path: str | Path,
    ) -> FileFingerprint:
        authorized = authorize_path(path, root=root_path)
        return fingerprint_file(authorized.path)

    def _iter_files(
        self,
        *,
        root: Path,
        include_patterns: list[str],
        exclude_patterns: list[str],
    ) -> list[Path]:
        files: list[Path] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if not any(fnmatch(relative, pattern) for pattern in include_patterns):
                continue
            if any(fnmatch(relative, pattern) for pattern in exclude_patterns):
                continue
            files.append(path)
        return sorted(files)
