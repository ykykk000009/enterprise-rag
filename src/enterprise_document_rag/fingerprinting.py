import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileFingerprint:
    canonical_path: str
    size_bytes: int
    mtime_ns: int
    sha256: str


def streaming_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_file(path: str | Path) -> FileFingerprint:
    resolved = Path(path).resolve(strict=True)
    stat = resolved.stat()
    return FileFingerprint(
        canonical_path=str(resolved),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        sha256=streaming_sha256(resolved),
    )

