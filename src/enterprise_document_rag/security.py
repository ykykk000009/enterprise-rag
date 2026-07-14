from dataclasses import dataclass
from pathlib import Path


class PathAuthorizationError(ValueError):
    pass


@dataclass(frozen=True)
class AuthorizedPath:
    root: Path
    path: Path


def resolve_authorized_root(root: str | Path) -> Path:
    resolved = Path(root).resolve(strict=True)
    if not resolved.is_dir():
        raise PathAuthorizationError(f"authorized root is not a directory: {root}")
    return resolved


def authorize_path(path: str | Path, *, root: str | Path) -> AuthorizedPath:
    resolved_root = resolve_authorized_root(root)
    resolved_path = Path(path).resolve(strict=True)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise PathAuthorizationError("path escapes authorized root") from exc
    return AuthorizedPath(root=resolved_root, path=resolved_path)

