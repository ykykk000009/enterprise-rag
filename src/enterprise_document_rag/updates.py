"""GitHub Release based application update checks and downloads."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from . import __version__
from .config import Settings

_VERSION_RE = re.compile(
    r"^v?(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>[0-9A-Za-z.-]+))?$"
)
_SHA256_RE = re.compile(r"\b([0-9a-fA-F]{64})\b")


class UpdateError(RuntimeError):
    """A recoverable update error that is safe to show in the UI."""


@dataclass(frozen=True)
class ReleaseAsset:
    name: str
    download_url: str
    size: int
    sha256: str | None


@dataclass(frozen=True)
class ReleaseInfo:
    version: str
    tag_name: str
    notes: str
    html_url: str
    published_at: str | None
    asset: ReleaseAsset | None
    checksum_url: str | None


def normalize_version(value: str) -> str:
    match = _VERSION_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"invalid semantic version: {value}")
    base = ".".join(match.group(name) for name in ("major", "minor", "patch"))
    prerelease = match.group("prerelease")
    return f"{base}-{prerelease}" if prerelease else base


def is_newer_version(candidate: str, current: str) -> bool:
    return _version_key(candidate) > _version_key(current)


def _version_key(value: str) -> tuple[int, int, int, tuple[tuple[int, int | str], ...]]:
    normalized = normalize_version(value)
    release, separator, prerelease = normalized.partition("-")
    major, minor, patch = (int(part) for part in release.split("."))
    if not separator:
        pre_key: tuple[tuple[int, int | str], ...] = ((2, 0),)
    else:
        identifiers: list[tuple[int, int | str]] = []
        for item in prerelease.split("."):
            identifiers.append((0, int(item)) if item.isdigit() else (1, item.lower()))
        pre_key = tuple(identifiers)
    return major, minor, patch, pre_key


def parse_release(payload: dict[str, Any], *, prefer_offline: bool = False) -> ReleaseInfo:
    tag_name = str(payload.get("tag_name") or "").strip()
    version = normalize_version(tag_name)
    assets = payload.get("assets") or []
    suffix = "-win-x64-offline.zip" if prefer_offline else "-win-x64.zip"
    expected_name = f"DocQA-v{version}{suffix}"
    zip_assets = [item for item in assets if str(item.get("name", "")).lower().endswith(".zip")]
    selected = next((item for item in zip_assets if item.get("name") == expected_name), None)
    if selected is None and prefer_offline:
        # A newer release may intentionally publish only the Transformers online
        # package. Older offline editions can still update to that package and
        # download Qwen3 on first launch.
        standard_name = f"DocQA-v{version}-win-x64.zip"
        selected = next((item for item in zip_assets if item.get("name") == standard_name), None)
        suffix = "-win-x64.zip"
    if selected is None:
        selected = next(
            (
                item
                for item in zip_assets
                if str(item.get("name", "")).startswith(("DocQA-", "EnterpriseDocumentRAG-"))
                and str(item.get("name", "")).lower().endswith(suffix)
            ),
            None,
        )
    asset = None
    checksum_url = None
    if selected is not None:
        digest = str(selected.get("digest") or "")
        sha256 = digest.removeprefix("sha256:").lower() if digest.startswith("sha256:") else None
        if sha256 is not None and _SHA256_RE.fullmatch(sha256) is None:
            sha256 = None
        asset = ReleaseAsset(
            name=str(selected["name"]),
            download_url=str(selected["browser_download_url"]),
            size=int(selected.get("size") or 0),
            sha256=sha256,
        )
        checksum_name = f"{asset.name}.sha256"
        checksum_asset = next((item for item in assets if item.get("name") == checksum_name), None)
        if checksum_asset is not None:
            checksum_url = str(checksum_asset["browser_download_url"])
    return ReleaseInfo(
        version=version,
        tag_name=tag_name,
        notes=str(payload.get("body") or "").strip(),
        html_url=str(payload.get("html_url") or ""),
        published_at=payload.get("published_at"),
        asset=asset,
        checksum_url=checksum_url,
    )


class UpdateService:
    def __init__(
        self,
        settings: Settings,
        *,
        opener: Callable[..., Any] | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.current_version = normalize_version(__version__)
        self.repository = settings.update_repository
        self.updates_dir = settings.application_data_path / "updates"
        self.state_path = self.updates_dir / "update-state.json"
        self._open = opener or urllib.request.urlopen
        self._now = now or (lambda: datetime.now(UTC))
        executable_dir = Path(sys.executable).resolve().parent
        # Published releases use one full package. Prefer the legacy offline
        # asset when it exists, then fall back to the Transformers package.
        # This also lets older installations migrate to the current format.
        self.prefer_offline = bool(getattr(sys, "frozen", False))
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._state = self._load_state()

    def status(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._state)
        result.update(
            {
                "current_version": self.current_version,
                "repository": self.repository,
                "enabled": self.settings.update_enabled,
                "install_supported": bool(getattr(sys, "frozen", False)),
                "edition": "offline" if self.prefer_offline else "online",
            }
        )
        return result

    def start_background_check(self, *, force: bool = False) -> dict[str, Any]:
        if not self.settings.update_enabled:
            return self.status()
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return self.status()
            if not force and not self._check_is_due():
                return self.status()
            self._state.update({"state": "checking", "error": None})
            self._save_state()
            self._worker = threading.Thread(
                target=self._check_worker,
                name="application-update-check",
                daemon=True,
            )
            self._worker.start()
            return self.status()

    def check_now(self) -> dict[str, Any]:
        with self._lock:
            self._state.update({"state": "checking", "error": None})
            self._save_state()
        self._check_worker()
        return self.status()

    def skip(self, version: str) -> dict[str, Any]:
        normalized = normalize_version(version)
        with self._lock:
            self._state["skipped_version"] = normalized
            if self._state.get("latest_version") == normalized:
                self._state["state"] = "skipped"
            self._save_state()
        return self.status()

    def remind_later(self) -> dict[str, Any]:
        with self._lock:
            self._state["remind_after"] = (self._now() + timedelta(hours=24)).isoformat()
            if self._state.get("state") == "available":
                self._state["state"] = "deferred"
            self._save_state()
        return self.status()

    def acknowledge_install(self) -> dict[str, Any]:
        with self._lock:
            self._state.pop("recently_installed_version", None)
            self._save_state()
        return self.status()

    def start_download(self) -> dict[str, Any]:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                raise UpdateError("已有更新任务正在运行")
            release = self._release_from_state()
            if release is None or release.asset is None:
                raise UpdateError("当前版本没有可下载的 Windows 更新包")
            if not is_newer_version(release.version, self.current_version):
                raise UpdateError("当前已经是最新版本")
            self._state.update(
                {
                    "state": "downloading",
                    "error": None,
                    "downloaded_bytes": 0,
                    "total_bytes": release.asset.size,
                }
            )
            self._save_state()
            self._worker = threading.Thread(
                target=self._download_worker,
                args=(release,),
                name="application-update-download",
                daemon=True,
            )
            self._worker.start()
            return self.status()

    def _check_worker(self) -> None:
        try:
            release = self._fetch_latest_release()
            now = self._now().isoformat()
            state = "up_to_date"
            if is_newer_version(release.version, self.current_version):
                state = (
                    "skipped"
                    if self._state.get("skipped_version") == release.version
                    else "available"
                )
                remind_after = _parse_datetime(self._state.get("remind_after"))
                if state == "available" and remind_after is not None and remind_after > self._now():
                    state = "deferred"
            with self._lock:
                self._state.update(
                    {
                        "state": state,
                        "error": None,
                        "last_checked_at": now,
                        "release": _release_payload(release),
                        "latest_version": release.version,
                    }
                )
                self._save_state()
        except Exception as exc:
            with self._lock:
                self._state.update(
                    {
                        "state": "check_failed",
                        "error": _safe_error(exc, "检查更新失败"),
                        "last_checked_at": self._now().isoformat(),
                    }
                )
                self._save_state()

    def _download_worker(self, release: ReleaseInfo) -> None:
        try:
            assert release.asset is not None
            expected_hashes: list[str] = []
            if release.asset.sha256:
                expected_hashes.append(release.asset.sha256)
            if release.checksum_url:
                expected_hashes.append(self._fetch_checksum(release.checksum_url))
            if not expected_hashes:
                raise UpdateError("Release 未提供 SHA-256 校验值，已拒绝下载")
            if len(set(expected_hashes)) != 1:
                raise UpdateError("GitHub digest 与 .sha256 文件不一致")

            target_dir = self.updates_dir / f"v{release.version}"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / "update.zip"
            partial = target.with_suffix(".zip.part")
            digest = hashlib.sha256()
            request = self._request(release.asset.download_url)
            with self._open(
                request, timeout=self.settings.update_request_timeout_seconds
            ) as response:
                total = int(response.headers.get("Content-Length") or release.asset.size or 0)
                downloaded = 0
                with partial.open("wb") as output:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        digest.update(block)
                        downloaded += len(block)
                        with self._lock:
                            self._state.update(
                                {"downloaded_bytes": downloaded, "total_bytes": total}
                            )
                            self._save_state()
            actual = digest.hexdigest()
            if actual != expected_hashes[0]:
                partial.unlink(missing_ok=True)
                raise UpdateError("安装包 SHA-256 校验失败，下载文件已删除")
            os.replace(partial, target)
            with self._lock:
                self._state.update(
                    {
                        "state": "downloaded",
                        "error": None,
                        "download_path": str(target),
                        "downloaded_bytes": target.stat().st_size,
                        "total_bytes": target.stat().st_size,
                        "sha256": actual,
                    }
                )
                self._save_state()
        except Exception as exc:
            with self._lock:
                self._state.update(
                    {"state": "download_failed", "error": _safe_error(exc, "更新下载失败")}
                )
                self._save_state()

    def _fetch_latest_release(self) -> ReleaseInfo:
        url = f"https://api.github.com/repos/{self.repository}/releases/latest"
        request = self._request(url)
        try:
            with self._open(
                request, timeout=self.settings.update_request_timeout_seconds
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise UpdateError("GitHub 仓库尚未发布正式 Release") from exc
            if exc.code == 403:
                return self._fetch_latest_release_from_page()
                raise UpdateError("GitHub API 请求受限，请稍后重试") from exc
            raise UpdateError(f"GitHub API 返回 HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise UpdateError("无法连接 GitHub，已保留当前版本") from exc
        return parse_release(payload, prefer_offline=self.prefer_offline)

    def _fetch_latest_release_from_page(self) -> ReleaseInfo:
        """Resolve the latest release without consuming the GitHub API quota."""
        page_url = f"https://github.com/{self.repository}/releases/latest"
        request = urllib.request.Request(
            page_url,
            headers={"Accept": "text/html", "User-Agent": f"DocQA/{self.current_version}"},
        )
        try:
            with self._open(request, timeout=self.settings.update_request_timeout_seconds) as response:
                final_url = str(response.geturl())
        except (urllib.error.URLError, TimeoutError, AttributeError) as exc:
            raise UpdateError("GitHub API 受限，且无法读取 Release 页面") from exc

        match = re.search(
            r"/releases/tag/(?P<tag>v?(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?)",
            final_url,
        )
        if match is None:
            raise UpdateError("无法从 GitHub Release 页面识别版本号")

        tag_name = match.group("tag")
        version = normalize_version(tag_name)
        names = [
            f"DocQA-v{version}-win-x64-offline.zip",
            f"DocQA-v{version}-win-x64.zip",
        ]
        if not self.prefer_offline:
            names.reverse()
        selected_name = next(
            (name for name in names if self._release_asset_exists(name)),
            None,
        )
        if selected_name is None:
            raise UpdateError("最新 Release 没有可用的 Windows 更新包")

        download_url = (
            f"https://github.com/{self.repository}/releases/latest/download/{selected_name}"
        )
        return ReleaseInfo(
            version=version,
            tag_name=tag_name,
            notes="",
            html_url=final_url,
            published_at=None,
            asset=ReleaseAsset(
                name=selected_name,
                download_url=download_url,
                size=0,
                sha256=None,
            ),
            checksum_url=f"{download_url}.sha256",
        )

    def _release_asset_exists(self, name: str) -> bool:
        url = f"https://github.com/{self.repository}/releases/latest/download/{name}"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/octet-stream",
                "Range": "bytes=0-0",
                "User-Agent": f"DocQA/{self.current_version}",
            },
        )
        try:
            with self._open(request, timeout=self.settings.update_request_timeout_seconds) as response:
                response.read(1)
            return True
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            return False

    def _fetch_checksum(self, url: str) -> str:
        try:
            with self._open(
                self._request(url), timeout=self.settings.update_request_timeout_seconds
            ) as response:
                content = response.read(16 * 1024).decode("ascii", errors="ignore")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise UpdateError("无法下载 SHA-256 校验文件") from exc
        match = _SHA256_RE.search(content)
        if match is None:
            raise UpdateError(".sha256 文件格式无效")
        return match.group(1).lower()

    def _request(self, url: str) -> urllib.request.Request:
        return urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": f"DocQA/{self.current_version}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

    def _check_is_due(self) -> bool:
        last_checked = _parse_datetime(self._state.get("last_checked_at"))
        if last_checked is None:
            return True
        interval = timedelta(hours=self.settings.update_check_interval_hours)
        return self._now() - last_checked >= interval

    def _release_from_state(self) -> ReleaseInfo | None:
        payload = self._state.get("release")
        if not isinstance(payload, dict):
            return None
        asset_payload = payload.get("asset")
        asset = ReleaseAsset(**asset_payload) if isinstance(asset_payload, dict) else None
        return ReleaseInfo(
            version=payload["version"],
            tag_name=payload["tag_name"],
            notes=payload.get("notes", ""),
            html_url=payload.get("html_url", ""),
            published_at=payload.get("published_at"),
            asset=asset,
            checksum_url=payload.get("checksum_url"),
        )

    def _load_state(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {"state": "idle"}
        except (OSError, ValueError):
            return {"state": "idle"}

    def _save_state(self) -> None:
        self.updates_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, self.state_path)


def backup_database(settings: Settings, version: str) -> Path | None:
    database = settings.database_path
    if database == Path(":memory:") or not database.is_file():
        return None
    backup_dir = settings.application_data_path / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / f"agent-before-v{normalize_version(version)}.db"
    shutil.copy2(database, target)
    return target


def launch_updater(
    *,
    settings: Settings,
    status: dict[str, Any],
    shutdown_callback: Callable[[], None] | None,
) -> dict[str, str]:
    if not getattr(sys, "frozen", False):
        raise UpdateError("开发模式只支持检查和下载更新，安装请在打包版中执行")
    if status.get("state") != "downloaded":
        raise UpdateError("请先完成更新包下载和校验")
    package = Path(str(status.get("download_path") or "")).resolve()
    if not package.is_file():
        raise UpdateError("已下载的更新包不存在，请重新下载")
    version = normalize_version(str(status.get("latest_version") or ""))
    install_dir = Path(sys.executable).resolve().parent
    bundled_updater = install_dir / "Updater.exe"
    if not bundled_updater.is_file():
        raise UpdateError("独立更新器不存在，请重新安装当前版本")

    database_backup = backup_database(settings, version)
    temporary_updater = Path(tempfile.gettempdir()) / "DocQA-Updater.exe"
    shutil.copy2(bundled_updater, temporary_updater)
    command = [
        str(temporary_updater),
        "--install-dir",
        str(install_dir),
        "--package",
        str(package),
        "--pid",
        str(os.getpid()),
        "--version",
        version,
        "--executable",
        Path(sys.executable).name,
        "--database",
        str(settings.database_path.resolve()),
    ]
    if database_backup is not None:
        command.extend(["--database-backup", str(database_backup.resolve())])
    creation_flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    )
    subprocess.Popen(
        command,
        cwd=install_dir,
        close_fds=True,
        creationflags=creation_flags,
    )

    def delayed_shutdown() -> None:
        time.sleep(1.0)
        if shutdown_callback is not None:
            shutdown_callback()
        else:
            os._exit(0)

    threading.Thread(target=delayed_shutdown, name="update-shutdown", daemon=True).start()
    return {"state": "installing", "version": version}


def _release_payload(release: ReleaseInfo) -> dict[str, Any]:
    result = asdict(release)
    return result


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    except ValueError:
        return None


def _safe_error(exc: Exception, fallback: str) -> str:
    if isinstance(exc, (UpdateError, ValueError)):
        return str(exc)
    return fallback
