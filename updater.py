"""Standalone Windows updater for Document RAG."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

ALWAYS_RESERVED_NAMES = {"user-data", "portable.mode"}
OFFLINE_ASSET_NAMES = {
    "models",
    "tools",
    "licenses",
    "offline.mode",
    "third_party_notices.md",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-dir", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--version", required=True)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--database-backup")
    return parser.parse_args()


def _wait_for_exit(pid: int, timeout: int = 60) -> None:
    if os.name == "nt":
        synchronize = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, pid)
        if not handle:
            return
        try:
            wait_result = ctypes.windll.kernel32.WaitForSingleObject(handle, timeout * 1000)
            if wait_result != 0:
                raise RuntimeError("旧版本未能在 60 秒内退出")
            return
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.25)
    raise RuntimeError("旧版本未能在 60 秒内退出")


def _safe_extract(archive: Path, target: Path) -> None:
    target_resolved = target.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            path = PurePosixPath(member.filename.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"更新包包含不安全路径：{member.filename}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise RuntimeError(f"更新包包含不支持的符号链接：{member.filename}")
            destination = (target / Path(*path.parts)).resolve()
            if destination != target_resolved and target_resolved not in destination.parents:
                raise RuntimeError(f"更新包路径越界：{member.filename}")
        source.extractall(target)


def _payload_root(extracted: Path, executable: str) -> Path:
    if (extracted / executable).is_file():
        return extracted
    children = [child for child in extracted.iterdir() if child.is_dir()]
    if len(children) == 1 and (children[0] / executable).is_file():
        return children[0]
    raise RuntimeError(f"更新包中找不到 {executable}")


def _validate_payload(payload: Path, executable: str) -> None:
    required = (executable, "Updater.exe", "docqa.ico", "portable.mode", "version.json")
    missing = [name for name in required if not (payload / name).is_file()]
    if missing:
        raise RuntimeError(f"更新包缺少必要文件：{', '.join(missing)}")


def _refresh_shell_icons() -> None:
    if os.name != "nt":
        return
    shell_change_associated = 0x08000000
    ctypes.windll.shell32.SHChangeNotify(shell_change_associated, 0, None, None)


def _move_existing_to_backup(
    install_dir: Path,
    backup_dir: Path,
    *,
    replace_offline_assets: bool,
) -> list[str]:
    moved: list[str] = []
    backup_dir.mkdir(parents=True, exist_ok=True)
    for item in install_dir.iterdir():
        item_name = item.name.lower()
        if item_name in ALWAYS_RESERVED_NAMES:
            continue
        if not replace_offline_assets and item_name in OFFLINE_ASSET_NAMES:
            continue
        destination = backup_dir / item.name
        if destination.exists():
            _remove(destination)
        shutil.move(str(item), str(destination))
        moved.append(item.name)
    return moved


def _copy_payload(
    payload: Path,
    install_dir: Path,
    *,
    replace_offline_assets: bool,
) -> list[str]:
    copied: list[str] = []
    for item in payload.iterdir():
        item_name = item.name.lower()
        if item_name == "user-data":
            continue
        if not replace_offline_assets and item_name in OFFLINE_ASSET_NAMES:
            continue
        destination = install_dir / item.name
        if item.name.lower() == "portable.mode" and destination.exists():
            continue
        if destination.exists():
            _remove(destination)
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)
        copied.append(item.name)
    return copied


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _rollback(
    install_dir: Path,
    backup_dir: Path,
    copied: list[str],
    database: Path,
    database_backup: Path | None,
    replace_offline_assets: bool,
) -> None:
    for name in copied:
        target = install_dir / name
        if target.name.lower() not in ALWAYS_RESERVED_NAMES and target.exists():
            _remove(target)
    for item in backup_dir.iterdir():
        destination = install_dir / item.name
        if destination.exists():
            _remove(destination)
        shutil.move(str(item), str(destination))
    if database_backup is not None and database_backup.is_file():
        database.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(database_backup, database)


def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with path.open("a", encoding="utf-8") as output:
        output.write(f"{timestamp} {message}\n")


def _mark_installed(update_root: Path, version: str) -> None:
    state_path = update_root.parent / "update-state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            state = {}
    except (OSError, ValueError):
        state = {}
    state["recently_installed_version"] = version
    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, state_path)


def run() -> int:
    args = _arguments()
    install_dir = Path(args.install_dir).resolve()
    package = Path(args.package).resolve()
    database = Path(args.database).resolve()
    database_backup = Path(args.database_backup).resolve() if args.database_backup else None
    update_root = install_dir / "user-data" / "updates" / f"v{args.version}"
    extract_dir = update_root / "extracted"
    backup_dir = update_root / "rollback" / "program"
    log_path = install_dir / "user-data" / "updates" / "updater.log"
    copied: list[str] = []
    replace_offline_assets = False
    try:
        _log(log_path, f"开始更新到 v{args.version}")
        _wait_for_exit(args.pid)
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True)
        _safe_extract(package, extract_dir)
        payload = _payload_root(extract_dir, args.executable)
        _validate_payload(payload, args.executable)
        replace_offline_assets = (payload / "offline.mode").is_file()
        _move_existing_to_backup(
            install_dir,
            backup_dir,
            replace_offline_assets=replace_offline_assets,
        )
        copied = _copy_payload(
            payload,
            install_dir,
            replace_offline_assets=replace_offline_assets,
        )
        executable = install_dir / args.executable
        if not executable.is_file():
            raise RuntimeError("替换后主程序不存在")
        _refresh_shell_icons()
        process = subprocess.Popen([str(executable)], cwd=install_dir)
        time.sleep(12)
        if process.poll() is not None:
            raise RuntimeError(f"新版启动失败，退出码 {process.returncode}")
        marker = update_root / "installed.json"
        marker.write_text(
            json.dumps(
                {"version": args.version, "installed_at": datetime.now(UTC).isoformat()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _mark_installed(update_root, args.version)
        _log(log_path, f"成功更新到 v{args.version}")
        return 0
    except Exception as exc:
        _log(log_path, f"更新失败：{exc}")
        try:
            if backup_dir.is_dir():
                _rollback(
                    install_dir,
                    backup_dir,
                    copied,
                    database,
                    database_backup,
                    replace_offline_assets=replace_offline_assets,
                )
                old_executable = install_dir / args.executable
                if old_executable.is_file():
                    _refresh_shell_icons()
                    subprocess.Popen([str(old_executable)], cwd=install_dir)
                _log(log_path, "已恢复旧版本程序和数据库")
        except Exception as rollback_error:
            _log(log_path, f"回滚失败：{rollback_error}")
        return 1


if __name__ == "__main__":
    sys.exit(run())
