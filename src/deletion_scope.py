"""Shared, read-only exclusions for deletion and recovery; no database access."""

from __future__ import annotations

import json
from pathlib import Path

from src.storage_boundaries import local_directory


def is_within(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def backup_directory(path: Path) -> bool:
    """Recognize old/current directory backups without opening their databases.

    Unknown or damaged manifests fail closed. A backup does not stop being
    protected just because it is incomplete, old, or no longer validates.
    """
    path = local_directory(path)
    if not path.is_dir():
        return False
    if ".incomplete-" in path.name:
        raise ValueError("清理范围内存在未完成的备份/导出目录，请先人工核对并移出清理范围。")
    manifest = path / "manifest.json"
    if not manifest.exists():
        return False
    local_directory(manifest)
    if not manifest.is_file() or manifest.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("无法安全识别目录 manifest，已停止清理以保留可能的历史备份。")
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("目录 manifest 无法读取，已停止清理以保留可能的历史备份。") from exc
    return isinstance(value, dict) and "backup_format" in value


def require_material_path(path: Path, protected: tuple[Path, ...]) -> Path:
    """Reject protected areas and any backup ancestor even for a recorded source."""
    path = local_directory(path)
    protected = tuple(local_directory(root) for root in protected)
    if is_within(path, protected):
        raise ValueError("待删除文件位于数据库、独立备份或配置保留区域，已停止操作。")
    if path.is_file() and path.suffix.lower() in {
        ".bak", ".backup", ".db", ".sqlite", ".sqlite3", ".zip", ".7z", ".tar", ".gz",
    }:
        raise ValueError("清理范围内存在可能的备份/数据库/压缩包，请先核对并移入独立备份目录。")
    if any(backup_directory(parent) for parent in path.parents):
        raise ValueError("待删除文件属于历史备份，已停止操作；请保留备份并核对资料路径。")
    return path
