"""Explicit file inventories for clearing an entire platform library, not user disks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.deletion_recovery import sha256_file
from src.deletion_scope import backup_directory, require_material_path
from src.storage_boundaries import check_storage_paths, local_directory


@dataclass(frozen=True)
class CleanupFile:
    path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class LibraryCleanupPlan:
    """All managed files, including unregistered originals, shown before confirmation."""

    roots: tuple[Path, ...]
    protected: tuple[Path, ...]
    files: tuple[CleanupFile, ...]


@dataclass(frozen=True)
class LibraryCleanupResult:
    complete: bool
    remaining: tuple[str, ...] = ()


def _protected(path: Path, protected: tuple[Path, ...]) -> bool:
    return any(path == item or item in path.parents for item in protected)


def capture_cleanup_plan(
    roots: tuple[Path, ...], protected: tuple[Path, ...]
) -> LibraryCleanupPlan:
    """Inventory only configured platform roots, preserving explicit backup/config areas."""
    roots = check_storage_paths("platform", roots)
    protected = tuple(local_directory(path) for path in protected)
    if any(_protected(root, protected) for root in roots):
        raise ValueError("资料/缓存目录与保留区域相交，不能安全生成全库清理清单。")
    files: dict[Path, CleanupFile] = {}
    retained = set(protected)

    def fail(_: OSError) -> None:
        raise ValueError("无法完整列出资料目录，不能确认或宣称清理完成。")

    for root in roots:
        if _protected(root, protected) or not root.exists():
            continue
        require_material_path(root, protected)
        if not root.is_dir():
            raise ValueError("资料/缓存根路径不是目录，不能核对清理范围。")
        for current, directories, names in os.walk(root, followlinks=False, onerror=fail):
            parent = local_directory(Path(current))
            if backup_directory(parent):
                retained.add(parent)
                directories[:] = []
                continue
            directories[:] = [
                name for name in directories
                if not _protected(local_directory(parent / name), protected)
            ]
            for name in names:
                path = local_directory(parent / name)
                if _protected(path, protected):
                    continue
                if not path.is_file():
                    raise ValueError("资料目录包含非普通文件，已停止清理。")
                if name == ".gitkeep" and path.stat().st_size == 0:
                    continue
                require_material_path(path, protected)
                files[path] = CleanupFile(path, path.stat().st_size, sha256_file(path))
    return LibraryCleanupPlan(
        roots, tuple(sorted(retained)), tuple(files[key] for key in sorted(files)),
    )


def remaining_approved_files(plan: LibraryCleanupPlan) -> tuple[CleanupFile, ...]:
    """Prior document deletions may remove approved files, but never authorize new files."""
    current = capture_cleanup_plan(plan.roots, plan.protected)
    if current.protected != plan.protected:
        raise ValueError("保留备份范围发生变化，请重新生成全库清理预览。")
    approved = {item.path: item for item in plan.files}
    for item in current.files:
        if approved.get(item.path) != item:
            raise ValueError("资料目录出现新增或变化的文件，请重新生成全库清理预览。")
    return current.files


def remove_empty_directories(plan: LibraryCleanupPlan) -> None:
    """Remove only empty material subdirectories; never remove protected areas or roots."""
    for root in plan.roots:
        if not root.exists() or _protected(root, plan.protected):
            continue
        candidates: list[Path] = []

        def fail(_: OSError) -> None:
            raise ValueError("无法核对资料子目录，清理尚未完成。")

        for current, directories, _ in os.walk(root, followlinks=False, onerror=fail):
            parent = local_directory(Path(current))
            directories[:] = [
                name for name in directories
                if not _protected(local_directory(parent / name), plan.protected)
            ]
            candidates.extend(parent / name for name in directories)
        for path in reversed(candidates):
            # A directory containing a protected child must remain as its parent.
            if any(path in protected.parents for protected in plan.protected):
                continue
            if path.exists() and not any(path.iterdir()):
                path.rmdir()
