"""Local directory ownership checks; never open materials or application databases.

Claims outlive a UI session so another process cannot reuse a study directory as
platform storage. Only path metadata is kept here; neither importer calls the
other importer. Old claims are deliberately not removed by relocation or restore.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal
from uuid import uuid4

Owner = Literal["platform", "study"]
STUDY_MARKER = ".nectivon-study-loop"


class StorageBoundaryError(ValueError):
    """An operation would cross storage ownership, or ownership is unreadable."""


def local_directory(path: Path) -> Path:
    """Reject relative/network paths and Windows junctions before resolving aliases."""
    path = path.expanduser()
    if not path.is_absolute() or str(path).startswith(("\\\\", "//")):
        raise StorageBoundaryError("存储隔离要求本机绝对路径，不能使用网络共享目录。")
    for entry in (path, *path.parents):
        try:
            info = entry.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise StorageBoundaryError("存储隔离不接受符号链接或目录联接，请使用实际目录。")
    return path.resolve(strict=False)


def _registry_dir() -> Path:
    if os.name == "nt":
        value = os.environ.get("LOCALAPPDATA")
        if not value:
            raise StorageBoundaryError("无法定位本机目录隔离记录，未执行存储操作。")
        base = Path(value)
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share")))
    return local_directory(base / "Nectivon" / "storage-boundaries")


def _overlaps(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _claims(directory: Path) -> list[tuple[Owner, Path]]:
    """Read only our small path registry; malformed records never mean an empty registry."""
    result: list[tuple[Owner, Path]] = []
    if not directory.exists():
        return result
    if not directory.is_dir():
        raise StorageBoundaryError("目录隔离记录位置异常，未执行存储操作。")
    for entry in directory.iterdir():
        if entry.suffix != ".json":
            continue
        local_directory(entry)
        if entry.stat().st_size > 32768:
            raise StorageBoundaryError("目录隔离记录异常，已停止操作；请保留记录并人工核对。")
        value = json.loads(entry.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("version") != 1
            or value.get("owner") not in ("platform", "study")
            or not isinstance(value.get("path"), str)
        ):
            raise StorageBoundaryError("目录隔离记录版本或内容无效，未执行存储操作。")
        result.append((value["owner"], local_directory(Path(value["path"]))))
    return result


def _check(owner: Owner, paths: tuple[Path, ...], directory: Path) -> None:
    claims = _claims(directory)
    configured_study = os.environ.get("EKB_STUDY_LOOP_ROOT", "").strip()
    if configured_study:
        claims.append(("study", local_directory(Path(configured_study))))
    for path in paths:
        if _overlaps(path, directory):
            raise StorageBoundaryError("资料目录不能包含本机目录隔离记录，或位于其内部。")
        for other_owner, other in claims:
            if owner != other_owner and _overlaps(path, other):
                raise StorageBoundaryError(
                    "平台资料与学习整理目录必须完全分离，不能相同或互相包含；"
                    "已取消操作，原有资料保持不变。"
                )
            if owner == other_owner == "study" and path != other and _overlaps(path, other):
                raise StorageBoundaryError("两个学习整理库不能互相包含。")
        # This also protects previously created, not-yet-registered study roots.
        for parent in (path, *path.parents):
            if os.path.lexists(parent / STUDY_MARKER) and (owner == "platform" or parent != path):
                raise StorageBoundaryError("此位置属于已有学习整理库，请选择独立目录。")


def _require_no_nested_study(root: Path) -> None:
    """Inspect marker names only inside an explicitly selected maintenance tree."""
    if not root.exists():
        return

    def fail(_: OSError) -> None:
        raise StorageBoundaryError("无法完整核对维护目录，未执行备份、恢复或迁移。")

    for current, directories, files in os.walk(root, followlinks=False, onerror=fail):
        if STUDY_MARKER in files or STUDY_MARKER in directories:
            raise StorageBoundaryError(
                "平台维护目录中存在学习整理库，已取消操作；请先人工分离并保留全部资料。"
            )
        for name in directories:
            local_directory(Path(current) / name)


@contextmanager
def _registry_lock(directory: Path) -> Iterator[None]:
    """Serialize claims across UI/workers; a stale lock blocks rather than being deleted."""
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "claims.lock"
    deadline = time.monotonic() + 2
    while True:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise StorageBoundaryError(
                    "目录隔离记录正被占用，请稍后重试或人工核对锁文件。"
                ) from None
            time.sleep(0.05)
    try:
        os.close(descriptor)
        yield
    finally:
        lock.unlink()


def check_storage_paths(owner: Owner, paths: Iterable[Path]) -> tuple[Path, ...]:
    """Read-only guard, also used immediately before repeated writes by cached services."""
    try:
        resolved = tuple(dict.fromkeys(local_directory(path) for path in paths))
        _check(owner, resolved, _registry_dir())
        return resolved
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StorageBoundaryError("无法核对本机目录隔离记录，未执行存储操作。") from exc


def claim_storage_paths(
    owner: Owner, paths: Iterable[Path], *, inspect_tree: bool = False
) -> tuple[Path, ...]:
    """Reserve paths before writes; a failure never releases an existing reservation."""
    try:
        resolved = check_storage_paths(owner, paths)
        directory = _registry_dir()
        with _registry_lock(directory):
            _check(owner, resolved, directory)
            if inspect_tree:
                for path in resolved:
                    if not any(other != path and other in path.parents for other in resolved):
                        _require_no_nested_study(path)
            for path in resolved:
                identity = hashlib.sha256(f"{owner}:{path}".encode()).hexdigest()
                target = directory / f"{identity}.json"
                if target.exists():
                    continue
                value = {"version": 1, "owner": owner, "path": str(path)}
                temporary = directory / f".{identity}-{uuid4().hex}.tmp"
                with temporary.open("x", encoding="utf-8") as stream:
                    json.dump(value, stream, ensure_ascii=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
        return resolved
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StorageBoundaryError("无法保存本机目录隔离记录，未执行存储操作。") from exc
