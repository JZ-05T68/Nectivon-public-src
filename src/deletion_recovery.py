"""Crash-safe reconciliation of unfinished document-deletion quarantines.

Every document deletion moves the document's recorded files into a
per-operation directory ``.deletion-quarantine/op-<uuid>/`` together with an
atomically written ``manifest.json`` before the database transaction runs.
If the process dies between those stages the operation directory survives,
and this module decides — at the next startup or on demand — how to finish
it, using only observable facts:

- the manifest (which files, which document, which hashes);
- the real filesystem state (which files exist where, with which content);
- the database (whether the document row still exists).

The database is the only authority on whether the deletion committed:

- document row exists  -> the deletion never committed; restore every
  quarantined file to its recorded original path (Case 1);
- document row absent  -> the deletion committed; verify every quarantined
  copy against its manifest SHA-256 and destroy the operation directory
  (Case 2). Files that reappeared at original paths are never touched.

Anything ambiguous — missing or corrupt manifest, illegal paths, hash
mismatches, conflicting copies, undeterminable document state — is
fail-closed: the operation directory is preserved untouched and reported as
``attention`` so a human can inspect it. Reconciliation is idempotent, and
one broken operation never blocks the others. Follows the project
discipline: no silent exception handling, all user-facing messages in
Chinese.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

from src.database import Database
from src.deletion_scope import require_material_path
from src.models import QuarantineOperationReport, QuarantineReconciliation
from src.storage_boundaries import check_storage_paths, local_directory

LOGGER = logging.getLogger(__name__)

QUARANTINE_DIR_NAME = ".deletion-quarantine"
MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1

STATUS_RESTORED = "restored"
STATUS_COMPLETED = "completed"
STATUS_ATTENTION = "attention"

_REQUIRED_MANIFEST_KEYS = (
    "version",
    "operation_id",
    "document_id",
    "document_title",
    "created_at",
    "files",
)
_REQUIRED_FILE_KEYS = ("original_path", "quarantine_path", "size_bytes", "sha256")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of one file, read in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON via a temporary file, flush+fsync, then atomic replace.

    The formal file is never written in place, so a crash can leave at most
    a harmless ``*.tmp-*`` sibling, never a half-written manifest.
    """

    temporary = path.with_name(f"{path.name}.tmp-{os.urandom(6).hex()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def move_verified_file(source: Path, destination: Path, expected_sha256: str) -> None:
    """Stage/restore across local volumes without discarding the only verified copy."""
    local_directory(source)
    local_directory(destination)
    if os.path.lexists(destination):
        raise OSError("移动目标已有文件，未覆盖。")
    if sha256_file(source) != expected_sha256:
        raise OSError("文件内容在确认后变化，未移动。")
    try:
        os.replace(source, destination)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV and getattr(exc, "winerror", None) != 17:
            raise
    with source.open("rb") as original, destination.open("xb") as staged:
        shutil.copyfileobj(original, staged, 1024 * 1024)
        staged.flush()
        os.fsync(staged.fileno())
    if sha256_file(destination) != expected_sha256 or sha256_file(source) != expected_sha256:
        raise OSError("跨卷复制校验失败，原文件和隔离副本均保留。")
    source.unlink()


def purge_verified_operation(operation: Path) -> None:
    """Remove only manifest-listed copies; unknown files/links are never swept away."""
    operation = local_directory(operation)
    manifest_path = local_directory(operation / MANIFEST_NAME)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problem = _validate_manifest(manifest, operation)
    if problem:
        raise ValueError(problem)
    expected: dict[Path, str] = {}
    directories = {operation, operation / "files"}
    for entry in manifest["files"]:
        path = local_directory(Path(entry["quarantine_path"]))
        if not path.is_relative_to(operation / "files") or path == operation / "files":
            raise ValueError("隔离文件不在本次操作的 files 目录内，未清理。")
        if path in expected:
            raise ValueError("隔离清单存在重复路径，未清理。")
        expected[path] = entry["sha256"]
        directories.update(parent for parent in path.parents if parent.is_relative_to(operation))
    actual: list[Path] = []
    metadata = [manifest_path]

    def fail(exc: OSError) -> None:
        raise OSError("无法完整核对隔离目录，已保留现场。") from exc

    for current, children, names in os.walk(operation, followlinks=False, onerror=fail):
        parent = local_directory(Path(current))
        for name in children:
            if local_directory(parent / name) not in directories:
                raise ValueError("隔离目录出现未登记子目录，已保留现场。")
        for name in names:
            path = local_directory(parent / name)
            if path == manifest_path:
                continue
            if parent == operation and name.startswith(MANIFEST_NAME + ".tmp-"):
                metadata.append(path)
            elif path not in expected or not path.is_file():
                raise ValueError("隔离目录出现未登记文件，已保留现场。")
            elif sha256_file(path) != expected[path]:
                raise ValueError("隔离副本摘要变化，已保留现场。")
            else:
                actual.append(path)
    # No recursive removal: a file introduced after the inventory makes rmdir
    # fail rather than becoming an implicitly authorized deletion.
    for path in actual:
        local_directory(path)
        if sha256_file(path) != expected[path]:
            raise ValueError("隔离副本在清理期间变化，未继续删除。")
        path.unlink()
    for path in sorted(directories - {operation}, key=lambda p: len(p.parts), reverse=True):
        if path.exists():
            local_directory(path).rmdir()
    for path in metadata:
        local_directory(path).unlink(missing_ok=True)
    operation.rmdir()


def reconcile_quarantine(
    *,
    database: Database,
    data_dir: Path,
    raw_dir: Path,
    pages_dir: Path,
    markdown_dir: Path,
    agent_readings_dir: Path | None = None,
    cache_dir: Path | None = None,
    backups_dir: Path | None = None,
) -> QuarantineReconciliation:
    """Inspect every quarantine operation directory and settle what is provable.

    Never raises for an individual operation: unexpected errors are logged
    and reported as ``attention`` (fail-closed) so one broken operation
    cannot block the reconciliation of the others or crash the caller.
    """

    root = Path(data_dir) / QUARANTINE_DIR_NAME
    check_storage_paths("platform", (Path(data_dir), root))
    require_material_path(root, (
        database.database_path.parent,
        *(() if backups_dir is None else (backups_dir,)),
    ))
    if not root.is_dir():
        return QuarantineReconciliation()
    reports: list[QuarantineOperationReport] = []
    for operation_dir in sorted(root.iterdir()):
        try:
            reports.append(
                _reconcile_operation(
                    operation_dir,
                    database=database,
                    library_roots=(
                        (Path(data_dir), Path(raw_dir), Path(pages_dir), Path(markdown_dir),
                         (Path(agent_readings_dir) if agent_readings_dir
                          else Path(data_dir) / "agent-readings"),
                         cache_dir)
                        if cache_dir is not None and backups_dir is not None else ()
                    ),
                    protected=(database.database_path.parent, root,
                               Path(data_dir) / ".residue-cleanup-snapshots",
                               Path(data_dir) / "ui_preferences.json",
                               *(() if backups_dir is None else (backups_dir,))),
                    managed_roots=(
                        Path(raw_dir),
                        Path(pages_dir),
                        Path(markdown_dir),
                        Path(agent_readings_dir)
                        if agent_readings_dir is not None
                        else Path(data_dir) / "agent-readings",
                    ),
                )
            )
        except Exception as exc:  # fail closed, keep the operation inspectable
            LOGGER.exception("隔离区对账出现异常，已保留现场：%s", operation_dir)
            reports.append(
                QuarantineOperationReport(
                    operation_id=operation_dir.name,
                    quarantine_path=operation_dir,
                    document_id=None,
                    status=STATUS_ATTENTION,
                    detail=(
                        f"对账过程出现异常：{exc}。"
                        "系统未自动删除或覆盖这些文件，请人工检查。"
                    ),
                )
            )
    return QuarantineReconciliation(operations=tuple(reports))


def _reconcile_operation(
    operation_dir: Path,
    *,
    database: Database,
    managed_roots: tuple[Path, ...],
    library_roots: tuple[Path, ...] = (),
    protected: tuple[Path, ...] = (),
) -> QuarantineOperationReport:
    """Settle one operation directory; see the module docstring for the rules."""

    operation_id = operation_dir.name
    local_directory(operation_dir)
    if not operation_dir.is_dir():
        return _attention(
            operation_id,
            operation_dir,
            None,
            "隔离区内存在无法识别的条目（不是操作目录）",
        )
    manifest_path = operation_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return _attention(
            operation_id,
            operation_dir,
            None,
            "缺少删除操作 manifest，无法判定该操作的归属与计划文件",
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _attention(
            operation_id,
            operation_dir,
            None,
            f"删除操作 manifest 损坏或无法解析：{exc}",
        )
    problem = _validate_manifest(manifest, operation_dir)
    if problem is not None:
        return _attention(operation_id, operation_dir, None, problem)

    document_id = int(manifest["document_id"])
    try:
        document = (
            bool(database.list_documents())
            if manifest.get("operation_kind") == "library_cleanup"
            else database.get_document(document_id)
        )
    except Exception as exc:
        LOGGER.exception("隔离区对账无法读取文档状态：document_id=%s", document_id)
        return _attention(
            operation_id,
            operation_dir,
            document_id,
            f"无法判定文档当前状态（数据库读取失败：{exc}），已跳过该操作",
        )

    library_cleanup = manifest.get("operation_kind") == "library_cleanup"
    if library_cleanup and not library_roots:
        return _attention(
            operation_id, operation_dir, document_id,
            "全库清理对账需要完整的缓存和备份目录配置，请从当前平台维护入口处理",
        )
    files, path_problem = _validated_file_entries(
        manifest["files"], operation_dir, library_roots if library_cleanup else managed_roots
    )
    if path_problem is not None:
        return _attention(operation_id, operation_dir, document_id, path_problem)

    for original, _, _ in files:
        try:
            require_material_path(original, protected)
        except (OSError, ValueError) as exc:
            return _attention(operation_id, operation_dir, document_id, str(exc))

    if library_cleanup:
        for original, _, _ in files:
            if any(original == path or path in original.parents for path in protected):
                return _attention(operation_id, operation_dir, document_id, "全库清理触及保留区域")
        if document or any(os.path.lexists(original) for original, _, _ in files):
            return _restore_operation(operation_id, operation_dir, document_id, files)
        return _complete_operation(operation_id, operation_dir, document_id, files)
    if document is not None:
        return _restore_operation(operation_id, operation_dir, document_id, files)
    return _complete_operation(operation_id, operation_dir, document_id, files)


def _validate_manifest(manifest: object, operation_dir: Path) -> str | None:
    """Return a fail-closed reason when the manifest cannot be trusted."""

    if not isinstance(manifest, dict):
        return "删除操作 manifest 不是有效的 JSON 对象"
    missing = [key for key in _REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing:
        return "删除操作 manifest 缺少必要字段：" + "、".join(missing)
    if manifest["version"] != MANIFEST_VERSION:
        return f"删除操作 manifest 版本不受支持：{manifest['version']}"
    if not isinstance(manifest["operation_id"], str) or not manifest["operation_id"]:
        return "删除操作 manifest 的 operation_id 无效"
    if operation_dir.name != f"op-{manifest['operation_id']}":
        return "删除操作 manifest 与操作目录名不一致，无法证明归属"
    is_library = manifest.get("operation_kind") == "library_cleanup"
    if type(manifest["document_id"]) is not int or (
        manifest["document_id"] != 0 if is_library else manifest["document_id"] <= 0
    ):
        return "删除操作 manifest 的 document_id 无效"
    if not isinstance(manifest["files"], list):
        return "删除操作 manifest 的文件清单无效"
    for entry in manifest["files"]:
        if not isinstance(entry, dict):
            return "删除操作 manifest 存在无效的文件条目"
        entry_missing = [key for key in _REQUIRED_FILE_KEYS if key not in entry]
        if entry_missing:
            return "删除操作 manifest 文件条目缺少字段：" + "、".join(entry_missing)
        sha256 = entry["sha256"]
        if not isinstance(sha256, str) or len(sha256) != 64:
            return "删除操作 manifest 文件条目的 SHA-256 无效"
    return None


def _validated_file_entries(
    entries: list[dict],
    operation_dir: Path,
    managed_roots: tuple[Path, ...],
) -> tuple[list[tuple[Path, Path, str]], str | None]:
    """Resolve and validate every manifest path; any anomaly fails closed."""

    resolved_roots = tuple(root.resolve() for root in managed_roots)
    operation_resolved = operation_dir.resolve()
    files: list[tuple[Path, Path, str]] = []
    for entry in entries:
        original = Path(entry["original_path"])
        quarantined = Path(entry["quarantine_path"])
        try:
            original = local_directory(original)
            quarantined = local_directory(quarantined)
            check_storage_paths("platform", (original, quarantined))
        except (OSError, ValueError) as exc:
            return [], f"manifest 路径隔离检查失败：{exc}"
        if ".." in original.parts or ".." in quarantined.parts:
            return [], f"manifest 中存在包含“..”段的路径：{entry}"
        if not quarantined.resolve().is_relative_to(operation_resolved):
            return [], f"manifest 中的隔离路径不在操作目录内：{quarantined}"
        if not any(
            original.resolve().is_relative_to(root) for root in resolved_roots
        ) or original.resolve() in resolved_roots:
            return [], f"manifest 中的原始路径不在受管数据目录内：{original}"
        for candidate in (original, quarantined):
            if candidate.is_symlink():
                return [], f"manifest 中的路径是符号链接：{candidate}"
        files.append((original, quarantined, str(entry["sha256"])))
    return files, None


def _restore_operation(
    operation_id: str,
    operation_dir: Path,
    document_id: int,
    files: list[tuple[Path, Path, str]],
) -> QuarantineOperationReport:
    """Case 1: the document still exists, so the deletion never committed.

    Every recorded file is returned to its original path. Decisions are made
    per file from the real filesystem state plus SHA-256, never from a flag.
    """

    problems: list[str] = []
    for original, quarantined, expected_sha256 in files:
        original_exists = original.is_file()
        quarantine_exists = quarantined.is_file()
        original_match = (
            original_exists and sha256_file(original) == expected_sha256
        )
        quarantine_match = (
            quarantine_exists and sha256_file(quarantined) == expected_sha256
        )
        if quarantine_exists and not quarantine_match:
            problems.append(f"隔离副本内容与登记不一致：{quarantined}")
        elif original_exists and not original_match:
            problems.append(
                f"原始位置出现与登记内容不同的文件，未覆盖：{original}"
            )
        elif quarantine_match and not original_exists:
            try:
                original.parent.mkdir(parents=True, exist_ok=True)
                move_verified_file(quarantined, original, expected_sha256)
            except OSError as exc:
                problems.append(f"恢复文件失败：{original}（{exc}）")
            else:
                if sha256_file(original) != expected_sha256:
                    problems.append(f"恢复后校验不一致：{original}")
        elif quarantine_match and original_match:
            # Both copies are provably the recorded file: the quarantine
            # copy is redundant and can be removed safely.
            try:
                quarantined.unlink()
            except OSError as exc:
                problems.append(f"移除多余隔离副本失败：{quarantined}（{exc}）")
        elif not quarantine_exists and original_match:
            pass  # Already restored by an earlier reconciliation run.
        else:
            problems.append(f"文件在原始位置与隔离区均缺失：{original}")
    if problems:
        LOGGER.warning(
            "删除操作 %s 恢复未完成（document_id=%s）：%s",
            operation_id,
            document_id,
            problems,
        )
        return _attention(
            operation_id,
            operation_dir,
            document_id,
            "删除未完成且无法安全恢复："
            + "；".join(problems)
            + "。系统未自动删除或覆盖这些文件，请人工检查",
        )
    try:
        purge_verified_operation(operation_dir)
    except (OSError, ValueError) as exc:
        return _attention(
            operation_id,
            operation_dir,
            document_id,
            f"文件已全部恢复原位，但操作目录未能移除：{exc}",
        )
    LOGGER.info(
        "删除操作 %s 已回滚恢复：document_id=%s files=%s",
        operation_id,
        document_id,
        len(files),
    )
    return QuarantineOperationReport(
        operation_id=operation_id,
        quarantine_path=operation_dir,
        document_id=document_id,
        status=STATUS_RESTORED,
        detail=f"删除未提交，已将 {len(files)} 个登记文件恢复原位。",
    )


def _complete_operation(
    operation_id: str,
    operation_dir: Path,
    document_id: int,
    files: list[tuple[Path, Path, str]],
) -> QuarantineOperationReport:
    """Case 2: the document row is gone, so the deletion committed.

    Only quarantine copies whose SHA-256 matches the manifest are destroyed.
    Anything that reappeared at an original path is reported but never
    touched — it may be a file the user put back themselves.
    """

    notes: list[str] = []
    for original, quarantined, expected_sha256 in files:
        if quarantined.is_file() and sha256_file(quarantined) != expected_sha256:
            return _attention(
                operation_id,
                operation_dir,
                document_id,
                f"隔离副本内容与登记不一致：{quarantined}。"
                "系统未自动删除或覆盖这些文件，请人工检查",
            )
        if original.is_file():
            if sha256_file(original) == expected_sha256:
                notes.append(f"原始位置存在内容一致的文件（可能是人工放回），未触碰：{original}")
            else:
                notes.append(f"原始位置存在内容不同的文件，未触碰：{original}")
    try:
        purge_verified_operation(operation_dir)
    except (OSError, ValueError) as exc:
        return _attention(
            operation_id,
            operation_dir,
            document_id,
            f"删除已提交，但隔离目录销毁失败（下次启动将重试）：{exc}",
        )
    LOGGER.info(
        "删除操作 %s 已完成清理：document_id=%s files=%s",
        operation_id,
        document_id,
        len(files),
    )
    detail = f"删除已提交，已销毁 {len(files)} 个隔离文件。"
    if notes:
        detail += " " + "；".join(notes) + "。"
    return QuarantineOperationReport(
        operation_id=operation_id,
        quarantine_path=operation_dir,
        document_id=document_id,
        status=STATUS_COMPLETED,
        detail=detail,
    )


def _attention(
    operation_id: str,
    operation_dir: Path,
    document_id: int | None,
    reason: str,
) -> QuarantineOperationReport:
    """Build a fail-closed report; the operation directory stays untouched."""

    LOGGER.warning(
        "删除隔离区操作需要人工处理：operation=%s document_id=%s 原因=%s 位置=%s",
        operation_id,
        document_id,
        reason,
        operation_dir,
    )
    return QuarantineOperationReport(
        operation_id=operation_id,
        quarantine_path=operation_dir,
        document_id=document_id,
        status=STATUS_ATTENTION,
        detail=f"{reason}。系统未自动删除或覆盖这些文件。",
    )
