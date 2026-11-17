"""Staged, verifiable deletion of one imported document and its own files.

The deletion never deletes anything in place. Recorded files are first
moved into a per-operation quarantine directory
(``.deletion-quarantine/op-<uuid>/``) with same-volume atomic renames —
but only after an atomically written ``manifest.json`` (operation id,
document id, and every planned file with its original path, quarantine
path, size and SHA-256) is reliably on disk, so an interrupted operation
can always be settled later by :mod:`src.deletion_recovery`. Only after
the single-transaction database delete (including per-table residue checks
and ``PRAGMA foreign_key_check``) has committed is the quarantine removed
permanently. Any failure before the commit rolls the database back and
moves every quarantined file back to its original location, verifying
presence and size. Files not recorded as belonging to the document are
never touched, and projects/tags are shared entities that always survive.
Follows the established project discipline: connections go through
``Database._connection()``, all user-facing errors are explicit Chinese
messages, and no exception is ever swallowed silently.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from src.aggregation_service import AggregationService
from src.database import Database, DatabaseError
from src.deletion_recovery import (
    MANIFEST_NAME,
    MANIFEST_VERSION,
    QUARANTINE_DIR_NAME,
    move_verified_file,
    purge_verified_operation,
    sha256_file,
    write_json_atomic,
)
from src.deletion_scope import require_material_path
from src.library_cleanup import (
    LibraryCleanupPlan,
    LibraryCleanupResult,
    capture_cleanup_plan,
    remaining_approved_files,
    remove_empty_directories,
)
from src.models import (
    DocumentDeletionFile,
    DocumentDeletionPreview,
    DocumentDeletionResult,
)
from src.storage_boundaries import (
    StorageBoundaryError,
    check_storage_paths,
    claim_storage_paths,
    local_directory,
)

LOGGER = logging.getLogger(__name__)

#: Knowledge objects mined from source material carry
#: ``epistemic_basis='source_derived'``; when they lose their last source
#: they become untraceable, searchable orphans and are removed with the
#: document. Objects resting on a personal basis (experience, judgment,
#: direct observation, decision, problem definition) are user knowledge:
#: they survive detached exactly like organized questions do.
_ORPHAN_CASCADE_EPISTEMIC_BASES = ("source_derived",)


def _knowledge_source_scope(
    alias: str, document_id: int, page_ids: list[int]
) -> tuple[str, list[object]]:
    """Build the polymorphic ``knowledge_object_sources`` scope predicate.

    Returns ``(condition_sql, parameters)`` matching every source link that
    points into ``document_id``: the document itself, one of its pages, or a
    note/evidence item belonging to either. ``alias`` prefixes the columns so
    the same predicate serves preview counting and transactional deletion.
    """

    placeholders = ",".join("?" for _ in page_ids) or "NULL"
    condition = (
        f"({alias}.source_type = 'document' AND {alias}.source_id = ?) "
        f"OR ({alias}.source_type = 'page' AND {alias}.source_id IN ({placeholders})) "
        f"OR ({alias}.source_type = 'note' AND {alias}.source_id IN ("
        f"SELECT id FROM notes WHERE document_id = ? OR page_id IN ({placeholders}))) "
        f"OR ({alias}.source_type = 'evidence' AND {alias}.source_id IN ("
        f"SELECT id FROM evidence_items WHERE document_id = ? "
        f"OR page_id IN ({placeholders})))"
    )
    parameters: list[object] = [
        document_id,
        *page_ids,
        document_id,
        *page_ids,
        document_id,
        *page_ids,
    ]
    return condition, parameters


def _select_cascade_orphan_candidates(
    connection, document_id: int, page_ids: list[int]  # noqa: ANN001
) -> list[int]:
    """Knowledge objects whose sources all live on this document (read-only).

    Restricted to ``source_derived`` objects: only they claim source
    traceability as their epistemic basis, so only they become evidence-less
    orphans when the document goes away. Called BEFORE the source links are
    deleted; the zero-source verdict itself is re-checked afterwards inside
    the same transaction.
    """

    bases = ",".join("?" for _ in _ORPHAN_CASCADE_EPISTEMIC_BASES)
    condition, parameters = _knowledge_source_scope("s", document_id, page_ids)
    rows = connection.execute(
        f"""
        SELECT DISTINCT ko.id FROM knowledge_objects ko
        JOIN knowledge_object_sources s ON s.knowledge_object_id = ko.id
        WHERE ko.epistemic_basis IN ({bases})
          AND ({condition})
          AND NOT EXISTS (
              SELECT 1 FROM knowledge_object_sources outside
              WHERE outside.knowledge_object_id = ko.id AND NOT ({condition})
          )
        """,
        (*_ORPHAN_CASCADE_EPISTEMIC_BASES, *parameters, *parameters),
    ).fetchall()
    return [int(row[0]) for row in rows]


class DocumentDeletionError(DatabaseError):
    """A document deletion was refused, aborted, or only partially recoverable."""


def preview_signature(preview: DocumentDeletionPreview) -> str:
    """Bind confirmation to the displayed identity, impact counts and file inventory."""
    payload = json.dumps(asdict(preview), default=str, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DocumentDeletionService:
    """Preview and execute the complete, staged deletion of one document."""

    def __init__(
        self,
        *,
        database: Database,
        raw_dir: Path,
        pages_dir: Path,
        markdown_dir: Path,
        data_dir: Path,
        agent_readings_dir: Path | None = None,
        app_version: str = "",
        cache_dir: Path | None = None,
        backups_dir: Path | None = None,
    ) -> None:
        self._database = database
        self._raw_dir = Path(raw_dir)
        self._pages_dir = Path(pages_dir)
        self._markdown_dir = Path(markdown_dir)
        self._data_dir = Path(data_dir)
        self._agent_readings_dir = (
            Path(agent_readings_dir)
            if agent_readings_dir is not None
            else self._data_dir / "agent-readings"
        )
        self._app_version = app_version
        self._cache_dir = cache_dir
        self._backups_dir = backups_dir
        self._quarantine_root = self._data_dir / QUARANTINE_DIR_NAME
        self._storage_claimed = False

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")

    # ------------------------------------------------------------- preview
    def preview_document_deletion(self, document_id: int) -> DocumentDeletionPreview:
        """Build a read-only deletion impact summary without writing anything.

        Raises :class:`DocumentDeletionError` when the document does not
        exist. Every recorded file path is validated against its owning root
        directory; suspicious paths are reported in ``path_anomalies`` and
        abort any later deletion instead of being followed.
        """

        self._check_storage()
        document = self._database.get_document(document_id)
        if document is None:
            raise DocumentDeletionError(f"找不到文档：{document_id}")
        pages = self._database.list_pages(document_id)
        page_ids = [page.id for page in pages]
        placeholders = ",".join("?" for _ in page_ids) or "NULL"

        with self._database._connection() as connection:
            self._require_idle(connection, document_id)
            note_counts = {"document": 0, "page": 0, "text_selection": 0, "image_region": 0}
            row = connection.execute(
                "SELECT COUNT(*) FROM notes WHERE document_id = ?", (document_id,)
            ).fetchone()
            note_counts["document"] = int(row[0])
            for row in connection.execute(
                f"SELECT note_type, COUNT(*) FROM notes "
                f"WHERE page_id IN ({placeholders}) GROUP BY note_type",
                page_ids,
            ).fetchall():
                note_counts[str(row[0])] = int(row[1])
            evidence_count = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM evidence_items "
                    f"WHERE document_id = ? OR page_id IN ({placeholders})",
                    (document_id, *page_ids),
                ).fetchone()[0]
            )
            search_count = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM page_search WHERE rowid IN ({placeholders})",
                    page_ids,
                ).fetchone()[0]
            )
            association_count = sum(
                int(connection.execute(query, parameters).fetchone()[0])
                for query, parameters in (
                    ("SELECT COUNT(*) FROM document_tags WHERE document_id = ?", (document_id,)),
                    (
                        f"SELECT COUNT(*) FROM page_tags WHERE page_id IN ({placeholders})",
                        page_ids,
                    ),
                    (
                        "SELECT COUNT(*) FROM project_documents WHERE document_id = ?",
                        (document_id,),
                    ),
                    (
                        f"SELECT COUNT(*) FROM project_pages WHERE page_id IN ({placeholders})",
                        page_ids,
                    ),
                )
            )
            import_record_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM import_records WHERE document_id = ?",
                    (document_id,),
                ).fetchone()[0]
            )
            knowledge_object_source_count = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*) FROM knowledge_object_sources
                    WHERE (source_type = 'document' AND source_id = ?)
                       OR (source_type = 'page' AND source_id IN ({placeholders}))
                       OR (source_type = 'note' AND source_id IN (
                               SELECT id FROM notes
                               WHERE document_id = ? OR page_id IN ({placeholders})
                           ))
                       OR (source_type = 'evidence' AND source_id IN (
                               SELECT id FROM evidence_items
                               WHERE document_id = ? OR page_id IN ({placeholders})
                           ))
                    """,
                    (
                        document_id,
                        *page_ids,
                        document_id,
                        *page_ids,
                        document_id,
                        *page_ids,
                    ),
                ).fetchone()[0]
            )
            # v0.8.8: source-derived knowledge objects that lose their last
            # source with this document are cascade-deleted (relations and
            # FTS rows go with them); personal-basis objects survive detached.
            knowledge_object_orphan_count = len(
                _select_cascade_orphan_candidates(connection, document_id, page_ids)
            )
            page_visual_index_count = (
                int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM page_visual_index "
                        f"WHERE page_id IN ({placeholders})",
                        page_ids,
                    ).fetchone()[0]
                )
                if page_ids
                else 0
            )
            # Learning-layer assets (schema v19+) organized on this document.
            # They survive the deletion detached (ON DELETE SET NULL); the
            # counts below are the user-visible "what stays, but loses its
            # source and its live evidence" impact.
            learning_question_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM question_items WHERE document_id = ?",
                    (document_id,),
                ).fetchone()[0]
            )
            question_placeholders = ",".join("?" for _ in page_ids) or "NULL"
            learning_family_count = int(
                connection.execute(
                    f"""
                    SELECT COUNT(DISTINCT family_id) FROM question_family_members
                    WHERE question_id IN (
                        SELECT id FROM question_items
                        WHERE document_id = ?
                           OR page_id IN ({question_placeholders})
                    )
                    """,
                    (document_id, *page_ids),
                ).fetchone()[0]
            )
            learning_mastery_record_count = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*) FROM mastery_records
                    WHERE question_id IN (
                        SELECT id FROM question_items
                        WHERE document_id = ?
                           OR page_id IN ({question_placeholders})
                    )
                    """,
                    (document_id, *page_ids),
                ).fetchone()[0]
            )
            learning_evidence_link_count = int(
                connection.execute(
                    f"""
                    SELECT COUNT(DISTINCT qie.id)
                    FROM question_item_evidence AS qie
                    LEFT JOIN question_group_materials AS qgm
                           ON qgm.id = qie.source_id
                    WHERE qie.question_item_id IN (
                        SELECT id FROM question_items
                        WHERE document_id = ?
                           OR page_id IN ({question_placeholders})
                    )
                       OR qie.page_id IN ({question_placeholders})
                       OR qgm.page_id IN ({question_placeholders})
                    """,
                    (
                        document_id,
                        *page_ids,
                        *page_ids,
                        *page_ids,
                    ),
                ).fetchone()[0]
            )

        files, missing_files, path_anomalies, total_size = self._collect_files(
            document_id=document_id,
            source_path=document.source_path,
            page_ids=page_ids,
            image_paths=[page.image_path for page in pages],
            markdown_paths=[
                page.markdown_path for page in pages if page.markdown_path is not None
            ],
        )
        aggregation_impact = AggregationService(
            self._database
        ).get_document_aggregation_impacts(document_id)
        return DocumentDeletionPreview(
            document_id=document.id,
            document_title=document.title,
            document_sha256=document.sha256,
            page_count=len(pages),
            document_note_count=note_counts["document"],
            page_note_count=note_counts["page"],
            text_selection_note_count=note_counts["text_selection"],
            image_region_note_count=note_counts["image_region"],
            evidence_item_count=evidence_count,
            search_record_count=search_count,
            association_count=association_count,
            import_record_count=import_record_count,
            knowledge_object_source_count=knowledge_object_source_count,
            knowledge_object_orphan_count=knowledge_object_orphan_count,
            page_visual_index_count=page_visual_index_count,
            learning_question_count=learning_question_count,
            learning_family_count=learning_family_count,
            learning_mastery_record_count=learning_mastery_record_count,
            learning_evidence_link_count=learning_evidence_link_count,
            files=files,
            total_size_bytes=total_size,
            missing_files=missing_files,
            path_anomalies=path_anomalies,
            aggregation_impact=aggregation_impact,
        )

    # ------------------------------------------------------------ deletion
    def delete_document(
        self, document_id: int, *, expected_title: str,
        expected_preview: DocumentDeletionPreview | None = None,
        expected_file_hashes: dict[Path, str] | None = None,
    ) -> DocumentDeletionResult:
        """Delete one document in verifiable stages, never faking success.

        ``expected_title`` is a mandatory title-confirmation invariant: it
        must equal the document title re-read at execution time exactly —
        no stripping, no case folding, no normalization. Any mismatch
        aborts before a single side effect (no quarantine directory, no
        file move, no database change). This is defense in depth on top of
        the UI's own confirmation gating, so no caller can delete by
        document id alone. Stage 1 then re-validates the document and
        every recorded path; any anomaly aborts before anything is
        touched. Stage 2 writes the per-operation quarantine manifest
        (every planned file with its SHA-256) atomically — before the
        first file moves, so a crash at any later point is recoverable by
        :mod:`src.deletion_recovery`. Stage 3 moves recorded files into
        the operation directory with atomic same-volume renames. Stage 4
        runs the single cascading ``DELETE`` plus per-table residue checks
        in one transaction, rolling back on any surprise. Stage 5-6
        permanently remove the quarantine only after the commit.
        """

        preview = self.preview_document_deletion(document_id)
        if expected_preview is not None and preview_signature(preview) != preview_signature(
            expected_preview
        ):
            raise DocumentDeletionError("资料或删除影响已变化，请重新生成预览并确认；未删除本项。")
        if expected_title != preview.document_title:
            raise DocumentDeletionError(
                "标题确认不匹配，已中止删除，未改动任何数据："
                f"文档“{preview.document_title}”（id={document_id}）"
            )
        if preview.path_anomalies:
            raise DocumentDeletionError(
                "检测到路径异常，已中止删除，未改动任何数据："
                + "；".join(preview.path_anomalies)
            )

        self._check_storage(claim=True)
        with self._database._connection() as connection:
            self._require_idle(connection, document_id)
        quarantine_dir, manifest_path, manifest, moved = self._stage_files(
            preview.files, document_id, preview.document_title,
            expected_hashes=expected_file_hashes,
        )

        try:
            self._delete_document_records(document_id)
        except Exception as exc:
            self._abort_with_restore(
                moved,
                quarantine_dir,
                f"删除文档数据库记录失败，已回滚数据库：{exc}",
            )
        self._update_manifest_phase(manifest_path, manifest, "db_committed")

        cleanup_warnings: list[str] = [
            "仅处理预览中已登记归属的文件；未登记或共享原件、未登记派生文件会保留，"
            "本次结果不表示存储目录已清空。"
        ]
        try:
            purge_verified_operation(quarantine_dir)
        except (OSError, ValueError) as exc:
            warning = f"删除已完成，但隔离目录未能清理，已保留：{quarantine_dir}（{exc}）"
            LOGGER.warning("隔离目录清理失败：%s（%s）", quarantine_dir, exc)
            cleanup_warnings.append(warning)
        for directory in (
            self._pages_dir / str(document_id),
            self._markdown_dir / str(document_id),
        ):
            try:
                directory.rmdir()
            except FileNotFoundError:
                pass
            except OSError:
                warning = f"删除已完成，但目录非空（含未登记文件），已保留：{directory}"
                LOGGER.warning("文档目录非空，已保留：%s", directory)
                cleanup_warnings.append(warning)

        LOGGER.info(
            "导入文件已删除：document_id=%s title=%s files=%s warnings=%s",
            document_id,
            preview.document_title,
            len(moved),
            len(cleanup_warnings),
        )
        return DocumentDeletionResult(
            document_id=preview.document_id,
            document_title=preview.document_title,
            preview=preview,
            deleted=True,
            cleanup_warnings=tuple(cleanup_warnings),
        )

    def _stage_files(
        self, files: tuple[DocumentDeletionFile, ...], document_id: int, title: str,
        *, operation_kind: str = "document",
        expected_hashes: dict[Path, str] | None = None,
    ) -> tuple[Path, Path, dict, list[tuple[Path, Path, int | None, str]]]:
        """Shared manifest/quarantine engine for document and confirmed full-library deletion."""
        operation_id = uuid4().hex
        quarantine_dir = self._quarantine_root / f"op-{operation_id}"
        files_dir = quarantine_dir / "files"
        movable = [entry for entry in files if entry.exists]
        manifest_files = []
        for index, entry in enumerate(movable):
            require_material_path(entry.path, self._protected_paths())
            manifest_files.append(
                {
                    "original_path": str(entry.path),
                    "quarantine_path": str(files_dir / f"{index:04d}-{entry.path.name}"),
                    "size_bytes": entry.size_bytes,
                    "sha256": sha256_file(entry.path),
                }
            )
        manifest_path = quarantine_dir / MANIFEST_NAME
        if expected_hashes is not None:
            for entry in manifest_files:
                if expected_hashes.get(Path(entry["original_path"])) != entry["sha256"]:
                    raise DocumentDeletionError("文件在预览后变化，已中止清理，请重新预览。")
        manifest = {
            "version": MANIFEST_VERSION,
            "operation_kind": operation_kind,
            "operation_id": operation_id,
            "document_id": document_id,
            "document_title": title,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "app_version": self._app_version,
            # Diagnostic only; recovery decisions never read this field.
            "phase": "prepared",
            "files": manifest_files,
        }
        created = False
        try:
            files_dir.mkdir(parents=True, exist_ok=False)
            created = True
            write_json_atomic(manifest_path, manifest)
        except OSError as exc:
            if created:
                try:
                    shutil.rmtree(quarantine_dir)
                except OSError:
                    LOGGER.warning("中止后隔离目录未能移除，已保留：%s", quarantine_dir)
            raise DocumentDeletionError(
                f"写入删除操作 manifest 失败，已中止删除，未改动任何数据：{exc}"
            ) from exc

        moved: list[tuple[Path, Path, int | None, str]] = []
        try:
            for entry in manifest_files:
                original = Path(entry["original_path"])
                destination = Path(entry["quarantine_path"])
                move_verified_file(original, destination, entry["sha256"])
                moved.append((original, destination, entry["size_bytes"], entry["sha256"]))
                if sha256_file(destination) != entry["sha256"]:
                    raise OSError("隔离后文件校验不一致，停止删除并保留现场。")
        except (OSError, ValueError) as exc:
            if destination.exists() and not any(path == destination for _, path, _, _ in moved):
                moved.append((original, destination, entry["size_bytes"], entry["sha256"]))
            self._abort_with_restore(
                moved,
                quarantine_dir,
                f"移动文件到隔离目录失败，已中止删除：{entry['original_path']}（{exc}）",
            )
        self._update_manifest_phase(manifest_path, manifest, "quarantined")

        return quarantine_dir, manifest_path, manifest, moved

    def current_document_ids(self) -> tuple[int, ...]:
        """Return the current platform scope without exposing a database to the bulk layer."""
        return tuple(document.id for document in self._database.list_documents())

    def library_cleanup_plan(self) -> LibraryCleanupPlan:
        """Preview all platform material files, explicitly including unregistered residue."""
        if self._cache_dir is None or self._backups_dir is None:
            raise DocumentDeletionError("请从已配置缓存和独立备份目录的平台文档管理入口清空资料。")
        self._check_storage()
        protected = self._protected_paths()
        roots = (self._data_dir, self._raw_dir, self._pages_dir, self._markdown_dir,
                 self._agent_readings_dir, self._cache_dir)
        return capture_cleanup_plan(roots, protected)

    def _protected_paths(self) -> tuple[Path, ...]:
        """Use the same exclusions for each document and the final directory sweep."""
        return (
            self._database.database_path.parent, self._quarantine_root,
            self._data_dir / ".residue-cleanup-snapshots",
            self._data_dir / "ui_preferences.json",
            *(() if self._backups_dir is None else (self._backups_dir,)),
        )

    def validate_library_plan(self, plan: LibraryCleanupPlan) -> None:
        """Preflight the entire displayed inventory before deleting the first document."""
        current = self.library_cleanup_plan()
        if current != plan:
            raise DocumentDeletionError("全库文件或备份保留范围已变化，请重新预览；尚未删除资料。")

    def clear_empty_library(
        self, plan: LibraryCleanupPlan, *, confirmed: bool
    ) -> LibraryCleanupResult:
        """Reuse the normal manifest/quarantine engine, then verify zero material residue."""
        if confirmed is not True:
            raise DocumentDeletionError("请先核对全库文件清单并确认清空残留。")
        current = self.library_cleanup_plan()
        if current.roots != plan.roots or current.protected != plan.protected:
            raise DocumentDeletionError("平台目录配置已变化，请重新生成清理预览。")
        claim_storage_paths("platform", plan.roots, inspect_tree=True)
        self._check_storage(claim=True)
        with self._database._connection() as connection:
            self._require_empty_library(connection)
        if self._quarantine_root.exists() and any(self._quarantine_root.iterdir()):
            raise DocumentDeletionError(
                "还有未完成的删除隔离操作，请先在备份与修复中处理，再重新预览。"
            )
        remaining = remaining_approved_files(plan)
        files = tuple(
            DocumentDeletionFile(item.path, "library_residue", True, item.size_bytes)
            for item in remaining
        )
        quarantine, manifest_path, manifest, moved = self._stage_files(
            files, 0, "清空全部平台资料残留", operation_kind="library_cleanup",
            expected_hashes={item.path: item.sha256 for item in remaining},
        )
        try:
            with self._database._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._require_empty_library(connection)
                connection.execute("DELETE FROM long_read_task")
                connection.execute("DELETE FROM import_queue")
                connection.execute("DELETE FROM import_records")
                connection.execute("DELETE FROM page_visual_index")
                residue = self._library_residue(connection)
                if residue:
                    raise DocumentDeletionError("全库记录尚有残留，保留文件：" + "；".join(residue))
        except Exception as exc:
            self._abort_with_restore(moved, quarantine, f"全库收尾未完成，正在保留文件：{exc}")
        self._update_manifest_phase(manifest_path, manifest, "db_committed")
        problems: list[str] = []
        try:
            purge_verified_operation(quarantine)
            remove_empty_directories(plan)
        except (OSError, ValueError) as exc:
            problems.append(f"隔离文件或空目录清理未完成：{exc}")
        final = self.library_cleanup_plan()
        problems.extend(str(item.path) for item in final.files)
        if self._quarantine_root.exists() and any(self._quarantine_root.iterdir()):
            problems.append("删除隔离区仍有残留")
        with self._database._connection() as connection:
            self._require_empty_library(connection)
            problems.extend(self._library_residue(connection))
        return LibraryCleanupResult(not problems, tuple(problems))

    @staticmethod
    def _library_residue(connection) -> tuple[str, ...]:
        """Check page-owned records before purging files and again before success."""
        problems = []
        for table in (
            "pages", "page_search", "notes", "evidence_items", "page_embeddings",
            "long_read_task", "long_read_page", "import_queue", "import_records",
            "page_visual_index", "page_visual_interpretations", "document_tags",
            "page_tags", "project_documents", "project_pages",
        ):
            if connection.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone():
                problems.append(f"资料、任务或派生记录仍有残留：{table}")
        if connection.execute("PRAGMA foreign_key_check").fetchone():
            problems.append("数据库存在外键异常，未宣称清理完成")
        return tuple(problems)

    @staticmethod
    def _require_empty_library(connection) -> None:
        if connection.execute("SELECT 1 FROM documents LIMIT 1").fetchone():
            raise DocumentDeletionError("资料库尚有文档，不能清空全库残留。")
        if connection.execute(
            "SELECT 1 FROM import_queue WHERE status IN ('queued','importing') LIMIT 1"
        ).fetchone() or connection.execute(
            "SELECT 1 FROM long_read_task WHERE status IN ('queued','running') LIMIT 1"
        ).fetchone():
            raise DocumentDeletionError("后台任务尚未停止，不能清空全库残留。")

    def _update_manifest_phase(
        self, manifest_path: Path, manifest: dict, phase: str
    ) -> None:
        """Best-effort diagnostic phase update; never fails the deletion.

        The phase field is diagnostic-only — crash recovery decides purely
        from the database and the filesystem — so a failed update is logged
        and otherwise ignored. The atomic rewrite guarantees the manifest
        on disk is always either the old or the new complete JSON.
        """

        try:
            write_json_atomic(manifest_path, {**manifest, "phase": phase})
        except OSError as exc:
            LOGGER.warning("更新删除操作 manifest 阶段标记失败：%s（%s）", manifest_path, exc)

    # ------------------------------------------------------- database stage
    def _check_storage(self, *, claim: bool = False) -> None:
        """Keep platform deletion and quarantine outside independent study storage."""
        paths = (self._data_dir, self._raw_dir, self._pages_dir, self._markdown_dir,
                 self._agent_readings_dir, self._quarantine_root)
        try:
            require_material_path(
                self._quarantine_root,
                tuple(path for path in self._protected_paths() if path != self._quarantine_root),
            )
            if claim:
                claim_storage_paths("platform", paths, inspect_tree=not self._storage_claimed)
                self._storage_claimed = True
            else:
                check_storage_paths("platform", paths)
        except (OSError, StorageBoundaryError) as exc:
            raise DocumentDeletionError(str(exc)) from exc

    @staticmethod
    def _require_idle(connection, document_id: int) -> None:
        """Refuse deletion during scheduled/running imports or this document's reading."""
        importing = connection.execute(
            "SELECT 1 FROM import_queue WHERE status IN ('queued', 'importing') LIMIT 1"
        ).fetchone()
        reading = connection.execute(
            "SELECT 1 FROM long_read_task WHERE document_id=? "
            "AND status IN ('queued','running') LIMIT 1", (document_id,),
        ).fetchone()
        if importing or reading:
            raise DocumentDeletionError(
                "仍有待执行/正在执行的导入或本资料读取任务，请先在任务界面停止并等待退出，"
                "再重新预览删除。"
            )

    def _delete_document_records(self, document_id: int) -> None:
        """Run the cascading delete and verify zero residue in one transaction.

        Executes one ``DELETE`` relying on the verified schema cascades and
        the FTS cleanup trigger, then checks every affected table inside the
        same connection so any surprise rolls the whole transaction back
        before commit.
        """

        with self._database._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_idle(connection, document_id)
            page_ids = [
                int(row[0])
                for row in connection.execute(
                    "SELECT id FROM pages WHERE document_id = ?", (document_id,)
                ).fetchall()
            ]
            placeholders = ",".join("?" for _ in page_ids) or "NULL"
            # v0.8.8: capture the source-derived knowledge objects whose
            # sources all live on this document BEFORE the links are removed,
            # so losing the last source can cascade instead of leaving
            # searchable zero-source orphans behind (round-1 defect P1).
            cascade_orphan_candidates = _select_cascade_orphan_candidates(
                connection, document_id, page_ids
            )
            # v0.5.2: knowledge-object source links are polymorphic and have
            # no declared SQLite foreign key. Remove them explicitly inside
            # the same transaction so a document deletion can never leave
            # dangling KO sources behind.
            connection.execute(
                "DELETE FROM knowledge_object_sources "
                "WHERE source_type = 'document' AND source_id = ?",
                (document_id,),
            )
            if page_ids:
                connection.execute(
                    "DELETE FROM knowledge_object_sources "
                    f"WHERE source_type = 'page' AND source_id IN ({placeholders})",
                    page_ids,
                )
                connection.execute(
                    "DELETE FROM knowledge_object_sources WHERE source_type = 'note' "
                    "AND source_id IN ("
                    "SELECT id FROM notes WHERE document_id = ? "
                    f"OR page_id IN ({placeholders}))",
                    (document_id, *page_ids),
                )
                connection.execute(
                    "DELETE FROM knowledge_object_sources "
                    "WHERE source_type = 'evidence' AND source_id IN ("
                    "SELECT id FROM evidence_items WHERE document_id = ? "
                    f"OR page_id IN ({placeholders}))",
                    (document_id, *page_ids),
                )
            else:
                connection.execute(
                    "DELETE FROM knowledge_object_sources WHERE source_type = 'note' "
                    "AND source_id IN ("
                    "SELECT id FROM notes WHERE document_id = ?)",
                    (document_id,),
                )
                connection.execute(
                    "DELETE FROM knowledge_object_sources "
                    "WHERE source_type = 'evidence' AND source_id IN ("
                    "SELECT id FROM evidence_items WHERE document_id = ?)",
                    (document_id,),
                )
            # v0.8.8 (round-1 P1): knowledge objects that just lost their
            # LAST source go with the document — relations cascade by foreign
            # key, the knowledge_object_search FTS rows by the schema-v13
            # delete trigger. Objects still referenced as a supersession
            # successor (ON DELETE RESTRICT) are kept and reported instead of
            # failing the whole deletion; the residue cleanup on the system
            # maintenance page surfaces them for manual resolution.
            orphan_ids: list[int] = []
            if cascade_orphan_candidates:
                candidate_placeholders = ",".join(
                    "?" for _ in cascade_orphan_candidates
                )
                orphan_ids = [
                    int(row[0])
                    for row in connection.execute(
                        f"""
                        SELECT id FROM knowledge_objects
                        WHERE id IN ({candidate_placeholders})
                          AND NOT EXISTS (
                              SELECT 1 FROM knowledge_object_sources s
                              WHERE s.knowledge_object_id = knowledge_objects.id
                          )
                          AND NOT EXISTS (
                              SELECT 1 FROM knowledge_objects successor
                              WHERE successor.superseded_by_ko_id
                                    = knowledge_objects.id
                          )
                        """,
                        cascade_orphan_candidates,
                    ).fetchall()
                ]
                protected = [
                    int(row[0])
                    for row in connection.execute(
                        f"""
                        SELECT id FROM knowledge_objects
                        WHERE id IN ({candidate_placeholders})
                          AND NOT EXISTS (
                              SELECT 1 FROM knowledge_object_sources s
                              WHERE s.knowledge_object_id = knowledge_objects.id
                          )
                          AND EXISTS (
                              SELECT 1 FROM knowledge_objects successor
                              WHERE successor.superseded_by_ko_id
                                    = knowledge_objects.id
                          )
                        """,
                        cascade_orphan_candidates,
                    ).fetchall()
                ]
                if protected:
                    LOGGER.warning(
                        "知识对象失去全部来源但仍被替代关系引用，本次保留待人工处置："
                        "document_id=%s ids=%s",
                        document_id,
                        protected,
                    )
            if orphan_ids:
                orphan_placeholders = ",".join("?" for _ in orphan_ids)
                connection.execute(
                    f"DELETE FROM knowledge_project_links "
                    f"WHERE target_type = 'knowledge_object' "
                    f"AND target_id IN ({orphan_placeholders})",
                    orphan_ids,
                )
                connection.execute(
                    f"DELETE FROM knowledge_objects WHERE id IN ({orphan_placeholders})",
                    orphan_ids,
                )
            # v0.8.8 (round-1 P2): page_visual_index is a plain FTS5 table
            # with no foreign keys — remove this document's rows explicitly,
            # otherwise deleted pages keep haunting visual retrieval.
            if page_ids:
                connection.execute(
                    f"DELETE FROM page_visual_index WHERE page_id IN ({placeholders})",
                    page_ids,
                )
            # v0.8.6: learning assets survive detached. Organized questions
            # keep their typed content (stem/answer/verdict/corrections).
            # Schema v29 evidence bindings point to question-group materials
            # and/or pages; their page references are detached by FK
            # ``ON DELETE SET NULL`` while the binding and confirmation state
            # survive. This replaces the pre-v29 ``question_id`` /
            # ``evidence_item_id`` / ``unavailable`` contract.
            affected_question_ids = [
                int(row[0])
                for row in connection.execute(
                    f"""
                    SELECT id FROM question_items
                    WHERE document_id = ? OR page_id IN ({placeholders})
                    """,
                    (document_id, *page_ids),
                ).fetchall()
            ]
            question_placeholders = (
                ",".join("?" for _ in affected_question_ids) or "NULL"
            )
            family_member_baseline = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*) FROM question_family_members
                    WHERE question_id IN ({question_placeholders})
                    """,
                    affected_question_ids,
                ).fetchone()[0]
            ) if affected_question_ids else 0
            mastery_baseline = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*) FROM mastery_records
                    WHERE question_id IN ({question_placeholders})
                    """,
                    affected_question_ids,
                ).fetchone()[0]
            ) if affected_question_ids else 0
            # V086-305/208: refresh the source snapshot right before the
            # delete, inside this same transaction. The asset survives the
            # document detached; the snapshot keeps "which paper, which
            # page" answerable afterwards. Only a minimal citation is stored
            # — never a copy of the deleted content — so the user's intent
            # to delete the original file stays intact.
            connection.execute(
                f"""
                UPDATE question_items SET
                    source_document_title_snapshot = COALESCE(
                        (SELECT title FROM documents
                         WHERE documents.id = question_items.document_id),
                        source_document_title_snapshot
                    ),
                    source_page_label_snapshot = COALESCE(
                        '第 ' || (SELECT page_number FROM pages
                                  WHERE pages.id = question_items.page_id) || ' 页',
                        source_page_label_snapshot
                    ),
                    updated_at = ?
                WHERE id IN ({question_placeholders})
                """,
                (self._utc_now(), *affected_question_ids),
            )
            connection.execute("DELETE FROM long_read_task WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM import_queue WHERE document_id = ?", (document_id,))
            cursor = connection.execute(
                "DELETE FROM documents WHERE id = ?", (document_id,)
            )
            if cursor.rowcount == 0:
                raise DocumentDeletionError(f"找不到文档：{document_id}")
            residue_checks = (
                ("页面记录", "SELECT COUNT(*) FROM pages WHERE document_id = ?", (document_id,)),
                (
                    "文档级笔记",
                    "SELECT COUNT(*) FROM notes WHERE document_id = ?",
                    (document_id,),
                ),
                (
                    "页面笔记",
                    f"SELECT COUNT(*) FROM notes WHERE page_id IN ({placeholders})",
                    page_ids,
                ),
                (
                    "证据项",
                    f"SELECT COUNT(*) FROM evidence_items "
                    f"WHERE document_id = ? OR page_id IN ({placeholders})",
                    (document_id, *page_ids),
                ),
                (
                    "搜索记录",
                    f"SELECT COUNT(*) FROM page_search WHERE rowid IN ({placeholders})",
                    page_ids,
                ),
                (
                    "文档标签关联",
                    "SELECT COUNT(*) FROM document_tags WHERE document_id = ?",
                    (document_id,),
                ),
                (
                    "页面标签关联",
                    f"SELECT COUNT(*) FROM page_tags WHERE page_id IN ({placeholders})",
                    page_ids,
                ),
                (
                    "文档项目关联",
                    "SELECT COUNT(*) FROM project_documents WHERE document_id = ?",
                    (document_id,),
                ),
                (
                    "页面项目关联",
                    f"SELECT COUNT(*) FROM project_pages WHERE page_id IN ({placeholders})",
                    page_ids,
                ),
                (
                    "页面嵌入向量",
                    f"SELECT COUNT(*) FROM page_embeddings "
                    f"WHERE page_id IN ({placeholders})",
                    page_ids,
                ),
                (
                    "知识对象来源残留",
                    "SELECT COUNT(*) FROM knowledge_object_sources "
                    f"WHERE (source_type = 'document' AND source_id = ?) "
                    f"OR (source_type = 'page' AND source_id IN ({placeholders}))",
                    (document_id, *page_ids),
                ),
                (
                    "页面视觉索引残留",
                    f"SELECT COUNT(*) FROM page_visual_index "
                    f"WHERE page_id IN ({placeholders})",
                    page_ids,
                ),
                # Import records survive with document_id set to NULL
                # (ON DELETE SET NULL); none may keep referencing the document.
                (
                    "导入记录引用",
                    "SELECT COUNT(*) FROM import_records WHERE document_id = ?",
                    (document_id,),
                ),
            )
            if orphan_ids:
                orphan_placeholders = ",".join("?" for _ in orphan_ids)
                residue_checks += (
                    (
                        "知识对象孤儿残留",
                        f"SELECT COUNT(*) FROM knowledge_objects "
                        f"WHERE id IN ({orphan_placeholders})",
                        orphan_ids,
                    ),
                    (
                        "知识关系残留",
                        f"SELECT COUNT(*) FROM knowledge_relations "
                        f"WHERE source_ko_id IN ({orphan_placeholders}) "
                        f"OR target_ko_id IN ({orphan_placeholders})",
                        (*orphan_ids, *orphan_ids),
                    ),
                    (
                        "知识检索索引残留",
                        f"SELECT COUNT(*) FROM knowledge_object_search "
                        f"WHERE rowid IN ({orphan_placeholders})",
                        orphan_ids,
                    ),
                    (
                        "题目知识关联残留",
                        f"SELECT COUNT(*) FROM question_knowledge_links "
                        f"WHERE knowledge_object_id IN ({orphan_placeholders})",
                        orphan_ids,
                    ),
                )
            if affected_question_ids:
                # Surviving learning assets must be detached, not deleted:
                # questions keep existing without document/page binding;
                # evidence relationships and confirmation state survive while
                # references to deleted pages are cleared by foreign keys;
                # family memberships plus mastery records stay intact.
                residue_checks += (
                    (
                        "学习题目脱离原资料",
                        f"""
                        SELECT COUNT(*) FROM question_items
                        WHERE id IN ({question_placeholders})
                          AND (document_id IS NOT NULL OR page_id IS NOT NULL)
                        """,
                        affected_question_ids,
                    ),
                    (
                        "学习题目证据仍引用已删页面",
                        f"""
                        SELECT COUNT(*) FROM question_item_evidence
                        WHERE question_item_id IN ({question_placeholders})
                          AND page_id IN ({placeholders})
                        """,
                        (*affected_question_ids, *page_ids),
                    ),
                    (
                        "学习题目成员关系丢失",
                        f"""
                        SELECT {family_member_baseline} - COUNT(*)
                        FROM question_family_members
                        WHERE question_id IN ({question_placeholders})
                        """,
                        affected_question_ids,
                    ),
                    (
                        "学习题目掌握记录丢失",
                        f"""
                        SELECT {mastery_baseline} - COUNT(*)
                        FROM mastery_records
                        WHERE question_id IN ({question_placeholders})
                        """,
                        affected_question_ids,
                    ),
                )
            failures = [
                label
                for label, query, parameters in residue_checks
                if int(connection.execute(query, parameters).fetchone()[0]) != 0
            ]
            violations = connection.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                failures.append(f"外键一致性检查（{len(violations)} 条违规）")
            if failures:
                raise DocumentDeletionError(
                    "删除后校验发现残留数据：" + "、".join(failures)
                )

    # ----------------------------------------------------------- file stage
    def _collect_files(
        self,
        *,
        document_id: int,
        source_path: Path,
        page_ids: list[int],
        image_paths: list[Path],
        markdown_paths: list[Path],
    ) -> tuple[tuple[DocumentDeletionFile, ...], tuple[Path, ...], tuple[str, ...], int]:
        """Validate and measure every recorded file exactly once.

        Paths come only from database records and are validated against
        their owning root. Duplicated records (same resolved path) collapse
        to one entry so nothing is measured or moved twice.
        """

        candidates = [
            (Path(source_path), "pdf", self._raw_dir),
            *(
                (Path(path), "page_image", self._pages_dir / str(document_id))
                for path in image_paths
            ),
            *(
                (Path(path), "markdown", self._markdown_dir / str(document_id))
                for path in markdown_paths
            ),
        ]
        # Agent readings are derived files keyed by database ids. They are not
        # database-recorded source paths, so missing files are normal; existing
        # ones must still join the same quarantine/rollback transaction to avoid
        # leaving stale read state after a document is deleted and re-uploaded.
        optional_reading_candidates = [
            (
                self._agent_readings_dir
                / "documents"
                / f"document_{document_id}.json",
                "agent_document_reading",
                self._agent_readings_dir / "documents",
            ),
            *(
                (
                    self._agent_readings_dir / "pages" / f"page_{page_id}.json",
                    "agent_page_reading",
                    self._agent_readings_dir / "pages",
                )
                for page_id in page_ids
            ),
        ]
        candidates.extend(
            candidate
            for candidate in optional_reading_candidates
            if candidate[0].exists() or candidate[0].is_symlink()
        )
        entries: list[DocumentDeletionFile] = []
        missing: list[Path] = []
        anomalies: list[str] = []
        total_size = 0
        seen: set[Path] = set()
        for path, kind, root in candidates:
            anomaly, resolved = self._validate_recorded_path(path, root)
            if anomaly is not None:
                anomalies.append(anomaly)
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            if path.is_file():
                size = path.stat().st_size
                total_size += size
                entries.append(
                    DocumentDeletionFile(path=path, kind=kind, exists=True, size_bytes=size)
                )
            else:
                missing.append(path)
                entries.append(
                    DocumentDeletionFile(path=path, kind=kind, exists=False, size_bytes=None)
                )
        return tuple(entries), tuple(missing), tuple(anomalies), total_size

    def _validate_recorded_path(
        self, path: Path, root: Path
    ) -> tuple[str | None, Path]:
        """Validate one database-recorded path against its owning root.

        Returns ``(anomaly_message, resolved_path)``; the message is ``None``
        only when the path stays inside ``root`` after resolution, is not a
        data root itself, contains no ``..`` segment, is not a symlink, and
        is not a directory masquerading as a file. No user input is ever
        used to build these paths; deletion aborts on any anomaly instead
        of following the path.
        """

        try:
            resolved = local_directory(path)
            check_storage_paths("platform", (resolved,))
            require_material_path(resolved, self._protected_paths())
        except (OSError, ValueError) as exc:
            return f"文件隔离检查失败：{exc}", path
        managed_roots = (
            self._raw_dir.resolve(),
            self._pages_dir.resolve(),
            self._markdown_dir.resolve(),
            self._data_dir.resolve(),
        )
        if ".." in path.parts:
            return f"路径包含“..”段：{path}", resolved
        if resolved in managed_roots:
            return f"路径指向数据根目录本身：{path}", resolved
        root_resolved = root.resolve()
        if not resolved.is_relative_to(root_resolved):
            return f"路径不在其归属目录（{root_resolved}）内：{path}", resolved
        if path.is_symlink():
            return f"路径是符号链接：{path}", resolved
        if path.exists() and not path.is_file():
            return f"路径不是普通文件：{path}", resolved
        return None, resolved

    def _restore_moved_files(
        self, moved: list[tuple[Path, Path, int | None, str]]
    ) -> list[str]:
        """Move every quarantined file back and verify presence and size."""

        failures: list[str] = []
        for original, quarantined, size_bytes, digest in reversed(moved):
            if os.path.lexists(original):
                failures.append(f"{original}（原位置已有文件，未覆盖；隔离副本保留待人工恢复）")
                continue
            try:
                move_verified_file(quarantined, original, digest)
            except (OSError, ValueError) as exc:
                failures.append(f"{original}（移回失败：{exc}）")
                continue
            if not original.is_file():
                failures.append(f"{original}（恢复后文件不存在）")
            elif size_bytes is not None and original.stat().st_size != size_bytes:
                failures.append(f"{original}（恢复后大小不一致）")
        return failures

    def _abort_with_restore(
        self,
        moved: list[tuple[Path, Path, int | None, str]],
        quarantine_dir: Path,
        reason: str,
    ) -> None:
        """Restore quarantined files after an abort and raise honestly.

        Restoration failures are logged as critical and spelled out in the
        raised message together with the quarantine location; the database
        is already rolled back at this point and the message says so.
        """

        restore_failures = self._restore_moved_files(moved)
        if restore_failures:
            LOGGER.critical(
                "删除中止且部分文件未能恢复：quarantine=%s failures=%s",
                quarantine_dir,
                restore_failures,
            )
            raise DocumentDeletionError(
                f"{reason}。数据库未改动，但以下文件未能恢复原位："
                + "；".join(restore_failures)
                + f"。这些文件保留在隔离目录：{quarantine_dir}"
            )
        try:
            purge_verified_operation(quarantine_dir)
        except (OSError, ValueError):
            LOGGER.warning("中止后隔离目录未能移除，已保留：%s", quarantine_dir)
        raise DocumentDeletionError(f"{reason}。数据库未改动，文件已全部恢复原位。")


# ------------------------------------------------------- residue cleanup
@dataclass(frozen=True, slots=True)
class OrphanKnowledgeObject:
    """One source-derived knowledge object with zero live sources."""

    knowledge_object_id: int
    title: str
    kind: str
    confirmation_status: str
    source_count: int
    live_source_count: int
    inbound_supersession_count: int

    @property
    def deletable(self) -> bool:
        """Refuse automatic deletion while a supersession still points here."""

        return self.inbound_supersession_count == 0


@dataclass(frozen=True, slots=True)
class ResidueCleanupPreview:
    """Read-only summary of historical deletion residue in the database."""

    orphan_knowledge_objects: tuple[OrphanKnowledgeObject, ...]
    knowledge_relation_count: int
    personal_zero_source_count: int
    page_visual_index_orphan_count: int
    stale_figure_index_entry_count: int
    figures_index_path: Path | None = None

    @property
    def deletable_knowledge_object_ids(self) -> tuple[int, ...]:
        """Ids the confirmed run would delete (supersession targets excluded)."""

        return tuple(
            item.knowledge_object_id
            for item in self.orphan_knowledge_objects
            if item.deletable
        )

    @property
    def protected_knowledge_objects(self) -> tuple[OrphanKnowledgeObject, ...]:
        """Zero-source objects a supersession protects from automatic removal."""

        return tuple(
            item for item in self.orphan_knowledge_objects if not item.deletable
        )


@dataclass(frozen=True, slots=True)
class ResidueCleanupResult:
    """Outcome of one confirmed residue cleanup run."""

    deleted_knowledge_object_count: int
    deleted_knowledge_relation_count: int
    deleted_visual_index_row_count: int
    pruned_figure_index_entry_count: int
    protected_knowledge_object_ids: tuple[int, ...] = ()
    snapshot_path: Path | None = None
    warnings: tuple[str, ...] = ()


class ResidueCleanupError(DocumentDeletionError):
    """A residue cleanup was refused, stale, or only partially verifiable."""


_LIVE_SOURCE_CONDITION = """
    (s.source_type = 'document'
        AND EXISTS (SELECT 1 FROM documents t WHERE t.id = s.source_id))
 OR (s.source_type = 'page'
        AND EXISTS (SELECT 1 FROM pages t WHERE t.id = s.source_id))
 OR (s.source_type = 'note'
        AND EXISTS (SELECT 1 FROM notes t WHERE t.id = s.source_id))
 OR (s.source_type = 'evidence'
        AND EXISTS (SELECT 1 FROM evidence_items t WHERE t.id = s.source_id))
"""


def _figure_entry_page_id(entry: dict) -> int:
    """Best-effort page id of one global figure-index entry."""

    try:
        return int(entry.get("page_id", 0))
    except (TypeError, ValueError):
        return 0


class ResidueCleanupService:
    """Clean historical deletion residue that predates the fixed cascade.

    Round-1 context: documents were already deleted on the live instance
    while the old cascade left zero-source knowledge objects, their
    relations, their FTS rows and ``page_visual_index`` rows behind. This
    service removes exactly that residue through the platform — no manual
    SQL, no manual file deletion — with a read-only preview, an explicit
    confirmation gate, a JSON snapshot of every removed row, and post-run
    residue verification inside the same transaction.

    Scope is deliberately narrow and idempotent:

    - knowledge objects with ``epistemic_basis='source_derived'`` whose
      live source count is zero (no source rows, or every row dangling at
      a deleted target). Objects resting on a personal basis keep their
      meaning without sources and are only counted, never removed;
    - objects still referenced as a supersession successor are reported
      for manual resolution instead of deleted (ON DELETE RESTRICT);
    - ``page_visual_index`` FTS rows whose page no longer exists;
    - global figure-index entries whose page no longer exists.

    Original PDFs, page images, notes, evidence, questions, families and
    mastery records are never touched. A second run finds nothing and is
    a verified no-op.
    """

    def __init__(
        self,
        *,
        database: Database,
        data_dir: Path,
        question_candidates_dir: Path | None = None,
    ) -> None:
        self._database = database
        self._data_dir = Path(data_dir)
        self._candidates_dir = Path(
            question_candidates_dir
            if question_candidates_dir is not None
            else self._data_dir / "question-candidates"
        )

    # ---------------------------------------------------------- preview
    def preview_cleanup(self) -> ResidueCleanupPreview:
        """Read-only residue inventory; never writes anything."""

        bases_placeholders = ",".join("?" for _ in _ORPHAN_CASCADE_EPISTEMIC_BASES)
        with self._database._connection() as connection:
            orphan_rows = connection.execute(
                f"""
                SELECT ko.id, ko.title, ko.kind, ko.confirmation_status,
                    (SELECT COUNT(*) FROM knowledge_object_sources s
                      WHERE s.knowledge_object_id = ko.id) AS source_count,
                    (SELECT COUNT(*) FROM knowledge_object_sources s
                      WHERE s.knowledge_object_id = ko.id
                        AND ({_LIVE_SOURCE_CONDITION})) AS live_source_count,
                    (SELECT COUNT(*) FROM knowledge_objects succ
                      WHERE succ.superseded_by_ko_id = ko.id
                    ) AS inbound_supersession_count
                FROM knowledge_objects ko
                WHERE ko.epistemic_basis IN ({bases_placeholders})
                ORDER BY ko.id
                """,
                _ORPHAN_CASCADE_EPISTEMIC_BASES,
            ).fetchall()
            orphans = tuple(
                OrphanKnowledgeObject(
                    knowledge_object_id=int(row["id"]),
                    title=str(row["title"]),
                    kind=str(row["kind"]),
                    confirmation_status=str(row["confirmation_status"]),
                    source_count=int(row["source_count"]),
                    live_source_count=int(row["live_source_count"]),
                    inbound_supersession_count=int(row["inbound_supersession_count"]),
                )
                for row in orphan_rows
                if int(row["live_source_count"]) == 0
            )
            deletable_ids = [
                item.knowledge_object_id for item in orphans if item.deletable
            ]
            relation_count = 0
            if deletable_ids:
                placeholders = ",".join("?" for _ in deletable_ids)
                relation_count = int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM knowledge_relations "
                        f"WHERE source_ko_id IN ({placeholders}) "
                        f"OR target_ko_id IN ({placeholders})",
                        (*deletable_ids, *deletable_ids),
                    ).fetchone()[0]
                )
            personal_zero_source_count = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*) FROM knowledge_objects ko
                    WHERE ko.epistemic_basis NOT IN ({bases_placeholders})
                      AND NOT EXISTS (
                          SELECT 1 FROM knowledge_object_sources s
                          WHERE s.knowledge_object_id = ko.id
                            AND ({_LIVE_SOURCE_CONDITION})
                      )
                    """,
                    _ORPHAN_CASCADE_EPISTEMIC_BASES,
                ).fetchone()[0]
            )
            visual_orphan_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM page_visual_index "
                    "WHERE page_id NOT IN (SELECT id FROM pages)"
                ).fetchone()[0]
            )
            valid_page_ids = {
                int(row[0])
                for row in connection.execute("SELECT id FROM pages").fetchall()
            }

        stale_figure_entries = 0
        figures_index_path: Path | None = None
        try:
            from src.figure_index_service import index_path, read_index

            candidate_index_path = index_path(self._candidates_dir)
            if candidate_index_path.is_file():
                figures_index_path = candidate_index_path
                stale_figure_entries = sum(
                    1
                    for entries in read_index(self._candidates_dir).values()
                    for entry in entries
                    if _figure_entry_page_id(entry) not in valid_page_ids
                )
        except (OSError, ValueError) as exc:
            LOGGER.warning("全局图形索引不可读，本次清理将跳过该部分：%s", exc)

        return ResidueCleanupPreview(
            orphan_knowledge_objects=orphans,
            knowledge_relation_count=relation_count,
            personal_zero_source_count=personal_zero_source_count,
            page_visual_index_orphan_count=visual_orphan_count,
            stale_figure_index_entry_count=stale_figure_entries,
            figures_index_path=figures_index_path,
        )

    # ---------------------------------------------------------- execute
    def execute_cleanup(
        self,
        *,
        confirmed: bool,
        expected_knowledge_object_ids: Sequence[int],
        expected_visual_index_row_count: int,
    ) -> ResidueCleanupResult:
        """Remove the previewed residue after an explicit confirmation.

        ``confirmed`` must be ``True`` and both expectations must still
        match a freshly recomputed preview, so a stale preview can never
        authorize a different deletion. Every removed database row is
        snapshotted to JSON first; the database delete runs in one
        transaction with residue verification, and the figure-index prune
        happens only after that commit.
        """

        if not confirmed:
            raise ResidueCleanupError(
                "未确认清理：残留清理是不可逆操作，必须先预览并显式确认。"
            )
        preview = self.preview_cleanup()
        expected_ids = {int(item) for item in expected_knowledge_object_ids}
        actual_ids = set(preview.deletable_knowledge_object_ids)
        if expected_ids != actual_ids:
            raise ResidueCleanupError(
                "预览已过期：待清理知识对象集合发生变化，请重新预览后再确认。"
                f"（预览 {len(expected_ids)} 个，当前 {len(actual_ids)} 个）"
            )
        if int(expected_visual_index_row_count) != preview.page_visual_index_orphan_count:
            raise ResidueCleanupError(
                "预览已过期：待清理视觉索引行数发生变化，请重新预览后再确认。"
            )
        if (
            not actual_ids
            and not preview.page_visual_index_orphan_count
            and not preview.stale_figure_index_entry_count
        ):
            return ResidueCleanupResult(
                deleted_knowledge_object_count=0,
                deleted_knowledge_relation_count=0,
                deleted_visual_index_row_count=0,
                pruned_figure_index_entry_count=0,
                protected_knowledge_object_ids=tuple(
                    item.knowledge_object_id
                    for item in preview.protected_knowledge_objects
                ),
            )

        snapshot_path = self._write_snapshot(preview)
        deletable = sorted(actual_ids)
        placeholders = ",".join("?" for _ in deletable) if deletable else "NULL"
        with self._database._connection() as connection:
            relation_count = 0
            if deletable:
                connection.execute(
                    f"DELETE FROM knowledge_project_links "
                    f"WHERE target_type = 'knowledge_object' "
                    f"AND target_id IN ({placeholders})",
                    deletable,
                )
                relation_count = int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM knowledge_relations "
                        f"WHERE source_ko_id IN ({placeholders}) "
                        f"OR target_ko_id IN ({placeholders})",
                        (*deletable, *deletable),
                    ).fetchone()[0]
                )
                connection.execute(
                    f"DELETE FROM knowledge_objects WHERE id IN ({placeholders})",
                    deletable,
                )
            visual_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM page_visual_index "
                    "WHERE page_id NOT IN (SELECT id FROM pages)"
                ).fetchone()[0]
            )
            connection.execute(
                "DELETE FROM page_visual_index "
                "WHERE page_id NOT IN (SELECT id FROM pages)"
            )
            failures: list[str] = []
            if deletable:
                for label, query, parameters in (
                    (
                        "知识对象残留",
                        f"SELECT COUNT(*) FROM knowledge_objects "
                        f"WHERE id IN ({placeholders})",
                        deletable,
                    ),
                    (
                        "知识关系残留",
                        f"SELECT COUNT(*) FROM knowledge_relations "
                        f"WHERE source_ko_id IN ({placeholders}) "
                        f"OR target_ko_id IN ({placeholders})",
                        (*deletable, *deletable),
                    ),
                    (
                        "知识检索索引残留",
                        f"SELECT COUNT(*) FROM knowledge_object_search "
                        f"WHERE rowid IN ({placeholders})",
                        deletable,
                    ),
                ):
                    if int(connection.execute(query, parameters).fetchone()[0]) != 0:
                        failures.append(label)
            if (
                int(
                    connection.execute(
                        "SELECT COUNT(*) FROM page_visual_index "
                        "WHERE page_id NOT IN (SELECT id FROM pages)"
                    ).fetchone()[0]
                )
                != 0
            ):
                failures.append("页面视觉索引残留")
            violations = connection.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                failures.append(f"外键一致性检查（{len(violations)} 条违规）")
            if failures:
                raise ResidueCleanupError(
                    "清理后校验发现残留数据，已回滚：" + "、".join(failures)
                )

        pruned = 0
        warnings: list[str] = []
        if preview.stale_figure_index_entry_count:
            try:
                from src.figure_index_service import prune_missing_pages

                with self._database._connection() as connection:
                    valid_page_ids = {
                        int(row[0])
                        for row in connection.execute(
                            "SELECT id FROM pages"
                        ).fetchall()
                    }
                pruned = prune_missing_pages(self._candidates_dir, valid_page_ids)
            except (OSError, ValueError) as exc:
                warning = f"数据库残留已清理，但全局图形索引清理失败：{exc}"
                LOGGER.warning(warning)
                warnings.append(warning)
        LOGGER.info(
            "残留清理完成：knowledge_objects=%s relations=%s visual_index=%s "
            "figure_index=%s snapshot=%s",
            len(deletable),
            relation_count,
            visual_count,
            pruned,
            snapshot_path,
        )
        return ResidueCleanupResult(
            deleted_knowledge_object_count=len(deletable),
            deleted_knowledge_relation_count=relation_count,
            deleted_visual_index_row_count=visual_count,
            pruned_figure_index_entry_count=pruned,
            protected_knowledge_object_ids=tuple(
                item.knowledge_object_id
                for item in preview.protected_knowledge_objects
            ),
            snapshot_path=snapshot_path,
            warnings=tuple(warnings),
        )

    # --------------------------------------------------------- snapshot
    def _write_snapshot(self, preview: ResidueCleanupPreview) -> Path | None:
        """Snapshot every row the confirmed run will remove (audit trail).

        The snapshot lives under ``data/.residue-cleanup-snapshots/`` and is
        never removed automatically. When it cannot be written the cleanup
        refuses to run — no irreversible delete without its audit record.
        """

        deletable = sorted(preview.deletable_knowledge_object_ids)
        if (
            not deletable
            and not preview.page_visual_index_orphan_count
            and not preview.stale_figure_index_entry_count
        ):
            return None
        snapshot_dir = (
            self._data_dir
            / ".residue-cleanup-snapshots"
            / f"cleanup-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
        )
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "reason": "历史删除残留清理（零来源溯源型知识对象与失效索引行）",
            "knowledge_objects": [],
            "knowledge_object_sources": [],
            "knowledge_relations": [],
            "page_visual_index_orphan_count": preview.page_visual_index_orphan_count,
            "stale_figure_index_entry_count": preview.stale_figure_index_entry_count,
        }
        if deletable:
            placeholders = ",".join("?" for _ in deletable)
            queries = (
                ("knowledge_objects",
                 f"SELECT * FROM knowledge_objects WHERE id IN ({placeholders})",
                 deletable),
                ("knowledge_object_sources",
                 f"SELECT * FROM knowledge_object_sources "
                 f"WHERE knowledge_object_id IN ({placeholders})",
                 deletable),
                ("knowledge_relations",
                 f"SELECT * FROM knowledge_relations "
                 f"WHERE source_ko_id IN ({placeholders}) "
                 f"OR target_ko_id IN ({placeholders})",
                 (*deletable, *deletable)),
            )
            with self._database._connection() as connection:
                for key, query, parameters in queries:
                    rows = connection.execute(query, parameters).fetchall()
                    payload[key] = [dict(row) for row in rows]
        try:
            write_json_atomic(snapshot_dir / "snapshot.json", payload)
            if preview.figures_index_path is not None:
                shutil.copy2(
                    preview.figures_index_path,
                    snapshot_dir / "figures-index.before.json",
                )
        except OSError as exc:
            raise ResidueCleanupError(
                f"无法写入清理快照，已取消清理（未改动任何数据）：{exc}"
            ) from exc
        return snapshot_dir / "snapshot.json"
