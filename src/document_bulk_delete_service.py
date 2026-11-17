"""Bulk document deletion as a thin orchestration layer (overnight §16).

Product rule: there is NO second deletion engine.  Every document goes
through the existing staged :class:`src.document_deletion_service.DocumentDeletionService`
(preview → validation → quarantine manifest → transaction → residue check →
recovery) exactly once per document.  This module only:

1. aggregates the per-document previews into one batch summary;
2. executes deletions one document at a time, isolating failures so the
   37th document failing can never corrupt the state of the other 36;
3. reports per-document outcomes so failed items can be retried.

Learning assets (题目/族/二级结论/掌握记录) are NOT deleted: the single
deletion service detaches their document/page provenance while preserving
evidence relationships and confirmation state — the preview says so explicitly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from uuid import uuid4

from src.document_deletion_service import (
    DocumentDeletionError,
    DocumentDeletionService,
    preview_signature,
)
from src.library_cleanup import LibraryCleanupPlan
from src.models import DocumentDeletionPreview

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BulkDeleteOutcome:
    """Per-document result of one bulk deletion run."""

    document_id: int
    document_title: str
    deleted: bool
    message: str = ""
    cleanup_warnings: tuple[str, ...] = ()
    library_cleanup_complete: bool | None = None


@dataclass(slots=True)
class BulkDeletePreview:
    """Aggregated read-only impact of deleting several documents."""

    previews: list[DocumentDeletionPreview] = field(default_factory=list)
    errors: list[tuple[int, str]] = field(default_factory=list)
    requested_ids: tuple[int, ...] = ()
    confirmation_id: str = field(default_factory=lambda: uuid4().hex)
    cleanup_plan: LibraryCleanupPlan | None = None

    @property
    def evidence_item_count(self) -> int:
        return sum(preview.evidence_item_count for preview in self.previews)

    @property
    def note_count(self) -> int:
        return sum(preview.note_count for preview in self.previews)

    @property
    def document_count(self) -> int:
        return len(self.previews)

    @property
    def page_count(self) -> int:
        return sum(preview.page_count for preview in self.previews)

    @property
    def total_size_bytes(self) -> int:
        return sum(preview.total_size_bytes for preview in self.previews)

    @property
    def surviving_learning_questions(self) -> int:
        """Learning question assets that stay (detached) after deletion."""

        return sum(preview.learning_question_count for preview in self.previews)

    @property
    def surviving_learning_families(self) -> int:
        return sum(preview.learning_family_count for preview in self.previews)

    def missing_or_anomalous(self) -> list[tuple[int, list[str]]]:
        return [
            (preview.document_id, list(preview.path_anomalies))
            for preview in self.previews
            if preview.path_anomalies
        ]


class DocumentBulkDeleteService:
    """One-by-one orchestration over the staged single-document deleter."""

    def __init__(self, deletion_service: DocumentDeletionService) -> None:
        self._deletion_service = deletion_service

    def preview(self, document_ids: list[int]) -> BulkDeletePreview:
        """Aggregate per-document previews; missing documents are errors."""

        if any(type(identifier) is not int or identifier <= 0 for identifier in document_ids):
            raise DocumentDeletionError("请选择有效的资料编号。")
        identifiers = tuple(dict.fromkeys(document_ids))
        aggregated = BulkDeletePreview(requested_ids=identifiers)
        for document_id in identifiers:
            try:
                aggregated.previews.append(
                    self._deletion_service.preview_document_deletion(document_id)
                )
            except Exception as exc:  # noqa: BLE001 - report every failed preview
                aggregated.errors.append((document_id, str(exc)))
        if identifiers and set(identifiers) == set(self._deletion_service.current_document_ids()):
            try:
                aggregated.cleanup_plan = self._deletion_service.library_cleanup_plan()
            except (OSError, ValueError, DocumentDeletionError) as exc:
                aggregated.errors.append((0, f"全库文件清单未完成：{exc}"))
        return aggregated

    def execute(
        self, preview: BulkDeletePreview, *, confirmed: bool = False,
        evidence_confirmed: bool = False,
    ) -> list[BulkDeleteOutcome]:
        """Execute only the displayed batch; retries require another preview/confirmation."""
        if not isinstance(preview, BulkDeletePreview) or confirmed is not True:
            raise DocumentDeletionError("请先生成批量删除预览并明确确认。")
        if (
            not preview.previews or preview.errors or preview.missing_or_anomalous()
            or tuple(p.document_id for p in preview.previews) != preview.requested_ids
        ):
            raise DocumentDeletionError("批量预览不完整或存在异常，请重新选择并生成预览。")
        if preview.evidence_item_count and evidence_confirmed is not True:
            raise DocumentDeletionError("请单独确认删除预览中的证据摘录、快照和用户批注。")
        if preview.cleanup_plan and set(preview.requested_ids) != set(
            self._deletion_service.current_document_ids()
        ):
            raise DocumentDeletionError("全库资料清单已变化，请重新预览；本次尚未删除资料。")
        if preview.cleanup_plan is not None:
            self._deletion_service.validate_library_plan(preview.cleanup_plan)
        approved_hashes = (
            {item.path: item.sha256 for item in preview.cleanup_plan.files}
            if preview.cleanup_plan is not None else None
        )

        # Check the whole batch before deleting its first document. Each single
        # deletion checks again, including changes caused by earlier batch items.
        for expected in preview.previews:
            current = self._deletion_service.preview_document_deletion(expected.document_id)
            if preview_signature(current) != preview_signature(expected):
                raise DocumentDeletionError("批量预览已过期，请重新预览并确认；本次尚未删除资料。")

        outcomes: list[BulkDeleteOutcome] = []
        for expected in preview.previews:
            document_id = expected.document_id
            try:
                result = self._deletion_service.delete_document(
                    document_id, expected_title=expected.document_title, expected_preview=expected,
                    expected_file_hashes=approved_hashes,
                )
            except DocumentDeletionError as exc:
                LOGGER.warning("批量删除单项失败：document_id=%s error=%s", document_id, exc)
                outcomes.append(
                    BulkDeleteOutcome(
                        document_id=document_id,
                        document_title=expected.document_title,
                        deleted=False,
                        message=str(exc),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - isolation is the contract
                LOGGER.exception("批量删除单项异常：document_id=%s", document_id)
                outcomes.append(
                    BulkDeleteOutcome(
                        document_id=document_id,
                        document_title=expected.document_title,
                        deleted=False,
                        message=f"未预期错误：{exc}",
                    )
                )
            else:
                outcomes.append(
                    BulkDeleteOutcome(
                        document_id=document_id,
                        document_title=expected.document_title,
                        deleted=result.deleted,
                        cleanup_warnings=result.cleanup_warnings,
                    )
                )
        if preview.cleanup_plan and outcomes:
            complete = False
            issues = ("本批仍有未删除资料，未清理全库残留。",)
            if all(outcome.deleted for outcome in outcomes):
                try:
                    cleanup = self._deletion_service.clear_empty_library(
                        preview.cleanup_plan, confirmed=True,
                    )
                    complete, issues = cleanup.complete, cleanup.remaining
                except Exception as exc:  # noqa: BLE001 - records and disk have separate outcomes
                    issues = (f"文档删除已完成，但磁盘收尾未完成：{exc}",)
            if complete:
                outcomes = [replace(outcome, cleanup_warnings=tuple(
                    warning for warning in outcome.cleanup_warnings
                    if not warning.startswith(("仅处理预览", "删除已完成，但目录非空"))
                )) for outcome in outcomes]
            outcomes[-1] = replace(
                outcomes[-1], library_cleanup_complete=complete,
                cleanup_warnings=(*outcomes[-1].cleanup_warnings, *issues),
            )
        return outcomes

    @staticmethod
    def failed_ids(outcomes: list[BulkDeleteOutcome]) -> list[int]:
        return [outcome.document_id for outcome in outcomes if not outcome.deleted]

    @staticmethod
    def summarize(outcomes: list[BulkDeleteOutcome]) -> str:
        succeeded = sum(1 for outcome in outcomes if outcome.deleted)
        failed = len(outcomes) - succeeded
        warnings = sum(bool(outcome.cleanup_warnings) for outcome in outcomes)
        result = f"已删除 {succeeded} 份，未删除 {failed} 份；{warnings} 份含文件保留或清理说明。"
        states = [
            o.library_cleanup_complete for o in outcomes if o.library_cleanup_complete is not None
        ]
        if states:
            result += (
                " 资料、派生和暂存文件已核对无残留；保留区域不在清空范围内。"
                if states[-1] else " 磁盘清理尚未完成，请处理剩余项后重新预览。"
            )
        return result
