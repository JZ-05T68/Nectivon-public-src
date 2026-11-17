"""Shared Streamlit UI for permanent imported-document deletion.

Single rendering path for the document lifecycle deletion flow: read-only
impact preview (per-note-type counts, evidence items, files), then the
multi-step confirmation (checkbox + exact document title + an independent
evidence-basket confirmation whenever the document still has evidence
items). The execute button stays disabled until every required
confirmation is satisfied and no path anomaly exists. All execution goes
through :class:`DocumentDeletionService` — the UI never deletes anything
itself.
"""

from __future__ import annotations

import logging
from uuid import uuid4

import streamlit as st

from src.document_deletion_service import DocumentDeletionService, preview_signature
from src.library_cleanup import LibraryCleanupPlan
from src.models import Document, DocumentAggregationImpact, DocumentDeletionPreview

LOGGER = logging.getLogger(__name__)


def render_library_cleanup_plan(plan: LibraryCleanupPlan) -> None:
    """Show the exact platform file scope and protected backups before confirmation."""
    st.warning(
        f"全库清空包含 {len(plan.files)} 个已登记或未登记的资料、派生及暂存文件，"
        f"共 {sum(item.size_bytes for item in plan.files) / (1024 * 1024):.1f} MB。"
    )
    with st.expander("查看清空范围与文件清单"):
        st.write("受管资料/缓存目录：", [str(path) for path in plan.roots])
        st.write("保留数据库、独立备份/恢复快照及配置：", [str(path) for path in plan.protected])
        st.caption("删除隔离区仅作事务保护，最终也必须无残留；未完成的旧隔离操作会阻止清空。")
        st.dataframe([
            {"路径": str(item.path), "字节数": item.size_bytes, "SHA-256": item.sha256}
            for item in plan.files
        ], hide_index=True, use_container_width=True)


def render_empty_library_cleanup(deletion_service: DocumentDeletionService) -> None:
    """Allow a new confirmed cleanup even when earlier versions left an empty document list."""
    st.subheader("清空资料、派生与暂存残留")
    st.caption("保留旧题目、独立历史备份和恢复快照；不处理平台之外自行保存的原文件。")
    previous = st.session_state.get("empty_library_cleanup_result")
    if previous is not None:
        if previous.complete:
            st.success("上次清理已核对：授权范围内无资料、派生、暂存及删除隔离残留。")
        else:
            st.warning("上次磁盘清理未完成：" + "；".join(previous.remaining))
    if st.button("生成磁盘清理预览", key="empty_library_preview"):
        st.session_state.pop("empty_library_plan", None)
        try:
            plan = deletion_service.library_cleanup_plan()
        except Exception as exc:
            st.error(f"清理预览未完成：{exc}")
        else:
            st.session_state["empty_library_plan"] = (uuid4().hex, plan)
    stored = st.session_state.get("empty_library_plan")
    if stored is None:
        return
    identity, plan = stored
    render_library_cleanup_plan(plan)
    confirmed = st.checkbox(
        "确认清空上述资料、派生与暂存文件，保留列出的备份和其他保留区域",
        key=f"empty_library_confirm_{identity}",
    )
    if st.button("执行已预览的磁盘清理", disabled=not confirmed, key="empty_library_execute"):
        try:
            result = deletion_service.clear_empty_library(plan, confirmed=confirmed)
        except Exception as exc:
            st.session_state.pop("empty_library_plan", None)
            st.error(f"磁盘清理未完成：{exc}。请处理原因后重新预览。")
        else:
            st.session_state["empty_library_cleanup_result"] = result
            st.session_state.pop("empty_library_plan", None)
            st.rerun()


def _format_file_size(size_bytes: int) -> str:
    """Format a byte count with human-readable binary units."""

    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"


def _render_learning_asset_impact(preview: DocumentDeletionPreview) -> None:
    """学习整理资产影响（v0.8.6）：整理成果属于用户，不属于原始 PDF。

    原资料删除后，已形成的整理题、题型/方法族与掌握记录按现有策略保留；
    学习证据关系与确认状态保留，指向被删页面的位置引用会安全解除。
    """

    st.markdown("**学习整理资产影响**")
    if (
        preview.learning_question_count == 0
        and preview.learning_family_count == 0
        and preview.learning_mastery_record_count == 0
    ):
        st.caption("此文档暂无学习整理资产（整理题/题型族/掌握记录）。")
        return
    st.caption(
        f"将保留的学习成果：整理题 {preview.learning_question_count} 道 · "
        f"涉及题型/方法族 {preview.learning_family_count} 个 · "
        f"掌握练习记录 {preview.learning_mastery_record_count} 条"
    )
    if preview.learning_evidence_link_count:
        st.caption(
            f"涉及 {preview.learning_evidence_link_count} 条学习证据关系：关系和确认状态会保留，"
            "其中指向本资料页面的位置引用会解除，不会留下失效的数据库引用。"
        )


def _render_aggregation_impact(impact: DocumentAggregationImpact) -> None:
    """Transparency block: which aggregation views this deletion changes.

    Display only — the existing confirmation chain (checkbox + exact title +
    evidence confirmation) already covers the risk, so no extra checkbox is
    added here.
    """

    st.markdown("**知识聚合影响**")
    if not impact.projects and not impact.tags:
        st.caption("此文档当前未出现在任何项目或标签知识聚合视图中。")
        return
    if impact.projects:
        lines = "\n".join(f"- {project.name}" for project in impact.projects)
        st.caption(f"项目聚合：{len(impact.projects)} 个")
        st.markdown(lines)
    if impact.tags:
        lines = "\n".join(f"- {tag.name}" for tag in impact.tags)
        st.caption(f"标签聚合：{len(impact.tags)} 个")
        st.markdown(lines)


def render_document_deletion_section(
    *,
    deletion_service: DocumentDeletionService,
    document: Document,
) -> None:
    """Render the impact preview and multi-step permanent-deletion flow."""

    st.markdown("**永久删除导入文档及关联数据**")
    try:
        deletion_preview = deletion_service.preview_document_deletion(document.id)
    except Exception as exc:
        LOGGER.exception("生成删除预览失败：document_id=%s", document.id)
        st.error(f"无法生成删除预览：{exc}")
        return
    st.warning(
        f"将把文档“{document.title}”及其全部页面、笔记、证据和派生工作文件"
        "移入删除隔离区（与批量删除同一套安全流程）。隔离区内容可在「备份与修复」"
        "页检查；一旦隔离区被清理或自动完成清理，数据无法再恢复。"
        "项目、标签与证据篮本身保留；历史备份中可能仍保留旧版本，"
        "不受本次删除影响。"
    )
    preview_metrics = st.columns(4)
    preview_metrics[0].metric("页面", deletion_preview.page_count)
    preview_metrics[1].metric("结构化笔记", deletion_preview.note_count)
    preview_metrics[2].metric("证据项", deletion_preview.evidence_item_count)
    preview_metrics[3].metric("搜索记录", deletion_preview.search_record_count)
    st.caption(
        f"笔记明细：文档级 {deletion_preview.document_note_count} 条 · "
        f"页面级 {deletion_preview.page_note_count} 条 · "
        f"文字选区 {deletion_preview.text_selection_note_count} 条 · "
        f"图片区域 {deletion_preview.image_region_note_count} 条　|　"
        f"标签与项目关联 {deletion_preview.association_count} 条　|　"
        f"导入记录 {deletion_preview.import_record_count} 条（保留，仅解除关联）"
    )
    st.caption(
        f"独占文件：PDF {deletion_preview.pdf_file_count} 个 · "
        f"页面图片 {deletion_preview.page_image_count} 个 · "
        f"Markdown {deletion_preview.markdown_file_count} 个，"
        f"共 {_format_file_size(deletion_preview.total_size_bytes)}"
    )
    if deletion_preview.knowledge_object_orphan_count:
        st.caption(
            f"将同时删除失去全部来源的溯源型知识点 "
            f"{deletion_preview.knowledge_object_orphan_count} 个"
            "（连同其知识关系与检索索引）；基于个人经验/判断的知识点"
            "与整理题、掌握记录一样保留，仅断开出处。"
        )
    if deletion_preview.page_visual_index_count:
        st.caption(
            f"将清理页面视觉检索索引 {deletion_preview.page_visual_index_count} 条，"
            "避免已删页面继续出现在视觉检索结果中。"
        )
    _render_learning_asset_impact(deletion_preview)
    _render_aggregation_impact(deletion_preview.aggregation_impact)
    if deletion_preview.missing_files:
        st.warning(
            "以下登记文件在磁盘上缺失，删除时将跳过：\n"
            + "\n".join(f"- {path}" for path in deletion_preview.missing_files)
        )
    if deletion_preview.path_anomalies:
        st.error(
            "检测到路径异常，已禁止删除：\n"
            + "\n".join(f"- {item}" for item in deletion_preview.path_anomalies)
        )
    confirmation_key = f"{document.id}_{preview_signature(deletion_preview)}"
    delete_confirmed = st.checkbox(
        "我确认永久删除此导入文档及其全部页面、笔记和派生数据。",
        key=f"doc_delete_confirm_{confirmation_key}",
    )
    delete_title = st.text_input(
        f"请输入文档标题“{document.title}”以确认删除",
        key=f"doc_delete_title_{document.id}",
    )
    # Evidence items may carry user excerpts, snapshots and annotations, so
    # they earn their own explicit confirmation instead of hiding behind the
    # generic warning above.
    evidence_confirmed = True
    if deletion_preview.evidence_item_count > 0:
        evidence_confirmed = st.checkbox(
            f"此文档还有 {deletion_preview.evidence_item_count} 条证据篮条目。"
            "继续删除将永久删除这些摘录、快照和用户批注。",
            key=f"doc_delete_evidence_{confirmation_key}",
        )
    if st.button(
        "永久删除此导入文档及关联数据",
        disabled=(
            not delete_confirmed
            or delete_title != document.title
            or not evidence_confirmed
            or bool(deletion_preview.path_anomalies)
        ),
        key=f"doc_delete_execute_{document.id}",
    ):
        try:
            deletion_result = deletion_service.delete_document(
                document.id, expected_title=delete_title, expected_preview=deletion_preview,
            )
        except Exception as exc:
            LOGGER.exception("删除导入文档失败：document_id=%s", document.id)
            st.error(f"删除失败：{exc}")
        else:
            st.session_state["doc_delete_reset_pending"] = True
            st.session_state["doc_delete_flash"] = (
                f"文档“{document.title}”及其 "
                f"{deletion_result.preview.page_count} 个页面与全部派生数据"
                "已从当前知识库删除。历史备份中可能仍保留旧版本。",
                deletion_result.cleanup_warnings,
            )
            st.query_params.clear()
            st.rerun()
