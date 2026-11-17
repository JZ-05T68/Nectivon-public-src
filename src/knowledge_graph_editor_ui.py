"""Shared service-backed content and relation editing for both graph views."""

from __future__ import annotations

import logging

import streamlit as st

from src.database import Database
from src.knowledge_object_service import KnowledgeObjectService
from src.models import KnowledgeRelationType, NoteImportance

LOGGER = logging.getLogger(__name__)


def render_selected_knowledge_editor(database: Database, selected: str | None) -> None:
    """Require explicit saves and a second action before deleting a relationship."""
    if not selected or not selected.startswith("knowledge:"):
        st.caption("选择一个知识点后，可在这里修改正文、重要程度与已有关系。")
        return
    service = KnowledgeObjectService(database)
    identifier = int(selected.split(":", 1)[1])
    current = service.get(identifier)
    if current is None:
        st.info("所选知识点已不存在，请刷新视图。")
        return
    st.markdown(f"**编辑：{current.title}**")
    prefix = f"graph_edit_{identifier}_{current.current_revision}"
    with st.form(prefix):
        title = st.text_input("知识点标题", value=current.title, max_chars=200,
                              key=f"{prefix}_title")
        content = st.text_area("知识正文", value=current.content, height=240,
                               key=f"{prefix}_content")
        importance = st.selectbox("重要程度", list(NoteImportance),
                                  index=list(NoteImportance).index(current.importance),
                                  format_func=lambda item: item.label, key=f"{prefix}_importance")
        confirm = st.checkbox("我已核对，保存时确认当前内容", key=f"{prefix}_confirm")
        st.caption("保存会保留修订历史；改动后的内容需要重新确认。拖动画布不会修改正文或关系。")
        if st.form_submit_button("保存知识修改", type="primary"):
            try:
                service.update_content(identifier, title=title, content=content,
                                       importance=importance)
                if confirm:
                    service.confirm(identifier)
            except Exception as exc:  # noqa: BLE001 - keep unsaved user input on failure
                LOGGER.exception("保存知识修改失败")
                st.error(f"保存知识修改失败：{exc}")
            else:
                st.session_state["graph_relation_notice"] = ("✅", "知识修改已保存，两种视图共用。")
                st.rerun()
    relations = service.relations(identifier)
    if not relations:
        st.caption("当前没有已保存的知识关系，可在“新增与关联”中添加。")
        return

    def label(relation_id: int) -> str:
        relation = next(item for item in relations if item.id == relation_id)
        source, target = service.get(relation.source_ko_id), service.get(relation.target_ko_id)
        return f"{source.title} → {relation.relation_type.label} → {target.title}"

    relation_id = st.selectbox("修改已有关系", [item.id for item in relations],
                               format_func=label, key=f"graph_edit_relation_{identifier}")
    relation = next(item for item in relations if item.id == relation_id)
    with st.form(f"graph_edit_relation_form_{relation_id}_{relation.relation_type.value}"):
        relation_type = st.selectbox("关系类型", list(KnowledgeRelationType),
                                     index=list(KnowledgeRelationType).index(relation.relation_type),
                                     format_func=lambda item: item.label)
        description = st.text_input("关系依据", value=relation.description, max_chars=1000)
        if st.form_submit_button("保存关系修改"):
            try:
                service.update_relation(relation_id, relation_type=relation_type,
                                        description=description)
            except Exception as exc:  # noqa: BLE001 - transaction preserves old relation
                LOGGER.exception("保存关系修改失败")
                st.error(f"保存关系修改失败：{exc}")
            else:
                st.rerun()
    pending_key = f"graph_edit_remove_{identifier}"
    if st.button("删除所选关系", key=f"graph_edit_remove_request_{identifier}"):
        st.session_state[pending_key] = relation_id
    pending = st.session_state.get(pending_key)
    if pending is not None:
        # Freeze the intended relation ID, even if the selection changes before confirmation.
        pending_relation = next((item for item in relations if item.id == pending), None)
        if pending_relation is None:
            st.session_state.pop(pending_key, None)
            return
        st.warning(f"确认删除：{label(pending)}？知识正文与来源会保留。")
        yes, no = st.columns(2)
        if yes.button("确认删除关系", key=f"graph_edit_remove_confirm_{identifier}"):
            try:
                service.remove_relation(pending)
            except Exception as exc:  # noqa: BLE001 - preserve pending state for retry
                LOGGER.exception("删除知识关系失败")
                st.error(f"删除知识关系失败：{exc}")
            else:
                st.session_state.pop(pending_key, None)
                st.rerun()
        if no.button("保留关系", key=f"graph_edit_remove_cancel_{identifier}"):
            st.session_state.pop(pending_key, None)
            st.rerun()
