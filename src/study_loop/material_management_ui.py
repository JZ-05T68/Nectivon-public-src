"""Explicit, recoverable removal of learning materials."""

from __future__ import annotations

import json

import streamlit as st

from .models import StudyError
from .storage import StudyStore


def _has_active_tasks(store: StudyStore) -> bool:
    """Avoid racing any in-flight import, export or paid analysis in this library."""
    if not store.path("tasks.sqlite3").exists():
        return False
    from .runtime import queue

    tasks = queue(store.root)
    return any(tasks.list(state=state, limit=1) for state in (
        "queued", "running", "pause_requested", "cancel_requested",
    ))


def render_material_management(store: StudyStore) -> None:
    """Hide removed sources and study entries together; restore all original versions."""
    sources = store.sources()
    if sources:
        source = st.selectbox(
            "要删除的学习文件", sources, format_func=lambda value: value.name,
            key="sl_delete_source_choice",
        )
        confirmed = st.checkbox(
            "删除这份文件及其学习条目（可在最近删除中恢复）",
            key="sl_delete_source_confirm_" + source.id,
        )
        if st.button("删除学习文件", disabled=not confirmed, key="sl_delete_source"):
            if _has_active_tasks(store):
                raise StudyError("请先等待或取消正在处理的任务，再删除材料。")
            store.delete_source(source.id, user_confirmed=confirmed)
            for key in ("sl_draft_choice", "sl_review_selected", "sl_summary_selection"):
                st.session_state.pop(key, None)
            st.rerun()
    deleted = store.deleted_sources()
    if deleted:
        st.markdown("**最近删除**")
        source = st.selectbox(
            "恢复学习文件", deleted, format_func=lambda value: value.name,
            key="sl_restore_source_choice",
        )
        if st.button("恢复文件", key="sl_restore_source"):
            store.restore_source(source.id, user_confirmed=True)
            st.rerun()
    if not sources and not deleted:
        st.info("还没有学习文件。")


def visible_session_items(store: StudyStore, session: dict) -> list:
    """Do not revive removed materials through an old review-session selection."""
    available = {item.id: item for item in store.items()}
    return [available[key] for key in json.loads(session["item_ids"]) if key in available]
