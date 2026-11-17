"""First-level local management of explicitly confirmed knowledge memories."""

from __future__ import annotations

import streamlit as st

from .chat_store import ChatStore
from .knowledge_memory_ui import render_memory_sources
from .memory_editor_ui import render_memory_editor
from .source_presentation import memory_source_details, memory_title
from .storage import StudyStore


def open_knowledge_memory() -> None:
    """Open knowledge management directly, without creating or confirming anything."""
    st.session_state["sl_learning_view"] = "memory"
    st.session_state["sl_ai_review_focus"] = False
    st.session_state.pop("sl_chat_id", None)


def _back(*, flushed: bool = False) -> None:
    from .color_editor import defer_active_color_action

    if not flushed and defer_active_color_action(_back, kwargs={"flushed": True}):
        return
    st.session_state["sl_learning_view"] = "home"


def _matches(store: StudyStore, memory: dict, query: str) -> bool:
    if not query.strip():
        return True
    names = [str(detail[field]) for detail in memory_source_details(store, memory)
             for field in ("filename", "dialogue_title", "type_name")]
    text = "\n".join([memory["subject"], memory["content"], *names])
    return query.strip().casefold() in text.casefold()


def render_knowledge_memory_home(store: StudyStore) -> None:
    """Search current confirmed revisions and manage invalid or recoverably deleted records."""
    from .ai_review_ui import persist_review_route

    st.session_state["sl_review_route_guard"] = persist_review_route
    st.button("返回学习整理", key="sl_memory_home_back", on_click=_back)
    st.subheader("知识记忆")
    chats = ChatStore(store)
    values = chats.memories(
        None, include_stale=True, include_unavailable=True, include_deleted=True,
    )
    subjects = list(dict.fromkeys(value["subject"] for value in values))
    course = st.selectbox("课程筛选", [None, *subjects], key="sl_memory_home_course",
                          format_func=lambda value: "全部课程" if value is None
                          else value or "未指定课程")
    query = st.text_input("搜索记忆", key="sl_memory_home_search",
                          placeholder="输入标题或正文关键词")
    status = st.selectbox("记忆状态", ["全部", "可用", "来源失效", "已删除"],
                          key="sl_memory_home_status")
    filtered = [value for value in values if (course is None or value["subject"] == course)
                and _matches(store, value, query)
                and (status == "全部" or (status == "已删除" and value["deleted"])
                     or (status == "可用" and value["source_status"]["active"])
                     or (status == "来源失效" and not value["deleted"]
                         and not value["source_status"]["active"]))]
    if not filtered:
        st.caption("没有符合条件的已确认知识记忆。")
        return
    by_id = {value["id"]: value for value in filtered}
    key = "sl_memory_home_selection"
    if st.session_state.get(key) not in by_id:
        st.session_state[key] = filtered[0]["id"]
    identity = st.selectbox(
        "知识记忆", list(by_id), key=key,
        format_func=lambda value: memory_title(by_id[value]) + " · "
        + (by_id[value]["subject"] or "未指定课程"),
        index=list(by_id).index(st.session_state[key]),
    )
    render_memory_editor(store, identity, key_prefix="sl_memory_home")
    render_memory_sources(store, chats.memory(identity), key_prefix="sl_memory_home")
