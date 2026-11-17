"""Learning-workflow knowledge candidates with explicit review and durable source identity."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from src.ai.provider import AIError, CompletionProvider
from src.time_display import format_beijing_time

from .chat_store import ChatStore
from .display import render_saved_text
from .knowledge_memory import KnowledgeMemoryService
from .memory_editor_ui import render_memory_editor
from .models import StudyError
from .source_presentation import SOURCE_NAMES, memory_source_details, memory_title
from .storage import StudyStore

PREFIX = "sl_knowledge_"


def _edit(store: StudyStore, identity: str) -> None:
    service = KnowledgeMemoryService(store)
    try:
        updated = service.update_candidate(
            identity, st.session_state[PREFIX + "body_" + identity],
            expected_revision=st.session_state.get(PREFIX + "revision_" + identity),
        )
        st.session_state[PREFIX + "revision_" + identity] = updated["working_revision"]
    except StudyError as exc:
        st.session_state[PREFIX + "error_" + identity] = str(exc)


def _render_extraction(store: StudyStore, value: dict) -> None:
    service = KnowledgeMemoryService(store)
    identity, source = value["id"], value["source"]
    st.caption(SOURCE_NAMES.get(source["type"], "知识记忆") + " · "
               + format_beijing_time(value["created_at"]))
    if value.get("confirmed_memory_id"):
        st.success("这份候选已确认写入长期知识记忆。")
        render_memory_editor(
            store, value["confirmed_memory_id"], key_prefix="confirmed_" + identity,
        )
        render_memory_sources(store, ChatStore(store).memory(value["confirmed_memory_id"]),
                              key_prefix="confirmed_" + identity)
        return
    state = service.status(identity)
    if not state["active"]:
        st.warning("；".join(state["reasons"]) + "。原候选和已确认记忆保留，请从当前来源重新提炼。")
    if value["status"] != "ready":
        st.info(value.get("error") or "提炼结果尚未完成，记录已保留。")
        return
    body_key, revision_key = PREFIX + "body_" + identity, PREFIX + "revision_" + identity
    if body_key not in st.session_state:
        st.session_state[body_key] = value["content"]
        st.session_state[revision_key] = value["working_revision"]
    st.text_area("编辑记忆候选", key=body_key, height=260, max_chars=30000,
                 on_change=_edit, args=(store, identity))
    if error := st.session_state.get(PREFIX + "error_" + identity):
        st.warning(error)
    with st.expander("预览记忆候选"):
        render_saved_text(st.session_state[body_key])
    checked = st.checkbox("我已核对内容、来源和适用条件，确认写入长期知识记忆",
                          key=PREFIX + "confirm_" + identity)
    if st.button("确认写入长期知识记忆", key=PREFIX + "save_" + identity,
                 disabled=not checked or not state["active"]):
        try:
            service.confirm(identity, st.session_state[body_key], confirmed=checked)
        except StudyError as exc:
            st.warning(str(exc))
        else:
            st.rerun()


def _history(store: StudyStore, values: list[dict], key: str) -> None:
    if not values:
        return
    ids = {value["id"]: value for value in values}
    selected = st.selectbox(
        "记忆候选", list(ids), key=PREFIX + "select_" + key,
        format_func=lambda identity: format_beijing_time(ids[identity]["created_at"])
        + (" · 已确认" if ids[identity].get("confirmed_memory_id") else " · 待审核"),
    )
    _render_extraction(store, ids[selected])


def render_review_memory(
    store: StudyStore, document: dict, version: dict, *,
    provider_factory: Callable[[], CompletionProvider],
) -> None:
    """Extract from the current saved human revision; never silently substitute a raw AI draft."""
    service = KnowledgeMemoryService(store)
    current = version["id"] == document.get("last_saved_version_id")
    st.caption(f"以已保存第{version['number']}版提炼；审核确认后才进入长期知识记忆。")
    if not current:
        st.info("当前选择的是历史版本。请打开最新保存版，或将历史版另存为草稿并保存后提炼。")
    if st.button("保存到记忆", key=PREFIX + "extract_review_" + document["id"],
                 disabled=not current or bool(document.get("deleted_at"))):
        try:
            with st.spinner("正在提炼记忆候选…"):
                value = service.extract_review(document["id"], provider_factory())
            st.session_state[PREFIX + "select_review_" + document["id"]] = value["id"]
        except (StudyError, AIError) as exc:
            st.warning(str(exc))
    values = [value for value in service.extractions(document["conversation_id"])
              if value["source"].get("document_id") == document["id"]]
    _history(store, values, "review_" + document["id"])


def render_course_memory(
    store: StudyStore, identity: str, provider_factory: Callable[[], CompletionProvider],
    *, model_available: bool = True,
) -> None:
    """Use a standalone course dialogue as the other explicit knowledge source."""
    service = KnowledgeMemoryService(store)
    if st.button("保存到记忆", key=PREFIX + "extract_course_" + identity,
                 disabled=not model_available):
        try:
            with st.spinner("正在提炼记忆候选…"):
                value = service.extract_course(identity, provider_factory())
            st.session_state[PREFIX + "select_course_" + identity] = value["id"]
        except (StudyError, AIError) as exc:
            st.warning(str(exc))
    values = [value for value in service.extractions(identity)
              if value["source"].get("type") == "course_dialogue_memory"]
    _history(store, values, "course_" + identity)


def render_memory_catalog(store: StudyStore, subject: str) -> None:
    """Show confirmed knowledge, its source type and retained outdated-version warnings."""
    chats = ChatStore(store)
    values = chats.memories(subject, include_stale=True, include_unavailable=True)
    if not values:
        st.caption("本课程尚无已确认的知识记忆。")
        return
    for memory in values:
        with st.expander(memory_title(memory)):
            render_memory_editor(store, memory["id"], key_prefix="catalog_" + subject)
            render_memory_sources(store, chats.memory(memory["id"]))


def render_memory_sources(
    store: StudyStore, memory: dict, *, key_prefix: str = "sl_knowledge",
) -> None:
    """Show readable provenance and navigate only to a resolvable local source."""
    details = memory_source_details(store, memory)
    for index, detail in enumerate(details):
        st.caption("来源类型：" + detail["type_name"])
        st.caption("来源课程：" + detail["subject"])
        source = memory.get("sources", [])[index] if index < len(memory.get("sources", [])) else {}
        if detail["filename"]:
            st.caption("来源题目：" + detail["filename"])
        if detail["review_version"] is not None:
            st.caption(f"来源版本：复盘第{detail['review_version']}版")
        if detail["dialogue_title"]:
            st.caption("来源会话：" + detail["dialogue_title"])
        if not detail["available"]:
            st.caption(detail["unavailable_reason"])
        if source.get("type") == "review_memory" and source.get("item_ids"):
            from .ai_review_ui import open_ai_review

            st.button("打开来源复盘", key=key_prefix + "_origin_" + memory["id"]
                      + "_" + source["version_id"], disabled=not detail["available"],
                      on_click=open_ai_review,
                      args=(source["item_ids"][0], source["document_id"], source["version_id"]))
        elif source.get("type") == "course_dialogue_memory":
            from .course_chat_ui import open_course_chat

            st.button("打开来源对话", key=key_prefix + "_origin_" + memory["id"]
                      + "_" + source["conversation_id"], disabled=not detail["available"],
                      on_click=open_course_chat, args=(source["conversation_id"],))
