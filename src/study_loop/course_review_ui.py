"""A course workspace for saved review documents and its separate live conversation."""

from __future__ import annotations

from collections import Counter

import streamlit as st

from src.time_display import format_beijing_time

from .chat_citations import visible_answer
from .chat_store import ChatStore
from .course_reviews import CourseReviewIndex
from .display import render_saved_text
from .models import StudyError
from .storage import StudyStore


def _labels(entries: list[dict]) -> dict[str, str]:
    """Keep same-file saved reviews distinguishable without exposing internal identifiers."""
    labels = ["、".join(entry["filenames"]) for entry in entries]
    duplicates = Counter(labels)
    return {
        entry["document_id"]: label + (f"（复盘{index + 1}）" if duplicates[label] > 1 else "")
        for index, (entry, label) in enumerate(zip(entries, labels, strict=True))
    }


def _render_memory_catalog(store: StudyStore, subject: str) -> None:
    from .knowledge_memory_ui import render_memory_catalog

    render_memory_catalog(store, subject)


def _render_details(entry: dict) -> None:
    st.text("来源：" + "、".join(entry["filenames"]))
    st.text("题目：" + " · ".join(dict.fromkeys(
        [name.rsplit(".", 1)[0] for name in entry["filenames"]]
        + [question["number"] for question in entry["questions"] if question["number"]]
    )))
    if entry["chapter"]:
        st.caption("章节：" + entry["chapter"])
    st.caption(f"第{entry['version_number']}版 · 已保存 · "
               + format_beijing_time(entry["updated_at"]))


def _render_comparison(
    index: CourseReviewIndex, subject: str, entries: list[dict], key: str
) -> None:
    with st.expander("复盘对照"):
        if len(entries) < 2:
            st.caption("保存两份复盘后可进行对照。")
            return
        labels = _labels(entries)
        selection_key = key + "compare"
        previous = st.session_state.get(selection_key, [])
        if any(identity not in labels for identity in previous):
            st.session_state[selection_key] = [
                identity for identity in previous if identity in labels
            ]
        selected = st.multiselect(
            "选择复盘资料", list(labels), format_func=labels.get, max_selections=10,
            key=selection_key,
        )
        if len(selected) < 2:
            st.caption("选择 2 至 10 份已保存复盘进行对照。")
            return
        comparison = index.compare(subject, selected)
        for start in range(0, len(comparison), 2):
            pair = comparison[start:start + 2]
            for column, entry in zip(st.columns(len(pair)), pair, strict=True):
                with column:
                    with st.container(border=True):
                        _render_details(entry)
                        with st.container(height=450):
                            render_saved_text(visible_answer(entry["content"]))


def _render_reviews(store: StudyStore, subject: str, identity: str) -> None:
    from .ai_review_ui import open_ai_review

    index = CourseReviewIndex(store)
    key = "sl_course_review_" + identity + "_"
    query = st.text_input("搜索复盘资料", key=key + "query", placeholder="题名、正文或来源文件名")
    entries = index.entries(subject, query)
    if entries:
        labels = _labels(entries)
        if st.session_state.get(key + "selected") not in labels:
            st.session_state.pop(key + "selected", None)
        document_id = st.selectbox(
            "已保存复盘", list(labels), format_func=labels.get, key=key + "selected",
        )
        entry = index.entry(document_id, subject)
        _render_details(entry)
        with st.container(height=480):
            render_saved_text(visible_answer(entry["content"]))
        st.button(
            "打开原复盘", key=key + "open", on_click=open_ai_review,
            args=(entry["item_ids"][0], entry["document_id"], entry["version_id"]),
        )
        from .ai_review import AIReviewService
        from .knowledge_import_ui import render_review_knowledge_import

        reviews = AIReviewService(store)
        render_review_knowledge_import(
            store, reviews.document(entry["document_id"]), reviews.version(entry["version_id"]),
            location="course_" + identity,
        )
    elif query:
        st.info("没有找到匹配的复盘资料。")
    else:
        st.info("暂无已保存的复盘资料。保存单题复盘后，即可在这里浏览和对照。")
    _render_comparison(index, subject, index.entries(subject), key)
    with st.expander("已确认知识记忆"):
        _render_memory_catalog(store, subject)


def render_course_workspace(store: StudyStore, identity: str) -> None:
    """Enter chat by default, keeping course materials in a separate locally browsable view."""
    from .course_chat_ui import (
        VIEW_KEY,
        leave_course_chat,
        render_course_chat,
        set_course_view,
    )

    conversation = ChatStore(store).conversation(identity)
    if conversation["kind"] != "course":
        raise StudyError("请选择课程自由复盘会话。")
    if st.session_state.get(VIEW_KEY, "chat") == "chat":
        render_course_chat(store, identity)
        return
    subject = conversation["subject"]
    heading, back = st.columns([4, 1])
    with heading:
        st.subheader("课程复盘资料")
        st.caption(subject)
    back.button("返回学习整理", key="sl_course_workspace_back",
                on_click=leave_course_chat)
    st.button("返回课程对话", key="sl_course_back_chat", on_click=set_course_view, args=("chat",))
    _render_reviews(store, subject, identity)
