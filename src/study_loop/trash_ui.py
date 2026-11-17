"""Categorized views over existing soft-delete records, with no parallel trash database."""

from __future__ import annotations

import json

import streamlit as st

from src.time_display import format_beijing_time

from .chat_store import ChatStore
from .models import StudyError
from .source_presentation import memory_title
from .storage import StudyStore

NAMES = {
    "memory": "记忆", "conversation": "对话", "archive": "已保存对话",
    "course": "课程设置", "candidate": "记忆候选", "review_document": "AI复盘稿件",
    "source": "学习文件与关联草稿",
}


def _restore(store: StudyStore, chats: ChatStore, kind: str, identity: str, prefix: str) -> None:
    try:
        if kind == "source":
            store.restore_source(identity, user_confirmed=True)
        elif kind == "review_document":
            from .ai_review import AIReviewService

            AIReviewService(store).restore_document(identity)
        else:
            chats.restore(kind, identity)
    except StudyError as exc:
        st.session_state[prefix + "_error"] = str(exc)
    except OSError:
        st.session_state[prefix + "_error"] = "原文件暂不可用，未恢复；请先检查原件。"
    else:
        st.session_state[prefix + "_notice"] = "已恢复所选内容。"


def _rows(store: StudyStore, chats: ChatStore, kind: str, subject: str | None) -> list[dict]:
    if kind == "source":
        return [{"identity": source.id, "label": source.name, "deleted_at": ""}
                for source in store.deleted_sources()]
    if kind == "review_document":
        from .ai_review import AIReviewService

        rows = []
        for document in AIReviewService(store).documents(include_deleted=True):
            if document.get("deleted_at"):
                names = [store.source(identity).name for identity in document["source_ids"]]
                rows.append({"identity": document["id"], "label": "、".join(names),
                             "deleted_at": document["deleted_at"]})
        return rows
    rows = [row for row in chats.trash() if row["kind"] == kind]
    if kind in {"memory", "candidate"}:
        for row in rows:
            # trash() has already shortened its legacy label; that may cut through a span tag.
            # Resolve the complete retained body before making a plain-text title.
            with chats.connect() as db:
                table = "memories" if kind == "memory" else "candidates"
                body = db.execute(f"SELECT content FROM {table} WHERE id=?",
                                  (row["identity"],)).fetchone()
            row["label"] = NAMES[kind] + " · " + (
                memory_title({"content": body[0]}) if body else "内容暂不可用"
            )
    if kind == "conversation":
        with chats.connect() as db:
            payloads = {row["id"]: json.loads(row["payload"]) for row in db.execute(
                "SELECT id,payload FROM conversations"
            )}
        if subject is not None:
            rows = [row for row in rows
                    if payloads.get(row["identity"], {}).get("kind") == "course"
                    and payloads[row["identity"]].get("subject") == subject]
        from .conversation_titles import ConversationTitles

        subjects = {payloads[row["identity"]]["subject"] for row in rows
                    if payloads.get(row["identity"], {}).get("kind") == "course"}
        titles = ConversationTitles(store)
        named = {entry["id"]: entry for course in subjects
                 for entry in titles.entries(course, include_deleted=True)}
        for row in rows:
            if row["identity"] in named:
                entry = named[row["identity"]]
                row["label"] = (
                    (entry["subject"] + " · ") if subject is None else ""
                ) + entry["title"]
    return rows


def render_classified_trash(
    store: StudyStore, chats: ChatStore | None = None, *, key_prefix: str = "sl_trash",
    scope: tuple[str, ...] | None = None, subject: str | None = None,
) -> None:
    """Restore selected records; a course drawer can show only its own course conversations."""
    chats = chats or ChatStore(store)
    allowed = [kind for kind in (scope or tuple(NAMES)) if kind in NAMES]
    if subject is not None:
        allowed = [kind for kind in allowed if kind == "conversation"]
    if not allowed:
        st.caption("此处没有可显示的回收站分类。")
        return
    key = key_prefix + "_category"
    if st.session_state.get(key) not in allowed:
        st.session_state[key] = allowed[0]
    category = st.selectbox("回收站分类", allowed, format_func=NAMES.get, key=key)
    if notice := st.session_state.pop(key_prefix + "_notice", None):
        st.success(notice)
    if error := st.session_state.pop(key_prefix + "_error", None):
        st.warning(error)
    rows = _rows(store, chats, category, subject)
    if not rows:
        st.caption("此分类暂无已删除内容。")
        return
    identities = {row["identity"]: row for row in rows}
    selection = key_prefix + "_selection_" + category
    if st.session_state.get(selection) not in identities:
        st.session_state[selection] = rows[0]["identity"]
    identity = st.selectbox(
        "已删除的内容", list(identities), key=selection,
        format_func=lambda value: identities[value]["label"]
        + (" · " + format_beijing_time(identities[value]["deleted_at"])
           if identities[value]["deleted_at"] else ""),
    )
    if category == "source":
        st.caption("恢复学习文件时，其关联的题目和草稿一同恢复。")
    st.button("恢复所选内容", key=key_prefix + "_restore", on_click=_restore,
              args=(store, chats, category, identity, key_prefix))
