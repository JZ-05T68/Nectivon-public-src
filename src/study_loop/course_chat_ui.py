"""Course-only immersive chat routing and controls, with an AI-free cold import path."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from .storage import StudyStore

LEARNING_PAGE = "pages/18_学习整理.py"
VIEW_KEY = "sl_course_workspace_view"

_STYLE = """
.stApp:has(#sl-course-chat-active) { --sl-course-sidebar:0px; }
.stApp:has(#sl-course-chat-active):has([data-testid="stSidebar"][aria-expanded="true"]) {
  --sl-course-sidebar:300px;
}
.stApp:has(#sl-course-chat-active) [data-testid="stSidebar"][aria-expanded="true"] {
  width:300px !important; min-width:300px !important; max-width:300px !important;
}
.stApp:has(#sl-course-chat-active) [data-testid="stHeader"] {
  height:0; background:transparent; pointer-events:none;
}
.stApp:has(#sl-course-chat-active) [data-testid="stHeader"] button {
  pointer-events:auto;
}
.stApp:has(#sl-course-chat-active) .st-key-workspace_topbar { display:none !important; }
.stApp:has(#sl-course-chat-active) [data-testid="stMainBlockContainer"] {
  max-width:1040px; height:auto !important; min-height:100dvh; flex:0 0 auto !important;
  padding:6rem 2rem 11rem !important;
}
.stApp:has(#sl-course-chat-active) [data-testid="stBottomBlockContainer"] {
  position:fixed !important; bottom:0; left:var(--sl-course-sidebar); right:0;
  transform:none; margin-inline:auto; width:calc(100% - var(--sl-course-sidebar));
  max-width:1040px; padding:.75rem 2rem 1rem; z-index:100;
  background:var(--ekb-gray-50,#f6f8f7);
}
.stApp:has(#sl-course-chat-active) .st-key-sl_course_chat_toolbar {
  position:fixed !important; top:0; left:var(--sl-course-sidebar); right:0;
  transform:none; margin-inline:auto; width:calc(100% - var(--sl-course-sidebar));
  max-width:1040px; z-index:101; padding:.5rem 2rem;
  background:var(--ekb-gray-50,#f6f8f7); border-bottom:1px solid var(--ekb-line,#e1e8e4);
}
.stApp:has(#sl-course-chat-active) .st-key-sl_course_chat_toolbar
[data-testid="stHorizontalBlock"] {
  flex-wrap:nowrap;
}
.stApp:has(#sl-course-chat-active) .st-key-sl_course_chat_toolbar [data-testid="stColumn"] {
  min-width:0 !important;
}
.stApp:has(#sl-course-chat-active) .st-key-sl_course_chat_toolbar h3 {
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-size:1.2rem;
}
.stApp:has(#sl-course-chat-active) .st-key-sl_course_chat_welcome {
  min-height:min(44dvh,360px); justify-content:center; text-align:center;
  color:var(--ekb-muted,#53665d);
}
@media(max-width:768px) {
  .stApp:has(#sl-course-chat-active):has([data-testid="stSidebar"][aria-expanded="true"]) {
    --sl-course-sidebar:0px;
  }
  .stApp:has(#sl-course-chat-active) [data-testid="stMainBlockContainer"] {
    padding:7.5rem 1rem 12rem !important;
  }
  .stApp:has(#sl-course-chat-active) [data-testid="stBottomBlockContainer"] {
    padding-left:1rem; padding-right:1rem;
  }
  .stApp:has(#sl-course-chat-active) .st-key-sl_course_chat_toolbar {
    padding-left:1rem; padding-right:1rem;
  }
}
"""


def is_course_chat_view(current: str) -> bool:
    """Check the exact active route read-only, without initializing any business schema."""
    if (current != LEARNING_PAGE or st.session_state.get("sl_ai_review_focus")
            or st.session_state.get(VIEW_KEY, "chat") != "chat"):
        return False
    identity, root = st.session_state.get("sl_chat_id"), st.session_state.get("sl_open_root")
    if not isinstance(identity, str) or not isinstance(root, str):
        return False
    from .runtime import hold_reason

    if hold_reason():
        return False
    try:
        path = (Path(root) / "conversations.sqlite3").resolve()
        if not path.is_file():
            return False
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            row = db.execute("SELECT payload FROM conversations WHERE id=?", (identity,)).fetchone()
            payload = json.loads(row[0]) if row else None
            if not isinstance(payload, dict) or payload.get("kind") != "course":
                return False
            deleted = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='deleted_records'"
            ).fetchone()
            return not (deleted and db.execute(
                "SELECT 1 FROM deleted_records WHERE kind='conversation' AND identity=?",
                (identity,),
            ).fetchone())
    except (OSError, sqlite3.Error, ValueError, TypeError):
        return False


def render_course_chat_shell() -> None:
    """Scope layout rules to a marker that disappears on every other route or view."""
    if (st.session_state.get("sl_course_sidebar_initialized")
            != st.session_state.get("sl_open_root")):
        st.set_page_config(initial_sidebar_state="expanded")
        st.session_state["sl_course_sidebar_initialized"] = st.session_state.get("sl_open_root")
    st.markdown('<span id="sl-course-chat-active" hidden></span><style>' + _STYLE + "</style>",
                unsafe_allow_html=True)


def open_course_chat(identity: str) -> None:
    """Open an existing course chat, keeping its actual identity and all saved messages."""
    st.session_state["sl_chat_id"] = identity
    st.session_state[VIEW_KEY] = "chat"
    st.session_state["sl_ai_review_focus"] = False
    st.session_state.pop("sl_chat_attempt_origin", None)


def set_course_view(view: str) -> None:
    if view not in {"chat", "materials"}:
        raise ValueError("Unknown course workspace view")
    st.session_state[VIEW_KEY] = view


def leave_course_chat() -> None:
    """Return to normal learning navigation rather than retaining an immersive-view flag."""
    st.session_state.pop("sl_chat_id", None)
    st.session_state.pop("sl_chat_attempt_origin", None)
    st.session_state.pop(VIEW_KEY, None)
    st.session_state.pop("sl_course_sidebar_initialized", None)


def _history_labels(entries: list[dict]) -> dict[str, str]:
    counts = Counter(entry["title"] for entry in entries)
    return {entry["id"]: entry["title"] + (f"（对话{index + 1}）"
            if counts[entry["title"]] > 1 else "") for index, entry in enumerate(entries)}


def _capture_title(identity: str) -> None:
    st.session_state["sl_course_title_draft_" + identity] = st.session_state[
        "sl_course_title_" + identity
    ]


def _render_history(store: StudyStore, conversation: dict) -> None:
    from src.time_display import format_beijing_time

    from .chat_ui import _delete_control
    from .conversation_titles import ConversationTitles
    from .models import StudyError
    from .trash_ui import render_classified_trash

    titles = ConversationTitles(store)
    identity, subject = conversation["id"], conversation["subject"]
    query = st.text_input("搜索对话", key="sl_course_history_query_" + identity)
    entries = titles.entries(subject, query)
    if entries:
        labels = _history_labels(entries)
        for entry in entries:
            st.button(labels[entry["id"]], key="sl_course_history_open_" + entry["id"],
                      type="primary" if entry["id"] == identity else "secondary",
                      width="stretch", on_click=open_course_chat, args=(entry["id"],))
    else:
        st.caption("没有找到本课程对话。")
    with st.expander("对话设置"):
        selected = identity
        selected_entry = next(entry for entry in titles.entries(subject) if entry["id"] == identity)
        st.caption("创建：" + format_beijing_time(selected_entry["created_at"]))
        st.caption("最近使用：" + format_beijing_time(selected_entry["updated_at"]))
        baseline_key, draft_key = ("sl_course_title_baseline_" + selected,
                                   "sl_course_title_draft_" + selected)
        current = titles.title(selected)
        if (draft_key not in st.session_state
                or st.session_state[draft_key] == st.session_state.get(baseline_key)):
            st.session_state[draft_key] = current
            st.session_state[baseline_key] = current
        st.session_state["sl_course_title_" + selected] = st.session_state[draft_key]
        title = st.text_input("对话标题", max_chars=48, key="sl_course_title_" + selected,
                              on_change=_capture_title, args=(selected,))
        if st.button("保存名称", key="sl_course_rename_" + selected):
            try:
                titles.rename(selected, title)
            except StudyError as exc:
                st.warning(str(exc))
            else:
                st.session_state[draft_key] = title.strip()
                st.session_state[baseline_key] = title.strip()
                st.rerun()
        _delete_control(titles.chats, "conversation", selected,
                        key="sl_course_history_remove", label="当前对话")
    with st.expander("本课程回收站"):
        render_classified_trash(store, chats=titles.chats, key_prefix="sl_course_history_trash",
                                scope=("conversation",), subject=subject)


def render_course_sidebar(store: StudyStore, conversation: dict) -> None:
    """Render an ordinary, visible native session sidebar in place of platform navigation."""
    from .conversation_titles import ConversationTitles

    titles = ConversationTitles(store)
    with st.sidebar:
        st.markdown("**Nectivon · 课程对话**")
        st.caption(conversation["subject"])

        def new_chat() -> None:
            open_course_chat(titles.chats.open_course(conversation["subject"], new=True))

        st.button("新建对话", key="sl_chat_new", width="stretch", on_click=new_chat)
        st.button("课程资料", key="sl_course_materials", width="stretch",
                  on_click=set_course_view, args=("materials",))
        _render_history(store, conversation)


def render_course_chat(store: StudyStore, identity: str) -> None:
    """Use native course history alongside unbounded messages and a fixed composer."""
    from .chat_store import ChatStore
    from .chat_ui import render_chat
    from .conversation_titles import ConversationTitles

    conversation = ChatStore(store).conversation(identity)
    titles = ConversationTitles(store)
    titles.sync_auto(identity)
    opened_key = "sl_course_last_opened"
    opened = (str(store.root), identity)
    if st.session_state.get(opened_key) != opened:
        titles.mark_opened(identity)
        st.session_state[opened_key] = opened
    render_course_sidebar(store, conversation)
    with st.container(key="sl_course_chat_toolbar"):
        heading, back = st.columns([4, 1])
        with heading:
            st.caption("课程自由复盘 · " + conversation["subject"])
            st.subheader(titles.title(identity), help=titles.title(identity))
        back.button("返回学习整理", key="sl_course_workspace_back", on_click=leave_course_chat)
    render_chat(store, identity, embedded=True, immersive=True)
