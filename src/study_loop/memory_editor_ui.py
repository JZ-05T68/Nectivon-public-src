"""One local editor for a confirmed memory, preserving its stable identity and provenance."""

from __future__ import annotations

import streamlit as st

from src.time_display import format_beijing_time

from .chat_store import ChatStore, digest
from .color_editor import (
    confirmed_color_body,
    deactivate_color_editor,
    defer_color_action,
    render_color_editor,
)
from .display import render_saved_text
from .models import StudyError
from .storage import StudyStore


def _prefix(store: StudyStore, identity: str, key_prefix: str) -> str:
    """Isolate entry-point drafts and prevent identical IDs in different libraries mixing input."""
    return "sl_memory_editor_" + digest([str(store.root), key_prefix, identity])[:20]


def _capture(prefix: str) -> None:
    """Keep text in a non-widget session key; navigation cleanup must not delete unsaved input."""
    if prefix + "_body" in st.session_state:
        st.session_state[prefix + "_draft"] = st.session_state[prefix + "_body"]


def _begin(store: StudyStore, identity: str, prefix: str) -> None:
    memory = ChatStore(store).memory(identity)
    if (prefix + "_draft" not in st.session_state
            or st.session_state[prefix + "_draft"] == st.session_state.get(prefix + "_baseline")):
        st.session_state[prefix + "_draft"] = memory["content"]
        st.session_state[prefix + "_base"] = memory["revision"]
        st.session_state[prefix + "_baseline"] = memory["content"]
    st.session_state[prefix + "_body"] = st.session_state[prefix + "_draft"]
    st.session_state[prefix + "_mode"] = "编辑"


def _defer(prefix: str, callback, *, args: tuple = (), kwargs: dict | None = None) -> bool:
    return bool(st.session_state.get(prefix + "_mode") == "编辑"
                and defer_color_action(prefix + "_body", callback, args=args, kwargs=kwargs))


def _preview(prefix: str, *, flushed: bool = False) -> None:
    if not flushed and _defer(prefix, _preview, args=(prefix,), kwargs={"flushed": True}):
        return
    try:
        st.session_state[prefix + "_draft"] = confirmed_color_body(
            prefix + "_body", flushed=flushed,
        )
    except StudyError as exc:
        st.session_state[prefix + "_error"] = str(exc)
        return
    st.session_state[prefix + "_mode"] = "预览"


def _save(store: StudyStore, identity: str, prefix: str, *, flushed: bool = False,
          expected_revision: int | None = None) -> None:
    if not flushed and _defer(prefix, _save, args=(store, identity, prefix),
                              kwargs={"flushed": True,
                                      "expected_revision": st.session_state[prefix + "_base"]}):
        return
    try:
        intended = confirmed_color_body(prefix + "_body", flushed=flushed)
        st.session_state[prefix + "_draft"] = intended
        updated = ChatStore(store).update_memory(
            identity, intended,
            expected_revision=(expected_revision if expected_revision is not None
                               else st.session_state[prefix + "_base"]),
        )
        if updated["content"] != intended:
            raise StudyError("保存后的正文与本次确认输入不一致，未显示保存成功。当前输入已保留。")
    except StudyError as exc:
        st.session_state[prefix + "_error"] = str(exc)
        st.session_state[prefix + "_mode"] = "编辑"
    else:
        st.session_state[prefix + "_draft"] = updated["content"]
        st.session_state[prefix + "_base"] = updated["revision"]
        st.session_state[prefix + "_baseline"] = updated["content"]
        st.session_state[prefix + "_body"] = updated["content"]
        st.session_state[prefix + "_mode"] = "阅读"
        st.session_state.pop(prefix + "_error", None)
        st.session_state[prefix + "_notice"] = "记忆修改已保存。"


def _continue_after_conflict(store: StudyStore, identity: str, prefix: str,
                             *, flushed: bool = False) -> None:
    """An explicit user choice refreshes the base revision while retaining every typed character."""
    if not flushed and _defer(prefix, _continue_after_conflict, args=(store, identity, prefix),
                              kwargs={"flushed": True}):
        return
    _capture(prefix)
    memory = ChatStore(store).memory(identity)
    st.session_state[prefix + "_base"] = memory["revision"]
    st.session_state[prefix + "_baseline"] = memory["content"]
    st.session_state.pop(prefix + "_error", None)
    st.session_state[prefix + "_mode"] = "编辑"
    st.session_state[prefix + "_notice"] = "输入已保留，请对照当前保存的正文核对后再保存修改。"


def _restore_version(
    store: StudyStore, identity: str, revision: int, expected_revision: int, prefix: str,
    *, flushed: bool = False,
) -> None:
    if not flushed and _defer(prefix, _restore_version,
                              args=(store, identity, revision, expected_revision, prefix),
                              kwargs={"flushed": True}):
        return
    _capture(prefix)
    chats = ChatStore(store)
    current = chats.memory(identity)
    draft = st.session_state.get(prefix + "_draft")
    try:
        updated = chats.restore_memory_revision(
            identity, revision, expected_revision=expected_revision,
        )
    except StudyError as exc:
        st.session_state[prefix + "_error"] = str(exc)
        return
    if draft is not None and draft != current["content"]:
        st.session_state[prefix + "_notice"] = "历史版本已恢复，未保存的输入已保留，可继续编辑。"
        st.session_state[prefix + "_mode"] = "阅读"
    else:
        st.session_state[prefix + "_draft"] = updated["content"]
        st.session_state[prefix + "_body"] = updated["content"]
        st.session_state[prefix + "_base"] = updated["revision"]
        st.session_state[prefix + "_baseline"] = updated["content"]
        st.session_state[prefix + "_mode"] = "阅读"
        st.session_state[prefix + "_notice"] = "所选历史正文已恢复。"


def _delete(store: StudyStore, identity: str, prefix: str, *, flushed: bool = False) -> None:
    if not flushed and _defer(prefix, _delete, args=(store, identity, prefix),
                              kwargs={"flushed": True}):
        return
    _capture(prefix)
    try:
        ChatStore(store).delete(
            "memory", identity, confirmed=bool(st.session_state.get(prefix + "_delete_confirm")),
        )
    except StudyError as exc:
        st.session_state[prefix + "_error"] = str(exc)


def render_memory_editor(store: StudyStore, memory_id: str, *, key_prefix: str) -> None:
    """Read the current stable-ID memory and save only after the user's explicit action."""
    chats = ChatStore(store)
    memory = chats.memory(memory_id)
    prefix = _prefix(store, memory_id, key_prefix)
    deactivate_color_editor(prefix + "_body")
    active_route = st.session_state.get("sl_memory_editor_route")
    if isinstance(active_route, dict) and active_route.get("body_key") == prefix + "_body":
        st.session_state.pop("sl_memory_editor_route", None)
    st.caption(f"当前记忆修订：第{memory['revision']}修订 · 已确认")
    if notice := st.session_state.pop(prefix + "_notice", None):
        st.success(notice)
    state = memory["source_status"]
    if not state["active"]:
        reasons = state["reasons"] or ["这条记忆已删除" if memory["deleted"] else "来源暂不可用"]
        st.caption("当前不参与自动检索：" + "；".join(reasons))
    if memory["deleted"]:
        render_saved_text(memory["content"])
        st.caption("这条记忆已在回收站，恢复后仍保留原有来源状态。")
        st.button("恢复这条记忆", key=prefix + "_restore_deleted",
                  on_click=chats.restore, args=("memory", memory_id))
        return
    mode = st.session_state.get(prefix + "_mode", "阅读")
    if error := st.session_state.get(prefix + "_error"):
        st.warning(error + " 当前输入已保留，可展开已保存正文进行核对。")
        with st.expander("当前已保存正文"):
            render_saved_text(memory["content"])
        st.button("核对当前正文后继续编辑", key=prefix + "_continue",
                  on_click=_continue_after_conflict, args=(store, memory_id, prefix))
    if mode == "阅读":
        render_saved_text(memory["content"])
        pending = (
            st.session_state.get(prefix + "_draft") != st.session_state.get(prefix + "_baseline")
        )
        st.button("继续编辑记忆" if pending else "编辑记忆", key=prefix + "_edit",
                  on_click=_begin, args=(store, memory_id, prefix))
    else:
        if mode == "编辑":
            st.session_state[prefix + "_body"] = st.session_state[prefix + "_draft"]
            render_color_editor("记忆正文", key=prefix + "_body", height=260, max_chars=30000,
                                on_change=_capture, args=(prefix,))
            st.session_state["sl_memory_editor_route"] = {
                "kind": "memory", "body_key": prefix + "_body", "study_root": str(store.root),
                "memory_id": memory_id, "key_prefix": key_prefix,
            }
            st.button("预览修改", key=prefix + "_preview", on_click=_preview, args=(prefix,))
        else:
            st.markdown("**修改预览**")
            render_saved_text(st.session_state[prefix + "_draft"])
            st.button("继续编辑记忆", key=prefix + "_edit",
                      on_click=_begin, args=(store, memory_id, prefix))
        st.button("保存修改", key=prefix + "_save", type="primary",
                  on_click=_save, args=(store, memory_id, prefix))
        st.caption("离开页面会保留本次会话中的编辑输入，只有“保存修改”会更新正式记忆。")
    with st.expander("高级设置", key=prefix + "_advanced", on_change="rerun"):
        versions = chats.memory_revisions(memory_id)
        if versions:
            by_revision = {version["revision"]: version for version in versions}
            selected = st.selectbox(
                "历史版本", list(by_revision), key=prefix + "_history",
                format_func=lambda number: f"第{number}修订 · "
                + format_beijing_time(by_revision[number]["created_at"]),
            )
            render_saved_text(by_revision[selected]["content"])
            st.button("恢复此版本", key=prefix + "_restore_version",
                      disabled=selected == memory["revision"], on_click=_restore_version,
                      args=(store, memory_id, selected, memory["revision"], prefix))
        confirmed = st.checkbox("确认将这条记忆移入回收站", key=prefix + "_delete_confirm")
        st.button("删除记忆", key=prefix + "_delete", disabled=not confirmed,
                  on_click=_delete, args=(store, memory_id, prefix))
