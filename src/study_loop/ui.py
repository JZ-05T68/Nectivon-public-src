"""Sole learning route; exam hold is checked before any candidate storage is opened."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import streamlit as st

from .models import StudyError
from .runtime import hold_reason
from .storage import StudyStore

LOGGER = logging.getLogger(__name__)


def render() -> None:
    """Initialize only by an explicit open/create action after the hold is released."""
    reason = hold_reason()
    if reason:
        st.title("学习整理与主动复盘")
        st.info(reason)
        return
    from .course_chat_ui import LEARNING_PAGE, is_course_chat_view

    if not is_course_chat_view(LEARNING_PAGE):
        st.title("学习整理与主动复盘")
    try:
        if (os.environ.get("EKB_STAGING_INSTANCE") == "1"
                and os.environ.get("EKB_STUDY_AUTO_OPEN") == "yes"
                and "sl_open_root" not in st.session_state):
            target = Path(os.environ["EKB_STUDY_LOOP_ROOT"])
            # Only a marked phase-three test instance can opt into unattended reopening.
            if (target.parent / ".nectivon-phase3-test.json").is_file():
                st.session_state["sl_open_root"] = str(StudyStore(target).root)
        if not st.session_state.get("sl_open_root"):
            _render_open()
        selected = st.session_state.get("sl_open_root")
        if not selected:
            return
        store = StudyStore(Path(selected))
        from .ai_review_ui import persist_ai_review

        if not persist_ai_review(store):
            st.warning("复盘草稿尚未保存，请返回编辑页保留当前输入后继续。")
            return
        # Detach existing input values from Streamlit's widget cleanup while visiting chat.
        # Buttons are boolean triggers and must never be restored as widget values.
        for key in list(st.session_state):
            value = st.session_state[key]
            retained_boolean = key.startswith(("sl_attempt_input_", "sl_draft_input_"))
            retained_boolean |= key.startswith("sl_attempt_") and key.endswith("_shown")
            if (key.startswith("sl_") and (not isinstance(value, bool) or retained_boolean)
                    and "upload" not in key and not key.startswith("sl_chat_input_")):
                st.session_state[key] = value
        from .ai_review_ui import render_ai_review
        from .chat_ui import course_entry, render_chat

        if st.session_state.get("sl_ai_review_focus"):
            render_ai_review(store, focused=True)
            return
        if st.session_state.get("sl_chat_id"):
            from .chat_store import ChatStore

            identity = st.session_state["sl_chat_id"]
            if ChatStore(store).conversation(identity)["kind"] == "course":
                from .course_review_ui import render_course_workspace

                render_course_workspace(store, identity)
            else:
                render_chat(store, identity)
            return
        from .knowledge_memory_home_ui import open_knowledge_memory, render_knowledge_memory_home

        if st.session_state.get("sl_learning_view") == "memory":
            render_knowledge_memory_home(store)
            return
        from .import_progress_ui import render_import_progress
        from .import_ui import render_drafts, render_input
        from .insights_ui import render_error_profile, render_recent_reviews
        from .review_ui import render_review, render_summary

        st.button("知识记忆", key="sl_memory_home_entry", on_click=open_knowledge_memory)
        course_entry(store)
        st.subheader("导入学习材料")
        render_input(store)
        render_import_progress(store)
        st.divider()
        import_tab, review_tab, summary_tab, errors_tab, recent_tab = st.tabs(
            ["待确认材料", "主动复盘", "复盘总结", "错题画像", "近期复盘"],
            key="sl_main_tab", on_change="rerun",
        )
        with import_tab:
            render_drafts(store)
        with review_tab:
            render_ai_review(store, advanced=False)
            with st.expander("按题练习／主动回忆"):
                render_review(store)
        with summary_tab:
            render_summary(store)
        with errors_tab:
            render_error_profile(store)
        with recent_tab:
            render_recent_reviews(store)
        _render_advanced(store)
    except StudyError as exc:
        st.warning(str(exc))
    except ImportError:
        LOGGER.error("study_loop component_missing")
        st.error("所需功能暂不可用，请检查安装后重试。")
    except Exception as exc:  # noqa: BLE001 - do not log sensitive paths, material or raw provider text
        LOGGER.error("study_loop operation_failed type=%s", type(exc).__name__)
        st.error("操作未完成，请稍后重试。")


def _render_advanced(store: StudyStore) -> None:
    """Keep infrequent controls together below the complete learning workflow."""
    from .ai_review_ui import render_ai_review_advanced
    from .chat_ui import render_saved_management
    from .import_ui import render_draft_advanced, render_model_action
    from .material_management_ui import render_material_management
    from .review_ui import render_review_advanced, render_tasks

    with st.expander("高级设置", expanded=False):
        source, correction, recognition, review, tasks, management, library, manuscripts = st.tabs(
            ["原始来源", "人工调整", "结构识别", "模型复盘", "任务", "删除与恢复", "学习库",
             "AI复盘稿件"]
        )
        with source:
            render_draft_advanced(store, section="source")
        with correction:
            render_draft_advanced(store, section="correction")
        with recognition:
            render_model_action(store)
        with review:
            render_review_advanced(store)
        with tasks:
            render_tasks(store)
        with management:
            render_material_management(store)
            render_saved_management(store)
        with library:
            _render_open()
        with manuscripts:
            render_ai_review_advanced(store)


def _render_open() -> None:
    with st.expander("打开独立学习整理库", expanded="sl_open_root" not in st.session_state):
        with st.form("sl_open"):
            root = st.text_input(
                "独立学习整理目录（与源码、平台资料及备份目录分开）",
                value=os.environ.get("EKB_STUDY_LOOP_ROOT", ""),
            )
            action = st.radio("操作", ["打开已有目录", "在空目录初始化"], horizontal=True)
            open_requested = st.form_submit_button("确认打开")
        if open_requested:
            if not root.strip():
                raise StudyError("请指定独立本地目录。")
            store = StudyStore(Path(root), create=action == "在空目录初始化")
            previous = st.session_state.get("sl_open_root")
            if previous != str(store.root):
                # Clear only this feature's ephemeral widgets, never user materials.
                for key in list(st.session_state):
                    if key.startswith("sl_"):
                        if key != "sl_open":
                            del st.session_state[key]
            st.session_state["sl_open_root"] = str(store.root)
            st.rerun()
