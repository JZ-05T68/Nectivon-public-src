"""Course-only layout, route restoration and title controls with synthetic offline data."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from test_ai_review_support import RecordingProvider, review_case

from src.study_loop.conversation_titles import ConversationTitles
from src.study_loop.course_chat_ui import LEARNING_PAGE, VIEW_KEY, is_course_chat_view


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("EKB_STUDY_LOOP_ACCEPTANCE_CONFIRMED", "yes")
    return review_case(tmp_path / "NON_REAL_IMMERSIVE_COURSE")


@pytest.fixture
def provider(monkeypatch):
    provider = RecordingProvider("合成测试回答，保留公式 $a=b+c$。")
    monkeypatch.setattr("src.study_loop.chat_ui.configured_chat_model",
                        lambda: ("NON_REAL_PROVIDER", "NON_REAL_MODEL"))
    monkeypatch.setattr("src.study_loop.chat_ui.StudyChatProvider", lambda selection: provider)
    return provider


def _app(case, monkeypatch, identity):
    def links(entries, current):
        st.text("SYNTHETIC_NORMAL_NAVIGATION")

    monkeypatch.setattr("src.workspace_ui._render_links", links)
    monkeypatch.setattr("src.workspace_ui._render_topbar", lambda *args: None)
    monkeypatch.setattr("src.workspace_ui.render_voice_input", lambda: None)
    monkeypatch.setattr("src.workspace_ui.read_font_level", lambda path: 3)
    app = AppTest.from_string(
        "from pathlib import Path\n"
        "import streamlit as st\n"
        "from src.workspace_ui import render_workspace\n"
        "from src.study_loop.storage import StudyStore\n"
        "from src.study_loop.chat_store import ChatStore\n"
        "from src.study_loop.chat_ui import render_chat\n"
        "from src.study_loop.course_review_ui import render_course_workspace\n"
        "render_workspace('pages/18_学习整理.py')\n"
        "store=StudyStore(Path(st.session_state['sl_open_root']))\n"
        "identity=st.session_state.get('sl_chat_id')\n"
        "if not identity:\n"
        "    st.header('SYNTHETIC_LEARNING_HOME')\n"
        "elif ChatStore(store).conversation(identity)['kind']=='course':\n"
        "    render_course_workspace(store,identity)\n"
        "else:\n"
        "    render_chat(store,identity)\n"
    )
    app.session_state["sl_open_root"] = str(case.store.root)
    app.session_state["sl_chat_id"] = identity
    return app.run(timeout=30)


def _has_fixed_message_box(app):
    return any(block.proto.height_config.pixel_height == 520 for block in app.get("flex_container"))


def _normal_navigation_visible(app):
    return any(value.value == "SYNTHETIC_NORMAL_NAVIGATION" for value in app.sidebar.text)


def test_predicate_is_exact_read_only_and_respects_hold_focus_and_view(case, monkeypatch, tmp_path):
    identity = case.chats.open_course(case.selected.subject)
    state = {"sl_open_root": str(case.store.root), "sl_chat_id": identity}
    monkeypatch.setattr(st, "session_state", state)
    before = case.chats.db.read_bytes()
    assert is_course_chat_view(LEARNING_PAGE)
    assert not is_course_chat_view("app.py")
    state[VIEW_KEY] = "materials"
    assert not is_course_chat_view(LEARNING_PAGE)
    state[VIEW_KEY] = "chat"
    state["sl_ai_review_focus"] = True
    assert not is_course_chat_view(LEARNING_PAGE)
    state["sl_ai_review_focus"] = False
    monkeypatch.delenv("EKB_STUDY_LOOP_ACCEPTANCE_CONFIRMED")
    assert not is_course_chat_view(LEARNING_PAGE)
    monkeypatch.setenv("EKB_STUDY_LOOP_ACCEPTANCE_CONFIRMED", "yes")
    assert case.chats.db.read_bytes() == before
    state["sl_chat_id"] = case.chats.open_question(case.selected.id)
    assert not is_course_chat_view(LEARNING_PAGE)
    state["sl_chat_id"] = "NON_REAL_UNKNOWN_CONVERSATION"
    assert not is_course_chat_view(LEARNING_PAGE)
    state["sl_chat_id"] = identity
    case.chats.delete("conversation", identity, confirmed=True)
    deleted_bytes = case.chats.db.read_bytes()
    assert not is_course_chat_view(LEARNING_PAGE)
    assert case.chats.db.read_bytes() == deleted_bytes
    state["sl_open_root"] = str(tmp_path / "NON_EXISTENT_LIBRARY")
    assert not is_course_chat_view(LEARNING_PAGE)
    assert not (tmp_path / "NON_EXISTENT_LIBRARY").exists()


def test_cold_route_module_does_not_import_ai_or_initialize_chat_storage():
    code = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.startswith('src.ai') or fullname.endswith('.chat_store'):
            raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from src.study_loop.course_chat_ui import is_course_chat_view
assert is_course_chat_view('app.py') is False
"""
    result = subprocess.run([sys.executable, "-c", code],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True,
                            text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr


def test_default_course_chat_uses_page_height_and_send_derives_title_once(
    case, provider, monkeypatch,
):
    identity = case.chats.open_course(case.selected.subject)
    app = _app(case, monkeypatch, identity)
    assert not app.exception and not app.error
    assert app.subheader[0].value == "新对话"
    assert app.text_input(key="sl_course_title_" + identity).value == "新对话"
    assert any(value.value.startswith("创建：") for value in app.caption)
    assert app.button(key="sl_course_materials")
    assert any("可以直接提问" in value.value and "课程资料" in value.value
               for value in app.markdown)
    assert not any(value.key == "sl_course_review_" + identity + "_query"
                   for value in app.text_input)
    assert not _has_fixed_message_box(app)
    assert not _normal_navigation_visible(app)
    assert app.sidebar.button(key="sl_chat_new")
    assert app.sidebar.button(key="sl_course_materials")
    assert app.sidebar.button(key="sl_course_history_open_" + identity)
    assert app.sidebar.text_input(key="sl_course_history_query_" + identity)
    assert app.session_state["sl_course_sidebar_initialized"] == str(case.store.root)
    assert provider.prompts == []
    app.chat_input[0].set_value("请解释合成课程概念之间的关系与适用条件。\n").run(timeout=30)
    assert not app.exception and not app.error
    assert len(provider.prompts) == 1
    title = ConversationTitles(case.store).title(identity)
    assert app.subheader[0].value == title and 8 <= len(title) <= 24
    assert app.text_input(key="sl_course_title_" + identity).value == title
    with case.chats.connect() as db:
        persisted = db.execute("SELECT * FROM conversation_titles WHERE conversation_id=?",
                               (identity,)).fetchone()
        assert persisted["mode"] == "auto" and persisted["title"] == title
        assert persisted["first_user_message_id"] == case.chats.messages(identity)[0]["id"]
    assert len(case.chats.messages(identity)) == 2
    assert not any("开始这次复习吧" in value.value for value in app.markdown)
    app.run(timeout=30)
    assert len(provider.prompts) == 1


def test_title_input_preserves_real_unsaved_manual_draft_when_derived_title_changes(
    case, provider, monkeypatch,
):
    identity = case.chats.open_course(case.selected.subject)
    app = _app(case, monkeypatch, identity)
    manual = "尚未保存的手动课程名称"
    app.text_input(key="sl_course_title_" + identity).set_value(manual).run(timeout=30)
    assert ConversationTitles(case.store).title(identity) == "新对话"
    app.chat_input[0].set_value("请说明合成概念之间的关联与适用条件。").run(timeout=30)
    derived = ConversationTitles(case.store).title(identity)
    assert derived not in {"新对话", manual}
    assert app.subheader[0].value == derived
    assert app.text_input(key="sl_course_title_" + identity).value == manual
    app.run(timeout=30)
    assert app.text_input(key="sl_course_title_" + identity).value == manual
    app.button(key="sl_course_rename_" + identity).click().run(timeout=30)
    assert not app.exception and ConversationTitles(case.store).title(identity) == manual
    assert app.subheader[0].value == manual
    assert len(provider.prompts) == 1


def test_materials_and_return_restore_navigation_and_question_chat_keeps_its_layout(
    case, provider, monkeypatch,
):
    identity = case.chats.open_course(case.selected.subject)
    app = _app(case, monkeypatch, identity)
    app.button(key="sl_course_materials").click().run(timeout=30)
    assert not app.exception and app.session_state[VIEW_KEY] == "materials"
    assert _normal_navigation_visible(app)
    assert not app.chat_input
    app.button(key="sl_course_back_chat").click().run(timeout=30)
    assert app.session_state[VIEW_KEY] == "chat" and not _normal_navigation_visible(app)
    app.button(key="sl_course_workspace_back").click().run(timeout=30)
    assert "sl_chat_id" not in app.session_state and VIEW_KEY not in app.session_state
    assert _normal_navigation_visible(app)
    question = case.chats.open_question(case.selected.id)
    app.session_state["sl_chat_id"] = question
    app.run(timeout=30)
    assert not app.exception and _normal_navigation_visible(app)
    assert _has_fixed_message_box(app)
    assert app.button(key="sl_chat_return")
    assert provider.prompts == []


def test_history_rename_search_open_and_new_chat_use_ids_and_do_not_mix_courses(
    case, provider, monkeypatch,
):
    first = case.chats.open_course(case.selected.subject)
    second = case.chats.open_course(case.selected.subject, new=True)
    foreign = case.chats.open_course(case.foreign.subject)
    titles = ConversationTitles(case.store)
    for identity in (first, second, foreign):
        titles.rename(identity, "同名合成课程对话")
    app = _app(case, monkeypatch, first)
    histories = [button for button in app.sidebar.button
                 if (button.key or "").startswith("sl_course_history_open_")]
    assert len(histories) == 2 and len({button.label for button in histories}) == 2
    assert not any(button.key == "sl_course_history_open_" + foreign for button in histories)
    app.sidebar.button(key="sl_course_history_open_" + second).click().run(timeout=30)
    assert app.session_state["sl_chat_id"] == second
    app.text_input(key="sl_course_title_" + second).set_value("人工重命名的课程主题")
    next(button for button in app.button if button.label == "保存名称").click().run(timeout=30)
    assert titles.title(second) == "人工重命名的课程主题"
    app.text_input(key="sl_course_history_query_" + second).set_value("人工重命名").run(timeout=30)
    histories = [button for button in app.sidebar.button
                 if (button.key or "").startswith("sl_course_history_open_")]
    assert len(histories) == 1 and histories[0].key == "sl_course_history_open_" + second
    app.button(key="sl_course_history_open_" + second).click().run(timeout=30)
    assert not app.exception and app.session_state["sl_chat_id"] == second
    assert app.subheader[0].value == "人工重命名的课程主题"
    app.button(key="sl_chat_new").click().run(timeout=30)
    newest = app.session_state["sl_chat_id"]
    assert newest not in {first, second, foreign}
    assert app.subheader[0].value == "新对话"
    assert case.chats.messages(first) == case.chats.messages(second) == []
    assert case.chats.messages(newest) == []
    assert provider.prompts == []


def test_course_entry_reopens_recent_used_chat_instead_of_empty_default(
    case, provider, monkeypatch,
):
    empty = case.chats.open_course(case.selected.subject)
    used = case.chats.open_course(case.selected.subject, new=True)
    case.chats.begin(used, "SYNTHETIC_USED", "请说明合成系统的条件边界与例子。", {})
    case.chats.finish(used, "SYNTHETIC_USED", "SYNTHETIC_ANSWER", {})
    before = case.chats.snapshot(used)
    app = AppTest.from_string(
        "from pathlib import Path\nimport streamlit as st\n"
        "from src.study_loop.storage import StudyStore\n"
        "from src.study_loop.chat_ui import course_entry\n"
        "course_entry(StudyStore(Path(st.session_state['sl_open_root'])))\n"
    )
    app.session_state["sl_open_root"] = str(case.store.root)
    app.run(timeout=30)
    app.button(key="sl_chat_course_open").click().run(timeout=30)
    assert not app.exception and app.session_state["sl_chat_id"] == used != empty
    assert case.chats.snapshot(used) == before and case.chats.messages(empty) == []
    assert provider.prompts == []


def test_new_formula_audit_is_text_status_once_without_rewriting_old_history(
    case, provider, monkeypatch,
):
    identity = case.chats.open_course(case.selected.subject)
    case.chats.begin(identity, "OLD", "OLD_QUESTION", {})
    case.chats.finish(identity, "OLD", "$a=b$", {})
    old = case.chats.messages(identity)
    app = _app(case, monkeypatch, identity)
    assert not any("公式与适用条件待人工核验" in item.value for item in app.caption)
    app.chat_input[0].set_value("请继续解释合成参数的条件与关系。").run(timeout=30)
    assert not app.exception and len(provider.prompts) == 1
    assert sum("公式与适用条件待人工核验" in item.value for item in app.caption) == 1
    assert case.chats.messages(identity)[:2] == old
