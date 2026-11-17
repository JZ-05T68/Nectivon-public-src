"""Synthetic AppTests for course review browsing, comparison and focused navigation."""

from __future__ import annotations

import copy

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from test_ai_review_support import RecordingProvider, review_case

from src.ai.provider import AIUnavailableError
from src.study_loop.ai_review import AIReviewService
from src.study_loop.chat_ui import render_chat as actual_render_chat

FIRST = "# 合成复盘甲\n\nSAVED_COURSE_ALPHA $F=\\frac{P}{v}$"
SECOND = "# 合成复盘乙\n\nSAVED_COURSE_BETA $q=\\sqrt{x}$"
UNSAVED = "UNSAVED_WORKING_MUST_NOT_ENTER_COURSE_VIEW"
FOREIGN = "FOREIGN_SAVED_REVIEW_MUST_NOT_ENTER_COURSE_VIEW"


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    case = review_case(tmp_path / "NON_REAL_COURSE_WORKSPACE")
    case.selected.chapter = "合成章节"
    case.store.confirm(case.selected, user_confirmed=True)
    service, provider = AIReviewService(case.store), RecordingProvider("NON_REAL_RAW_AI_ONLY")
    documents, versions = [], []
    for item, content in ((case.selected, FIRST), (case.selected, SECOND), (case.foreign, FOREIGN)):
        document = service.generate([item.id], provider)
        service.update_working(document["id"], content)
        versions.append(service.save_version(document["id"]))
        documents.append(document)
    service.update_working(documents[1]["id"], UNSAVED)
    identity = case.chats.open_course(case.selected.subject)
    calls = []

    def memory_catalog(store, subject):
        calls.append(("memory", subject))
        st.caption("合成记忆目录")

    def embedded_chat(store, selected_identity, *, embedded=False, immersive=False):
        calls.append(("chat", selected_identity, embedded, immersive))
        st.caption("合成独立课程对话")

    def forbidden_model():
        raise AssertionError("Course browsing must not bind a runtime model")

    monkeypatch.setattr("src.study_loop.course_review_ui._render_memory_catalog", memory_catalog)
    monkeypatch.setattr("src.study_loop.chat_ui.render_chat", embedded_chat)
    monkeypatch.setattr("src.study_loop.ai_review_ui.model_provider", forbidden_model)
    return case, service, provider, documents, versions, identity, calls


def _app(workspace, *, view="materials"):
    case, _, _, _, _, identity, _ = workspace
    app = AppTest.from_string(
        "from pathlib import Path\n"
        "import streamlit as st\n"
        "from src.study_loop.storage import StudyStore\n"
        "from src.study_loop.course_review_ui import render_course_workspace\n"
        "if st.session_state.get('sl_ai_review_focus'):\n"
        "    st.text('SYNTHETIC_AI_FOCUSED_PAGE')\n"
        "    def return_to_course():\n"
        "        st.session_state['sl_ai_review_focus'] = False\n"
        "    st.button('返回课程', on_click=return_to_course)\n"
        "elif st.session_state.get('sl_chat_id'):\n"
        "    render_course_workspace(StudyStore(Path(st.session_state['test_root'])), "
        "st.session_state['sl_chat_id'])\n"
        "else:\n"
        "    st.text('SYNTHETIC_LEARNING_HOME')\n"
    )
    app.session_state["test_root"] = str(case.store.root)
    app.session_state["sl_chat_id"] = identity
    app.session_state["sl_course_workspace_view"] = view
    return app.run(timeout=30)


def _visible(app):
    return "\n".join(value.value for group in (app.text, app.markdown, app.caption, app.info)
                     for value in group)


def test_browse_search_and_tabs_only_show_selected_course_saved_versions(workspace):
    case, service, provider, documents, _, identity, calls = workspace
    baseline = copy.deepcopy(service.documents())
    study, chat = case.store.db.read_bytes(), case.chats.db.read_bytes()
    app = _app(workspace)
    assert not app.exception and not app.error
    assert app.subheader[0].value == "课程复盘资料"
    assert app.button(key="sl_course_back_chat")
    key = "sl_course_review_" + identity + "_"
    choices = app.selectbox(key=key + "selected")
    assert len(choices.options) == 2 and len(set(choices.options)) == 2
    assert choices.options == [f"{case.source.name}（复盘1）", f"{case.source.name}（复盘2）"]
    assert choices.value in [document["id"] for document in documents[:2]]
    text = _visible(app)
    assert case.source.name in text and "合成章节" in text and "已保存" in text
    assert UNSAVED not in text and FOREIGN not in text and "NON_REAL_RAW_AI_ONLY" not in text
    assert not any(document["id"] in text for document in documents)
    app.text_input(key=key + "query").set_value("SAVED_COURSE_ALPHA").run(timeout=30)
    assert not app.exception and app.selectbox(key=key + "selected").value == documents[0]["id"]
    assert FIRST in _visible(app) and SECOND not in _visible(app)
    app.text_input(key=key + "query").set_value("NO_SYNTHETIC_MATCH").run(timeout=30)
    assert any("没有找到匹配" in info.value for info in app.info)
    assert service.documents() == baseline
    assert case.store.db.read_bytes() == study and case.chats.db.read_bytes() == chat
    assert len(provider.prompts) == 3
    assert not any(call[0] == "chat" for call in calls)
    assert all(call == ("memory", case.selected.subject) for call in calls if call[0] == "memory")


def test_comparison_reads_two_current_saved_bodies_without_mutating_history(workspace):
    case, service, provider, documents, _, identity, _ = workspace
    baseline = copy.deepcopy(service.documents())
    before_messages = copy.deepcopy(case.chats.messages(identity))
    app = _app(workspace)
    key = "sl_course_review_" + identity + "_compare"
    assert app.multiselect(key=key).options == [
        f"{case.source.name}（复盘1）", f"{case.source.name}（复盘2）",
    ]
    app.multiselect(key=key).set_value(
        [document["id"] for document in documents[:2]]
    ).run(timeout=30)
    assert not app.exception and not app.error
    assert FIRST in _visible(app) and SECOND in _visible(app)
    assert UNSAVED not in _visible(app) and FOREIGN not in _visible(app)
    assert app.multiselect(key=key).proto.max_selections == 10
    assert service.documents() == baseline
    assert case.chats.messages(identity) == before_messages == []
    assert len(provider.prompts) == 3


def test_open_original_keeps_course_conversation_and_exact_saved_version(workspace):
    _, _, provider, documents, versions, identity, _ = workspace
    app = _app(workspace)
    key = "sl_course_review_" + identity + "_"
    app.selectbox(key=key + "selected").set_value(documents[0]["id"]).run(timeout=30)
    app.button(key=key + "open").click().run(timeout=30)
    assert not app.exception and app.session_state["sl_chat_id"] == identity
    assert app.session_state["sl_ai_review_requested"] == {
        "kind": "document", "id": documents[0]["id"], "version_id": versions[0]["id"],
    }
    assert "SYNTHETIC_AI_FOCUSED_PAGE" in _visible(app)
    app.button[0].click().run(timeout=30)
    assert app.session_state["sl_chat_id"] == identity
    assert app.subheader[0].value == "课程复盘资料"
    assert len(provider.prompts) == 3
    app.button(key="sl_course_workspace_back").click().run(timeout=30)
    assert "sl_chat_id" not in app.session_state
    assert "SYNTHETIC_LEARNING_HOME" in _visible(app)


def test_empty_course_has_chinese_reason_and_no_empty_review_dropdown(workspace):
    case, _, provider, _, _, _, _ = workspace
    case.chats.set_course("NON_REAL_EMPTY_COURSE", [], "")
    empty_identity = case.chats.open_course("NON_REAL_EMPTY_COURSE")
    app = _app(workspace)
    app.session_state["sl_chat_id"] = empty_identity
    app.run(timeout=30)
    assert not app.exception and not app.error
    assert any("暂无已保存的复盘资料" in info.value for info in app.info)
    assert not app.selectbox and not app.multiselect
    assert len(provider.prompts) == 3


def test_new_course_chat_and_history_switch_keep_chat_view_and_independent_messages(
    workspace, monkeypatch,
):
    case, _, provider, _, _, identity, _ = workspace
    case.chats.begin(identity, "NON_REAL_COURSE_REQUEST", "NON_REAL_OLD_COURSE_QUESTION", {})
    case.chats.finish(identity, "NON_REAL_COURSE_REQUEST", "NON_REAL_OLD_COURSE_REPLY", {})
    original_messages = copy.deepcopy(case.chats.messages(identity))
    monkeypatch.setattr("src.study_loop.chat_ui.render_chat", actual_render_chat)

    def offline_model():
        raise AIUnavailableError("合成离线测试")

    monkeypatch.setattr("src.study_loop.chat_ui.configured_chat_model", offline_model)
    app = _app(workspace, view="chat")
    app.button(key="sl_chat_new").click().run(timeout=30)
    assert not app.exception and not app.error
    new_identity = app.session_state["sl_chat_id"]
    assert new_identity != identity
    assert app.session_state["sl_course_workspace_view"] == "chat"
    assert case.chats.messages(new_identity) == []
    assert case.chats.messages(identity) == original_messages
    assert {conversation["id"] for conversation in case.chats.conversations(
        "course", case.selected.subject
    )} == {identity, new_identity}
    app.button(key="sl_course_history_open_" + identity).click().run(timeout=30)
    assert not app.exception and app.session_state["sl_chat_id"] == identity
    assert app.session_state["sl_course_workspace_view"] == "chat"
    assert case.chats.messages(identity) == original_messages
    assert case.chats.messages(new_identity) == []
    assert len(provider.prompts) == 3
