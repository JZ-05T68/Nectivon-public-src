"""Legacy/manual candidate links reopen current stable-ID memory without reconfirming copies."""

from __future__ import annotations

import copy

import pytest
from color_editor_support import edit_color_body
from streamlit.testing.v1 import AppTest
from test_ai_review_support import review_case


@pytest.fixture
def case(tmp_path):
    case = review_case(tmp_path / "NON_REAL_LEGACY_MEMORY_ENTRY")
    identity = case.chats.open_course(case.selected.subject)
    case.chats.begin(identity, "NON_REAL_LEGACY_REQ", "NON_REAL_LEGACY_USER", {})
    case.chats.finish(identity, "NON_REAL_LEGACY_REQ", "NON_REAL_LEGACY_AI", {})
    archive = case.chats.archive(identity)
    candidate_id = case.chats.add_candidate(archive, "NON_REAL_ORIGINAL_CANDIDATE")
    candidate = next(value for value in case.chats.candidates(identity)
                     if value["id"] == candidate_id)
    return case, identity, candidate


def _app(case):
    data, _, candidate = case
    app = AppTest.from_string(
        "from pathlib import Path\n"
        "import streamlit as st\n"
        "from src.study_loop.storage import StudyStore\n"
        "from src.study_loop.chat_store import ChatStore\n"
        "from src.study_loop.chat_ui import render_candidate_confirmation\n"
        "chats=ChatStore(StudyStore(Path(st.session_state['test_root'])))\n"
        "render_candidate_confirmation(chats,st.session_state['test_candidate'])\n"
    )
    app.session_state["test_root"] = str(data.store.root)
    app.session_state["test_candidate"] = copy.deepcopy(candidate)
    return app.run(timeout=30)


def _button(app, label):
    return next(value for value in app.button if value.label == label)


def test_confirmed_manual_candidate_immediately_edits_same_memory_not_a_second_confirmation(case):
    data, identity, candidate = case
    messages = copy.deepcopy(data.chats.messages(identity))
    app = _app(case)
    next(value for value in app.text_area if value.label == "编辑知识记忆").set_value(
        "NON_REAL_FIRST_CONFIRMED"
    )
    next(value for value in app.checkbox if "我已核对来源" in value.label).check()
    _button(app, "确认写入知识记忆").click().run(timeout=30)
    assert not app.exception and not app.error
    memories = data.chats.memories(None)
    assert len(memories) == 1 and memories[0]["content"] == "NON_REAL_FIRST_CONFIRMED"
    memory_id = memories[0]["id"]
    assert not any(value.label == "编辑知识记忆" for value in app.text_area)
    _button(app, "编辑记忆").click().run(timeout=30)
    edit_color_body(app, "记忆正文", r"NON_REAL_CURRENT_MEMORY $a=\frac{b}{c}$")
    _button(app, "预览修改").click().run(timeout=30)
    _button(app, "保存修改").click().run(timeout=30)
    assert not app.exception and not app.error
    assert [value["id"] for value in data.chats.memories(None)] == [memory_id]
    assert data.chats.memory(memory_id)["content"] == r"NON_REAL_CURRENT_MEMORY $a=\frac{b}{c}$"
    assert data.chats.messages(identity) == messages
    assert next(value for value in data.chats.candidates(identity)
                if value["id"] == candidate["id"])["content"] == candidate["content"]


def test_legacy_confirmed_candidate_uses_current_body_and_deleted_memory_can_restore_same_id(case):
    data, _, candidate = case
    memory_id = data.chats.confirm(candidate["id"], "NON_REAL_INITIAL_CONFIRMATION", confirmed=True)
    memory = data.chats.memory(memory_id)
    data.chats.update_memory(memory_id, "NON_REAL_UPDATED_BODY",
                             expected_revision=memory["revision"])
    app = _app(case)
    assert not app.exception
    assert any(value.value == "NON_REAL_UPDATED_BODY" for value in app.markdown)
    assert not any(value.label == "确认写入知识记忆" for value in app.button)
    data.chats.delete("memory", memory_id, confirmed=True)
    app.run(timeout=30)
    assert not app.exception and _button(app, "恢复这条记忆")
    _button(app, "恢复这条记忆").click().run(timeout=30)
    assert not app.exception and not data.chats.memory(memory_id)["deleted"]
    assert data.chats.memory(memory_id)["content"] == "NON_REAL_UPDATED_BODY"
    assert [value["id"] for value in data.chats.memories(None)] == [memory_id]
