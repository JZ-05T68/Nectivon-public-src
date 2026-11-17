"""The learning home exposes confirmed-memory management without entering a chatbot."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from color_editor_support import edit_color_body
from streamlit.testing.v1 import AppTest
from test_ai_review_support import RecordingProvider, review_case

from src.study_loop.ai_review import AIReviewService
from src.study_loop.knowledge_memory import KnowledgeMemoryService


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("EKB_STUDY_LOOP_ACCEPTANCE_CONFIRMED", "yes")
    source = review_case(tmp_path / "NON_REAL_MEMORY_HOME")
    reviews = AIReviewService(source.store)
    document = reviews.generate([source.selected.id], RecordingProvider("NON_REAL_RAW_AI"))
    for number in range(1, 5):
        reviews.update_working(document["id"], f"NON_REAL_SAVED_REVIEW_{number}")
        reviews.save_version(document["id"])
    memories = KnowledgeMemoryService(source.store)
    extraction = memories.extract_review(document["id"], RecordingProvider("额定计算初始记忆"))
    identity = memories.confirm(extraction["id"], "额定计算初始记忆", confirmed=True)
    source.chats.update_memory(
        identity, "三相异步电动机额定计算\n\n已有人工修订。", expected_revision=1,
    )
    source.chats.add_candidate(extraction["archive_id"], "NON_REAL_UNCONFIRMED_CANDIDATE")
    bindings = []

    def no_model(*args, **kwargs):
        bindings.append(True)
        raise AssertionError("Opening memory management must not initialize a model")

    monkeypatch.setattr("src.study_loop.ai_review_ui.model_provider", no_model)
    monkeypatch.setattr("src.study_loop.chat_ui.StudyChatProvider", no_model)
    return SimpleNamespace(**vars(source), reviews=reviews, document=document,
                           memory=identity, extraction=extraction, bindings=bindings)


def _app(case, *, home=False):
    app = AppTest.from_string("from src.study_loop.ui import render\nrender()")
    app.session_state["sl_open_root"] = str(case.store.root)
    if home:
        app.session_state["sl_learning_view"] = "memory"
    return app.run(timeout=30)


def test_first_level_home_entry_searches_confirmed_current_revision_and_keeps_source_versions(case):
    memory = copy.deepcopy(case.chats.memory(case.memory))
    versions = copy.deepcopy(case.reviews.versions(case.document["id"]))
    study = case.store.db.read_bytes()
    app = _app(case)
    assert not app.exception and not app.error
    app.button(key="sl_memory_home_entry").click().run(timeout=30)
    assert not app.exception and not app.error
    assert app.session_state["sl_learning_view"] == "memory"
    assert "sl_chat_id" not in app.session_state
    app.text_input(key="sl_memory_home_search").set_value(case.source.name).run(timeout=30)
    assert app.selectbox(key="sl_memory_home_selection").value == case.memory
    app.text_input(key="sl_memory_home_search").set_value("额定计算").run(timeout=30)
    app.selectbox(key="sl_memory_home_course").select("NON_REAL_COURSE_A").run(timeout=30)
    assert app.selectbox(key="sl_memory_home_selection").value == case.memory
    assert len(app.selectbox(key="sl_memory_home_selection").options) == 1
    captions = "\n".join(caption.value for caption in app.caption)
    assert "当前记忆修订：第2修订" in captions
    assert "来源版本：复盘第4版" in captions
    assert "来源类型：单题复盘记忆" in captions and case.source.name in captions
    assert "NON_REAL_UNCONFIRMED_CANDIDATE" not in "\n".join(text.value for text in app.markdown)
    app.button(key="sl_memory_home_back").click().run(timeout=30)
    assert not app.exception and not app.error
    assert app.button(key="sl_memory_home_entry")
    assert case.chats.memory(case.memory) == memory
    assert case.reviews.versions(case.document["id"]) == versions
    assert case.store.db.read_bytes() == study and case.bindings == []


def test_home_saves_same_memory_id_and_new_session_reads_current_revision(case):
    app = _app(case, home=True)
    sources = copy.deepcopy(case.chats.memory(case.memory)["sources"])
    next(button for button in app.button if button.label == "编辑记忆").click().run(timeout=30)
    edited = '<span style="color:#D62828">重点 $P=UI$</span>\n\n人工修改。'
    edit_color_body(app, "记忆正文", edited)
    assert case.chats.memory(case.memory)["revision"] == 2
    next(button for button in app.button if button.label == "预览修改").click().run(timeout=30)
    next(button for button in app.button if button.label == "保存修改").click().run(timeout=30)
    assert not app.exception and not app.error
    assert app.selectbox(key="sl_memory_home_selection").value == case.memory
    reopened = _app(case, home=True)
    assert not reopened.exception and not reopened.error
    current = case.chats.memory(case.memory)
    assert current["content"] == edited and current["revision"] == 3
    assert current["sources"] == sources
    assert any("第3修订" in caption.value for caption in reopened.caption)
    assert any("复盘第4版" in caption.value for caption in reopened.caption)
    assert case.bindings == []


@pytest.mark.parametrize("state", ["stale", "unavailable"])
def test_home_manages_deleted_and_invalid_memories_without_reactivating_their_sources(case, state):
    if state == "stale":
        case.reviews.update_working(case.document["id"], "NON_REAL_NEW_SAVED_SOURCE")
        case.reviews.save_version(case.document["id"])
    else:
        case.reviews.delete_document(case.document["id"], confirmed=True)
    before = copy.deepcopy(case.chats.memory(case.memory))
    app = _app(case, home=True)
    app.selectbox(key="sl_memory_home_status").select("来源失效").run(timeout=30)
    assert app.selectbox(key="sl_memory_home_selection").value == case.memory
    assert any("当前不参与自动检索" in caption.value for caption in app.caption)
    confirmation = next(box for box in app.checkbox if box.label == "确认将这条记忆移入回收站")
    confirmation.check().run(timeout=30)
    next(button for button in app.button if button.label == "删除记忆").click().run(timeout=30)
    app.selectbox(key="sl_memory_home_status").select("已删除").run(timeout=30)
    assert app.selectbox(key="sl_memory_home_selection").value == case.memory
    next(button for button in app.button if button.label == "恢复这条记忆").click().run(timeout=30)
    assert not app.exception and not app.error
    app.selectbox(key="sl_memory_home_status").select("来源失效").run(timeout=30)
    current = case.chats.memory(case.memory)
    assert current["content"] == before["content"] and current["revision"] == before["revision"]
    assert current["sources"] == before["sources"] and current["source_status"][state]
    assert case.chats.memories("NON_REAL_COURSE_A") == [] and case.bindings == []
