"""Explicit memory extraction and confirmation AppTests with temporary synthetic sources."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest
from test_ai_review_support import RecordingProvider, review_case

from src.study_loop import chat
from src.study_loop.ai_review import AIReviewService
from src.study_loop.knowledge_memory import KnowledgeMemoryService


@pytest.fixture
def case(tmp_path):
    source = review_case(tmp_path / "NON_REAL_MEMORY_UI")
    reviews = AIReviewService(source.store)
    document = reviews.generate([source.selected.id], RecordingProvider("NON_REAL_ORIGINAL_AI"))
    reviews.update_working(document["id"], r"NON_REAL_SAVED_MEMORY_BASIS $k=\frac{x}{y}$。")
    version = reviews.save_version(document["id"])
    provider = RecordingProvider(r"NON_REAL_EXTRACTED_CANDIDATE $k=\frac{x}{y}$。")
    bindings = []

    def factory():
        bindings.append(True)
        return provider

    return SimpleNamespace(
        **vars(source), reviews=reviews, document=document, version=version,
        memories=KnowledgeMemoryService(source.store), provider=provider,
        factory=factory, bindings=bindings, course_id=None,
    )


def _app(case, *, kind="review", version=None, model_available=True):
    if kind == "course" and case.course_id is None:
        case.course_id = case.chats.open_course("NON_REAL_COURSE_A", new=True)
        chat.send(case.chats, case.course_id, "NON_REAL_ACTUAL_COURSE_USER", "ORIGIN",
                  RecordingProvider("NON_REAL_ACTUAL_COURSE_AI"), allow_remote=True)
    script = (
        "from pathlib import Path\nimport streamlit as st\n"
        "from src.study_loop.storage import StudyStore\n"
        "from src.study_loop.ai_review import AIReviewService\n"
        "from src.study_loop.knowledge_memory_ui import (\n"
        "    render_review_memory, render_course_memory,\n)\n"
        "store = StudyStore(Path(st.session_state['test_root']))\n"
    )
    if kind == "review":
        script += (
            "service = AIReviewService(store)\n"
            "render_review_memory(store, service.document(st.session_state['test_document']), "
            "service.version(st.session_state['test_version']), "
            "provider_factory=st.session_state['test_factory'])"
        )
    else:
        script += (
            "render_course_memory(store, st.session_state['test_course'], "
            "provider_factory=st.session_state['test_factory'], "
            "model_available=st.session_state['test_model_available'])"
        )
    app = AppTest.from_string(script)
    app.session_state["test_root"] = str(case.store.root)
    app.session_state["test_document"] = case.document["id"]
    app.session_state["test_version"] = (version or case.version)["id"]
    app.session_state["test_course"] = case.course_id
    app.session_state["test_factory"] = case.factory
    app.session_state["test_model_available"] = model_available
    return app.run(timeout=30)


@pytest.mark.parametrize("kind", ["review", "course"])
def test_extraction_edit_preview_and_confirmation_are_separate_explicit_ui_actions(case, kind):
    app = _app(case, kind=kind)
    assert not app.exception and not app.error
    study_before = case.store.db.read_bytes()
    identity = case.document["id"] if kind == "review" else case.course_id
    conversation_id = case.document["conversation_id"] if kind == "review" else case.course_id
    original = copy.deepcopy(case.chats.snapshot(conversation_id))
    assert case.bindings == [] and case.provider.prompts == []
    app.button(key=f"sl_knowledge_extract_{kind}_" + identity).click().run(timeout=30)
    assert not app.exception and not app.error
    extraction = case.memories.extractions(conversation_id)[0]
    key = extraction["id"]
    assert extraction["source"]["type"] == (
        "review_memory" if kind == "review" else "course_dialogue_memory"
    )
    assert len(case.bindings) == len(case.provider.prompts) == 1
    assert app.button(key="sl_knowledge_save_" + key).disabled
    assert not app.checkbox(key="sl_knowledge_confirm_" + key).value
    edited = r"NON_REAL_MANUAL_MEMORY $k=\frac{x}{y}$；限定 $y>0$。"
    app.text_area(key="sl_knowledge_body_" + key).set_value(edited).run(timeout=30)
    assert any(value.label == "预览记忆候选" for value in app.expander)
    assert any(edited in value.value for value in app.markdown)
    app.run(timeout=30)
    assert case.memories.extraction(key)["content"] == edited
    assert case.memories.extraction(key)["raw_ai"] == case.provider.content
    assert case.chats.memories("NON_REAL_COURSE_A") == []
    assert len(case.bindings) == len(case.provider.prompts) == 1
    app.checkbox(key="sl_knowledge_confirm_" + key).check().run(timeout=30)
    app.button(key="sl_knowledge_save_" + key).click().run(timeout=30)
    assert not app.exception and not app.error
    memory = case.chats.memories("NON_REAL_COURSE_A")[0]
    assert memory["content"] == edited
    assert memory["sources"][0] == extraction["source"]
    assert case.chats.snapshot(conversation_id) == original
    assert len(case.provider.prompts) == 1 and len(case.bindings) == 1
    assert case.store.db.read_bytes() == study_before
    assert not case.store.sessions() and not case.store.attempts()


def test_historical_review_cannot_extract_and_stale_candidate_cannot_be_confirmed(case):
    identity = case.document["id"]
    case.reviews.update_working(identity, "NON_REAL_CURRENT_SAVED_VERSION")
    second = case.reviews.save_version(identity)
    app = _app(case, version=case.version)
    assert not app.exception and not app.error
    assert app.button(key="sl_knowledge_extract_review_" + identity).disabled
    assert case.bindings == [] and case.provider.prompts == []
    app.session_state["test_version"] = second["id"]
    app.run(timeout=30)
    assert not app.button(key="sl_knowledge_extract_review_" + identity).disabled
    app.button(key="sl_knowledge_extract_review_" + identity).click().run(timeout=30)
    extraction = case.memories.extractions(case.document["conversation_id"])[0]
    assert extraction["source"]["version_id"] == second["id"]
    case.reviews.update_working(identity, "NON_REAL_REPLACEMENT_AFTER_EXTRACTION")
    case.reviews.save_version(identity)
    app.run(timeout=30)
    assert not app.exception and not app.error and app.warning
    app.checkbox(key="sl_knowledge_confirm_" + extraction["id"]).check().run(timeout=30)
    assert app.button(key="sl_knowledge_save_" + extraction["id"]).disabled
    assert case.chats.memories("NON_REAL_COURSE_A") == []
    assert len(case.bindings) == len(case.provider.prompts) == 1


def test_candidate_edits_reopen_in_a_new_session_without_extraction_or_confirmation(case):
    app = _app(case)
    app.button(key="sl_knowledge_extract_review_" + case.document["id"]).click().run(timeout=30)
    extraction = case.memories.extractions(case.document["conversation_id"])[0]
    key = extraction["id"]
    content = r"NON_REAL_DURABLE_MEMORY_CANDIDATE $y>0$。"
    app.text_area(key="sl_knowledge_body_" + key).set_value(content).run(timeout=30)
    reopened = _app(case)
    assert not reopened.exception and not reopened.error
    assert reopened.text_area(key="sl_knowledge_body_" + key).value == content
    assert not reopened.checkbox(key="sl_knowledge_confirm_" + key).value
    assert reopened.button(key="sl_knowledge_save_" + key).disabled
    assert case.memories.extraction(key)["raw_ai"] == case.provider.content
    assert len(case.memories.extractions()) == 1
    assert len(case.bindings) == len(case.provider.prompts) == 1
    assert case.chats.memories("NON_REAL_COURSE_A") == []


def test_offline_course_disables_extraction_without_binding_the_provider_factory(case):
    app = _app(case, kind="course", model_available=False)
    assert not app.exception and not app.error
    assert app.button(key="sl_knowledge_extract_course_" + case.course_id).disabled
    app.run(timeout=30)
    assert not app.exception and not app.error
    assert case.bindings == [] and case.provider.prompts == []
    assert case.memories.extractions(case.course_id) == []
    assert case.chats.memories("NON_REAL_COURSE_A") == []


@pytest.mark.parametrize("deleted_kind", ["conversation", "archive"])
def test_memory_source_navigation_checks_the_live_conversation_not_its_archive(case, deleted_kind):
    identity = case.chats.open_course("NON_REAL_COURSE_A", new=True)
    chat.send(case.chats, identity, "NON_REAL_SOURCE_QUESTION", "SOURCE",
              RecordingProvider("NON_REAL_SOURCE_ANSWER"), allow_remote=True)
    history = copy.deepcopy(case.chats.snapshot(identity))
    extraction = case.memories.extract_course(identity, case.provider)
    memory_id = case.memories.confirm(extraction["id"], case.provider.content, confirmed=True)
    memory = copy.deepcopy(case.chats.memory(memory_id))
    deleted_id = identity if deleted_kind == "conversation" else extraction["archive_id"]
    case.chats.delete(deleted_kind, deleted_id, confirmed=True)
    app = AppTest.from_string(
        "from pathlib import Path\nimport streamlit as st\n"
        "from src.study_loop.storage import StudyStore\n"
        "from src.study_loop.knowledge_memory_ui import render_memory_catalog\n"
        "render_memory_catalog(StudyStore(Path(st.session_state['test_root'])), "
        "'NON_REAL_COURSE_A')"
    )
    app.session_state["test_root"] = str(case.store.root)
    app.run(timeout=30)
    assert not app.exception and not app.error
    key = "sl_knowledge_origin_" + memory_id + "_" + identity
    assert app.button(key=key).disabled is (deleted_kind == "conversation")
    assert "sl_chat_id" not in app.session_state
    if deleted_kind == "conversation":
        assert case.chats.memory_status(memory_id)["unavailable"]
        assert any("对话已删除" in caption.value and "回收站" in caption.value
                   for caption in app.caption)
        case.chats.restore("conversation", identity)
        app.run(timeout=30)
        assert not app.exception and not app.error
        assert not app.button(key=key).disabled
    app.button(key=key).click().run(timeout=30)
    assert not app.exception and not app.error
    assert app.session_state["sl_chat_id"] == identity
    assert app.session_state["sl_course_workspace_view"] == "chat"
    assert not app.session_state["sl_ai_review_focus"]
    assert case.chats.snapshot(identity) == history
    assert case.chats.memory(memory_id)["content"] == memory["content"]
    assert case.chats.memory(memory_id)["revision"] == memory["revision"]
    assert case.chats.memory(memory_id)["sources"] == memory["sources"]
    assert case.bindings == [] and len(case.provider.prompts) == 1
