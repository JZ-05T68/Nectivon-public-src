"""Local review navigation and restart regressions; no runtime model calls from the UI."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from color_editor_support import color_body
from streamlit.testing.v1 import AppTest
from test_ai_review_support import RecordingProvider, review_case

from src.study_loop.ai_review import AIReviewService
from src.study_loop.storage import StudyStore


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("EKB_STUDY_LOOP_ACCEPTANCE_CONFIRMED", "yes")
    source = review_case(tmp_path / "NON_REAL_RESUME_UI")
    service, provider = AIReviewService(source.store), RecordingProvider("NON_REAL_INITIAL_AI")
    document = service.generate([source.selected.id], provider)
    bindings = []

    def no_model():
        bindings.append(True)
        raise AssertionError("Navigation must not initialize a runtime model")

    monkeypatch.setattr("src.study_loop.ai_review_ui.model_provider", no_model)
    return SimpleNamespace(**vars(source), service=service, provider=provider,
                           document=document, bindings=bindings)


def _app(case, *, origin_version=None, chat_id=None):
    script = "from src.study_loop.ui import render\nrender()"
    if origin_version:
        script = (
            "import streamlit as st\n"
            "from src.study_loop.ai_review_ui import open_ai_review\n"
            "st.button('返回记忆来源版本', key='test_origin_version', "
            "on_click=open_ai_review, args=st.session_state['test_origin'])\n" + script
        )
    app = AppTest.from_string(script)
    app.session_state["sl_open_root"] = str(case.store.root)
    app.session_state["sl_ai_review_focus"] = True
    if chat_id:
        app.session_state["sl_chat_id"] = chat_id
    if origin_version:
        app.session_state["test_origin"] = (case.selected.id, case.document["id"], origin_version)
    return app.run(timeout=30)


def test_immediate_leave_preserves_exact_editor_input_without_guard_call_or_new_version(case):
    identity = case.document["id"]
    before = case.store.db.read_bytes()
    app = _app(case)
    app.radio(key="sl_ai_review_view_" + identity).set_value("编辑").run(timeout=30)
    source = color_body(app, "复盘正文")
    content = "NON_REAL_UNSAVED_INPUT\n\n" + r"$q=\frac{a}{b}$，保留本人的修改。"
    source.set_value(content)
    app.button(key="sl_ai_review_back").click().run(timeout=30)
    assert not app.exception and not app.error
    assert not app.session_state["sl_ai_review_focus"]
    assert not any(value.key == "sl_ai_review_keep_switch" for value in app.button)
    assert case.service.document(identity)["working"] == content
    assert case.service.versions(identity) == []
    app.button(key="sl_draft_ai_review_" + case.selected.id).click().run(timeout=30)
    assert not app.exception and not app.error
    assert app.session_state["sl_ai_review_document_id"] == identity
    visible = [value.value for value in app.markdown] + [value.value for value in app.text_area]
    visible += [color_body(app, "复盘正文").value] if app.get("component_instance") else []
    assert content in visible
    assert case.service.versions(identity) == []
    assert case.store.db.read_bytes() == before
    assert case.bindings == [] and len(case.provider.prompts) == 1
    assert not case.store.sessions() and not case.store.attempts()


def test_new_session_resumes_last_manual_document_and_viewed_version_ahead_of_new_ai(case):
    identity = case.document["id"]
    case.service.update_working(identity, "NON_REAL_MANUAL_SAVED_FIRST")
    first = case.service.save_version(identity)
    case.service.update_working(identity, "NON_REAL_MANUAL_SAVED_SECOND")
    second = case.service.save_version(identity)
    case.service.remember_view(identity, first["id"])
    case.provider.content = "NON_REAL_NEWEST_AI_MUST_NOT_REPLACE_MANUAL_DOCUMENT"
    newer = case.service.generate([case.selected.id], case.provider)
    course = case.chats.open_course("NON_REAL_COURSE_A")
    app = _app(case, chat_id=course)
    assert not app.exception and not app.error
    assert app.session_state["sl_ai_review_document_id"] == identity
    assert app.session_state["sl_ai_review_selected_version"] == first["id"]
    assert any(value.value == first["content"] for value in app.markdown)
    assert not app.chat_input  # Focused review has priority over a retained chat route.
    restarted = AIReviewService(StudyStore(case.store.root))
    resumed = restarted.resume_document()
    assert resumed["id"] == identity and resumed["viewed_version_id"] == first["id"]
    assert restarted.document(identity)["working"] == second["content"]
    assert {version["id"] for version in restarted.versions(identity)} == {
        first["id"], second["id"],
    }
    assert restarted.document(newer["id"])["raw_ai"] == case.provider.content
    assert case.bindings == [] and len(case.provider.prompts) == 2


def test_soft_delete_and_restore_from_ui_preserve_original_working_and_saved_version(case):
    identity = case.document["id"]
    case.service.update_working(identity, "NON_REAL_MANUAL_DOCUMENT_TO_RESTORE")
    saved = case.service.save_version(identity)
    original = case.store.read_source(case.source)
    app = _app(case)
    app.checkbox(key="sl_ai_review_delete_confirm").check().run(timeout=30)
    app.button(key="sl_ai_review_delete_document").click().run(timeout=30)
    assert not app.exception and not app.error
    assert case.service.document(identity)["deleted_at"]
    assert case.service.documents() == []
    assert case.service.version(saved["id"]) == saved
    app = _app(case)  # Recovery must also work after closing and reopening the local library.
    assert not app.exception and not app.error
    app.button(key="sl_ai_review_restore_document").click().run(timeout=30)
    assert not app.exception and not app.error
    assert app.session_state["sl_ai_review_document_id"] == identity
    assert not case.service.document(identity).get("deleted_at")
    assert case.service.document(identity)["working"] == saved["content"]
    assert case.service.versions(identity) == [saved]
    assert case.store.read_source(case.source) == original
    assert case.chats.memories(None) == []
    assert case.bindings == [] and len(case.provider.prompts) == 1


def test_opening_exact_source_version_of_current_document_updates_the_view_without_a_call(case):
    identity = case.document["id"]
    case.service.update_working(identity, "NON_REAL_OLDER_SOURCE_VERSION")
    first = case.service.save_version(identity)
    case.service.update_working(identity, "NON_REAL_LATEST_WORKING_CONTENT")
    second = case.service.save_version(identity)
    case.service.remember_view(identity)
    app = _app(case, origin_version=first["id"])
    assert app.session_state["sl_ai_review_document_id"] == identity
    app.button(key="test_origin_version").click().run(timeout=30)
    assert not app.exception and not app.error
    assert app.session_state["sl_ai_review_selected_version"] == first["id"]
    assert any(value.value == first["content"] for value in app.markdown)
    assert case.service.document(identity)["working"] == second["content"]
    assert case.service.document(identity)["viewed_version_id"] == first["id"]
    assert case.service.versions(identity) == [second, first]
    assert case.bindings == [] and len(case.provider.prompts) == 1
