"""Synthetic AppTest coverage for the real AI-review editor and navigation boundaries."""

from __future__ import annotations

import copy

import pytest
from color_editor_support import color_body, edit_color_body
from streamlit.testing.v1 import AppTest
from test_ai_review_support import FORMULA_ANSWER, RecordingProvider, review_case

from src.ai.provider import AIUnavailableError
from src.study_loop.ai_review import AIReviewService
from src.study_loop.storage import StudyStore


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("EKB_STUDY_LOOP_ACCEPTANCE_CONFIRMED", "yes")
    return review_case(tmp_path / "NON_REAL_AI_REVIEW_UI")


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("EKB_STUDY_LOOP_ACCEPTANCE_CONFIRMED", "yes")
    value = RecordingProvider()
    monkeypatch.setattr("src.study_loop.ai_review_ui.model_provider", lambda: value)
    return value


def _app(case) -> AppTest:
    """Use the real focused route, including its established widget-state preservation."""
    app = AppTest.from_string("from src.study_loop.ui import render\nrender()")
    app.session_state["sl_open_root"] = str(case.store.root)
    app.session_state["sl_ai_review_focus"] = True
    return app.run(timeout=30)


def _generate(app: AppTest) -> str:
    app.button(key="sl_ai_review_generate").click().run(timeout=30)
    assert not app.exception and not app.error
    return app.session_state["sl_ai_review_document_id"]


def _edit(app: AppTest, identity: str, content: str) -> None:
    app.radio(key="sl_ai_review_view_" + identity).set_value("编辑").run(timeout=30)
    edit_color_body(app, "复盘正文", content)
    assert not app.exception and not app.error


def test_latest_course_version_continues_original_working_draft_without_replacing_history(
    case, provider,
):
    service = AIReviewService(case.store)
    app = _app(case)
    identity = _generate(app)
    _edit(app, identity, "NON_REAL_SAVED_VERSION_ONE")
    app.button(key="sl_ai_review_save_" + identity).click().run(timeout=30)
    first = copy.deepcopy(service.versions(identity)[0])
    unsaved = "NON_REAL_RETAINED_WORKING_FORMULA $P=UI$"
    _edit(app, identity, unsaved)
    app.session_state["sl_ai_review_requested"] = {
        "kind": "document", "id": identity, "version_id": first["id"],
    }
    app.run(timeout=30)
    assert app.session_state["sl_ai_review_selected_version"] == first["id"]
    app.radio(key="sl_ai_review_view_" + identity).set_value("编辑").run(timeout=30)
    assert not app.exception and not app.error
    assert color_body(app, "复盘正文").value == unsaved
    assert service.document(identity).get("viewed_version_id") is None
    app.button(key="sl_ai_review_save_" + identity).click().run(timeout=30)
    versions = service.versions(identity)
    assert len(versions) == 2 and versions[0]["content"] == unsaved
    assert service.version(first["id"]) == first
    assert len(service.documents()) == 1 and len(provider.prompts) == 1
    app.session_state["sl_ai_review_requested"] = {
        "kind": "document", "id": identity, "version_id": first["id"],
    }
    app.run(timeout=30)
    app.radio(key="sl_ai_review_view_" + identity).set_value("编辑").run(timeout=30)
    assert not any(widget.label == "复盘正文" for widget in app.text_area)
    assert any("历史版本" in value.value for value in app.info)
    assert service.document(identity)["working"] == unsaved
    assert service.version(first["id"]) == first


def test_pending_material_can_be_selected_and_rerenders_do_not_bind_or_call_a_model(
    case, monkeypatch,
):
    factories = []

    def forbidden_factory():
        factories.append(True)
        raise AssertionError("A read-only rerender must not bind any runtime model")

    monkeypatch.setattr("src.study_loop.ai_review_ui.model_provider", forbidden_factory)
    before = case.store.db.read_bytes()
    app = _app(case)
    assert not app.exception and not app.error
    assert app.selectbox(key="sl_ai_review_material").value == case.source.id
    assert len(app.selectbox(key="sl_ai_review_material").options) == 2
    assert app.selectbox(key="sl_ai_review_material").options.count(case.source.name) == 1
    assert app.selectbox(key="sl_ai_review_question").value == case.selected.id
    assert not app.button(key="sl_ai_review_generate").disabled
    assert not any(widget.label == "我已实际尝试作答" for widget in app.checkbox)
    assert not any("核对原题" in widget.label for widget in app.checkbox)
    app.text_area(key="sl_ai_review_requirements").set_value("NON_REAL_PERSIST_REQUIREMENT").run(
        timeout=30
    )
    assert app.session_state["sl_ai_review_item_id"] == case.selected.id
    assert app.text_area(key="sl_ai_review_requirements").value == "NON_REAL_PERSIST_REQUIREMENT"
    app.run(timeout=30)
    app.selectbox(key="sl_ai_review_question").set_value(case.sibling.id).run(timeout=30)
    assert not app.exception and not app.error and factories == []
    assert not AIReviewService(case.store).documents()
    assert case.store.db.read_bytes() == before


def test_generation_defaults_to_rendered_math_then_edit_preview_save_and_reopen(case, provider):
    service = AIReviewService(case.store)
    before = case.store.db.read_bytes()
    app = _app(case)
    app.text_area(key="sl_ai_review_requirements").set_value("NON_REAL_EXPLICIT_REQUIREMENT")
    identity = _generate(app)
    assert len(provider.prompts) == 1
    assert "NON_REAL_EXPLICIT_REQUIREMENT" in provider.prompts[-1]
    assert app.radio(key="sl_ai_review_view_" + identity).value == "阅读"
    assert any(r"$$F=\frac{P}{v}$$" in value.value for value in app.markdown)
    assert not any(value.label == "复盘正文" for value in app.text_area)
    assert not app.get("component_instance")
    edited = r"NON_REAL_REVISED_STUDY_CONTENT $F=\frac{P}{v}$"
    _edit(app, identity, edited)
    assert service.document(identity)["working"] == edited
    assert service.document(identity)["raw_ai"] == FORMULA_ANSWER
    assert service.versions(identity) == []
    app.radio(key="sl_ai_review_view_" + identity).set_value("预览").run(timeout=30)
    assert any(edited in value.value for value in app.markdown)
    app.button(key="sl_ai_review_save_" + identity).click().run(timeout=30)
    assert not app.exception and not app.error
    assert service.versions(identity)[0]["content"] == edited
    app.run(timeout=30)
    reopened = _app(case)
    assert reopened.session_state["sl_ai_review_document_id"] == identity
    assert reopened.radio(key="sl_ai_review_view_" + identity).value == "阅读"
    assert any(edited in value.value for value in reopened.markdown)
    assert len(service.versions(identity)) == 1 and len(provider.prompts) == 1
    assert case.store.db.read_bytes() == before
    assert case.chats.memories(None) == []


def test_regeneration_requires_unsaved_decision_and_preserves_original_working_copy(case, provider):
    service = AIReviewService(case.store)
    app = _app(case)
    first = _generate(app)
    edited = "NON_REAL_UNSAVED_FIRST_MANUSCRIPT"
    _edit(app, first, edited)
    frozen = copy.deepcopy(service.document(first))
    app.button(key="sl_ai_review_generate").click().run(timeout=30)
    assert not app.exception and len(provider.prompts) == 1
    assert app.session_state["sl_ai_review_document_id"] == first
    assert app.button(key="sl_ai_review_keep_switch")
    app.button(key="sl_ai_review_cancel_switch").click().run(timeout=30)
    assert app.session_state["sl_ai_review_document_id"] == first
    assert len(service.documents()) == 1 and len(provider.prompts) == 1
    app.button(key="sl_ai_review_generate").click().run(timeout=30)
    app.button(key="sl_ai_review_keep_switch").click().run(timeout=30)
    assert not app.exception and not app.error
    second = app.session_state["sl_ai_review_document_id"]
    assert second != first and len(provider.prompts) == 2
    assert service.document(first) == frozen
    assert service.document(second)["raw_ai"] == FORMULA_ANSWER
    assert service.document(second)["working"] == FORMULA_ANSWER
    assert edited not in provider.prompts[-1]
    assert not case.store.sessions() and not case.store.attempts()


def test_standalone_pending_cancel_does_not_restore_stale_widget_text(case, provider):
    """An early rerun once dropped the editor and later overwrote its durable working copy."""
    app = AppTest.from_string(
        "from pathlib import Path\nimport streamlit as st\n"
        "from src.study_loop.storage import StudyStore\n"
        "from src.study_loop.ai_review_ui import render_ai_review\n"
        "render_ai_review(StudyStore(Path(st.session_state['test_root'])))"
    )
    app.session_state["test_root"] = str(case.store.root)
    app.run(timeout=30)
    identity = _generate(app)
    content = "NON_REAL_KEEP_EDITOR_WHILE_AWAITING_SWITCH"
    _edit(app, identity, content)
    for _ in range(2):
        app.button(key="sl_ai_review_generate").click().run(timeout=30)
        assert color_body(app, "复盘正文").value == content
        app.button(key="sl_ai_review_cancel_switch").click().run(timeout=30)
        assert color_body(app, "复盘正文").value == content
    assert not app.exception and not app.error
    assert AIReviewService(case.store).document(identity)["working"] == content
    assert len(provider.prompts) == 1


@pytest.mark.parametrize("decision", ["autosaved_draft", "saved_version"])
@pytest.mark.parametrize("target", ["question", "source"])
def test_changing_material_or_question_keeps_unsaved_work_and_calls_no_model(
    case, provider, decision, target,
):
    service = AIReviewService(case.store)
    app = _app(case)
    identity = _generate(app)
    edited = "NON_REAL_EDIT_BEFORE_CHANGING_MATERIAL"
    _edit(app, identity, edited)
    if decision == "saved_version":
        app.button(key="sl_ai_review_save_" + identity).click().run(timeout=30)
    if target == "question":
        app.selectbox(key="sl_ai_review_question").set_value(case.sibling.id).run(timeout=30)
        destination = case.sibling.id
    else:
        app.selectbox(key="sl_ai_review_material").set_value(case.foreign.source_id).run(timeout=30)
        destination = case.foreign.id
    assert not app.exception and not app.error
    assert app.session_state["sl_ai_review_item_id"] == destination
    assert service.document(identity)["working"] == edited
    assert len(provider.prompts) == 1
    assert not any(value.key == "sl_ai_review_keep_switch" for value in app.button)
    assert service.document(identity)["working"] == edited
    if decision == "saved_version":
        assert service.versions(identity)[0]["content"] == edited
    else:
        assert service.versions(identity) == []
    assert len(provider.prompts) == 1
    assert case.chats.memories(None) == []


def test_saved_revision_is_ai_extracted_then_confirmed_without_using_unsaved_work(case, provider):
    from src.ai.conversation import decode_conversation
    from src.study_loop.knowledge_memory import KnowledgeMemoryService

    service = AIReviewService(case.store)
    app = _app(case)
    identity = _generate(app)
    _edit(app, identity, "NON_REAL_OLD_SAVED_REVISION")
    app.button(key="sl_ai_review_save_" + identity).click().run(timeout=30)
    old = service.versions(identity)[0]
    current = r"NON_REAL_CURRENT_SAVED_REVISION $F=\frac{P}{v}$"
    _edit(app, identity, current)
    app.button(key="sl_ai_review_save_" + identity).click().run(timeout=30)
    latest = service.versions(identity)[0]
    _edit(app, identity, "UNSAVED_WORK_MUST_NOT_BE_MEMORY_INPUT")
    app.selectbox(key="sl_ai_review_history_version_" + identity).set_value(old).run(timeout=30)
    assert app.button(key="sl_knowledge_extract_review_" + identity).disabled
    app.selectbox(key="sl_ai_review_history_version_" + identity).set_value(latest).run(timeout=30)
    provider.content = r"NON_REAL_EXTRACTED_MEMORY $F=\frac{P}{v}$"
    app.button(key="sl_knowledge_extract_review_" + identity).click().run(timeout=30)
    assert not app.exception and not app.error
    extraction = KnowledgeMemoryService(case.store).extractions()[0]
    inputs = decode_conversation(provider.prompts[-1])
    import json

    selected_input = json.loads(inputs[1]["content"].split("\n", 1)[1])
    assert selected_input["saved_review_content"] == current
    assert "UNSAVED_WORK_MUST_NOT_BE_MEMORY_INPUT" not in provider.prompts[-1]
    assert "NON_REAL_OLD_SAVED_REVISION" not in provider.prompts[-1]
    assert case.chats.memories(None) == []
    edited = "NON_REAL_HUMAN_REVIEWED_KNOWLEDGE"
    app.text_area(key="sl_knowledge_body_" + extraction["id"]).set_value(edited).run(timeout=30)
    app.checkbox(key="sl_knowledge_confirm_" + extraction["id"]).check().run(timeout=30)
    app.button(key="sl_knowledge_save_" + extraction["id"]).click().run(timeout=30)
    assert not app.exception and not app.error
    assert case.chats.memories(case.selected.subject)[0]["content"] == edited
    assert service.document(identity)["raw_ai"] == FORMULA_ANSWER
    assert service.document(identity)["working"] == "UNSAVED_WORK_MUST_NOT_BE_MEMORY_INPUT"
    assert service.version(old["id"])["content"] == "NON_REAL_OLD_SAVED_REVISION"
    assert not case.store.sessions() and not case.store.attempts()
    assert len(provider.prompts) == 2


def test_deleted_selected_source_blocks_a_stale_generation_action(case, provider):
    app = _app(case)
    original = case.store.read_source(case.source)
    case.store.delete_source(case.source.id, user_confirmed=True)
    app.button(key="sl_ai_review_generate").click().run(timeout=30)
    assert not app.exception and not app.error
    assert provider.prompts == [] and AIReviewService(case.store).documents() == []
    assert app.warning
    assert case.store.path(case.source.blob).read_bytes() == original


def test_existing_review_can_be_read_and_edited_without_model_credentials(case, monkeypatch):
    service = AIReviewService(case.store)
    provider = RecordingProvider()
    document = service.generate([case.selected.id], provider)
    service.save_version(document["id"])
    factories = []

    def unavailable():
        factories.append(True)
        raise AIUnavailableError("NON_REAL_NO_MODEL_CONFIGURED")

    monkeypatch.setattr("src.study_loop.ai_review_ui.model_provider", unavailable)
    app = _app(case)
    assert not app.exception and not app.error and factories == []
    _edit(app, document["id"], "NON_REAL_OFFLINE_EDIT")
    app.button(key="sl_ai_review_save_" + document["id"]).click().run(timeout=30)
    assert not app.exception and not app.error and factories == []
    assert service.versions(document["id"])[0]["content"] == "NON_REAL_OFFLINE_EDIT"
    assert len(provider.prompts) == 1


def test_stale_editor_keeps_both_concurrent_edits_instead_of_overwriting(case, provider):
    service = AIReviewService(case.store)
    first_app = _app(case)
    identity = _generate(first_app)
    first_app.radio(key="sl_ai_review_view_" + identity).set_value("编辑").run(timeout=30)
    second_app = _app(case)
    second_app.radio(key="sl_ai_review_view_" + identity).set_value("编辑").run(timeout=30)
    edit_color_body(first_app, "复盘正文", "FIRST_TAB_EDIT")
    edit_color_body(second_app, "复盘正文", "SECOND_TAB_EDIT")
    assert not second_app.exception and not second_app.error
    documents = service.documents(case.selected.id)
    assert {document["working"] for document in documents} == {"FIRST_TAB_EDIT", "SECOND_TAB_EDIT"}
    assert len(documents) == 2
    assert all(document["raw_ai"] == FORMULA_ANSWER for document in documents)
    assert service.document(identity)["working"] == "FIRST_TAB_EDIT"
    assert len(provider.prompts) == 1


def test_empty_library_provides_import_navigation_without_manual_prerequisites(tmp_path, provider):
    class EmptyCase:
        store = StudyStore(tmp_path / "NON_REAL_EMPTY_REVIEW", create=True)

    app = _app(EmptyCase())
    assert not app.exception and not app.error
    assert any("请先导入材料" in value.value for value in app.info)
    assert app.button(key="sl_ai_review_go_import")
    assert not app.checkbox and not app.text_area
    assert provider.prompts == []


def test_main_draft_entry_and_recent_history_reopen_exact_review_without_calls(
    case, provider, monkeypatch,
):
    monkeypatch.setenv("EKB_STUDY_LOOP_ACCEPTANCE_CONFIRMED", "yes")
    app = AppTest.from_string("from src.study_loop.ui import render\nrender()")
    app.session_state["sl_open_root"] = str(case.store.root)
    app.session_state["sl_main_tab"] = "待确认材料"
    app.run(timeout=30)
    assert not app.exception and not app.error
    study_before = case.store.db.read_bytes()
    assert any(value.label == "按题练习／主动回忆" for value in app.expander)
    app.button(key="sl_draft_ai_review_" + case.selected.id).click().run(timeout=30)
    assert not app.exception and not app.error
    assert app.session_state["sl_ai_review_focus"]
    assert app.session_state["sl_ai_review_item_id"] == case.selected.id
    assert provider.prompts == []
    identity = _generate(app)
    app.button(key="sl_ai_review_back").click().run(timeout=30)
    assert not app.exception and not app.error
    assert not app.session_state["sl_ai_review_focus"]
    app.button(key="sl_draft_ai_review_" + case.selected.id).click().run(timeout=30)
    assert app.session_state["sl_ai_review_document_id"] == identity
    _edit(app, identity, "NON_REAL_MAIN_ROUTE_UNSAVED_WORK")
    app.button(key="sl_ai_review_back").click().run(timeout=30)
    assert not app.session_state["sl_ai_review_focus"]
    assert AIReviewService(case.store).document(identity)["working"] == (
        "NON_REAL_MAIN_ROUTE_UNSAVED_WORK"
    )
    assert AIReviewService(case.store).versions(identity) == []
    assert not app.exception and not app.error
    assert not app.session_state["sl_ai_review_focus"]
    app.button(key="sl_recent_ai_review_" + identity).click().run(timeout=30)
    assert not app.exception and not app.error
    assert app.session_state["sl_ai_review_document_id"] == identity
    visible = [value.value for value in app.markdown] + [value.value for value in app.text_area]
    visible += [color_body(app, "复盘正文").value] if app.get("component_instance") else []
    assert any("NON_REAL_MAIN_ROUTE_UNSAVED_WORK" in value for value in visible)
    assert AIReviewService(case.store).document(identity)["working"] == (
        "NON_REAL_MAIN_ROUTE_UNSAVED_WORK"
    )
    assert len(provider.prompts) == 1
    assert case.store.db.read_bytes() == study_before
    assert not case.store.items() and not case.store.sessions() and not case.store.attempts()


def test_pending_file_selector_keeps_second_question_bound_through_ai_and_return(case, provider):
    app = AppTest.from_string("from src.study_loop.ui import render\nrender()")
    app.session_state["sl_open_root"] = str(case.store.root)
    app.session_state["sl_main_tab"] = "待确认材料"
    app.run(timeout=30)
    assert not app.exception and not app.error
    assert app.selectbox(key="sl_draft_choice").options.count(case.source.name) == 1
    key = "sl_draft_question_" + case.source.id
    app.selectbox(key=key).set_value(1).run(timeout=30)
    assert not app.exception and not app.error
    app.button(key="sl_draft_ai_review_" + case.sibling.id).click().run(timeout=30)
    assert app.session_state["sl_ai_review_item_id"] == case.sibling.id
    assert app.selectbox(key="sl_ai_review_material").value == case.source.id
    assert app.selectbox(key="sl_ai_review_question").value == case.sibling.id
    assert provider.prompts == []
    identity = _generate(app)
    document = AIReviewService(case.store).document(identity)
    assert document["item_ids"] == [case.sibling.id]
    assert "UNSELECTED_5_1_SENTINEL" in provider.prompts[-1]
    assert "SELECTED_4_2_QUESTION" not in provider.prompts[-1]
    app.button(key="sl_ai_review_back").click().run(timeout=30)
    assert not app.exception and not app.error
    assert app.selectbox(key=key).value == 1
    app.button(key="sl_draft_ai_review_" + case.sibling.id).click().run(timeout=30)
    assert not app.exception and not app.error
    assert app.session_state["sl_ai_review_document_id"] == identity
    assert len(provider.prompts) == 1
    assert not case.store.items() and not case.store.sessions() and not case.store.attempts()
