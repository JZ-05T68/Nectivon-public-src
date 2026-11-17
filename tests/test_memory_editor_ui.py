"""Confirmed-memory editing uses one stable ID and never rewrites its extraction or source."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from color_editor_support import color_body, edit_color_body
from streamlit.testing.v1 import AppTest
from test_ai_review_support import RecordingProvider, review_case

from src.study_loop.ai_review import AIReviewService
from src.study_loop.knowledge_memory import KnowledgeMemoryService


@pytest.fixture
def case(tmp_path):
    source = review_case(tmp_path / "NON_REAL_CONFIRMED_EDITOR")
    reviews, review_provider = AIReviewService(source.store), RecordingProvider("NON_REAL_RAW_AI")
    document = reviews.generate([source.selected.id], review_provider)
    reviews.save_version(document["id"])
    service = KnowledgeMemoryService(source.store)
    provider = RecordingProvider("ORIGINAL_MEMORY_BODY")
    extraction = service.extract_review(document["id"], provider)
    memory = service.confirm(extraction["id"], "ORIGINAL_MEMORY_BODY", confirmed=True)
    return SimpleNamespace(**vars(source), reviews=reviews, document=document, service=service,
                           extraction=extraction, memory=memory, provider=provider,
                           review_provider=review_provider)


def _app(case, *, entry="editor", prefix="test_editor"):
    app = AppTest.from_string(
        "from pathlib import Path\nimport streamlit as st\n"
        "from src.study_loop.storage import StudyStore\n"
        "from src.study_loop.ai_review import AIReviewService\n"
        "from src.study_loop.memory_editor_ui import render_memory_editor\n"
        "from src.study_loop.knowledge_memory_ui import (\n"
        "    render_review_memory, render_memory_catalog,\n)\n"
        "store = StudyStore(Path(st.session_state['test_root']))\n"
        "entry = st.session_state['test_entry']\n"
        "def no_model():\n    raise AssertionError('Memory editing must not initialize AI')\n"
        "if entry == 'editor':\n"
        "    render_memory_editor(store, st.session_state['test_memory'], "
        "key_prefix=st.session_state['test_prefix'])\n"
        "elif entry == 'review':\n"
        "    service = AIReviewService(store)\n"
        "    document = service.document(st.session_state['test_document'])\n"
        "    render_review_memory(store, document,\n"
        "        service.version(document['last_saved_version_id']), provider_factory=no_model)\n"
        "elif entry == 'catalog':\n    render_memory_catalog(store, 'NON_REAL_COURSE_A')\n"
        "elif entry == 'management':\n"
        "    from src.study_loop.chat_ui import render_saved_management\n"
        "    render_saved_management(store)\n"
        "else:\n    st.write('NON_REAL_OTHER_PAGE')"
    )
    app.session_state["test_root"] = str(case.store.root)
    app.session_state["test_memory"] = case.memory
    app.session_state["test_document"] = case.document["id"]
    app.session_state["test_entry"] = entry
    app.session_state["test_prefix"] = prefix
    return app.run(timeout=30)


def _button(app, label):
    return next(button for button in app.button if button.label == label)


def _edit(app, content):
    button = next(button for button in app.button if button.label in {"编辑记忆", "继续编辑记忆"})
    button.click().run(timeout=30)
    edit_color_body(app, "记忆正文", content)
    assert not app.exception and not app.error


def test_confirmed_review_editor_saves_same_id_and_all_entry_points_show_current_body(case):
    before = copy.deepcopy(case.service.extraction(case.extraction["id"]))
    source = copy.deepcopy(case.chats.memory(case.memory)["sources"])
    study_bytes = case.store.db.read_bytes()
    app = _app(case, entry="review")
    edited = r"LATEST_MEMORY_BODY $q=\frac{a}{b}$，限定 $b>0$。"
    _edit(app, edited)
    assert case.chats.memory(case.memory)["content"] == "ORIGINAL_MEMORY_BODY"
    _button(app, "预览修改").click().run(timeout=30)
    assert any(edited in text.value for text in app.markdown)
    assert case.chats.memory(case.memory)["revision"] == 1
    _button(app, "保存修改").click().run(timeout=30)
    assert not app.exception and not app.error
    current = case.chats.memory(case.memory)
    assert current["content"] == edited and current["revision"] == 2
    assert current["id"] == case.memory and current["sources"] == source
    assert case.service.extraction(case.extraction["id"]) == before
    for entry in ("review", "catalog", "management"):
        reopened = _app(case, entry=entry)
        assert not reopened.exception and not reopened.error
        assert reopened.markdown[0].value == edited
        assert any(button.label == "编辑记忆" for button in reopened.button)
        assert any(button.label == "删除记忆" for button in reopened.button)
    assert case.store.db.read_bytes() == study_bytes
    assert len(case.provider.prompts) == len(case.review_provider.prompts) == 1
    assert not case.store.sessions() and not case.store.attempts()


def test_concurrent_input_survives_conflict_navigation_and_explicit_rebase(case):
    first = _app(case, prefix="first_tab")
    second = _app(case, prefix="second_tab")
    _edit(first, "FIRST_TAB_UNSAVED_INPUT")
    _edit(second, "SECOND_TAB_SAVED_BODY")
    _button(second, "保存修改").click().run(timeout=30)
    _button(first, "保存修改").click().run(timeout=30)
    assert not first.exception and not first.error and first.warning
    assert case.chats.memory(case.memory)["content"] == "SECOND_TAB_SAVED_BODY"
    assert case.chats.memory(case.memory)["revision"] == 2
    assert color_body(first, "记忆正文").value == (
        "FIRST_TAB_UNSAVED_INPUT"
    )
    first.session_state["test_entry"] = "other"
    first.run(timeout=30)
    first.session_state["test_entry"] = "editor"
    first.run(timeout=30)
    assert color_body(first, "记忆正文").value == (
        "FIRST_TAB_UNSAVED_INPUT"
    )
    _button(first, "核对当前正文后继续编辑").click().run(timeout=30)
    _button(first, "保存修改").click().run(timeout=30)
    assert not first.exception and not first.error
    assert case.chats.memory(case.memory)["content"] == "FIRST_TAB_UNSAVED_INPUT"
    assert case.chats.memory(case.memory)["revision"] == 3
    assert case.chats.memory_revisions(case.memory)[1]["content"] == "SECOND_TAB_SAVED_BODY"
    assert len(case.provider.prompts) == len(case.review_provider.prompts) == 1


def test_history_restore_appends_revision_and_keeps_source_and_confirmation_records(case):
    original = copy.deepcopy(case.service.extraction(case.extraction["id"]))
    source = copy.deepcopy(case.chats.memory(case.memory)["sources"])
    app = _app(case)
    _edit(app, "MEMORY_SECOND_REVISION")
    _button(app, "保存修改").click().run(timeout=30)
    next(box for box in app.selectbox if box.label == "历史版本").set_value(1).run(timeout=30)
    _button(app, "恢复此版本").click().run(timeout=30)
    assert not app.exception and not app.error
    memory = case.chats.memory(case.memory)
    assert memory["id"] == case.memory and memory["revision"] == 3
    assert memory["content"] == "ORIGINAL_MEMORY_BODY" and memory["sources"] == source
    versions = case.chats.memory_revisions(case.memory)
    assert [version["revision"] for version in versions] == [3, 2, 1]
    assert versions[0]["action"] == "restored" and versions[0]["restored_from_revision"] == 1
    assert case.service.extraction(case.extraction["id"]) == original
    assert len(case.provider.prompts) == len(case.review_provider.prompts) == 1


def test_restoring_history_keeps_uncommitted_input_separate_from_the_confirmed_body(case):
    app = _app(case)
    _edit(app, "SAVED_SECOND_VERSION")
    _button(app, "保存修改").click().run(timeout=30)
    _edit(app, "UNSAVED_INPUT_BEFORE_RESTORE")
    next(box for box in app.selectbox if box.label == "历史版本").set_value(1).run(timeout=30)
    _button(app, "恢复此版本").click().run(timeout=30)
    assert not app.exception and not app.error
    assert case.chats.memory(case.memory)["content"] == "ORIGINAL_MEMORY_BODY"
    assert case.chats.memory(case.memory)["revision"] == 3
    _button(app, "继续编辑记忆").click().run(timeout=30)
    assert color_body(app, "记忆正文").value == (
        "UNSAVED_INPUT_BEFORE_RESTORE"
    )
    assert case.chats.memory(case.memory)["content"] == "ORIGINAL_MEMORY_BODY"
    assert case.chats.memory_status(case.memory)["active"]


def test_management_edit_of_nonfirst_memory_keeps_its_stable_selection(case):
    identity = case.chats.open_course("NON_REAL_COURSE_A")
    case.chats.begin(identity, "OTHER_SOURCE", "NON_REAL_USER", {})
    case.chats.finish(identity, "OTHER_SOURCE", "NON_REAL_AI", {})
    candidate = case.chats.add_candidate(case.chats.archive(identity), "OTHER_NEWER_MEMORY")
    other = case.chats.confirm(candidate, "OTHER_NEWER_MEMORY", confirmed=True)
    app = _app(case, entry="management")
    assert app.selectbox(key="sl_manage_memory").value == other
    app.selectbox(key="sl_manage_memory").set_value(case.memory).run(timeout=30)
    _edit(app, "NONFIRST_MEMORY_UPDATED_BODY")
    _button(app, "保存修改").click().run(timeout=30)
    assert not app.exception and not app.error
    assert app.selectbox(key="sl_manage_memory").value == case.memory
    assert app.markdown[0].value == "NONFIRST_MEMORY_UPDATED_BODY"
    assert case.chats.memory(case.memory)["content"] == "NONFIRST_MEMORY_UPDATED_BODY"
    assert case.chats.memory(other)["content"] == "OTHER_NEWER_MEMORY"
    assert len(case.provider.prompts) == len(case.review_provider.prompts) == 1


@pytest.mark.parametrize("status", ["stale", "unavailable"])
def test_inactive_memory_edit_delete_restore_does_not_reactivate_source(case, status):
    if status == "stale":
        case.reviews.update_working(case.document["id"], "NEW_SAVED_SOURCE_VERSION")
        case.reviews.save_version(case.document["id"])
    else:
        case.reviews.delete_document(case.document["id"], confirmed=True)
    assert case.chats.memory_status(case.memory)[status]
    app = _app(case)
    assert any("当前不参与自动检索" in caption.value for caption in app.caption)
    _edit(app, "MANUALLY_EDITED_INACTIVE_MEMORY")
    _button(app, "保存修改").click().run(timeout=30)
    assert case.chats.memory(case.memory)["content"] == "MANUALLY_EDITED_INACTIVE_MEMORY"
    assert case.chats.memory_status(case.memory)[status]
    assert case.chats.memories("NON_REAL_COURSE_A") == []
    assert _button(app, "删除记忆").disabled
    confirmed = next(box for box in app.checkbox if box.label == "确认将这条记忆移入回收站")
    confirmed.check().run(timeout=30)
    _button(app, "删除记忆").click().run(timeout=30)
    assert not app.exception and not app.error
    assert case.chats.memory(case.memory)["deleted"]
    _button(app, "恢复这条记忆").click().run(timeout=30)
    assert not app.exception and not app.error
    assert not case.chats.memory(case.memory)["deleted"]
    assert case.chats.memory(case.memory)["content"] == "MANUALLY_EDITED_INACTIVE_MEMORY"
    assert case.chats.memory_status(case.memory)[status]
    assert case.chats.memories("NON_REAL_COURSE_A") == []
    assert len(case.provider.prompts) == len(case.review_provider.prompts) == 1
