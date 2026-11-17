"""Categorized recovery reuses existing stores and cannot leak other courses into a drawer."""

from __future__ import annotations

import copy

import pytest
from streamlit.testing.v1 import AppTest
from test_ai_review_support import RecordingProvider, review_case

from src.study_loop.ai_review import AIReviewService
from src.study_loop.conversation_titles import ConversationTitles
from src.study_loop.models import StudyError


def _app(store, *, scope=None, subject=None):
    app = AppTest.from_string(
        "from pathlib import Path\nimport streamlit as st\n"
        "from src.study_loop.storage import StudyStore\n"
        "from src.study_loop.trash_ui import render_classified_trash\n"
        "render_classified_trash(StudyStore(Path(st.session_state['test_root'])), "
        "key_prefix='test_trash', scope=st.session_state['test_scope'], "
        "subject=st.session_state['test_subject'])"
    )
    app.session_state["test_root"] = str(store.root)
    app.session_state["test_scope"] = scope
    app.session_state["test_subject"] = subject
    return app.run(timeout=30)


def test_global_trash_classifies_and_restores_existing_record_types_without_new_storage(tmp_path):
    case = review_case(tmp_path / "NON_REAL_CLASSIFIED_TRASH")
    chats = case.chats
    conversation = chats.open_course("NON_REAL_COURSE_A")
    chats.begin(conversation, "TEST_TURN", "NON_REAL_USER", {})
    chats.finish(conversation, "TEST_TURN", "NON_REAL_AI", {})
    archive = chats.archive(conversation)
    candidate = chats.add_candidate(archive, "NON_REAL_MEMORY")
    memory = chats.confirm(candidate, "NON_REAL_MEMORY", confirmed=True)
    reviews, provider = AIReviewService(case.store), RecordingProvider("NON_REAL_REVIEW")
    document = reviews.generate([case.selected.id], provider)
    version = reviews.save_version(document["id"])
    original = copy.deepcopy(chats.snapshot(conversation))
    source_bytes = case.store.read_source(case.source)
    identities = {
        "memory": memory, "conversation": conversation, "archive": archive,
        "course": "NON_REAL_COURSE_A", "candidate": candidate,
        "review_document": document["id"], "source": case.source.id,
    }
    for kind in ("memory", "conversation", "archive", "course", "candidate"):
        chats.delete(kind, identities[kind], confirmed=True)
    reviews.delete_document(document["id"], confirmed=True)
    case.store.delete_source(case.source.id, user_confirmed=True)
    app = _app(case.store)
    assert not app.exception and not app.error
    assert len(app.selectbox(key="test_trash_category").options) == 7
    for kind, identity in identities.items():
        app.selectbox(key="test_trash_category").set_value(kind).run(timeout=30)
        assert not app.exception and not app.error
        selection = app.selectbox(key="test_trash_selection_" + kind)
        assert selection.value == identity
        if kind != "course":  # A user-provided course name is its readable identity.
            assert identity not in "\n".join(selection.options)
        app.button(key="test_trash_restore").click().run(timeout=30)
        assert not app.exception and not app.error
    assert chats.trash() == [] and case.store.deleted_sources() == []
    assert not reviews.document(document["id"]).get("deleted_at")
    assert reviews.version(version["id"]) == version
    assert chats.snapshot(conversation) == original
    assert chats.memory(memory)["content"] == "NON_REAL_MEMORY"
    assert case.store.read_source(case.source) == source_bytes
    assert len(provider.prompts) == 1
    assert not case.store.sessions() and not case.store.attempts()


def test_course_drawer_trash_restores_only_its_own_course_conversations(tmp_path):
    case = review_case(tmp_path / "NON_REAL_COURSE_TRASH")
    chats = case.chats
    own = chats.open_course("NON_REAL_COURSE_A")
    titles = ConversationTitles(case.store)
    titles.rename(own, "本课程手动设置的标题")
    foreign = chats.open_course("NON_REAL_COURSE_B")
    question = chats.open_question(case.selected.id)
    review = AIReviewService(case.store).generate([case.selected.id], RecordingProvider("NON_REAL"))
    for identity in (own, foreign, question, review["conversation_id"]):
        chats.delete("conversation", identity, confirmed=True)
    app = _app(case.store, scope=("conversation", "memory"), subject="NON_REAL_COURSE_A")
    assert not app.exception and not app.error
    assert app.selectbox(key="test_trash_category").options == ["对话"]
    selected = app.selectbox(key="test_trash_selection_conversation")
    assert selected.value == own and len(selected.options) == 1
    assert "本课程手动设置的标题" in selected.options[0]
    app.button(key="test_trash_restore").click().run(timeout=30)
    assert not app.exception and not app.error
    assert chats.conversation(own)["subject"] == "NON_REAL_COURSE_A"
    assert titles.title(own) == "本课程手动设置的标题"
    for identity in (foreign, question, review["conversation_id"]):
        with pytest.raises(StudyError):
            chats.conversation(identity)
