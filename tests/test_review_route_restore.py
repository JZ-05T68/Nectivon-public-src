"""Cross-page workspace hook regressions for stale client widget replay."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import streamlit as st
from color_editor_support import color_body, edit_color_body
from streamlit.testing.v1 import AppTest
from test_ai_review_support import RecordingProvider, review_case

from src.study_loop.ai_review import AIReviewService

LEARNING = "pages/18_学习整理.py"


@pytest.fixture
def shell(monkeypatch):
    """Keep the real workspace route hook; omit unrelated sidebar links and voice widgets."""
    monkeypatch.setenv("EKB_STUDY_LOOP_ACCEPTANCE_CONFIRMED", "yes")
    monkeypatch.setattr("src.workspace_ui._render_links", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.workspace_ui.render_voice_input", lambda: None)
    monkeypatch.setattr("src.workspace_ui._render_topbar", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.workspace_ui._render_home_topbar", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.workspace_ui.read_font_level", lambda _: 3)


@pytest.fixture
def case(tmp_path, monkeypatch, shell):
    source = review_case(tmp_path / "NON_REAL_ROUTE_RESTORE")
    service, provider = AIReviewService(source.store), RecordingProvider("NON_REAL_OLD_AI_BODY")
    # The active document belongs to the second file, so a reset to the first is observable.
    document = service.generate([source.foreign.id], provider)
    version = service.save_version(document["id"])
    bindings = []

    def no_model():
        bindings.append(True)
        raise AssertionError("Changing workspace pages must not bind a model")

    monkeypatch.setattr("src.study_loop.ai_review_ui.model_provider", no_model)
    return SimpleNamespace(**vars(source), service=service, provider=provider,
                           document=document, version=version, bindings=bindings)


def test_real_workspace_hook_ignores_old_other_page_widgets_and_restores_sqlite_authority(
    case, monkeypatch,
):
    switches = []

    def logical_page_switch(target):
        # This from_string fixture has no pages/ directory. Model only the
        # router while keeping the actual workspace guard, component ACK and
        # SQLite services. The real browser covers the repository's pages.
        assert target in {LEARNING, "app.py"}
        switches.append(target)
        st.session_state["test_route"] = target
        st.rerun()

    monkeypatch.setattr(st, "switch_page", logical_page_switch)
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from src.workspace_ui import render_workspace\n"
        "route = st.session_state['test_route']\n"
        "render_workspace(route)\n"
        f"if route == {LEARNING!r}:\n"
        "    from src.study_loop.ui import render\n    render()\n"
        "else:\n    st.write('NON_REAL_OTHER_WORKSPACE_PAGE')\n"
    )
    app.session_state["test_route"] = LEARNING
    app.session_state["sl_open_root"] = str(case.store.root)
    app.session_state["sl_ai_review_focus"] = True
    app.run(timeout=30)
    identity = case.document["id"]
    assert app.session_state["sl_ai_review_document_id"] == identity
    assert callable(app.session_state["sl_review_route_guard"])
    app.radio(key="sl_ai_review_view_" + identity).set_value("编辑").run(timeout=30)
    draft = "NON_REAL_DURABLE_WORKING_REVIEW\n\n" + r"$q=\frac{a}{b}$；保留人工补充。"
    edit_color_body(app, "复盘正文", draft)
    persisted = case.service.document(identity)
    assert persisted["working"] == draft
    versions = case.service.versions(identity)
    original = case.store.db.read_bytes()

    for lose_binding in (False, True):
        app.session_state["test_route"] = "app.py"
        app.run(timeout=30)
        assert not app.exception and not app.error
        assert switches[:2] == [LEARNING, "app.py"]
        # This is replay from another page, not a text-area edit callback.
        app.session_state["sl_ai_review_body_" + identity] = "NON_REAL_OLD_AI_BODY"
        app.session_state["sl_ai_review_material"] = case.source.id
        app.session_state["sl_ai_review_question"] = case.selected.id
        app.run(timeout=30)
        assert not app.exception and not app.error
        assert case.service.document(identity)["working"] == draft
        assert case.service.document(identity)["working_revision"] == persisted["working_revision"]
        # A late client replay must also be replaced before learning widgets are instantiated.
        app.session_state["sl_ai_review_body_" + identity] = "NON_REAL_OLD_AI_BODY"
        app.session_state["sl_ai_review_loaded_" + identity] = "NON_REAL_OLD_AI_BODY"
        app.session_state["sl_ai_review_revision_" + identity] = 0
        app.session_state["sl_ai_review_material"] = case.source.id
        app.session_state["sl_ai_review_question"] = case.selected.id
        if lose_binding:
            app.session_state["sl_ai_review_document_id"] = None
            app.session_state["sl_ai_review_item_id"] = None
        app.session_state["test_route"] = LEARNING
        app.run(timeout=30)
        assert not app.exception and not app.error
        assert app.session_state["sl_ai_review_document_id"] == identity
        assert app.session_state["sl_ai_review_item_id"] == case.foreign.id
        assert app.selectbox(key="sl_ai_review_material").value == case.foreign.source_id
        assert app.session_state["sl_ai_review_question"] == case.foreign.id
        assert app.session_state["sl_ai_review_body_" + identity] == draft
        assert any(draft in value.value for value in app.markdown)
        assert case.service.document(identity)["working"] == draft
        assert case.service.document(identity)["working_revision"] == persisted["working_revision"]
        assert case.service.versions(identity) == versions
        # Replay must not create a conflict-recovery copy.
        assert len(case.service.documents()) == 1
    assert case.store.db.read_bytes() == original
    assert case.bindings == [] and len(case.provider.prompts) == 1
    assert not case.store.sessions() and not case.store.attempts()


def test_workspace_only_invokes_an_explicit_callable_route_guard(shell, monkeypatch):
    services, calls = [], []

    def no_review_service(*args, **kwargs):
        services.append(True)
        raise AssertionError("An unrelated workspace page must not initialize review storage")

    monkeypatch.setattr("src.study_loop.ai_review_ui.review_service", no_review_service)
    app = AppTest.from_string(
        "from src.workspace_ui import render_workspace\nrender_workspace('app.py')"
    ).run(timeout=30)
    assert not app.exception and not app.error and services == []
    app.session_state["sl_review_route_guard"] = "NON_REAL_NON_CALLABLE"
    app.run(timeout=30)
    assert not app.exception and not app.error and services == []
    app.session_state["sl_review_route_guard"] = calls.append
    app.run(timeout=30)
    assert not app.exception and not app.error
    assert calls == ["app.py"] and services == []


def test_nonfirst_memory_page_departure_preserves_latest_input_without_confirming_it(
    case, monkeypatch,
):
    """A pending memory ACK must remount that memory, rather than the default first row."""
    from src.study_loop.knowledge_memory import KnowledgeMemoryService

    memories = KnowledgeMemoryService(case.store)
    extraction = memories.extract_review(
        case.document["id"], RecordingProvider("NON_REAL_SELECTED_MEMORY"),
    )
    identity = memories.confirm(extraction["id"], "NON_REAL_SELECTED_MEMORY", confirmed=True)
    conversation = case.chats.open_course("NON_REAL_COURSE_A")
    case.chats.begin(conversation, "NON_REAL_OTHER", "NON_REAL_USER", {})
    case.chats.finish(conversation, "NON_REAL_OTHER", "NON_REAL_AI", {})
    candidate = case.chats.add_candidate(
        case.chats.archive(conversation), "NON_REAL_NEWER_DEFAULT_MEMORY",
    )
    other = case.chats.confirm(candidate, "NON_REAL_NEWER_DEFAULT_MEMORY", confirmed=True)
    switches = []

    def logical_page_switch(target):
        assert target in {LEARNING, "app.py"}
        switches.append(target)
        st.session_state["test_route"] = target
        st.rerun()

    monkeypatch.setattr(st, "switch_page", logical_page_switch)
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from src.workspace_ui import render_workspace\n"
        "route = st.session_state['test_route']\n"
        "render_workspace(route)\n"
        f"if route == {LEARNING!r}:\n"
        "    from src.study_loop.ui import render\n    render()\n"
        "else:\n    st.write('NON_REAL_OTHER_WORKSPACE_PAGE')\n"
    )
    app.session_state["test_route"] = LEARNING
    app.session_state["sl_open_root"] = str(case.store.root)
    app.session_state["sl_learning_view"] = "memory"
    app.run(timeout=30)
    assert app.selectbox(key="sl_memory_home_selection").value == other
    app.selectbox(key="sl_memory_home_selection").set_value(identity).run(timeout=30)
    next(button for button in app.button if button.label == "编辑记忆").click().run(timeout=30)
    editor = color_body(app, "记忆正文")
    intended = '<span style="color:#2563EB">NON_REAL_PENDING_MEMORY $q=a/b$</span>'
    editor.set_value(intended)
    before = case.chats.memory(identity)
    other_before = case.chats.memory(other)
    revisions = case.chats.memory_revisions(identity)
    app.session_state["test_route"] = "app.py"
    app.run(timeout=30)
    assert not app.exception and not app.error
    assert switches == [LEARNING, "app.py"]
    assert app.session_state["test_route"] == "app.py"
    assert any("NON_REAL_OTHER_WORKSPACE_PAGE" in value.value for value in app.markdown)
    assert case.chats.memory(identity) == before
    assert case.chats.memory(other) == other_before
    assert case.chats.memory_revisions(identity) == revisions
    app.session_state["test_route"] = LEARNING
    app.run(timeout=30)
    assert not app.exception and not app.error
    app.selectbox(key="sl_memory_home_selection").set_value(identity).run(timeout=30)
    assert color_body(app, "记忆正文").value == intended
    assert case.chats.memory(identity) == before
    assert case.chats.memory_revisions(identity) == revisions
    assert case.bindings == [] and len(case.provider.prompts) == 1
    assert not case.store.sessions() and not case.store.attempts()
