"""Colored memory titles stay readable while source bodies and history remain unchanged."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest
from test_ai_review_support import review_case

from src.study_loop.memory_editor_ui import _prefix


@pytest.fixture
def case(tmp_path):
    source = review_case(tmp_path / "NON_REAL_COLORED_LABELS")
    identity = source.chats.open_course("NON_REAL_COURSE_A", new=True)
    source.chats.begin(identity, "NON_REAL_SEED", "NON_REAL_SEED_USER", {})
    source.chats.finish(identity, "NON_REAL_SEED", "NON_REAL_SEED_AI", {})
    archive = source.chats.archive(identity)
    # A legacy 60-character trash label cuts through this complete, valid span tag.
    red = '# <span' + ' ' * 75 + 'style="color:#D62828">红色记忆（含公式）</span>\n\n$P=UI$'
    blue = '# <span style="color:#2563EB">蓝色记忆</span>\n\n限制条件。'
    candidate = source.chats.add_candidate(archive, red)
    memory = source.chats.confirm(candidate, red, confirmed=True)
    other = source.chats.confirm(candidate, blue, confirmed=True)
    return SimpleNamespace(**vars(source), identity=identity, candidate=candidate,
                           memory=memory, other=other, red=red, blue=blue)


def _app(case, entry):
    app = AppTest.from_string(
        "from pathlib import Path\nimport streamlit as st\n"
        "from src.study_loop.storage import StudyStore\n"
        "from src.study_loop.chat_store import ChatStore\n"
        "store = StudyStore(Path(st.session_state['test_root']))\n"
        "chats = ChatStore(store)\n"
        "entry = st.session_state['test_entry']\n"
        "if entry == 'chat':\n"
        "    from src.study_loop.chat_ui import _memory\n"
        "    _memory(chats, st.session_state['test_conversation'], None)\n"
        "elif entry == 'management':\n"
        "    from src.study_loop.chat_ui import render_saved_management\n"
        "    render_saved_management(store)\n"
        "elif entry == 'trash':\n"
        "    from src.study_loop.trash_ui import render_classified_trash\n"
        "    render_classified_trash(store, key_prefix='test_trash')\n"
        "else:\n"
        "    from src.study_loop.memory_editor_ui import render_memory_editor\n"
        "    render_memory_editor(store, st.session_state['test_memory'], "
        "key_prefix='test_history')"
    )
    app.session_state["test_root"] = str(case.store.root)
    app.session_state["test_entry"] = entry
    app.session_state["test_conversation"] = case.identity
    app.session_state["test_memory"] = case.memory
    return app.run(timeout=30)


@pytest.mark.parametrize("entry", ["chat", "management"])
def test_confirmed_memory_lists_keep_plain_titles_and_unchanged_colored_bodies(case, entry):
    before = copy.deepcopy(case.chats.snapshot(case.identity))
    memories = [copy.deepcopy(case.chats.memory(identity))
                for identity in (case.memory, case.other)]
    app = _app(case, entry)
    assert not app.exception and not app.error
    labels = [expander.label for expander in app.expander]
    labels += [option for select in app.selectbox for option in select.options]
    assert any("红色记忆（含公式）" in label for label in labels)
    assert any("蓝色记忆" in label for label in labels)
    assert not any("<span" in label or "</span>" in label or "style=" in label for label in labels)
    app.run(timeout=30)
    assert not app.exception and not app.error
    assert [case.chats.memory(identity) for identity in (case.memory, case.other)] == memories
    assert case.chats.snapshot(case.identity) == before


def test_memory_and_candidate_trash_titles_read_full_body_before_shortening(case):
    case.chats.delete("memory", case.memory, confirmed=True)
    case.chats.delete("candidate", case.candidate, confirmed=True)
    original_labels = [row["label"] for row in case.chats.trash()]
    assert any("<span" in label and "</span>" not in label for label in original_labels)
    before = case.chats.db.read_bytes()
    app = _app(case, "trash")
    for category in ("memory", "candidate"):
        app.selectbox(key="test_trash_category").set_value(category).run(timeout=30)
        assert not app.exception and not app.error
        options = app.selectbox(key="test_trash_selection_" + category).options
        assert any("红色记忆（含公式）" in option for option in options)
        assert not any("<span" in option or "style=" in option for option in options)
    assert case.chats.memory(case.memory)["content"] == case.red
    assert case.chats.db.read_bytes() == before


def test_memory_advanced_open_state_survives_history_confirmation_and_restore_reruns(case):
    case.chats.update_memory(case.memory, case.blue + "\n另一修订。", expected_revision=1)
    app = _app(case, "editor")
    prefix = _prefix(case.store, case.memory, "test_history")
    advanced = next(expander for expander in app.expander if expander.label == "高级设置")
    identity = advanced.proto.id
    assert identity

    def rerun_with_expander(opened=True):
        # AppTest models expanders as blocks, so its run() omits the browser's Boolean
        # widget state. Send that real frontend payload along with the selected action.
        states = app._tree.get_widget_states()
        assert identity not in {state.id for state in states.widgets}
        state = states.widgets.add()
        state.id = identity
        state.bool_value = opened
        app._run(widget_state=states, timeout=30)

    rerun_with_expander()
    assert next(value for value in app.expander if value.label == "高级设置").proto.expanded
    # A negative control proves this is the browser's actual open/close state,
    # rather than a test that forces the product to remain permanently expanded.
    rerun_with_expander(False)
    assert not next(value for value in app.expander if value.label == "高级设置").proto.expanded
    rerun_with_expander()
    assert next(value for value in app.expander if value.label == "高级设置").proto.expanded
    app.selectbox(key=prefix + "_history").set_value(1)
    rerun_with_expander()
    app.checkbox(key=prefix + "_delete_confirm").check()
    rerun_with_expander()
    app.button(key=prefix + "_restore_version").click()
    rerun_with_expander()
    assert not app.exception and not app.error
    advanced = next(expander for expander in app.expander if expander.label == "高级设置")
    assert advanced.proto.id == identity and advanced.proto.expanded
    memory = case.chats.memory(case.memory)
    assert memory["revision"] == 3 and memory["content"] == case.red and not memory["deleted"]
    assert app.selectbox(key=prefix + "_history").value == 1
    assert app.checkbox(key=prefix + "_delete_confirm").value
