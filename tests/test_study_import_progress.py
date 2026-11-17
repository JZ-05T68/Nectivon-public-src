"""Synthetic AppTest coverage for finite, read-only local import refresh."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from src.study_loop import import_progress_ui as progress_ui


def _task(state="running", *, kind="study_import_batch", result=None):
    return SimpleNamespace(state=state, kind=kind, done=1, total=3, result=result)


class LocalStatusOnly:
    """The UI may only read its selected task, never list or dispatch other tasks."""

    def __init__(self, states):
        self.states = list(states)
        self.reads = []

    def get(self, identity):
        assert identity == "NON_REAL_IMPORT_TASK"
        self.reads.append(identity)
        if len(self.states) > 1:
            return self.states.pop(0)
        return self.states[0]

    def __getattr__(self, name):
        pytest.fail("Progress must not dispatch, resume, claim or list tasks: " + name)


@pytest.fixture
def ui_case(monkeypatch):
    store = SimpleNamespace(
        root=Path("NON_REAL_UNUSED_ROOT"),
        source=lambda identity: SimpleNamespace(
            name={"failed-file": "未完成的材料.docx"}[identity],
        ),
    )
    monkeypatch.setattr(progress_ui, "_test_store", store, raising=False)

    def app(*, tracking=True):
        instance = AppTest.from_string(
            "import streamlit as st\n"
            "from src.study_loop.import_progress_ui import render_import_progress, _test_store\n"
            "st.session_state['page_renders'] = st.session_state.get('page_renders',0)+1\n"
            "render_import_progress(_test_store)\n"
            "st.text('MAIN_MATERIALS_VISIBLE')\n"
        )
        if tracking:
            instance.session_state[progress_ui.IMPORT_TASK_KEY] = "NON_REAL_IMPORT_TASK"
        return instance

    return app


def test_idle_page_has_no_status_reads_or_fragment_timer(ui_case, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Idle page must not poll, dispatch or create a timer")

    monkeypatch.setattr(progress_ui, "queue", forbidden)
    monkeypatch.setattr(progress_ui.st, "fragment", forbidden)
    app = ui_case(tracking=False).run()
    assert not app.exception and not app.get("progress")
    assert app.session_state["page_renders"] == 1


def test_pending_import_uses_one_local_only_fragment(ui_case, monkeypatch):
    tasks = LocalStatusOnly([_task()])
    monkeypatch.setattr(progress_ui, "queue", lambda root: tasks)
    app = ui_case().run()
    assert not app.exception
    assert len(app.get("progress")) == 1
    assert "正在导入学习材料" in app.get("progress")[0].proto.text
    assert app.session_state[progress_ui.IMPORT_TASK_KEY] == "NON_REAL_IMPORT_TASK"
    assert app.session_state["page_renders"] == 1
    assert len(tasks.reads) == 2  # Initial probe and first fragment body, no paid task enumeration.


def test_terminal_fragment_refreshes_materials_once_then_removes_tracking(ui_case, monkeypatch):
    tasks = LocalStatusOnly([_task(), _task("completed", result={"file_count": 3})])
    monkeypatch.setattr(progress_ui, "queue", lambda root: tasks)
    app = ui_case().run()
    assert not app.exception and not app.get("progress")
    assert app.session_state["page_renders"] == 2
    assert len(app.success) == 1 and app.success[0].value == "材料已导入。"
    assert progress_ui.IMPORT_TASK_KEY not in app.session_state
    assert len(tasks.reads) == 2
    app.run()
    assert not app.exception and not app.success and not app.get("progress")
    assert app.session_state["page_renders"] == 3 and len(tasks.reads) == 2


def test_partial_import_failure_names_are_readable_and_shown_once(ui_case, monkeypatch):
    result = {"file_count": 3, "failed_file_count": 1,
              "failed_files": [{"source_id": "failed-file", "error_type": "SECRET_ERROR_CODE"}]}
    tasks = LocalStatusOnly([_task("completed", result=result)])
    monkeypatch.setattr(progress_ui, "queue", lambda root: tasks)
    app = ui_case().run()
    assert not app.exception and not app.get("progress")
    assert "已导入 2 份材料；1 份未能导入。" in app.warning[0].value
    visible = " ".join(element.value for element in [*app.warning, *app.text])
    assert "未完成的材料.docx" in visible
    assert "failed-file" not in visible and "SECRET_ERROR_CODE" not in visible
    assert app.session_state["page_renders"] == 2
    app.run()
    assert not app.warning and len(tasks.reads) == 1


@pytest.mark.parametrize("state", ["failed", "cancelled", "paused", "interrupted"])
def test_stopped_import_has_no_remaining_timer(ui_case, monkeypatch, state):
    tasks = LocalStatusOnly([_task(state)])
    monkeypatch.setattr(progress_ui, "queue", lambda root: tasks)
    app = ui_case().run()
    assert not app.exception and not app.get("progress")
    assert progress_ui.IMPORT_TASK_KEY not in app.session_state
    assert app.session_state["page_renders"] == 2
    assert len(tasks.reads) == 1


def test_wrong_task_kind_is_never_polled_as_a_model_job(ui_case, monkeypatch):
    tasks = LocalStatusOnly([_task(kind="study_model_review")])
    monkeypatch.setattr(progress_ui, "queue", lambda root: tasks)
    app = ui_case().run()
    assert not app.exception and not app.get("progress")
    assert "导入状态暂时无法读取" in app.warning[0].value
    assert progress_ui.IMPORT_TASK_KEY not in app.session_state
    assert len(tasks.reads) == 1


def test_expired_public_lease_stops_refresh_without_mutating_the_task(ui_case, monkeypatch):
    expired = _task()
    expired.lease_until = 1.0
    tasks = LocalStatusOnly([expired])
    monkeypatch.setattr(progress_ui, "queue", lambda root: tasks)

    def forbidden(*args, **kwargs):
        pytest.fail("Expired worker must not retain a polling timer")

    monkeypatch.setattr(progress_ui.st, "fragment", forbidden)
    app = ui_case().run()
    assert not app.exception and not app.get("progress")
    assert "导入已中断" in app.warning[0].value
    assert progress_ui.IMPORT_TASK_KEY not in app.session_state
    assert len(tasks.reads) == 1 and expired.state == "running"
    app.run()
    assert not app.warning and len(tasks.reads) == 1
