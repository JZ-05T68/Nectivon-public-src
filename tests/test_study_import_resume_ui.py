"""NON_REAL_TEST_DATA: resuming a local batch cannot dispatch unrelated queued model work."""

from __future__ import annotations

from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from src.study_loop import review_ui


def test_resume_local_batch_restarts_only_that_batch_and_restores_progress(tmp_path, monkeypatch):
    actions = []
    identity = "a" * 32
    task = SimpleNamespace(
        id=identity, state="paused", kind="study_import_batch", created_at=1,
        result=None, error_message="",
    )
    tasks = SimpleNamespace(
        list=lambda **kwargs: [task],
        resume=lambda selected: actions.append(("resume", selected)),
    )
    monkeypatch.setattr(review_ui, "queue", lambda root: tasks)
    monkeypatch.setattr(
        review_ui, "launch_once",
        lambda root, **options: actions.append(("launch", options)),
    )
    monkeypatch.setattr(review_ui, "_test_store", SimpleNamespace(root=tmp_path), raising=False)
    app = AppTest.from_string(
        "from src.study_loop.review_ui import render_tasks, _test_store\n"
        "render_tasks(_test_store)\n"
    ).run(timeout=30)
    assert not app.exception and actions == []
    app.button(key="resume_" + identity).click().run(timeout=30)
    assert not app.exception
    assert actions == [("resume", identity),
                       ("launch", {"kind": "study_import_batch", "task_id": identity})]
    assert app.session_state["sl_import_batch_task_id"] == identity
    app.run(timeout=30)
    assert len(actions) == 2
