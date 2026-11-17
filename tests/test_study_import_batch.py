"""Finite mixed import execution, with no model dispatch or real user materials."""

from __future__ import annotations

import pytest

from src.study_loop import chat, runtime
from src.study_loop.chat_store import ChatStore
from src.study_loop.models import StudyError
from src.study_loop.storage import StudyStore
from src.study_loop.worker import registry, run_import_batch


@pytest.fixture
def batch_case(tmp_path, monkeypatch):
    monkeypatch.setenv("EKB_STUDY_LOOP_ACCEPTANCE_CONFIRMED", "yes")
    study = StudyStore(tmp_path / "study", create=True)
    sources = [study.receive(f"NON_REAL_{i}.txt", f"1. NON_REAL_{i}".encode(), "txt")
               for i in range(3)]
    tasks = runtime.queue(study.root)
    return study, sources, tasks


def test_one_batch_imports_all_selected_sources_without_claiming_earlier_paid_task(batch_case):
    study, sources, tasks = batch_case
    unrelated = tasks.submit("study_model_review", 1, {"NON_REAL": True})
    batch = registry(study.root).submit(
        tasks, "study_import_batch", 1, {"source_ids": [s.id for s in sources]}, total=3,
    )
    run_import_batch(tasks, study.root, batch.id)
    assert tasks.get(batch.id).state == "completed"
    assert tasks.get(batch.id).result == {"file_count": 3, "draft_count": 3}
    assert len(study.drafts()) == 3
    assert tasks.get(unrelated.id).state == "queued"


def test_busy_queue_is_rejected_before_creating_a_stranded_batch(batch_case, monkeypatch):
    study, sources, tasks = batch_case
    active = tasks.submit("NON_REAL_TASK", 1, {})
    tasks.claim("NON_REAL_OWNER")
    with pytest.raises(StudyError, match="已有任务正在处理"):
        runtime.start_import_batch(study.root, [s.id for s in sources])
    assert [task.id for task in tasks.list()] == [active.id]


def test_failed_file_does_not_prevent_remaining_local_imports(batch_case, monkeypatch, caplog):
    from src.study_loop import components

    study, sources, tasks = batch_case
    original = components.import_source
    calls = []

    def local_import(store, source_id):
        calls.append(source_id)
        if source_id == sources[1].id:
            raise RuntimeError("SENSITIVE_FILE_PARSER_DETAIL")
        return original(store, source_id)

    monkeypatch.setattr(components, "import_source", local_import)
    batch = registry(study.root).submit(
        tasks, "study_import_batch", 1, {"source_ids": [s.id for s in sources]}, total=3,
    )
    run_import_batch(tasks, study.root, batch.id)
    finished = tasks.get(batch.id)
    assert finished.state == "completed"
    assert finished.done == finished.total == 3
    assert calls == [s.id for s in sources]
    assert finished.result["draft_count"] == 2
    assert finished.result["failed_file_count"] == 1
    assert finished.result["failed_files"][0]["source_id"] == sources[1].id
    assert finished.result["failed_files"][0]["error_type"] == "RuntimeError"
    assert "SENSITIVE_FILE_PARSER_DETAIL" not in str(finished.result) + caplog.text
    assert {draft.source_id for draft in study.drafts()} == {sources[0].id, sources[2].id}
    assert study.read_source(sources[1]) == b"1. NON_REAL_1"


def test_batch_wait_expiry_cancels_only_its_unclaimed_local_task(batch_case):
    study, sources, tasks = batch_case
    unrelated = tasks.submit("NON_REAL_TASK", 1, {})
    tasks.claim("NON_REAL_OWNER")
    batch = registry(study.root).submit(
        tasks, "study_import_batch", 1, {"source_ids": [s.id for s in sources]}, total=3,
    )
    run_import_batch(tasks, study.root, batch.id, timeout_seconds=0)
    assert tasks.get(batch.id).state == "cancelled"
    assert tasks.get(unrelated.id).state == "running"
    assert study.drafts() == []


def test_start_import_batch_launches_exactly_one_restricted_worker(batch_case, monkeypatch):
    study, sources, tasks = batch_case
    calls = []
    monkeypatch.setattr(
        runtime, "launch_once", lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    batch = runtime.start_import_batch(study.root, [s.id for s in sources])
    assert batch.kind == "study_import_batch"
    assert len(calls) == 1
    assert calls[0][1] == {"kind": "study_import_batch", "task_id": batch.id}
    assert tasks.get(batch.id).payload["source_ids"] == [s.id for s in sources]


@pytest.mark.parametrize("kind", ["study_import", "study_verify", "study_export"])
def test_resumed_local_task_cannot_process_a_source_deleted_while_paused(
    batch_case, monkeypatch, kind,
):
    from nectivon_tasks import Worker

    from src.study_loop import components

    study, sources, tasks = batch_case
    components.import_source(study, sources[0].id)
    item = study.drafts()[0]
    study.confirm(item, user_confirmed=True)
    payload = {"source_id": sources[0].id} if kind == "study_import" else {"item_ids": [item.id]}
    if kind == "study_export":
        payload["run_id"] = "a" * 32
    task = registry(study.root).submit(tasks, kind, 1, payload, total=1)
    tasks.request_pause(task.id)
    assert tasks.get(task.id).state == "paused"
    study.delete_source(sources[0].id, user_confirmed=True)
    tasks.resume(task.id)

    def forbidden(*args, **kwargs):
        pytest.fail("Deleted source reached a resumed local handler")

    for operation in ("import_source", "verify_sources", "export_review"):
        monkeypatch.setattr(components, operation, forbidden)
    assert Worker(tasks, registry(study.root)).run_once()
    assert tasks.get(task.id).state == "failed"
    assert study.read_source(sources[0]) == b"1. NON_REAL_0"
    assert list(study.path("exports").iterdir()) == []


@pytest.mark.parametrize("operation", ["send", "propose"])
def test_deleted_memory_in_existing_history_blocks_reuse_without_rewriting_it(
    batch_case, operation,
):
    study, sources, _ = batch_case
    chats = ChatStore(study)
    chats.set_course("NON_REAL_COURSE", [s.id for s in sources], "")
    identity = chats.open_course("NON_REAL_COURSE")
    chats.begin(identity, "initial", "NON_REAL_QUERY", {})
    chats.finish(identity, "initial", "NON_REAL_ANSWER", {})
    candidate = chats.add_candidate(chats.archive(identity), "NON_REAL_MEMORY")
    memory_id = chats.confirm(candidate, "NON_REAL_MEMORY", confirmed=True)
    chats.begin(identity, "memory-round", "NON_REAL_QUERY", {"evidence": [{
        "label": "M" + memory_id[:12], "text": "NON_REAL_MEMORY", "memory_links": [],
    }]})
    chats.finish(identity, "memory-round", "NON_REAL_ANSWER", {})
    before = chats.messages(identity)
    chats.delete("memory", memory_id, confirmed=True)

    class Forbidden:
        def complete(self, *args, **kwargs):
            pytest.fail("Removed memory reached a model")

    with pytest.raises(StudyError, match="已删除的记忆"):
        if operation == "send":
            chat.send(chats, identity, "NON_REAL_QUERY", "next", Forbidden(), allow_remote=True)
        else:
            chat.propose(chats, identity, Forbidden(), allow_remote=True)
    assert chats.messages(identity) == before
