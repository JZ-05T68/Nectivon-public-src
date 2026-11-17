"""Non-learning transport/storage fixtures, never evidence of learning quality."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.study_loop import components, model_review_records
from src.study_loop.model_review import parse_answer
from src.study_loop.models import StudyError
from src.study_loop.storage import StudyStore
from src.study_loop.worker import registry


@pytest.fixture
def store(tmp_path: Path) -> StudyStore:
    return StudyStore(tmp_path / "isolated-study", create=True)


def test_failed_import_preserves_original_and_explicit_resume_is_idempotent(
    store: StudyStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure after page publication must not duplicate pages on explicit retry."""
    from nectivon_tasks import Queue, Worker

    source = store.receive("parser-fixture.txt", b"Non-learning parser fixture.\n", "txt")
    tasks = Queue(store.path("tasks.sqlite3"), max_active=1)
    original = components.import_source
    calls = 0

    def fail_after_import(current: StudyStore, source_id: str) -> dict:
        nonlocal calls
        calls += 1
        result = original(current, source_id)
        if calls == 1:
            raise OSError("injected local I/O failure")
        return result

    monkeypatch.setattr(components, "import_source", fail_after_import)
    handlers = registry(store.root)
    task = handlers.submit(
        tasks, "study_import", 1, {"source_id": source.id},
        idempotency_key=source.id, total=1,
    )
    worker = Worker(tasks, handlers)
    assert worker.run_once()
    assert tasks.get(task.id).state == "failed"
    pages, drafts = store.pages(source.id), store.drafts()
    assert store.read_source(source) == b"Non-learning parser fixture.\n"
    assert not worker.run_once()  # Failure is never automatically retried.
    tasks.resume(task.id)
    assert worker.run_once()
    assert tasks.get(task.id).state == "completed"
    assert store.pages(source.id) == pages and store.drafts() == drafts
    assert calls == 2
    duplicate = handlers.submit(
        tasks, "study_import", 1, {"source_id": source.id},
        idempotency_key=source.id, total=1,
    )
    assert duplicate.id == task.id
    assert not worker.run_once()


def test_cancelled_queued_import_never_runs_or_deletes_original(store: StudyStore) -> None:
    from nectivon_tasks import Queue, Worker

    source = store.receive("cancel-fixture.txt", b"Non-learning cancellation fixture.", "txt")
    tasks = Queue(store.path("tasks.sqlite3"), max_active=1)
    handlers = registry(store.root)
    task = handlers.submit(tasks, "study_import", 1, {"source_id": source.id})
    tasks.request_cancel(task.id)
    assert not Worker(tasks, handlers).run_once()
    assert tasks.get(task.id).state == "cancelled"
    assert store.read_source(source) == b"Non-learning cancellation fixture."
    assert store.pages(source.id) == []


def test_raw_record_publication_failure_retains_recoverable_bytes(
    store: StudyStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transport diagnostic bytes survive a failed no-overwrite publication."""
    run = "1" * 32
    record = {"text": "Non-learning transport token: \ud800", "status": "unstructured"}

    def deny_link(*args, **kwargs) -> None:
        raise OSError("injected publication failure")

    monkeypatch.setattr(model_review_records.os, "link", deny_link)
    with pytest.raises(StudyError, match="暂存副本保留"):
        model_review_records.save_record(store, run, "analysis_0_raw", record)
    assert model_review_records.load_record(store, run, "analysis_0_raw") is None
    names = model_review_records.pending_record_names(store, run)
    assert len(names) == 1
    temporary = store.path(f"model-reviews/{run}/{names[0]}")
    import json

    assert json.loads(temporary.read_text(encoding="ascii")) == record


@pytest.mark.parametrize("text", ["", "[]", '{"method":"first","method":"second"}', "null"])
def test_unstructured_transport_tokens_are_not_promoted(text: str) -> None:
    assert parse_answer(text) is None
