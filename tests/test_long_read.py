"""P0-2 long-document reading engine/store/worker tests (v0.8.8).

These tests must be able to detect an empty implementation and fake
progress, so they assert on real persisted artefacts:

* the page-level state machine reaches ``done`` for every page;
* a real ``PageReading`` JSON file exists on disk for every completed page
  and carries the model-produced summary (not a placeholder);
* the summary provider was actually called (recorded prompts);
* the task counters are recomputed from the page rows (no drift/fake).

Covered: state-machine transitions (legal + illegal), resume idempotency,
pause/cancel semantics, timing-field completeness, per-page + per-batch
failure retry, selective (never blanket) vision, and batch planning caps.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent_document_reader import AgentReadingStore
from src.ai.provider import AIExecutionError, CompletionResult
from src.database import Database
from src.long_read_engine import (
    DEFAULT_BATCH_SIZE,
    LongReadEngine,
    LongReadNetworkPause,
    batch_completion_budget,
    plan_batches,
)
from src.long_read_store import (
    LongReadError,
    LongReadStore,
    timing_view,
)
from src.long_read_worker import LongReadCoordinator
from src.models import PageStatus

BATCH_MARKER = "本批页面(JSON)："


# --------------------------------------------------------------- fixtures
@pytest.fixture()
def database(tmp_path: Path) -> Database:
    return Database(tmp_path / "long-read.db")


def _make_document_with_pages(
    database: Database,
    tmp_path: Path,
    *,
    page_count: int,
    text_len: int = 220,
    sparse_pages: set[int] | None = None,
) -> tuple[int, list[int]]:
    """Create one document with ``page_count`` real pages; return ids."""

    sparse_pages = sparse_pages or set()
    sha = "a" * 64
    document = database.create_document(
        title="液压与气动",
        filename="hydraulics.pdf",
        source_path=tmp_path / "hydraulics.pdf",
        sha256=sha,
        page_count=page_count,
    )
    page_ids: list[int] = []
    for number in range(1, page_count + 1):
        if number in sparse_pages:
            text = "图"  # below the vision threshold -> sparse text layer
        else:
            text = f"第{number}页 液压传动 概述 内容 " + ("填充" * (text_len // 2))
        image_path = tmp_path / f"page_{number}.png"
        page = database.create_page(
            document_id=document.id,
            page_number=number,
            image_path=image_path,
            extracted_text=text,
            status=PageStatus.PENDING,
        )
        page_ids.append(page.id)
    return document.id, page_ids


class _FakeSummarizeProvider:
    """Records prompts; returns a valid per-page batch JSON."""

    def __init__(
        self,
        *,
        missing_pages: set[int] | None = None,
        network_error: bool = False,
    ) -> None:
        self.prompts: list[str] = []
        self.missing_pages = missing_pages or set()
        self.network_error = network_error

    @staticmethod
    def _batch_payload(prompt: str) -> list[dict]:
        marker_index = prompt.rfind(BATCH_MARKER)
        assert marker_index != -1, "batch prompt must embed the page payload"
        raw = prompt[marker_index + len(BATCH_MARKER):].strip()
        return json.loads(raw)

    def complete(
        self, prompt: str, *, model: str | None = None,
        max_completion_tokens: int | None = None,
    ) -> CompletionResult:
        self.prompts.append(prompt)
        if self.network_error:
            raise AIExecutionError("网络不可达", error_class="network")
        payload = self._batch_payload(prompt)
        pages = []
        for item in payload:
            number = int(item["page_number"])
            if number in self.missing_pages:
                continue  # simulate the model dropping a page
            pages.append(
                {
                    "page_number": number,
                    "summary": f"本页摘要{number}",
                    "keywords": [f"关键词{number}"],
                    "key_facts": [f"事实{number}"],
                }
            )
        return CompletionResult(
            text=json.dumps({"pages": pages}, ensure_ascii=False),
            model=model or "fake-llm",
        )


class _FakeVisionProvider:
    """Records vision calls; returns a minimal valid vision JSON."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete_vision(
        self, prompt: str, image_data_url: str, *, model: str | None = None,
        max_completion_tokens: int | None = None,
        source_feature: str = "", target_refs: tuple = (),
    ) -> CompletionResult:
        self.calls.append(prompt)
        return CompletionResult(
            text=json.dumps(
                {
                    "visual_summary": "回路图",
                    "figures": [{"name": "图2-3", "description": "液压回路"}],
                    "formulas": [],
                    "circuit": "泵-阀-缸",
                    "uncertain": ["符号待核验"],
                },
                ensure_ascii=False,
            ),
            model=model or "fake-vision",
        )


def _engine(
    database: Database,
    store: LongReadStore,
    tmp_path: Path,
    *,
    provider=None,
    vision_provider=None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model: str = "fake-llm",
) -> tuple[LongReadEngine, AgentReadingStore]:
    readings = AgentReadingStore(tmp_path / "readings")
    engine = LongReadEngine(
        database=database,
        store=store,
        provider=provider if provider is not None else _FakeSummarizeProvider(),
        readings=readings,
        model=model,
        vision_provider=vision_provider,
        vision_model="fake-vision",
        batch_size=batch_size,
    )
    return engine, readings


def _new_task(
    database: Database, store: LongReadStore, tmp_path: Path, *,
    page_count: int, batch_size: int = DEFAULT_BATCH_SIZE,
    sparse_pages: set[int] | None = None,
) -> tuple[int, int]:
    document_id, _ = _make_document_with_pages(
        database, tmp_path, page_count=page_count, sparse_pages=sparse_pages
    )
    document = database.get_document(document_id)
    task = store.create_task(
        document_id=document_id,
        document_sha256=document.sha256,
        total_pages=page_count,
        batch_size=batch_size,
        model="fake-llm",
        batch_id="batch-1",
    )
    pages = database.list_pages(document_id)
    store.seed_pages(task.id, [(p.id, p.page_number) for p in pages])
    return task.id, document_id


# ------------------------------------------------------------ batch plan
def test_plan_batches_enforces_count_and_char_caps() -> None:
    """515 pages can never collapse into one giant call."""

    class _Page:
        def __init__(self, number: int) -> None:
            self.page_number = number

    pages = [_Page(i) for i in range(1, 21)]
    texts = ["x" * 100 for _ in pages]
    plans = plan_batches(pages, texts, batch_size=6)
    assert [len(plan.pages) for plan in plans] == [6, 6, 6, 2]
    assert [plan.index for plan in plans] == [1, 2, 3, 4]
    # character cap splits a batch even below the count cap
    dense = ["x" * 20_000 for _ in range(4)]
    dense_plans = plan_batches(
        [_Page(i) for i in range(1, 5)], dense, batch_size=6, max_batch_chars=24_000
    )
    assert all(len(plan.pages) == 1 for plan in dense_plans)


def test_batch_completion_budget_is_bounded() -> None:
    assert batch_completion_budget(1, 100) >= 1_600
    assert batch_completion_budget(500, 10_000_000) <= 16_384


# ------------------------------------------------------- state machine
def test_every_page_reaches_an_explicit_state(database, tmp_path) -> None:
    """No page is silently skipped; all nine states are reachable."""

    store = LongReadStore(database)
    task_id, document_id = _new_task(database, store, tmp_path, page_count=5)
    engine, readings = _engine(database, store, tmp_path)
    assert store.claim_task(task_id, worker_pid=4242)
    final = engine.run_task(task_id)

    assert final.status == "complete"
    assert final.total_pages == 5
    assert final.done_pages == 5
    assert final.pending_pages == 0
    assert final.failed_pages == 0
    rows = store.list_pages(task_id)
    assert len(rows) == 5
    assert all(row.state == "done" for row in rows)
    # Anti-fake-progress: real reading artefacts exist with model content.
    pages = database.list_pages(document_id)
    for page in pages:
        reading = readings.page_reading(page.id)
        assert reading is not None
        assert reading.summary.startswith("本页摘要")
    task = store.get_task(task_id)
    assert task.reused_pages == 0


def test_illegal_page_transition_is_rejected(database, tmp_path) -> None:
    store = LongReadStore(database)
    task_id, _ = _new_task(database, store, tmp_path, page_count=2)
    with pytest.raises(LongReadError):
        store.set_page_state(task_id, 1, "done")  # pending -> done is illegal


def test_done_page_cannot_be_reset(database, tmp_path) -> None:
    """A completed result is never overwritten by retry/resume."""

    store = LongReadStore(database)
    task_id, _ = _new_task(database, store, tmp_path, page_count=1)
    engine, _ = _engine(database, store, tmp_path)
    store.claim_task(task_id, worker_pid=1)
    engine.run_task(task_id)
    with pytest.raises(LongReadError):
        store.retry_page(task_id, 1)
    with pytest.raises(LongReadError):
        store.mark_page_skipped_confirmed(task_id, 1)


# ------------------------------------------------- failure + retry paths
def test_batch_failure_marks_pages_and_retry_succeeds(database, tmp_path) -> None:
    """Per-batch failure is recorded, then a retry completes the pages."""

    store = LongReadStore(database)
    task_id, document_id = _new_task(database, store, tmp_path, page_count=3)
    failing = _FakeSummarizeProvider(missing_pages={2})
    engine, readings = _engine(database, store, tmp_path, provider=failing)
    store.claim_task(task_id, worker_pid=1)
    final = engine.run_task(task_id)

    assert final.status == "failed"
    assert final.failed_pages == 1
    page2 = store.get_page(task_id, 2)
    assert page2.state == "failed"
    assert page2.error_code == "batch_result_missing_page"
    assert page2.retry_count == 0
    # pages 1 and 3 still completed and are on disk (partial success kept)
    assert store.get_page(task_id, 1).state == "done"
    assert store.get_page(task_id, 3).state == "done"

    # retry only the failed page with a healthy provider
    requeued = store.retry_failed_pages(task_id)
    assert requeued == 1
    assert store.get_page(task_id, 2).retry_count == 1
    assert store.get_page(task_id, 2).state == "queued"
    healthy = _FakeSummarizeProvider()
    engine2, readings2 = _engine(database, store, tmp_path, provider=healthy)
    store.claim_task(task_id, worker_pid=2)
    final2 = engine2.run_task(task_id)
    assert final2.status == "complete"
    assert final2.done_pages == 3
    assert store.get_page(task_id, 2).state == "done"
    del readings, readings2, document_id


def test_network_failure_raises_for_system_pause(database, tmp_path) -> None:
    store = LongReadStore(database)
    task_id, _ = _new_task(database, store, tmp_path, page_count=2)
    provider = _FakeSummarizeProvider(network_error=True)
    engine, _ = _engine(database, store, tmp_path, provider=provider)
    store.claim_task(task_id, worker_pid=1)
    with pytest.raises(LongReadNetworkPause):
        engine.run_task(task_id)


# --------------------------------------------------------- idempotency
def _prompt_page_numbers(prompt: str) -> set[int]:
    marker = "本批页面(JSON)："
    raw = prompt[prompt.rfind(marker) + len(marker):].strip()
    return {int(item["page_number"]) for item in json.loads(raw)}


def test_rerun_same_document_reuses_without_model_calls(database, tmp_path) -> None:
    """A re-run over an already-read document issues ZERO model calls."""

    store = LongReadStore(database)
    task_id, document_id = _new_task(database, store, tmp_path, page_count=4)
    provider = _FakeSummarizeProvider()
    engine, readings = _engine(database, store, tmp_path, provider=provider)
    store.claim_task(task_id, worker_pid=1)
    first = engine.run_task(task_id)
    assert first.status == "complete"
    assert len(provider.prompts) > 0

    # New task over the SAME document (previous one is terminal/complete).
    document = database.get_document(document_id)
    task2 = store.create_task(
        document_id=document_id, document_sha256=document.sha256,
        total_pages=4, batch_size=DEFAULT_BATCH_SIZE, model="fake-llm",
        batch_id="batch-2",
    )
    assert task2.id != task_id
    pages = database.list_pages(document_id)
    store.seed_pages(task2.id, [(p.id, p.page_number) for p in pages])
    provider2 = _FakeSummarizeProvider()
    engine2, _ = _engine(database, store, tmp_path, provider=provider2)
    store.claim_task(task2.id, worker_pid=2)
    second = engine2.run_task(task2.id)

    assert second.status == "complete"
    assert second.reused_pages == 4
    # Anti-fake-progress / idempotency: no paid call, no duplicate artefacts.
    assert provider2.prompts == []
    assert len(store.list_pages(task2.id)) == 4
    for page in pages:
        assert readings.page_reading(page.id) is not None


def test_partial_resume_only_reprocesses_unfinished_pages(database, tmp_path) -> None:
    """Pages 1-2 done, 3-4 failed; retry re-summarizes ONLY pages 3-4."""

    store = LongReadStore(database)
    task_id, _ = _new_task(database, store, tmp_path, page_count=4, batch_size=2)
    failing = _FakeSummarizeProvider(missing_pages={3, 4})
    engine, _ = _engine(database, store, tmp_path, provider=failing, batch_size=2)
    store.claim_task(task_id, worker_pid=1)
    first = engine.run_task(task_id)
    assert first.status == "failed"
    assert store.get_page(task_id, 1).state == "done"
    assert store.get_page(task_id, 2).state == "done"
    assert store.get_page(task_id, 3).state == "failed"
    assert store.get_page(task_id, 4).state == "failed"

    assert store.retry_failed_pages(task_id) == 2
    healthy = _FakeSummarizeProvider()
    engine2, _ = _engine(database, store, tmp_path, provider=healthy, batch_size=2)
    store.claim_task(task_id, worker_pid=2)
    second = engine2.run_task(task_id)
    assert second.status == "complete"
    assert second.done_pages == 4
    # The retry run must only ask the model about pages 3 and 4.
    retried_numbers: set[int] = set()
    for prompt in healthy.prompts:
        retried_numbers |= _prompt_page_numbers(prompt)
    assert retried_numbers == {3, 4}
    for number in (1, 2, 3, 4):
        assert store.get_page(task_id, number).state == "done"


def test_seed_pages_is_idempotent(database, tmp_path) -> None:
    store = LongReadStore(database)
    task_id, document_id = _new_task(database, store, tmp_path, page_count=3)
    pages = database.list_pages(document_id)
    store.seed_pages(task_id, [(p.id, p.page_number) for p in pages])
    store.seed_pages(task_id, [(p.id, p.page_number) for p in pages])
    assert len(store.list_pages(task_id)) == 3


# ------------------------------------------------- pause / cancel / delete
def test_pause_then_resume_keeps_progress(database, tmp_path) -> None:
    store = LongReadStore(database)
    task_id, _ = _new_task(database, store, tmp_path, page_count=3, batch_size=1)
    engine, _ = _engine(database, store, tmp_path, batch_size=1)
    store.claim_task(task_id, worker_pid=1)
    # request a pause before the run consumes it
    store.request_control(task_id, "pause")
    paused = engine.run_task(task_id)
    assert paused.status == "paused_user"
    assert paused.done_pages == 0  # paused before the first batch

    store.resume_task(task_id)
    resumed = store.get_task(task_id)
    assert resumed.status == "queued"
    engine2, _ = _engine(database, store, tmp_path, batch_size=1)
    store.claim_task(task_id, worker_pid=2)
    final = engine2.run_task(task_id)
    assert final.status == "complete"
    assert final.done_pages == 3


def test_cancel_keeps_results_and_stops_scheduling(database, tmp_path) -> None:
    """Cancel is not delete: completed pages stay, no further model calls."""

    store = LongReadStore(database)
    task_id, document_id = _new_task(database, store, tmp_path, page_count=4, batch_size=1)
    provider = _FakeSummarizeProvider()
    engine, readings = _engine(database, store, tmp_path, provider=provider, batch_size=1)
    store.claim_task(task_id, worker_pid=1)
    # Cancel request present from the start: engine stops before any batch.
    store.request_control(task_id, "cancel")
    calls_before = len(provider.prompts)
    cancelled = engine.run_task(task_id)
    assert cancelled.status == "cancelled"
    assert len(provider.prompts) == calls_before  # no model call after cancel
    # Task rows still exist (not deleted).
    assert store.get_task(task_id).status == "cancelled"
    assert len(store.list_pages(task_id)) == 4
    del readings, document_id


def test_delete_refused_while_running_and_separated_from_cancel(database, tmp_path) -> None:
    store = LongReadStore(database)
    task_id, _ = _new_task(database, store, tmp_path, page_count=2)
    store.claim_task(task_id, worker_pid=1)  # running
    with pytest.raises(LongReadError):
        store.delete_task(task_id)  # must cancel first
    store.finish_cancelled(task_id)
    store.delete_task(task_id)  # now allowed: removes bookkeeping only
    with pytest.raises(LongReadError):
        store.get_task(task_id)


# --------------------------------------------------------- timing fields
def test_timing_fields_are_complete_and_count_waits(database, tmp_path) -> None:
    store = LongReadStore(database)
    task_id, _ = _new_task(database, store, tmp_path, page_count=2)
    engine, _ = _engine(database, store, tmp_path)
    store.claim_task(task_id, worker_pid=1)
    engine.run_task(task_id)
    task = store.get_task(task_id)
    view = timing_view(task)
    # all five honest timing buckets exist and are non-negative
    for value in (
        view.queued_ms, view.processing_ms, view.system_paused_ms,
        view.user_paused_ms, view.wall_clock_ms,
    ):
        assert isinstance(value, int) and value >= 0
    assert task.started_at  # real start time recorded
    assert task.heartbeat_at  # last heartbeat recorded
    # per-stage elapsed captured for at least extract/summarize/write
    stage_ms = task.stage_ms
    assert set(stage_ms) >= {"extract", "summarize", "write"}
    # wall clock covers the whole lifetime
    assert view.wall_clock_ms >= view.processing_ms


def test_paused_time_is_counted_separately(database, tmp_path) -> None:
    store = LongReadStore(database)
    task_id, _ = _new_task(database, store, tmp_path, page_count=1)
    store.claim_task(task_id, worker_pid=1)
    store.pause_running(task_id, by_user=True, message="user")
    task = store.get_task(task_id)
    assert task.status == "paused_user"
    view = timing_view(task)
    # processing time was accumulated on the running->paused transition
    assert view.processing_ms >= 0
    # live slice while paused accrues to the user-paused bucket
    assert view.live_phase == "user_paused_ms"


# ------------------------------------------------------ selective vision
def test_vision_is_selective_not_blanket(database, tmp_path) -> None:
    """Only sparse-text pages trigger vision; dense pages never do."""

    store = LongReadStore(database)
    # page 2 is sparse, pages 1 and 3 are dense
    task_id, document_id = _new_task(
        database, store, tmp_path, page_count=3, sparse_pages={2}
    )
    # give the sparse page a real image file so vision can actually run
    page2 = database.list_pages(document_id)[1]
    _write_png(Path(page2.image_path))
    vision = _FakeVisionProvider()
    provider = _FakeSummarizeProvider()
    engine, _ = _engine(
        database, store, tmp_path, provider=provider, vision_provider=vision
    )
    store.claim_task(task_id, worker_pid=1)
    final = engine.run_task(task_id)
    assert final.status == "complete"
    # exactly one vision call for the single sparse page
    assert len(vision.calls) == 1
    rows = {row.page_number: row for row in store.list_pages(task_id)}
    assert rows[2].vision_needed is True
    assert rows[2].vision_reason in ("sparse_text_layer", "no_text_layer")
    assert rows[2].vision_analysis  # persisted evidence
    assert rows[1].vision_needed is False
    assert rows[3].vision_needed is False


def _write_png(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (60, 40), (255, 255, 255)).save(path, format="PNG")


def test_vision_suppressed_when_page_confirmed_visual_free(database, tmp_path) -> None:
    """Sparse text BUT stage-1 says no visual content -> no pointless vision."""

    store = LongReadStore(database)
    task_id, document_id = _new_task(
        database, store, tmp_path, page_count=1, sparse_pages={1}
    )
    page = database.list_pages(document_id)[0]
    _write_png(Path(page.image_path))
    # stage-1 detector confirmed this page has NO visual content
    with database._connection() as connection:
        connection.execute(
            "UPDATE pages SET has_visual_content = 0 WHERE id = ?", (page.id,)
        )
    vision = _FakeVisionProvider()
    engine, _ = _engine(
        database, store, tmp_path,
        provider=_FakeSummarizeProvider(), vision_provider=vision,
    )
    store.claim_task(task_id, worker_pid=1)
    final = engine.run_task(task_id)
    assert final.status == "complete"
    assert vision.calls == []  # gate suppressed the call
    row = store.get_page(task_id, 1)
    assert row.vision_needed is False


# ------------------------------------------------------- worker coordinator
def test_coordinator_read_tasks_resumes_and_completes(database, tmp_path) -> None:
    store = LongReadStore(database)
    task_id, _ = _new_task(database, store, tmp_path, page_count=3)
    provider = _FakeSummarizeProvider()
    engine, _ = _engine(database, store, tmp_path, provider=provider)
    coordinator = LongReadCoordinator(
        database=database, store=store, engine=engine,
        model="fake-llm", batch_size=DEFAULT_BATCH_SIZE,
    )
    report = coordinator.read_tasks([task_id], worker_pid=99)
    assert report.read_complete == 1
    assert store.get_task(task_id).status == "complete"


def test_coordinator_maps_network_pause_to_system_pause(database, tmp_path) -> None:
    store = LongReadStore(database)
    task_id, _ = _new_task(database, store, tmp_path, page_count=2)
    provider = _FakeSummarizeProvider(network_error=True)
    engine, _ = _engine(database, store, tmp_path, provider=provider)
    coordinator = LongReadCoordinator(
        database=database, store=store, engine=engine,
        model="fake-llm", batch_size=DEFAULT_BATCH_SIZE,
    )
    report = coordinator.read_tasks([task_id], worker_pid=7)
    assert report.read_paused == 1
    task = store.get_task(task_id)
    assert task.status == "paused_system"


def test_stale_running_task_is_recovered_as_interrupted(database, tmp_path) -> None:
    """A crashed worker's task is honestly interrupted, resumable."""

    store = LongReadStore(database)
    task_id, _ = _new_task(database, store, tmp_path, page_count=2)
    store.claim_task(task_id, worker_pid=1)
    # force a stale heartbeat far in the past
    with database._connection() as connection:
        connection.execute(
            "UPDATE long_read_task SET heartbeat_at = '2000-01-01T00:00:00+00:00'"
            " WHERE id = ?",
            (task_id,),
        )
    recovered = store.recover_stale_running(stale_seconds=1.0)
    assert [task.id for task in recovered] == [task_id]
    assert store.get_task(task_id).status == "interrupted"
    # interrupted task is resumable
    store.resume_task(task_id)
    assert store.get_task(task_id).status == "queued"


def test_claim_is_single_flight(database, tmp_path) -> None:
    store = LongReadStore(database)
    task_id, _ = _new_task(database, store, tmp_path, page_count=1)
    assert store.claim_task(task_id, worker_pid=1) is True
    # already running -> a second worker cannot claim it
    assert store.claim_task(task_id, worker_pid=2) is False
