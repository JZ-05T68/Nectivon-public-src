"""Background worker + coordinator for long-document import/reading (P0-2).

The task lifetime must be **independent of the Streamlit page component**:
navigating away, closing the browser, or an app restart must never erase a
running batch. Nectivon is a local single-user app bound to 127.0.0.1 and
has no always-on service manager, so the honest architecture is a detached
OS subprocess (the *worker*) that owns the batch:

* the page only stages the uploaded bytes to durable disk, appends rows to
  the existing ``import_queue``, creates ``long_read_task`` rows and spawns
  one detached worker;
* the worker imports each staged file (reusing :class:`DocumentService`),
  seeds the page state machine, then drives :class:`LongReadEngine`
  batch-by-batch, heart-beating so a crash is detectable;
* everything durable is in SQLite, so after an app shutdown/crash the next
  page load marks dead-heartbeat tasks ``interrupted`` and offers 继续读取 /
  取消读取 — never a silent restart, never deleting completed work.

Pause / resume / cancel / delete stay strictly separated (see
:mod:`src.long_read_store`). Cancel stops scheduling and keeps results; it
is not delete. A cancel/pause request is consumed *before* the next model
call, so a cancelled task never keeps calling the model.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

from src.config import PROJECT_ROOT, runtime_settings
from src.database import Database
from src.document_service import DocumentService
from src.import_queue_service import DEDUP_NOTICE, ImportQueueService
from src.long_read_engine import LongReadEngine, LongReadNetworkPause
from src.long_read_store import LongReadStore

LOGGER = logging.getLogger(__name__)

#: Windows process-creation flags for a detached, console-less worker that
#: survives the parent Streamlit process exiting.
_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_CREATE_NO_WINDOW = 0x08000000

#: How often the worker refreshes ``heartbeat_at`` while a task runs.
HEARTBEAT_INTERVAL_SECONDS = 15.0

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._\u4e00-\u9fff-]+")


def _safe_component(name: str) -> str:
    cleaned = _SAFE_NAME.sub("_", name).strip("_")
    return cleaned[:120] or "file"


def staging_dir_for(data_dir: Path) -> Path:
    """Durable directory where the page parks uploaded bytes for the worker."""

    return Path(data_dir) / "import-staging"


def staging_path(staging_dir: Path, batch_id: str, filename: str) -> Path:
    """Deterministic staged-file path shared by the page and the worker."""

    return Path(staging_dir) / _safe_component(batch_id) / _safe_component(filename)


def stage_uploads(
    staging_dir: Path, batch_id: str, uploads: list
) -> dict[str, Path]:
    """Persist each upload's bytes to durable disk; return name -> path.

    The uploaded bytes never survive Streamlit navigation, so they are
    written to local disk *before* the worker is spawned. The worker reads
    them back and hands them to :class:`DocumentService`.
    """

    staged: dict[str, Path] = {}
    for upload in uploads:
        name = str(upload.name)
        path = staging_path(staging_dir, batch_id, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(upload.getvalue())
        staged[name] = path
    return staged


@dataclass(slots=True)
class CoordinatorReport:
    """Honest outcome of one worker run (no beautified counters)."""

    imported: int = 0
    duplicate: int = 0
    import_failed: int = 0
    read_complete: int = 0
    read_incomplete: int = 0
    read_cancelled: int = 0
    read_paused: int = 0
    read_failed: int = 0
    task_ids: list[int] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


class _Heartbeat:
    """Background thread that keeps ``heartbeat_at`` fresh while a task runs."""

    def __init__(
        self, store: LongReadStore, task_id: int,
        interval: float = HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        self._store = store
        self._task_id = task_id
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _Heartbeat:
        self._thread = threading.Thread(
            target=self._run, name=f"long-read-hb-{self._task_id}", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 1.0)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._store.heartbeat(self._task_id)
            except Exception:  # noqa: BLE001 - a heartbeat must never kill work
                LOGGER.exception("心跳写入失败：task=%s", self._task_id)


class LongReadCoordinator:
    """Tie import-queue + document-service + long-read engine into one run."""

    def __init__(
        self,
        *,
        database: Database,
        store: LongReadStore,
        engine: LongReadEngine,
        model: str,
        batch_size: int,
        document_service: DocumentService | None = None,
        queue: ImportQueueService | None = None,
        staging_dir: Path | None = None,
    ) -> None:
        self._database = database
        self._store = store
        self._engine = engine
        self._model = model
        self._batch_size = batch_size
        self._document_service = document_service
        self._queue = queue
        self._staging_dir = staging_dir

    # --------------------------------------------------------- import+read
    def process_batch(
        self, batch_id: str, *, run_id: str, worker_pid: int
    ) -> CoordinatorReport:
        """Import every staged file of one batch, then read each document."""

        report = CoordinatorReport()
        if self._queue is None or self._document_service is None or self._staging_dir is None:
            raise RuntimeError("process_batch 需要 document_service、queue 与 staging_dir。")
        queue = self._queue
        queue.recover_stale_inflight(run_id)
        for entry in queue.queued_entries(batch_id):
            staged = staging_path(self._staging_dir, batch_id, entry.filename)
            if not staged.is_file():
                queue.mark_failed(
                    entry.id,
                    message="导入失败：暂存文件不可用（请重新选择文件后再导入）。",
                )
                report.import_failed += 1
                continue
            queue.start_entry(entry.id, run_id=run_id)
            try:
                content = staged.read_bytes()
                result = self._document_service.import_document(
                    file_content=content, filename=entry.filename
                )
            except Exception as exc:  # noqa: BLE001 - isolate per-file failures
                LOGGER.exception("资料导入失败：filename=%s", entry.filename)
                queue.mark_failed(entry.id, message=f"导入失败：{exc}")
                report.import_failed += 1
                continue
            document = result.document
            if result.duplicate:
                queue.mark_complete(
                    entry.id, document_id=document.id, message=DEDUP_NOTICE
                )
                report.duplicate += 1
                _safe_unlink(staged)
                continue
            report.imported += 1
            queue.mark_imported(entry.id, document_id=document.id)
            pages = self._database.list_pages(document.id)
            task = self._store.create_task(
                document_id=document.id,
                document_sha256=document.sha256,
                total_pages=len(pages),
                batch_size=self._batch_size,
                model=self._model,
                batch_id=batch_id,
                queue_entry_id=entry.id,
            )
            self._store.seed_pages(
                task.id, [(page.id, page.page_number) for page in pages]
            )
            report.task_ids.append(task.id)
            outcome = self._read_task(task.id, worker_pid=worker_pid)
            self._sync_queue_after_read(queue, entry.id, document.id, task.id, outcome)
            if outcome in ("complete",):
                report.read_complete += 1
            elif outcome == "cancelled":
                report.read_cancelled += 1
            elif outcome == "network_paused":
                report.read_paused += 1
            elif outcome in ("failed", "budget_failed", "error"):
                report.read_failed += 1
            else:
                report.read_incomplete += 1
            _safe_unlink(staged)
        return report

    def _sync_queue_after_read(
        self, queue: ImportQueueService, entry_id: int,
        document_id: int, task_id: int, outcome: str,
    ) -> None:
        task = self._store.get_task(task_id)
        if task.status == "complete":
            queue.mark_complete(entry_id, document_id=document_id, message="")
        elif task.status == "cancelled":
            queue.mark_imported(
                entry_id, document_id=document_id,
                message="用户取消了 AI 阅读；资料已导入，已完成的页面结果保留。",
            )
        elif task.status in ("paused_user", "paused_system", "interrupted"):
            queue.mark_imported(
                entry_id, document_id=document_id,
                message=(
                    f"资料已导入，AI 阅读已暂停/中断（{task.done_pages}/"
                    f"{task.total_pages} 页已读）；可点「继续读取」。"
                ),
            )
        else:
            queue.mark_imported(
                entry_id, document_id=document_id,
                message=task.error_message or "资料已导入，AI 阅读未全部完成。",
            )
        del outcome

    # ---------------------------------------------------------- read-only
    def read_tasks(self, task_ids: list[int], *, worker_pid: int) -> CoordinatorReport:
        """Resume/continue existing tasks (no import); used by 继续读取."""

        report = CoordinatorReport()
        for task_id in task_ids:
            task = self._store.get_task(task_id)
            if task.status in ("paused_user", "paused_system", "interrupted", "failed"):
                self._store.resume_task(task_id)
            report.task_ids.append(task_id)
            outcome = self._read_task(task_id, worker_pid=worker_pid)
            if outcome == "complete":
                report.read_complete += 1
            elif outcome == "cancelled":
                report.read_cancelled += 1
            elif outcome == "network_paused":
                report.read_paused += 1
            elif outcome in ("failed", "budget_failed", "error"):
                report.read_failed += 1
            else:
                report.read_incomplete += 1
        return report

    def _read_task(self, task_id: int, *, worker_pid: int) -> str:
        """Claim one task, run the engine with a heartbeat, map the outcome."""

        if not self._store.claim_task(task_id, worker_pid=worker_pid):
            LOGGER.info("任务已被其他 worker 认领，跳过：task=%s", task_id)
            return "skipped"
        with _Heartbeat(self._store, task_id):
            try:
                self._engine.run_task(task_id)
            except LongReadNetworkPause as exc:
                LOGGER.warning("网络故障，系统安全暂停：task=%s detail=%s", task_id, exc)
                self._store.pause_running(
                    task_id, by_user=False,
                    message=f"网络故障已安全暂停，检查点已保存：{exc}",
                )
                return "network_paused"
            except Exception as exc:  # noqa: BLE001 - never lose the checkpoint
                from src.ai.provider import AIBudgetExceededError

                LOGGER.exception("长文档读取异常：task=%s", task_id)
                if isinstance(exc, AIBudgetExceededError):
                    self._store.fail_task(task_id, message=f"AI 预算限制：{exc}")
                    return "budget_failed"
                self._store.fail_task(task_id, message=f"读取异常：{exc}")
                return "error"
        final = self._store.get_task(task_id)
        return final.status


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        LOGGER.warning("暂存文件清理失败（不影响导入）：%s", path)


# --------------------------------------------------------------------- CLI
def _build_coordinator(
    database: Database, *, batch_size: int | None
) -> tuple[LongReadCoordinator, LongReadStore]:
    """Bootstrap the real runtime services for a spawned worker process."""

    from src.agent_document_reader import AgentReadingStore
    from src.runtime import (
        application_ai_provider,
        application_document_service,
        application_question_vision_provider,
        application_settings,
    )

    settings = application_settings()
    store = LongReadStore(database)
    readings = AgentReadingStore(settings.agent_readings_dir)
    provider = application_ai_provider()
    vision_provider = application_question_vision_provider()
    model = settings.ai_llm_model
    engine = LongReadEngine(
        database=database,
        store=store,
        provider=provider,
        readings=readings,
        model=model,
        vision_provider=vision_provider,
        vision_model=settings.ai_vision_model,
        batch_size=batch_size or _default_batch_size(),
    )
    coordinator = LongReadCoordinator(
        database=database,
        store=store,
        engine=engine,
        model=model,
        batch_size=batch_size or _default_batch_size(),
        document_service=application_document_service(),
        queue=ImportQueueService(database),
        staging_dir=staging_dir_for(settings.data_dir),
    )
    return coordinator, store


def _default_batch_size() -> int:
    """Benchmark-derived default (see scripts/bench_long_read.py)."""

    from src.long_read_engine import DEFAULT_BATCH_SIZE

    return DEFAULT_BATCH_SIZE


def main(argv: list[str] | None = None) -> int:
    """Worker process entry point (spawned detached by the import page)."""

    parser = argparse.ArgumentParser(prog="long_read_worker")
    parser.add_argument("--batch", action="append", default=[], dest="batches")
    parser.add_argument("--task", action="append", default=[], dest="tasks")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--batch-size", type=int, default=0)
    args = parser.parse_args(argv)

    from src.runtime import application_database, application_settings

    settings = application_settings()
    from src.runtime import configure_logging

    configure_logging(settings.log_path)
    database = application_database()
    worker_pid = os.getpid()
    coordinator, store = _build_coordinator(
        database, batch_size=args.batch_size or None
    )
    # Recover any dead-heartbeat tasks from a previous crashed run first.
    store.recover_stale_running()
    report = CoordinatorReport()
    run_id = args.run_id or f"worker-{worker_pid}"
    for batch_id in args.batches:
        partial = coordinator.process_batch(
            batch_id, run_id=run_id, worker_pid=worker_pid
        )
        _merge(report, partial)
    if args.tasks:
        task_ids = [int(value) for value in args.tasks]
        partial = coordinator.read_tasks(task_ids, worker_pid=worker_pid)
        _merge(report, partial)
    LOGGER.info(
        "长文档读取 worker 结束：pid=%s imported=%s dup=%s import_failed=%s "
        "read_complete=%s read_incomplete=%s read_paused=%s read_cancelled=%s "
        "read_failed=%s",
        worker_pid, report.imported, report.duplicate, report.import_failed,
        report.read_complete, report.read_incomplete, report.read_paused,
        report.read_cancelled, report.read_failed,
    )
    return 0


def _merge(target: CoordinatorReport, source: CoordinatorReport) -> None:
    target.imported += source.imported
    target.duplicate += source.duplicate
    target.import_failed += source.import_failed
    target.read_complete += source.read_complete
    target.read_incomplete += source.read_incomplete
    target.read_cancelled += source.read_cancelled
    target.read_paused += source.read_paused
    target.read_failed += source.read_failed
    target.task_ids.extend(source.task_ids)
    target.messages.extend(source.messages)


def spawn_worker(
    *,
    batches: list[str] | None = None,
    task_ids: list[int] | None = None,
    run_id: str = "",
    batch_size: int = 0,
    python_exe: str | None = None,
    project_root: Path | None = None,
) -> int:
    """Launch a detached worker subprocess; return its pid.

    The worker is a separate OS process, so it keeps running after the user
    navigates away from the import page or closes the browser tab. On
    Windows it is detached + console-less; elsewhere ``start_new_session``
    gives the same independence.
    """

    exe = python_exe or sys.executable
    root = Path(project_root or PROJECT_ROOT)
    command = [exe, "-m", "src.long_read_worker"]
    for batch_id in batches or []:
        command += ["--batch", batch_id]
    for task_id in task_ids or []:
        command += ["--task", str(int(task_id))]
    if run_id:
        command += ["--run-id", run_id]
    if batch_size:
        command += ["--batch-size", str(int(batch_size))]
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "long_read_worker.log"
    kwargs: dict[str, object] = {
        "cwd": str(root),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            _DETACHED_PROCESS | _CREATE_NEW_PROCESS_GROUP | _CREATE_NO_WINDOW
        )
    else:
        kwargs["start_new_session"] = True
    with open(log_path, "ab") as sink:
        kwargs["stdout"] = sink
        kwargs["stderr"] = sink
        process = subprocess.Popen(command, **kwargs)  # noqa: S603 - fixed argv
    LOGGER.info(
        "已启动后台读取 worker：pid=%s batches=%s tasks=%s log=%s",
        process.pid, batches or [], task_ids or [], log_path,
    )
    return int(process.pid)


def worker_available() -> bool:
    """Whether the runtime can resolve a real AI provider for reading."""

    try:
        settings = runtime_settings()
        del settings
    except Exception:  # noqa: BLE001 - settings problems surface elsewhere
        return False
    return True


__all__ = [
    "CoordinatorReport",
    "LongReadCoordinator",
    "spawn_worker",
    "stage_uploads",
    "staging_dir_for",
    "staging_path",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
