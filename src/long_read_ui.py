"""Import/long-read dashboard view-model + control actions (v0.8.8 P0-3).

Keeps the Streamlit page thin and testable: all data is read from the
durable stores (``import_queue`` + ``long_read_task``/``long_read_page``),
never from session state, so the status survives page switches and app
restarts. The page renders :class:`ImportDashboard` and calls the control
helpers, which write control flags / spawn the detached worker.

Control semantics are strictly separated (see :mod:`src.long_read_store`):

* 暂停 (pause)   -> mailbox flag; the worker checkpoints and stops.
* 继续读取 (resume) -> re-queue + spawn a fresh worker; completed work kept.
* 取消读取 (cancel) -> stop scheduling, keep completed results (NOT delete).
* 删除记录 (delete) -> remove task bookkeeping only; refused while running.

Rendering contract (P0-3, the eleven required elements):

1. file selection + import area      :func:`render_file_picker` /
   :func:`render_import_action`
2. imported-file list                :func:`_render_file_rows`
3. per-file independent status       ``FileRow.queue_label`` + ``task_label``
4. global progress                   ``ImportDashboard.global_progress_ratio``
5. current page and stage            ``FileRow.current_page_number`` /
   ``current_stage`` / ``current_batch``
6. real cumulative elapsed time      :func:`src.long_read_store.timing_view`
7. estimated remaining time          ``FileRow.eta_seconds`` + confidence
8. failure count with error detail   :func:`_render_failures`
9./10./11. pause / resume / cancel   :func:`_render_control_bar` and the
   per-file control buttons

**No displayed number is ever a literal.** Every count, page number,
duration and estimate is read back from ``LongReadStore`` / ``import_queue``
on each render, which is what lets the tests in
``tests/test_long_read_ui.py`` fail if a value is hard-coded.

Routing (no parallel import flow): both paths reuse ``import_queue`` +
``DocumentService``. Long native-text PDFs are read by the detached worker
so navigation can never kill a 515-page batch; everything else — photos,
Office files, short or scanned PDFs — keeps the existing in-page path
unchanged, because that path drives ``PageImageReader`` and therefore also
produces question candidates and figure registration.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import streamlit as st

from src.database import Database
from src.import_queue_service import ImportQueueService, new_run_id
from src.long_read_store import (
    TASK_STATUS_LABELS,
    LongReadStore,
    LongReadTask,
    TaskTimingView,
    timing_view,
)
from src.long_read_worker import spawn_worker, stage_uploads, staging_dir_for, staging_path
from src.time_display import format_beijing_time

LOGGER = logging.getLogger(__name__)

#: Batch-id prefix marking a batch as owned by the detached long-read worker.
#: The worker imports such a batch from staged bytes in its own OS process;
#: the in-page import loop must never adopt one (it has no live upload object
#: for those rows and would wrongly mark them「文件内容不可用」).
WORKER_BATCH_PREFIX: Final = "lr-"

#: A PDF with at least this many pages is read by the durable background
#: worker instead of inside one Streamlit script run — V086-301 proved a long
#: in-page batch dies when the user navigates away. All eight 液压与气动
#: courseware files (22-121 pages) are above it; short documents keep the
#: existing in-page path with its full per-page visual reading.
BACKGROUND_READ_MIN_PAGES: Final = 20

#: Minimum fraction of pages carrying a non-empty native text layer for the
#: batch reader to be the right path. Measured on the real 液压与气动
#: courseware: all 8 files score 1.000 (every one of the 515 pages has a text
#: layer), so the margin is wide. A scanned file scores ~0 and stays on
#: the existing per-page visual path, which is the one that produces question
#: candidates and figure registration.
BACKGROUND_TEXT_PAGE_RATIO: Final = 0.5

#: Pages sampled by the routing probe so a very large upload never turns the
#: routing decision into a noticeable wait.
_TEXT_PROBE_PAGE_LIMIT: Final = 30

#: How often the live panel re-reads durable state while a task runs.
LIVE_REFRESH_SECONDS: Final = 2.0

#: Accepted upload extensions — identical to the v0.8.7 import page so the
#: uploader contract (asserted by ``tests/test_import_ui.py``) is unchanged.
UPLOAD_EXTENSIONS: Final = [
    "pdf", "doc", "docx", "ppt", "pptx", "jpg", "jpeg", "png", "webp",
]


#: File-level (import_queue) status labels shown to the user.
QUEUE_STATUS_LABELS = {
    "queued": "等待导入",
    "importing": "导入中",
    "imported": "已导入，AI 读取未完成",
    "complete": "已完成",
    "failed": "导入失败",
    "interrupted": "已中断（可继续）",
}
_QUEUE_STATUS_ICONS = {
    "queued": "⏳",
    "importing": "🔄",
    "imported": "⚠️",
    "complete": "✅",
    "failed": "❌",
    "interrupted": "⏸️",
}
_ETA_CONFIDENCE_LABELS = {"low": "低", "medium": "中", "high": "高"}
#: Conservatism order used to fold several tasks' estimates into one batch
#: estimate — the batch is never presented as more reliable than its weakest
#: contributing task.
_ETA_CONFIDENCE_RANK = {"": 0, "low": 1, "medium": 2, "high": 3}


def _merge_eta_confidence(current: str, incoming: str) -> str:
    """Keep the most conservative of two ETA confidence labels."""

    if not current:
        return incoming
    if not incoming:
        return current
    return min(current, incoming, key=lambda value: _ETA_CONFIDENCE_RANK.get(value, 0))


@dataclass(frozen=True, slots=True)
class PageFailure:
    """One failed page's viewable reason + retry count."""

    filename: str
    page_number: int
    state_label: str
    error_code: str
    error_message: str
    retry_count: int


@dataclass(slots=True)
class FileRow:
    """One imported file's combined queue + long-read status."""

    filename: str
    queue_status: str
    queue_label: str
    queue_icon: str
    document_id: int | None
    error_message: str
    batch_id: str = ""
    queue_entry_id: int | None = None
    task_id: int | None = None
    task_status: str = ""
    task_label: str = ""
    total_pages: int = 0
    done_pages: int = 0
    pending_pages: int = 0
    failed_pages: int = 0
    reused_pages: int = 0
    current_page_number: int | None = None
    current_stage: str = ""
    current_batch: int = 0
    batch_count: int = 0
    eta_seconds: int | None = None
    eta_confidence: str = ""
    timing: TaskTimingView | None = None
    resumable: bool = False
    cancellable: bool = False
    deletable: bool = False
    running: bool = False

    @property
    def resumable_queue_row(self) -> bool:
        """Queue row the worker can re-import from staged bytes, no re-selection.

        Only worker-owned batches stage their uploads to durable disk, so only
        those rows can be continued by a click. An in-page batch row is
        reported by :attr:`needs_reselection` / :attr:`adoptable` instead —
        never by a button that would silently do nothing.
        """

        return (
            self.task_id is None
            and self.queue_status in ("queued", "interrupted")
            and self.batch_id.startswith(WORKER_BATCH_PREFIX)
        )

    @property
    def adoptable(self) -> bool:
        """In-page row whose document already exists, so only the reading is left.

        Such a row can be adopted by the background reader without the
        original bytes: the document and its pages are already on local disk.
        """

        return (
            self.task_id is None
            and self.document_id is not None
            and self.queue_status in ("queued", "interrupted", "imported")
            and not self.batch_id.startswith(WORKER_BATCH_PREFIX)
        )

    @property
    def needs_reselection(self) -> bool:
        """In-page row whose import never finished: the bytes are truly gone."""

        return (
            self.task_id is None
            and self.document_id is None
            and self.queue_status in ("queued", "interrupted")
            and not self.batch_id.startswith(WORKER_BATCH_PREFIX)
        )


@dataclass(slots=True)
class ImportDashboard:
    """Everything the import page needs to render, from durable state only."""

    batch_id: str | None
    files: list[FileRow] = field(default_factory=list)
    failures: list[PageFailure] = field(default_factory=list)
    batch_ids: list[str] = field(default_factory=list)
    global_total_pages: int = 0
    global_done_pages: int = 0
    global_pending_pages: int = 0
    global_failed_pages: int = 0
    global_reused_pages: int = 0
    global_wall_clock_ms: int = 0
    global_eta_seconds: int | None = None
    global_eta_confidence: str = ""
    has_active_task: bool = False
    has_resumable_task: bool = False
    has_resumable_queue_row: bool = False
    has_adoptable_row: bool = False
    has_reselection_rows: bool = False
    queue_counts: dict[str, int] = field(default_factory=dict)

    @property
    def global_progress_ratio(self) -> float:
        if self.global_total_pages <= 0:
            return 0.0
        resolved = self.global_done_pages + self.global_failed_pages
        return max(0.0, min(1.0, resolved / self.global_total_pages))

    @property
    def file_count(self) -> int:
        return len(self.files)


@dataclass(frozen=True, slots=True)
class UploadRoute:
    """Which reading path each selected file takes, and why (Chinese reason)."""

    background: tuple[Any, ...]
    in_page: tuple[Any, ...]
    reasons: dict[str, str] = field(default_factory=dict)

    @property
    def mixed(self) -> bool:
        return bool(self.background) and bool(self.in_page)


@dataclass(frozen=True, slots=True)
class ResumePlan:
    """What one click can really continue — and what it honestly cannot."""

    task_ids: tuple[int, ...] = ()
    batch_ids: tuple[str, ...] = ()
    resumed_files: tuple[str, ...] = ()
    needs_reselection: tuple[str, ...] = ()
    adoptable_files: tuple[str, ...] = ()
    adoptable_rows: tuple[FileRow, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    worker_pid: int = 0

    @property
    def did_anything(self) -> bool:
        return bool(self.task_ids or self.batch_ids)



def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def format_elapsed(milliseconds: int) -> str:
    """Human, honest elapsed time (never hides waits)."""

    if milliseconds < 0:
        milliseconds = 0
    seconds = milliseconds // 1000
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}小时{minutes}分{secs}秒"
    if minutes:
        return f"{minutes}分{secs}秒"
    return f"{secs}秒"


def format_eta(seconds: int | None, confidence: str) -> str:
    if seconds is None:
        return "计算中……"
    label = _ETA_CONFIDENCE_LABELS.get(confidence, "")
    text = format_elapsed(int(seconds) * 1000)
    return f"约 {text}（可信度{label}）" if label else f"约 {text}"


def build_dashboard(
    database: Database,
    *,
    recover: bool = True,
    batch_ids: Sequence[str] | None = None,
) -> ImportDashboard:
    """Assemble the dashboard from durable state (survives page switches).

    Nothing is read from ``st.session_state``: every number comes from the
    ``import_queue`` rows and the ``long_read_task``/``long_read_page`` state
    machine, so switching away from the import page and back — or restarting
    the app — re-renders exactly the same real status.

    ``batch_ids`` defaults to :meth:`ImportQueueService.dashboard_batch_ids`,
    which covers every batch that still needs attention plus the newest one.
    """

    queue = ImportQueueService(database)
    store = LongReadStore(database)
    if recover:
        # Honest crash recovery: a dead-heartbeat running task is interrupted,
        # never left pretending to run.
        store.recover_stale_running()
    if batch_ids is None:
        resolved = queue.dashboard_batch_ids()
        if not resolved:
            latest = queue.latest_batch_id()
            resolved = [latest] if latest else []
    else:
        # Caller-supplied ids are de-duplicated, order-preserved, empties dropped.
        resolved = [batch_id for batch_id in dict.fromkeys(batch_ids) if batch_id]
    dashboard = ImportDashboard(batch_id=resolved[0] if resolved else None)
    dashboard.batch_ids = list(resolved)
    for batch_id in resolved:
        _append_batch_to_dashboard(store, queue, dashboard, batch_id)
    return dashboard


def _append_batch_to_dashboard(
    store: LongReadStore,
    queue: ImportQueueService,
    dashboard: ImportDashboard,
    batch_id: str,
) -> None:
    """Add one batch's file rows, failures and counters to ``dashboard``."""

    entries = queue.list_batch(batch_id)
    for status, count in ImportQueueService.summarize(entries).items():
        dashboard.queue_counts[status] = dashboard.queue_counts.get(status, 0) + count
    tasks_by_entry: dict[int, LongReadTask] = {}
    tasks_by_document: dict[int, LongReadTask] = {}
    for task in store.list_tasks(batch_id):
        if task.queue_entry_id is not None:
            tasks_by_entry[task.queue_entry_id] = task
        tasks_by_document[task.document_id] = task

    for entry in entries:
        task = tasks_by_entry.get(entry.id)
        if task is None and entry.document_id is not None:
            task = tasks_by_document.get(entry.document_id)
        row = FileRow(
            filename=entry.filename,
            queue_status=entry.status,
            queue_label=QUEUE_STATUS_LABELS.get(entry.status, entry.status),
            queue_icon=_QUEUE_STATUS_ICONS.get(entry.status, "•"),
            document_id=entry.document_id,
            error_message=entry.error_message,
            batch_id=batch_id,
            queue_entry_id=entry.id,
        )
        if row.resumable_queue_row:
            dashboard.has_resumable_queue_row = True
        if row.adoptable:
            dashboard.has_adoptable_row = True
        if row.needs_reselection:
            dashboard.has_reselection_rows = True
        if task is not None:
            view = timing_view(task)
            row.task_id = task.id
            row.task_status = task.status
            row.task_label = TASK_STATUS_LABELS.get(task.status, task.status)
            row.total_pages = task.total_pages
            row.done_pages = task.done_pages
            row.pending_pages = task.pending_pages
            row.failed_pages = task.failed_pages
            row.reused_pages = task.reused_pages
            row.current_page_number = task.current_page_number
            row.current_stage = task.current_stage
            row.current_batch = task.current_batch
            row.batch_count = task.batch_count
            row.eta_seconds = task.eta_seconds
            row.eta_confidence = task.eta_confidence
            row.timing = view
            row.running = task.status == "running"
            row.resumable = task.status in (
                "paused_user", "paused_system", "interrupted", "failed"
            )
            row.cancellable = not task.terminal
            row.deletable = task.status not in ("running", "queued")
            dashboard.global_total_pages += task.total_pages
            dashboard.global_done_pages += task.done_pages
            dashboard.global_pending_pages += task.pending_pages
            dashboard.global_failed_pages += task.failed_pages
            dashboard.global_reused_pages += task.reused_pages
            dashboard.global_wall_clock_ms += view.wall_clock_ms
            if not task.terminal and task.eta_seconds is not None:
                dashboard.global_eta_seconds = (
                    (dashboard.global_eta_seconds or 0) + int(task.eta_seconds)
                )
                # The batch estimate is only as trustworthy as its weakest task.
                dashboard.global_eta_confidence = _merge_eta_confidence(
                    dashboard.global_eta_confidence, task.eta_confidence
                )
            if task.status in ("running", "queued"):
                dashboard.has_active_task = True
            if row.resumable:
                dashboard.has_resumable_task = True
            if task.failed_pages:
                for page in store.list_pages(task.id, states=("failed",)):
                    dashboard.failures.append(
                        PageFailure(
                            filename=entry.filename,
                            page_number=page.page_number,
                            state_label=page.state_label,
                            error_code=page.error_code,
                            error_message=page.error_message,
                            retry_count=page.retry_count,
                        )
                    )
        dashboard.files.append(row)


# --------------------------------------------------------------- routing
def probe_pdf_text_layer(content: bytes) -> tuple[int, float] | None:
    """Return ``(page_count, text_layer_ratio)`` for PDF bytes, else ``None``.

    ``text_layer_ratio`` is the fraction of sampled pages whose native text
    layer is **non-empty** — the same criterion used to establish that all
    515 courseware pages carry a text layer. It deliberately does *not*
    require a page to be text-rich: the batch reader was designed for thin
    pages and answers them with its own selective-vision gate
    (``VISION_TEXT_THRESHOLD_CHARS``), so a figure-heavy page is not a reason
    to fall back to the slower in-page path.

    Metadata/text-layer detection is explicitly allowed automation and costs
    ~1 ms per page on the real courseware, so the routing decision never
    needs a model call. At most :data:`_TEXT_PROBE_PAGE_LIMIT` pages, evenly
    spread, are sampled. ``None`` means "not a readable PDF" and always falls
    back to the existing in-page import path — a routing probe must never
    reject a file the user chose.
    """

    try:
        from src.pdf_service import _load_pymupdf

        pymupdf = _load_pymupdf()
        document = pymupdf.open(stream=bytes(content), filetype="pdf")
    except Exception:  # noqa: BLE001 - unreadable bytes keep the existing path
        return None
    try:
        if getattr(document, "needs_pass", False):
            return None
        page_count = int(document.page_count)
        if page_count <= 0:
            return None
        step = max(1, page_count // _TEXT_PROBE_PAGE_LIMIT)
        sampled = list(range(0, page_count, step))[:_TEXT_PROBE_PAGE_LIMIT]
        with_text = 0
        for index in sampled:
            text = document.load_page(index).get_text("text") or ""
            if text.strip():
                with_text += 1
        return page_count, (with_text / len(sampled) if sampled else 0.0)
    except Exception:  # noqa: BLE001 - a probe failure must not block import
        LOGGER.exception("PDF 路由探测失败，改用页面内导入路径")
        return None
    finally:
        document.close()


def classify_uploads(uploads: Sequence[Any]) -> UploadRoute:
    """Split selected files into background-read and existing in-page groups.

    One import queue, one document service — only the *reading* path differs,
    and the reason is recorded in Chinese so the screen explains itself:

    * long native-text PDF -> detached worker (survives navigation);
    * photo / Office / short PDF / scanned PDF -> the unchanged in-page path,
      which drives ``PageImageReader`` and therefore still produces question
      candidates and figure registration.
    """

    background: list[Any] = []
    in_page: list[Any] = []
    reasons: dict[str, str] = {}
    for upload in uploads:
        name = str(upload.name)
        suffix = Path(name).suffix.lower()
        if suffix != ".pdf":
            in_page.append(upload)
            reasons[name] = (
                "图片与 Office 文件走原有逐页识别路径（题目候选与图文关联由该路径产生）。"
                if suffix in {".jpg", ".jpeg", ".png", ".webp"}
                else "Office 文件需要先在本机转换为 PDF，沿用原有页面内导入路径。"
            )
            continue
        probe = None
        try:
            probe = probe_pdf_text_layer(upload.getvalue())
        except Exception:  # noqa: BLE001 - an unreadable upload is not rejected here
            LOGGER.exception("读取上传内容用于路由判断失败：%s", name)
        if probe is None:
            in_page.append(upload)
            reasons[name] = "无法在本机确认这份 PDF 的页数与文字层，沿用原有页面内导入路径。"
            continue
        page_count, ratio = probe
        if page_count >= BACKGROUND_READ_MIN_PAGES and ratio >= BACKGROUND_TEXT_PAGE_RATIO:
            background.append(upload)
            reasons[name] = (
                f"共 {page_count} 页、约 {int(round(ratio * 100))}% 的页面自带文字层，"
                "改由后台独立进程分批读取：离开本页或关闭浏览器都不会中断。"
            )
        elif page_count < BACKGROUND_READ_MIN_PAGES:
            in_page.append(upload)
            reasons[name] = (
                f"共 {page_count} 页（少于 {BACKGROUND_READ_MIN_PAGES} 页），"
                "沿用原有页面内导入路径。"
            )
        else:
            in_page.append(upload)
            reasons[name] = (
                f"共 {page_count} 页，但只有约 {int(round(ratio * 100))}% 的页面自带文字层，"
                "属于扫描件，需要走原有逐页视觉识别路径。"
            )
    return UploadRoute(
        background=tuple(background), in_page=tuple(in_page), reasons=reasons
    )


def _resolve_data_dir(data_dir: Path | None) -> Path:
    """Resolve the durable data dir lazily, so rendering never needs settings."""

    if data_dir is not None:
        return Path(data_dir)
    from src.runtime import application_settings

    return Path(application_settings().data_dir)



# ------------------------------------------------------------ control ops
def new_worker_batch_id() -> str:
    """A batch id that is unambiguously owned by the detached worker."""

    return f"{WORKER_BATCH_PREFIX}{new_run_id()}"


def start_import_batch(
    database: Database,
    uploads: Sequence[Any],
    *,
    data_dir: Path | None = None,
    spawn: bool = True,
    run_id: str | None = None,
    batch_size: int = 0,
) -> str:
    """Stage bytes + enqueue + (optionally) spawn the detached worker.

    Returns the durable batch id. Uploaded bytes are written to local disk
    first so the worker can import them in a separate process; the batch is
    persisted before any processing, so navigation can never erase it.

    The batch is always a **worker-owned** batch (:data:`WORKER_BATCH_PREFIX`):
    it continues the worker's own open batch when there is one, and it never
    adopts an in-page batch (whose rows have no staged bytes), so the two
    consumers of ``import_queue`` cannot steal each other's rows.
    """

    queue = ImportQueueService(database)
    staging_dir = staging_dir_for(_resolve_data_dir(data_dir))
    names = [str(upload.name) for upload in uploads]
    if not names:
        raise ValueError("至少需要选择一个文件才能开始后台导入。")
    open_batch = queue.latest_open_batch_id(batch_id_prefix=WORKER_BATCH_PREFIX)
    pending_names: set[str] = set()
    if open_batch:
        pending_names = {entry.filename for entry in queue.queued_entries(open_batch)}
    new_names = [name for name in names if name not in pending_names]
    if new_names:
        batch_id = queue.enqueue_batch(
            new_names, batch_id=open_batch or new_worker_batch_id()
        )
    else:
        batch_id = open_batch or queue.enqueue_batch(
            names, batch_id=new_worker_batch_id()
        )
    # Stage every selected upload (re-staging an existing name is harmless).
    stage_uploads(staging_dir, batch_id, list(uploads))
    if spawn:
        try:
            spawn_worker(
                batches=[batch_id],
                run_id=run_id or new_run_id(),
                batch_size=batch_size,
            )
        except Exception:  # noqa: BLE001 - spawn failure must not lose the batch
            LOGGER.exception("后台读取 worker 启动失败；批次已持久化，可稍后继续")
    return batch_id


def plan_resume(
    dashboard: ImportDashboard, *, staging_dir: Path
) -> ResumePlan:
    """Work out what one click can really continue, without touching state.

    A queue row can only be re-imported by the worker while its staged bytes
    are still on disk; when they are gone the honest answer is「需要重新选择
    该文件」rather than a button that silently does nothing.
    """

    task_ids: list[int] = []
    batch_ids: list[str] = []
    resumed: list[str] = []
    missing: list[str] = []
    adoptable: list[FileRow] = []
    for row in dashboard.files:
        if row.task_id is not None and row.resumable:
            task_ids.append(row.task_id)
            resumed.append(row.filename)
        elif row.resumable_queue_row:
            staged = staging_path(staging_dir, row.batch_id, row.filename)
            if staged.is_file():
                if row.batch_id not in batch_ids:
                    batch_ids.append(row.batch_id)
                resumed.append(row.filename)
            else:
                missing.append(row.filename)
        elif row.adoptable:
            adoptable.append(row)
        elif row.needs_reselection:
            missing.append(row.filename)
    return ResumePlan(
        task_ids=tuple(dict.fromkeys(task_ids)),
        batch_ids=tuple(batch_ids),
        resumed_files=tuple(resumed),
        needs_reselection=tuple(missing),
        adoptable_files=tuple(row.filename for row in adoptable),
        adoptable_rows=tuple(adoptable),
    )


def adopt_queue_row_for_background_read(
    database: Database, row: FileRow, *, batch_size: int = 0
) -> int | None:
    """Give an already-imported document a durable background reading task.

    This is the honest one-click continuation for a row left behind by the
    v0.8.7 in-page path: the document and its pages are already on local
    disk, so no re-selection and no second import are needed — only a
    ``long_read_task`` the detached worker can pick up. Returns the new task
    id, or ``None`` when the document has no readable pages (reported to the
    user instead of being silently ignored).
    """

    if row.document_id is None:
        return None
    document = database.get_document(row.document_id)
    if document is None:
        LOGGER.warning("无法接管：找不到资料 document_id=%s", row.document_id)
        return None
    pages = database.list_pages(document.id)
    if not pages:
        LOGGER.warning("无法接管：资料没有页面 document_id=%s", document.id)
        return None
    store = LongReadStore(database)
    task = store.create_task(
        document_id=document.id,
        document_sha256=document.sha256,
        total_pages=len(pages),
        batch_size=batch_size or _default_batch_size(),
        model=_resolve_model(),
        batch_id=row.batch_id,
        queue_entry_id=row.queue_entry_id,
    )
    store.seed_pages(task.id, [(page.id, page.page_number) for page in pages])
    return task.id


def _default_batch_size() -> int:
    """Benchmark-derived engine default (see ``scripts/bench_long_read.py``)."""

    from src.long_read_engine import DEFAULT_BATCH_SIZE

    return DEFAULT_BATCH_SIZE


def _resolve_model() -> str:
    """The configured LLM model, resolved lazily so rendering never needs it."""

    from src.runtime import application_settings

    return str(application_settings().ai_llm_model)


def resume_everything(
    database: Database,
    dashboard: ImportDashboard,
    *,
    data_dir: Path | None = None,
    spawn: bool = True,
    batch_size: int = 0,
) -> ResumePlan:
    """One-click 继续: re-queue resumable tasks and re-importable queue rows.

    Replaces the v0.8.7 instruction「重新选择同样的文件即可继续」(Reviewer
    offload item #6). Completed pages are never re-queued and never re-paid
    for — :meth:`LongReadStore.resume_task` only re-queues rows left
    mid-flight, and the engine reuses a still-valid reading before any call.

    Rows whose document already exists but which were left by the in-page
    path are *adopted* into a durable background task, so they too continue
    with one click instead of a re-selection.
    """

    staging_dir = staging_dir_for(_resolve_data_dir(data_dir))
    plan = plan_resume(dashboard, staging_dir=staging_dir)
    store = LongReadStore(database)
    for task_id in plan.task_ids:
        store.resume_task(task_id)
    adopted: list[int] = []
    adopted_names: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    for row in plan.adoptable_rows:
        try:
            task_id = adopt_queue_row_for_background_read(
                database, row, batch_size=batch_size
            )
        except Exception as exc:  # noqa: BLE001 - one row must not block others
            LOGGER.exception("接管队列行失败：filename=%s", row.filename)
            errors.append(f"「{row.filename}」无法一键继续：{exc}")
            continue
        if task_id is None:
            warnings.append(
                f"「{row.filename}」已导入但没有可读取的页面，无法一键继续；"
                "请在「我的资料」中检查这份资料。"
            )
            continue
        adopted.append(task_id)
        adopted_names.append(row.filename)
    spawn_tasks = list(dict.fromkeys([*plan.task_ids, *adopted]))
    worker_pid = 0
    if spawn and (spawn_tasks or plan.batch_ids):
        try:
            worker_pid = spawn_worker(
                batches=list(plan.batch_ids),
                task_ids=spawn_tasks,
                run_id=new_run_id(),
                batch_size=batch_size,
            )
        except Exception:  # noqa: BLE001 - the re-queue already persisted
            LOGGER.exception("一键继续的 worker 启动失败；任务与队列行已重新排队")
            errors.append(
                "后台读取进程启动失败；任务已重新排队，请再点一次「继续读取」。"
            )
    return ResumePlan(
        task_ids=tuple(spawn_tasks),
        batch_ids=plan.batch_ids,
        resumed_files=(*plan.resumed_files, *adopted_names),
        needs_reselection=plan.needs_reselection,
        adoptable_files=plan.adoptable_files,
        adoptable_rows=plan.adoptable_rows,
        warnings=tuple(warnings),
        errors=tuple(errors),
        worker_pid=worker_pid,
    )



def pause_task(database: Database, task_id: int) -> None:
    """Leave a pause request for the running worker (checkpoint + stop)."""

    LongReadStore(database).request_control(task_id, "pause")


def cancel_task(database: Database, task_id: int) -> None:
    """Cancel: stop scheduling, keep completed results (not a delete)."""

    LongReadStore(database).cancel_task(task_id)


def resume_task(database: Database, task_id: int, *, spawn: bool = True,
                batch_size: int = 0) -> None:
    """Re-queue a paused/interrupted/failed task and spawn a fresh worker."""

    store = LongReadStore(database)
    store.resume_task(task_id)
    if spawn:
        try:
            spawn_worker(task_ids=[task_id], batch_size=batch_size)
        except Exception:  # noqa: BLE001 - resume persists even if spawn fails
            LOGGER.exception("继续读取的 worker 启动失败；任务已排队，可再次点击继续")


def retry_failed_pages(database: Database, task_id: int, *, spawn: bool = True,
                       batch_size: int = 0) -> int:
    """Re-queue every failed page of a task, then continue in the background."""

    store = LongReadStore(database)
    count = store.retry_failed_pages(task_id)
    if count and spawn:
        try:
            spawn_worker(task_ids=[task_id], batch_size=batch_size)
        except Exception:  # noqa: BLE001
            LOGGER.exception("重试失败页的 worker 启动失败；页面已重新排队")
    return count


def retry_page(database: Database, task_id: int, page_number: int,
               *, spawn: bool = True) -> None:
    store = LongReadStore(database)
    store.retry_page(task_id, page_number)
    if spawn:
        try:
            spawn_worker(task_ids=[task_id])
        except Exception:  # noqa: BLE001
            LOGGER.exception("单页重试的 worker 启动失败；该页已重新排队")


def mark_page_skipped(database: Database, task_id: int, page_number: int) -> None:
    """User confirms a page needs no processing (explicit, never silent)."""

    LongReadStore(database).mark_page_skipped_confirmed(task_id, page_number)


def delete_task_record(database: Database, task_id: int) -> None:
    """Delete task bookkeeping only; refused while running (cancel first)."""

    LongReadStore(database).delete_task(task_id)


# ------------------------------------------------------------- rendering
def _refresh_now() -> None:
    """Re-render immediately after a control action (only inside a real run).

    ``st.rerun`` unwinds the script via an exception, so it must never be
    wrapped in a broad ``except``. Outside a Streamlit run context (bare mode,
    tests) there is nothing to refresh and the live fragment will pick the new
    durable state up on its next pass.
    """

    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx() is not None:
        st.rerun(scope="app")


def _file_progress_ratio(row: FileRow) -> float:
    """One file's resolved-page ratio, from the page state machine only."""

    if row.total_pages <= 0:
        return 0.0
    resolved = row.done_pages + row.failed_pages
    return max(0.0, min(1.0, resolved / row.total_pages))


def _timing_lines(view: TaskTimingView) -> str:
    """Honest timing breakdown: every wait is named, none is folded away."""

    return (
        f"处理 {format_elapsed(view.processing_ms)} · "
        f"排队 {format_elapsed(view.queued_ms)} · "
        f"系统暂停 {format_elapsed(view.system_paused_ms)} · "
        f"用户暂停 {format_elapsed(view.user_paused_ms)}"
    )


def render_file_picker(*, key: str = "long_read_file_picker") -> list:
    """Element 1a: the multi-file selection area. Returns the chosen uploads.

    The accepted extensions and ``accept_multiple_files`` are the v0.8.7
    contract (locked by ``tests/test_import_ui.py``); only the help text
    changed, to say which files are read in the background.
    """

    uploads = st.file_uploader(
        "选择 PDF、Word、PowerPoint 或图片文件（可多选）",
        type=UPLOAD_EXTENSIONS,
        accept_multiple_files=True,
        key=key,
        help=(
            "支持 PDF、DOC、DOCX、PPT、PPTX，以及 JPG/JPEG/PNG/WEBP 图片；"
            "相同内容不会重复保存，重复上传会明确提示。图片按原样保存为一份一页资料。"
            f"页数不少于 {BACKGROUND_READ_MIN_PAGES} 页且自带文字层的 PDF "
            "会交给后台独立进程分批读取，离开本页也不会中断。"
        ),
    )
    return list(uploads or [])


def render_import_action(
    database: Database,
    *,
    uploads: Sequence[Any],
    data_dir: Path | None = None,
    batch_size: int = 0,
) -> list:
    """Element 1b: the import area. Returns the uploads the page must handle
    through its **existing** in-page path.

    One button, one import queue, one document service. Long native-text PDFs
    are staged to durable disk, enqueued and handed to the detached worker;
    everything else is returned untouched so the page keeps its v0.8.7
    behaviour (including full per-page visual reading for photos).
    """

    if not uploads:
        return []
    route = classify_uploads(uploads)
    st.caption(
        f"已选择 {len(uploads)} 个文件：" + "、".join(str(u.name) for u in uploads)
    )
    if not st.button("导入并让 Agent 逐份阅读", type="primary"):
        return []
    if route.background:
        try:
            start_import_batch(
                database,
                list(route.background),
                data_dir=data_dir,
                batch_size=batch_size,
            )
        except Exception as exc:  # noqa: BLE001 - import failure must be readable
            LOGGER.exception("后台导入批次创建失败")
            st.error(f"后台导入没有开始：{exc}")
        else:
            st.success(
                f"已开始导入 {len(route.background)} 份文件。"
                "可以离开本页或关闭浏览器，读取会继续；进度在下方实时显示。"
            )
    return list(route.in_page)


def render_progress_panel(
    database: Database,
    *,
    data_dir: Path | None = None,
    batch_size: int = 0,
    live: bool = True,
) -> None:
    """Elements 2-11: the durable progress panel with pause/resume/cancel.

    While any task is active the panel body is wrapped in a Streamlit
    fragment that re-reads durable state every :data:`LIVE_REFRESH_SECONDS`
    seconds, so progress, elapsed time and the control buttons stay on screen
    and up to date without a full page rerun. When nothing is running the body
    is called directly — deliberately *not* through ``st.fragment(run_every=
    None)``, because that form does not invoke its body outside a real script
    run, which would silently render an empty panel.
    """

    probe = build_dashboard(database)

    def _body() -> None:
        _render_panel(database, data_dir=data_dir, batch_size=batch_size)

    if live and probe.has_active_task:
        st.fragment(run_every=LIVE_REFRESH_SECONDS)(_body)()
    else:
        _body()


def _render_panel(
    database: Database, *, data_dir: Path | None, batch_size: int
) -> None:
    dashboard = build_dashboard(database)
    st.subheader("导入队列与读取进度")
    st.caption(
        "长文档由后台独立进程读取：离开本页或关闭浏览器都不会中断，返回本页仍能看到进度。"
        "其余文件在当前导入页面打开期间处理，离开本页会暂停未完成的文件。"
        "未完成或已中断的文件列在下方，可一键继续，已完成的内容不会丢失。"
    )
    if not dashboard.files:
        st.info("还没有导入记录。选择文件后点「导入并让 Agent 逐份阅读」即可开始。")
        return
    _render_global_progress(dashboard)
    _render_control_bar(database, dashboard, data_dir=data_dir, batch_size=batch_size)
    _render_failures(database, dashboard, batch_size=batch_size)
    _render_file_rows(database, dashboard, batch_size=batch_size)


def _render_global_progress(dashboard: ImportDashboard) -> None:
    """Element 4 (global progress) + the batch-level honest counters."""

    counts = dashboard.queue_counts
    st.progress(
        dashboard.global_progress_ratio,
        text=(
            f"总体进度：已读 {dashboard.global_done_pages} / "
            f"{dashboard.global_total_pages} 页 · 失败 "
            f"{dashboard.global_failed_pages} 页 · 待处理 "
            f"{dashboard.global_pending_pages} 页"
        ),
    )
    columns = st.columns(6)
    columns[0].metric("文件", dashboard.file_count)
    columns[1].metric("总页数", dashboard.global_total_pages)
    columns[2].metric("已读页", dashboard.global_done_pages)
    columns[3].metric("失败页", dashboard.global_failed_pages)
    # Element 6 at batch level: real wall-clock, waits included.
    columns[4].metric("累计耗时", format_elapsed(dashboard.global_wall_clock_ms))
    # Element 7 at batch level.
    columns[5].metric(
        "预计剩余",
        format_eta(dashboard.global_eta_seconds, dashboard.global_eta_confidence),
    )
    st.caption(
        f"文件状态：已完成 {counts.get('complete', 0)} · 已导入未读完 "
        f"{counts.get('imported', 0)} · 等待 {counts.get('queued', 0)} · "
        f"处理中 {counts.get('importing', 0)} · 已中断 "
        f"{counts.get('interrupted', 0)} · 导入失败 {counts.get('failed', 0)}"
        + (
            " · 资料导入与 AI 阅读是两个独立状态。"
            if counts.get("imported", 0)
            else ""
        )
    )
    st.caption(
        "累计耗时为各文件墙钟时间之和，包含排队、处理、系统暂停与用户暂停；"
        "不会为了好看而扣除等待时间。"
    )


def _render_control_bar(
    database: Database,
    dashboard: ImportDashboard,
    *,
    data_dir: Path | None,
    batch_size: int,
) -> None:
    """Elements 9/10/11, always visible while anything can still be controlled.

    The buttons are rendered on every pass and only *disabled* when they do
    not apply, so 暂停 and 取消 never disappear from a running task's screen.
    """

    rows = [row for row in dashboard.files if row.task_id is not None]
    running = [row for row in rows if row.running]
    queued = [row for row in rows if row.task_status == "queued"]
    resumable_tasks = [row for row in rows if row.resumable]
    can_pause = bool(running)
    can_cancel = bool(running or queued or resumable_tasks)
    can_resume = bool(
        resumable_tasks
        or dashboard.has_resumable_queue_row
        or dashboard.has_adoptable_row
    )
    if not (can_pause or can_cancel or can_resume):
        return
    st.markdown("**任务控制**")
    columns = st.columns(4)
    if columns[0].button(
        "暂停",
        key="long_read_pause_all",
        disabled=not can_pause,
        help="当前批次读完这一批后停下并保存检查点；已完成的页面不会丢失。",
    ):
        for row in running:
            pause_task(database, int(row.task_id))
        st.info(f"已发出暂停请求：{len(running)} 份文件会在当前批次结束后停下。")
        _refresh_now()
    if columns[1].button(
        "继续读取",
        key="long_read_resume_all",
        disabled=not can_resume,
        help=(
            "一键继续所有已暂停/已中断/失败的任务，以及未导入完或只读完一部分的文件；"
            "已完成的页面不会重复读取，也不会重复付费。"
        ),
    ):
        plan = resume_everything(
            database, dashboard, data_dir=data_dir, batch_size=batch_size
        )
        if plan.did_anything:
            st.success(
                f"已继续 {len(plan.resumed_files)} 份文件"
                + (f"（后台进程 {plan.worker_pid}）" if plan.worker_pid else "")
                + "；已完成的页面不会重复读取。"
            )
        else:
            st.info("没有可继续的任务。")
        for message in plan.warnings:
            st.warning(message)
        for message in plan.errors:
            st.error(message)
        if plan.needs_reselection:
            st.warning(
                "以下文件的导入从未完成、本机也没有暂存内容，需要重新选择后再导入："
                + "、".join(plan.needs_reselection)
            )
        _refresh_now()
    if columns[2].button(
        "取消读取",
        key="long_read_cancel_all",
        disabled=not can_cancel,
        help="停止调度并保留已完成的结果；取消不等于删除，资料与已读页面都还在。",
    ):
        cancelled = 0
        for row in rows:
            if row.task_id is None or not row.cancellable:
                continue
            cancel_task(database, int(row.task_id))
            cancelled += 1
        st.info(f"已取消 {cancelled} 份文件的读取；已完成的结果与资料全部保留。")
        _refresh_now()
    columns[3].button(
        "刷新",
        key="long_read_refresh",
        help="立即从本机数据库重新读取进度（进度也会自动刷新）。",
    )


def _render_failures(
    database: Database, dashboard: ImportDashboard, *, batch_size: int
) -> None:
    """Element 8: failure count plus the real per-page error detail."""

    del batch_size  # retries reuse the task's own persisted batch size
    total = len(dashboard.failures)
    if not total:
        st.caption("失败页：0")
        return
    with st.expander(f"失败明细（{total} 页）", expanded=True):
        for failure in dashboard.failures:
            st.markdown(
                f"- **{failure.filename}** 第 {failure.page_number} 页 · "
                f"{failure.state_label} · 已重试 {failure.retry_count} 次"
            )
            st.caption(f"错误码 {failure.error_code or '未记录'}：{failure.error_message}")
            row = next(
                (
                    item
                    for item in dashboard.files
                    if item.filename == failure.filename and item.task_id is not None
                ),
                None,
            )
            if row is None:
                continue
            if st.button(
                "重试该页",
                key=f"long_read_retry_page_{row.task_id}_{failure.page_number}",
            ):
                try:
                    retry_page(database, int(row.task_id), failure.page_number)
                except Exception as exc:  # noqa: BLE001 - surface a readable reason
                    LOGGER.exception(
                        "单页重试失败：task=%s page=%s", row.task_id, failure.page_number
                    )
                    st.error(f"重试第 {failure.page_number} 页失败：{exc}")
                else:
                    st.success(f"第 {failure.page_number} 页已重新排队。")
                    _refresh_now()


def _render_file_rows(
    database: Database, dashboard: ImportDashboard, *, batch_size: int
) -> None:
    """Elements 2/3/5/6/7: the imported-file list with independent statuses."""

    st.markdown("**已导入文件**")
    for row in dashboard.files:
        with st.container(border=True):
            # Same one-line format as the v0.8.7 queue list: the file's own
            # honest message stays attached to its own row (V086-301 §6.4).
            line = f"{row.queue_icon} **{row.filename}** —— {row.queue_label}"
            if row.error_message:
                line += f"：{row.error_message}"
            st.markdown(line)
            if row.task_id is None:
                if row.adoptable:
                    st.caption(
                        "这份资料已保存在本机，只剩 AI 阅读没有完成；"
                        "点上方「继续读取」即可在后台接着读，不需要重新选择文件。"
                    )
                elif row.needs_reselection:
                    st.caption(
                        "这份文件的导入从未完成，本机也没有暂存内容；"
                        "需要重新选择同一个文件后再导入，已完成的其他文件不会重复导入。"
                    )
                if row.document_id is not None and row.queue_status in (
                    "complete",
                    "imported",
                ):
                    _render_open_document_button(row)
                continue
            _render_one_task(database, row, batch_size=batch_size)


def _render_open_document_button(row: FileRow) -> None:
    if st.button(
        "查看这份资料", key=f"open_document_{row.queue_entry_id}_{row.document_id}"
    ):
        st.switch_page(
            "pages/17_我的资料.py", query_params={"document": str(row.document_id)}
        )


def _render_one_task(database: Database, row: FileRow, *, batch_size: int) -> None:
    """One file's independent long-read status (elements 3/5/6/7)."""

    view = row.timing
    st.markdown(f"读取状态：{row.task_label}（任务 #{row.task_id}）")
    st.progress(
        _file_progress_ratio(row),
        text=(
            f"已读 {row.done_pages} / {row.total_pages} 页 · 失败 "
            f"{row.failed_pages} 页 · 待处理 {row.pending_pages} 页 · 复用 "
            f"{row.reused_pages} 页"
        ),
    )
    # Element 5: current page and stage, straight from the worker heartbeat.
    stage = row.current_stage or "暂无"
    batch_text = f" · 第 {row.current_batch} 批" if row.current_batch else ""
    page_text = (
        f"第 {row.current_page_number} 页" if row.current_page_number else "暂无当前页"
    )
    st.caption(f"当前：{page_text} · {stage}{batch_text}")
    if view is not None:
        # Element 6: real cumulative elapsed time, waits included.
        st.caption(f"累计耗时：{format_elapsed(view.wall_clock_ms)}")
        with st.expander("计时明细", expanded=False):
            st.markdown(_timing_lines(view))
            st.caption(
                f"开始：{format_beijing_time(view.started_at)} · "
                f"最后心跳：{format_beijing_time(view.heartbeat_at)} · "
                f"结束：{format_beijing_time(view.finished_at)}"
            )
    # Element 7: estimated remaining time, with its honest confidence.
    st.caption(f"预计剩余：{format_eta(row.eta_seconds, row.eta_confidence)}")
    if row.error_message and row.task_id is not None:
        st.warning(row.error_message)
    _render_row_controls(database, row, batch_size=batch_size)
    if row.document_id is not None:
        _render_open_document_button(row)


def _render_row_controls(
    database: Database, row: FileRow, *, batch_size: int
) -> None:
    """Per-file 暂停 / 继续读取 / 取消读取 (elements 9/10/11)."""

    task_id = int(row.task_id or 0)
    if not task_id:
        return
    columns = st.columns(4)
    if columns[0].button(
        "暂停", key=f"long_read_pause_{task_id}", disabled=not row.running
    ):
        try:
            pause_task(database, task_id)
        except Exception as exc:  # noqa: BLE001 - readable Chinese UI error
            LOGGER.exception("暂停任务失败：task=%s", task_id)
            st.error(f"暂停失败：{exc}")
        else:
            st.info("已发出暂停请求；当前批次结束后停下，检查点会保存。")
            _refresh_now()
    if columns[1].button(
        "继续读取", key=f"long_read_resume_{task_id}", disabled=not row.resumable
    ):
        try:
            resume_task(database, task_id, batch_size=batch_size)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("继续读取失败：task=%s", task_id)
            st.error(f"继续读取失败：{exc}")
        else:
            st.success("已重新排队并启动后台读取；已完成的页面不会重复读取。")
            _refresh_now()
    if columns[2].button(
        "取消读取", key=f"long_read_cancel_{task_id}", disabled=not row.cancellable
    ):
        try:
            cancel_task(database, task_id)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("取消读取失败：task=%s", task_id)
            st.error(f"取消读取失败：{exc}")
        else:
            st.info("已取消读取；已完成的结果与资料全部保留（取消不等于删除）。")
            _refresh_now()
    if columns[3].button(
        "重试失败页",
        key=f"long_read_retry_{task_id}",
        disabled=not row.failed_pages,
    ):
        try:
            count = retry_failed_pages(database, task_id, batch_size=batch_size)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("重试失败页出错：task=%s", task_id)
            st.error(f"重试失败页出错：{exc}")
        else:
            st.success(f"已重新排队 {count} 个失败页。")
            _refresh_now()


__all__ = [
    "BACKGROUND_READ_MIN_PAGES",
    "BACKGROUND_TEXT_PAGE_RATIO",
    "FileRow",
    "ImportDashboard",
    "LIVE_REFRESH_SECONDS",
    "PageFailure",
    "QUEUE_STATUS_LABELS",
    "ResumePlan",
    "UPLOAD_EXTENSIONS",
    "UploadRoute",
    "WORKER_BATCH_PREFIX",
    "adopt_queue_row_for_background_read",
    "build_dashboard",
    "cancel_task",
    "classify_uploads",
    "delete_task_record",
    "format_elapsed",
    "format_eta",
    "mark_page_skipped",
    "new_worker_batch_id",
    "pause_task",
    "plan_resume",
    "probe_pdf_text_layer",
    "render_file_picker",
    "render_import_action",
    "render_progress_panel",
    "resume_everything",
    "resume_task",
    "retry_failed_pages",
    "retry_page",
    "start_import_batch",
]
