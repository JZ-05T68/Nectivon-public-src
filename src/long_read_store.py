"""Durable store for long-document reading tasks (v0.8.8 P0-2, schema v39).

The 515-page courseware reading must survive Streamlit page switches, app
restarts and crashes, so **all** state lives in SQLite — never in session
state.  This module owns the two v39 tables:

* ``long_read_task`` — one row per document reading task with an explicit
  status, a control mailbox (UI writes ``pause``/``cancel`` requests, the
  worker consumes them), honest timing buckets and a heartbeat;
* ``long_read_page`` — one row per page with the page-level state machine,
  retry counter, failure reason/code and selective-vision evidence.

Design rules enforced here:

1. **Explicit states only.** Every page row always carries one of the nine
   states; nothing is silently skipped. Page transitions are validated
   against :data:`PAGE_TRANSITIONS` — an illegal transition raises instead
   of corrupting the checkpoint.
2. **Honest timing.** Wall-clock time is ``created_at .. finished_at`` and
   is never shortened by excluding waits: queued time, processing time,
   system-paused time and user-paused time are accumulated separately on
   every status transition, and the live view adds the still-running
   current phase on top.
3. **Idempotent resume.** Page seeding uses ``INSERT OR IGNORE`` against
   ``UNIQUE(task_id, page_number)``; re-running a task never duplicates
   page rows, never re-queues ``done`` pages and never overwrites a valid
   completed result.
4. **Separated semantics.** Pause / resume / cancel / delete are four
   distinct operations. Cancel keeps completed results and only stops
   scheduling; delete removes *task bookkeeping rows* (never documents,
   page readings or queue rows) and is refused while a task is running.

Rollback note (schema v39): the two tables are append-only additions; see
``_apply_version_thirty_nine`` in :mod:`src.migrations` for the exact
rollback statements.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from src.database import Database, DatabaseError

LOGGER = logging.getLogger(__name__)

#: Page-level state machine states (UI shows the Chinese labels below).
PAGE_STATES: Final = (
    "pending",
    "queued",
    "extracting",
    "vision",
    "summarizing",
    "writing",
    "done",
    "failed",
    "skipped_confirmed",
)

PAGE_STATE_LABELS: Final = {
    "pending": "待处理",
    "queued": "排队中",
    "extracting": "提取中",
    "vision": "视觉分析中",
    "summarizing": "总结中",
    "writing": "写入中",
    "done": "已完成",
    "failed": "失败",
    "skipped_confirmed": "用户确认无需处理",
}

#: Legal page transitions. ``done`` is terminal: a valid completed result is
#: never overwritten by a resume or retry pass (idempotency requirement).
PAGE_TRANSITIONS: Final = {
    "pending": {"queued", "skipped_confirmed", "failed"},
    "queued": {"extracting", "pending", "failed", "skipped_confirmed"},
    "extracting": {"vision", "summarizing", "queued", "failed"},
    "vision": {"summarizing", "queued", "failed"},
    "summarizing": {"writing", "queued", "failed"},
    "writing": {"done", "queued", "failed"},
    "failed": {"queued", "pending", "skipped_confirmed"},
    "done": set(),
    "skipped_confirmed": {"pending"},
}

#: Page states that mean "still work in flight or waiting".
ACTIVE_PAGE_STATES: Final = (
    "pending",
    "queued",
    "extracting",
    "vision",
    "summarizing",
    "writing",
)

TASK_STATUSES: Final = (
    "queued",
    "running",
    "paused_user",
    "paused_system",
    "cancelled",
    "complete",
    "failed",
    "interrupted",
)

TASK_STATUS_LABELS: Final = {
    "queued": "排队中",
    "running": "读取中",
    "paused_user": "已暂停（用户）",
    "paused_system": "已暂停（系统）",
    "cancelled": "已取消（结果保留）",
    "complete": "已完成",
    "failed": "失败（可重试）",
    "interrupted": "已中断（可继续）",
}

#: Legal task status transitions.
TASK_TRANSITIONS: Final = {
    "queued": {"running", "cancelled", "failed"},
    "running": {
        "paused_user",
        "paused_system",
        "cancelled",
        "complete",
        "failed",
        "interrupted",
    },
    "paused_user": {"queued", "cancelled", "failed"},
    "paused_system": {"queued", "cancelled", "failed"},
    "interrupted": {"queued", "cancelled", "failed"},
    "failed": {"queued", "cancelled"},
    "complete": set(),
    "cancelled": set(),
}

#: Status -> timing bucket that accumulates while the task sits in it.
_STATUS_TIME_BUCKET: Final = {
    "queued": "queued_ms",
    "running": "processing_ms",
    "paused_user": "user_paused_ms",
    "paused_system": "system_paused_ms",
    "interrupted": "system_paused_ms",
}

#: Control requests the UI may leave in the task mailbox.
CONTROL_REQUESTS: Final = ("pause", "cancel")

#: A running task whose heartbeat is older than this is considered dead
#: (crash / app shutdown) and is honestly marked ``interrupted``.
STALE_HEARTBEAT_SECONDS: Final = 120.0


class LongReadError(DatabaseError):
    """A long-read store operation was refused or could not complete."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="microseconds")


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class LongReadTask:
    """One document's durable reading task row."""

    id: int
    batch_id: str
    queue_entry_id: int | None
    document_id: int
    document_sha256: str
    status: str
    control: str
    total_pages: int
    done_pages: int
    reused_pages: int
    pending_pages: int
    failed_pages: int
    current_batch: int
    batch_count: int
    batch_size: int
    current_page_number: int | None
    current_stage: str
    model: str
    worker_pid: int | None
    created_at: str
    started_at: str | None
    heartbeat_at: str | None
    updated_at: str
    finished_at: str | None
    queued_ms: int
    processing_ms: int
    system_paused_ms: int
    user_paused_ms: int
    stage_ms_json: str
    eta_seconds: int | None
    eta_confidence: str
    error_message: str

    @property
    def terminal(self) -> bool:
        return self.status in ("complete", "cancelled")

    @property
    def resumable(self) -> bool:
        """Task can be continued without losing completed work."""

        return self.status in ("queued", "paused_user", "paused_system",
                               "interrupted", "failed")

    @property
    def stage_ms(self) -> dict[str, int]:
        try:
            payload = json.loads(self.stage_ms_json or "{}")
        except json.JSONDecodeError:
            return {}
        return {
            str(key): int(value)
            for key, value in payload.items()
            if isinstance(value, (int, float))
        }


@dataclass(frozen=True, slots=True)
class LongReadPage:
    """One page's durable state-machine row."""

    id: int
    task_id: int
    page_id: int
    page_number: int
    state: str
    batch_index: int
    retry_count: int
    error_code: str
    error_message: str
    vision_needed: bool
    vision_reason: str
    vision_analysis: str
    source_text_sha256: str
    stage_ms_json: str
    created_at: str
    updated_at: str

    @property
    def state_label(self) -> str:
        return PAGE_STATE_LABELS.get(self.state, self.state)

    @property
    def stage_ms(self) -> dict[str, int]:
        try:
            payload = json.loads(self.stage_ms_json or "{}")
        except json.JSONDecodeError:
            return {}
        return {
            str(key): int(value)
            for key, value in payload.items()
            if isinstance(value, (int, float))
        }


@dataclass(frozen=True, slots=True)
class TaskTimingView:
    """Honest wall-clock view: every wait is counted, nothing is hidden."""

    queued_ms: int
    processing_ms: int
    system_paused_ms: int
    user_paused_ms: int
    wall_clock_ms: int
    started_at: str | None
    finished_at: str | None
    heartbeat_at: str | None
    live_phase: str


def timing_view(task: LongReadTask, *, now: datetime | None = None) -> TaskTimingView:
    """Add the still-running current phase to the persisted buckets.

    ``updated_at`` is only rewritten on status transitions, so
    ``now - updated_at`` is exactly how long the task has been sitting in
    its current phase; that live slice is attributed to the same bucket the
    transition would use. Waiting time is therefore always included.
    """

    now = now or _utc_now()
    queued = task.queued_ms
    processing = task.processing_ms
    system_paused = task.system_paused_ms
    user_paused = task.user_paused_ms
    live_phase = ""
    bucket = _STATUS_TIME_BUCKET.get(task.status)
    updated = _parse_iso(task.updated_at)
    if bucket is not None and updated is not None:
        live_ms = max(0, int((now - updated).total_seconds() * 1000))
        live_phase = bucket
        if bucket == "queued_ms":
            queued += live_ms
        elif bucket == "processing_ms":
            processing += live_ms
        elif bucket == "system_paused_ms":
            system_paused += live_ms
        else:
            user_paused += live_ms
    created = _parse_iso(task.created_at)
    finished = _parse_iso(task.finished_at)
    wall = 0
    if created is not None:
        wall = int(((finished or now) - created).total_seconds() * 1000)
    return TaskTimingView(
        queued_ms=queued,
        processing_ms=processing,
        system_paused_ms=system_paused,
        user_paused_ms=user_paused,
        wall_clock_ms=max(0, wall),
        started_at=task.started_at,
        finished_at=task.finished_at,
        heartbeat_at=task.heartbeat_at,
        live_phase=live_phase,
    )


class LongReadStore:
    """SQLite-backed task/page state machine shared by UI and worker."""

    def __init__(self, database: Database) -> None:
        self._database = database

    # ------------------------------------------------------------ creation
    def create_task(
        self,
        *,
        document_id: int,
        document_sha256: str,
        total_pages: int,
        batch_size: int,
        model: str,
        batch_id: str = "",
        queue_entry_id: int | None = None,
    ) -> LongReadTask:
        """Create one task row, or return the still-open task for the document.

        Idempotent per document: an unfinished task (anything except
        ``complete``/``cancelled``) is returned unchanged instead of
        creating a duplicate that would double the model calls.
        """

        if total_pages <= 0:
            raise LongReadError("任务总页数必须大于 0。")
        if batch_size <= 0:
            raise LongReadError("批次大小必须大于 0。")
        existing = self.open_task_for_document(document_id)
        if existing is not None:
            return existing
        timestamp = _iso(_utc_now())
        with self._database._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO long_read_task(
                    batch_id, queue_entry_id, document_id, document_sha256,
                    status, control, total_pages, pending_pages, batch_size,
                    model, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'queued', '', ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    queue_entry_id,
                    document_id,
                    document_sha256,
                    total_pages,
                    total_pages,
                    batch_size,
                    model,
                    timestamp,
                    timestamp,
                ),
            )
            task_id = int(cursor.lastrowid or 0)
        return self.get_task(task_id)

    def seed_pages(self, task_id: int, pages: Sequence[tuple[int, int]]) -> None:
        """Insert one row per ``(page_id, page_number)``; ignore duplicates.

        ``UNIQUE(task_id, page_number)`` plus ``INSERT OR IGNORE`` makes
        re-seeding on resume a no-op — page rows are never duplicated and
        existing states (including ``done``) are never reset.
        """

        timestamp = _iso(_utc_now())
        with self._database._connection() as connection:
            for page_id, page_number in pages:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO long_read_page(
                        task_id, page_id, page_number, state,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, 'pending', ?, ?)
                    """,
                    (task_id, page_id, page_number, timestamp, timestamp),
                )
        self.recount(task_id)

    # --------------------------------------------------------------- reads
    def get_task(self, task_id: int) -> LongReadTask:
        with self._database._connection() as connection:
            row = connection.execute(
                "SELECT * FROM long_read_task WHERE id = ?", (task_id,)
            ).fetchone()
        if row is None:
            raise LongReadError(f"读取任务不存在：{task_id}")
        return _task_from_row(row)

    def open_task_for_document(self, document_id: int) -> LongReadTask | None:
        with self._database._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM long_read_task
                WHERE document_id = ? AND status NOT IN ('complete', 'cancelled')
                ORDER BY id DESC LIMIT 1
                """,
                (document_id,),
            ).fetchone()
        return _task_from_row(row) if row is not None else None

    def list_tasks(self, batch_id: str | None = None) -> list[LongReadTask]:
        with self._database._connection() as connection:
            if batch_id is None:
                rows = connection.execute(
                    "SELECT * FROM long_read_task ORDER BY id DESC"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM long_read_task WHERE batch_id = ? ORDER BY id",
                    (batch_id,),
                ).fetchall()
        return [_task_from_row(row) for row in rows]

    def open_tasks(self, batch_id: str | None = None) -> list[LongReadTask]:
        return [
            task
            for task in self.list_tasks(batch_id)
            if not task.terminal
        ]

    def list_pages(
        self, task_id: int, *, states: Iterable[str] | None = None
    ) -> list[LongReadPage]:
        with self._database._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM long_read_page WHERE task_id = ? ORDER BY page_number",
                (task_id,),
            ).fetchall()
        pages = [_page_from_row(row) for row in rows]
        if states is None:
            return pages
        wanted = set(states)
        return [page for page in pages if page.state in wanted]

    def get_page(self, task_id: int, page_number: int) -> LongReadPage | None:
        with self._database._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM long_read_page
                WHERE task_id = ? AND page_number = ?
                """,
                (task_id, page_number),
            ).fetchone()
        return _page_from_row(row) if row is not None else None

    def recount(self, task_id: int) -> LongReadTask:
        """Recompute done/pending/failed counters from the page rows.

        The page rows are the single source of truth, so counters can never
        drift from the real state machine (a fake-progress guard).
        """

        with self._database._connection() as connection:
            rows = connection.execute(
                "SELECT state, COUNT(*) AS n FROM long_read_page WHERE task_id = ?"
                " GROUP BY state",
                (task_id,),
            ).fetchall()
            counts = {str(row["state"]): int(row["n"]) for row in rows}
            done = counts.get("done", 0)
            failed = counts.get("failed", 0)
            pending = sum(counts.get(state, 0) for state in ACTIVE_PAGE_STATES)
            connection.execute(
                """
                UPDATE long_read_task
                SET done_pages = ?, failed_pages = ?, pending_pages = ?
                WHERE id = ?
                """,
                (done, failed, pending, task_id),
            )
        return self.get_task(task_id)

    # ---------------------------------------------------- page transitions
    def set_page_state(
        self,
        task_id: int,
        page_number: int,
        state: str,
        *,
        stage: str = "",
        stage_ms: int = 0,
        error_code: str = "",
        error_message: str = "",
        retry_increment: int = 0,
        batch_index: int | None = None,
        vision_needed: bool | None = None,
        vision_reason: str | None = None,
        vision_analysis: str | None = None,
        source_text_sha256: str | None = None,
    ) -> LongReadPage:
        """Validate and persist one page transition (never a silent skip)."""

        if state not in PAGE_STATES:
            raise LongReadError(f"页面状态无效：{state}")
        page = self.get_page(task_id, page_number)
        if page is None:
            raise LongReadError(
                f"任务 {task_id} 没有第 {page_number} 页的状态行；禁止静默跳过。"
            )
        if state != page.state and state not in PAGE_TRANSITIONS[page.state]:
            raise LongReadError(
                f"非法页面状态流转：{page.state} -> {state}（第 {page_number} 页）"
            )
        assignments = ["state = ?", "updated_at = ?"]
        values: list[object] = [state, _iso(_utc_now())]
        record_stage = bool(stage)
        if record_stage:
            # Always record a named stage, even at ~0 ms, so the per-stage
            # timing view is complete and never silently drops a fast stage.
            merged = dict(page.stage_ms)
            merged[stage] = int(merged.get(stage, 0)) + max(0, int(stage_ms))
            assignments.append("stage_ms_json = ?")
            values.append(json.dumps(merged, ensure_ascii=False, sort_keys=True))
        if error_code or error_message:
            assignments.append("error_code = ?")
            values.append(error_code)
            assignments.append("error_message = ?")
            values.append(error_message[:2000])
        if state == "done":
            assignments.append("error_code = ''")
            assignments.append("error_message = ''")
        if retry_increment:
            assignments.append("retry_count = retry_count + ?")
            values.append(int(retry_increment))
        if batch_index is not None:
            assignments.append("batch_index = ?")
            values.append(int(batch_index))
        if vision_needed is not None:
            assignments.append("vision_needed = ?")
            values.append(1 if vision_needed else 0)
        if vision_reason is not None:
            assignments.append("vision_reason = ?")
            values.append(vision_reason[:500])
        if vision_analysis is not None:
            assignments.append("vision_analysis = ?")
            values.append(vision_analysis)
        if source_text_sha256 is not None:
            assignments.append("source_text_sha256 = ?")
            values.append(source_text_sha256)
        values.extend([task_id, page_number])
        with self._database._connection() as connection:
            connection.execute(
                f"""
                UPDATE long_read_page SET {', '.join(assignments)}
                WHERE task_id = ? AND page_number = ?
                """,
                values,
            )
            if record_stage:
                self._accumulate_task_stage_ms(connection, task_id, stage, stage_ms)
        return self.get_page(task_id, page_number) or page

    @staticmethod
    def _accumulate_task_stage_ms(
        connection, task_id: int, stage: str, stage_ms: int
    ) -> None:
        row = connection.execute(
            "SELECT stage_ms_json FROM long_read_task WHERE id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return
        try:
            merged = json.loads(str(row["stage_ms_json"] or "{}"))
        except json.JSONDecodeError:
            merged = {}
        merged[stage] = int(merged.get(stage, 0)) + int(stage_ms)
        connection.execute(
            "UPDATE long_read_task SET stage_ms_json = ? WHERE id = ?",
            (json.dumps(merged, ensure_ascii=False, sort_keys=True), task_id),
        )

    def retry_page(self, task_id: int, page_number: int) -> LongReadPage:
        """Re-queue exactly one failed/active page; ``done`` is never reset."""

        page = self.get_page(task_id, page_number)
        if page is None:
            raise LongReadError(f"任务 {task_id} 没有第 {page_number} 页。")
        if page.state == "done":
            raise LongReadError("已完成的页面不会被重试覆盖（幂等保护）。")
        if page.state == "skipped_confirmed":
            raise LongReadError("用户已确认无需处理的页面不参与重试。")
        updated = self.set_page_state(
            task_id, page_number, "queued", retry_increment=1
        )
        self._revive_task_for_retry(task_id)
        return updated

    def retry_failed_pages(self, task_id: int) -> int:
        """Re-queue every failed page of one task; return how many."""

        failed = self.list_pages(task_id, states=("failed",))
        for page in failed:
            self.set_page_state(
                task_id, page.page_number, "queued", retry_increment=1
            )
        if failed:
            self._revive_task_for_retry(task_id)
        return len(failed)

    def _revive_task_for_retry(self, task_id: int) -> None:
        """Bring a ``failed`` task back to ``queued`` after pages were re-queued.

        A finished-at/eta reset keeps the honest timing view correct: the
        retry run is a fresh processing phase, the earlier failure time is
        already banked in the buckets.
        """

        task = self.get_task(task_id)
        if task.status != "failed":
            return
        self._set_task_status(task_id, "queued", error_message="", finished=False)
        with self._database._connection() as connection:
            connection.execute(
                "UPDATE long_read_task SET finished_at = NULL, eta_seconds = NULL,"
                " eta_confidence = '' WHERE id = ?",
                (task_id,),
            )
        self.recount(task_id)

    def mark_page_skipped_confirmed(self, task_id: int, page_number: int) -> LongReadPage:
        """User explicitly confirms a page needs no processing (never silent)."""

        page = self.get_page(task_id, page_number)
        if page is None:
            raise LongReadError(f"任务 {task_id} 没有第 {page_number} 页。")
        if page.state == "done":
            raise LongReadError("已完成的页面不能改为无需处理。")
        return self.set_page_state(task_id, page_number, "skipped_confirmed")

    # --------------------------------------------------- task transitions
    def _set_task_status(
        self,
        task_id: int,
        status: str,
        *,
        error_message: str | None = None,
        worker_pid: int | None = None,
        finished: bool | None = None,
    ) -> LongReadTask:
        """Transition with validation + honest timing-bucket accumulation."""

        if status not in TASK_STATUSES:
            raise LongReadError(f"任务状态无效：{status}")
        task = self.get_task(task_id)
        if status != task.status and status not in TASK_TRANSITIONS[task.status]:
            raise LongReadError(
                f"非法任务状态流转：{task.status} -> {status}（任务 {task_id}）"
            )
        now = _utc_now()
        assignments = ["status = ?", "updated_at = ?"]
        values: list[object] = [status, _iso(now)]
        bucket = _STATUS_TIME_BUCKET.get(task.status)
        previous = _parse_iso(task.updated_at)
        if bucket is not None and previous is not None and status != task.status:
            elapsed = max(0, int((now - previous).total_seconds() * 1000))
            assignments.append(f"{bucket} = {bucket} + ?")
            values.append(elapsed)
        if status == "running":
            if task.started_at is None:
                assignments.append("started_at = ?")
                values.append(_iso(now))
            if worker_pid is not None:
                assignments.append("worker_pid = ?")
                values.append(int(worker_pid))
        if finished if finished is not None else status in ("complete", "cancelled", "failed"):
            assignments.append("finished_at = ?")
            values.append(_iso(now))
            assignments.append("current_stage = ''")
        if error_message is not None:
            assignments.append("error_message = ?")
            values.append(error_message[:2000])
        values.append(task_id)
        with self._database._connection() as connection:
            connection.execute(
                f"UPDATE long_read_task SET {', '.join(assignments)} WHERE id = ?",
                values,
            )
        return self.get_task(task_id)

    def claim_task(self, task_id: int, *, worker_pid: int) -> bool:
        """Atomically claim a resumable task for this worker (single flight).

        Returns ``False`` when another worker already owns it, so two
        workers can never process the same task (no duplicate model calls).
        The queued-time bucket is accumulated up to the claim moment, so
        waiting time is never dropped from the honest timing view.
        """

        now = _utc_now()
        now_iso = _iso(now)
        with self._database._connection() as connection:
            row = connection.execute(
                "SELECT status, updated_at FROM long_read_task WHERE id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                raise LongReadError(f"读取任务不存在：{task_id}")
            previous = _parse_iso(str(row["updated_at"]))
            queued_elapsed = 0
            if str(row["status"]) == "queued" and previous is not None:
                queued_elapsed = max(0, int((now - previous).total_seconds() * 1000))
            cursor = connection.execute(
                """
                UPDATE long_read_task
                SET status = 'running', updated_at = ?, worker_pid = ?,
                    started_at = COALESCE(started_at, ?), control = '',
                    queued_ms = queued_ms + ?
                WHERE id = ? AND status IN ('queued', 'interrupted')
                """,
                (now_iso, int(worker_pid), now_iso, queued_elapsed, task_id),
            )
            claimed = cursor.rowcount == 1
        if claimed:
            LOGGER.info("长文档读取任务已认领：task=%s pid=%s", task_id, worker_pid)
        return claimed

    def heartbeat(
        self,
        task_id: int,
        *,
        stage: str = "",
        page_number: int | None = None,
        current_batch: int | None = None,
        batch_count: int | None = None,
        eta_seconds: int | None = None,
        eta_confidence: str | None = None,
    ) -> None:
        """Liveness + "current page/stage" for the UI. Never touches updated_at."""

        assignments = ["heartbeat_at = ?"]
        values: list[object] = [_iso(_utc_now())]
        if stage:
            assignments.append("current_stage = ?")
            values.append(stage[:80])
        if page_number is not None:
            assignments.append("current_page_number = ?")
            values.append(int(page_number))
        if current_batch is not None:
            assignments.append("current_batch = ?")
            values.append(int(current_batch))
        if batch_count is not None:
            assignments.append("batch_count = ?")
            values.append(int(batch_count))
        if eta_seconds is not None:
            assignments.append("eta_seconds = ?")
            values.append(int(eta_seconds))
        if eta_confidence is not None:
            assignments.append("eta_confidence = ?")
            values.append(eta_confidence[:20])
        values.append(task_id)
        with self._database._connection() as connection:
            connection.execute(
                f"UPDATE long_read_task SET {', '.join(assignments)} WHERE id = ?",
                values,
            )

    def set_batch_plan(self, task_id: int, *, batch_size: int, batch_count: int) -> None:
        with self._database._connection() as connection:
            connection.execute(
                "UPDATE long_read_task SET batch_size = ?, batch_count = ? WHERE id = ?",
                (int(batch_size), int(batch_count), task_id),
            )

    # ------------------------------------------------------- control paths
    def request_control(self, task_id: int, control: str) -> LongReadTask:
        """UI leaves a pause/cancel request in the mailbox of a running task."""

        if control not in CONTROL_REQUESTS:
            raise LongReadError(f"控制指令无效：{control}")
        task = self.get_task(task_id)
        if task.terminal:
            raise LongReadError("任务已结束，不能再发送控制指令。")
        with self._database._connection() as connection:
            connection.execute(
                "UPDATE long_read_task SET control = ? WHERE id = ?",
                (control, task_id),
            )
        return self.get_task(task_id)

    def consume_control(self, task_id: int) -> str:
        """Worker reads and clears the mailbox; returns '' when empty."""

        with self._database._connection() as connection:
            row = connection.execute(
                "SELECT control FROM long_read_task WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise LongReadError(f"读取任务不存在：{task_id}")
            control = str(row["control"] or "")
            if control:
                connection.execute(
                    "UPDATE long_read_task SET control = '' WHERE id = ?",
                    (task_id,),
                )
        return control

    def pause_running(self, task_id: int, *, by_user: bool, message: str = "") -> LongReadTask:
        """Worker-side safe pause after the current model call finished."""

        status = "paused_user" if by_user else "paused_system"
        self._set_task_status(
            task_id, status, error_message=message or None, finished=False
        )
        return self.recount(task_id)

    def resume_task(self, task_id: int) -> LongReadTask:
        """Explicit user 继续读取: paused/interrupted/failed -> queued."""

        task = self.get_task(task_id)
        if task.status not in ("paused_user", "paused_system", "interrupted", "failed"):
            raise LongReadError(f"当前状态（{task.status}）不能继续读取。")
        updated = self._set_task_status(
            task_id, "queued", error_message="", finished=False
        )
        with self._database._connection() as connection:
            connection.execute(
                "UPDATE long_read_task SET finished_at = NULL, eta_seconds = NULL,"
                " eta_confidence = '' WHERE id = ?",
                (task_id,),
            )
            # Pages left mid-flight by the dead/paused worker go back to
            # queued so resume reprocesses them (done pages are untouched).
            connection.execute(
                """
                UPDATE long_read_page SET state = 'queued', updated_at = ?
                WHERE task_id = ? AND state IN
                    ('extracting', 'vision', 'summarizing', 'writing')
                """,
                (_iso(_utc_now()), task_id),
            )
        return self.recount(updated.id)

    def cancel_task(self, task_id: int, *, message: str = "") -> LongReadTask:
        """Cancel != delete: completed results stay, scheduling stops.

        A running task gets the cancel mailbox flag (the worker stops before
        the next model call); any other non-terminal status flips to
        ``cancelled`` immediately.
        """

        task = self.get_task(task_id)
        if task.terminal:
            return task
        if task.status == "running":
            return self.request_control(task_id, "cancel")
        cancelled = self._set_task_status(
            task_id,
            "cancelled",
            error_message=message or "用户取消了读取；已完成的结果全部保留。",
        )
        return self.recount(cancelled.id)

    def finish_cancelled(self, task_id: int, *, message: str = "") -> LongReadTask:
        """Worker-side finalization after consuming a cancel request."""

        cancelled = self._set_task_status(
            task_id,
            "cancelled",
            error_message=message or "用户取消了读取；已完成的结果全部保留。",
        )
        return self.recount(cancelled.id)

    def add_reused(self, task_id: int, count: int) -> None:
        """Accumulate pages whose existing valid reading was reused."""

        if count <= 0:
            return
        with self._database._connection() as connection:
            connection.execute(
                "UPDATE long_read_task SET reused_pages = reused_pages + ? WHERE id = ?",
                (int(count), task_id),
            )

    def fail_task(self, task_id: int, *, message: str) -> LongReadTask:
        task = self._set_task_status(task_id, "failed", error_message=message)
        return self.recount(task.id)

    def complete_task(self, task_id: int) -> LongReadTask:
        self.recount(task_id)
        return self._set_task_status(task_id, "complete", error_message="")

    def mark_interrupted(self, task_id: int, *, message: str) -> LongReadTask:
        task = self.get_task(task_id)
        if task.status != "running":
            return task
        interrupted = self._set_task_status(
            task_id, "interrupted", error_message=message, finished=False
        )
        with self._database._connection() as connection:
            connection.execute(
                """
                UPDATE long_read_page SET state = 'queued', updated_at = ?
                WHERE task_id = ? AND state IN
                    ('extracting', 'vision', 'summarizing', 'writing')
                """,
                (_iso(_utc_now()), task_id),
            )
        return self.recount(interrupted.id)

    def recover_stale_running(
        self, *, stale_seconds: float = STALE_HEARTBEAT_SECONDS
    ) -> list[LongReadTask]:
        """Crash recovery: running tasks with a dead heartbeat -> interrupted.

        Called on worker start and on import-page load. Completed page
        results are kept; mid-flight pages return to ``queued``; the UI then
        offers 继续读取 / 取消读取.
        """

        recovered: list[LongReadTask] = []
        with self._database._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM long_read_task WHERE status = 'running'"
            ).fetchall()
        for row in rows:
            task = _task_from_row(row)
            beat = _parse_iso(task.heartbeat_at) or _parse_iso(task.updated_at)
            if beat is None:
                stale = True
            else:
                age = (_utc_now() - beat).total_seconds()
                stale = age > stale_seconds
            if not stale:
                continue
            recovered.append(
                self.mark_interrupted(
                    task.id,
                    message=(
                        "读取进程已停止（应用关闭或异常退出）；"
                        f"最后心跳 {task.heartbeat_at or '无'}。"
                        "已完成的页面结果全部保留，可点「继续读取」。"
                    ),
                )
            )
        if recovered:
            LOGGER.info(
                "长文档读取恢复：标记 %s 个中断任务（心跳超时）", len(recovered)
            )
        return recovered

    # -------------------------------------------------------------- delete
    def delete_task(self, task_id: int) -> None:
        """Delete task bookkeeping only — never documents or reading results.

        Refused while the task is running or claimed: cancel first. This
        keeps 取消 (stop scheduling, keep results) and 删除 (remove the
        bookkeeping rows) strictly separated.
        """

        task = self.get_task(task_id)
        if task.status in ("running", "queued"):
            raise LongReadError(
                "任务还在调度中，请先取消读取，再删除任务记录。"
            )
        with self._database._connection() as connection:
            connection.execute(
                "DELETE FROM long_read_page WHERE task_id = ?", (task_id,)
            )
            connection.execute(
                "DELETE FROM long_read_task WHERE id = ?", (task_id,)
            )
        LOGGER.info("长文档读取任务记录已删除：task=%s（资料与阅读结果保留）", task_id)


def _task_from_row(row) -> LongReadTask:
    return LongReadTask(
        id=int(row["id"]),
        batch_id=str(row["batch_id"] or ""),
        queue_entry_id=(
            int(row["queue_entry_id"]) if row["queue_entry_id"] is not None else None
        ),
        document_id=int(row["document_id"]),
        document_sha256=str(row["document_sha256"] or ""),
        status=str(row["status"]),
        control=str(row["control"] or ""),
        total_pages=int(row["total_pages"]),
        done_pages=int(row["done_pages"]),
        reused_pages=int(row["reused_pages"]),
        pending_pages=int(row["pending_pages"]),
        failed_pages=int(row["failed_pages"]),
        current_batch=int(row["current_batch"]),
        batch_count=int(row["batch_count"]),
        batch_size=int(row["batch_size"]),
        current_page_number=(
            int(row["current_page_number"])
            if row["current_page_number"] is not None
            else None
        ),
        current_stage=str(row["current_stage"] or ""),
        model=str(row["model"] or ""),
        worker_pid=int(row["worker_pid"]) if row["worker_pid"] is not None else None,
        created_at=str(row["created_at"]),
        started_at=str(row["started_at"]) if row["started_at"] else None,
        heartbeat_at=str(row["heartbeat_at"]) if row["heartbeat_at"] else None,
        updated_at=str(row["updated_at"]),
        finished_at=str(row["finished_at"]) if row["finished_at"] else None,
        queued_ms=int(row["queued_ms"]),
        processing_ms=int(row["processing_ms"]),
        system_paused_ms=int(row["system_paused_ms"]),
        user_paused_ms=int(row["user_paused_ms"]),
        stage_ms_json=str(row["stage_ms_json"] or "{}"),
        eta_seconds=(
            int(row["eta_seconds"]) if row["eta_seconds"] is not None else None
        ),
        eta_confidence=str(row["eta_confidence"] or ""),
        error_message=str(row["error_message"] or ""),
    )


def _page_from_row(row) -> LongReadPage:
    return LongReadPage(
        id=int(row["id"]),
        task_id=int(row["task_id"]),
        page_id=int(row["page_id"]),
        page_number=int(row["page_number"]),
        state=str(row["state"]),
        batch_index=int(row["batch_index"]),
        retry_count=int(row["retry_count"]),
        error_code=str(row["error_code"] or ""),
        error_message=str(row["error_message"] or ""),
        vision_needed=bool(row["vision_needed"]),
        vision_reason=str(row["vision_reason"] or ""),
        vision_analysis=str(row["vision_analysis"] or ""),
        source_text_sha256=str(row["source_text_sha256"] or ""),
        stage_ms_json=str(row["stage_ms_json"] or "{}"),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


__all__ = [
    "ACTIVE_PAGE_STATES",
    "CONTROL_REQUESTS",
    "LongReadError",
    "LongReadPage",
    "LongReadStore",
    "LongReadTask",
    "PAGE_STATE_LABELS",
    "PAGE_STATES",
    "PAGE_TRANSITIONS",
    "STALE_HEARTBEAT_SECONDS",
    "TASK_STATUSES",
    "TASK_STATUS_LABELS",
    "TASK_TRANSITIONS",
    "TaskTimingView",
    "timing_view",
]
