"""Development hold and explicit local task entry points; no import-time I/O."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .models import StudyError

DEVELOPMENT_END = datetime(2026, 11, 11, 12, 10, tzinfo=UTC)
COMPONENTS = (
    "nectivon-import-core",
    "nectivon-evidence-core",
    "nectivon-search-core",
    "nectivon-export-core",
    "nectivon-model-adapter",
    "nectivon-task-engine",
)


def hold_reason() -> str:
    """Time alone cannot release the hold; explicit post-exam human authorization is required."""
    if datetime.now(UTC) < DEVELOPMENT_END:
        return "考前开发冻结中：2026-11-11 20:10（北京时间）前不运行新功能或验收。"
    if os.environ.get("EKB_STUDY_LOOP_ACCEPTANCE_CONFIRMED") != "yes":
        return "等待用户确认启动测试与验收，当前不初始化学习库。"
    return ""


def require_released() -> None:
    reason = hold_reason()
    if reason:
        raise StudyError(reason)


def dependencies() -> dict[str, str]:
    """Inspect installed metadata only; do not install or import candidates automatically."""
    found = {}
    for package in COMPONENTS:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = "未安装"
    return found


def queue(root: Path):
    """Construct a separate candidate Task Engine database after the hold is released."""
    from nectivon_tasks import Queue

    from .storage import StudyStore

    require_released()
    store = StudyStore(root)
    return Queue(store.path("tasks.sqlite3"), max_active=1)


def submit(root: Path, kind: str, payload: dict, key: str):
    """Persist one task only after a UI action, never on view or rerun."""
    from .worker import registry

    require_released()
    return registry(root).submit(
        queue(root),
        kind,
        1,
        payload,
        idempotency_key=key,
        # The engine requires exact completion counts for a declared total.
        # Review branches have bounded calls but a variable number of steps.
        total=(
            None if kind == "study_model_review"
            else len(payload["source_ids"]) if kind == "study_import_batch"
            else 1
        ),
    )


def start_import_batch(root: Path, source_ids: list[str]):
    """Launch one finite local import batch without dispatching unrelated model work."""
    from uuid import uuid4

    selected = list(dict.fromkeys(source_ids))
    if not selected or len(selected) > 100:
        raise StudyError("每次请选择 1 至 100 份材料。")
    tasks = queue(root)
    if any(tasks.list(state=state, limit=1)
           for state in ("running", "pause_requested", "cancel_requested")):
        raise StudyError("已有任务正在处理，请稍后再次导入；所选材料已保存在本地。")
    task = submit(root, "study_import_batch", {"source_ids": selected}, "batch:" + uuid4().hex)
    launch_once(root, kind="study_import_batch", task_id=task.id)
    return task


def launch_once(root: Path, *, kind: str | None = None, task_id: str | None = None) -> None:
    """Run one queued task in a hidden child main thread, avoiding MuPDF thread conflicts."""
    from .storage import StudyStore

    require_released()
    root = StudyStore(root).root
    if (kind, task_id) != (None, None) and (
        kind != "study_import_batch" or not task_id
        or len(task_id) != 32 or any(c not in "0123456789abcdef" for c in task_id)
    ):
        raise StudyError("导入任务无效。")
    arguments = [sys.executable, "-m", "src.study_loop.worker", "--root", str(root)]
    if kind:
        arguments.extend(["--kind", kind, "--task-id", task_id])
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    subprocess.Popen(
        arguments,
        cwd=Path(__file__).resolve().parents[2],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        **kwargs,
    )
