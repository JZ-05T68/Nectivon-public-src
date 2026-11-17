"""Active-only local import progress; no model calls or task dispatch on refresh."""

from __future__ import annotations

import logging
import time

import streamlit as st

from .runtime import queue
from .storage import StudyStore

LOGGER = logging.getLogger(__name__)
IMPORT_TASK_KEY = "sl_import_batch_task_id"
_NOTICE_KEY = "sl_import_batch_notice"
_PENDING = {"queued", "running", "pause_requested", "cancel_requested"}


def _lease_expired(task) -> bool:
    """Read the public lease timestamp without recovering or mutating any task."""
    expiry = getattr(task, "lease_until", None)
    return (
        task.state in {"running", "pause_requested", "cancel_requested"}
        and type(expiry) in (int, float) and expiry <= time.time()
    )


def _terminal_notice(store: StudyStore, task) -> dict:
    """Describe the outcome without exposing task identifiers or engine error codes."""
    if task.state == "completed":
        result = task.result if isinstance(task.result, dict) else {}
        failures = result.get("failed_files", [])
        names = []
        for failure in failures:
            try:
                names.append(store.source(failure["source_id"]).name)
            except Exception as exc:  # noqa: BLE001 - status display must not break the study page
                LOGGER.warning("study_import_progress filename_unavailable type=%s",
                               type(exc).__name__)
                names.append("名称暂时无法读取的材料")
        failed = result.get("failed_file_count", len(names))
        count = result.get("file_count", 0)
        if failed:
            return {"level": "warning", "message": f"已导入 {max(0, count - failed)} 份材料；"
                    f"{failed} 份未能导入。", "names": names}
        return {"level": "success", "message": "材料已导入。", "names": []}
    if task.state == "cancelled":
        return {"level": "info", "message": "导入已取消。", "names": []}
    if task.state == "paused":
        return {"level": "info", "message": "导入已暂停，可在高级设置中继续。", "names": []}
    if _lease_expired(task):
        return {"level": "warning", "message": "导入已中断，请在高级设置中查看。", "names": []}
    return {"level": "warning", "message": "导入未完成，请重新导入所选材料。", "names": []}


def _finish(identity: str, notice: dict) -> None:
    """Detach the timer before requesting one complete refresh of the material list."""
    if st.session_state.get(IMPORT_TASK_KEY) != identity:
        return
    st.session_state.pop(IMPORT_TASK_KEY, None)
    st.session_state[_NOTICE_KEY] = notice
    st.rerun(scope="app")


def _read_task(store: StudyStore, identity: str):
    """Only inspect the explicitly submitted local batch; never claim or execute a task."""
    task = queue(store.root).get(identity)
    if task.kind != "study_import_batch":
        raise ValueError("Unexpected import task kind")
    return task


def _poll_once(store: StudyStore, identity: str) -> None:
    if st.session_state.get(IMPORT_TASK_KEY) != identity:
        return
    try:
        task = _read_task(store, identity)
    except Exception as exc:  # noqa: BLE001 - keep low-level status failures out of the UI
        LOGGER.warning("study_import_progress read_failed type=%s", type(exc).__name__)
        _finish(identity, {"level": "warning", "message": "导入状态暂时无法读取，请刷新页面。",
                           "names": []})
        return
    if task.state not in _PENDING or _lease_expired(task):
        _finish(identity, _terminal_notice(store, task))
        return
    ratio = min(1.0, max(0.0, task.done / task.total)) if task.total else 0.0
    st.progress(ratio, text="正在导入学习材料…")


def render_import_progress(store: StudyStore) -> None:
    """Install a one-second fragment only while the selected local import is pending."""
    notice = st.session_state.pop(_NOTICE_KEY, None)
    if notice:
        renderer = {"success": st.success, "warning": st.warning, "info": st.info}
        renderer[notice["level"]](notice["message"])
        if notice["names"]:
            st.text("、".join(notice["names"]))
    identity = st.session_state.get(IMPORT_TASK_KEY)
    if not identity:
        return
    try:
        initial = _read_task(store, identity)
    except Exception as exc:  # noqa: BLE001 - no raw paths, codes or provider data on the page
        LOGGER.warning("study_import_progress read_failed type=%s", type(exc).__name__)
        _finish(identity, {"level": "warning", "message": "导入状态暂时无法读取，请刷新页面。",
                           "names": []})
        return
    if initial.state not in _PENDING or _lease_expired(initial):
        _finish(identity, _terminal_notice(store, initial))
        return

    def progress() -> None:
        _poll_once(store, identity)

    st.fragment(run_every="1s")(progress)()
