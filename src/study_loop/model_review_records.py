"""Immutable local review artifacts beside the study store; no schema migration."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from uuid import uuid4

from .models import StudyError
from .storage import StudyStore, encode

CALL_STAGES = ("analysis_0", "analysis_1", "verify_0", "verify_1", "summary")
STAGES = {
    "input",
    "analysis_0",
    "analysis_1",
    "verify_0",
    "verify_1",
    "summary",
    "report",
    "failure",
    *(stage + "_raw" for stage in CALL_STAGES),
}
LOGGER = logging.getLogger(__name__)


def record_digest(value: dict) -> str:
    """Keep prior valid-text checkpoint hashes; also hash rejected Unicode losslessly."""
    return hashlib.sha256(encode(value).encode("utf-8", errors="surrogatepass")).hexdigest()


def _record_bytes(value: dict) -> bytes:
    # Escaping retains even malformed provider Unicode in the immutable raw record.
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def record_path(store: StudyStore, run_id: str, stage: str) -> Path:
    if re.fullmatch(r"[0-9a-f]{32}", run_id) is None or stage not in STAGES:
        raise StudyError("复盘记录标识无效。")
    return store.path(f"model-reviews/{run_id}/{stage}.json")


def save_record(store: StudyStore, run_id: str, stage: str, value: dict) -> dict:
    """Publish once; resume may reuse identical records but never rewrite raw answers."""
    path = record_path(store, run_id, stage)
    raw = _record_bytes(value)
    if path.exists():
        if _record_bytes(load_record(store, run_id, stage)) != raw:
            raise StudyError("同一次模型复盘已有不同记录，未覆盖原始答案。")
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = store.path(f"model-reviews/{run_id}/.{stage}-{uuid4().hex}.tmp")
    published = False
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory hard link provides no-overwrite publication on local NTFS.
        try:
            os.link(temporary, path)
        except FileExistsError:
            if _record_bytes(load_record(store, run_id, stage)) != raw:
                raise StudyError("复盘记录发生并发冲突，未覆盖。") from None
        published = True
    except OSError:
        raise StudyError("复盘记录写入或发布失败；已写入的暂存副本保留，未重发模型请求。") from None
    finally:
        # A failed publish may leave the only answer copy here. Never erase it.
        if published:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("model_review temporary_retained_after_publish")
    return value


def load_record(store: StudyStore, run_id: str, stage: str) -> dict | None:
    path = record_path(store, run_id, stage)
    if not path.exists():
        return None
    try:
        if path.stat().st_size > 64 * 1024 * 1024:
            raise StudyError("复盘记录过大，请保留文件并人工核对。")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, RecursionError):
        raise StudyError("复盘记录无法读取，原文件保留，请人工核对。") from None
    if not isinstance(value, dict):
        raise StudyError("复盘记录格式无效，未尝试覆盖或修复。")
    return value


def list_runs(store: StudyStore, item_id: str, unit_id: str) -> tuple[list[dict], list[str]]:
    """A damaged history must not hide other records or the task cancellation controls."""
    root = store.path("model-reviews")
    if not root.exists():
        return [], []
    results, warnings = [], []
    for path in root.iterdir():
        if re.fullmatch(r"[0-9a-f]{32}", path.name) is None:
            continue
        try:
            value = load_record(store, path.name, "input")
        except StudyError:
            warnings.append(f"记录 {path.name[:8]} 的输入无法读取，原件保留，未阻断其他记录。")
            continue
        if value and value.get("item_id") == item_id and value.get("unit_id") == unit_id:
            if (
                value.get("run_id") != path.name
                or type(value.get("created_at")) is not str
                or type(value.get("evidence_verification")) is not dict
                or type(value.get("source")) is not dict
                or any(type(value["source"].get(k)) is not str for k in ("name", "sha256"))
            ):
                warnings.append(f"记录 {path.name[:8]} 的来源字段不完整，文件保留，请人工核对。")
                continue
            results.append(value)
    return sorted(results, key=lambda v: v["created_at"], reverse=True), warnings


def pending_record_names(store: StudyStore, run_id: str) -> list[str]:
    """List retained temporary record names only; never recover or delete them implicitly."""
    root = record_path(store, run_id, "input").parent
    if not root.exists():
        return []
    return sorted(
        p.name for p in root.iterdir() if p.name.startswith(".") and p.name.endswith(".tmp")
    )
