"""Append-oriented sidecar store, never a connection to the application's learning DB."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from src.config import configured_platform_storage_paths
from src.storage_boundaries import (
    StorageBoundaryError,
    check_storage_paths,
    claim_storage_paths,
    local_directory,
)

from .models import KINDS, Draft, Page, Source, StudyError, now, validate_draft, validate_page_ids

MARKER = "nectivon-study-loop-v1"
MAX_BYTES = 64 * 1024 * 1024


def encode(value: object) -> str:
    """Stable JSON for snapshots and idempotency comparisons."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def private_root(path: Path) -> Path:
    """Require an explicitly chosen local directory outside any source checkout."""
    try:
        root = local_directory(path)
        check_storage_paths("study", (root,))
        for platform_root in configured_platform_storage_paths():
            platform_root = local_directory(platform_root)
            if (
                root == platform_root
                or root in platform_root.parents
                or platform_root in root.parents
            ):
                raise StudyError("学习整理目录必须与平台资料、备份及运行目录分离，不能互相包含。")
    except (OSError, ValueError) as exc:
        raise StudyError("无法安全打开学习目录：" + str(exc)) from exc
    repo = Path(__file__).resolve().parents[2]
    if root == repo or repo in root.parents:
        raise StudyError("学习数据必须放在源码目录之外。")
    for parent in (root, *root.parents):
        if (parent / ".git").exists():
            raise StudyError("学习数据目录不能位于 Git 仓库内。")
    return root


class StudyStore:
    """Explicit create/open with ownership markers and no legacy migrations."""

    def __init__(self, root: Path, *, create: bool = False) -> None:
        self.root = private_root(root)
        self.db = self.root / "study.sqlite3"
        marker = self.root / ".nectivon-study-loop"
        if not marker.exists():
            if not create or (self.root.exists() and any(self.root.iterdir())):
                raise StudyError("请选择空的新目录，或此前初始化的学习整理目录。")
            self._claim_root()
            self.root.mkdir(parents=True, exist_ok=True)
            with marker.open("x", encoding="utf-8") as stream:
                stream.write(MARKER)
            for name in ("originals", "imports", "exports"):
                (self.root / name).mkdir()
            with sqlite3.connect(self.db) as connection:
                connection.executescript(
                    """
                    PRAGMA application_id = 1314081841;
                    PRAGMA user_version = 1;
                    CREATE TABLE sources(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                    CREATE TABLE pages(id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
                        payload TEXT NOT NULL, FOREIGN KEY(source_id) REFERENCES sources(id));
                    CREATE TABLE drafts(id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                        confirmed INTEGER NOT NULL DEFAULT 0);
                    CREATE TABLE items(id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                        confirmed_at TEXT NOT NULL);
                    CREATE TABLE sessions(id TEXT PRIMARY KEY, item_ids TEXT NOT NULL,
                        started_at TEXT NOT NULL, ended_at TEXT);
                    CREATE TABLE attempts(id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                        item_id TEXT NOT NULL, unit_id TEXT NOT NULL, payload TEXT NOT NULL,
                        FOREIGN KEY(session_id) REFERENCES sessions(id),
                        FOREIGN KEY(item_id) REFERENCES items(id));
                    """
                )
        if (
            marker.is_symlink()
            or self.db.is_symlink()
            or marker.read_text(encoding="utf-8") != MARKER
            or not self.db.is_file()
        ):
            raise StudyError("目录标记或学习库缺失，未尝试修复或覆盖。")
        self._claim_root()
        with self.connect() as connection:
            if connection.execute("PRAGMA application_id").fetchone()[0] != 1314081841:
                raise StudyError("该文件不属于学习整理库，拒绝打开。")
            if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise StudyError("学习整理库版本不兼容，原库保持不变。")

    def _deleted_source_ids(self) -> set[str]:
        """Read an optional trash index without migrating a library on browse."""
        with self.connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='deleted_sources'"
            ).fetchone()
            if not exists:
                return set()
            return {row[0] for row in connection.execute("SELECT source_id FROM deleted_sources")}

    def is_source_deleted(self, source_id: str) -> bool:
        return source_id in self._deleted_source_ids()

    def delete_source(self, source_id: str, *, user_confirmed: bool) -> None:
        """Move a material and its study entries out of use, preserving recoverability."""
        if user_confirmed is not True:
            raise StudyError("请确认删除这份材料。")
        self.source(source_id)
        with self.connect(write=True) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS deleted_sources("
                "source_id TEXT PRIMARY KEY, deleted_at TEXT NOT NULL,"
                "FOREIGN KEY(source_id) REFERENCES sources(id))"
            )
            connection.execute(
                "INSERT OR IGNORE INTO deleted_sources VALUES (?,?)", (source_id, now())
            )

    def deleted_sources(self) -> list[Source]:
        """List recoverable materials without reading their private contents."""
        deleted = self._deleted_source_ids()
        with self.connect() as connection:
            return [
                Source(**json.loads(row[0]))
                for row in connection.execute("SELECT payload FROM sources ORDER BY rowid DESC")
                if json.loads(row[0])["id"] in deleted
            ]

    def restore_source(self, source_id: str, *, user_confirmed: bool) -> None:
        """Restore exactly the selected material and its original study history."""
        if user_confirmed is not True:
            raise StudyError("请确认恢复这份材料。")
        if source_id not in self._deleted_source_ids():
            raise StudyError("这份材料不在最近删除中。")
        self.read_source(self.source(source_id))
        with self.connect(write=True) as connection:
            connection.execute("DELETE FROM deleted_sources WHERE source_id=?", (source_id,))

    def _claim_root(self) -> None:
        """Reserve both domains before any store or component writes can occur."""
        try:
            claim_storage_paths("platform", configured_platform_storage_paths())
            claim_storage_paths("study", (self.root,))
        except (OSError, ValueError) as exc:
            raise StudyError("学习目录隔离检查未完成：" + str(exc)) from exc

    @contextmanager
    def connect(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        """Open only the existing owned DB; each operation owns its transaction."""
        try:
            check_storage_paths("study", (self.root,))
            self.path("study.sqlite3")
        except StorageBoundaryError as exc:
            raise StudyError(str(exc)) from exc
        connection = sqlite3.connect(self.db.as_uri() + "?mode=rw", uri=True, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                if write:
                    connection.execute("BEGIN IMMEDIATE")
                yield connection
        finally:
            connection.close()

    def path(self, relative: str) -> Path:
        """Resolve a stored asset without accepting outside paths or symlinks."""
        value = self.root / relative
        if Path(relative).is_absolute() or not relative or value.is_symlink():
            raise StudyError("来源路径无效。")
        try:
            result = local_directory(value)
        except (OSError, ValueError) as exc:
            raise StudyError("来源路径隔离检查失败。") from exc
        if self.root not in result.parents:
            raise StudyError("来源路径超出学习整理目录。")
        return result

    def receive(self, name: str, data: bytes, kind: str, *, transcript: bool = False) -> Source:
        """Save original bytes once, SHA-256 deduplicate and never overwrite an original."""
        if kind not in KINDS or not data or len(data) > MAX_BYTES:
            raise StudyError("输入类型无效、内容为空或文件超过 64 MiB。")
        digest = hashlib.sha256(data).hexdigest()
        identity = f"{kind}-{digest}"
        if self.is_source_deleted(identity):
            raise StudyError("这份文件在最近删除中，请先恢复。")
        suffix = Path(name).suffix.lower()
        extensions = {
            "image": {".png", ".jpg", ".jpeg"},
            "pdf": {".pdf"},
            "docx": {".docx"},
            "txt": {".txt"},
            "md": {".md", ".markdown"},
            "json": {".json"},
            "html": {".html", ".htm"},
            "manual": {".txt"},
            "voice": {".wav"},
        }
        if suffix not in extensions[kind]:
            raise StudyError("文件扩展名不支持。")
        with self.connect(write=True) as connection:
            prior = connection.execute(
                "SELECT payload FROM sources WHERE id=?", (identity,)
            ).fetchone()
            if prior:
                source = Source(**json.loads(prior[0]))
                self.read_source(source)
                if source.transcript != transcript:
                    raise StudyError(
                        "相同原件已按另一种对话标记导入，原记录未改变；"
                        "请在已有草稿中人工整理，不能静默改变历史解析方式。"
                    )
                return source
            relative = f"originals/{identity}{suffix}"
            target = self.path(relative)
            try:
                with target.open("xb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
            except FileExistsError:
                if target.read_bytes() != data:
                    raise StudyError("同名原件内容异常，已保留现场。") from None
            source = Source(
                identity, kind, Path(name).name, digest, relative, transcript=transcript
            )
            connection.execute(
                "INSERT INTO sources VALUES (?,?)", (identity, encode(asdict(source)))
            )
            return source

    def read_source(self, source: Source) -> bytes:
        """Read and compare against the historical receipt hash."""
        data = self.path(source.blob).read_bytes()
        if hashlib.sha256(data).hexdigest() != source.sha256:
            raise StudyError("原始材料已改变，停止处理；请保留原件并人工核对。")
        return data

    def source(self, source_id: str) -> Source:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload FROM sources WHERE id=?", (source_id,)
            ).fetchone()
        if row is None:
            raise StudyError("找不到材料来源。")
        return Source(**json.loads(row[0]))

    def sources(self) -> list[Source]:
        deleted = self._deleted_source_ids()
        with self.connect() as connection:
            return [
                Source(**json.loads(row[0]))
                for row in connection.execute("SELECT payload FROM sources ORDER BY rowid DESC")
                if json.loads(row[0])["id"] not in deleted
            ]

    def put_page(self, page: Page) -> None:
        """Idempotent immutable page snapshot; parser changes require a new version."""
        with self.connect(write=True) as connection:
            old = connection.execute("SELECT payload FROM pages WHERE id=?", (page.id,)).fetchone()
            payload = encode(asdict(page))
            if old and old[0] != payload:
                raise StudyError("同一来源段出现不同解析结果，未覆盖原记录。")
            connection.execute(
                "INSERT OR IGNORE INTO pages VALUES (?,?,?)",
                (
                    page.id,
                    page.source_id,
                    payload,
                ),
            )

    def pages(self, source_id: str) -> list[Page]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM pages WHERE source_id=? ORDER BY rowid",
                (source_id,),
            )
            return [Page(**json.loads(row[0])) for row in rows]

    def referenced_pages(self, source_id: str, page_ids: list[str]) -> list[Page]:
        """Resolve an exact, bounded source scope; never silently omit invalid references."""
        validate_page_ids(page_ids)
        pages = {page.id: page for page in self.pages(source_id)}
        if not set(page_ids) <= pages.keys():
            raise StudyError("题目引用了不属于该材料的来源段。")
        return [pages[identity] for identity in page_ids]

    def add_draft(self, draft: Draft) -> None:
        """Keep the first extraction immutable, including unconfirmed observed answers."""
        with self.connect(write=True) as connection:
            old = connection.execute(
                "SELECT payload FROM drafts WHERE id=?", (draft.id,)
            ).fetchone()
            payload = encode(draft.to_dict())
            if old and encode(Draft.from_dict(json.loads(old[0])).to_dict()) != payload:
                raise StudyError("同一草稿出现不同内容，原草稿已保留；请创建独立修订。")
            connection.execute(
                "INSERT OR IGNORE INTO drafts(id,payload) VALUES (?,?)",
                (
                    draft.id,
                    payload,
                ),
            )

    def drafts(self) -> list[Draft]:
        deleted = self._deleted_source_ids()
        with self.connect() as connection:
            return [
                Draft.from_dict(json.loads(row[0]))
                for row in connection.execute(
                    "SELECT payload FROM drafts WHERE confirmed=0 ORDER BY rowid"
                )
                if json.loads(row[0])["source_id"] not in deleted
            ]

    def confirm(self, draft: Draft, *, user_confirmed: bool) -> None:
        """Commit a human-edited snapshot once, keeping the original extraction separately."""
        if user_confirmed is not True:
            raise StudyError("请核对原始材料后确认保存。")
        if self.is_source_deleted(draft.source_id):
            raise StudyError("这份材料已删除，请先恢复材料。")
        validate_draft(draft)
        self.read_source(self.source(draft.source_id))
        self.referenced_pages(draft.source_id, draft.page_ids)
        with self.connect(write=True) as connection:
            original = connection.execute(
                "SELECT payload FROM drafts WHERE id=?",
                (draft.id,),
            ).fetchone()
            if original is None or json.loads(original[0])["source_id"] != draft.source_id:
                raise StudyError("草稿不存在或来源不匹配。")
            extraction = Draft.from_dict(json.loads(original[0]))
            if extraction.revision_of != draft.revision_of:
                raise StudyError("修订来源与原草稿不一致，未保存。")
            linked_ids = set(draft.conflict_item_ids)
            if draft.revision_of:
                linked_ids.add(draft.revision_of)
            for linked_id in linked_ids:
                if (
                    linked_id == draft.id
                    or connection.execute("SELECT 1 FROM items WHERE id=?", (linked_id,)).fetchone()
                    is None
                ):
                    raise StudyError("修订或冲突引用必须指向已保存的其他材料。")
            payload = encode(draft.to_dict())
            prior = connection.execute(
                "SELECT payload FROM items WHERE id=?", (draft.id,)
            ).fetchone()
            if prior and encode(Draft.from_dict(json.loads(prior[0])).to_dict()) != payload:
                raise StudyError("已保存版本不会被覆盖；请另建修订草稿。")
            connection.execute(
                "INSERT OR IGNORE INTO items VALUES (?,?,?)", (draft.id, payload, now())
            )
            connection.execute("UPDATE drafts SET confirmed=1 WHERE id=?", (draft.id,))

    def create_revision(self, item_id: str, *, user_requested: bool) -> Draft:
        """Append a pending revision; preserve the prior item, source and attempts verbatim."""
        if user_requested is not True:
            raise StudyError("请主动选择需要修订的材料。")
        original = self.item(item_id)
        if self.is_source_deleted(original.source_id):
            raise StudyError("这份材料已删除，请先恢复材料。")
        self.read_source(self.source(original.source_id))
        revision = deepcopy(original)
        revision.id = uuid4().hex
        revision.revision_of = original.id
        revision.verification_status = "pending"
        revision.verification_note = ""
        revision.extraction = "user_revision"
        revision.warnings = [*original.warnings, "此为人工修订草稿；原版本与历史作答保持不变。"]
        self.add_draft(revision)
        return revision

    def items(self) -> list[Draft]:
        deleted = self._deleted_source_ids()
        with self.connect() as connection:
            return [
                Draft.from_dict(json.loads(row[0]))
                for row in connection.execute("SELECT payload FROM items ORDER BY rowid")
                if json.loads(row[0])["source_id"] not in deleted
            ]

    def item(self, item_id: str) -> Draft:
        with self.connect() as connection:
            row = connection.execute("SELECT payload FROM items WHERE id=?", (item_id,)).fetchone()
        if row is None:
            raise StudyError("题目不存在。")
        return Draft.from_dict(json.loads(row[0]))

    def start_review(self, item_ids: list[str], *, user_started: bool) -> str:
        """Create a session only on a human action; there are no schedules."""
        known = {item.id for item in self.items()}
        selected = list(dict.fromkeys(item_ids))
        if user_started is not True or not selected or not set(selected) <= known:
            raise StudyError("请主动选择有效内容并开始复盘。")
        identity = uuid4().hex
        with self.connect(write=True) as connection:
            connection.execute(
                "INSERT INTO sessions VALUES (?,?,?,NULL)",
                (
                    identity,
                    encode(selected),
                    now(),
                ),
            )
        return identity

    def sessions(self) -> list[dict]:
        with self.connect() as connection:
            return [
                dict(row)
                for row in connection.execute("SELECT * FROM sessions ORDER BY rowid DESC")
            ]

    def end_review(self, session_id: str) -> None:
        """Explicitly end a session without reminders or subsequent tasks."""
        with self.connect(write=True) as connection:
            connection.execute(
                "UPDATE sessions SET ended_at=COALESCE(ended_at,?) WHERE id=?",
                (now(), session_id),
            )

    def record_attempt(
        self,
        session_id: str,
        item_id: str,
        unit_id: str,
        *,
        outcome: str,
        answer: str,
        note: str,
        independent: bool,
        attempted: bool,
        error_tags: list[str],
        submission_id: str,
        error_confirmed: bool | None = None,
    ) -> None:
        """Record actual user work, never an AI explanation or an inferred answer."""
        item = self.item(item_id)
        if self.is_source_deleted(item.source_id):
            raise StudyError("这份材料已删除，请先恢复材料。")
        units = {unit.id: unit for unit in item.question.units()}
        if unit_id not in units or outcome not in {"not_yet", "partial", "correct"}:
            raise StudyError("小问或结果无效。")
        if attempted is not True or not (answer.strip() or note.strip()):
            raise StudyError("请先实际尝试，并记录作答或无法作答的具体障碍。")
        if outcome == "correct" and (independent is not True or not answer.strip()):
            raise StudyError("独立正确需要独立作答及实际答案，讲解记录不能代替。")
        if error_confirmed is not None and type(error_confirmed) is not bool:
            raise StudyError("作答错误确认状态无效。")
        if error_confirmed is True and (outcome == "correct" or not answer.strip()):
            raise StudyError("确认作答错误需要实际作答文字，且不能同时记录为独立正确。")
        if not submission_id or len(submission_id) > 128:
            raise StudyError("提交标识无效。")
        payload = {
            "session_id": session_id,
            "item_id": item_id,
            "unit_id": unit_id,
            "outcome": outcome,
            "answer": answer,
            "note": note,
            "independent": independent,
            "error_tags": list(dict.fromkeys(t.strip() for t in error_tags if t.strip())),
            "source": "user_attempt",
            "subject": item.subject,
            "chapter": item.chapter,
            "knowledge": units[unit_id].knowledge or item.question.knowledge,
            "page_ids": item.page_ids,
        }
        # Optional metadata only: old snapshots and idempotent old submissions stay unchanged.
        if error_confirmed is not None:
            payload["error_confirmed"] = error_confirmed
        with self.connect(write=True) as connection:
            session = connection.execute(
                "SELECT * FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
            if not session or item_id not in json.loads(session["item_ids"]):
                raise StudyError("该题不属于本次主动复盘。")
            prior = connection.execute(
                "SELECT payload FROM attempts WHERE id=?",
                (submission_id,),
            ).fetchone()
            if prior:
                previous = json.loads(prior[0])
                previous.pop("created_at")
                if previous != payload:
                    raise StudyError("同一次提交的内容发生冲突，未覆盖历史作答。")
                return
            if session["ended_at"]:
                raise StudyError("复盘已结束，请主动开始新一轮。")
            payload["created_at"] = now()
            connection.execute(
                "INSERT INTO attempts VALUES (?,?,?,?,?)",
                (
                    submission_id,
                    session_id,
                    item_id,
                    unit_id,
                    encode(payload),
                ),
            )

    def attempts(self) -> list[dict]:
        with self.connect() as connection:
            return [
                json.loads(row[0])
                for row in connection.execute("SELECT payload FROM attempts ORDER BY rowid")
            ]

    def attempt_records(self) -> list[dict]:
        """Read stable event identities without changing the historical payload contract.

        Identity columns are authoritative; a damaged payload is flagged for the
        read-only view instead of being silently linked to a different material.
        """
        with self.connect() as connection:
            records = []
            for row in connection.execute("SELECT * FROM attempts ORDER BY rowid"):
                payload = json.loads(row["payload"])
                identities = {key: row[key] for key in ("session_id", "item_id", "unit_id")}
                records.append(
                    {
                        **payload,
                        **identities,
                        "id": row["id"],
                        "identity_valid": all(payload.get(k) == v for k, v in identities.items()),
                    }
                )
            return records
