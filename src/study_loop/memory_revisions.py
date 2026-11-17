"""Append-only confirmed-memory revisions; identifiers and original provenance stay stable."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from .models import StudyError, now

if TYPE_CHECKING:
    from .chat_store import ChatStore


def initialize(db) -> None:
    """Add history and baseline old content atomically, without altering existing memory rows."""
    if db.execute("SELECT 1 FROM sqlite_master WHERE name='memory_revisions'").fetchone():
        return
    if not db.in_transaction:
        db.execute("BEGIN IMMEDIATE")
    db.execute("""CREATE TABLE IF NOT EXISTS memory_revisions(
        memory_id TEXT NOT NULL, revision INTEGER NOT NULL, content TEXT NOT NULL,
        created_at TEXT NOT NULL, action TEXT NOT NULL, restored_from_revision INTEGER,
        PRIMARY KEY(memory_id, revision), FOREIGN KEY(memory_id) REFERENCES memories(id))""")
    db.execute("""INSERT OR IGNORE INTO memory_revisions
        SELECT id,1,content,created_at,'confirmed',NULL FROM memories""")


def baseline(db, memory: dict) -> None:
    """Preserve an imported legacy row before its first edit or confirmation link."""
    db.execute("INSERT OR IGNORE INTO memory_revisions VALUES (?,?,?,?,?,?)",
               (memory["id"], 1, memory["content"], memory["created_at"], "confirmed", None))


def read(db, identity: str) -> dict:
    row = db.execute("SELECT * FROM memories WHERE id=?", (identity,)).fetchone()
    if row is None:
        raise StudyError("知识记忆不存在。")
    memory = dict(row)
    latest = db.execute(
        "SELECT revision,created_at FROM memory_revisions WHERE memory_id=? "
        "ORDER BY revision DESC LIMIT 1", (identity,),
    ).fetchone()
    memory.update(revision=latest["revision"] if latest else 1,
                  updated_at=latest["created_at"] if latest else memory["created_at"])
    return memory


class MemoryRevisions:
    """Explicit local edits never modify AI output, candidate text or confirmation snapshots."""

    def __init__(self, chats: ChatStore) -> None:
        self.chats = chats

    def history(self, identity: str) -> list[dict]:
        with self.chats.connect() as db:
            memory = read(db, identity)
            versions = [dict(row) for row in db.execute(
                "SELECT * FROM memory_revisions WHERE memory_id=? ORDER BY revision DESC",
                (identity,),
            )]
        return versions or [{"memory_id": identity, "revision": 1, "content": memory["content"],
                             "created_at": memory["created_at"], "action": "confirmed",
                             "restored_from_revision": None}]

    def update(
        self, identity: str, content: str, *, expected_revision: int,
        restored_from_revision: int | None = None,
    ) -> dict:
        if not isinstance(content, str) or not content.strip() or len(content) > 30000:
            raise StudyError("知识正文不能为空，且不能超过30000字符。")
        if type(expected_revision) is not int or expected_revision < 1:
            raise StudyError("修订号无效，请重新打开这份记忆。")
        with self.chats.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            memory = read(db, identity)
            if memory["revision"] != expected_revision:
                raise StudyError("这份记忆已在其他页面更新，请保留当前输入并重新打开后再保存。")
            collisions = db.execute(
                "SELECT id FROM memories WHERE subject=? AND content=? AND id<>?",
                (memory["subject"], content, identity),
            ).fetchall()
            # Inactive sources may later be restored. Rejecting their duplicate
            # bodies too prevents restoration from producing two active copies.
            if collisions:
                raise StudyError("本课程已有另一份记忆使用相同正文，请保留输入并打开该记忆。")
            if content != memory["content"] or restored_from_revision is not None:
                baseline(db, memory)
                db.execute("INSERT INTO memory_revisions VALUES (?,?,?,?,?,?)", (
                    identity, memory["revision"] + 1, content, now(),
                    "restored" if restored_from_revision is not None else "edited",
                    restored_from_revision,
                ))
                db.execute("UPDATE memories SET content=? WHERE id=?", (content, identity))
                saved_revision = memory["revision"] + 1
            else:
                saved_revision = memory["revision"]
        # A returned transaction-local dict does not prove the committed bytes.
        # Read both the current row and its immutable revision on a new read-only
        # connection before allowing a caller to display success.
        try:
            with sqlite3.connect(self.chats.db.as_uri() + "?mode=ro", uri=True) as check:
                check.row_factory = sqlite3.Row
                check.execute("PRAGMA query_only=ON")
                check.execute("BEGIN")
                current = read(check, identity)
                saved = check.execute(
                    "SELECT content FROM memory_revisions WHERE memory_id=? AND revision=?",
                    (identity, saved_revision),
                ).fetchone()
        except sqlite3.Error as exc:
            raise StudyError("记忆保存后的独立核对失败，输入已保留，请重新打开核对。") from exc
        if (current["revision"] != saved_revision or current["content"] != content
                or saved is None or saved["content"] != content):
            raise StudyError("记忆保存后的正文或修订号不一致，输入已保留，请重新打开核对。")
        result = self.chats.memory(identity)
        if result["revision"] != saved_revision or result["content"] != content:
            raise StudyError("这份记忆在保存后又被更新，当前输入已保留，请重新打开核对。")
        return result

    def restore(self, identity: str, revision: int, *, expected_revision: int) -> dict:
        selected = next((row for row in self.history(identity)
                         if row["revision"] == revision), None)
        if selected is None:
            raise StudyError("历史修订不存在，请重新选择。")
        return self.update(identity, selected["content"], expected_revision=expected_revision,
                           restored_from_revision=revision)
