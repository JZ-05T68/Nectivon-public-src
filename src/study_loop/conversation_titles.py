"""Local course-chat titles kept outside immutable conversations, messages and archives."""

from __future__ import annotations

import json
import re

from .chat_store import ChatStore
from .models import StudyError, now
from .storage import StudyStore


def derive_title(question: str, subject: str = "") -> str:
    """Extract a request's topic locally, discarding source scaffolding before shortening."""
    if not question.strip():
        return "新对话"
    text = re.sub(r"\$\$.*?\$\$|\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\\]", " ",
                  question, flags=re.DOTALL)
    sentences = [part.strip() for part in re.split(r"[。！？!?\n]", text) if part.strip()]
    if not sentences:
        return "课程学习问题讨论"
    topic, suffix = sentences[0], ""
    verbs = ("总结|梳理|整理|归纳|比较|对比|解释|说明|分析|推导|证明|计算|"
             "回顾|找到|查找|检索|复述|介绍|讲解")
    for index, sentence in enumerate(sentences[:8]):
        sentence = re.sub(r"^(?:(?:请你|请问|请|帮我|能否|我想了解|我想知道)\s*)+", "", sentence)
        if index == 0:
            topic = sentence
        if "为什么" in sentence:
            before, after = sentence.split("为什么", 1)
            topic = before or re.split(r"需要|必须|应当|要|会|能|是|用|采用", after, maxsplit=1)[0]
            if len(re.findall(r"[\u3400-\u9fff]", topic)) < 4:
                topic = re.sub(r"需要(?:使用)?|必须|应当|使用|采用|会", "", before + after)
            suffix = "计算依据" if "计算" in after else "原因分析"
            break
        match = re.search(r"(?:^|[，,：:]|请|帮我|系统|重点|如何|"
                          r"(?:知识记忆|记忆|资料|材料)中?)(" + verbs + r")(.+)", sentence)
        if match:
            action, topic = match.groups()
            suffix = ("对比" if action in {"比较", "对比"} else
                      "整理" if action in {"总结", "梳理", "整理", "归纳"} else
                      "回顾" if action == "回顾" else "")
            break
        if re.search(r"有哪些|是什么|怎么|如何|是否", sentence):
            topic, suffix = sentence, "整理" if "有哪些" in sentence else ""
            break
    topic = re.split(r"[，,；;：:]|并(?:请|说明|告诉|复述|列出)", topic, maxsplit=1)[0]
    topic = re.sub(r"^(?:我(?:已经|已|之前)?(?:修改并保存|修改|保存|确认|导入)(?:的)?)+", "", topic)
    topic = re.sub(r"(?:这门|本门|该门|本|该)(?:课程|科目)", subject or "课程", topic)
    topic = re.sub(r"(?:需要|应当|必须)(?:重点)?掌握的?|需要重点|题目?对应的|计算题对应的|"
                   r"对应的|之间的|有关的|相关的|的", "", topic)
    topic = re.sub(r"有哪些|是什么|如何|怎么|是否|[吗呢]$", "", topic)
    chinese = "".join(re.findall(r"[\u3400-\u9fff]", topic))
    if not chinese:
        return "课程学习问题讨论"
    title = chinese[:24 - len(suffix)] + suffix if not chinese.endswith(suffix) else chinese[:24]
    return title if len(title) >= 8 else "关于" + title + "的学习讨论"


class ConversationTitles:
    """Keep titles and recent-use metadata separate from immutable chat context."""

    def __init__(self, store: StudyStore) -> None:
        self.chats = ChatStore(store)

    @staticmethod
    def _metadata(db) -> dict[str, dict]:
        exists = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='conversation_titles'"
        ).fetchone()
        return ({row["conversation_id"]: dict(row)
                 for row in db.execute("SELECT * FROM conversation_titles")} if exists else {})

    @staticmethod
    def _first_question(db, identity: str) -> str:
        row = db.execute(
            "SELECT content FROM messages WHERE conversation_id=? AND role='user' "
            "AND trim(content)<>'' ORDER BY id LIMIT 1", (identity,),
        ).fetchone()
        return row[0] if row else ""

    @staticmethod
    def _ensure_metadata(db) -> None:
        """Migrate the optional rename-only side table without changing old renames."""
        db.execute(
            "CREATE TABLE IF NOT EXISTS conversation_titles("
            "conversation_id TEXT PRIMARY KEY,title TEXT NOT NULL,updated_at TEXT NOT NULL,"
            "mode TEXT NOT NULL DEFAULT 'manual',first_user_message_id INTEGER,"
            "last_opened_at TEXT,FOREIGN KEY(conversation_id) REFERENCES conversations(id))"
        )
        columns = {row[1] for row in db.execute("PRAGMA table_info(conversation_titles)")}
        for name, declaration in (
            ("mode", "TEXT NOT NULL DEFAULT 'manual'"),
            ("first_user_message_id", "INTEGER"), ("last_opened_at", "TEXT"),
        ):
            if name not in columns:
                db.execute(f"ALTER TABLE conversation_titles ADD COLUMN {name} {declaration}")

    def sync_auto(self, identity: str, *, opened: bool = False) -> str:
        """Persist the first real question's title once; an explicit rename always wins."""
        conversation = self.chats.conversation(identity)
        if conversation["kind"] != "course":
            raise StudyError("请选择课程对话。")
        with self.chats.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute(
                "SELECT 1 FROM deleted_records WHERE kind='conversation' AND identity=?",
                (identity,),
            ).fetchone():
                raise StudyError("对话已删除，请先恢复后再打开。")
            first = db.execute(
                "SELECT id,content FROM messages WHERE conversation_id=? AND role='user' "
                "AND trim(content)<>'' ORDER BY id LIMIT 1", (identity,),
            ).fetchone()
            existing = self._metadata(db).get(identity)
            manual = existing and existing.get("mode", "manual") == "manual"
            title = (existing["title"] if manual else
                     derive_title(first["content"] if first else "", conversation["subject"]))
            if not first and not opened and not existing:
                return title
            self._ensure_metadata(db)
            if not existing:
                db.execute(
                    "INSERT INTO conversation_titles "
                    "(conversation_id,title,updated_at,mode,first_user_message_id,last_opened_at) "
                    "VALUES (?,?,?,'auto',?,?)",
                    (identity, title, now(), first["id"] if first else None,
                     now() if opened else None),
                )
            elif not manual and (existing["title"] != title
                                 or existing.get("first_user_message_id") != (
                                     first["id"] if first else None)):
                db.execute(
                    "UPDATE conversation_titles SET title=?,updated_at=?,first_user_message_id=? "
                    "WHERE conversation_id=?",
                    (title, now(), first["id"] if first else None, identity),
                )
            if opened:
                db.execute(
                    "UPDATE conversation_titles SET last_opened_at=? WHERE conversation_id=?",
                    (now(), identity),
                )
        return title

    def mark_opened(self, identity: str) -> None:
        """Record explicit navigation locally so re-entering a course resumes that chat."""
        self.sync_auto(identity, opened=True)

    def recent(self, subject: str) -> str | None:
        """Choose the most recently opened or used active chat in this exact course."""
        entries = self.entries(subject)
        return entries[0]["id"] if entries else None

    def title(self, identity: str) -> str:
        """Use a manual rename when present, otherwise the first real user question."""
        conversation = self.chats.conversation(identity)
        if conversation["kind"] != "course":
            raise StudyError("请选择课程对话。")
        with self.chats.connect() as db:
            metadata = self._metadata(db).get(identity)
            if metadata and metadata.get("mode", "manual") == "manual":
                return metadata["title"]
            # Compatibility for old histories and an auto placeholder opened before
            # the first message. The UI also calls sync_auto to persist this label.
            return derive_title(self._first_question(db, identity), conversation["subject"])

    def rename(self, identity: str, title: str) -> str:
        """Rename an active course chat without editing its context or any historic snapshot."""
        title = title.strip()
        if not title or len(title) > 48 or "\n" in title or "\r" in title:
            raise StudyError("请填写 1 至 48 字的单行对话标题。")
        conversation = self.chats.conversation(identity)
        if conversation["kind"] != "course":
            raise StudyError("请选择课程对话。")
        with self.chats.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute(
                "SELECT 1 FROM deleted_records WHERE kind='conversation' AND identity=?",
                (identity,),
            ).fetchone():
                raise StudyError("对话已删除，请先恢复后再修改标题。")
            self._ensure_metadata(db)
            db.execute(
                "INSERT INTO conversation_titles(conversation_id,title,updated_at,mode) "
                "VALUES (?,?,?,'manual') ON CONFLICT(conversation_id) DO UPDATE SET "
                "title=excluded.title,updated_at=excluded.updated_at,mode='manual'",
                (identity, title, now()),
            )
        return title

    def entries(
        self, subject: str, query: str = "", *, include_deleted: bool = False
    ) -> list[dict]:
        """Search only this course; retain identity for duplicate names and restored chats."""
        with self.chats.connect() as db:
            metadata = self._metadata(db)
            deleted = {row["identity"]: row["deleted_at"] for row in db.execute(
                "SELECT identity,deleted_at FROM deleted_records WHERE kind='conversation'"
            )}
            rows = db.execute("SELECT payload FROM conversations ORDER BY rowid DESC").fetchall()
            values = []
            for row in rows:
                conversation = json.loads(row[0])
                identity = conversation["id"]
                if (conversation["kind"] != "course" or conversation["subject"] != subject
                        or (identity in deleted and not include_deleted)):
                    continue
                meta = metadata.get(identity)
                manual = meta and meta.get("mode", "manual") == "manual"
                title = (meta["title"] if manual else
                         derive_title(self._first_question(db, identity), subject))
                if meta and not manual and meta["title"] != title:
                    # Upgrade only automatic labels; a concurrent explicit rename
                    # must remain authoritative even while history is being read.
                    db.execute(
                        "UPDATE conversation_titles SET title=? WHERE conversation_id=? "
                        "AND mode='auto' AND title=?", (title, identity, meta["title"]),
                    )
                messages = db.execute(
                    "SELECT content,created_at FROM messages WHERE conversation_id=? ORDER BY id",
                    (identity,),
                ).fetchall()
                if query and query.casefold() not in (
                    title + "\n" + "\n".join(message["content"] for message in messages)
                ).casefold():
                    continue
                values.append({
                    "id": identity, "title": title, "subject": subject,
                    "manual_title": bool(manual), "created_at": conversation["created_at"],
                    "updated_at": max(filter(None, (
                        meta["updated_at"] if meta else None,
                        meta.get("last_opened_at") if meta else None,
                        messages[-1]["created_at"] if messages else None,
                        conversation["created_at"],
                    ))),
                    "deleted": identity in deleted, "deleted_at": deleted.get(identity),
                })
        return sorted(values, key=lambda value: (value["updated_at"], value["id"]), reverse=True)
