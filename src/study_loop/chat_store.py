"""Independent, transactional conversation/memory sidecar; StudyStore v1 stays unchanged."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from uuid import uuid4

from .models import StudyError, now
from .storage import StudyStore, encode

APPLICATION_ID = 1313032241


def digest(value: object) -> str:
    return hashlib.sha256(encode(value).encode()).hexdigest()


class ChatStore:
    """Open only inside the selected StudyStore, with no access to platform databases."""

    def __init__(self, study: StudyStore) -> None:
        self.study = study
        self.db = study.path("conversations.sqlite3")
        if not self.db.exists():
            # Exclusive creation prevents two initializers replacing one another.
            try:
                with self.db.open("xb"):
                    pass
            except FileExistsError:
                pass
            else:
                with sqlite3.connect(self.db) as db:
                    db.executescript(f"""
                        BEGIN IMMEDIATE;
                        PRAGMA application_id = {APPLICATION_ID};
                        PRAGMA user_version = 1;
                        CREATE TABLE courses(subject TEXT PRIMARY KEY, payload TEXT NOT NULL);
                        CREATE TABLE conversations(id TEXT PRIMARY KEY, binding TEXT UNIQUE,
                            payload TEXT NOT NULL);
                        CREATE TABLE messages(id INTEGER PRIMARY KEY, conversation_id TEXT NOT NULL,
                            request_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
                            status TEXT NOT NULL, created_at TEXT NOT NULL, metadata TEXT NOT NULL,
                            UNIQUE(conversation_id, request_id, role),
                            FOREIGN KEY(conversation_id) REFERENCES conversations(id));
                        CREATE TABLE archives(id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
                            snapshot TEXT NOT NULL, created_at TEXT NOT NULL);
                        CREATE TABLE candidates(id TEXT PRIMARY KEY, archive_id TEXT NOT NULL,
                            content TEXT NOT NULL, created_at TEXT NOT NULL,
                            FOREIGN KEY(archive_id) REFERENCES archives(id));
                        CREATE TABLE memories(id TEXT PRIMARY KEY, subject TEXT NOT NULL,
                            content TEXT NOT NULL, created_at TEXT NOT NULL);
                        CREATE TABLE memory_links(memory_id TEXT NOT NULL, archive_id TEXT NOT NULL,
                            candidate_id TEXT NOT NULL, UNIQUE(memory_id, archive_id, candidate_id),
                            FOREIGN KEY(memory_id) REFERENCES memories(id),
                            FOREIGN KEY(archive_id) REFERENCES archives(id),
                            FOREIGN KEY(candidate_id) REFERENCES candidates(id));
                        COMMIT;
                    """)
        with self.connect() as db:
            if (
                db.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID
                or db.execute("PRAGMA user_version").fetchone()[0] != 1
            ):
                raise StudyError("对话库标识或版本不兼容，原文件保留，未迁移或覆盖。")
            db.execute(
                "CREATE TABLE IF NOT EXISTS deleted_records("
                "kind TEXT NOT NULL, identity TEXT NOT NULL, deleted_at TEXT NOT NULL, "
                "PRIMARY KEY(kind, identity))"
            )
            # Additive metadata only: existing tables, snapshots and user_version
            # remain unchanged. Readers need no write lock after initialization.
            existing_extractions = db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='knowledge_extractions'").fetchone()
            if existing_extractions is None:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "CREATE TABLE IF NOT EXISTS knowledge_extractions("
                    "id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
                )
            from .memory_revisions import initialize

            initialize(db)

    def _deleted(self, kind: str) -> set[str]:
        """Read reversible deletions without changing any original dialogue snapshots."""
        with self.connect() as db:
            return {r[0] for r in db.execute(
                "SELECT identity FROM deleted_records WHERE kind=?", (kind,)
            )}

    @contextmanager
    def connect(self):
        """One transaction per operation, including uniqueness checks and confirmation."""
        self.study.path("conversations.sqlite3")
        db = sqlite3.connect(self.db.as_uri() + "?mode=rw", uri=True, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def set_course(self, subject: str, source_ids: list[str], exam_notes: str) -> None:
        """Persist an explicitly chosen course scope, not model-inferred exam predictions."""
        if not subject.strip() or len(exam_notes) > 10000:
            raise StudyError("请填写课程名称，并控制考试范围说明长度。")
        available = {source.id for source in self.study.sources()}
        for identity in source_ids:
            if identity not in available:
                raise StudyError("所选材料已删除或不存在，请重新选择。")
        payload = {
            "subject": subject,
            "source_ids": list(dict.fromkeys(source_ids)),
            "exam_notes": exam_notes,
        }
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO courses VALUES (?,?)", (subject, encode(payload)))
            db.execute("DELETE FROM deleted_records WHERE kind='course' AND identity=?", (subject,))

    def courses(self) -> list[dict]:
        deleted = self._deleted("course")
        with self.connect() as db:
            return [
                json.loads(r[0]) for r in db.execute("SELECT payload FROM courses ORDER BY rowid")
                if json.loads(r[0])["subject"] not in deleted
            ]

    def course_scope(self, subject: str, source_id: str | None = None) -> dict[str, str]:
        """Resolve current user exam constraints without changing any saved conversation.

        A question's explicit subject takes precedence over source membership.
        Otherwise only an unambiguous current source-to-course relationship may
        supply a course; shared materials never combine unrelated exam scopes.
        """
        courses = self.courses()
        resolved = subject if subject.strip() else ""
        if not resolved and source_id:
            matches = [course for course in courses if source_id in course["source_ids"]]
            if len(matches) == 1:
                resolved = matches[0]["subject"]
        selected = next((course for course in courses if course["subject"] == resolved), None)
        return {"subject": resolved, "exam_notes": selected["exam_notes"] if selected else ""}

    def open_question(self, item_id: str) -> str:
        """Stable binding per immutable question revision, including unconfirmed drafts."""
        items = {i.id: i for i in [*self.study.drafts(), *self.study.items()]}
        item = items.get(item_id)
        if item is None:
            raise StudyError("原题不存在，请返回材料列表重新选择。")
        source = self.study.source(item.source_id)
        self.study.read_source(source)
        context = {
            "item": item.to_dict(),
            "source": {"id": source.id, "name": source.name, "sha256": source.sha256},
            "confirmed_material": item_id in {i.id for i in self.study.items()},
        }
        courses = [c for c in self.courses() if item.source_id in c["source_ids"]]
        subject = item.subject or (courses[0]["subject"] if len(courses) == 1 else "")
        # Confirmation may edit a draft before its first immutable saved version.
        # Preserve the old dialogue and bind the new context to a separate snapshot.
        # Match existing snapshots first so sidecars created by the earlier binding
        # rule reopen normally without a destructive database migration.
        for previous in self.conversations("question"):
            if previous["subject"] == subject and previous["context"] == context:
                return previous["id"]
        key = item_id + ":" + digest([subject, context])
        return self._create("question", subject, [item.source_id], "", context, key)

    def open_course(self, subject: str, *, new: bool = False) -> str:
        course = next((c for c in self.courses() if c["subject"] == subject), None)
        if course is None:
            raise StudyError("请先保存课程与材料范围。")
        return self._create(
            "course",
            subject,
            course["source_ids"],
            course["exam_notes"],
            {},
            uuid4().hex if new else "default:" + digest(course),
        )

    def open_review(self, subject: str, source_ids: list[str], context: dict) -> str:
        """Create a separate, immutable selection for one explicit AI review generation."""
        available = {source.id for source in self.study.sources()}
        if not source_ids or not set(source_ids) <= available:
            raise StudyError("所选复盘材料已删除或不存在。")
        return self._create("review", subject, source_ids, "", context, uuid4().hex)

    def _create(
        self, kind: str, subject: str, sources: list[str], exam: str, context: dict, key: str
    ) -> str:
        binding = kind + ":" + key
        value = {
            "id": uuid4().hex,
            "kind": kind,
            "subject": subject,
            "source_ids": sources,
            "exam_notes": exam,
            "context": context,
            "created_at": now(),
        }
        with self.connect() as db:
            # Reopening a deleted default binding creates a new empty conversation.
            db.execute(
                "UPDATE conversations SET binding=NULL WHERE binding=? AND id IN "
                "(SELECT identity FROM deleted_records WHERE kind='conversation')", (binding,),
            )
            db.execute(
                "INSERT OR IGNORE INTO conversations VALUES (?,?,?)",
                (value["id"], binding, encode(value)),
            )
            return db.execute(
                "SELECT id FROM conversations WHERE binding=?", (binding,)
            ).fetchone()[0]

    def conversation(self, identity: str) -> dict:
        if identity in self._deleted("conversation"):
            raise StudyError("这段对话已删除，可在高级设置的回收站中恢复。")
        with self.connect() as db:
            row = db.execute("SELECT payload FROM conversations WHERE id=?", (identity,)).fetchone()
        if row is None:
            raise StudyError("对话不存在或不属于当前学习库。")
        return json.loads(row[0])

    def conversations(self, kind: str, subject: str = "") -> list[dict]:
        deleted = self._deleted("conversation")
        with self.connect() as db:
            values = [
                json.loads(r[0])
                for r in db.execute("SELECT payload FROM conversations ORDER BY rowid DESC")
            ]
        return [v for v in values if v["id"] not in deleted and v["kind"] == kind
                and (not subject or v["subject"] == subject)]

    def messages(self, identity: str) -> list[dict]:
        self.conversation(identity)
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM messages WHERE conversation_id=? ORDER BY id", (identity,)
            ).fetchall()
        return [{**dict(r), "metadata": json.loads(r["metadata"])} for r in rows]

    def begin(self, identity: str, request_id: str, text: str, metadata: dict) -> bool:
        """Commit the user's exact message before calling a provider; never auto retry."""
        self.conversation(identity)
        if not text.strip() or len(text) > 16000:
            raise StudyError("消息不能为空，且每条不能超过 16000 字符。")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute(
                "SELECT 1 FROM messages WHERE conversation_id=? AND request_id=?",
                (identity, request_id),
            ).fetchone():
                return False
            if db.execute(
                "SELECT 1 FROM messages WHERE conversation_id=? AND status='pending'", (identity,)
            ).fetchone():
                raise StudyError("上一条请求仍在处理中或中断未决，请先结束等待状态；不会自动重发。")
            db.execute(
                "INSERT INTO messages VALUES (NULL,?,?,?,?,?,?,?)",
                (identity, request_id, "user", text, "pending", now(), encode(metadata)),
            )
        return True

    def finish(
        self, identity: str, request_id: str, text: str, metadata: dict, *, error: bool = False
    ) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status FROM messages WHERE conversation_id=? "
                "AND request_id=? AND role='user'",
                (identity, request_id),
            ).fetchone()
            if row is None or row[0] != "pending":
                raise StudyError("请求已结束，未覆盖已有回答。")
            db.execute(
                "UPDATE messages SET status=? WHERE conversation_id=? AND request_id=?",
                ("failed" if error else "complete", identity, request_id),
            )
            db.execute(
                "INSERT INTO messages VALUES (NULL,?,?,?,?,?,?,?)",
                (
                    identity,
                    request_id,
                    "assistant",
                    text,
                    "error" if error else "complete",
                    now(),
                    encode(metadata),
                ),
            )

    def interrupt_pending(self, identity: str) -> None:
        """Explicit recovery after a killed process; in-flight requests cannot be unsent."""
        for message in self.messages(identity):
            if message["status"] == "pending":
                self.finish(
                    identity,
                    message["request_id"],
                    "请求被用户标记为中断。未自动重发；原请求可能已被供应商计费。",
                    {},
                    error=True,
                )

    def snapshot(self, identity: str) -> dict:
        return {"conversation": self.conversation(identity), "messages": self.messages(identity)}

    def archive(
        self, identity: str, *, saved_review_version: dict | None = None,
        memory_source: dict | None = None,
    ) -> str:
        """Save every round and provenance as an immutable original dialogue snapshot."""
        snapshot = self.snapshot(identity)
        if saved_review_version is not None:
            snapshot["conversation"] = {
                **snapshot["conversation"],
                "context": {**snapshot["conversation"]["context"],
                            "saved_manual_review_version": saved_review_version},
            }
        if memory_source is not None:
            snapshot["knowledge_memory_source"] = memory_source
        if not snapshot["messages"]:
            raise StudyError("当前没有可保存的对话。")
        if any(m["status"] == "pending" for m in snapshot["messages"]):
            raise StudyError("请等待当前回答结束，再保存完整对话快照。")
        key = digest(snapshot)
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO archives VALUES (?,?,?,?)",
                (key, identity, encode(snapshot), now()),
            )
            db.execute("DELETE FROM deleted_records WHERE kind='archive' AND identity=?", (key,))
        return key

    def archived(self, identity: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT snapshot FROM archives WHERE id=?", (identity,)).fetchone()
        if row is None:
            raise StudyError("原始对话快照缺失。")
        snapshot = json.loads(row[0])
        if digest(snapshot) != identity:
            raise StudyError("原始对话快照指纹不匹配。")
        return snapshot

    def add_candidate(self, archive_id: str, content: str) -> str:
        snapshot = self.archived(archive_id)
        if (archive_id in self._deleted("archive")
                or snapshot["conversation"]["id"] in self._deleted("conversation")):
            raise StudyError("原始对话已删除，请先恢复。")
        if not content.strip() or len(content) > 30000:
            raise StudyError("记忆候选为空或过长，未保存为知识。")
        identity = digest([archive_id, content])
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO candidates VALUES (?,?,?,?)",
                (identity, archive_id, content, now()),
            )
            db.execute("DELETE FROM deleted_records WHERE kind='candidate' AND identity=?",
                       (identity,))
        return identity

    def candidates(self, conversation_id: str) -> list[dict]:
        deleted = self._deleted("candidate")
        archives = self._deleted("archive")
        if conversation_id in self._deleted("conversation"):
            return []
        with self.connect() as db:
            values = [
                dict(r)
                for r in db.execute(
                    "SELECT c.* FROM candidates c JOIN archives a ON a.id=c.archive_id "
                    "WHERE a.conversation_id=? ORDER BY c.rowid DESC",
                    (conversation_id,),
                )
                if r["id"] not in deleted and r["archive_id"] not in archives
            ]
            drafts = {
                value.get("candidate_id"): value
                for row in db.execute("SELECT payload FROM knowledge_extractions")
                if (value := json.loads(row[0])).get("candidate_id")
            }
        for value in values:
            if value["id"] in drafts:
                draft = drafts[value["id"]]
                value.update(raw_content=value["content"], content=draft["content"],
                             extraction_id=draft["id"], source=draft["source"])
        return values

    def confirm(self, candidate_id: str, content: str, *, confirmed: bool) -> str:
        """User-edited content only; exact duplicates retain links to all original snapshots."""
        if not confirmed or not content.strip() or len(content) > 30000:
            raise StudyError("请预览、核对或修改候选，并明确确认后保存。")
        if candidate_id in self._deleted("candidate"):
            raise StudyError("这份候选已删除，请先恢复。")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM candidates WHERE id=?", (candidate_id,)).fetchone()
            if row is None:
                raise StudyError("记忆候选不存在。")
            if row["archive_id"] in self._deleted("archive"):
                raise StudyError("这份候选的原始对话已删除，请先恢复。")
            archive = self.archived(row["archive_id"])
            if archive["conversation"]["id"] in self._deleted("conversation"):
                raise StudyError("这份候选的会话已删除，请先恢复。")
            source = self.memory_source(archive)
            if source is not None:
                state = self.memory_source_status(source)
                if not state["active"]:
                    raise StudyError("这份候选的来源已变化或不可用，请恢复来源或重新提炼。")
            subject = (source["subject"] if source is not None
                       else archive["conversation"]["subject"])
            # IDs remain stable after editing. Deduplicate by current content,
            # never by assuming that an old hash ID still describes that content.
            matches = db.execute(
                "SELECT id FROM memories WHERE subject=? AND content=? ORDER BY rowid",
                (subject, content.strip()),
            ).fetchall()
            if matches:
                deleted = self._deleted("memory")
                identity = min((row["id"] for row in matches), key=lambda key: (
                    key in deleted, not self.memory_status(key)["active"],
                ))
            else:
                identity = digest([subject, content.strip()])
                if db.execute("SELECT 1 FROM memories WHERE id=?", (identity,)).fetchone():
                    identity = uuid4().hex
                db.execute("INSERT INTO memories VALUES (?,?,?,?)",
                           (identity, subject, content.strip(), now()))
            from .memory_revisions import baseline, read

            baseline(db, read(db, identity))
            db.execute("DELETE FROM deleted_records WHERE kind='memory' AND identity=?",
                       (identity,))
            db.execute(
                "INSERT OR IGNORE INTO memory_links VALUES (?,?,?)",
                (identity, row["archive_id"], candidate_id),
            )
            for extraction in db.execute("SELECT id,payload FROM knowledge_extractions").fetchall():
                value = json.loads(extraction["payload"])
                if value.get("candidate_id") == candidate_id:
                    if value["content"] != content.strip():
                        value["working_revision"] += 1
                    value.update(content=content.strip(), confirmed_memory_id=identity,
                                 updated_at=now())
                    value.setdefault("confirmations", []).append(
                        {"memory_id": identity, "content": content.strip(), "created_at": now()}
                    )
                    db.execute("UPDATE knowledge_extractions SET payload=? WHERE id=?",
                               (encode(value), extraction["id"]))
        return identity

    def memories(
        self, subject: str | None, query: str = "", *, include_stale: bool = False,
        include_unavailable: bool = False, include_deleted: bool = False,
    ) -> list[dict]:
        deleted = self._deleted("memory")
        with self.connect() as db:
            values = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM memories WHERE (? IS NULL OR subject=?) ORDER BY rowid DESC",
                    (subject, subject),
                )
                if include_deleted or r["id"] not in deleted
            ]
            from .memory_revisions import read

            for value in values:
                value.update(read(db, value["id"]))
                value["links"] = [
                    dict(r)
                    for r in db.execute(
                        "SELECT archive_id,candidate_id FROM memory_links WHERE memory_id=?",
                        (value["id"],),
                    )
                ]
        result = []
        seen = set()
        for value in values:
            state = self.memory_status(value["id"])
            value.update(source_status=state, sources=state["sources"], stale=state["stale"],
                         deleted=value["id"] in deleted)
            if (state["active"] or (include_stale and state["stale"]
                                    and not state["unavailable"])
                    or (include_unavailable and state["unavailable"])
                    or (include_deleted and value["deleted"])) and (
                not query or query.casefold() in value["content"].casefold()
            ):
                key = (value["subject"], value["content"])
                if not (include_stale or include_unavailable or include_deleted) and key in seen:
                    continue
                seen.add(key)
                result.append(value)
        return result

    def memory(self, identity: str) -> dict:
        """Read the current confirmed body by stable ID, including recoverable/inactive records."""
        from .memory_revisions import read

        with self.connect() as db:
            value = read(db, identity)
            value["links"] = [dict(row) for row in db.execute(
                "SELECT archive_id,candidate_id FROM memory_links WHERE memory_id=?", (identity,),
            )]
        state = self.memory_status(identity)
        value.update(source_status=state, sources=state["sources"], stale=state["stale"],
                     deleted=identity in self._deleted("memory"))
        return value

    def update_memory(self, identity: str, content: str, *, expected_revision: int) -> dict:
        """Append a human revision without changing source links or confirmation records."""
        from .memory_revisions import MemoryRevisions

        return MemoryRevisions(self).update(identity, content, expected_revision=expected_revision)

    def memory_revisions(self, identity: str) -> list[dict]:
        from .memory_revisions import MemoryRevisions

        return MemoryRevisions(self).history(identity)

    def restore_memory_revision(
        self, identity: str, revision: int, *, expected_revision: int,
    ) -> dict:
        """Copy old content into a new revision without restoring retrieval authority."""
        from .memory_revisions import MemoryRevisions

        return MemoryRevisions(self).restore(
            identity, revision, expected_revision=expected_revision,
        )

    @staticmethod
    def memory_source(snapshot: dict) -> dict | None:
        """Read explicit provenance, including the existing saved-review archive shape."""
        explicit = snapshot.get("knowledge_memory_source")
        if isinstance(explicit, dict):
            return explicit
        conversation = snapshot.get("conversation", {})
        context = conversation.get("context", {})
        version = context.get("saved_manual_review_version")
        if isinstance(version, dict) and version.get("document_id") and version.get("id"):
            return {
                "type": "review_memory", "subject": conversation.get("subject", ""),
                "conversation_id": conversation.get("id"),
                "source_ids": conversation.get("source_ids", []),
                "item_ids": [selected["item"]["id"]
                             for selected in context.get("review_items", [])],
                "document_id": version["document_id"], "version_id": version["id"],
                "content_sha256": hashlib.sha256(version["content"].encode()).hexdigest(),
            }
        return None

    def memory_source_status(self, source: dict) -> dict:
        """Check live authority without replacing the immutable knowledge or its snapshot."""
        state = {"source": source, "active": True, "stale": False,
                 "unavailable": False, "reasons": []}

        def invalidate(reason: str, *, stale: bool = False) -> None:
            state["active"] = False
            state["stale" if stale else "unavailable"] = True
            state["reasons"].append(reason)

        subject = source.get("subject", "")
        source_ids = source.get("source_ids", [])
        if not isinstance(source_ids, list) or not all(isinstance(s, str) for s in source_ids):
            invalidate("来源标识无效")
            return state
        available = {item.id for item in self.study.sources()}
        if any(identity not in available for identity in source_ids):
            invalidate("来源材料已删除或不存在")
        courses = {course["subject"]: course for course in self.courses()}
        if subject in self._deleted("course") or (
            source.get("course_was_configured") and subject not in courses
        ):
            invalidate("课程已删除或不存在")
        elif (source.get("type") == "course_dialogue_memory" and subject in courses
              and not set(source_ids) <= set(courses[subject]["source_ids"])):
            invalidate("来源材料的课程范围已变化")
        try:
            conversation = self.conversation(source["conversation_id"])
        except (StudyError, KeyError):
            invalidate("来源会话已删除或不存在")
            return state
        if (source.get("type") == "course_dialogue_memory"
                and conversation["subject"] != subject):
            invalidate("来源课程归属已变化")
        if source.get("type") == "review_memory":
            with self.connect() as db:
                tables = {row[0] for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")}
                document_row = (
                    db.execute("SELECT payload FROM ai_review_documents WHERE id=?",
                               (source.get("document_id"),)).fetchone()
                    if {"ai_review_documents", "ai_review_versions"} <= tables else None
                )
                version = (db.execute("SELECT * FROM ai_review_versions WHERE id=?",
                                      (source.get("version_id"),)).fetchone()
                           if document_row else None)
            if document_row is None or version is None:
                invalidate("来源复盘稿或保存版本不存在")
                return state
            document = json.loads(document_row[0])
            if document.get("deleted_at"):
                invalidate("来源复盘稿已删除")
            if (document.get("conversation_id") != source.get("conversation_id")
                    or document.get("source_ids") != source_ids
                    or document.get("item_ids") != source.get("item_ids")):
                invalidate("来源复盘稿关联已变化")
            if (version["document_id"] != document["id"] or
                    hashlib.sha256(version["content"].encode()).hexdigest()
                    != source.get("content_sha256")):
                invalidate("来源保存版本校验失败")
            if document.get("last_saved_version_id") != source.get("version_id"):
                invalidate("来源复盘已保存新版本，请重新提炼", stale=True)
            items = {item.id: item for item in [*self.study.drafts(), *self.study.items()]}
            for identity in source.get("item_ids", []):
                item = items.get(identity)
                if item is None:
                    invalidate("原题已删除或不存在")
                elif self.course_scope(item.subject, item.source_id)["subject"] != subject:
                    invalidate("原题课程归属已变化")
                elif item.source_id not in source_ids:
                    invalidate("原题来源归属已变化")
        elif source.get("type") == "course_dialogue_memory":
            selected = set(source.get("message_ids", []))
            messages = [self.memory_message(message)
                        for message in self.messages(conversation["id"])
                        if message["id"] in selected]
            if len(messages) != len(selected) or digest(messages) != source.get("content_sha256"):
                invalidate("来源对话记录已变化", stale=True)
            if source.get("review_dependencies"):
                from .course_reviews import CourseReviewIndex

                try:
                    CourseReviewIndex(self.study).validate_evidence(
                        subject, source["review_dependencies"])
                except StudyError:
                    invalidate("来源对话引用的复盘版本已变化或不可用", stale=True)
        else:
            invalidate("记忆来源类型无效")
        return state

    @staticmethod
    def memory_message(message: dict) -> dict:
        """Content-only identity for immutable selected dialogue turns, excluding diagnostics."""
        return {key: message[key] for key in ("id", "role", "content", "status", "created_at")}

    def memory_status(self, identity: str) -> dict:
        """At least one live independent source keeps deduplicated knowledge available."""
        with self.connect() as db:
            if db.execute("SELECT 1 FROM memories WHERE id=?", (identity,)).fetchone() is None:
                raise StudyError("知识记忆不存在。")
            links = db.execute("SELECT archive_id FROM memory_links WHERE memory_id=?",
                               (identity,)).fetchall()
        states = []
        legacy = False
        deleted_archives = self._deleted("archive")
        for link in links:
            try:
                source = self.memory_source(self.archived(link["archive_id"]))
                if source is None:
                    legacy = True  # Existing untyped memories retain their established behavior.
                else:
                    state = self.memory_source_status(source)
                    if link["archive_id"] in deleted_archives:
                        state.update(active=False, unavailable=True)
                        state["reasons"].append("来源快照已删除")
                    states.append(state)
            except StudyError:
                states.append({"source": None, "active": False, "stale": False,
                               "unavailable": True, "reasons": ["原始来源快照不可用"]})
        active = legacy or not links or any(state["active"] for state in states)
        stale = not active and any(state["stale"] for state in states)
        unavailable = not active and (not stale or all(state["unavailable"] for state in states))
        if identity in self._deleted("memory"):
            active, unavailable = False, True
        reasons = list(dict.fromkeys(reason for state in states for reason in state["reasons"]))
        if identity in self._deleted("memory"):
            reasons.append("此记忆已删除")
        return {"active": active, "stale": stale, "unavailable": unavailable,
                "sources": [state["source"] for state in states if state["source"]],
                "reasons": reasons}

    def archives(self, conversation_id: str | None = None) -> list[dict]:
        """List saved dialogue snapshots that have not been moved to the trash."""
        deleted = self._deleted("archive")
        conversations = self._deleted("conversation")
        with self.connect() as db:
            return [dict(r) for r in db.execute(
                "SELECT id,conversation_id,created_at FROM archives "
                "WHERE (? IS NULL OR conversation_id=?) ORDER BY rowid DESC",
                (conversation_id, conversation_id),
            ) if r["id"] not in deleted and r["conversation_id"] not in conversations]

    def delete(self, kind: str, identity: str, *, confirmed: bool) -> None:
        """Move one explicitly selected record to the trash, retaining recovery and links.

        A deleted conversation's snapshots and candidates disappear with it. Confirmed
        memories remain separate records and can be deleted independently.
        """
        tables = {"conversation": "conversations", "archive": "archives",
                  "candidate": "candidates", "memory": "memories", "course": "courses"}
        if not confirmed:
            raise StudyError("请确认删除所选内容。")
        if kind not in tables:
            raise StudyError("无法删除这种内容。")
        column = "subject" if kind == "course" else "id"
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute(
                f"SELECT 1 FROM {tables[kind]} WHERE {column}=?", (identity,),
            ).fetchone() is None:
                raise StudyError("所选内容不存在。")
            if kind == "conversation" and db.execute(
                "SELECT 1 FROM messages WHERE conversation_id=? AND status='pending'", (identity,),
            ).fetchone():
                raise StudyError("请先等待回答完成或结束等待，再删除对话。")
            db.execute("INSERT OR IGNORE INTO deleted_records VALUES (?,?,?)",
                       (kind, identity, now()))

    def trash(self) -> list[dict]:
        """Return deleted records with readable names for a deliberate restore action."""
        labels = {"conversation": "对话", "archive": "已保存对话", "candidate": "记忆候选",
                  "memory": "记忆", "course": "课程设置"}
        with self.connect() as db:
            values = [dict(r) for r in db.execute(
                "SELECT * FROM deleted_records ORDER BY deleted_at DESC"
            )]
            for value in values:
                kind, identity = value["kind"], value["identity"]
                if kind == "conversation":
                    row = db.execute("SELECT payload FROM conversations WHERE id=?",
                                     (identity,)).fetchone()
                    payload = json.loads(row[0]) if row else {}
                    name = payload.get("subject") or "题目追问"
                elif kind in {"memory", "candidate"}:
                    table = "memories" if kind == "memory" else "candidates"
                    row = db.execute(f"SELECT content FROM {table} WHERE id=?",
                                     (identity,)).fetchone()
                    name = row[0][:60] if row else ""
                elif kind == "archive":
                    row = db.execute("SELECT snapshot FROM archives WHERE id=?",
                                     (identity,)).fetchone()
                    name = json.loads(row[0])["conversation"]["subject"] if row else ""
                else:
                    name = identity
                value["label"] = labels.get(kind, "记录") + " · " + name
        return values

    def restore(self, kind: str, identity: str) -> None:
        """Restore one record; a replaced default conversation keeps its new binding."""
        with self.connect() as db:
            db.execute("DELETE FROM deleted_records WHERE kind=? AND identity=?", (kind, identity))
