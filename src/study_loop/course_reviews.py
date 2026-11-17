"""Dynamic course views over saved review versions, without another copy of their text."""

from __future__ import annotations

import hashlib
import json

from .chat_store import ChatStore, digest
from .models import StudyError
from .storage import StudyStore


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class CourseReviewIndex:
    """Resolve current course ownership and read only each document's latest saved version."""

    def __init__(self, store: StudyStore) -> None:
        self.store = store
        self.chats = ChatStore(store)

    def entries(self, subject: str, query: str = "") -> list[dict]:
        """Read live latest versions; working text, raw AI and earlier versions stay separate."""
        courses = {course["subject"] for course in self.chats.courses()}
        if subject not in courses:
            return []
        items = {item.id: item for item in [*self.store.drafts(), *self.store.items()]}
        sources = {source.id: source for source in self.store.sources()}
        with self.chats.connect() as db:
            tables = {row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            if not {"ai_review_documents", "ai_review_versions"} <= tables:
                return []
            rows = db.execute(
                "SELECT d.id,d.payload,v.id AS version_id,v.number,v.content,v.created_at "
                "FROM ai_review_documents d JOIN ai_review_versions v ON v.document_id=d.id "
                "WHERE v.number=(SELECT MAX(number) FROM ai_review_versions WHERE document_id=d.id)"
            ).fetchall()
        results = []
        for row in rows:
            document = json.loads(row["payload"])
            source_ids, item_ids = document["source_ids"], document["item_ids"]
            if (document.get("deleted_at") or not source_ids or not item_ids
                    or not set(source_ids) <= sources.keys() or not set(item_ids) <= items.keys()
                    or document.get("last_saved_version_id") != row["version_id"]):
                continue
            selected = [items[identity] for identity in item_ids]
            if {item.source_id for item in selected} != set(source_ids):
                continue
            owners = {
                self.chats.course_scope(item.subject, item.source_id)["subject"]
                for item in selected
            }
            if owners != {subject}:
                continue
            questions = [{
                "id": item.id, "number": item.question.number,
                "title": item.question.content, "chapter": item.chapter,
            } for item in selected]
            chapters = list(dict.fromkeys(item.chapter for item in selected if item.chapter))
            entry = {
                "id": row["id"], "document_id": row["id"],
                "version_id": row["version_id"], "version_number": row["number"],
                "subject": subject, "source_ids": list(source_ids),
                "filenames": [sources[identity].name for identity in source_ids],
                "item_ids": list(item_ids), "questions": questions,
                "title": "；".join(
                    ((question["number"] + " · ") if question["number"] else "")
                    + question["title"][:100] for question in questions
                ),
                "chapter": "、".join(chapters), "status": "saved_review",
                "updated_at": row["created_at"], "content": row["content"],
            }
            searchable = "\n".join([
                entry["title"], entry["chapter"], *entry["filenames"], entry["content"],
            ]).casefold()
            if not query or query.casefold() in searchable:
                results.append(entry)
        return sorted(results, key=lambda entry: (entry["updated_at"], entry["document_id"]),
                      reverse=True)

    def entry(self, document_id: str, subject: str | None = None) -> dict:
        """Resolve a current live course entry, rather than falling back to an old snapshot."""
        subjects = ([subject] if subject is not None else
                    [c["subject"] for c in self.chats.courses()])
        for course in subjects:
            for entry in self.entries(course):
                if entry["document_id"] == document_id:
                    return entry
        raise StudyError("这份已保存复盘不在当前课程资料中，可能已删除或课程归属已变更。")

    def compare(self, subject: str, document_ids: list[str]) -> list[dict]:
        """Read the selected latest saved versions from one course in the requested order."""
        if (not document_ids or len(document_ids) > 10
                or len(set(document_ids)) != len(document_ids)):
            raise StudyError("请选择 1 至 10 份不重复的复盘资料。")
        entries = {entry["document_id"]: entry for entry in self.entries(subject)}
        if not set(document_ids) <= entries.keys():
            raise StudyError("所选复盘资料已删除或不属于当前课程，请重新选择。")
        return [entries[identity] for identity in document_ids]

    @staticmethod
    def _evidence(entry: dict, start: int) -> dict:
        content = entry["content"]
        sha256 = _sha256(content)
        end = min(start + 5500, len(content))
        return {
            "kind": "saved_review",
            "label": "R" + digest([
                entry["document_id"], entry["version_id"], entry["source_ids"], start, sha256,
            ])[:12],
            "name": "、".join(entry["filenames"]) + f" · 复盘第{entry['version_number']}版",
            "subject": entry["subject"], "source_ids": entry["source_ids"],
            "document_id": entry["document_id"], "version_id": entry["version_id"],
            "version_number": entry["version_number"], "item_ids": entry["item_ids"],
            "sha256": sha256, "start": start, "end": end,
            "location": f"已保存复盘第{entry['version_number']}版 · 字符 {start} 起",
            "text": content[start:end],
        }

    def retrieve(
        self, subject: str, query: str, source_ids: list[str] | None = None
    ) -> list[dict]:
        """Select at most five relevant saved-version excerpts, bounded to 20000 characters."""
        import jieba

        tokens = {part.casefold() for part in jieba.cut(query) if len(part.strip()) > 1}
        candidates = []
        allowed = set(source_ids) if source_ids is not None else None
        for entry in self.entries(subject):
            if allowed is not None and not set(entry["source_ids"]) <= allowed:
                continue
            excerpts = []
            for start in range(0, len(entry["content"]), 5000):
                ref = self._evidence(entry, start)
                searchable = (entry["title"] + "\n" + ref["text"]).casefold()
                score = sum(searchable.count(token) for token in tokens)
                excerpts.append((score, ref))
            if excerpts:
                candidates.append(max(excerpts, key=lambda result: result[0]))
        candidates.sort(key=lambda result: result[0], reverse=True)
        results, size = [], 0
        for _, ref in candidates:
            if len(results) >= 5 or size + len(ref["text"]) > 20000:
                continue
            for source_id in ref["source_ids"]:
                self.store.read_source(self.store.source(source_id))
            results.append(ref)
            size += len(ref["text"])
        return results

    def validate_evidence(self, subject: str, refs: list[dict]) -> None:
        """Reject replaced, deleted or reassigned historical versions without rewriting history."""
        reviews = [ref for ref in refs if ref.get("kind") == "saved_review"
                   or str(ref.get("label", "")).startswith("R")]
        if not reviews:
            return
        entries = {entry["document_id"]: entry for entry in self.entries(subject)}
        verified_sources = set()
        for ref in reviews:
            entry = entries.get(ref.get("document_id"))
            start = ref.get("start")
            if entry is None or not isinstance(start, int) or start < 0:
                raise StudyError("此对话引用的复盘已删除、更新或归属变更，请新建课程会话后继续。")
            expected = self._evidence(entry, start)
            fields = ("label", "subject", "source_ids", "document_id", "version_id",
                      "version_number", "item_ids", "sha256", "start", "end", "text")
            if any(ref.get(field) != expected[field] for field in fields):
                raise StudyError("此对话引用的复盘已删除、更新或归属变更，请新建课程会话后继续。")
            for source_id in entry["source_ids"]:
                if source_id not in verified_sources:
                    self.store.read_source(self.store.source(source_id))
                    verified_sources.add(source_id)
