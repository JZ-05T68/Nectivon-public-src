"""Human source names projected from immutable evidence and current local metadata."""

from __future__ import annotations

import json
import re

from .chat_store import ChatStore
from .models import StudyError
from .storage import StudyStore

UNAVAILABLE = "来源信息暂不可用"
SOURCE_NAMES = {"review_memory": "单题复盘记忆", "course_dialogue_memory": "课程自由对话记忆"}
_KEY = re.compile(
    r"(?<![A-Za-z0-9_])(?:[SMR][0-9a-fA-F]{8,64}|[0-9a-fA-F]{32,64}|"
    r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})(?![A-Za-z0-9_])"
)


def human_name(value: object) -> str:
    """Keep ordinary titles while removing identifier-shaped parts from display-only names."""
    if not isinstance(value, str):
        return ""
    value = _KEY.sub("", value)
    value = re.sub(r"</?span\b[^>]*>", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(?:memory_revision|content_sha256|source_id|document_id|version_id)\b",
                   "", value)
    value = value.strip(" #\t\r\n[]:：·,，;；`")
    # A real title may end in a parenthesized qualifier. Only empty wrappers
    # left by a removed identity should disappear; do not unbalance that title.
    if re.fullmatch(r"[\s\[\]（）():：·,，;；`]*", value):
        return ""
    return value


def memory_title(memory: dict) -> str:
    """Use a short content title, never the stable storage identity."""
    first = next((line for line in memory.get("content", "").splitlines() if line.strip()), "")
    return human_name(first)[:70] or "知识记忆"


def evidence_name(ref: dict, store: StudyStore | None = None) -> str:
    """Name a cited saved review or memory revision without revealing backend keys.

    Snapshot fields suffice for exports. An optional store only resolves missing names;
    it never replaces a historical reference's explicit revision with the current one.
    """
    if ref.get("kind") == "saved_review" or str(ref.get("label", "")).startswith("R"):
        name = human_name(ref.get("name"))
        version = ref.get("version_number")
        if name:
            return name + (f" · 复盘第{version}版" if isinstance(version, int)
                           and not re.search(r"(?:复盘)?第\s*" + str(version) + r"\s*版", name)
                           else "")
        return UNAVAILABLE
    if ref.get("memory_id") or str(ref.get("label", "")).startswith("M"):
        title = memory_title({"content": ref.get("text", "")})
        revision = ref.get("memory_revision")
        if store is not None and ref.get("memory_id") and title == "知识记忆":
            try:
                title = memory_title(ChatStore(store).memory(ref["memory_id"]))
            except StudyError:
                return UNAVAILABLE
        name = "知识记忆：" + title if title != "知识记忆" else "已确认知识记忆"
        return name + (f" · 第{revision}修订" if isinstance(revision, int) else "")
    return human_name(ref.get("name")) or UNAVAILABLE


def memory_source_details(store: StudyStore, memory: dict) -> list[dict]:
    """Resolve display-only provenance, keeping review versions separate from memory revisions."""
    chats = ChatStore(store)
    details = []
    for source in memory.get("sources", []):
        value = {
            "type_name": SOURCE_NAMES.get(source.get("type"), "已有知识记忆"),
            "subject": human_name(source.get("subject")) or "未指定课程",
            "filename": "", "review_version": None, "dialogue_title": "",
            "memory_revision": memory.get("revision", 1),
            "available": False, "unavailable_reason": UNAVAILABLE,
        }
        if source.get("type") == "review_memory":
            names = []
            for identity in source.get("source_ids", []):
                try:
                    names.append(human_name(store.source(identity).name))
                except StudyError:
                    names.append(UNAVAILABLE)
            value["filename"] = "、".join(name for name in names if name) or UNAVAILABLE
            with chats.connect() as db:
                tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master")}
                version = db.execute("SELECT number,document_id FROM ai_review_versions WHERE id=?",
                                     (source.get("version_id"),)).fetchone() if (
                                         "ai_review_versions" in tables) else None
                document = db.execute("SELECT payload FROM ai_review_documents WHERE id=?",
                                      (source.get("document_id"),)).fetchone() if (
                                          "ai_review_documents" in tables) else None
            if (version is not None and document is not None
                    and version["document_id"] == source.get("document_id")):
                value["review_version"] = version["number"]
                payload = json.loads(document[0])
                value["available"] = not payload.get("deleted_at")
                value["unavailable_reason"] = ""
                if not value["available"]:
                    value["unavailable_reason"] = "来源复盘已删除，请从回收站恢复。"
                if not set(source.get("source_ids", [])) <= {item.id for item in store.sources()}:
                    value.update(available=False, unavailable_reason="来源材料已删除或不存在。")
                try:
                    chats.conversation(source["conversation_id"])
                except (StudyError, KeyError):
                    value.update(available=False, unavailable_reason="来源对话已删除或不存在。")
        elif source.get("type") == "course_dialogue_memory":
            from .conversation_titles import ConversationTitles

            try:
                value["dialogue_title"] = human_name(
                    ConversationTitles(store).title(source["conversation_id"])
                ) or UNAVAILABLE
                value.update(available=True, unavailable_reason="")
            except (StudyError, KeyError):
                value["dialogue_title"] = UNAVAILABLE
                value["unavailable_reason"] = "来源对话已删除或不存在，请从回收站恢复后再打开。"
        details.append(value)
    if not details:
        details.append({
            "type_name": "已有知识记忆", "subject": human_name(memory.get("subject")),
            "filename": "", "review_version": None, "dialogue_title": "",
            "memory_revision": memory.get("revision", 1), "available": False,
            "unavailable_reason": UNAVAILABLE,
        })
    return details
