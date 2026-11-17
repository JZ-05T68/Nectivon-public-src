"""Explicit saved-review identities bridging two otherwise independent local stores.

Only a confirmed import creates additive mapping tables in the platform DB.
No review text is copied to a parallel document library. All knowledge writes
reuse KnowledgeObjectService and participate in one atomic transaction.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

from src.database import Database
from src.knowledge_mining_service import (
    KnowledgeMiningAIService,
    KnowledgeMiningDraft,
    KnowledgeMiningSaveResult,
)
from src.knowledge_object_service import KnowledgeObjectService
from src.study_loop.ai_review import AIReviewService
from src.study_loop.models import StudyError, now
from src.study_loop.storage import StudyStore, encode


@dataclass(frozen=True)
class SavedReviewKnowledgeDraft:
    """A source-bound candidate; merely constructing it never writes knowledge."""

    source: dict
    mining: KnowledgeMiningDraft


class _TransactionDatabase(Database):
    """Bind existing service methods to the bridge's outer atomic transaction."""

    def __init__(self, database: Database, connection: sqlite3.Connection) -> None:
        self.database_path = database.database_path
        self.image_readings_dir = database.image_readings_dir
        self.last_backup_path = database.last_backup_path
        self.connection = connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        yield self.connection

    @contextmanager
    def knowledge_transaction(self) -> Iterator[sqlite3.Connection]:
        yield self.connection


def _has_imports(connection: sqlite3.Connection) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='study_knowledge_imports'"
    ).fetchone() is not None


def knowledge_review_refs(
    database: Database, knowledge_id: int, *, connection: sqlite3.Connection | None = None,
) -> list[dict]:
    """Read saved-review lineage without initializing or changing either store."""
    with (nullcontext(connection) if connection is not None else database._connection()) as current:
        if not _has_imports(current):
            return []
        rows = current.execute(
            "SELECT i.source_ref,l.origin,l.evidence FROM study_knowledge_links l "
            "JOIN study_knowledge_imports i ON i.revision_id=l.revision_id "
            "AND i.library_id=l.library_id "
            "WHERE l.knowledge_object_id=? ORDER BY i.created_at", (knowledge_id,),
        ).fetchall()
    return [dict(json.loads(row[0]), origin=row[1], evidence=row[2]) for row in rows]


class StudyKnowledgeBridge:
    """Resolve saved versions, extract candidates and explicitly confirm imports."""

    def __init__(self, database: Database, store: StudyStore) -> None:
        self.database = database
        self.store = store

    def saved_source(self, revision_id: str) -> tuple[dict, str]:
        """Resolve exact identities; never bind by a title, keyword or text similarity."""
        reviews = AIReviewService(self.store)
        version = reviews.version(revision_id)
        document = reviews.document(version["document_id"])
        if document.get("deleted_at"):
            raise StudyError("复盘已删除，请先恢复后再导入。")
        conversation = reviews.chats.conversation(document["conversation_id"])
        selected = conversation["context"].get("review_items", [])
        if not selected or {entry["item"]["id"] for entry in selected} != set(document["item_ids"]):
            raise StudyError("无法可靠关联复盘与原始题目，未按标题补关联。")
        live = {item.id: item for item in [*self.store.items(), *self.store.drafts()]}
        materials, questions, page_refs, courses = {}, [], [], set()
        for entry in selected:
            item, snapshot = live.get(entry["item"]["id"]), entry["item"]
            if item is None or item.question.id != snapshot["question"]["id"]:
                raise StudyError("原题身份缺失或已变化，无法可靠关联。")
            source = self.store.source(snapshot["source_id"])
            if source.sha256 != entry["source"]["sha256"]:
                raise StudyError("复盘记录与原材料身份不一致，拒绝关联。")
            self.store.read_source(source)
            materials[source.id] = {"material_id": source.id, "filename": source.name,
                                    "sha256": source.sha256}
            questions.append({"item_id": item.id, "question_id": snapshot["question"]["id"],
                              "material_id": source.id})
            courses.add(reviews.chats.course_scope(item.subject, source.id)["subject"])
            for page in self.store.referenced_pages(source.id, snapshot["page_ids"]):
                ref = {"material_id": source.id, "study_page_id": page.id,
                       "page_number": page.number}
                if ref not in page_refs:
                    page_refs.append(ref)
        if set(materials) != set(document["source_ids"]):
            raise StudyError("复盘材料身份映射不完整，未生成伪造来源。")
        source_ref = {
            "library_id": hashlib.sha256(
                str(self.store.root).casefold().encode("utf-8")
            ).hexdigest(),
            "study_root": str(self.store.root), "study_review_id": document["id"],
            "study_review_revision_id": revision_id, "revision_number": version["number"],
            # courses.subject is the actual primary key, not a guessed numeric ID.
            "course_id": next(iter(courses)) if len(courses) == 1 else "",
            "course_ids": sorted(courses), "materials": list(materials.values()),
            "questions": questions, "source_page_refs": page_refs,
            "content_sha256": hashlib.sha256(version["content"].encode()).hexdigest(),
        }
        return source_ref, version["content"]

    def imports_for_review(self, review_id: str) -> list[dict]:
        """Report imported versions and their nodes, including historical revisions."""
        with self.database._connection() as connection:
            if not _has_imports(connection):
                return []
            rows = connection.execute(
                "SELECT revision_id,source_ref FROM study_knowledge_imports "
                "WHERE library_id=? AND review_id=? ORDER BY created_at",
                (self.library_id, review_id),
            ).fetchall()
            return [dict(json.loads(row[1]), knowledge_ids=[link[0] for link in connection.execute(
                "SELECT DISTINCT knowledge_object_id FROM study_knowledge_links "
                "WHERE library_id=? AND revision_id=?",
                (self.library_id, row[0]),
            )]) for row in rows]

    @property
    def library_id(self) -> str:
        """Namespace independent stores by their resolved, case-insensitive local root."""
        return hashlib.sha256(str(self.store.root).casefold().encode("utf-8")).hexdigest()

    def mine_saved_review(
        self, revision_id: str, miner: KnowledgeMiningAIService,
    ) -> SavedReviewKnowledgeDraft:
        """One bounded audited extraction, always reading the saved version afresh."""
        source, content = self.saved_source(revision_id)
        if any(ref["study_review_revision_id"] == revision_id
               for ref in self.imports_for_review(source["study_review_id"])):
            raise StudyError("此复盘版本已经导入，无需再次调用模型。")
        mining = miner.mine_saved_review(
            content, review_id=source["study_review_id"], revision_id=revision_id,
            title=f"{source['course_id'] or '未分类'} · 已保存复盘第{source['revision_number']}版",
        )
        return SavedReviewKnowledgeDraft(source, mining)

    def confirm(
        self, draft: SavedReviewKnowledgeDraft, *, user_confirmed: bool,
        bindings: Mapping[str, int | None] | None = None,
    ) -> KnowledgeMiningSaveResult:
        """Atomically persist an inspected candidate, provenance and typed relations.

        Explicit bindings reuse existing identities without rewriting their
        content. A later review version requires an explicit reuse/new choice
        for every point, so regeneration cannot silently multiply or overwrite.
        """
        if user_confirmed is not True:
            raise StudyError("请先检查知识点和关系，再确认导入。")
        revision_id = draft.source["study_review_revision_id"]
        source, _ = self.saved_source(revision_id)
        if source != draft.source:
            raise StudyError("复盘或原始来源身份已变化，请重新检查候选。")
        if not draft.mining.points:
            raise StudyError("请至少选择一个知识点。")
        bindings = dict(bindings or {})
        prior = self.imports_for_review(source["study_review_id"])
        if prior and not any(ref["study_review_revision_id"] == revision_id for ref in prior):
            if any(point.title not in bindings for point in draft.mining.points):
                raise StudyError("新版本请逐项选择复用已有知识点或明确新增，旧知识不会被覆盖。")
        with self.database.knowledge_transaction() as connection:
            self._initialize(connection)
            existing = connection.execute(
                "SELECT source_ref FROM study_knowledge_imports "
                "WHERE library_id=? AND revision_id=?", (self.library_id, revision_id),
            ).fetchone()
            if existing:
                if json.loads(existing[0]) != source:
                    raise StudyError("已导入版本内容指纹不一致，请停止并核对原库。")
                ids = tuple(row[0] for row in connection.execute(
                    "SELECT DISTINCT knowledge_object_id FROM study_knowledge_links "
                    "WHERE library_id=? AND revision_id=? ORDER BY knowledge_object_id",
                    (self.library_id, revision_id),
                ))
                return KnowledgeMiningSaveResult(knowledge_ids=ids)
            service = KnowledgeObjectService(_TransactionDatabase(self.database, connection))
            connection.execute(
                "INSERT INTO study_knowledge_imports VALUES (?,?,?,?,?)",
                (self.library_id, revision_id, source["study_review_id"], encode(source), now()),
            )
            title_to_id: dict[str, int] = {}
            for point in draft.mining.points:
                existing_id = bindings.get(point.title)
                if existing_id is not None:
                    obj = service.get_view(existing_id).knowledge_object
                    if obj.lifecycle.value != "active":
                        raise StudyError("只能复用现行知识点。")
                    identity = obj.id
                else:
                    view = service.create(
                        kind="concept", title=point.title, content=point.content,
                        importance=point.importance, classifications=point.subject_classifications,
                        epistemic_basis=("personal_judgment" if point.origin == "inferred"
                                         else "source_derived"),
                    )
                    identity = view.knowledge_object.id
                    service.confirm(identity)
                title_to_id[point.title] = identity
                connection.execute(
                    "INSERT INTO study_knowledge_links VALUES (?,?,?,?,?,?)",
                    (self.library_id, revision_id, point.title, identity, point.origin,
                     point.evidence),
                )
            count = 0
            for relation in draft.mining.relations:
                left, right = (title_to_id.get(relation.source_title),
                               title_to_id.get(relation.target_title))
                if left is None or right is None or left == right:
                    continue
                if not relation.description.strip():
                    raise StudyError("知识关系缺少依据，未保存本次导入。")
                duplicate = connection.execute(
                    "SELECT id FROM knowledge_relations WHERE source_ko_id=? "
                    "AND target_ko_id=? AND relation_type=?",
                    (left, right, relation.relation_type.value),
                ).fetchone()
                if duplicate is None:
                    service.add_relation(left, right, relation_type=relation.relation_type,
                                         description=relation.description)
                    count += 1
        return KnowledgeMiningSaveResult(tuple(dict.fromkeys(title_to_id.values())), count)

    @staticmethod
    def _initialize(connection: sqlite3.Connection) -> None:
        # execute individually: executescript would commit the outer transaction.
        connection.execute(
            "CREATE TABLE IF NOT EXISTS study_knowledge_imports ("
            "library_id TEXT NOT NULL, revision_id TEXT NOT NULL, review_id TEXT NOT NULL, "
            "source_ref TEXT NOT NULL, created_at TEXT NOT NULL, "
            "PRIMARY KEY(library_id,revision_id))"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS study_knowledge_links ("
            "library_id TEXT NOT NULL, revision_id TEXT NOT NULL, "
            "point_key TEXT NOT NULL, knowledge_object_id INTEGER NOT NULL, "
            "origin TEXT NOT NULL, evidence TEXT NOT NULL, "
            "PRIMARY KEY(library_id,revision_id,point_key), "
            "FOREIGN KEY(library_id,revision_id) "
            "REFERENCES study_knowledge_imports(library_id,revision_id))"
        )
