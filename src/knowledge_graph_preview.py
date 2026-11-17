"""Runtime-only saved candidates for reviewing both graphs before confirmation.

No function in this module writes knowledge objects, relations, or StudyStore.
Candidates carry a separate identity namespace and cannot enter formal exports.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.database import Database
from src.knowledge_graph_generation import GraphGenerationStore
from src.knowledge_graph_service import KnowledgeGraph
from src.knowledge_mining_service import (
    KnowledgeMiningDraft,
    MinedKnowledgePoint,
    MinedKnowledgeRelation,
)
from src.models import KnowledgeRelationType
from src.study_knowledge_bridge import SavedReviewKnowledgeDraft

_MAX_PREVIEW_BYTES = 3_000_000


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def preview_identity(draft: SavedReviewKnowledgeDraft) -> str:
    """Bind user controls and active-pointer changes to this exact candidate."""
    return hashlib.sha256(_json(asdict(draft)).encode("utf-8")).hexdigest()


def _directory(database: Database, runtime_dir: Path) -> tuple[Path, str]:
    store = GraphGenerationStore(runtime_dir, database.database_path,
                                 library_uuid=database.get_knowledge_base_uuid())
    return runtime_dir / "knowledge-preview" / store.scope, store.scope


def _draft(value: dict[str, Any]) -> SavedReviewKnowledgeDraft:
    source, mining = value["source"], value["mining"]
    if (not isinstance(source, dict) or not source.get("study_review_revision_id")
            or not source.get("study_review_id") or not source.get("library_id")):
        raise ValueError("候选缺少已保存复盘的可靠来源身份。")
    if len(mining.get("points", [])) > 40 or len(mining.get("relations", [])) > 80:
        raise ValueError("候选知识或关系数量超出单次提炼上限。")
    points = tuple(MinedKnowledgePoint(
        **{**point, "page_ids": tuple(point.get("page_ids", [])),
           "page_numbers": tuple(point.get("page_numbers", [])),
           "classifications": tuple(tuple(pair) for pair in point.get("classifications", []))},
    ) for point in mining.get("points", []))
    if not points or len({point.title for point in points}) != len(points):
        raise ValueError("候选知识为空或标题重复，请重新检查候选。")
    titles = {point.title for point in points}
    relations = tuple(MinedKnowledgeRelation(
        **{**relation, "relation_type": KnowledgeRelationType(relation["relation_type"])},
    ) for relation in mining.get("relations", []))
    if any(relation.source_title not in titles or relation.target_title not in titles
           or not relation.description.strip() for relation in relations):
        raise ValueError("候选关系缺少端点或依据。")
    return SavedReviewKnowledgeDraft(source, KnowledgeMiningDraft(
        document_id=mining["document_id"], document_title=mining["document_title"],
        points=points, relations=relations, truncated=bool(mining.get("truncated", False)),
        model=str(mining.get("model", "")),
    ))


def _write(path: Path, value: Any) -> None:
    encoded = _json(value)
    if len(encoded.encode("utf-8")) > _MAX_PREVIEW_BYTES:
        raise ValueError("候选预览文件超过安全大小。")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid4().hex}.tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_preview(database: Database, runtime_dir: Path,
                 draft: SavedReviewKnowledgeDraft) -> str:
    """Save an exact, library-scoped draft without confirming any knowledge."""
    _draft(asdict(draft))
    directory, scope = _directory(database, runtime_dir)
    identity = preview_identity(draft)
    _write(directory / f"{identity}.json", {
        "schema_version": 1, "scope": scope, "id": identity,
        "created_at": datetime.now(UTC).isoformat(), "draft": asdict(draft),
    })
    _write(directory / "active.json", {"id": identity})
    return identity


def load_preview(database: Database, runtime_dir: Path,
                 preview_id: str | None = None) -> SavedReviewKnowledgeDraft | None:
    """Read and verify a candidate; loading never writes or confirms it."""
    directory, scope = _directory(database, runtime_dir)
    if preview_id is None:
        pointer = directory / "active.json"
        if not pointer.exists():
            return None
        preview_id = json.loads(pointer.read_text(encoding="utf-8")).get("id")
    if not isinstance(preview_id, str) or len(preview_id) != 64 or any(
        char not in "0123456789abcdef" for char in preview_id
    ):
        raise ValueError("候选预览标识无效。")
    path = directory / f"{preview_id}.json"
    if not path.exists():
        raise ValueError("候选预览文件不存在，未改动正式知识库。")
    if path.stat().st_size > _MAX_PREVIEW_BYTES:
        raise ValueError("候选预览文件超过安全大小。")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("scope") != scope:
        raise ValueError("候选预览所属资料库不匹配。")
    result = _draft(value["draft"])
    if value.get("id") != preview_id or preview_identity(result) != preview_id:
        raise ValueError("候选预览内容校验失败。")
    return result


def list_previews(database: Database, runtime_dir: Path) -> list[dict[str, Any]]:
    """List this library's saved draft identities without copying private text."""
    directory, _ = _directory(database, runtime_dir)
    result = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "active.json":
            continue
        draft = load_preview(database, runtime_dir, path.stem)
        if draft is not None:
            result.append({"id": path.stem, "title": draft.mining.document_title,
                           "revision_id": draft.source["study_review_revision_id"],
                           "node_count": len(draft.mining.points),
                           "edge_count": len(draft.mining.relations)})
    return result


def clear_preview(database: Database, runtime_dir: Path,
                  preview_id: str | None = None) -> None:
    """Clear only the active pointer after confirmation; retain candidate history."""
    directory, _ = _directory(database, runtime_dir)
    pointer = directory / "active.json"
    if not pointer.exists():
        return
    if preview_id is not None and json.loads(pointer.read_text(encoding="utf-8")).get(
        "id"
    ) != preview_id:
        return
    pointer.unlink()


def preview_graph(draft: SavedReviewKnowledgeDraft, database: Database) -> KnowledgeGraph:
    """Project explicitly labeled candidates in a namespace no formal editor accepts."""
    _draft(asdict(draft))
    # Scope prevents accidental identity reuse when independent libraries happen
    # to contain the same review/version names.
    scope = database.get_knowledge_base_uuid()
    identity = hashlib.sha256((scope + preview_identity(draft)).encode("utf-8")).hexdigest()[:24]
    identifiers = {point.title: f"preview:{identity}:{index}"
                   for index, point in enumerate(draft.mining.points)}
    nodes = {}
    courses = list(dict.fromkeys(draft.source.get("course_ids", [])
                                or [draft.source.get("course_id", "")]))
    courses = [course for course in courses if course]
    for index, point in enumerate(draft.mining.points):
        identifier = identifiers[point.title]
        nodes[identifier] = {
            "id": identifier, "entity_kind": "knowledge", "title": point.title,
            "summary": point.content, "path": "",
            "tags": courses,
            "meta": {"is_preview": True, "confirmation_status": "ai_candidate",
                     "confirmation_current": False, "importance": point.importance,
                     "preview_point_index": index,
                     "academic_subject_tags": [subject + (f"/{child}" if child else "")
                                               for subject, child in point.subject_classifications],
                     "source_refs": [draft.source], "revision_refs": [draft.source],
                     "origin": point.origin, "evidence": point.evidence},
        }
    links = [{"source": identifiers[relation.source_title],
              "target": identifiers[relation.target_title],
              "label": relation.relation_type.label, "note": relation.description,
              "relation_type": relation.relation_type.value,
              "confirmation_status": "ai_candidate"}
             for relation in draft.mining.relations]
    return KnowledgeGraph(nodes, links, len(nodes), len(nodes), is_preview=True)
