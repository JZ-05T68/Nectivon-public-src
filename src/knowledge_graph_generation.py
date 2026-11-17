"""Validated, reversible local graph artifacts; content stays owned by SQLite.

The artifact is a projection, not a second knowledge database. Regenerating it
never edits knowledge, relationships, layouts, or renderer preferences.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from src.models import KnowledgeRelationType

if TYPE_CHECKING:
    from src.knowledge_graph_service import KnowledgeGraph

MAX_GRAPH_BYTES = 30_000_000


def _encoded(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_encoded(value)).hexdigest()


def graph_contract(graph: KnowledgeGraph) -> dict[str, Any]:
    """Project one snapshot for both views without inventing semantic edges.

    Only persisted graph links enter this contract. Mining candidates are held
    by their draft workflow and never appear here. An unconfirmed saved object
    remains visible with an explicit status, rather than acquiring confirmation.
    """
    types = {item.label: item.value for item in KnowledgeRelationType}
    nodes = []
    for identifier, node in sorted(graph.nodes.items()):
        meta = node.get("meta", {})
        if not graph.is_preview and (identifier.startswith("preview:") or meta.get("is_preview")):
            raise ValueError("候选预览不能改作正式知识图。")
        status = ("ai_candidate" if graph.is_preview else
                  "confirmed" if meta.get("confirmation_current") else
                  "stale" if meta.get("confirmation_status") == "confirmed" else "unconfirmed")
        nodes.append({
            **node, "id": identifier,
            "source_refs": list(meta.get("source_refs", [])),
            "subject_tags": list(node.get("tags", [])),
            "importance": meta.get("importance", "normal"),
            "confirmation_status": status if node.get("entity_kind") == "knowledge"
            else "not_applicable",
            "revision_refs": list(meta.get("revision_refs", [])),
        })
    edges = [{**edge, "relation_type": types.get(edge.get("label", ""), "source_reference"),
              "confirmation_status": "ai_candidate" if graph.is_preview else "persisted"}
             for edge in graph.links]
    result = {"schema_version": 1, "mode": "preview" if graph.is_preview else "formal",
              "nodes": nodes, "edges": edges,
              "source_refs": {n["id"]: n["source_refs"] for n in nodes if n["source_refs"]},
              "subject_tags": sorted({tag for n in nodes for tag in n["subject_tags"]}),
              "importance": {n["id"]: n["importance"] for n in nodes},
              "confirmation_status": {n["id"]: n["confirmation_status"] for n in nodes},
              "revision_refs": {n["id"]: n["revision_refs"] for n in nodes
                                if n["revision_refs"]}}
    validate_graph_contract(result)
    return result


def validate_graph_contract(value: dict[str, Any]) -> None:
    """Reject duplicate IDs, dangling endpoints, candidates, and malformed data."""
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("星图数据版本不兼容。")
    nodes, edges = value.get("nodes"), value.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("星图节点或关系格式无效。")
    identifiers: set[str] = set()
    preview = value.get("mode") == "preview"
    if value.get("mode", "formal") not in {"formal", "preview"}:
        raise ValueError("知识图模式无效。")
    for node in nodes:
        identifier = node.get("id") if isinstance(node, dict) else None
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("星图节点标识为空或重复。")
        identifiers.add(identifier)
        if preview and (not identifier.startswith("preview:")
                        or node.get("confirmation_status") != "ai_candidate"):
            raise ValueError("候选预览必须使用独立候选身份及明确的候选状态。")
        if not preview and (identifier.startswith("preview:")
                            or node.get("confirmation_status") == "ai_candidate"):
            raise ValueError("正式知识图不能包含候选预览。")
        if not all(isinstance(node.get(field), list) for field in
                   ("source_refs", "subject_tags", "revision_refs")):
            raise ValueError("星图来源、学科或版本映射格式无效。")
    for edge in edges:
        if (not isinstance(edge, dict) or edge.get("source") not in identifiers
                or edge.get("target") not in identifiers
                or not edge.get("relation_type")
                or edge.get("confirmation_status") != ("ai_candidate" if preview else "persisted")):
            raise ValueError("星图关系包含悬空端点、未确认候选或缺少关系类型。")
    if len(_encoded(value)) > MAX_GRAPH_BYTES:
        raise ValueError("星图数据超过安全大小，请缩小资料库范围。")


class GraphGenerationStore:
    """Preview and atomically publish a library-scoped, rollbackable graph file."""

    def __init__(self, runtime_dir: Path, database_path: Path, *, library_uuid: str = "") -> None:
        library = os.path.normcase(str(database_path.resolve(strict=False)))
        self.scope = _fingerprint(f"{library}:{library_uuid}")
        self.path = runtime_dir / "knowledge-graph" / self.scope / "knowledge_graph.json"
        self.previous_path = self.path.with_name("knowledge_graph.previous.json")

    def load(self) -> dict[str, Any] | None:
        """Read a validated artifact without initializing or modifying anything."""
        return self._load(self.path)

    def renderer_contract(self, graph: KnowledgeGraph) -> dict[str, Any]:
        """Use a matching generated artifact, otherwise the current validated view.

        A stale or rolled-back export must never mask a user's later SQLite
        edits. This method does not automatically publish a changed artifact.
        """
        current = graph_contract(graph)
        generated = self.load()
        if generated and generated["fingerprint"] == _fingerprint(current):
            return self._content(generated)
        return current

    def _load(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        if path.stat().st_size > MAX_GRAPH_BYTES:
            raise ValueError("星图生成文件超过安全大小。")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("scope") != self.scope:
            raise ValueError("星图文件所属资料库不匹配。")
        validate_graph_contract(value)
        if value.get("mode") == "preview":
            raise ValueError("正式生成文件不能载入候选预览。")
        if value.get("fingerprint") != _fingerprint(self._content(value)):
            raise ValueError("星图生成文件校验失败。")
        return value

    @staticmethod
    def _content(value: dict[str, Any]) -> dict[str, Any]:
        return {key: item for key, item in value.items()
                if key not in {"scope", "fingerprint", "generated_at", "generation_mode"}}

    def preview(self, graph: KnowledgeGraph, *, mode: str = "incremental") -> dict[str, Any]:
        """Return exact changes and an optimistic token; no file is written."""
        if graph.is_preview:
            raise ValueError("AI 候选预览尚未人工确认，不能生成正式知识图文件。")
        if mode not in {"incremental", "full"}:
            raise ValueError("请选择增量更新或全量重新生成。")
        if graph.shown < graph.total:
            raise ValueError("当前图谱仅包含部分节点，请清除搜索并提高上限后再生成。")
        current = graph_contract(graph)
        previous = self.load()
        before = {node["id"]: node for node in previous["nodes"]} if previous else {}
        after = {node["id"]: node for node in current["nodes"]}
        fingerprint = _fingerprint(current)
        old_fingerprint = previous["fingerprint"] if previous else ""
        return {
            "mode": mode, "fingerprint": fingerprint,
            "token": _fingerprint([self.scope, mode, fingerprint, old_fingerprint]),
            "added": len(after.keys() - before.keys()),
            "removed": len(before.keys() - after.keys()),
            "changed": sum(before[key] != after[key] for key in before.keys() & after.keys()),
            "node_count": len(after), "edge_count": len(current["edges"]),
            "relations_changed": (previous is None or previous["edges"] != current["edges"]),
            "warnings": ["未确认的已保存知识明确标记；生成不等于人工确认。"]
            if any(n["confirmation_status"] in {"unconfirmed", "stale"}
                   for n in current["nodes"]) else [],
        }

    def generate(self, graph: KnowledgeGraph, *, preview_token: str,
                 mode: str = "incremental") -> dict[str, Any]:
        """Commit only the reviewed snapshot, preserving the last valid artifact."""
        preview = self.preview(graph, mode=mode)
        if preview_token != preview["token"]:
            raise ValueError("知识图或已有生成文件已变化，请重新预览后再生成。")
        content = graph_contract(graph)
        previous = self.load()
        if previous and previous["fingerprint"] == preview["fingerprint"]:
            return previous
        result = {**content, "scope": self.scope, "fingerprint": preview["fingerprint"],
                  "generation_mode": mode, "generated_at": datetime.now(UTC).isoformat()}
        if previous:
            self._write(self.previous_path, previous)
        self._write(self.path, result)
        return result

    def rollback(self) -> dict[str, Any]:
        """Restore the last valid data projection; knowledge and layouts stay intact."""
        previous = self._load(self.previous_path)
        if previous is None:
            raise ValueError("没有可回滚的上一次星图生成文件。")
        self._write(self.path, previous)
        return previous

    @staticmethod
    def _write(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{uuid4().hex}.tmp")
        try:
            temporary.write_bytes(_encoded(value))
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
