"""Read-only graph projection and library-scoped, durable canvas preferences.

Only the JSON view state is writable here. Knowledge entities, source bindings
and relations remain owned by their existing platform services.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.knowledge_graph_service import KnowledgeGraph

GRAPH_ID = "knowledge-connections-v1"
MAX_STATE_BYTES = 2_000_000


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _number(value: object, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("布局坐标或缩放格式无效。")
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError("布局坐标或缩放超出安全范围。")
    return float(value)


def _validated_state(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("布局内容格式无效。")
    positions = value.get("positions", {})
    if not isinstance(positions, dict) or len(positions) > 10000:
        raise ValueError("布局节点数量或格式无效。")
    clean: dict[str, dict[str, float]] = {}
    for identifier, point in positions.items():
        if not isinstance(identifier, str) or len(identifier) > 100:
            raise ValueError("布局节点标识无效。")
        if not isinstance(point, dict):
            raise ValueError("布局节点坐标无效。")
        clean[identifier] = {
            axis: _number(point.get(axis), lower=-1_000_000, upper=1_000_000)
            for axis in ("x", "y")
        }
    viewport = value.get("viewport", {"x": 0, "y": 0, "scale": 1})
    if not isinstance(viewport, dict):
        raise ValueError("画布状态格式无效。")
    camera = {axis: _number(viewport.get(axis), lower=-1_000_000, upper=1_000_000)
              for axis in ("x", "y")}
    camera["scale"] = _number(viewport.get("scale"), lower=0.00001, upper=4)
    collapsed = value.get("collapsed", [])
    if not isinstance(collapsed, list) or len(collapsed) > 10000 or any(
        not isinstance(item, str) or len(item) > 100 for item in collapsed
    ):
        raise ValueError("折叠状态格式无效。")
    return {"positions": clean, "viewport": camera, "collapsed": sorted(set(collapsed))}


@dataclass(frozen=True)
class LoadedLayout:
    """A usable layout plus a user-visible explanation of any safe fallback."""

    state: dict[str, Any]
    warning: str = ""
    saved_at: str = ""


class MindmapLayoutStore:
    """Persist only explicit layout saves, atomically and outside the content DB."""

    def __init__(self, runtime_dir: Path, database_path: Path,
                 graph_id: str = GRAPH_ID, *, library_uuid: str = "") -> None:
        library = os.path.normcase(str(database_path.resolve(strict=False)))
        # The existing database UUID also distinguishes a replacement library
        # created later at the same path, where integer entity IDs may restart.
        self.library_id = _digest(f"{library}:{library_uuid}")
        self.graph_id = graph_id
        self.scope = _digest(f"{self.library_id}:{graph_id}")
        self.path = runtime_dir / "view-state" / "mindmap" / self.library_id / (
            _digest(graph_id) + ".json"
        )

    def load(self) -> LoadedLayout:
        """Read without creating files; malformed state never damages source data."""
        empty = _validated_state({})
        try:
            if not self.path.exists():
                return LoadedLayout(empty)
            if self.path.stat().st_size > MAX_STATE_BYTES:
                raise ValueError("layout too large")
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if (not isinstance(value, dict) or value.get("version") != 1
                    or value.get("scope") != self.scope):
                raise ValueError("layout namespace mismatch")
            state = _validated_state(value.get("state"))
            return LoadedLayout(state, saved_at=str(value.get("saved_at", "")))
        except (OSError, ValueError, TypeError, RecursionError):
            return LoadedLayout(empty, "已保存布局不可读取或已损坏，当前使用默认布局；"
                                "知识内容与关系未改动。重新保存只替换本图布局。")

    def save(self, event: dict[str, Any], graph: KnowledgeGraph) -> LoadedLayout:
        """Accept known current nodes only, retaining positions outside a filter."""
        from datetime import UTC, datetime

        if event.get("scope") != self.scope:
            raise ValueError("布局所属资料库或图已变化，请刷新后重试。")
        incoming = _validated_state(event.get("layout"))
        if not set(incoming["positions"]).issubset(graph.nodes):
            raise ValueError("布局包含当前视图之外的节点，请刷新后重试。")
        documents = {node_id for node_id, node in graph.nodes.items()
                     if node["entity_kind"] == "document"}
        if not set(incoming["collapsed"]).issubset(documents):
            raise ValueError("折叠状态包含无效文档。")
        previous = self.load().state
        positions = dict(previous["positions"])
        positions.update(incoming["positions"])
        # Filtered-out layouts must survive a filtered save. Removed entities
        # may retain harmless coordinates; IDs are never used to invent nodes.
        collapsed = (set(previous["collapsed"]) - set(graph.nodes)) | set(incoming["collapsed"])
        state = _validated_state({**incoming, "positions": positions,
                                  "collapsed": sorted(collapsed)})
        saved_at = datetime.now(UTC).isoformat()
        payload = json.dumps({"version": 1, "scope": self.scope, "state": state,
                              "saved_at": saved_at}, ensure_ascii=False, allow_nan=False)
        if len(payload.encode("utf-8")) > MAX_STATE_BYTES:
            raise ValueError("布局文件超过安全大小，未保存。")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f".{uuid4().hex}.tmp")
        try:
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
        return LoadedLayout(state, saved_at=saved_at)


def mindmap_payload(graph: KnowledgeGraph, layout: LoadedLayout, scope: str) -> dict[str, Any]:
    """Preserve all graph semantics; only page membership defines a fold branch."""
    children: dict[str, list[str]] = {}
    for edge in graph.links:
        if (edge["label"] == "所属文档" and edge["source"].startswith("page:")
                and edge["target"].startswith("document:")):
            children.setdefault(edge["target"], []).append(edge["source"])
    nodes = []
    for identifier, node in graph.nodes.items():
        meta = node.get("meta", {})
        status = ""
        if node["entity_kind"] == "knowledge":
            status = ("AI 候选 · 待确认" if graph.is_preview else
                      "已确认" if meta.get("confirmation_current") else
                      "确认已过期" if meta.get("confirmation_status") == "confirmed" else "未确认")
        nodes.append({"id": identifier, "title": node["title"],
                      "kind": node["entity_kind"], "status": status})
    visible_ids = set(graph.nodes)
    state = {**layout.state,
             "positions": {key: point for key, point in layout.state["positions"].items()
                           if key in visible_ids},
             "collapsed": [key for key in layout.state["collapsed"] if key in children]}
    return {"scope": scope, "is_preview": graph.is_preview,
            "nodes": nodes, "links": graph.links,
            "children": {key: sorted(set(value)) for key, value in children.items()},
            "layout": state, "has_saved_layout": bool(layout.saved_at),
            "saved_at": layout.saved_at, "total": graph.total, "shown": graph.shown}
