"""Explicit 3D position saves, independent of knowledge data regeneration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.knowledge_graph_generation import GraphGenerationStore
from src.knowledge_graph_service import KnowledgeGraph


class StarmapLayoutStore:
    """Store user-pinned node positions in a library-specific runtime namespace."""

    def __init__(self, runtime_dir: Path, database_path: Path, *, library_uuid: str = "") -> None:
        generation = GraphGenerationStore(runtime_dir, database_path, library_uuid=library_uuid)
        self.generation = generation
        self.scope = generation.scope
        self.path = runtime_dir / "view-state" / "starmap" / self.scope / "layout.json"

    @staticmethod
    def _positions(value: Any) -> dict[str, dict[str, float]]:
        if not isinstance(value, dict) or len(value) > 10000:
            raise ValueError("三维布局节点格式或数量无效。")
        positions = {}
        for identifier, point in value.items():
            if (not isinstance(identifier, str) or len(identifier) > 100
                    or not isinstance(point, dict)):
                raise ValueError("三维布局节点标识或坐标无效。")
            clean = {}
            for axis in ("x", "y", "z"):
                number = point.get(axis)
                if (isinstance(number, bool) or not isinstance(number, (float, int))
                        or not math.isfinite(number) or abs(number) > 1_000_000):
                    raise ValueError("三维布局坐标超出安全范围。")
                clean[axis] = float(number)
            positions[identifier] = clean
        return positions

    def load(self) -> dict[str, Any]:
        """Read only; malformed saved positions are reported, never silently replaced."""
        if not self.path.exists():
            return {"positions": {}}
        if self.path.stat().st_size > 2_000_000:
            raise ValueError("三维布局文件超过安全大小。")
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if value.get("scope") != self.scope:
            raise ValueError("三维布局所属资料库已变化。")
        return {"positions": self._positions(value.get("positions"))}

    def payload(self, graph: KnowledgeGraph) -> dict[str, Any]:
        """Supply positions separately from the graph's semantic contract."""
        return {**graph.star_payload(), "knowledge_graph": self.generation.renderer_contract(graph),
                "scope": self.scope,
                "layout": self.load(), "layout_editing": True}

    def save(self, event: dict[str, Any], graph: KnowledgeGraph) -> None:
        """Merge an explicit save while retaining positions outside a filtered view."""
        if event.get("scope") != self.scope:
            raise ValueError("三维布局所属资料库已变化，请刷新后重试。")
        incoming = self._positions(event.get("layout", {}).get("positions"))
        if not set(incoming).issubset(graph.nodes):
            raise ValueError("三维布局包含当前视图之外的节点，请刷新后重试。")
        positions = self.load()["positions"]
        positions.update(incoming)
        value = json.dumps({"scope": self.scope, "positions": positions},
                           ensure_ascii=False, allow_nan=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f".{uuid4().hex}.tmp")
        try:
            temporary.write_text(value, encoding="utf-8")
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
