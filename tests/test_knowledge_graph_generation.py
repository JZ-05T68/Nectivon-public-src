"""Isolated graph generation, rollback, and durable 3D manual-layout contracts."""

from __future__ import annotations

import copy
import json

import pytest

from src.database import Database
from src.knowledge_graph_generation import (
    GraphGenerationStore,
    graph_contract,
    validate_graph_contract,
)
from src.knowledge_graph_service import KnowledgeGraph, KnowledgeGraphService
from src.knowledge_object_service import KnowledgeObjectService
from src.knowledge_starmap_layout import StarmapLayoutStore


@pytest.fixture()
def library(tmp_path):
    """Create synthetic user-owned nodes and one manually saved relation."""
    database = Database(tmp_path / "library.db")
    service = KnowledgeObjectService(database)
    first = service.create(kind="concept", title="磁通", content="合成测试资料",
                           epistemic_basis="personal_judgment").knowledge_object
    second = service.create(
        kind="concept", title="感应电势", content="合成测试资料",
        epistemic_basis="personal_judgment", subject="电气工程",
    ).knowledge_object
    service.add_relation(second.id, first.id, relation_type="requires", description="人工确认前提")
    return database, service, GraphGenerationStore(tmp_path / "runtime", database.database_path)


def test_preview_no_writes_and_exact_type_direction_and_confirmation(library):
    database, _, store = library
    graph = KnowledgeGraphService(database).snapshot()
    preview = store.preview(graph)
    assert preview["added"] == 2 and preview["edge_count"] == 1
    assert not store.path.exists()
    value = graph_contract(graph)
    assert [n["id"] for n in value["nodes"]] == ["knowledge:1", "knowledge:2"]
    assert value["edges"][0]["source"] == "knowledge:2"
    assert value["edges"][0]["target"] == "knowledge:1"
    assert value["edges"][0]["relation_type"] == "requires"
    assert all(n["confirmation_status"] == "unconfirmed" for n in value["nodes"])
    assert value["nodes"][0]["subject_tags"] == []  # Renderer keeps unclassified nebula.
    assert value["nodes"][0]["revision_refs"] == [{"knowledge_object_id": 1, "revision": 1}]


def test_full_regeneration_keeps_manual_relation_and_layout_and_updates_content(library, tmp_path):
    database, service, store = library
    graph_service = KnowledgeGraphService(database)
    graph = graph_service.snapshot()
    positions = StarmapLayoutStore(tmp_path / "runtime", database.database_path)
    positions.save({"scope": positions.scope, "layout": {"positions": {
        "knowledge:1": {"x": 11, "y": -12, "z": 13}}}}, graph)
    before = positions.path.read_bytes()
    store.generate(graph, preview_token=store.preview(graph)["token"])
    service.update_content(1, title="人工修改磁通")
    current = graph_service.snapshot()
    preview = store.preview(current, mode="full")
    assert preview["changed"] == 1
    result = store.generate(current, mode="full", preview_token=preview["token"])
    assert result["nodes"][0]["title"] == "人工修改磁通"
    assert result["edges"][0]["relation_type"] == "requires"
    assert positions.path.read_bytes() == before
    assert StarmapLayoutStore(tmp_path / "runtime", database.database_path).payload(
        current)["layout"]["positions"]["knowledge:1"]["z"] == 13
    restored = store.rollback()
    assert restored["nodes"][0]["title"] == "磁通"
    assert service.get(1).title == "人工修改磁通"  # Artifact rollback is not content rollback.
    assert service.relations(1)
    assert positions.payload(current)["knowledge_graph"]["nodes"][0]["title"] == "人工修改磁通"


def test_stale_preview_and_partial_graph_refused_without_artifact_change(library):
    database, service, store = library
    graph = KnowledgeGraphService(database).snapshot()
    preview = store.preview(graph)
    service.update_content(1, title="更新")
    with pytest.raises(ValueError, match="重新预览"):
        store.generate(KnowledgeGraphService(database).snapshot(), preview_token=preview["token"])
    assert not store.path.exists()
    with pytest.raises(ValueError, match="部分节点"):
        store.preview(KnowledgeGraph(graph.nodes, graph.links, graph.total + 1, graph.shown))


def test_atomic_write_failure_preserves_last_valid_artifact(library, monkeypatch):
    database, service, store = library
    graph_service = KnowledgeGraphService(database)
    graph = graph_service.snapshot()
    store.generate(graph, preview_token=store.preview(graph)["token"])
    before = store.path.read_bytes()
    service.update_content(1, title="新版本")
    current = graph_service.snapshot()
    preview = store.preview(current)
    original = type(store.path).replace

    def fail_current(path, target):
        if target == store.path:
            raise OSError("synthetic disk failure")
        return original(path, target)

    monkeypatch.setattr(type(store.path), "replace", fail_current)
    with pytest.raises(OSError, match="disk failure"):
        store.generate(current, preview_token=preview["token"])
    assert store.path.read_bytes() == before
    assert store.load()["nodes"][0]["title"] == "磁通"


def test_generation_noop_preserves_timestamp_and_contract_rejects_candidates(library):
    database, _, store = library
    graph = KnowledgeGraphService(database).snapshot()
    store.generate(graph, preview_token=store.preview(graph)["token"])
    before = store.path.read_bytes()
    store.generate(graph, mode="full", preview_token=store.preview(graph, mode="full")["token"])
    assert store.path.read_bytes() == before
    bad = copy.deepcopy(graph_contract(graph))
    bad["edges"][0]["confirmation_status"] = "ai_candidate"
    with pytest.raises(ValueError, match="未确认候选"):
        validate_graph_contract(bad)
    bad = copy.deepcopy(graph_contract(graph))
    bad["nodes"].append(bad["nodes"][0])
    with pytest.raises(ValueError, match="重复"):
        validate_graph_contract(bad)
    bad = json.loads(before)
    bad["nodes"][0]["title"] = "tampered"
    store.path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="校验失败"):
        store.load()


def test_3d_filtered_layout_save_merges_and_rejects_cross_library(library, tmp_path):
    database, _, _ = library
    graph = KnowledgeGraphService(database).snapshot()
    store = StarmapLayoutStore(tmp_path / "runtime", database.database_path)
    point = {"x": 1, "y": 2, "z": 3}
    store.save({"scope": store.scope, "layout": {"positions": {"knowledge:1": point}}}, graph)
    filtered = KnowledgeGraphService(database).snapshot(query="感应电势")
    store.save({"scope": store.scope, "layout": {"positions": {"knowledge:2": point}}}, filtered)
    assert len(store.load()["positions"]) == 2
    with pytest.raises(ValueError, match="资料库"):
        store.save({"scope": "another-library", "layout": {"positions": {}}}, graph)
    with pytest.raises(ValueError, match="坐标"):
        store.save({"scope": store.scope, "layout": {"positions": {
            "knowledge:1": {"x": float("nan"), "y": 2, "z": 3}}}}, graph)


def test_complete_generation_is_independent_of_display_cap_and_keeps_all_relations(library):
    database, _, store = library
    # Duplicate synthetic rows directly to exercise >1000 nodes without 1000
    # unrelated source-confirmation service calls. No production data is used.
    with database._connection() as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(knowledge_objects)")
                   if row[1] != "id"]
        fields = ",".join(columns)
        template = connection.execute(
            f"SELECT {fields} FROM knowledge_objects WHERE id=1").fetchone()
        connection.executemany(
            f"INSERT INTO knowledge_objects ({fields}) VALUES ({','.join('?' for _ in columns)})",
            [tuple(template)] * 1001,
        )
    service = KnowledgeGraphService(database)
    display = service.snapshot(limit=1000)
    assert display.shown == 1000 and display.total == 1003
    complete = service.generation_snapshot()
    assert complete.shown == complete.total == 1003
    assert len(complete.links) == 1
    preview = store.preview(complete, mode="full")
    assert preview["node_count"] == 1003
    with database._connection() as connection:
        connection.executemany(
            f"INSERT INTO knowledge_objects ({fields}) VALUES ({','.join('?' for _ in columns)})",
            [tuple(template)] * 9000,
        )
    with pytest.raises(ValueError, match="10000"):
        service.generation_snapshot()
