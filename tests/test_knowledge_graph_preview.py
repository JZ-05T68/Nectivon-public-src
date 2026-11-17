"""Both graph previews show exact saved candidates without formal persistence."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from src.database import Database
from src.knowledge_graph_generation import (
    GraphGenerationStore,
    graph_contract,
    validate_graph_contract,
)
from src.knowledge_graph_preview import (
    clear_preview,
    list_previews,
    load_preview,
    preview_graph,
    save_preview,
)
from src.knowledge_graph_service import KnowledgeGraphService
from src.knowledge_mindmap import LoadedLayout, mindmap_payload
from src.knowledge_mining_service import (
    KnowledgeMiningDraft,
    MinedKnowledgePoint,
    MinedKnowledgeRelation,
)
from src.models import KnowledgeRelationType
from src.study_knowledge_bridge import SavedReviewKnowledgeDraft


@pytest.fixture()
def candidate(tmp_path):
    database = Database(tmp_path / "library.db")
    source = {"library_id": "study-library", "study_review_id": "review-1",
              "study_review_revision_id": "version-1", "revision_number": 1,
              "course_id": "电机驱动与运动控制", "course_ids": ["电机驱动与运动控制"],
              "materials": [{"material_id": "material-1", "sha256": "a" * 64}],
              "source_page_refs": [{"material_id": "material-1", "page_number": 3}]}
    draft = SavedReviewKnowledgeDraft(source, KnowledgeMiningDraft(
        document_id=0, document_title="电机驱动与运动控制已保存复盘", model="fake-test",
        points=(MinedKnowledgePoint("转矩", "合成定义", "primary", page_numbers=(3,),
                                    classifications=(("电气工程", "电机与电器"),), evidence="原文"),
                MinedKnowledgePoint("电流", "合成前提", "secondary", origin="inferred")),
        relations=(MinedKnowledgeRelation("转矩", "电流", KnowledgeRelationType.REQUIRES,
                                          "复盘明确说明前置条件"),),
    ))
    return database, tmp_path / "runtime", draft


def test_saved_preview_roundtrip_is_namespaced_and_does_not_write_formal_graph(candidate, tmp_path):
    database, runtime, draft = candidate
    before = database.database_path.read_bytes()
    identity = save_preview(database, runtime, draft)
    assert load_preview(database, runtime) == draft
    assert list_previews(database, runtime)[0]["id"] == identity
    assert list_previews(database, runtime)[0]["node_count"] == 2
    assert not KnowledgeGraphService(database).snapshot().nodes
    assert database.database_path.read_bytes() == before
    other = Database(tmp_path / "other.db")
    assert load_preview(other, runtime) is None
    with pytest.raises(ValueError, match="不存在"):
        load_preview(other, runtime, identity)
    clear_preview(database, runtime, identity)
    assert load_preview(database, runtime) is None
    assert load_preview(database, runtime, identity) == draft  # Candidate history retained.


def test_2d_3d_candidates_share_ids_course_hub_provenance_and_typed_direction(candidate):
    database, _, draft = candidate
    graph = preview_graph(draft, database)
    assert graph.is_preview
    assert all(identifier.startswith("preview:") for identifier in graph.nodes)
    assert {tuple(node["tags"]) for node in graph.nodes.values()} == {("电机驱动与运动控制",)}
    assert graph.nodes[next(iter(graph.nodes))]["meta"]["preview_point_index"] == 0
    first_meta = graph.nodes[next(iter(graph.nodes))]["meta"]
    assert first_meta["academic_subject_tags"] == ["电气工程/电机与电器"]
    two_d = mindmap_payload(graph, LoadedLayout({"positions": {}, "collapsed": [],
                                               "viewport": {"x": 0, "y": 0, "scale": 1}}), "scope")
    three_d = graph.star_payload()["knowledge_graph"]
    assert three_d["mode"] == "preview" and two_d["is_preview"]
    assert {n["id"] for n in two_d["nodes"]} == {n["id"] for n in three_d["nodes"]}
    assert all(n["status"] == "AI 候选 · 待确认" for n in two_d["nodes"])
    assert three_d["edges"][0]["source"] == list(graph.nodes)[0]
    assert three_d["edges"][0]["target"] == list(graph.nodes)[1]
    assert three_d["edges"][0]["relation_type"] == "requires"
    assert three_d["edges"][0]["confirmation_status"] == "ai_candidate"
    assert three_d["nodes"][0]["revision_refs"][0] == draft.source
    validate_graph_contract(three_d)


def test_candidates_cannot_enter_formal_export_or_be_relabelled(candidate):
    database, runtime, draft = candidate
    graph = preview_graph(draft, database)
    with pytest.raises(ValueError, match="尚未人工确认"):
        GraphGenerationStore(runtime, database.database_path).preview(graph)
    with pytest.raises(ValueError, match="不能改作正式"):
        graph_contract(replace(graph, is_preview=False))
    value = graph_contract(graph)
    value["mode"] = "formal"
    with pytest.raises(ValueError, match="不能包含候选"):
        validate_graph_contract(value)


def test_preview_tampering_and_paths_are_rejected(candidate):
    database, runtime, draft = candidate
    identity = save_preview(database, runtime, draft)
    path = next(runtime.glob(f"knowledge-preview/*/{identity}.json"))
    value = json.loads(path.read_text(encoding="utf-8"))
    value["draft"]["mining"]["points"][0]["content"] = "tampered"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="校验失败"):
        load_preview(database, runtime)
    with pytest.raises(ValueError, match="标识无效"):
        load_preview(database, runtime, "../other")


def test_formal_import_course_mapping_preserves_shared_identity_and_academic_metadata():
    nodes = {"knowledge:1": {"tags": ["电气工程/电机与电器"], "meta": {"revision_refs": [
        {"course_ids": ["电机驱动与运动控制", "控制理论"]}]}},
             "page:7": {"tags": [], "meta": {}}, "document:2": {"tags": [], "meta": {}}}
    edges = [{"source": "knowledge:1", "target": "page:7", "label": "来自页面"},
             {"source": "page:7", "target": "document:2", "label": "所属文档"}]
    KnowledgeGraphService._apply_review_course_tags(nodes, edges)
    for node in nodes.values():
        assert set(node["tags"]) == {"电机驱动与运动控制", "控制理论"}
    assert len(nodes) == 3
    assert nodes["knowledge:1"]["meta"]["academic_subject_tags"] == ["电气工程/电机与电器"]
