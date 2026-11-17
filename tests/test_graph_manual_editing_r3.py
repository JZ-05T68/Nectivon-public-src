"""R3 manual edits preserve identity, history, and explicit removal consent."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.database import Database
from src.knowledge_graph_service import KnowledgeGraphService
from src.knowledge_graph_views import search_graph, subject_graph
from src.knowledge_mindmap import MindmapLayoutStore
from src.knowledge_object_service import KnowledgeObjectError, KnowledgeObjectService


@pytest.fixture
def graph_case(tmp_path):
    database = Database(tmp_path / "graph.db")
    service = KnowledgeObjectService(database)
    objects = [service.create(kind="concept", title=f"概念{number}", content=f"正文{number}",
                              epistemic_basis="personal_judgment").knowledge_object
               for number in range(3)]
    return database, service, objects


def test_relation_change_is_atomic_and_preserves_id_direction(graph_case):
    database, service, objects = graph_case
    a, b, _ = objects
    relation = service.add_relation(a.id, b.id, relation_type="supports", description="原依据")
    changed = service.update_relation(relation.id, relation_type="requires", description="新依据")
    assert (changed.id, changed.source_ko_id, changed.target_ko_id) == (relation.id, a.id, b.id)
    assert changed.created_at == relation.created_at
    assert changed.relation_type.value == "requires"
    service.add_relation(a.id, b.id, relation_type="supports")
    with pytest.raises(KnowledgeObjectError, match="已经存在"):
        service.update_relation(relation.id, relation_type="supports", description="不能覆盖")
    persisted = database.get_knowledge_relation(relation.id)
    assert persisted.description == "新依据" and persisted.relation_type.value == "requires"
    assert len(service.relations(a.id)) == 2


def _editor_app(graph_case):
    database, _, objects = graph_case
    app = AppTest.from_string("""
import streamlit as st
from src.database import Database
from src.knowledge_graph_editor_ui import render_selected_knowledge_editor
render_selected_knowledge_editor(Database(st.session_state['db']), st.session_state['selected'])
""")
    app.session_state["db"] = str(database.database_path)
    app.session_state["selected"] = f"knowledge:{objects[0].id}"
    return app.run(timeout=30)


def test_manual_content_save_retains_revision_and_both_views(graph_case):
    database, service, objects = graph_case
    a = objects[0]
    service.confirm(a.id)
    app = _editor_app(graph_case)
    assert not app.exception
    next(item for item in app.text_input if item.label == "知识点标题").set_value("人工修改")
    next(item for item in app.text_area if item.label == "知识正文").set_value("新正文")
    next(item for item in app.button if item.label == "保存知识修改").click().run()
    assert not app.exception
    assert service.get(a.id).content == "新正文"
    assert service.get(a.id).confirmation_is_stale
    assert any(item.before_content == a.content for item in service.revisions(a.id))
    graph = KnowledgeGraphService(database).snapshot()
    assert graph.nodes[f"knowledge:{a.id}"]["title"] == "人工修改"
    assert next(item for item in graph.star_payload()["vault"]["notes"]
                if item["id"] == f"knowledge:{a.id}")["title"] == "人工修改"


def test_manual_relation_removal_requires_second_action(graph_case):
    _, service, objects = graph_case
    a, b, c = objects
    removed = service.add_relation(a.id, b.id, relation_type="requires")
    retained = service.add_relation(a.id, c.id, relation_type="supports")
    app = _editor_app(graph_case)
    next(item for item in app.button if item.label == "删除所选关系").click().run()
    assert not app.exception
    assert len(service.relations(a.id)) == 2
    # A changed picker must not switch which relation the pending confirmation removes.
    app.selectbox(key=f"graph_edit_relation_{a.id}").set_value(retained.id).run()
    next(item for item in app.button if item.label == "确认删除关系").click().run()
    assert not app.exception
    assert [item.id for item in service.relations(a.id)] == [retained.id]
    assert removed.id != retained.id
    assert all(service.get(item.id) for item in objects)


def test_secondary_graph_controls_live_inside_single_advanced_panel():
    """Check hierarchy instead of merely checking that a label appears somewhere."""
    import ast

    page = Path(__file__).parents[1] / "pages" / "20_知识串联.py"
    tree = ast.parse(page.read_text(encoding="utf-8"))
    advanced = [node for node in ast.walk(tree) if isinstance(node, ast.With)
                and any(isinstance(item.context_expr, ast.Call)
                        and item.context_expr.args
                        and isinstance(item.context_expr.args[0], ast.Constant)
                        and item.context_expr.args[0].value == "高级设置" for item in node.items)]
    assert len(advanced) == 1
    calls = [node.func.id for node in ast.walk(advanced[0])
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert {"render_graph_editor", "render_material_mining",
            "render_selected_knowledge_editor"}.issubset(calls)


def test_subject_maps_share_identity_but_keep_independent_layout(graph_case, tmp_path):
    database, service, objects = graph_case
    from src.knowledge_subject_service import KnowledgeSubjectService

    classifications = KnowledgeSubjectService(database)
    classifications.add_subjects(objects[0].id, ["数学", "物理"])
    classifications.add_subjects(objects[1].id, ["数学"])
    classifications.add_subjects(objects[2].id, ["物理"])
    relation = service.add_relation(objects[0].id, objects[1].id, relation_type="requires")
    graph = KnowledgeGraphService(database).snapshot()
    math, physics = subject_graph(graph, "数学"), subject_graph(graph, "物理")
    shared = f"knowledge:{objects[0].id}"
    assert set(math.nodes) == {shared, f"knowledge:{objects[1].id}"}
    assert set(physics.nodes) == {shared, f"knowledge:{objects[2].id}"}
    assert math.links[0]["source"] == shared and math.links[0]["target"] in math.nodes
    assert not physics.links and database.get_knowledge_relation(relation.id) == relation
    stores = [MindmapLayoutStore(tmp_path, database.database_path, graph_id=subject)
              for subject in ("数学", "物理")]
    stores[0].save({"scope": stores[0].scope, "layout": {
        "positions": {shared: {"x": 4, "y": 7}}}}, math)
    assert stores[1].load().state["positions"] == {}
    assert shared in stores[0].load().state["positions"]


def test_unclassified_map_remains_accessible_and_preview_search_retains_identity(graph_case):
    database, _, objects = graph_case
    graph = KnowledgeGraphService(database).snapshot()
    unclassified = subject_graph(graph, "未归类")
    assert set(unclassified.nodes) == {f"knowledge:{item.id}" for item in objects}
    filtered = search_graph(graph, objects[1].title)
    assert list(filtered.nodes) == [f"knowledge:{objects[1].id}"]
    assert not search_graph(graph, "不存在的词").nodes
