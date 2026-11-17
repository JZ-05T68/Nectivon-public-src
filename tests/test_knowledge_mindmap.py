"""NON_REAL_TEST_DATA: graph semantics and durable view-state contracts."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import src.knowledge_mindmap_ui as mindmap_ui
import src.knowledge_starmap_component as star_component
import src.runtime as runtime
import src.workspace_ui as workspace
from src.database import Database
from src.knowledge_graph_service import KnowledgeGraph, KnowledgeGraphService
from src.knowledge_mindmap import MindmapLayoutStore, mindmap_payload
from src.knowledge_object_service import KnowledgeObjectService


def _graph(*ids: str) -> KnowledgeGraph:
    return KnowledgeGraph({identifier: {"id": identifier, "title": f"TEST_NODE_{identifier}",
                                       "entity_kind": identifier.split(":")[0], "meta": {}}
                           for identifier in ids}, [], len(ids), len(ids))


def _event(store, positions, **kwargs):
    return {"scope": store.scope, "layout": {"positions": positions, **kwargs}}


def test_layout_explicit_save_reopen_and_cross_library_graph_isolation(tmp_path):
    graph = _graph("knowledge:1")
    store = MindmapLayoutStore(tmp_path / "runtime", tmp_path / "a.db")
    assert not store.load().saved_at and not store.path.exists()
    store.save(_event(store, {"knowledge:1": {"x": 123, "y": -45}}), graph)
    reopened = MindmapLayoutStore(tmp_path / "runtime", tmp_path / "a.db")
    assert reopened.load().state["positions"] == {"knowledge:1": {"x": 123, "y": -45}}
    for other in (MindmapLayoutStore(tmp_path / "runtime", tmp_path / "b.db"),
                  MindmapLayoutStore(tmp_path / "runtime", tmp_path / "a.db", "other-graph")):
        assert other.scope != store.scope and not other.load().state["positions"]
        with pytest.raises(ValueError, match="资料库或图"):
            other.save(_event(store, {"knowledge:1": {"x": 1, "y": 1}}), graph)


def test_replaced_library_at_same_path_does_not_reuse_layout(tmp_path):
    old = MindmapLayoutStore(tmp_path / "runtime", tmp_path / "a.db", library_uuid="TEST_KB_1")
    old.save(_event(old, {"knowledge:1": {"x": 1, "y": 2}}), _graph("knowledge:1"))
    new = MindmapLayoutStore(tmp_path / "runtime", tmp_path / "a.db", library_uuid="TEST_KB_2")
    assert new.scope != old.scope and not new.load().state["positions"]


def test_fitted_large_graph_zoom_is_durable(tmp_path):
    store = MindmapLayoutStore(tmp_path / "runtime", tmp_path / "a.db")
    store.save(_event(store, {"knowledge:1": {"x": -999999, "y": 999999}},
                      viewport={"x": 500, "y": 250, "scale": .0002}), _graph("knowledge:1"))
    assert store.load().state["viewport"]["scale"] == .0002


def test_filtered_save_new_nodes_and_removed_nodes_preserve_existing_layout(tmp_path):
    store = MindmapLayoutStore(tmp_path / "runtime", tmp_path / "a.db")
    first = _graph("knowledge:1", "knowledge:2")
    store.save(_event(store, {"knowledge:1": {"x": 1, "y": 2},
                              "knowledge:2": {"x": 3, "y": 4}}), first)
    store.save(_event(store, {"knowledge:1": {"x": 7, "y": 8}}), _graph("knowledge:1"))
    updated = _graph("knowledge:1", "knowledge:2", "knowledge:3")
    payload = mindmap_payload(updated, store.load(), store.scope)
    assert payload["layout"]["positions"]["knowledge:2"] == {"x": 3, "y": 4}
    assert "knowledge:3" not in payload["layout"]["positions"]
    removed = mindmap_payload(_graph("knowledge:2"), store.load(), store.scope)
    assert set(removed["layout"]["positions"]) == {"knowledge:2"}


@pytest.mark.parametrize("content", ["broken", "[]", '{"version":1,"scope":"wrong"}',
                                      '{"version":1,"state":{"positions":[]}}'])
def test_corrupt_layout_is_read_only_safe_fallback(tmp_path, content):
    store = MindmapLayoutStore(tmp_path / "runtime", tmp_path / "a.db")
    store.path.parent.mkdir(parents=True)
    store.path.write_text(content)
    assert store.load().warning and not store.load().state["positions"]
    assert store.path.read_text() == content


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1e8, "12", True])
def test_nonfinite_or_invalid_positions_rejected_without_write(tmp_path, bad):
    store = MindmapLayoutStore(tmp_path / "runtime", tmp_path / "a.db")
    with pytest.raises(ValueError):
        store.save(_event(store, {"knowledge:1": {"x": bad, "y": 0}}), _graph("knowledge:1"))
    assert not store.path.exists()


def test_unknown_node_and_false_hierarchy_save_rejected(tmp_path):
    store = MindmapLayoutStore(tmp_path / "runtime", tmp_path / "a.db")
    with pytest.raises(ValueError, match="之外"):
        store.save(_event(store, {"knowledge:999": {"x": 0, "y": 0}}), _graph("knowledge:1"))
    with pytest.raises(ValueError, match="折叠"):
        store.save(_event(store, {}, collapsed=["knowledge:1"]), _graph("knowledge:1"))


def test_real_snapshot_ids_status_sources_cycles_and_multi_parents(tmp_path):
    database = Database(tmp_path / "knowledge.db")
    service = KnowledgeObjectService(database)
    document = database.create_document(
        title="TEST_DOCUMENT", filename="NON_REAL_TEST_DATA.pdf",
        source_path=tmp_path / "x.pdf", sha256="0" * 64, page_count=1,
    )
    page = database.create_page(document_id=document.id, page_number=1,
                                image_path=tmp_path / "p.png", extracted_text="NON_REAL_TEST_DATA")
    records = [service.create(
        kind="concept", title=f"TEST_NODE_{index}", content="NON_REAL_TEST_DATA",
        epistemic_basis="source_derived", source_links=[("page", page.id, "NON_REAL_TEST_DATA")],
    ).knowledge_object for index in range(3)]
    for source, target, relation in [(0, 1, "requires"), (1, 0, "relates_to"), (2, 1, "supports")]:
        service.add_relation(records[source].id, records[target].id, relation_type=relation)
    graph = KnowledgeGraphService(database).snapshot()
    before = copy.deepcopy(graph)
    store = MindmapLayoutStore(tmp_path / "runtime", database.database_path)
    payload = mindmap_payload(graph, store.load(), store.scope)
    assert graph == before
    assert payload["links"] == graph.links
    assert payload["children"] == {f"document:{document.id}": [f"page:{page.id}"]}
    assert all(node["status"] == "未确认" for node in payload["nodes"]
               if node["kind"] == "knowledge")
    assert len([edge for edge in payload["links"] if edge["target"] == "knowledge:2"]) == 2
    store.save(_event(store, {"knowledge:1": {"x": 2, "y": 4}}), graph)
    assert KnowledgeGraphService(database).snapshot() == before


def test_confirmation_requires_current_revision_and_missing_source_is_not_invented(tmp_path):
    database = Database(tmp_path / "knowledge.db")
    service = KnowledgeObjectService(database)
    item = service.create(kind="concept", title="TEST_NODE_A", content="NON_REAL_TEST_DATA",
                           epistemic_basis="personal_judgment").knowledge_object
    service.confirm(item.id)
    store = MindmapLayoutStore(tmp_path / "runtime", database.database_path)
    graph = KnowledgeGraphService(database).snapshot()
    assert mindmap_payload(graph, store.load(), store.scope)["nodes"][0]["status"] == "已确认"
    service.update_content(item.id, content="NON_REAL_TEST_DATA_REVISED")
    graph = KnowledgeGraphService(database).snapshot()
    assert mindmap_payload(graph, store.load(), store.scope)["nodes"][0]["status"] != "已确认"
    assert not graph.links


def test_invalid_source_id_is_not_replaced_by_same_title_material(tmp_path):
    database = Database(tmp_path / "knowledge.db")
    service = KnowledgeObjectService(database)
    for digest in ("a", "b"):
        document = database.create_document(
            title="TEST_DOCUMENT_SAME_TITLE", filename=f"NON_REAL_TEST_DATA_{digest}.pdf",
            source_path=tmp_path / f"{digest}.pdf", sha256=digest * 64, page_count=1,
        )
        page = database.create_page(document_id=document.id, page_number=1,
                                    image_path=tmp_path / f"{digest}.png",
                                    extracted_text="NON_REAL_TEST_DATA_IDENTICAL_TEXT")
    item = service.create(kind="concept", title="TEST_NODE_A", content="NON_REAL_TEST_DATA",
                           epistemic_basis="source_derived",
                           source_links=[("page", page.id, "NON_REAL_TEST_DATA")]).knowledge_object
    # Simulate an unavailable target in this disposable fixture without deleting
    # any real material. Same-title existing documents must never substitute.
    with database._connection() as connection:
        connection.execute("UPDATE knowledge_object_sources SET source_id = 999999 "
                           "WHERE knowledge_object_id = ?", (item.id,))
    graph = KnowledgeGraphService(database).snapshot()
    assert set(graph.nodes) == {f"knowledge:{item.id}"} and not graph.links
    assert service.source_views(item.id)[0].status.value == "missing"


def test_atomic_save_failure_retains_previous_file(tmp_path, monkeypatch):
    store = MindmapLayoutStore(tmp_path / "runtime", tmp_path / "a.db")
    graph = _graph("knowledge:1")
    store.save(_event(store, {"knowledge:1": {"x": 1, "y": 2}}), graph)
    before = store.path.read_bytes()

    def fail_replace(self, target):
        raise OSError("NON_REAL_TEST_DATA simulated unavailable destination")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError):
        store.save(_event(store, {"knowledge:1": {"x": 9, "y": 10}}), graph)
    assert store.path.read_bytes() == before
    assert not list(store.path.parent.glob("*.tmp"))


def test_page_component_save_selection_and_replayed_event_are_idempotent(tmp_path, monkeypatch):
    database = Database(tmp_path / "knowledge.db")
    service = KnowledgeObjectService(database)
    for label in ("A", "B"):
        service.create(kind="concept", title=f"TEST_NODE_{label}",
                       content="NON_REAL_TEST_DATA", epistemic_basis="personal_judgment")
    monkeypatch.setattr(runtime, "application_database", lambda: database)
    monkeypatch.setattr(runtime, "application_settings", lambda: SimpleNamespace(
        runtime_dir=tmp_path / "runtime", cache_dir=tmp_path / "cache"))
    monkeypatch.setattr(runtime, "application_ai_provider", lambda: None)
    monkeypatch.setattr(workspace, "render_workspace", lambda _: None)
    store = MindmapLayoutStore(tmp_path / "runtime", database.database_path,
                              graph_id="knowledge-subject:未归类",
                              library_uuid=database.get_knowledge_base_uuid())
    event = {"action": "save", "event_id": "NON_REAL_TEST_DATA_SAVE_1",
             **_event(store, {"knowledge:1": {"x": 77, "y": -3}},
                      viewport={"x": 4, "y": 9, "scale": .7})}
    monkeypatch.setattr(mindmap_ui, "render_mindmap", lambda payload: event)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "pages/20_知识串联.py")).run()
    assert not app.exception and store.path.exists()
    saved = store.path.read_bytes()
    app.run()
    assert not app.exception and store.path.read_bytes() == saved
    event = {"action": "select", "event_id": "NON_REAL_TEST_DATA_SELECT_1",
             "scope": store.scope, "node_id": "knowledge:1"}
    app.run()
    assert not app.exception and app.selectbox(key="graph_node_picker").value == "knowledge:1"
    assert store.path.read_bytes() == saved


def test_empty_and_single_node_payload(tmp_path):
    store = MindmapLayoutStore(tmp_path / "runtime", tmp_path / "a.db")
    assert mindmap_payload(_graph(), store.load(), store.scope)["nodes"] == []
    payload = mindmap_payload(_graph("knowledge:1"), store.load(), store.scope)
    assert len(payload["nodes"]) == 1 and not payload["links"] and not payload["children"]


def test_large_projection_explicit_counts_and_source_semantics(tmp_path):
    graph = _graph(*(f"knowledge:{index}" for index in range(1000)))
    graph.links.extend({"source": f"knowledge:{index}", "target": f"knowledge:{index + 1}",
                        "label": "相关", "note": "NON_REAL_TEST_DATA"} for index in range(999))
    store = MindmapLayoutStore(tmp_path / "runtime", tmp_path / "a.db")
    payload = mindmap_payload(graph, store.load(), store.scope)
    assert payload["shown"] == payload["total"] == len(payload["nodes"]) == 1000
    assert len(payload["links"]) == 999 and not payload["children"]


def test_view_default_switch_and_details_reuse(tmp_path, monkeypatch):
    database = Database(tmp_path / "knowledge.db")
    original = "NON_REAL_TEST_DATA $x^2$ ![TEST_IMAGE](https://invalid.test/never-load.png)"
    KnowledgeObjectService(database).create(kind="concept", title="TEST_NODE_A",
                                            content=original,
                                            epistemic_basis="personal_judgment")
    monkeypatch.setattr(runtime, "application_database", lambda: database)
    monkeypatch.setattr(runtime, "application_settings", lambda: SimpleNamespace(
        runtime_dir=tmp_path / "runtime", cache_dir=tmp_path / "cache"))
    monkeypatch.setattr(runtime, "application_ai_provider", lambda: None)
    monkeypatch.setattr(workspace, "render_workspace", lambda _: None)
    seen = []
    monkeypatch.setattr(mindmap_ui, "render_mindmap", lambda payload: seen.append("2d"))
    monkeypatch.setattr(star_component, "render_starmap", lambda payload: seen.append("star"))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "pages/20_知识串联.py")).run()
    assert not app.exception
    assert app.radio(key="graph_view_mode").value == "二维思维导图" and "2d" in seen
    assert any("未确认" in caption.value for caption in app.caption)
    assert any("尚未关联平台资料来源" in item.value for item in app.info)
    reading = [item.value for item in app.markdown if "never-load.png" in item.value]
    assert reading and all("![" not in text for text in reading)
    assert any("$x^2$" in text for text in reading)
    assert KnowledgeObjectService(database).get(1).content == original
    app.radio(key="graph_view_mode").set_value("知识星图").run()
    assert not app.exception and "star" in seen
    app.radio(key="graph_view_mode").set_value("二维思维导图").run()
    assert not app.exception and seen.count("2d") == 2


def test_component_has_no_network_or_html_text_interpolation():
    root = Path(__file__).parents[1] / "src/components/knowledge_mindmap"
    html = (root / "index.html").read_text("utf-8")
    js = (root / "mindmap.js").read_text("utf-8")
    assert "connect-src 'none'" in html and "font-src 'none'" in html
    assert "https://" not in html + js and "http://" not in html
    assert "innerHTML" not in js and "textContent" in js


def test_node_geometry_contract():
    import shutil

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js unavailable")
    result = subprocess.run([node, "--test", str(Path(__file__).parent / "js/mindmap.test.cjs")],
                            capture_output=True, text=True, encoding="utf-8", check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_layout_read_error_warns_and_wrong_scope_file_never_loads(tmp_path):
    store = MindmapLayoutStore(tmp_path / "runtime", tmp_path / "a.db")
    store.path.parent.mkdir(parents=True)
    store.path.mkdir()
    assert store.load().warning
    other = MindmapLayoutStore(tmp_path / "runtime", tmp_path / "b.db")
    other.path.parent.mkdir(parents=True)
    other.path.write_text(json.dumps({"version": 1, "scope": store.scope, "state": {}}))
    assert other.load().warning


@pytest.fixture
def multipage_source(tmp_path):
    """NON_REAL_TEST_DATA: physical page 2 deliberately has stable database ID 3."""
    database = Database(tmp_path / "knowledge.db")
    earlier = database.create_document(
        title="TEST_DOCUMENT_EARLIER", filename="NON_REAL_TEST_DATA_EARLIER.pdf",
        source_path=tmp_path / "earlier.pdf", sha256="a" * 64, page_count=1,
    )
    database.create_page(document_id=earlier.id, page_number=1,
                         image_path=tmp_path / "earlier.png", extracted_text="NON_REAL_TEST_DATA")
    target = database.create_document(
        title="TEST_DOCUMENT_TARGET", filename="NON_REAL_TEST_DATA_TARGET.pdf",
        source_path=tmp_path / "target.pdf", sha256="b" * 64, page_count=2,
    )
    for number in (1, 2):
        page = database.create_page(document_id=target.id, page_number=number,
                                    image_path=tmp_path / f"target_{number}.png",
                                    extracted_text=f"NON_REAL_TEST_DATA PAGE_{number}")
    knowledge = KnowledgeObjectService(database).create(
        kind="concept", title="TEST_NODE_PAGE_2", content="NON_REAL_TEST_DATA",
        epistemic_basis="source_derived", source_links=[("page", page.id, "NON_REAL_TEST_DATA")],
    ).knowledge_object
    with database._connection() as connection:
        cursor = connection.execute(
            "INSERT INTO question_items (document_id,page_id,question_kind,stem_text,"
            "created_at,updated_at) VALUES (?,?,'typical','NON_REAL_TEST_DATA',?,?)",
            (target.id, page.id, "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
        )
        question_id = cursor.lastrowid
    assert page.id != page.page_number and page.page_number == 2
    return database, earlier, target, page, knowledge, question_id


@pytest.mark.parametrize("entry", ["page", "question", "knowledge"])
def test_source_navigation_uses_physical_second_page_and_exact_identity(
    tmp_path, monkeypatch, multipage_source, entry,
):
    database, _, document, page, knowledge, question_id = multipage_source
    monkeypatch.setattr(runtime, "application_database", lambda: database)
    monkeypatch.setattr(runtime, "application_settings", lambda: SimpleNamespace(
        runtime_dir=tmp_path / "runtime", cache_dir=tmp_path / "cache"))
    monkeypatch.setattr(runtime, "application_ai_provider", lambda: None)
    monkeypatch.setattr(workspace, "render_workspace", lambda _: None)
    monkeypatch.setattr(mindmap_ui, "render_mindmap", lambda payload: None)
    routes = []
    monkeypatch.setattr(st, "switch_page", lambda name, *, query_params: routes.append(
        (name, query_params)))
    identifier = {"page": page.id, "question": question_id, "knowledge": knowledge.id}[entry]
    app = AppTest.from_file(str(Path(__file__).parents[1] / "pages/20_知识串联.py"))
    app.session_state["graph_node_picker"] = f"{entry}:{identifier}"
    app.run()
    assert not app.exception
    if entry == "knowledge":
        graph = KnowledgeGraphService(database).snapshot()
        edges = [edge for edge in graph.links
                 if f"knowledge:{knowledge.id}" in (edge["source"], edge["target"])]
        index = next(i for i, edge in enumerate(edges) if edge["target"] == f"page:{page.id}")
        app.button(key=f"graph_edge_{index}").click().run()
        assert app.selectbox(key="graph_node_picker").value == f"page:{page.id}"
    key = "graph_question_source" if entry == "question" else "graph_page_source"
    app.button(key=key).click().run()
    assert not app.exception
    assert routes[-1] == ("pages/17_我的资料.py", {
        "document": str(document.id), "page": "2", "page_id": str(page.id), "tab": "阅读",
    })


@pytest.mark.parametrize("missing", [False, True])
def test_historical_source_missing_or_cross_document_reference_refuses_jump(
    tmp_path, monkeypatch, multipage_source, missing,
):
    database, earlier, _, page, _, question_id = multipage_source
    with database._connection() as connection:
        connection.execute("UPDATE question_items SET document_id=?,page_id=? WHERE id=?",
                           (earlier.id, None if missing else page.id, question_id))
    monkeypatch.setattr(runtime, "application_database", lambda: database)
    monkeypatch.setattr(runtime, "application_settings", lambda: SimpleNamespace(
        runtime_dir=tmp_path / "runtime", cache_dir=tmp_path / "cache"))
    monkeypatch.setattr(runtime, "application_ai_provider", lambda: None)
    monkeypatch.setattr(workspace, "render_workspace", lambda _: None)
    monkeypatch.setattr(mindmap_ui, "render_mindmap", lambda payload: None)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "pages/20_知识串联.py"))
    app.session_state["graph_node_picker"] = f"question:{question_id}"
    app.run()
    assert not app.exception
    assert all(button.key != "graph_question_source" for button in app.button)
    assert any("引用不一致" in warning.value for warning in app.warning)
