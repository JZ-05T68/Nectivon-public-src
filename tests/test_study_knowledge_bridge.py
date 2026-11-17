"""Saved-review cross-store import and failure isolation, without real models or data."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace

import pytest
from test_ai_review_support import RecordingProvider, review_case

from src.database import Database
from src.knowledge_graph_service import KnowledgeGraphService
from src.knowledge_mining_service import KnowledgeMiningAIService, KnowledgeMiningError
from src.knowledge_object_service import KnowledgeObjectService
from src.models import KnowledgeRelationType
from src.study_knowledge_bridge import StudyKnowledgeBridge, knowledge_review_refs
from src.study_loop.ai_review import AIReviewService
from src.study_loop.models import StudyError
from src.study_loop.storage import StudyStore

CONTENT = "NON_REAL_REVIEW 反馈减小误差。稳定性是控制系统的要求。"


def payload() -> dict:
    return {"knowledge_points": [
        {"title": "NON_REAL反馈", "content": "反馈减小误差。", "origin": "explicit",
         "evidence": "反馈减小误差", "importance": "primary", "pages": [999]},
        {"title": "NON_REAL推导", "content": "这是推导候选。", "origin": "inferred",
         "evidence": "稳定性是控制系统的要求"},
    ], "relations": [
        {"from_title": "NON_REAL反馈", "to_title": "NON_REAL推导", "type": "supports",
         "description": "NON_REAL人工核对前的候选依据", "evidence": "反馈减小误差"},
    ]}


@pytest.fixture
def case(tmp_path):
    data = review_case(tmp_path / "NON_REAL_LIBRARY")
    reviews = AIReviewService(data.store)
    document = reviews.generate([data.selected.id], RecordingProvider(CONTENT))
    version = reviews.save_version(document["id"])
    database = Database(tmp_path / "NON_REAL_PLATFORM" / "knowledge.sqlite3")
    return data, reviews, document, version, database, StudyKnowledgeBridge(database, data.store)


def mine(bridge, version, value=None):
    provider = RecordingProvider(json.dumps(value or payload(), ensure_ascii=False))
    draft = bridge.mine_saved_review(version["id"], KnowledgeMiningAIService(provider))
    return draft, provider


def test_saved_revision_only_candidates_and_precise_source_identity(case):
    data, reviews, document, version, database, bridge = case
    reviews.update_working(document["id"], "UNSAVED_EDITOR_MUST_NOT_BE_READ")
    original = data.store.db.read_bytes()
    draft, provider = mine(bridge, version)
    assert CONTENT in provider.prompts[0]
    assert "UNSAVED_EDITOR_MUST_NOT_BE_READ" not in provider.prompts[0]
    assert len(provider.prompts) == 1
    assert '"origin":"explicit"' in provider.prompts[0]
    assert '"evidence":"从下文复盘逐字复制的连续短句"' in provider.prompts[0]
    assert draft.source["study_review_id"] == document["id"]
    assert draft.source["study_review_revision_id"] == version["id"]
    assert draft.source["course_id"] == "NON_REAL_COURSE_A"
    assert draft.source["questions"] == [{
        "item_id": data.selected.id, "question_id": data.selected.question.id,
        "material_id": data.source.id,
    }]
    assert draft.source["source_page_refs"][0]["study_page_id"] == "NON_REAL_SHARED_PAGE"
    assert draft.mining.points[0].page_ids == ()  # Model cannot fabricate platform pages.
    assert KnowledgeObjectService(database).count() == 0
    assert not bridge.imports_for_review(document["id"])
    assert knowledge_review_refs(database, 1) == []
    with database._connection() as connection:
        assert not connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name='study_knowledge_imports'"
        ).fetchone()
    assert data.store.db.read_bytes() == original


def test_confirmation_is_atomic_idempotent_and_shared_graph_source(case):
    data, reviews, document, version, database, bridge = case
    draft, _ = mine(bridge, version)
    with pytest.raises(StudyError, match="确认"):
        bridge.confirm(draft, user_confirmed=False)
    saved = bridge.confirm(draft, user_confirmed=True)
    again = bridge.confirm(draft, user_confirmed=True)
    assert saved.knowledge_ids == again.knowledge_ids
    assert len(saved.knowledge_ids) == 2 and saved.relation_count == 1
    assert KnowledgeObjectService(database).count() == 2
    with pytest.raises(StudyError, match="已经导入"):
        mine(bridge, version)
    ref = knowledge_review_refs(database, saved.knowledge_ids[1])[0]
    assert ref["origin"] == "inferred"
    assert ref["materials"][0]["sha256"] == data.source.sha256
    obj = KnowledgeObjectService(database).get(saved.knowledge_ids[1])
    assert obj.epistemic_basis.value == "personal_judgment"
    assert obj.confirmation_is_current
    graph = KnowledgeGraphService(database).snapshot()
    assert all(f"knowledge:{identity}" in graph.nodes for identity in saved.knowledge_ids)
    assert any(edge["label"] == "支持" for edge in graph.links)
    assert reviews.version(version["id"])["content"] == CONTENT


def test_new_revision_requires_explicit_bindings_and_preserves_edited_knowledge(case):
    _, reviews, document, version, database, bridge = case
    first, _ = mine(bridge, version)
    saved = bridge.confirm(first, user_confirmed=True)
    service = KnowledgeObjectService(database)
    service.update_content(saved.knowledge_ids[0], content="NON_REAL_USER_EDIT")
    for relation in service.relations(saved.knowledge_ids[0]):
        service.remove_relation(relation.id)
    manual = service.add_relation(*saved.knowledge_ids, relation_type="supports",
                                  description="NON_REAL_USER_RELATION_NOTE")
    reviews.update_working(document["id"], CONTENT + "\n新增的复盘版本。")
    next_version = reviews.save_version(document["id"])
    second, _ = mine(bridge, next_version)
    with pytest.raises(StudyError, match="逐项"):
        bridge.confirm(second, user_confirmed=True)
    bindings = {point.title: identity for point, identity
                in zip(second.mining.points, saved.knowledge_ids, strict=True)}
    second_saved = bridge.confirm(second, user_confirmed=True, bindings=bindings)
    assert second_saved.knowledge_ids == saved.knowledge_ids
    assert second_saved.relation_count == 0
    assert service.count() == 2
    assert service.get(saved.knowledge_ids[0]).content == "NON_REAL_USER_EDIT"
    assert database.get_knowledge_relation(manual.id).description == "NON_REAL_USER_RELATION_NOTE"
    assert len(knowledge_review_refs(database, saved.knowledge_ids[0])) == 2
    assert len(bridge.imports_for_review(document["id"])) == 2
    assert reviews.version(version["id"])["content"] == CONTENT


def test_failed_relation_rolls_back_all_knowledge_and_lineage(case):
    _, reviews, _, version, database, bridge = case
    draft, _ = mine(bridge, version)
    broken_relation = replace(draft.mining.relations[0], description="")
    broken = replace(draft, mining=replace(draft.mining, relations=(broken_relation,)))
    with pytest.raises(StudyError, match="缺少依据"):
        bridge.confirm(broken, user_confirmed=True)
    assert KnowledgeObjectService(database).count() == 0
    assert knowledge_review_refs(database, 1) == []
    assert reviews.version(version["id"])["content"] == CONTENT


def test_unsaved_version_missing_source_and_invented_quotes_are_rejected(case):
    data, reviews, document, version, database, bridge = case
    with pytest.raises(StudyError, match="不存在"):
        bridge.mine_saved_review(document["id"], KnowledgeMiningAIService(RecordingProvider()))
    bad = payload()
    for point in bad["knowledge_points"]:
        point["evidence"] = "NON_REAL_INVENTED_QUOTE"
    with pytest.raises(KnowledgeMiningError, match="引文"):
        mine(bridge, version, bad)
    data.store.delete_source(data.source.id, user_confirmed=True)
    with pytest.raises(StudyError):
        mine(bridge, version)
    assert KnowledgeObjectService(database).count() == 0
    assert reviews.version(version["id"])["content"] == CONTENT


def test_reimport_preserves_manual_relationships(case):
    _, _, _, version, database, bridge = case
    draft, _ = mine(bridge, version)
    saved = bridge.confirm(draft, user_confirmed=True)
    service = KnowledgeObjectService(database)
    manual = service.add_relation(*reversed(saved.knowledge_ids),
                                  relation_type=KnowledgeRelationType.REQUIRES,
                                  description="NON_REAL_USER_RELATION")
    bridge.confirm(draft, user_confirmed=True)
    assert database.get_knowledge_relation(manual.id).description == "NON_REAL_USER_RELATION"


def test_independent_library_namespace_prevents_revision_id_collision(case, tmp_path):
    data, _, document, version, database, bridge = case
    first, _ = mine(bridge, version)
    saved = bridge.confirm(first, user_confirmed=True)
    copied_root = tmp_path / "NON_REAL_INDEPENDENT_COPY"
    shutil.copytree(data.store.root, copied_root)
    independent = StudyKnowledgeBridge(database, StudyStore(copied_root))
    assert independent.library_id != bridge.library_id
    assert independent.imports_for_review(document["id"]) == []
    second, _ = mine(independent, version)
    second_saved = independent.confirm(second, user_confirmed=True)
    assert not set(second_saved.knowledge_ids).intersection(saved.knowledge_ids)
    assert len(bridge.imports_for_review(document["id"])) == 1
    assert len(independent.imports_for_review(document["id"])) == 1
    assert knowledge_review_refs(database, saved.knowledge_ids[0])[0]["study_root"] == str(
        data.store.root
    )
    assert knowledge_review_refs(database, second_saved.knowledge_ids[0])[0]["study_root"] == str(
        copied_root
    )


def test_lineage_does_not_prevent_existing_knowledge_deletion(case):
    _, _, document, version, database, bridge = case
    draft, _ = mine(bridge, version)
    saved = bridge.confirm(draft, user_confirmed=True)
    with database.knowledge_transaction() as connection:
        connection.execute("DELETE FROM knowledge_relations")
        connection.execute("DELETE FROM knowledge_objects WHERE id=?", (saved.knowledge_ids[0],))
    assert KnowledgeObjectService(database).get(saved.knowledge_ids[0]) is None
    imported = bridge.imports_for_review(document["id"])
    assert imported[0]["knowledge_ids"] == list(saved.knowledge_ids)


def test_node_source_panel_reads_original_question_snapshot_and_saved_revision(case, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    data, reviews, document, version, database, bridge = case
    draft, _ = mine(bridge, version)
    saved = bridge.confirm(draft, user_confirmed=True)
    reviews.update_working(document["id"], "NON_REAL_UNSAVED_TEXT_NOT_A_SOURCE")
    checked = []

    def source_panel(store, source_id, page_ids, *, prefix):
        checked.append((str(store.root), source_id, page_ids))
        st.caption("NON_REAL_VERIFIED_SOURCE_PANEL")

    monkeypatch.setattr("src.study_loop.import_ui.render_source", source_panel)
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from src.database import Database\n"
        "from src.study_loop.knowledge_import_ui import render_study_review_sources\n"
        "render_study_review_sources(Database(st.session_state['test_database']), "
        "st.session_state['test_knowledge_id'])\n"
    )
    app.session_state["test_database"] = str(database.database_path)
    app.session_state["test_knowledge_id"] = saved.knowledge_ids[0]
    app.run(timeout=30)
    assert not app.exception
    assert any(data.selected.question.content in item.value for item in app.markdown)
    assert not any("NON_REAL_UNSAVED_TEXT_NOT_A_SOURCE" in item.value for item in app.markdown)
    assert checked == [(str(data.store.root), data.source.id, data.selected.page_ids)]
    assert any("返回此版复盘与原题" == item.label for item in app.button)


def test_persisted_preview_confirmation_requires_action_and_keeps_only_selected_points(
    case, tmp_path, monkeypatch,
):
    from streamlit.testing.v1 import AppTest

    from src.knowledge_graph_preview import (
        list_previews,
        load_preview,
        preview_identity,
        save_preview,
    )

    _, _, _, version, database, bridge = case
    draft, _ = mine(bridge, version)
    runtime_dir = tmp_path / "NON_REAL_PREVIEW_RUNTIME"
    preview_id = save_preview(database, runtime_dir, draft)

    def forbidden_provider(*args, **kwargs):
        raise AssertionError("Opening or confirming a stored preview must not bind a provider")

    monkeypatch.setattr(
        "src.ai.knowledge_generation.configured_knowledge_generation_provider", forbidden_provider,
    )
    app = AppTest.from_string(
        "from pathlib import Path\n"
        "import streamlit as st\n"
        "from src.database import Database\n"
        "from src.study_loop.knowledge_import_ui import render_graph_review_preview\n"
        "render_graph_review_preview(Database(st.session_state['test_database']), "
        "Path(st.session_state['test_runtime']))\n"
    )
    app.session_state["test_database"] = str(database.database_path)
    app.session_state["test_runtime"] = str(runtime_dir)
    app.run(timeout=30)
    key = f"graph_review_preview_{preview_identity(draft)}_"
    assert not app.exception
    assert app.button(key=key + "save").disabled
    assert KnowledgeObjectService(database).count() == 0
    app.checkbox(key=key + "point_1").uncheck().run(timeout=30)
    app.checkbox(key=key + "confirm").check().run(timeout=30)
    assert KnowledgeObjectService(database).count() == 0
    app.button(key=key + "save").click().run(timeout=30)
    assert not app.exception
    assert KnowledgeObjectService(database).count() == 1
    assert load_preview(database, runtime_dir) is None
    assert load_preview(database, runtime_dir, preview_id) == draft
    assert len(list_previews(database, runtime_dir)) == 1
    imports = bridge.imports_for_review(draft.source["study_review_id"])
    assert len(imports[0]["knowledge_ids"]) == 1
    assert database.list_knowledge_relations(imports[0]["knowledge_ids"][0]) == []


def test_persisted_preview_source_change_does_not_clear_or_confirm_candidate(
    case, tmp_path,
):
    from streamlit.testing.v1 import AppTest

    from src.knowledge_graph_preview import load_preview, save_preview

    data, _, _, version, database, bridge = case
    draft, _ = mine(bridge, version)
    runtime_dir = tmp_path / "NON_REAL_PREVIEW_RUNTIME"
    save_preview(database, runtime_dir, draft)
    data.store.delete_source(data.source.id, user_confirmed=True)
    app = AppTest.from_string(
        "from pathlib import Path\n"
        "import streamlit as st\n"
        "from src.database import Database\n"
        "from src.study_loop.knowledge_import_ui import render_graph_review_preview\n"
        "render_graph_review_preview(Database(st.session_state['test_database']), "
        "Path(st.session_state['test_runtime']))\n"
    )
    app.session_state["test_database"] = str(database.database_path)
    app.session_state["test_runtime"] = str(runtime_dir)
    app.run(timeout=30)
    assert not app.exception
    assert any("暂不能确认" in warning.value for warning in app.warning)
    assert KnowledgeObjectService(database).count() == 0
    assert load_preview(database, runtime_dir) == draft


def _preview_confirmation_app(database, runtime_dir):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string(
        "from pathlib import Path\n"
        "import streamlit as st\n"
        "from src.database import Database\n"
        "from src.study_loop.knowledge_import_ui import render_graph_review_preview\n"
        "render_graph_review_preview(Database(st.session_state['test_database']), "
        "Path(st.session_state['test_runtime']))\n"
    )
    app.session_state["test_database"] = str(database.database_path)
    app.session_state["test_runtime"] = str(runtime_dir)
    return app.run(timeout=30)


def _changed_candidate(draft):
    point = replace(draft.mining.points[0], content="NON_REAL_DIFFERENT_CANDIDATE_CONTENT")
    return replace(draft, mining=replace(draft.mining, points=(point, *draft.mining.points[1:])))


def test_same_revision_replacement_cannot_inherit_candidate_confirmation(case, tmp_path):
    from src.knowledge_graph_preview import preview_identity, save_preview

    _, _, _, version, database, bridge = case
    first, _ = mine(bridge, version)
    runtime_dir = tmp_path / "NON_REAL_PREVIEW_RUNTIME"
    save_preview(database, runtime_dir, first)
    app = _preview_confirmation_app(database, runtime_dir)
    first_key = f"graph_review_preview_{preview_identity(first)}_"
    app.checkbox(key=first_key + "confirm").check().run(timeout=30)
    assert not app.button(key=first_key + "save").disabled
    second = _changed_candidate(first)
    save_preview(database, runtime_dir, second)
    app.run(timeout=30)
    second_key = f"graph_review_preview_{preview_identity(second)}_"
    assert second_key != first_key
    assert not app.exception
    assert app.checkbox(key=second_key + "confirm").value is False
    assert app.button(key=second_key + "save").disabled
    assert KnowledgeObjectService(database).count() == 0


def test_confirming_older_candidate_preserves_new_active_pointer(case, tmp_path, monkeypatch):
    from src.knowledge_graph_preview import load_preview, preview_identity, save_preview

    _, _, _, version, database, bridge = case
    first, _ = mine(bridge, version)
    second = _changed_candidate(first)
    runtime_dir = tmp_path / "NON_REAL_PREVIEW_RUNTIME"
    save_preview(database, runtime_dir, first)
    confirm = StudyKnowledgeBridge.confirm

    def concurrent_publish(self, draft, **kwargs):
        result = confirm(self, draft, **kwargs)
        save_preview(database, runtime_dir, second)
        return result

    monkeypatch.setattr(StudyKnowledgeBridge, "confirm", concurrent_publish)
    app = _preview_confirmation_app(database, runtime_dir)
    key = f"graph_review_preview_{preview_identity(first)}_"
    app.checkbox(key=key + "confirm").check().run(timeout=30)
    app.button(key=key + "save").click().run(timeout=30)
    assert not app.exception
    assert load_preview(database, runtime_dir) == second
    assert KnowledgeObjectService(database).count() == len(first.mining.points)
    assert all(obj.content != "NON_REAL_DIFFERENT_CANDIDATE_CONTENT"
               for obj in KnowledgeObjectService(database).list())
