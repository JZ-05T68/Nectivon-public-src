"""Residue cleanup for historical deletions (round-1 P1/P2, 系统维护 page).

Simulates the pre-fix state on the live instance: documents already gone,
but source-derived knowledge objects with dangling/zero sources, their
relations, their FTS rows and page_visual_index rows still present. The
cleanup must be reachable through service code only (no manual SQL, no
manual file deletion), preview-first, confirmation-gated, snapshot-logged,
idempotent, and strictly limited to zero-live-source records.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.database import Database
from src.document_deletion_service import (
    ResidueCleanupError,
    ResidueCleanupService,
)
from src.figure_index_service import read_index, update_page_figures
from src.knowledge_object_service import KnowledgeObjectService


@pytest.fixture()
def residue_workspace(tmp_path: Path) -> tuple[Database, Path, dict[str, int]]:
    """Orphaned knowledge residue plus one healthy surviving document."""

    database = Database(tmp_path / "data" / "database" / "knowledge.db")
    (tmp_path / "data" / "raw").mkdir(parents=True)
    ids: dict[str, int] = {}

    def make_document(name: str, title: str) -> tuple[int, int]:
        raw = tmp_path / "data" / "raw" / f"{name}.pdf"
        raw.write_bytes(f"%PDF-1.7 {name}".encode())
        document = database.create_document(
            title=title,
            filename=f"{name}.pdf",
            source_path=raw,
            sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
            page_count=1,
            import_status="completed",
        )
        image = tmp_path / "data" / "pages" / str(document.id) / "page-1.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"png")
        page = database.create_page(
            document_id=document.id, page_number=1, image_path=image,
            extracted_text="页面文本层", status="ready"
        )
        return document.id, page.id

    ids["dead_doc"], ids["dead_page"] = make_document("dead", "已删除的测试资料")
    ids["live_doc"], ids["live_page"] = make_document("live", "仍在库中的课件")

    knowledge = KnowledgeObjectService(database)
    orphan = knowledge.create(
        kind="concept", title="行政处罚", content="来自已删除测试资料的归纳。",
        epistemic_basis="source_derived",
        source_links=[("page", ids["dead_page"], "AI 归纳")],
    )
    partner = knowledge.create(
        kind="concept", title="行政强制", content="同一批归纳。",
        epistemic_basis="source_derived",
        source_links=[("page", ids["dead_page"], "AI 归纳")],
    )
    knowledge.add_relation(
        orphan.knowledge_object.id,
        partner.knowledge_object.id,
        relation_type="relates_to",
        description="同批残留",
    )
    healthy = knowledge.create(
        kind="concept", title="液压回路", content="来自仍在库课件的归纳。",
        epistemic_basis="source_derived",
        source_links=[("page", ids["live_page"], "AI 归纳")],
    )
    personal = knowledge.create(
        kind="experience", title="个人心得（挂在已删页面上）", content="我自己的总结。",
        epistemic_basis="personal_experience",
        source_links=[("page", ids["dead_page"], "手工关联")],
    )
    ids["ko_orphan"] = orphan.knowledge_object.id
    ids["ko_partner"] = partner.knowledge_object.id
    ids["ko_healthy"] = healthy.knowledge_object.id
    ids["ko_personal"] = personal.knowledge_object.id

    with database._connection() as connection:  # noqa: SLF001
        connection.execute(
            "INSERT INTO page_visual_index(visual_text, page_id) VALUES (?, ?)",
            ("残留 视觉 索引", ids["dead_page"]),
        )
        connection.execute(
            "INSERT INTO page_visual_index(visual_text, page_id) VALUES (?, ?)",
            ("健康 视觉 索引", ids["live_page"]),
        )
        # Simulate the OLD deletion behaviour: the document row disappears
        # (pages/notes cascade), knowledge_object_sources stay dangling and
        # the zero/one-source knowledge objects stay searchable.
        connection.execute(
            "DELETE FROM documents WHERE id = ?", (ids["dead_doc"],)
        )

    candidates_dir = tmp_path / "data" / "question-candidates"
    update_page_figures(
        candidates_dir,
        document_id=ids["dead_doc"], page_id=ids["dead_page"], page_number=1,
        document_title="已删除的测试资料",
        figures=[{"name": "图6-33", "description": "调速回路",
                  "bbox": [0, 0, 100, 100]}],
    )
    update_page_figures(
        candidates_dir,
        document_id=ids["live_doc"], page_id=ids["live_page"], page_number=1,
        document_title="仍在库中的课件",
        figures=[{"name": "图6-9", "description": "换向回路",
                  "bbox": [0, 0, 100, 100]}],
    )
    ids["candidates_dir"] = 0  # path carried separately via tmp_path
    return database, tmp_path, ids


def _service(database: Database, tmp_path: Path) -> ResidueCleanupService:
    return ResidueCleanupService(
        database=database,
        data_dir=tmp_path / "data",
        question_candidates_dir=tmp_path / "data" / "question-candidates",
    )


def test_preview_inventories_residue_without_writing(
    residue_workspace,
) -> None:
    database, tmp_path, ids = residue_workspace
    preview = _service(database, tmp_path).preview_cleanup()
    assert set(preview.deletable_knowledge_object_ids) == {
        ids["ko_orphan"], ids["ko_partner"],
    }
    assert preview.knowledge_relation_count == 1
    # Personal-basis objects with dead sources are counted, never deleted.
    assert preview.personal_zero_source_count == 1
    assert preview.page_visual_index_orphan_count == 1
    assert preview.stale_figure_index_entry_count == 1


def test_execute_refuses_without_confirmation(residue_workspace) -> None:
    database, tmp_path, ids = residue_workspace
    with pytest.raises(ResidueCleanupError, match="未确认"):
        _service(database, tmp_path).execute_cleanup(
            confirmed=False,
            expected_knowledge_object_ids=[ids["ko_orphan"], ids["ko_partner"]],
            expected_visual_index_row_count=1,
        )
    with database._connection() as connection:  # noqa: SLF001
        assert connection.execute(
            "SELECT COUNT(*) FROM knowledge_objects"
        ).fetchone()[0] == 4


def test_execute_refuses_stale_preview(residue_workspace) -> None:
    database, tmp_path, ids = residue_workspace
    with pytest.raises(ResidueCleanupError, match="预览已过期"):
        _service(database, tmp_path).execute_cleanup(
            confirmed=True,
            expected_knowledge_object_ids=[ids["ko_orphan"]],
            expected_visual_index_row_count=1,
        )
    with pytest.raises(ResidueCleanupError, match="视觉索引"):
        _service(database, tmp_path).execute_cleanup(
            confirmed=True,
            expected_knowledge_object_ids=[ids["ko_orphan"], ids["ko_partner"]],
            expected_visual_index_row_count=99,
        )


def test_execute_removes_residue_with_snapshot_and_is_idempotent(
    residue_workspace,
) -> None:
    database, tmp_path, ids = residue_workspace
    service = _service(database, tmp_path)
    preview = service.preview_cleanup()
    result = service.execute_cleanup(
        confirmed=True,
        expected_knowledge_object_ids=preview.deletable_knowledge_object_ids,
        expected_visual_index_row_count=preview.page_visual_index_orphan_count,
    )
    assert result.deleted_knowledge_object_count == 2
    assert result.deleted_knowledge_relation_count == 1
    assert result.deleted_visual_index_row_count == 1
    assert result.pruned_figure_index_entry_count == 1
    assert result.snapshot_path is not None and result.snapshot_path.is_file()
    snapshot = json.loads(result.snapshot_path.read_text(encoding="utf-8"))
    assert {row["id"] for row in snapshot["knowledge_objects"]} == {
        ids["ko_orphan"], ids["ko_partner"],
    }
    assert len(snapshot["knowledge_relations"]) == 1

    with database._connection() as connection:  # noqa: SLF001
        assert connection.execute(
            "SELECT COUNT(*) FROM knowledge_objects"
        ).fetchone()[0] == 2  # healthy + personal survive
        assert connection.execute(
            "SELECT COUNT(*) FROM knowledge_objects WHERE id IN (?, ?)",
            (ids["ko_orphan"], ids["ko_partner"]),
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM knowledge_relations"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM knowledge_object_search WHERE rowid IN (?, ?)",
            (ids["ko_orphan"], ids["ko_partner"]),
        ).fetchone()[0] == 0
        # Healthy FTS rows and the healthy visual index row are untouched.
        assert connection.execute(
            "SELECT COUNT(*) FROM knowledge_object_search WHERE rowid = ?",
            (ids["ko_healthy"],),
        ).fetchone()[0] == 1
        visual_rows = connection.execute(
            "SELECT page_id FROM page_visual_index"
        ).fetchall()
    assert [int(row["page_id"]) for row in visual_rows] == [ids["live_page"]]

    index = read_index(tmp_path / "data" / "question-candidates")
    assert index.get("图6-9") and not index.get("图6-33")
    before_copy = result.snapshot_path.parent / "figures-index.before.json"
    assert before_copy.is_file()

    # Idempotent: a second pass finds nothing and changes nothing.
    second_preview = service.preview_cleanup()
    assert second_preview.deletable_knowledge_object_ids == ()
    assert second_preview.page_visual_index_orphan_count == 0
    assert second_preview.stale_figure_index_entry_count == 0
    second = service.execute_cleanup(
        confirmed=True,
        expected_knowledge_object_ids=[],
        expected_visual_index_row_count=0,
    )
    assert second.deleted_knowledge_object_count == 0
    assert second.snapshot_path is None


def test_supersession_successor_orphan_is_reported_not_deleted(
    residue_workspace,
) -> None:
    database, tmp_path, ids = residue_workspace
    knowledge = KnowledgeObjectService(database)
    successor = knowledge.create(
        kind="concept", title="替代后的归纳", content="仍被旧对象引用。",
        epistemic_basis="source_derived",
        source_links=[("page", ids["live_page"], "AI 归纳")],
    )
    predecessor = knowledge.create(
        kind="concept", title="旧归纳", content="将被替代。",
        epistemic_basis="source_derived",
        source_links=[("page", ids["live_page"], "AI 归纳")],
    )
    knowledge.supersede(
        predecessor.knowledge_object.id, successor.knowledge_object.id
    )
    with database._connection() as connection:  # noqa: SLF001
        # Simulate the old cascade: sources removed, object left behind.
        connection.execute(
            "DELETE FROM knowledge_object_sources WHERE knowledge_object_id = ?",
            (successor.knowledge_object.id,),
        )
    service = _service(database, tmp_path)
    preview = service.preview_cleanup()
    protected = preview.protected_knowledge_objects
    assert [item.knowledge_object_id for item in protected] == [
        successor.knowledge_object.id
    ]
    result = service.execute_cleanup(
        confirmed=True,
        expected_knowledge_object_ids=preview.deletable_knowledge_object_ids,
        expected_visual_index_row_count=preview.page_visual_index_orphan_count,
    )
    assert result.protected_knowledge_object_ids == (
        successor.knowledge_object.id,
    )
    with database._connection() as connection:  # noqa: SLF001
        assert connection.execute(
            "SELECT COUNT(*) FROM knowledge_objects WHERE id = ?",
            (successor.knowledge_object.id,),
        ).fetchone()[0] == 1
