"""Deletion cascade for source-derived knowledge objects (round-1 P1).

Policy under test: deleting a document removes knowledge objects that lose
their LAST source with it — together with their knowledge_relations, their
knowledge_object_search FTS rows and their question links — while
personal-basis knowledge objects, manually created zero-source objects and
every genuine learning asset (questions, families, mastery records) survive
detached exactly as before. page_visual_index rows of the deleted pages are
removed in the same transaction.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.database import Database
from src.document_deletion_service import DocumentDeletionService
from src.knowledge_object_service import KnowledgeObjectService
from tests.historical_rows import seed_question


@pytest.fixture()
def workspace(tmp_path: Path) -> tuple[Database, Path, dict[str, int]]:
    """Two documents; knowledge objects, relations, FTS and visual index rows."""

    database = Database(tmp_path / "data" / "database" / "knowledge.db")
    (tmp_path / "data" / "raw").mkdir(parents=True)
    ids: dict[str, int] = {}
    for name, title in (("doc1", "液压与气动课件"), ("doc2", "另一份课件")):
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
        ids[f"{name}_id"] = document.id
        ids[f"{name}_page_id"] = page.id

    knowledge = KnowledgeObjectService(database)
    mined = knowledge.create(
        kind="concept", title="行政处罚的种类", content="警告、罚款、没收违法所得等。",
        epistemic_basis="source_derived",
        source_links=[("page", ids["doc1_page_id"], "AI 归纳")],
    )
    mined_other = knowledge.create(
        kind="concept", title="液压基本回路", content="调压、调速、换向回路。",
        epistemic_basis="source_derived",
        source_links=[("page", ids["doc2_page_id"], "AI 归纳")],
    )
    personal = knowledge.create(
        kind="experience", title="我自己的复盘心得", content="先画回路图再列方程。",
        epistemic_basis="personal_experience",
        source_links=[("page", ids["doc1_page_id"], "手工关联")],
    )
    manual_zero = knowledge.create(
        kind="concept", title="手抄概念（从未挂来源）", content="随手记录。",
        epistemic_basis="source_derived",
    )
    knowledge.add_relation(
        mined.knowledge_object.id,
        mined_other.knowledge_object.id,
        relation_type="relates_to",
        description="跨课件关联",
    )
    ids["ko_mined"] = mined.knowledge_object.id
    ids["ko_mined_other"] = mined_other.knowledge_object.id
    ids["ko_personal"] = personal.knowledge_object.id
    ids["ko_manual_zero"] = manual_zero.knowledge_object.id

    question = seed_question(database,
        document_id=ids["doc1_id"],
        page_id=ids["doc1_page_id"],
        question_kind="error",
        question_number="6-2",
        stem_text="画出调速回路并说明原理。",
        stem_confidence="probable",
    )
    ids["question_id"] = question.id
    with database._connection() as connection:  # noqa: SLF001
        connection.execute(
            "INSERT INTO question_knowledge_links "
            "(question_id, knowledge_object_id, note, created_at) "
            "VALUES (?, ?, '', '2026-11-04T00:00:00+00:00')",
            (question.id, ids["ko_mined"]),
        )
        for page_id in (ids["doc1_page_id"], ids["doc2_page_id"]):
            connection.execute(
                "INSERT INTO page_visual_index(visual_text, page_id) VALUES (?, ?)",
                ("液压 回路 调速", page_id),
            )
    return database, tmp_path, ids


def _service(database: Database, data_dir: Path) -> DocumentDeletionService:
    return DocumentDeletionService(
        database=database,
        raw_dir=data_dir / "data" / "raw",
        pages_dir=data_dir / "data" / "pages",
        markdown_dir=data_dir / "data" / "markdown",
        data_dir=data_dir / "data",
        app_version="0.8.8-test",
    )


def _counts(database: Database, ids: dict[str, int]) -> dict[str, int]:
    with database._connection() as connection:  # noqa: SLF001
        return {
            "ko_rows": connection.execute(
                "SELECT COUNT(*) FROM knowledge_objects"
            ).fetchone()[0],
            "mined": connection.execute(
                "SELECT COUNT(*) FROM knowledge_objects WHERE id = ?",
                (ids["ko_mined"],),
            ).fetchone()[0],
            "mined_other": connection.execute(
                "SELECT COUNT(*) FROM knowledge_objects WHERE id = ?",
                (ids["ko_mined_other"],),
            ).fetchone()[0],
            "personal": connection.execute(
                "SELECT COUNT(*) FROM knowledge_objects WHERE id = ?",
                (ids["ko_personal"],),
            ).fetchone()[0],
            "manual_zero": connection.execute(
                "SELECT COUNT(*) FROM knowledge_objects WHERE id = ?",
                (ids["ko_manual_zero"],),
            ).fetchone()[0],
            "relations": connection.execute(
                "SELECT COUNT(*) FROM knowledge_relations"
            ).fetchone()[0],
            "fts_mined": connection.execute(
                "SELECT COUNT(*) FROM knowledge_object_search WHERE rowid = ?",
                (ids["ko_mined"],),
            ).fetchone()[0],
            "fts_mined_other": connection.execute(
                "SELECT COUNT(*) FROM knowledge_object_search WHERE rowid = ?",
                (ids["ko_mined_other"],),
            ).fetchone()[0],
            "question_links": connection.execute(
                "SELECT COUNT(*) FROM question_knowledge_links"
            ).fetchone()[0],
            "questions": connection.execute(
                "SELECT COUNT(*) FROM question_items"
            ).fetchone()[0],
            "personal_sources": connection.execute(
                "SELECT COUNT(*) FROM knowledge_object_sources "
                "WHERE knowledge_object_id = ?",
                (ids["ko_personal"],),
            ).fetchone()[0],
            "visual_index": connection.execute(
                "SELECT COUNT(*) FROM page_visual_index"
            ).fetchone()[0],
        }


def test_preview_counts_orphan_knowledge_objects_and_visual_index(
    workspace,
) -> None:
    database, data_dir, ids = workspace
    preview = _service(database, data_dir).preview_document_deletion(ids["doc1_id"])
    # ko_mined loses its only source; personal/manual-zero/other-doc stay.
    assert preview.knowledge_object_orphan_count == 1
    assert preview.page_visual_index_count == 1
    assert preview.path_anomalies == ()


def test_delete_cascades_orphan_ko_relations_fts_and_visual_index(
    workspace,
) -> None:
    database, data_dir, ids = workspace
    before = _counts(database, ids)
    assert before["mined"] == 1 and before["fts_mined"] == 1

    result = _service(database, data_dir).delete_document(
        ids["doc1_id"], expected_title="液压与气动课件"
    )
    assert result.deleted is True

    after = _counts(database, ids)
    # Orphaned source-derived object, its relation, its FTS row and its
    # question link are gone; the visual index row of the deleted page too.
    assert after["mined"] == 0
    assert after["relations"] == 0
    assert after["fts_mined"] == 0
    assert after["question_links"] == 0
    assert after["visual_index"] == 1  # only doc2's row survives
    # Genuine learning assets and non-source-derived knowledge survive.
    assert after["questions"] == 1
    assert after["personal"] == 1
    assert after["personal_sources"] == 0  # detached, not deleted
    assert after["manual_zero"] == 1
    assert after["mined_other"] == 1
    assert after["fts_mined_other"] == 1


def test_question_survives_detached_after_orphan_ko_cascade(workspace) -> None:
    database, data_dir, ids = workspace
    _service(database, data_dir).delete_document(
        ids["doc1_id"], expected_title="液压与气动课件"
    )
    with database._connection() as connection:  # noqa: SLF001
        row = connection.execute(
            "SELECT document_id, page_id, stem_text FROM question_items WHERE id = ?",
            (ids["question_id"],),
        ).fetchone()
    assert row is not None
    assert row["document_id"] is None and row["page_id"] is None
    assert row["stem_text"] == "画出调速回路并说明原理。"


def test_cascade_keeps_supersession_successor_and_reports_it(
    workspace, caplog: pytest.LogCaptureFixture
) -> None:
    database, data_dir, ids = workspace
    knowledge = KnowledgeObjectService(database)
    old = knowledge.create(
        kind="concept", title="旧版归纳", content="将被替代。",
        epistemic_basis="source_derived",
        source_links=[("page", ids["doc1_page_id"], "AI 归纳")],
    )
    new = knowledge.create(
        kind="concept", title="新版归纳", content="替代后的内容。",
        epistemic_basis="source_derived",
        source_links=[("page", ids["doc1_page_id"], "AI 归纳")],
    )
    knowledge.supersede(old.knowledge_object.id, new.knowledge_object.id)

    _service(database, data_dir).delete_document(
        ids["doc1_id"], expected_title="液压与气动课件"
    )
    with database._connection() as connection:  # noqa: SLF001
        old_rows = connection.execute(
            "SELECT COUNT(*) FROM knowledge_objects WHERE id = ?",
            (old.knowledge_object.id,),
        ).fetchone()[0]
        new_rows = connection.execute(
            "SELECT COUNT(*) FROM knowledge_objects WHERE id = ?",
            (new.knowledge_object.id,),
        ).fetchone()[0]
    # The predecessor lost its last source and no successor points at it.
    assert old_rows == 0
    # The successor is protected by ON DELETE RESTRICT and stays for manual
    # resolution — the deletion must not fail because of it.
    assert new_rows == 1
    assert any("替代关系" in record.getMessage() for record in caplog.records)
