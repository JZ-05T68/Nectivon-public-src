"""Deletion policy for learning assets (v0.8.6 duty B).

Policy under test: organized questions deliberately SURVIVE document
deletion, detached from the document/page (ON DELETE SET NULL). Current
schema evidence relationships and confirmation state survive while page
provenance is detached; family memberships and mastery records stay intact.
The preview must show this impact to the user before anything happens.
"""

from __future__ import annotations

import hashlib
from contextlib import closing
from pathlib import Path

import pytest

from src.database import Database
from src.document_deletion_service import DocumentDeletionService
from src.models import DocumentAggregationImpact
from tests.historical_rows import seed_question, seed_row


@pytest.fixture()
def organized_workspace(tmp_path: Path) -> tuple[Database, Path]:
    """One document with pages, organized questions, evidence and mastery."""
    database = Database(tmp_path / "data" / "database" / "knowledge.db")
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "pages" / "1").mkdir(parents=True)
    raw = tmp_path / "data" / "raw" / "试卷.pdf"
    raw.write_bytes(b"%PDF-1.7 fixture")
    document = database.create_document(
        title="高三数学周测",
        filename="试卷.pdf",
        source_path=raw,
        sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        page_count=2,
        import_status="completed",
    )
    page_one = tmp_path / "data" / "pages" / "1" / "page-1.png"
    page_one.write_bytes(b"png1")
    page_two = tmp_path / "data" / "pages" / "1" / "page-2.png"
    page_two.write_bytes(b"png2")
    database.create_page(
        document_id=document.id, page_number=1, image_path=page_one, status="ready"
    )
    database.create_page(
        document_id=document.id, page_number=2, image_path=page_two, status="ready"
    )
    question = seed_question(
        database,
        document_id=document.id,
        page_id=1,
        question_kind="error",
        question_number="7",
        stem_text="设函数 f(x) 求单调区间。",
        stem_confidence="probable",
        teacher_verdict="incorrect",
    )
    same_paper_question = seed_question(
        database,
        document_id=document.id,
        page_id=2,
        question_kind="good",
        question_number="12",
        stem_text="证明数列单调有界。",
    )
    # Seed retained historical rows explicitly; exercise the CURRENT deleter.
    group_id = seed_row(
        database,
        "question_groups",
        document_id=document.id,
        group_number="7",
        source_pages=[1],
        status="user_confirmed",
    )
    with database._connection() as connection:
        connection.execute(
            "UPDATE question_items SET group_id=? WHERE id=?", (group_id, question.id)
        )
    material_id = seed_row(
        database,
        "question_group_materials",
        group_id=group_id,
        material_kind="figure",
        material_label="函数图像",
        page_id=1,
    )
    seed_row(
        database,
        "question_item_evidence",
        question_item_id=question.id,
        evidence_type="figure",
        source_id=material_id,
        page_id=1,
        status="user_confirmed",
        confidence="confirmed",
    )
    family_id = seed_row(
        database,
        "question_families",
        family_kind="type",
        title="函数与导数：单调区间",
        description="先定义域后求导。",
    )
    for item in (question, same_paper_question):
        seed_row(database, "question_family_members", question_id=item.id, family_id=family_id)
    seed_row(database, "mastery_records", question_id=question.id, outcome="incorrect")
    return database, tmp_path


def _delete_document(database: Database, data_dir: Path, document_id: int) -> None:
    service = DocumentDeletionService(
        database=database,
        raw_dir=data_dir / "data" / "raw",
        pages_dir=data_dir / "data" / "pages",
        markdown_dir=data_dir / "data" / "markdown",
        data_dir=data_dir / "data",
        app_version="0.8.6-test",
    )
    result = service.delete_document(document_id, expected_title="高三数学周测")
    assert result.deleted is True


def test_preview_reports_learning_layer_impact(organized_workspace) -> None:
    database, data_dir = organized_workspace
    service = DocumentDeletionService(
        database=database,
        raw_dir=data_dir / "data" / "raw",
        pages_dir=data_dir / "data" / "pages",
        markdown_dir=data_dir / "data" / "markdown",
        data_dir=data_dir / "data",
        app_version="0.8.6-test",
    )

    preview = service.preview_document_deletion(1)

    assert preview.learning_question_count == 2
    assert preview.learning_family_count == 1
    assert preview.learning_mastery_record_count == 1
    assert preview.learning_evidence_link_count == 1
    assert preview.path_anomalies == ()


def test_delete_survives_questions_and_detaches_evidence_page_provenance(
    organized_workspace,
) -> None:
    database, data_dir = organized_workspace
    _delete_document(database, data_dir, 1)

    with closing(sqlite3_connection(database)) as connection:
        questions = connection.execute(
            "SELECT id, document_id, page_id, stem_text FROM question_items ORDER BY id"
        ).fetchall()
        evidence_links = connection.execute(
            "SELECT question_item_id, source_id, page_id, status, confidence "
            "FROM question_item_evidence ORDER BY id"
        ).fetchall()
        materials = connection.execute(
            "SELECT id, page_id FROM question_group_materials ORDER BY id"
        ).fetchall()
        members = connection.execute("SELECT COUNT(*) FROM question_family_members").fetchone()[0]
        mastery = connection.execute("SELECT COUNT(*) FROM mastery_records").fetchone()[0]
        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    assert len(questions) == 2
    assert all(row["document_id"] is None and row["page_id"] is None for row in questions)
    assert questions[0]["stem_text"] == "设函数 f(x) 求单调区间。"
    assert evidence_links[0]["question_item_id"] == questions[0]["id"]
    assert evidence_links[0]["status"] == "user_confirmed"
    assert evidence_links[0]["confidence"] == "confirmed"
    assert evidence_links[0]["page_id"] is None
    assert evidence_links[0]["source_id"] == materials[0]["id"]
    assert materials[0]["page_id"] is None
    assert members == 2
    assert mastery == 1
    assert violations == []

    # 学习层 FTS 仍然可检索（幸存内容不被静默移除）
    with closing(sqlite3_connection(database)) as connection:
        hits = connection.execute(
            "SELECT rowid FROM question_search WHERE question_search MATCH '单调'"
        ).fetchall()
    assert hits


def sqlite3_connection(database: Database):
    import sqlite3

    connection = sqlite3.connect(database.database_path)
    connection.row_factory = sqlite3.Row
    return connection


def test_delete_document_without_learning_assets_reports_zero_impact(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "data" / "database" / "knowledge.db")
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "pages" / "1").mkdir(parents=True)
    raw = tmp_path / "data" / "raw" / "plain.pdf"
    raw.write_bytes(b"%PDF-1.7 plain")
    document = database.create_document(
        title="普通文档",
        filename="plain.pdf",
        source_path=raw,
        sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        import_status="completed",
    )
    image = tmp_path / "data" / "pages" / "1" / "page-1.png"
    image.write_bytes(b"png")
    database.create_page(document_id=document.id, page_number=1, image_path=image, status="pending")
    service = DocumentDeletionService(
        database=database,
        raw_dir=tmp_path / "data" / "raw",
        pages_dir=tmp_path / "data" / "pages",
        markdown_dir=tmp_path / "data" / "markdown",
        data_dir=tmp_path / "data",
        app_version="0.8.6-test",
    )

    preview = service.preview_document_deletion(document.id)

    assert preview.learning_question_count == 0
    assert preview.learning_family_count == 0
    assert preview.learning_mastery_record_count == 0
    assert preview.learning_evidence_link_count == 0
    assert preview.aggregation_impact == DocumentAggregationImpact()
    result = service.delete_document(document.id, expected_title="普通文档")
    assert result.deleted is True


@pytest.fixture()
def database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "data" / "database" / "knowledge.db")
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "pages" / "1").mkdir(parents=True)
    raw = tmp_path / "data" / "raw" / "试卷.pdf"
    raw.write_bytes(b"%PDF-1.7 fixture")
    image = tmp_path / "data" / "pages" / "1" / "page-1.png"
    image.write_bytes(b"png")
    document = database.create_document(
        title="高三数学周测",
        filename="试卷.pdf",
        source_path=raw,
        sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        page_count=1,
        import_status="completed",
    )
    database.create_page(document_id=document.id, page_number=1, image_path=image, status="ready")
    return database


def test_current_confirmation_requires_existing_page_in_same_source(tmp_path: Path) -> None:
    from copy import deepcopy

    from src.study_loop.components import import_source
    from src.study_loop.models import StudyError
    from tests.study_migration_support import text_case

    store, source, draft = text_case(tmp_path / "study")
    other = store.receive("OTHER.txt", b"NON_REAL_TEST_DATA OTHER", "txt")
    import_source(store, other.id)
    for page_id in ("MISSING_PAGE", store.pages(other.id)[0].id):
        changed = deepcopy(draft)
        changed.page_ids = [page_id]
        with pytest.raises(StudyError, match="不属于"):
            store.confirm(changed, user_confirmed=True)
        assert not store.items()
        assert not store.attempts()
        assert next(d for d in store.drafts() if d.id == draft.id).to_dict() == draft.to_dict()
    store.confirm(draft, user_confirmed=True)
    assert store.item(draft.id).source_id == source.id
    assert store.item(draft.id).page_ids == draft.page_ids
