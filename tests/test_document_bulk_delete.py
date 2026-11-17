"""Overnight-round tests: bulk delete orchestration (§16)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.database import Database
from src.document_bulk_delete_service import (
    DocumentBulkDeleteService,
)
from src.document_deletion_service import DocumentDeletionError, DocumentDeletionService
from tests.historical_rows import read_question, seed_question, seed_row


def _make_service(database: Database, tmp_path: Path) -> DocumentDeletionService:
    return DocumentDeletionService(
        database=database,
        raw_dir=tmp_path / "data" / "raw",
        pages_dir=tmp_path / "data" / "pages",
        markdown_dir=tmp_path / "data" / "markdown",
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "data" / "cache",
        backups_dir=tmp_path / "backups",
    )


@pytest.fixture()
def two_documents(tmp_path: Path) -> tuple[Database, Path, list[int]]:
    db = Database(tmp_path / "data" / "database" / "knowledge.db")
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "pages").mkdir(parents=True)
    (tmp_path / "data" / "markdown").mkdir(parents=True)
    ids: list[int] = []
    for index in (1, 2):
        raw = tmp_path / "data" / "raw" / f"试卷{index}.pdf"
        raw.write_bytes(b"%PDF-1.7 " + bytes([index]))
        image_dir = tmp_path / "data" / "pages" / str(index)
        image_dir.mkdir(parents=True, exist_ok=True)
        image = image_dir / "page-1.png"
        image.write_bytes(b"png")
        document = db.create_document(
            title=f"试卷{index}",
            filename=f"试卷{index}.pdf",
            source_path=raw,
            sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
            page_count=1,
            import_status="completed",
        )
        db.create_page(
            document_id=document.id,
            page_number=1,
            image_path=image,
            extracted_text=f"第{index}题",
            status="ready",
        )
        ids.append(document.id)
    return db, tmp_path, ids


def test_bulk_delete_removes_all_documents_and_keeps_learning_assets(
    two_documents,
) -> None:
    db, tmp_path, ids = two_documents
    service = DocumentBulkDeleteService(_make_service(db, tmp_path))

    question = seed_question(db,
        document_id=ids[0],
        page_id=1,
        question_kind="error",
        question_number="1",
        stem_text="题干",
    )
    seed_row(db, "mastery_records", question_id=question.id, outcome="incorrect")

    preview = service.preview(ids)
    assert preview.document_count == 2
    assert preview.page_count == 2
    assert preview.surviving_learning_questions == 1
    assert preview.errors == []
    assert not preview.missing_or_anomalous()

    with pytest.raises(DocumentDeletionError, match="明确确认"):
        service.execute(preview)
    outcomes = service.execute(preview, confirmed=True)
    assert all(outcome.deleted for outcome in outcomes)
    assert service.summarize(outcomes) == (
        "已删除 2 份，未删除 0 份；0 份含文件保留或清理说明。 "
        "资料、派生和暂存文件已核对无残留；保留区域不在清空范围内。"
    )

    assert db.get_document(ids[0]) is None
    assert db.get_document(ids[1]) is None
    # The learning asset survives, detached, and mastery records stay.
    survivor = read_question(db, question.id)
    assert survivor.document_id is None and survivor.page_id is None
    with db._connection() as connection:
        assert connection.execute("SELECT count(*) FROM mastery_records WHERE question_id=?",
                                  (question.id,)).fetchone()[0] == 1


def test_bulk_delete_isolates_partial_failures(two_documents, monkeypatch) -> None:
    db, tmp_path, ids = two_documents
    deletion_service = _make_service(db, tmp_path)
    service = DocumentBulkDeleteService(deletion_service)

    # Preflight now rejects anomalous batches as a whole. Exercise an actual
    # per-document failure occurring after a valid preview instead.
    real_delete = deletion_service.delete_document

    def fail_first(document_id, **kwargs):
        if document_id == ids[0]:
            raise DocumentDeletionError("synthetic late failure")
        return real_delete(document_id, **kwargs)

    monkeypatch.setattr(deletion_service, "delete_document", fail_first)
    preview = service.preview(ids)
    assert preview.errors == []
    assert not preview.missing_or_anomalous()
    outcomes = service.execute(preview, confirmed=True)
    failed = service.failed_ids(outcomes)
    assert failed == [ids[0]]
    assert not outcomes[0].deleted
    assert outcomes[0].message
    assert outcomes[1].deleted
    assert db.get_document(ids[1]) is None
    # The anomalous document still exists (abort means nothing changed).
    assert db.get_document(ids[0]) is not None

    # Retry still reports the same honest failure.
    retry_outcomes = service.execute(service.preview(failed), confirmed=True)
    assert retry_outcomes[0].deleted is False


def test_bulk_preview_aggregates_and_flags_missing_documents(two_documents) -> None:
    db, tmp_path, ids = two_documents
    service = DocumentBulkDeleteService(_make_service(db, tmp_path))
    preview = service.preview([ids[0], 99999])
    assert preview.document_count == 1
    assert preview.errors and preview.errors[0][0] == 99999


def test_learning_assets_still_resolve_after_bulk_delete(two_documents) -> None:
    db, tmp_path, ids = two_documents
    service = DocumentBulkDeleteService(_make_service(db, tmp_path))
    question = seed_question(db,
        document_id=ids[0], page_id=1, question_kind="good", stem_text="好题"
    )
    family_id = seed_row(db, "question_families", family_kind="type", title="函数与导数求极值")
    seed_row(db, "question_family_members", question_id=question.id, family_id=family_id)
    service.execute(service.preview([ids[0]]), confirmed=True)
    with db._connection() as connection:
        families = connection.execute(
            "SELECT f.title FROM question_families f JOIN question_family_members m "
            "ON m.family_id=f.id WHERE m.question_id=?", (question.id,),
        ).fetchall()
        assert [family[0] for family in families] == ["函数与导数求极值"]
        assert connection.execute("SELECT count(*) FROM question_family_members WHERE family_id=?",
                                  (family_id,)).fetchone()[0] == 1


def test_anomalous_preview_aborts_entire_batch(two_documents) -> None:
    """A known unsafe path cannot become a partial deletion of an approved batch."""
    db, tmp_path, ids = two_documents
    rogue = tmp_path / "outside.png"
    rogue.write_bytes(b"NON_REAL_TEST_DATA")
    with db._connection() as connection:
        connection.execute("UPDATE pages SET image_path=? WHERE document_id=?",
                           (str(rogue), ids[0]))
    service = DocumentBulkDeleteService(_make_service(db, tmp_path))
    with pytest.raises(DocumentDeletionError, match="异常"):
        service.execute(service.preview(ids), confirmed=True)
    assert all(db.get_document(identifier) is not None for identifier in ids)
    assert rogue.read_bytes() == b"NON_REAL_TEST_DATA"
