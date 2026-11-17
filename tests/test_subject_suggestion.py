"""AI 建议学科 + 一键确认 (v0.8.8 AI-First offload #2).

The subject pick must become AI-suggests + one-click user confirmation while
preserving the human right to override, and it must never auto-write a
subject.  Suggestions are derived from locally stored evidence only (title +
native text layer + optional reading keywords), so the path works with AI
disabled and never becomes a startup dependency.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.database import Database
from src.subject_suggestion_service import (
    build_document_evidence,
    ordered_candidates,
    suggest_subjects,
)


# ---------------------------------------------------------------- pure service
def test_suggest_subjects_prefers_direct_title_match() -> None:
    evidence = "液压与气动 第三章 液压执行元件\n溢流阀、方向阀是常见元件。"
    suggestions = suggest_subjects(evidence, ["数学", "液压与气动", "机械工程", "英语"])
    assert suggestions
    assert suggestions[0][0] == "液压与气动"
    # A subject absent from the evidence is never suggested.
    assert all(subject != "英语" for subject, _ in suggestions)


def test_suggest_subjects_returns_nothing_without_evidence() -> None:
    assert suggest_subjects("", ["数学", "液压与气动"]) == []
    assert suggest_subjects("无关内容", []) == []


def test_suggest_subjects_never_fabricates_unmatched_candidate() -> None:
    suggestions = suggest_subjects("只谈天气", ["液压与气动", "机械工程"])
    assert suggestions == []


def test_ordered_candidates_puts_saved_and_existing_first() -> None:
    candidates = ordered_candidates(
        ("数学", "物理"), saved_subject="液压与气动", existing_subjects=["液压与气动"]
    )
    assert candidates[0] == "液压与气动"
    assert "数学" in candidates and "物理" in candidates
    # Deduplicated.
    assert candidates.count("液压与气动") == 1


def test_build_document_evidence_uses_title_and_text_layer(tmp_path: Path) -> None:
    database = Database(tmp_path / "knowledge.db")
    raw = tmp_path / "raw" / "课件.pdf"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b"%PDF-1.7 fixture")
    image = tmp_path / "pages" / "1" / "page-1.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"png")
    document = database.create_document(
        title="液压与气动课件",
        filename="课件.pdf",
        source_path=raw,
        sha256="d" * 64,
        import_status="completed",
    )
    database.create_page(
        document_id=document.id,
        page_number=1,
        image_path=image,
        extracted_text="液压基本回路由溢流阀与方向阀组成。",
        status="ready",
    )
    evidence = build_document_evidence(database, document.id)
    assert "液压与气动课件" in evidence
    assert "溢流阀" in evidence


def test_build_document_evidence_merges_injected_reading_keywords(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "knowledge.db")
    raw = tmp_path / "raw" / "课件.pdf"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b"%PDF-1.7 fixture")
    image = tmp_path / "pages" / "1" / "page-1.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"png")
    document = database.create_document(
        title="课件",
        filename="课件.pdf",
        source_path=raw,
        sha256="e" * 64,
        import_status="completed",
    )
    page = database.create_page(
        document_id=document.id,
        page_number=1,
        image_path=image,
        extracted_text="",
        status="ready",
    )
    store = SimpleNamespace(
        page_reading=lambda page_id: (
            SimpleNamespace(summary="本页讲气动元件", keywords=("气压传动", "气缸"))
            if page_id == page.id
            else None
        )
    )
    evidence = build_document_evidence(database, document.id, reading_store=store)
    assert "气压传动" in evidence and "气缸" in evidence


# ------------------------------------------------------------------- UI flow


def test_model_subject_prefill_requires_explicit_current_draft_confirmation(tmp_path: Path) -> None:
    import json

    from src.study_loop.extraction import save_extraction
    from tests.study_migration_support import panel, text_case

    store, source, local = text_case(tmp_path / "study", subject="")
    original = store.read_source(source)
    save_extraction(
        store,
        source.id,
        local.page_ids[0],
        {
            "text": json.dumps(
                {
                    "subject": "TEST_MODEL_COURSE",
                    "questions": [{"content": "NON_REAL_TEST_DATA MODEL_CONTENT"}],
                }
            ),
            "model": "OFFLINE_TEST_RESPONSE",
        },
        "TEST_SUBJECT_RUN",
    )
    draft = next(d for d in store.drafts() if d.extraction == "model_draft")
    questions = [d for d in store.drafts() if d.source_id == source.id]
    app = panel(store, "drafts")
    app.selectbox(key="sl_draft_choice").select(questions[0]).run()
    index = next(index for index, question in enumerate(questions) if question.id == draft.id)
    app.selectbox(key="sl_draft_question_" + source.id).set_value(index).run()
    assert not app.exception
    assert app.session_state["sl_selected_draft_id"] == draft.id
    assert next(t for t in app.text_input if t.label == "科目").value == "TEST_MODEL_COURSE"
    assert not store.items()
    # Reading never writes; the explicit save action confirms this draft.
    next(b for b in app.button if b.label == "保存学习材料").click().run()
    assert not app.exception
    assert store.item(draft.id).subject == "TEST_MODEL_COURSE"
    assert store.item(draft.id).verification_status == "pending"
    assert local.id in {question.id for question in store.drafts()}
    assert store.read_source(source) == original


def test_local_subject_prefill_and_confirmation_work_without_ai(
    tmp_path: Path, monkeypatch
) -> None:
    import src.runtime as runtime
    from tests.study_migration_support import panel, text_case

    monkeypatch.setattr(runtime, "application_ai_provider", lambda: None)
    store, source, draft = text_case(tmp_path / "study")
    original = store.read_source(source)
    app = panel(store, "drafts")
    assert not app.exception
    assert next(t for t in app.text_input if t.label == "科目").value == "TEST_COURSE"
    assert not store.items()
    next(b for b in app.button if b.label == "保存学习材料").click().run()
    assert not app.exception
    assert store.item(draft.id).subject == "TEST_COURSE"
    assert store.read_source(source) == original


def test_manual_subject_override_preserves_original_and_extraction(tmp_path: Path) -> None:
    import json

    from src.study_loop.storage import StudyStore
    from tests.study_migration_support import panel, text_case

    store, source, draft = text_case(tmp_path / "study")
    original = store.read_source(source)
    app = panel(store, "drafts")
    next(t for t in app.text_input if t.label == "科目").set_value("TEST_MANUAL_COURSE")
    next(b for b in app.button if b.label == "保存学习材料").click().run()
    assert not app.exception
    reopened = StudyStore(store.root)
    assert reopened.item(draft.id).subject == "TEST_MANUAL_COURSE"
    assert reopened.read_source(source) == original
    with reopened.connect() as connection:
        extraction = json.loads(
            connection.execute("SELECT payload FROM drafts WHERE id=?", (draft.id,)).fetchone()[0]
        )
    assert extraction == draft.to_dict()
    assert extraction["subject"] == "TEST_COURSE"
