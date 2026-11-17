"""NON-LEARNING synthetic storage/format regression cases, never learning-effect evidence."""

from __future__ import annotations

import hashlib
import io
import json
import socket
from copy import deepcopy

import fitz
import pytest
from PIL import Image

from src.study_loop.components import export_review, import_source, search, verify_sources
from src.study_loop.models import Draft, StudyError, validate_draft
from src.study_loop.review import summary, unit_states
from src.study_loop.storage import StudyStore, encode


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No test in this module may contact a network service."""
    def reject(*args, **kwargs):
        raise AssertionError("Non-learning evidence regressions must remain offline")

    monkeypatch.setattr(socket, "create_connection", reject)
    monkeypatch.setattr(socket.socket, "connect", reject)


@pytest.fixture
def store(tmp_path):
    return StudyStore(tmp_path / "study", create=True)


def saved_text(store, text="NON-LEARNING FIXTURE: printer inventory record"):
    source = store.receive("infrastructure.txt", text.encode(), "txt")
    import_source(store, source.id)
    draft = next(d for d in store.drafts() if d.source_id == source.id)
    draft.material_kind = "note"
    store.confirm(draft, user_confirmed=True)
    return source, draft


def saved_pdf(store):
    """A one-page format fixture with no question, academic result or human history."""
    with fitz.open() as document:
        document.new_page().insert_text((40, 40), "NON-LEARNING FORMAT FIXTURE: printer asset")
        data = document.tobytes()
    source = store.receive("format-fixture.pdf", data, "pdf")
    import_source(store, source.id)
    draft = next(d for d in store.drafts() if d.source_id == source.id)
    draft.material_kind = "note"
    store.confirm(draft, user_confirmed=True)
    return source, draft


def test_original_duplicate_and_transcript_interpretation_are_preserved(store):
    source, _ = saved_text(store)
    raw = store.read_source(source)
    assert store.receive("renamed.txt", raw, "txt") == source
    with pytest.raises(StudyError, match="另一种对话标记"):
        store.receive("renamed.txt", raw, "txt", transcript=True)
    assert len(store.sources()) == 1
    assert store.read_source(source) == raw


def test_draft_collision_never_silently_discards_changed_extraction(store):
    _, draft = saved_text(store)
    with store.connect() as connection:
        original = Draft.from_dict(json.loads(connection.execute(
            "SELECT payload FROM drafts WHERE id=?", (draft.id,),
        ).fetchone()[0]))
    store.add_draft(original)
    changed = deepcopy(original)
    changed.question.content = "NON-LEARNING FIXTURE: changed extraction"
    with pytest.raises(StudyError, match="同一草稿"):
        store.add_draft(changed)


def test_legacy_payload_defaults_and_confirm_idempotency(store):
    _, draft = saved_text(store)
    legacy = draft.to_dict()
    for field in ("verification_status", "verification_note", "conflict_item_ids", "revision_of"):
        legacy.pop(field)
    with store.connect(write=True) as connection:
        connection.execute("UPDATE items SET payload=? WHERE id=?", (encode(legacy), draft.id))
        connection.execute("UPDATE drafts SET payload=? WHERE id=?", (encode(legacy), draft.id))
    restored = store.item(draft.id)
    assert restored.verification_status == "pending"
    store.confirm(restored, user_confirmed=True)
    assert len(store.items()) == 1


@pytest.mark.parametrize("state", ["verified", "error"])
def test_manual_verification_requires_evidence_note(store, state):
    _, draft = saved_text(store)
    revision = store.create_revision(draft.id, user_requested=True)
    revision.verification_status = state
    with pytest.raises(StudyError, match="人工核对依据"):
        validate_draft(revision)
    revision.verification_note = "NON-LEARNING FIXTURE: operator checked inventory label"
    store.confirm(revision, user_confirmed=True)
    assert store.item(revision.id).verification_status == state
    assert store.item(draft.id).verification_status == "pending"


def test_revision_preserves_original_raw_item_and_history(store):
    source, draft = saved_text(store)
    before = store.item(draft.id).to_dict()
    raw = store.read_source(source)
    session = store.start_review([draft.id], user_started=True)
    with pytest.raises(StudyError, match="主动选择"):
        store.create_revision(draft.id, user_requested=False)
    revision = store.create_revision(draft.id, user_requested=True)
    revision.question.content = "NON-LEARNING FIXTURE: corrected printer inventory"
    revision.question.answer_status = "user_note"
    revision.question.reference_answer = "NON-LEARNING FIXTURE: label checked"
    store.confirm(revision, user_confirmed=True)
    assert store.item(draft.id).to_dict() == before
    assert store.read_source(source) == raw
    assert revision.revision_of == draft.id
    assert store.sessions()[0]["id"] == session
    assert store.attempts() == []
    assert set(unit_states(store.items(), store.attempts()).values()) == {"unknown"}


def test_conflict_links_require_existing_source_and_preserve_both_records(store):
    first_source, first = saved_text(store)
    second_source, second = saved_text(store, "NON-LEARNING FIXTURE: second inventory record")
    revised = store.create_revision(first.id, user_requested=True)
    revised.conflict_item_ids = ["missing"]
    revised.verification_note = "NON-LEARNING FIXTURE: two inventory labels disagree"
    with pytest.raises(StudyError, match="指向已保存"):
        store.confirm(revised, user_confirmed=True)
    revised.conflict_item_ids = [second.id]
    store.confirm(revised, user_confirmed=True)
    text = summary(store, [revised.id])
    assert first_source.sha256 in text and second_source.sha256 in text
    assert second.id in text and first.id in text
    assert "未自动裁定" in text
    assert len(store.items()) == 3


def test_revision_pointer_cannot_be_forged_during_confirmation(store):
    _, first = saved_text(store)
    _, second = saved_text(store, "NON-LEARNING FIXTURE: different receipt")
    revised = store.create_revision(first.id, user_requested=True)
    revised.revision_of = second.id
    with pytest.raises(StudyError, match="修订来源"):
        store.confirm(revised, user_confirmed=True)


def test_search_reuses_persistent_index_for_reference_and_verification_notes(store):
    _, draft = saved_text(store)
    revised = store.create_revision(draft.id, user_requested=True)
    revised.question.answer_status = "user_note"
    revised.question.reference_answer = "assetlocatorquartz"
    revised.question.reference_analysis = "maintenancerecordviolet"
    revised.verification_note = "inspectorremarksilver"
    store.confirm(revised, user_confirmed=True)
    for query in ("assetlocatorquartz", "maintenancerecordviolet", "inspectorremarksilver"):
        assert revised.id in search(store, query)
    reopened = StudyStore(store.root)
    assert revised.id in search(reopened, "assetlocatorquartz")
    assert store.path("search.sqlite3").is_file()


@pytest.mark.parametrize("repetitions", [7875, 12475])
def test_long_confirmed_leaf_does_not_break_other_library_searches(store, repetitions):
    _, long_item = saved_text(store, "printer fixture " * repetitions)
    _, short_item = saved_text(store, "NON-LEARNING FIXTURE: shortlookupmarker")
    assert len(long_item.question.content) > 100_000
    assert short_item.id in search(store, "shortlookupmarker")
    assert long_item.id in search(store, "printer")


def test_large_saved_reference_is_fully_indexed_and_fragment_hits_are_deduplicated(store):
    _, draft = saved_text(store)
    revised = store.create_revision(draft.id, user_requested=True)
    revised.question.answer_status = "user_note"
    revised.question.reference_answer = "printer fixture " * 14000
    revised.question.reference_analysis = "tailreferencequartz"
    store.confirm(revised, user_confirmed=True)
    assert search(store, "tailreferencequartz") == [revised.id]
    assert search(store, "printer").count(revised.id) == 1
    assert store.item(revised.id).question.reference_answer == revised.question.reference_answer


@pytest.mark.parametrize("selection", [[], ["missing"]])
def test_verification_rejects_empty_or_unknown_selection(store, selection):
    with pytest.raises(StudyError):
        verify_sources(store, selection)


def test_verification_rejects_partially_missing_selection(store):
    _, draft = saved_text(store)
    with pytest.raises(StudyError, match="不存在"):
        verify_sources(store, [draft.id, "missing"])
    with pytest.raises(StudyError, match="无重复"):
        verify_sources(store, [draft.id, draft.id])
    result = verify_sources(store, [draft.id])
    assert result["byte_checked_sources"] == 1
    assert result["pdf_page_references"] == []
    assert store.item(draft.id).verification_status == "pending"


def test_mixed_export_checks_non_pdf_original_bytes(store):
    _, pdf = saved_pdf(store)
    source, text = saved_text(store)
    store.path(source.blob).write_bytes(b"NON-LEARNING FIXTURE: tampered receipt")
    with pytest.raises(StudyError, match="原始材料已改变"):
        export_review(store, [pdf.id, text.id], "mixed")
    assert not store.path("exports/mixed").exists()


def test_export_selection_is_pinned_and_idempotent(store):
    _, pdf = saved_pdf(store)
    _, text = saved_text(store)
    first = export_review(store, [pdf.id], "snapshot")
    second = export_review(store, [pdf.id], "snapshot")
    assert first == second
    selection = store.path("exports/snapshot/attachments/selection.json").read_text()
    assert json.loads(selection) == [pdf.id]
    with pytest.raises(StudyError, match="本次所选材料不一致"):
        export_review(store, [pdf.id, text.id], "snapshot")
    with pytest.raises(StudyError, match="导出标识"):
        export_review(store, [pdf.id], "../originals")


def test_mpo_camera_image_preserves_all_frames_and_original(store):
    output = io.BytesIO()
    first = Image.new("RGB", (80, 60), "red")
    second = Image.new("RGB", (40, 30), "blue")
    first.save(output, format="MPO", save_all=True, append_images=[second])
    raw = output.getvalue()
    source = store.receive("non-learning-camera-fixture.jpg", raw, "image")
    result = import_source(store, source.id)
    pages = store.pages(source.id)
    assert len(pages) == 2 and result["draft_count"] == 2
    assert store.read_source(source) == raw
    assert [p.metadata["source_frame"] for p in pages] == [1, 2]
    for page in pages:
        assert page.number is None
        assert page.needs_review is True
        assert page.metadata["source_sha256"] == hashlib.sha256(raw).hexdigest()
        assert hashlib.sha256(store.path(page.image).read_bytes()).hexdigest() == page.image_sha256
    assert import_source(store, source.id) == result
    assert len(store.pages(source.id)) == 2
    assert store.items() == []
