"""Current image/source contracts; migration mapping is in docs/migration20-20261112.

No legacy attachment editor is emulated. All inputs are NON_REAL_TEST_DATA.
"""

from __future__ import annotations

import copy
import hashlib
import io
import struct
import zlib
from pathlib import Path

import pytest
from PIL import Image

from src.study_loop.components import import_source
from src.study_loop.models import Page, StudyError
from src.study_loop.storage import MAX_BYTES, StudyStore
from tests.study_migration_support import image_bytes, image_case, panel, pdf_case, submit_attempt


def test_attempt_scope_never_mingles_with_source_or_other_item(tmp_path: Path) -> None:
    store, source, page, first, raw = image_case(tmp_path / "study")
    second_source = store.receive(source.name, image_bytes("red"), "image")
    import_source(store, second_source.id)
    second = next(d for d in store.drafts() if d.source_id == second_source.id)
    first.question.number = second.question.number = "7"
    store.confirm(first, user_confirmed=True)
    store.confirm(second, user_confirmed=True)
    session = store.start_review([first.id, second.id], user_started=True)
    before = [item.to_dict() for item in store.items()]
    submit_attempt(store, session, first)
    submit_attempt(store, session, second, index=2)
    records = store.attempt_records()
    assert [r["item_id"] for r in records] == [first.id, second.id]
    assert records[0]["page_ids"] == [page.id]
    assert records[1]["page_ids"] == second.page_ids != first.page_ids
    assert [r["answer"] for r in records] == [
        "NON_REAL_TEST_DATA RESPONSE_1",
        "NON_REAL_TEST_DATA RESPONSE_2",
    ]
    assert [item.to_dict() for item in store.items()] == before
    assert store.read_source(source) == raw
    with pytest.raises(StudyError, match="小问"):
        store.record_attempt(
            session,
            first.id,
            second.question.id,
            outcome="partial",
            answer="NON_REAL_TEST_DATA",
            note="",
            independent=True,
            attempted=True,
            error_tags=[],
            submission_id="WRONG_UNIT",
        )
    assert len(store.attempts()) == 2


def test_original_images_preserved_mpo_png_derivatives_and_deduplication(tmp_path: Path) -> None:
    store, source, page, draft, raw = image_case(tmp_path / "study", jpeg=True)
    assert store.read_source(source) == raw
    assert source.sha256 == hashlib.sha256(raw).hexdigest()
    with Image.open(io.BytesIO(raw)) as original:
        assert original.getexif()[0x010F] == "NON_REAL_TEST_DATA_CAMERA"
    derived = store.path(page.image).read_bytes()
    assert hashlib.sha256(derived).hexdigest() == page.image_sha256
    with Image.open(io.BytesIO(derived)) as picture:
        assert picture.format == "JPEG"
        assert picture.size == (80, 60)
        assert picture.getexif()[0x010F] == "NON_REAL_TEST_DATA_CAMERA"
    assert derived == raw  # Current Import Core retains ordinary JPEG originals.
    again = store.receive(source.name, raw, "image")
    import_source(store, again.id)
    assert again == source
    assert store.pages(source.id) == [page]
    assert [d.to_dict() for d in store.drafts()] == [draft.to_dict()]
    assert len(list(store.path("originals").iterdir())) == 1

    # Current PNG conversion exists for MPO camera frames, not answer attachments.
    output = io.BytesIO()
    first, second = Image.new("RGB", (80, 60), "red"), Image.new("RGB", (40, 30), "blue")
    exif = Image.Exif()
    exif[0x010F] = "NON_REAL_TEST_DATA_CAMERA"
    first.save(output, format="MPO", save_all=True, append_images=[second], exif=exif)
    mpo = output.getvalue()
    with Image.open(io.BytesIO(mpo)) as original_mpo:
        assert original_mpo.getexif()[0x010F] == "NON_REAL_TEST_DATA_CAMERA"
    camera = store.receive("NON_REAL_TEST_DATA_CAMERA.jpg", mpo, "image")
    import_source(store, camera.id)
    frames = store.pages(camera.id)
    assert len(frames) == 2
    assert store.read_source(camera) == mpo
    assert [p.metadata["source_frame"] for p in frames] == [1, 2]
    for frame, size in zip(frames, [(80, 60), (40, 30)], strict=True):
        data = store.path(frame.image).read_bytes()
        with Image.open(io.BytesIO(data)) as converted:
            assert converted.format == "PNG"
            assert converted.size == size
            assert not converted.info.get("exif")
        assert hashlib.sha256(data).hexdigest() == frame.image_sha256
    import_source(store, camera.id)
    assert store.pages(camera.id) == frames
    assert store.read_source(camera) == mpo


def _oversized_png() -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", 100_000, 10_000, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IEND", b"")


def test_receipt_and_image_import_reject_invalid_payloads(tmp_path: Path) -> None:
    store = StudyStore(tmp_path / "study", create=True)
    for name, data, kind in [
        ("empty.png", b"", "image"),
        ("large.png", b"x" * (MAX_BYTES + 1), "image"),
        ("wrong.exe", b"NON_REAL_TEST_DATA", "image"),
        ("bad.png", b"NON_REAL_TEST_DATA", "unsupported"),
    ]:
        with pytest.raises(StudyError):
            store.receive(name, data, kind)
    assert not store.sources()
    assert not list(store.path("originals").iterdir())
    for raw in (b"NON_REAL_TEST_DATA NOT_AN_IMAGE", _oversized_png()):
        source = store.receive("NON_REAL_TEST_DATA.png", raw, "image")
        with pytest.raises(StudyError, match="图像|图片"):
            import_source(store, source.id)
        assert store.read_source(source) == raw
        assert not store.pages(source.id)
    assert not store.drafts()
    assert not store.items()
    assert not store.attempts()


def test_revision_removes_reference_but_preserves_original_and_page(tmp_path: Path) -> None:
    store, source, pages, draft, raw = pdf_case(tmp_path / "study")
    store.confirm(draft, user_confirmed=True)
    previous = store.item(draft.id).to_dict()
    images = [store.path(p.image).read_bytes() for p in pages]
    revision = store.create_revision(draft.id, user_requested=True)
    revision.page_ids = [pages[1].id]
    store.confirm(revision, user_confirmed=True)
    assert store.item(revision.id).page_ids == [pages[1].id]
    assert store.item(revision.id).revision_of == draft.id
    assert store.item(draft.id).to_dict() == previous
    assert store.pages(source.id) == pages
    assert [store.path(p.image).read_bytes() for p in pages] == images
    assert store.read_source(source) == raw


def test_confirmation_rejects_invalid_duplicate_and_foreign_references(tmp_path: Path) -> None:
    store, source, page, draft, raw = image_case(tmp_path / "study")
    other = store.receive("NON_REAL_TEST_DATA.txt", b"NON_REAL_TEST_DATA", "txt")
    import_source(store, other.id)
    foreign = store.pages(other.id)[0]
    before = [d.to_dict() for d in store.drafts()]
    for references in (["missing"], [page.id, page.id], [foreign.id], [], [None]):
        edited = copy.deepcopy(draft)
        edited.page_ids = references
        with pytest.raises(StudyError, match="来源"):
            store.confirm(edited, user_confirmed=True)
        assert not store.items()
        assert not store.attempts()
        assert [d.to_dict() for d in store.drafts()] == before
    assert store.read_source(source) == raw


def test_revised_page_selection_survives_reopen_without_resurrection(tmp_path: Path) -> None:
    store, source, pages, draft, raw = pdf_case(tmp_path / "study")
    store.confirm(draft, user_confirmed=True)
    revision = store.create_revision(draft.id, user_requested=True)
    revision.page_ids = [pages[1].id]
    store.confirm(revision, user_confirmed=True)
    reopened = StudyStore(store.root)
    next_revision = reopened.create_revision(revision.id, user_requested=True)
    assert next_revision.page_ids == [pages[1].id]
    app = panel(reopened, "drafts")
    app.selectbox(key="sl_draft_choice").select(next_revision).run()
    assert not app.exception
    assert not app.multiselect
    assert not app.get("image")
    from tests.study_migration_support import source_panel

    source_view = source_panel(reopened, source.id, next_revision.page_ids)
    assert not source_view.exception
    assert len(source_view.get("image")) == 1
    assert not source_view.get("download_button")
    assert reopened.item(draft.id).page_ids == [p.id for p in pages]
    assert reopened.read_source(source) == raw
    assert all(reopened.path(p.image).is_file() for p in pages)


def test_attempt_and_manual_revision_preserve_extraction_and_history(tmp_path: Path) -> None:
    store, source, _, draft, raw = image_case(tmp_path / "study")
    extraction = draft.to_dict()
    draft.question.reference_answer = "NON_REAL_TEST_DATA USER_NOTE"
    draft.question.answer_status = "user_note"
    store.confirm(draft, user_confirmed=True)
    original = store.item(draft.id).to_dict()
    session = store.start_review([draft.id], user_started=True)
    submit_attempt(store, session, draft)
    history = store.attempt_records()
    assert store.item(draft.id).to_dict() == original
    revision = store.create_revision(draft.id, user_requested=True)
    revision.question.content = "NON_REAL_TEST_DATA MANUAL_CORRECTION"
    store.confirm(revision, user_confirmed=True)
    assert store.item(revision.id).question.content == revision.question.content
    assert store.item(revision.id).question.reference_answer == draft.question.reference_answer
    assert store.item(revision.id).extraction == "user_revision"
    assert store.item(draft.id).to_dict() == original
    assert store.attempt_records() == history
    assert history[0]["item_id"] == draft.id != revision.id
    with store.connect() as connection:
        from json import loads

        saved = loads(
            connection.execute("SELECT payload FROM drafts WHERE id=?", (draft.id,)).fetchone()[0]
        )
    assert saved == extraction
    assert store.read_source(source) == raw


def test_current_source_scope_limit_rejects_201_and_accepts_200(tmp_path: Path) -> None:
    store, source, page, draft, raw = image_case(tmp_path / "study")
    for index in range(200):
        store.put_page(
            Page(
                f"TEST_SEGMENT_{index}", source.id, None, text=f"NON_REAL_TEST_DATA SEGMENT_{index}"
            )
        )
    draft.page_ids = [p.id for p in store.pages(source.id)]
    assert len(draft.page_ids) == 201
    with pytest.raises(StudyError, match="来源"):
        store.confirm(draft, user_confirmed=True)
    assert not store.items()
    draft.page_ids = draft.page_ids[:200]
    store.confirm(draft, user_confirmed=True)
    assert store.item(draft.id).page_ids == draft.page_ids
    assert len(store.pages(source.id)) == 201
    assert store.pages(source.id)[0] == page
    assert store.read_source(source) == raw


def test_review_renders_source_and_saves_only_explicit_attempt(tmp_path: Path) -> None:
    store, source, page, draft, raw = image_case(tmp_path / "study")
    store.confirm(draft, user_confirmed=True)
    session = store.start_review([draft.id], user_started=True)
    app = panel(store, "review", advanced=True)
    assert not app.exception
    assert len(app.get("image")) == 1
    assert not app.get("download_button")
    assert not store.attempts()
    assert source.name in " ".join(t.value for t in app.text)
    next(t for t in app.text_area if t.label == "我的实际作答 / 回忆内容").set_value(
        "NON_REAL_TEST_DATA UI_RESPONSE"
    )
    next(c for c in app.checkbox if c.label.startswith("我已实际尝试")).check()
    next(b for b in app.button if b.label == "记录本次结果").click().run()
    assert not app.exception
    records = store.attempt_records()
    assert len(records) == 1
    assert records[0]["answer"] == "NON_REAL_TEST_DATA UI_RESPONSE"
    assert records[0]["session_id"] == session
    assert records[0]["page_ids"] == [page.id]
    app.run()
    assert len(store.attempts()) == 1
    assert "NON_REAL_TEST_DATA UI_RESPONSE" in " ".join(m.value for m in app.markdown)
    assert store.read_source(source) == raw
