"""Synthetic UI regressions for mixed-file import and unobtrusive source handling."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from src.study_loop import import_ui
from src.study_loop.models import StudyError
from src.study_loop.storage import StudyStore
from tests.study_migration_support import image_case, panel, source_panel, text_case


class Upload(io.BytesIO):
    """A selected synthetic file without requiring browser upload machinery."""

    def __init__(self, name: str, data: bytes) -> None:
        super().__init__(data)
        self.name = name


def test_mixed_file_formats_are_detected_per_file_and_keep_originals(tmp_path, monkeypatch):
    store = StudyStore(tmp_path / "study", create=True)
    enqueued = []
    monkeypatch.setattr(import_ui, "_enqueue", lambda saved, identity: enqueued.append(identity))
    uploads = [
        Upload("SOURCE.PDF", b"NON_REAL_TEST_DATA PDF"),
        Upload("SOURCE.docx", b"NON_REAL_TEST_DATA WORD"),
        Upload("SOURCE.png", b"NON_REAL_TEST_DATA IMAGE"),
        Upload("SOURCE.md", b"NON_REAL_TEST_DATA MARKDOWN"),
        Upload("SOURCE.json", b'{"messages":[{"role":"user","content":"TEST"}]}'),
        Upload("SOURCE.txt", "用户：NON_REAL_TEST_DATA\nAI：TEST".encode()),
    ]
    identities = [import_ui.receive_upload(store, uploaded) for uploaded in uploads]
    assert enqueued == identities
    assert [store.source(identity).kind for identity in identities] == [
        "pdf", "docx", "image", "md", "json", "txt",
    ]
    for identity, uploaded in zip(identities, uploads, strict=True):
        assert store.read_source(store.source(identity)) == uploaded.getvalue()
    assert store.source(identities[-1]).transcript
    assert not store.drafts()  # Queueing is independent of extraction and human saving.


def test_one_bad_upload_does_not_discard_other_selected_materials(tmp_path, monkeypatch):
    store = StudyStore(tmp_path / "study", create=True)
    uploads = [Upload("bad.json", b"invalid json"), Upload("good.txt", b"NON_REAL_TEST_DATA")]
    batches = []
    monkeypatch.setattr(
        import_ui, "start_import_batch",
        lambda root, identities: (batches.append(identities), SimpleNamespace(id="TEST_BATCH"))[1],
    )
    monkeypatch.setattr(import_ui, "_enqueue", lambda *args: pytest.fail("must use one batch"))
    monkeypatch.setattr(import_ui, "_test_store", store, raising=False)
    monkeypatch.setattr(import_ui.st, "file_uploader", lambda *args, **kwargs: uploads)
    app = AppTest.from_string(
        "from src.study_loop.import_ui import render_input, _test_store\n"
        "render_input(_test_store)\n"
    ).run()
    assert not app.exception and not app.radio
    app.button(key="sl_receive").click().run()
    assert not app.exception
    assert len(app.error) == 1 and "bad.json" in app.error[0].value
    assert len(batches) == 1 and len(batches[0]) == 1
    assert store.source(batches[0][0]).name == "good.txt"
    assert {source.name for source in store.sources()} == {"bad.json", "good.txt"}
    assert all(store.read_source(source) for source in store.sources())


def test_unsupported_format_is_rejected_without_creating_a_source(tmp_path, monkeypatch):
    store = StudyStore(tmp_path / "study", create=True)
    monkeypatch.setattr(import_ui, "_enqueue", lambda *args: pytest.fail("must not enqueue"))
    with pytest.raises(StudyError, match="格式"):
        import_ui.receive_upload(store, Upload("SOURCE.exe", b"NON_REAL_TEST_DATA"))
    assert not store.sources()


def test_reading_a_draft_has_no_editing_questionnaire_or_technical_warnings(tmp_path: Path):
    store, source, draft = text_case(tmp_path / "study")
    original = store.read_source(source)
    app = panel(store, "drafts")
    assert not app.exception
    assert [widget.label for widget in app.text_input] == ["科目"]
    assert not app.text_area and not app.checkbox and not app.multiselect
    assert not app.warning and not app.get("download_button")
    assert "NON_REAL_TEST_DATA UNIT_A" in " ".join(element.value for element in app.markdown)
    assert not store.items()
    next(button for button in app.button if button.label == "保存学习材料").click().run()
    assert not app.exception
    saved = store.item(draft.id)
    assert saved.verification_status == "pending"
    assert saved.question == draft.question
    assert saved.page_ids == draft.page_ids
    assert store.read_source(source) == original


def test_source_images_keep_integrity_check_but_no_fingerprints_or_downloads(tmp_path: Path):
    store, source, page, _, _ = image_case(tmp_path / "study")
    app = source_panel(store, source.id, [page.id])
    assert not app.exception and len(app.get("image")) == 1
    assert not app.text_area and not app.get("download_button")
    visible = " ".join(element.value for element in [*app.text, *app.caption, *app.warning])
    assert source.sha256 not in visible
    store.path(page.image).write_bytes(b"NON_REAL_TEST_DATA CHANGED_IMAGE")
    failed = source_panel(store, source.id, [page.id])
    assert not failed.exception and failed.warning
    assert not failed.get("image")
