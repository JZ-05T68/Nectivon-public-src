"""Saved-review download UI contracts using temporary data and local-only conversion."""

from __future__ import annotations

import copy
import io
import zipfile
from datetime import datetime
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest
from color_editor_support import edit_color_body
from streamlit.delta_generator import DeltaGenerator
from streamlit.testing.v1 import AppTest
from test_ai_review_support import RecordingProvider, review_case

from src.study_loop import ai_review_export
from src.study_loop.ai_review import AIReviewService

SAVED = (
    "# NON_REAL_SAVED_REVIEW\n\n人工保存的复盘版本。\n\n"
    r"$$q=\frac{a}{b}+\sqrt{x^2+y^2}$$"
    "\n\n|变量|说明|\n|---|---|\n|$b$|合成条件 $b>0$|\n"
)


@pytest.fixture
def export_case(tmp_path, monkeypatch):
    """Capture real column download widgets without replacing rendering or storage."""
    monkeypatch.setenv("EKB_STUDY_LOOP_ACCEPTANCE_CONFIRMED", "yes")
    case = review_case(tmp_path / "NON_REAL_EXPORT_UI")
    service, provider = AIReviewService(case.store), RecordingProvider("NON_REAL_ORIGINAL_AI")
    document = service.generate([case.selected.id], provider)
    downloads, conversions, model_bindings = {}, [], []
    render_download = DeltaGenerator.download_button
    convert = ai_review_export.docx

    def capture(self, label, data, *args, **kwargs):
        downloads[kwargs["key"]] = {
            "label": label, "data": data, "filename": args[0], **kwargs,
        }
        return render_download(self, label, data, *args, **kwargs)

    def local_docx(service, version_id):
        conversions.append(version_id)
        return convert(service, version_id)

    def no_model_binding():
        model_bindings.append(True)
        raise AssertionError("Opening, saving or downloading a review must not bind a model")

    monkeypatch.setattr(DeltaGenerator, "download_button", capture)
    monkeypatch.setattr(ai_review_export, "docx", local_docx)
    monkeypatch.setattr("src.study_loop.ai_review_ui.model_provider", no_model_binding)
    return SimpleNamespace(
        case=case, service=service, provider=provider, document=document,
        downloads=downloads, conversions=conversions, model_bindings=model_bindings,
    )


def _app(fixture) -> AppTest:
    app = AppTest.from_string("from src.study_loop.ui import render\nrender()")
    app.session_state["sl_open_root"] = str(fixture.case.store.root)
    app.session_state["sl_ai_review_focus"] = True
    return app.run(timeout=30)


def _buttons(fixture, version, *, location="current"):
    prefix = "sl_ai_review_export_" + location + "_"
    markdown = fixture.downloads[prefix + "md_" + version["id"]]
    word = fixture.downloads[prefix + "docx_" + version["id"]]
    assert markdown["label"] == "下载 Markdown" and markdown["mime"] == "text/markdown"
    assert markdown["data"] == version["content"]
    assert word["label"] == "下载 Word"
    assert word["mime"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert callable(word["data"])
    assert markdown["on_click"] == word["on_click"] == "ignore"
    return markdown, word


def _assert_word(data: bytes, content: str) -> None:
    with zipfile.ZipFile(io.BytesIO(data)) as package:
        assert package.read("nectivon/transcript.md") == content.encode()
        xml = ET.fromstring(package.read("word/document.xml"))
        namespace = {"m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}
        assert xml.find(".//m:f", namespace) is not None
        assert xml.find(".//m:rad", namespace) is not None


def _assert_manuscript_unchanged(before: dict, after: dict, viewed: str | None = None) -> None:
    """Opening a view updates view metadata, never AI/body/version/provenance fields."""
    view_fields = {"last_opened_at", "viewed_version_id"}
    assert {k: v for k, v in after.items() if k not in view_fields} == {
        k: v for k, v in before.items() if k not in view_fields
    }
    assert after["viewed_version_id"] == viewed
    assert datetime.fromisoformat(after["last_opened_at"]) >= datetime.fromisoformat(
        before.get("last_opened_at", before["created_at"])
    )


def test_unsaved_review_shows_save_hint_without_downloads_or_conversion(export_case):
    fixture = export_case
    before = copy.deepcopy(fixture.service.document(fixture.document["id"]))
    app = _app(fixture)
    assert not app.exception and not app.error
    assert any(value.value == "保存复盘后可导出 Markdown 和 Word。" for value in app.caption)
    assert app.button(key="sl_ai_review_save_" + fixture.document["id"])
    assert not app.get("download_button") and fixture.downloads == {}
    app.run(timeout=30)
    assert fixture.conversions == fixture.model_bindings == []
    _assert_manuscript_unchanged(before, fixture.service.document(fixture.document["id"]))
    assert len(fixture.provider.prompts) == 1


def test_explicit_save_offers_exact_markdown_and_deferred_native_math_word(export_case):
    fixture = export_case
    identity = fixture.document["id"]
    app = _app(fixture)
    app.radio(key="sl_ai_review_view_" + identity).set_value("编辑").run(timeout=30)
    edit_color_body(app, "复盘正文", SAVED)
    app.button(key="sl_ai_review_save_" + identity).click().run(timeout=30)
    assert not app.exception and not app.error
    version = fixture.service.versions(identity)[0]
    assert version["content"] == SAVED
    markdown, word = _buttons(fixture, version)
    assert markdown["filename"].endswith("复盘-第1版.md")
    assert word["filename"].endswith("复盘-第1版.docx")
    assert any(value.value == "导出第1版（已保存内容）" for value in app.caption)
    assert len(app.get("download_button")) == 4  # Current saved copy plus advanced history.
    assert fixture.conversions == []  # Merely displaying the page never converts Word.
    before = fixture.service.chats.db.read_bytes()
    _assert_word(word["data"](), SAVED)
    assert fixture.conversions == [version["id"]]
    assert fixture.service.chats.db.read_bytes() == before
    assert fixture.service.document(identity)["working"] == SAVED
    assert fixture.service.document(identity)["raw_ai"] == "NON_REAL_ORIGINAL_AI"
    assert fixture.model_bindings == [] and len(fixture.provider.prompts) == 1


def test_changed_or_empty_working_text_still_exports_the_last_saved_revision(export_case):
    fixture = export_case
    identity = fixture.document["id"]
    fixture.service.update_working(identity, SAVED)
    version = fixture.service.save_version(identity)
    app = _app(fixture)
    for working in ("NON_REAL_UNSAVED_MUST_NOT_EXPORT", ""):
        fixture.service.update_working(identity, working)
        before = copy.deepcopy(fixture.service.document(identity))
        fixture.downloads.clear()
        app.run(timeout=30)
        assert not app.exception and not app.error
        _, word = _buttons(fixture, version)
        assert fixture.conversions == []
        _assert_manuscript_unchanged(before, fixture.service.document(identity))
        assert [d['id'] for d in fixture.service.documents()] == [identity]
        assert fixture.service.version(version["id"])["content"] == SAVED
        # The empty working-copy branch must also retain a usable deferred Word download.
        if working == "":
            _assert_word(word["data"](), SAVED)
            assert fixture.conversions == [version["id"]]
    assert fixture.service.document(identity)["working"] == ""
    assert fixture.model_bindings == [] and len(fixture.provider.prompts) == 1


def test_history_selection_and_opened_version_download_the_exact_older_saved_copy(export_case):
    fixture = export_case
    identity = fixture.document["id"]
    fixture.service.update_working(identity, SAVED)
    first = fixture.service.save_version(identity)
    fixture.service.update_working(identity, "NON_REAL_SECOND_SAVED_COPY")
    second = fixture.service.save_version(identity)
    fixture.service.update_working(identity, "NON_REAL_CURRENT_UNSAVED_COPY")
    before = copy.deepcopy(fixture.service.document(identity))
    app = _app(fixture)
    _buttons(fixture, second)
    fixture.downloads.clear()
    app.selectbox(key="sl_ai_review_history_version_" + identity).set_value(first).run(timeout=30)
    assert not app.exception and not app.error
    _buttons(fixture, second)  # Choosing a history row does not replace the current view.
    _, history_word = _buttons(fixture, first, location="history")
    assert fixture.conversions == []
    fixture.downloads.clear()
    app.button(key="sl_ai_review_read_version").click().run(timeout=30)
    assert not app.exception and not app.error
    _, current_word = _buttons(fixture, first)
    _assert_word(history_word["data"](), SAVED)
    _assert_word(current_word["data"](), SAVED)
    assert fixture.conversions == [first["id"], first["id"]]
    _assert_manuscript_unchanged(before, fixture.service.document(identity), first["id"])
    assert fixture.service.versions(identity) == [second, first]
    assert fixture.case.chats.memories(None) == []
    assert fixture.model_bindings == [] and len(fixture.provider.prompts) == 1
