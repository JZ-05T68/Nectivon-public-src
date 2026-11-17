"""Saved-review export regressions with synthetic local materials and a fake provider."""

from __future__ import annotations

import copy
import io
import zipfile
from xml.etree import ElementTree as ET

import pytest
from test_ai_review_support import RecordingProvider, review_case

from src.study_loop import ai_review_export
from src.study_loop.ai_review import AIReviewService
from src.study_loop.models import StudyError

SAVED = (
    "# 合成学习复盘\n\n人工修订的第一版，保留条件 $v>0$。\n\n"
    r"$$F=\frac{P}{v},\qquad q=\sqrt{x^2+y^2}$$"
    "\n\n|量|含义|\n|---|---|\n|$F$|合成示例|\n"
)


@pytest.fixture
def saved_case(tmp_path):
    case = review_case(tmp_path / "NON_REAL_EXPORT_LIBRARY")
    service, provider = AIReviewService(case.store), RecordingProvider("NON_REAL_RAW_AI_ONLY")
    document = service.generate([case.selected.id], provider)
    service.update_working(document["id"], SAVED)
    version = service.save_version(document["id"])
    return case, service, provider, document, version


def test_exports_use_exact_selected_saved_revision_and_leave_all_state_unchanged(saved_case):
    case, service, provider, document, first = saved_case
    service.update_working(document["id"], "NON_REAL_SECOND_SAVED_VERSION")
    second = service.save_version(document["id"])
    service.update_working(document["id"], "NON_REAL_UNSAVED_WORKING_MUST_NOT_EXPORT")
    stored = copy.deepcopy(service.document(document["id"]))
    versions = copy.deepcopy(service.versions(document["id"]))
    conversation = copy.deepcopy(case.chats.snapshot(document["conversation_id"]))
    study_bytes, chat_bytes = case.store.db.read_bytes(), service.chats.db.read_bytes()

    assert ai_review_export.markdown(service, first["id"]) == SAVED
    assert ai_review_export.markdown(service, second["id"]) == second["content"]
    with zipfile.ZipFile(io.BytesIO(ai_review_export.docx(service, first["id"]))) as package:
        assert package.read("nectivon/transcript.md").decode("utf-8") == SAVED
        body = package.read("word/document.xml").decode("utf-8")
        assert "NON_REAL_RAW_AI_ONLY" not in body
        assert "NON_REAL_SECOND_SAVED_VERSION" not in body
        assert "NON_REAL_UNSAVED_WORKING_MUST_NOT_EXPORT" not in body

    assert service.document(document["id"]) == stored
    assert service.versions(document["id"]) == versions
    assert case.chats.snapshot(document["conversation_id"]) == conversation
    assert case.store.db.read_bytes() == study_bytes
    assert service.chats.db.read_bytes() == chat_bytes
    assert len(provider.prompts) == 1
    assert not case.chats.memories(None) and not case.chats.candidates(document["conversation_id"])


def test_word_keeps_native_equations_tables_and_exact_markdown(saved_case):
    _, service, _, _, version = saved_case
    ns = {
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
        "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    }
    with zipfile.ZipFile(io.BytesIO(ai_review_export.docx(service, version["id"]))) as package:
        root = ET.fromstring(package.read("word/document.xml"))
        assert root.find(".//m:f", ns) is not None
        assert root.find(".//m:rad", ns) is not None
        assert root.find(".//m:sSup", ns) is not None
        assert root.find(".//w:tbl", ns) is not None
        assert "人工修订的第一版" in "".join(root.itertext())
        assert package.read("nectivon/transcript.md") == SAVED.encode("utf-8")
        assert not any(name.startswith("word/media/") for name in package.namelist())


def test_another_generation_does_not_change_exported_human_revision(saved_case):
    _, service, provider, document, version = saved_case
    provider.content = "NON_REAL_NEW_AI_CANDIDATE"
    new = service.generate(document["item_ids"], provider)
    assert new["id"] != document["id"]
    assert ai_review_export.markdown(service, version["id"]) == SAVED
    assert service.document(document["id"])["raw_ai"] == "NON_REAL_RAW_AI_ONLY"
    assert len(provider.prompts) == 2


def test_unpersisted_document_id_is_not_accepted_as_a_saved_version(saved_case):
    _, service, _, document, _ = saved_case
    with pytest.raises(StudyError, match="保存的复盘版本不存在"):
        ai_review_export.markdown(service, document["id"])
    with pytest.raises(StudyError, match="保存的复盘版本不存在"):
        ai_review_export.docx(service, "NON_REAL_UNSAVED_VERSION")


def test_download_names_use_readable_source_names_and_visible_version_numbers():
    for extension in ("md", "docx"):
        name = ai_review_export.filename(["电工学习.docx"], 2, extension)
        assert name == f"电工学习-复盘-第2版.{extension}"
        sanitized = ai_review_export.filename([r"C:\folder\合成:材料?.docx"], 1, extension)
        assert sanitized == f"合成材料-复盘-第1版.{extension}"
        assert ai_review_export.filename([], 1, extension) == f"复盘-第1版.{extension}"
    with pytest.raises(StudyError, match="导出格式"):
        ai_review_export.filename([], 1, "pdf")
