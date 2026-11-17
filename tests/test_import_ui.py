"""Focused tests for the one-click document-reading page.

The page contract (V086-101/104/105/102 fixes, extended by v0.8.8 P0-3):

- the uploader really accepts multiple files and every file is processed;
- a duplicate upload produces explicit feedback instead of silence;
- the import status and the AI reading status are reported separately:
  a reading failure is presented as "资料已导入，AI 阅读未完成", never as
  an import failure;
- technical terms never leak into the user-visible surface;
- the page never invokes the dormant local OCR engine (Reviewer red line #1);
- v0.8.8: a long native-text PDF is handed to the detached background reader
  (its bytes staged to durable disk first), everything else keeps this page's
  own in-page path, and the durable panel replaces the v0.8.7 instruction to
  re-select the same files with a one-click 继续读取 (Reviewer offload #6).
"""

from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import streamlit as st

import src.runtime as runtime
from src.database import Database
from src.models import PageStatus
from tests.streamlit_doubles import StreamlitLog, install

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPORT_PAGE = PROJECT_ROOT / "pages" / "1_导入资料.py"

#: The page's single import trigger. The v0.8.8 panel adds its own control
#: buttons, so a click is now selected by label instead of "any button".
_IMPORT_LABEL = "导入并让 Agent 逐份阅读"


class Upload:
    """A minimal ``UploadedFile`` stand-in (name + bytes)."""

    def __init__(self, name: str, content: bytes = b"office-document") -> None:
        self.name = name
        self.size = len(content)
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


def _run_import_page(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    uploads: list[Any],
    service: Any,
    agent_client_class: Any,
    click_labels: tuple[str, ...] = (_IMPORT_LABEL,),
) -> StreamlitLog:
    """Run ``pages/1`` in bare mode with the runtime stubbed; return the log."""

    log = StreamlitLog(
        uploads=list(uploads),
        # The general picker is keyed (``render_file_picker``); the page's
        # image-only picker keeps its own key and must stay empty here.
        uploads_by_key={"long_read_file_picker": list(uploads)},
        click_labels=set(click_labels),
    )
    install(monkeypatch, log)
    monkeypatch.setattr(st, "stop", lambda: (_ for _ in ()).throw(SystemExit))
    monkeypatch.setattr(runtime, "application_page_image_reader", lambda: None)
    monkeypatch.setattr(runtime, "application_document_service", lambda: service)
    monkeypatch.setattr(
        runtime,
        "application_settings",
        lambda: SimpleNamespace(
            agent_readings_dir=tmp_path / "readings",
            data_dir=tmp_path / "data",
            ai_llm_model="synthetic-test-model",
        ),
    )
    monkeypatch.setattr(
        runtime, "application_database", lambda: Database(tmp_path / "import-queue.db")
    )
    monkeypatch.setattr(
        runtime,
        "application_ai_provider",
        lambda: SimpleNamespace(default_model="synthetic-test-model"),
    )
    monkeypatch.setattr(
        "src.agent.local_client.LocalDocumentAgentClient", agent_client_class
    )
    runpy.run_path(str(IMPORT_PAGE), run_name="__main__")
    return log


def test_multi_file_import_reads_each_and_reports_honestly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two new Office files: both imported and read in-page; text stays honest."""

    pages = tuple(
        SimpleNamespace(id=index, page_number=index, status=PageStatus.PENDING)
        for index in range(1, 4)
    )
    result = SimpleNamespace(
        document=SimpleNamespace(id=7, title="隔离资料", sha256="a" * 64),
        pages=pages,
        duplicate=False,
        import_record=None,
    )

    imported: list[dict[str, Any]] = []
    read_documents: list[int] = []

    class Service:
        @staticmethod
        def import_document(**kwargs: Any):
            imported.append(kwargs)
            return result

        @staticmethod
        def run_page_ocr(page_id: int) -> None:
            del page_id
            raise AssertionError("导入页不得接线本地 OCR（Reviewer 红线 #1）")

    class AgentClient:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def read_document(self, document_id: int, **kwargs: Any) -> None:
            read_documents.append(document_id)
            callback = kwargs["progress_callback"]
            for position in range(1, 4):
                callback(position, 3)

    log = _run_import_page(
        monkeypatch,
        tmp_path,
        uploads=[Upload("隔离资料.docx"), Upload("另一份.docx")],
        service=Service(),
        agent_client_class=AgentClient,
    )

    assert [item["filename"] for item in imported] == [
        "隔离资料.docx",
        "另一份.docx",
    ]
    assert read_documents == [7, 7]
    progress_texts = [text for _, text in log.progress_calls()]
    assert "上传队列处理完成。" in progress_texts
    visible_text = log.all_text()
    for hidden_term in ("embedding", "chunk", "RAG", "索引"):
        assert hidden_term not in visible_text
    # v0.8.8 execution model, stated honestly for BOTH paths: the background
    # reader survives navigation, the in-page path still pauses on leaving,
    # and completed work is never lost.
    assert "离开本页或关闭浏览器都不会中断" in visible_text
    assert "离开本页会暂停" in visible_text
    assert "不会丢失" in visible_text
    # Reviewer offload #6: the page must no longer tell the user to re-pick
    # the same files; the panel offers a one-click continuation instead.
    assert "重新选择同样的文件即可继续" not in visible_text
    assert "隔离资料.docx" in visible_text
    assert "另一份.docx" in visible_text
    # V086-212: success text must never surface as an error_message line.
    assert "Agent 已逐页读完" not in visible_text


def test_duplicate_upload_gets_explicit_feedback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """V086-105: an existing document must say so instead of staying silent."""

    existing = SimpleNamespace(
        document=SimpleNamespace(id=3, title="已有资料", sha256="b" * 64),
        pages=(),
        duplicate=True,
        import_record=None,
    )
    imported: list[dict[str, Any]] = []
    read_documents: list[int] = []

    class Service:
        @staticmethod
        def import_document(**kwargs: Any):
            imported.append(kwargs)
            return existing

        @staticmethod
        def run_page_ocr(page_id: int) -> None:
            del page_id
            raise AssertionError("duplicate import must not re-run OCR")

    class AgentClient:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def read_document(self, document_id: int, **kwargs: Any) -> None:
            read_documents.append(document_id)

    log = _run_import_page(
        monkeypatch,
        tmp_path,
        # Not a real PDF: the routing probe cannot read it, so the file stays
        # on the existing in-page path instead of being rejected.
        uploads=[Upload("已有资料.pdf", b"pdf-bytes")],
        service=Service(),
        agent_client_class=AgentClient,
    )

    assert len(imported) == 1  # import_document was invoked (dedup happens inside)
    assert read_documents == []  # duplicates never re-trigger paid reading
    assert "该资料已存在" in log.all_text()


def test_reading_failure_reports_import_and_reading_separately(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """V086-101/203: import completed + AI reading failed must be honest."""

    from src.agent_document_reader import AgentDocumentReadingError

    pages = tuple(
        SimpleNamespace(id=index, page_number=index, status=PageStatus.PENDING)
        for index in (1, 2)
    )
    result = SimpleNamespace(
        document=SimpleNamespace(id=9, title="截断资料", sha256="c" * 64),
        pages=pages,
        duplicate=False,
        import_record=None,
    )
    read_documents: list[int] = []

    class Service:
        @staticmethod
        def import_document(**kwargs: Any):
            del kwargs
            return result

        @staticmethod
        def run_page_ocr(page_id: int) -> None:
            del page_id
            raise AssertionError("导入页不得接线本地 OCR（Reviewer 红线 #1）")

    class AgentClient:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def read_document(self, document_id: int, **kwargs: Any) -> None:
            read_documents.append(document_id)
            raise AgentDocumentReadingError(
                "AI 阅读未全部完成：成功 1/2 页，失败页码：2。"
                "已成功页面的阅读结果不会丢失，可稍后重试失败页。"
            )

    log = _run_import_page(
        monkeypatch,
        tmp_path,
        uploads=[Upload("截断资料.pdf", b"pdf-bytes")],
        service=Service(),
        agent_client_class=AgentClient,
    )

    assert read_documents == [9]
    visible = log.all_text()
    assert "资料已导入" in visible
    assert "AI 阅读未全部完成" in visible
    # The misleading wording from Wave 1 must be gone.
    assert "这份资料还没有读完" not in visible
    # v0.8.8: a row whose document exists but whose reading was cut is offered
    # as a one-click background continuation, not a re-selection.
    assert "只剩 AI 阅读没有完成" in visible
    assert "继续读取" in log.button_labels()
    resume = [
        call
        for call in log.calls_of("button")
        if call.kwargs.get("key") == "long_read_resume_all"
    ]
    # The one-click resume must be enabled, not a greyed-out placeholder.
    assert resume and resume[0].kwargs.get("disabled") is False


def test_long_native_text_pdf_is_routed_to_the_background_reader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """v0.8.8 routing: a real long text-layer PDF never runs in-page."""

    from src.long_read_ui import WORKER_BATCH_PREFIX, staging_dir_for
    from src.long_read_worker import staging_path
    from src.pdf_service import _load_pymupdf

    pymupdf = _load_pymupdf()
    pdf_path = tmp_path / "courseware.pdf"
    source = pymupdf.open()
    for index in range(24):
        page = source.new_page()
        page.insert_text((72, 72), f"第{index + 1}页 液压传动 概述 内容 填充 填充 填充")
    source.save(pdf_path)
    source.close()
    content = pdf_path.read_bytes()

    spawned: list[dict[str, Any]] = []

    class Service:
        @staticmethod
        def import_document(**kwargs: Any):
            del kwargs
            raise AssertionError("长文档 PDF 不得在页面内同步导入")

        @staticmethod
        def run_page_ocr(page_id: int) -> None:
            del page_id
            raise AssertionError("导入页不得接线本地 OCR（Reviewer 红线 #1）")

    class AgentClient:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def read_document(self, document_id: int, **kwargs: Any) -> None:
            del document_id, kwargs
            raise AssertionError("长文档 PDF 不得在页面内同步阅读")

    monkeypatch.setattr(
        "src.long_read_ui.spawn_worker", lambda **kwargs: spawned.append(kwargs) or 4321
    )
    data_dir = tmp_path / "data"
    log = _run_import_page(
        monkeypatch,
        tmp_path,
        uploads=[Upload("液压课件.pdf", content)],
        service=Service(),
        agent_client_class=AgentClient,
    )

    # The detached worker was asked for exactly one worker-owned batch, so the
    # in-page loop and the worker can never fight over the same queue rows.
    assert len(spawned) == 1
    batch_ids = spawned[0]["batches"]
    assert len(batch_ids) == 1
    assert batch_ids[0].startswith(WORKER_BATCH_PREFIX)
    # The uploaded bytes were staged to durable disk BEFORE the spawn, so the
    # worker (another process) can import them and a resume needs no re-pick.
    staged = staging_path(staging_dir_for(data_dir), batch_ids[0], "液压课件.pdf")
    assert staged.is_file(), f"暂存文件不存在：{staged}"
    assert staged.read_bytes() == content
    visible = log.all_text()
    assert "后台独立进程" in visible
    assert "液压课件.pdf" in visible
    # The queued row is listed, so the user can see the file is waiting.
    assert "等待导入" in visible


def test_uploader_accepts_pdf_word_powerpoint_and_multiple_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploader_calls: list[dict[str, Any]] = []

    def file_uploader(*args: Any, **kwargs: Any) -> None:
        del args
        uploader_calls.append(kwargs)
        return None

    monkeypatch.setattr(st, "set_page_config", lambda **kwargs: None)
    monkeypatch.setattr(st, "title", lambda *args, **kwargs: None)
    monkeypatch.setattr(st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(st, "file_uploader", file_uploader)
    monkeypatch.setattr(
        st,
        "camera_input",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("导入页不得请求摄像头权限")
        ),
    )
    monkeypatch.setattr(st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(st, "stop", lambda: (_ for _ in ()).throw(SystemExit))
    # Bare-mode Streamlit session state is a process-wide singleton that
    # earlier tests in this module may have populated; isolate it.
    monkeypatch.setattr(st, "session_state", {})

    with pytest.raises(SystemExit):
        runpy.run_path(str(IMPORT_PAGE), run_name="__main__")

    # General PDF/Office/image and dedicated image-only uploaders remain;
    # rendering either must not invoke camera_input.
    assert uploader_calls, "the page must render its uploaders before stopping"
    general = uploader_calls[0]
    assert general["type"] == [
        "pdf",
        "doc",
        "docx",
        "ppt",
        "pptx",
        "jpg",
        "jpeg",
        "png",
        "webp",
    ]
    assert general["accept_multiple_files"] is True
    image_only = [
        call for call in uploader_calls[1:] if call.get("type") == ["jpg", "jpeg", "png", "webp"]
    ]
    assert image_only, "the dedicated image-only uploader must exist"
    assert image_only[0]["accept_multiple_files"] is True
