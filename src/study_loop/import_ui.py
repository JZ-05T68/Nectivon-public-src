"""Simple mixed-file import and reading, with exceptional corrections kept in advanced settings."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import streamlit as st

from src.voice_input_ui import render_voice_input

from .conversation_import import html_pages
from .display import render_saved_text
from .models import Draft, Page, Question, StudyError
from .runtime import launch_once, start_import_batch, submit
from .storage import StudyStore
from .structure import is_transcript, local_drafts
from .vesti import decode_text, json_pages, text_pages

FILE_KINDS = {
    "png": "image", "jpg": "image", "jpeg": "image", "docx": "docx", "pdf": "pdf",
    "txt": "txt", "md": "md", "markdown": "md", "json": "json", "html": "html", "htm": "html",
}


class UploadedMaterial(Protocol):
    """Minimal uploaded-file interface, independent of Streamlit widget internals."""

    name: str

    def getvalue(self) -> bytes:
        """Return the selected local file's bytes."""
        ...


def _page_label(page: Page, *, source_kind: str = "") -> str:
    """Use a readable page label without exposing extraction implementation details."""
    if page.number is not None:
        return f"第 {page.number} 页"
    frame = page.metadata.get("source_frame")
    if frame is not None:
        return f"第 {frame} 张图片"
    title = page.metadata.get("conversation", {}).get("title")
    return str(title) if title else "材料文字"


def _enqueue(store: StudyStore, source_id: str) -> None:
    task = submit(store.root, "study_import", {"source_id": source_id}, f"import:{source_id}")
    if task.state == "queued":
        launch_once(store.root)


def receive_upload(store: StudyStore, uploaded: UploadedMaterial, *, enqueue: bool = True) -> str:
    """Detect each file separately and validate text before queueing; keep the original bytes."""
    kind = FILE_KINDS.get(Path(uploaded.name).suffix.lower().lstrip("."))
    if kind is None:
        raise StudyError("暂不支持这种文件格式。")
    data = uploaded.getvalue()
    transcript = kind in {"txt", "md"} and is_transcript(decode_text(data))
    source = store.receive(uploaded.name, data, kind, transcript=transcript)
    if kind in {"txt", "md", "json", "html"}:
        parser = {"json": json_pages, "html": html_pages}.get(kind, text_pages)
        parser(source, data)
    if enqueue:
        _enqueue(store, source.id)
    return source.id


def render_input(store: StudyStore) -> None:
    """Accept different file formats together; a click authorizes the selected local imports."""
    uploaded = st.file_uploader(
        "添加学习材料", type=list(FILE_KINDS), accept_multiple_files=True,
        max_upload_size=64, key="sl_upload_files",
    )
    if st.button("导入", disabled=not uploaded, key="sl_receive", type="primary"):
        if len(uploaded) > 100:
            raise StudyError("一次最多导入 100 份材料。")
        source_ids = []
        for selected in uploaded:
            try:
                source_ids.append(receive_upload(store, selected, enqueue=False))
            except StudyError as exc:
                st.error(f"{Path(selected.name).name}：{exc}")
        if source_ids:
            batch = start_import_batch(store.root, source_ids)
            st.session_state["sl_import_batch_task_id"] = batch.id
    with st.expander("人工导入"):
        render_voice_input()
        with st.form("sl_manual"):
            title = st.text_input("材料名称", value="学习笔记")
            text = st.text_area("学习材料", height=180)
            transcript = st.checkbox("作为对话记录保存", key="sl_manual_dialogue")
            save = st.form_submit_button("添加文字")
        if save:
            source = store.receive(
                f"{title or '学习笔记'}.txt", text.encode("utf-8"), "manual",
                transcript=transcript or is_transcript(text),
            )
            _enqueue(store, source.id)
            st.success("已添加。")


def _verified_image(store: StudyStore, page: Page) -> bytes:
    """Read a derived image without leaking a raw filesystem exception into the UI."""
    try:
        image = store.path(page.image).read_bytes()
    except OSError as exc:
        raise StudyError("来源页图暂时无法打开。") from exc
    if hashlib.sha256(image).hexdigest() != page.image_sha256:
        raise StudyError("来源页图发生变化，暂时无法显示。")
    return image


def render_source(store: StudyStore, source_id: str, page_ids: list[str], *, prefix: str) -> None:
    """Show verified original page images, keeping fingerprints and extracted text in storage."""
    source = store.source(source_id)
    store.read_source(source)
    pages = store.referenced_pages(source_id, page_ids)
    for page in pages:
        if page.image:
            _verified_image(store, page)
    st.text(source.name)
    images = [page for page in pages if page.image]
    for page in images:
        if len(images) > 1:
            st.caption(_page_label(page, source_kind=source.kind))
        st.image(_verified_image(store, page))
    if not images:
        st.info("这份材料没有原始页图。")


def _edit_question(question: Question, prefix: str, depth: int = 0) -> None:
    """Allow an exceptional wording correction without altering hidden answer/source metadata."""
    key = f"{prefix}_{question.id}"
    label = f"修正小问 {question.number}" if depth else "修正文字"
    question.content = st.text_area(label, question.content, height=140, key=f"sl_c_{key}")
    for child in question.children:
        _edit_question(child, prefix, depth + 1)


def _read_question(question: Question) -> None:
    """Display learning content directly instead of presenting an editing questionnaire."""
    render_saved_text(question.content)
    for child in question.children:
        _read_question(child)


def render_drafts(store: StudyStore) -> None:
    """Read imported content and save explicitly without requiring field-by-field editing."""
    drafts = store.drafts()
    if not drafts:
        st.info("暂无待整理材料。")
        return
    by_source = {}
    for draft in drafts:
        by_source.setdefault(draft.source_id, draft)
    materials = list(by_source.values())
    material = st.selectbox(
        "选择材料", materials,
        format_func=lambda d: store.source(d.source_id).name,
        key="sl_draft_choice",
    )
    questions = [draft for draft in drafts if draft.source_id == material.source_id]
    if len(questions) > 1:
        selected = st.selectbox(
            "题目", range(len(questions)), format_func=lambda index: f"材料内第{index + 1}题",
            key="sl_draft_question_" + material.source_id,
        )
        draft = questions[selected]
    else:
        draft = questions[0]
    st.session_state["sl_selected_draft_id"] = draft.id
    from .ai_review_ui import ai_review_entry
    from .chat_ui import question_entry

    ai_review_entry(store, draft.id, key="sl_draft_ai_review_" + draft.id)
    question_entry(store, draft.id, key="sl_draft_chat_" + draft.id)
    _read_question(draft.question)
    working = copy.deepcopy(draft)
    with st.form(f"sl_edit_{draft.id}"):
        working.subject = st.text_input(
            "科目", working.subject, key="sl_draft_input_" + draft.id + "_subject",
        )
        save = st.form_submit_button("保存学习材料")
    if save:
        store.confirm(working, user_confirmed=True)
        st.success("已保存。")
        st.rerun()


def _selected_draft(store: StudyStore) -> Draft | None:
    drafts = store.drafts()
    identity = st.session_state.get("sl_selected_draft_id")
    return next((draft for draft in drafts if draft.id == identity), drafts[0] if drafts else None)


def render_draft_advanced(store: StudyStore, *, section: str) -> None:
    """Render only inside the single bottom advanced-settings menu."""
    draft = _selected_draft(store)
    if draft is None:
        st.info("请先选择一份待整理材料。")
        return
    if section == "source":
        render_source(store, draft.source_id, draft.page_ids, prefix="draft")
    elif section == "correction":
        working = copy.deepcopy(draft)
        with st.form(f"sl_correct_{draft.id}"):
            _edit_question(working.question, draft.id)
            save = st.form_submit_button("保存文字修正")
        if save:
            store.confirm(working, user_confirmed=True)
            st.success("已保存。")
            st.rerun()
        _manual_structure(store, draft)


def _manual_structure(store: StudyStore, draft: Draft) -> None:
    st.markdown("**重新分题 / 补充小问**")
    with st.form(f"sl_restructure_{draft.id}"):
        text = st.text_area("分题文字", height=160, help="用 1.、2. 分题，用 (1)、(2) 标记小问。")
        save = st.form_submit_button("重新分题")
    if save:
        if not text.strip():
            raise StudyError("请输入整理内容。")
        source = replace(store.source(draft.source_id), transcript=False)
        page = Page(draft.page_ids[0], source.id, None, text=text)
        for new in local_drafts(source, page):
            new.id = uuid4().hex
            new.extraction = "user_structure"
            new.subject, new.chapter = draft.subject, draft.chapter
            new.page_ids = draft.page_ids
            store.add_draft(new)
        st.rerun()


def render_model_action(store: StudyStore) -> None:
    """Offer explicit structure recognition from within advanced settings."""
    sources = store.sources()
    if not sources:
        return
    source = st.selectbox(
        "选择材料", sources, format_func=lambda s: s.name, key="sl_ai_source",
    )
    pages = store.pages(source.id)
    if not pages:
        st.info("材料正在导入。")
        return
    page = st.selectbox(
        "选择页面", pages,
        format_func=lambda selected: _page_label(selected, source_kind=source.kind),
        key="sl_ai_page",
    )
    model = st.text_input("识别模型", key="sl_ai_model")
    use_image = st.checkbox(
        "识别这张页图", value=bool(page.image), disabled=not page.image, key="sl_ai_image",
    )
    if st.button("识别所选材料", key="sl_ai_submit"):
        if not model.strip():
            raise StudyError("请填写已配置的识别模型。")
        run_id = uuid4().hex
        submit(
            store.root, "study_extract",
            {
                "source_id": source.id, "page_id": page.id, "model": model,
                "use_image": bool(page.image) and use_image, "allow_remote": True, "run_id": run_id,
            },
            f"extract:{run_id}",
        )
        launch_once(store.root)
        st.success("识别已开始。")
