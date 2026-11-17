"""Thin bindings to the audited public candidate APIs; never edit component repositories."""

from __future__ import annotations

import hashlib
import io
import os
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from PIL import Image

from .conversation_import import html_pages
from .models import Draft, Page, Source, StudyError
from .storage import StudyStore, encode
from .structure import local_drafts
from .vesti import json_pages, text_pages

if TYPE_CHECKING:
    from nectivon_import import ImportCore, Receipt


def _publish_frame(target: Path, data: bytes) -> None:
    """Publish a complete derived frame without replacing an existing asset."""
    staged = target.with_name(f".frame-{uuid4().hex}.tmp")
    try:
        with staged.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staged, target)
        except FileExistsError:
            if target.read_bytes() != data:
                raise StudyError("相机照片派生帧与原件不一致，未覆盖；请保留现场。") from None
    finally:
        # This is only the temporary file created by this call, never a source or page.
        staged.unlink(missing_ok=True)


def _submit_file(
    core: ImportCore,
    store: StudyStore,
    source: Source,
    data: bytes,
) -> tuple[Receipt, dict[int, str]]:
    """Adapt camera MPO frames to Import Core's existing photo-sequence contract."""
    if source.kind != "image":
        return core.submit_stream(source.name, io.BytesIO(data)), {}
    try:
        picture = Image.open(io.BytesIO(data))
    except (OSError, Image.DecompressionBombError) as exc:
        raise StudyError("图片无法读取或像素超过安全上限，原件已保留；请人工核对。") from exc
    with picture:
        if picture.format != "MPO":
            return core.submit_stream(source.name, io.BytesIO(data)), {}
        if not 1 <= picture.n_frames <= 200:
            raise StudyError("相机照片帧数超过 200，原件已保留；请人工分批转换。")
        directory = store.path(f"imports/{source.id}-mpo")
        directory.mkdir(parents=True, exist_ok=True)
        paths, hashes = [], {}
        pixels = 0
        for index in range(picture.n_frames):
            picture.seek(index)
            pixels += picture.width * picture.height
            if pixels > 100_000_000:
                raise StudyError("相机照片累计像素超过本次导入上限，原件已保留；请人工分批转换。")
            output = io.BytesIO()
            picture.convert("RGB").save(output, format="PNG")
            frame = output.getvalue()
            target = store.path(f"imports/{source.id}-mpo/frame_{index + 1:04d}.png")
            _publish_frame(target, frame)
            paths.append(target)
            hashes[index + 1] = hashlib.sha256(frame).hexdigest()
    return core.submit_photos(paths), hashes


def import_source(store: StudyStore, source_id: str) -> dict:
    """Use Import Core for files; TXT/manual/voice converge on the same page/draft flow."""
    source = store.source(source_id)
    data = store.read_source(source)
    if source.kind in {"txt", "md", "json", "html", "manual"}:
        parser = {"json": json_pages, "html": html_pages}.get(source.kind, text_pages)
        parsed = parser(source, data)
        for page in parsed:
            store.put_page(page)
    elif source.kind == "voice":
        if not store.pages(source.id):
            raise StudyError("录音需先经本地识别、人工编辑并确认文字。")
    else:
        from nectivon_import import ImportCore

        core = ImportCore(store.path("imports"))
        receipt, frame_hashes = _submit_file(core, store, source, data)
        state = core.status(receipt.job_id)
        if state.state in {"failed", "interrupted", "paused"}:
            core.control(
                receipt.job_id, "resume"
            )  # This handler runs only after an explicit start.
        state = core.run(receipt.job_id)
        if state.state != "completed":
            raise StudyError("文件导入尚未完整完成，原件已保留；请核对任务后主动继续。")
        core.verify(receipt.job_id)
        if state.total > 200:
            raise StudyError("材料超过 200 页，原件和导入页已保留；请分批整理。")
        for imported in core.pages(receipt.job_id):
            # Only public Page fields are used, not the component's SQLite/converted.pdf layout.
            image = imported.image_path.resolve().relative_to(store.root).as_posix()
            page = Page(
                f"{source.id}:p{imported.page_number}",
                source.id,
                None if frame_hashes else imported.source_page_number,
                imported.text,
                image,
                imported.image_sha256,
                imported.needs_review,
                receipt.job_id,
                metadata=(
                    {
                        "source_format": "MPO",
                        "source_frame": imported.page_number,
                        "source_sha256": source.sha256,
                        "derived_source_sha256": frame_hashes[imported.page_number],
                        "warning": "相机 MPO 多帧照片；此段为原件的第 "
                        f"{imported.page_number} 帧，派生 PNG 不替代原始照片。",
                    }
                    if frame_hashes
                    else {}
                ),
            )
            expected_hash = frame_hashes.get(imported.page_number, source.sha256)
            if imported.source_sha256 != expected_hash:
                raise StudyError("Import Core 来源摘要与收件记录不一致。")
            store.put_page(page)
    count = 0
    if source.kind == "docx":
        from .word_import import structured_word, supplement

        page = supplement(store, source, data)
        for draft in structured_word(source, page, [p.id for p in store.pages(source.id)]):
            store.add_draft(draft)
            count += 1
        return {"source_id": source.id, "draft_count": count}
    for page in store.pages(source.id):
        for draft in local_drafts(source, page):
            store.add_draft(draft)
            count += 1
    return {"source_id": source.id, "draft_count": count}


def search(store: StudyStore, query: str) -> list[str]:
    """Rebuild derived search rows from authoritative confirmed items before a user query."""
    from nectivon_search_core import Page as SearchPage
    from nectivon_search_core import SearchIndex

    snapshots: dict[str, list[SearchPage]] = defaultdict(list)
    owners: dict[tuple[str, str], str] = {}
    for item in store.items():
        fields = [item.subject, item.chapter, item.verification_note]
        pending = [item.question]
        while pending:
            node = pending.pop()
            fields.extend(
                [
                    node.number,
                    node.content,
                    *node.knowledge,
                    node.reference_answer,
                    node.reference_analysis,
                    node.obstacle,
                    node.obstacle_evidence,
                ]
            )
            pending.extend(reversed(node.children))
        content = "\n".join(fields)
        # The existing core accepts 200k/page and queries up to 512 characters.
        # Overlap keeps boundary phrases searchable; no saved text is truncated.
        starts = (0,) if len(content) <= 200_000 else range(0, len(content), 200_000 - 512)
        for part, start in enumerate(starts):
            fragment = item.id if len(content) <= 200_000 else f"{item.id}:part:{part}"
            snapshots[item.source_id].append(
                SearchPage(item.source_id, fragment, content[start : start + 200_000])
            )
            owners[item.source_id, fragment] = item.id
    with SearchIndex(store.path("search.sqlite3")) as index:
        for source_id, pages in snapshots.items():
            # Replace only this derived source snapshot, leaving original items untouched.
            index.replace_document(source_id, pages)
        hits: list[str] = []
        seen: set[str] = set()
        offset = 0
        while offset <= 1_000_000:
            batch = index.search(query, limit=100, offset=offset)
            for hit in batch:
                item_id = owners.get((hit.document_id, hit.page_id))
                if item_id is not None and item_id not in seen:
                    hits.append(item_id)
                    seen.add(item_id)
            if len(batch) < 100:
                return hits
            offset += 100
        raise StudyError("搜索结果过多，请缩小关键词范围。")


def _selected_items(store: StudyStore, item_ids: list[str]) -> list[Draft]:
    """Reject stale or mixed valid/missing selections instead of reporting partial success."""
    if not item_ids or len(item_ids) != len(set(item_ids)):
        raise StudyError("请选择非空、无重复的已保存材料范围。")
    items = [store.item(identity) for identity in item_ids]
    for item in items:
        pages = {page.id for page in store.pages(item.source_id)}
        if not item.page_ids or not set(item.page_ids) <= pages:
            raise StudyError("所选材料的来源段缺失，停止核验或导出。")
    return items


def pdf_sources(store: StudyStore, items: list[Draft]) -> list:
    """Build trusted SourcePage mappings only for original PDFs, never guessed Office paths."""
    from nectivon_evidence import SourcePage

    pages = {}
    for item in items:
        source = store.source(item.source_id)
        if source.kind != "pdf":
            continue
        for page in store.pages(source.id):
            if page.id in item.page_ids:
                pages[page.id] = SourcePage(
                    source.id,
                    page.id,
                    page.number,
                    store.path(source.blob),
                    source.sha256,
                )
    return list(pages.values())


def verify_sources(store: StudyStore, item_ids: list[str]) -> dict:
    """Run in a process main thread. Byte integrity and PDF PAGE verification stay distinct."""
    from nectivon_evidence import Catalog, EvidenceCore, Registry

    items = _selected_items(store, item_ids)
    for source_id in {item.source_id for item in items}:
        store.read_source(store.source(source_id))
    mappings = pdf_sources(store, items)
    references = []
    with Registry(store.path("evidence.sqlite3")) as registry:
        core = EvidenceCore(Catalog(mappings), registry)
        for page in mappings:
            citation = core.capture_page(page.document_id, page.page_id)
            location = core.verify(citation)
            references.append({"citation": citation, "page_id": location.page_id})
    return {
        "pdf_page_references": references,
        "byte_checked_sources": len({i.source_id for i in items}),
        "scope": "PDF物理页可定位；其他格式仅核对原始字节，不证明答案正确。",
    }


def export_review(store: StudyStore, item_ids: list[str], export_id: str) -> dict:
    """Reuse Export Core for real PDF evidence; plain text summaries remain UI downloads."""
    from nectivon_evidence import Catalog, EvidenceCore, Registry
    from nectivon_export_core import Attachment, Evidence, export_package, verify_package

    from .review import summary

    items = _selected_items(store, item_ids)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", export_id):
        raise StudyError("导出标识无效。")
    for source_id in {item.source_id for item in items}:
        store.read_source(store.source(source_id))
    mappings = pdf_sources(store, items)
    if not mappings:
        raise StudyError("当前没有 PDF 物理页证据，请使用界面的 Markdown 总结下载。")
    destination = store.path(f"exports/{export_id}")
    with Registry(store.path("evidence.sqlite3")) as registry:
        core = EvidenceCore(Catalog(mappings), registry)
        if destination.exists():
            verify_package(destination, verifier=core)
            selection = store.path(f"exports/{export_id}/attachments/selection.json")
            if not selection.is_file() or selection.read_text(encoding="utf-8") != encode(item_ids):
                raise StudyError("现有证据包与本次所选材料不一致，未覆盖；请创建新导出任务。")
            return {"directory": str(destination)}
        evidence = []
        for page in mappings:
            token = core.capture_page(page.document_id, page.page_id)
            evidence.append(
                Evidence(
                    token,
                    "PAGE",
                    page.document_id,
                    page.page_id,
                    page.page_number,
                    page.document_sha256,
                    source_description=store.source(page.document_id).name,
                )
            )
        records = [
            asdict(store.source(identity)) for identity in {item.source_id for item in items}
        ]
        export_package(
            destination,
            "学习复盘原始材料引用",
            evidence,
            [
                Attachment("review-summary.md", summary(store, item_ids).encode("utf-8")),
                Attachment("source-records.json", encode(records).encode("utf-8")),
                Attachment("selection.json", encode(item_ids).encode("utf-8")),
            ],
            verifier=core,
        )
    return {"directory": str(destination)}
