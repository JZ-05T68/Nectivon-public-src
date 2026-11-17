"""Read user-selected VESTI exports, preserving roles and original JSON pointers.

Contracts audited at VESTI e6cd072db804bda24f972d9289cf77e9fa065d5b.
No browser database access, link fetching, HTML rendering or attachment resolution.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .models import Page, Source, StudyError

MAX_SEGMENTS = 5000
MAX_TEXT = 200_000
CAUTION = (
    "Vesti 导出为历史材料，不代表答案已核对或用户已掌握；"
    "引用、附件描述和制品摘录不等于完整原件，缺失原件需主动补充。"
)


def decode_text(data: bytes) -> str:
    """Accept common local text encodings without replacing undecodable bytes."""
    encodings = ("utf-8-sig", "gb18030")
    if data.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        encodings = ("utf-32",)
    elif data.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings = ("utf-16",)
    for encoding in encodings:
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if any(ord(char) < 32 and char not in "\t\n\r\f" for char in text):
            raise StudyError("文本含二进制控制字符，请核对文件类型或另存为 UTF-8。")
        return text
    raise StudyError("文本编码无法识别，请另存为 UTF-8 后导入。")


def _objects(value: Any, label: str) -> list[dict]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise StudyError(f"Vesti {label} 结构无法识别；原件保留，请改用 TXT / Markdown。")
    if len(value) > MAX_SEGMENTS:
        raise StudyError("Vesti 导出记录过多，请在 Vesti 中选择较小范围重新导出。")
    return value


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise StudyError("JSON 中存在重复字段，无法安全判断来源；请重新导出。")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise StudyError(f"JSON 包含非标准数值 {value}，请重新导出。")


def _validate_unicode(value: Any) -> None:
    """Reject lone escaped surrogates before identifiers or SQLite require UTF-8."""
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, str):
            try:
                current.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise StudyError(
                    "JSON 含孤立代理字符，无法完整保存为 UTF-8；请重新导出，原件已保留。"
                ) from exc
        elif isinstance(current, dict):
            pending.extend(current.keys())
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)


def json_pages(source: Source, data: bytes) -> list[Page]:
    """Support Vesti contracts and bounded, explicitly shaped conversation exports."""
    from .conversation_import import content_pages, generic_json_pages

    if len(data) > 20 * 1024 * 1024:
        raise StudyError("JSON 超过 20 MiB，请按对话分批导出；原件已保留。")
    try:
        raw = json.loads(
            decode_text(data), object_pairs_hook=_unique_object, parse_constant=_invalid_constant
        )
    except StudyError:
        raise
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise StudyError("JSON 无法完整解析；请核对导出格式，原件已保留。") from exc
    _validate_unicode(raw)
    if not isinstance(raw, dict):
        return generic_json_pages(source, raw)
    archive = raw.get("schema_version") == "vesti_export.v1"
    selected = (
        "schema_version" not in raw
        and "exported_at" in raw
        and isinstance(raw.get("content_mode"), str)
        and raw.get("content_mode") in {"full", "compact", "summary"}
        and isinstance(raw.get("conversations"), list)
    )
    if not archive and not selected:
        if "schema_version" in raw:
            raise StudyError("JSON 版本未识别。请改用 TXT / Markdown，勿手工改版本号。")
        return generic_json_pages(source, raw)
    body = raw.get("data") if archive else raw
    if not isinstance(body, dict):
        raise StudyError("Vesti 档案缺少 data 对象。")
    conversations = _objects(body.get("conversations"), "conversations")
    pages: list[Page] = []
    mode = "full" if archive else raw["content_mode"]
    common = {
        "format": "vesti_export.v1" if archive else "vesti_selected_threads",
        "exported_at": raw.get("exported_at"),
        "content_mode": mode,
        "timezone": raw.get("timezone"),
        "app_version": raw.get("app_version"),
        "compact_variant": raw.get("compact_variant"),
        "warning": CAUTION,
    }

    def add(pointer: str, value: dict, kind: str, conversation: dict | None = None) -> None:
        # Unknown fields and rich AST remain in the immutable original, addressed by pointer.
        body_key = {
            "message": "content_text",
            "annotation": "content_text",
            "summary": "compressed_content" if selected else "content",
            "compact": "compressed_content",
            "weekly_report": "content",
        }.get(kind)
        body_text = value.get(body_key) if body_key else None
        text = body_text if isinstance(body_text, str) and body_text.strip() else _json(value)
        if kind == "message":
            if value.get("role") is not None and not isinstance(value["role"], str):
                raise StudyError("Vesti 消息角色字段无效，原件已保留。")
            role = {"user": "用户原话", "assistant": "AI 原话"}.get(value.get("role"), "角色不明")
            text = f"【{role}；历史记录，待人工核对】\n{text}"
            for key, label in (
                ("citations", "原记录引用"),
                ("attachments", "附件描述"),
                ("artifacts", "制品记录（可能仅有摘录）"),
            ):
                if value.get(key):
                    text += f"\n\n{label}：\n{_json(value[key])}"
        if len(pages) >= MAX_SEGMENTS:
            raise StudyError("Vesti 单条内容过长或条数过多，请分批导出；没有截断或丢弃原件。")
        metadata = {
            **common,
            "pointer": pointer,
            "record_kind": kind,
            "conversation": conversation or {},
            "role": value.get("role"),
            "created_at": value.get("created_at_iso", value.get("created_at")),
            "message_id": value.get("id") if kind == "message" else None,
        }
        identity = hashlib.sha256(f"vesti-v1:{source.id}:{pointer}".encode()).hexdigest()
        if len(text) > MAX_TEXT:
            pages.extend(content_pages(source, text, {
                **metadata,
                "character_basis": "分块前的展示文本（可能含角色提示与附件描述），"
                "0 起始、右边界不包含；不是原 JSON 字符偏移",
            }, identity))
            if len(pages) > MAX_SEGMENTS:
                raise StudyError("Vesti 来源段过多，请分批导出；原件已保留。")
        else:
            pages.append(Page(identity, source.id, None, text=text, metadata=metadata))

    lookup: dict[str, dict] = {}
    for index, conversation in enumerate(conversations):
        identifier = conversation.get("id")
        if type(identifier) not in {int, str} or str(identifier) in lookup:
            raise StudyError("Vesti 对话标识缺失或重复，无法安全关联消息。")
        context = {
            k: v
            for k, v in conversation.items()
            if k not in {"messages", "compressed_content", "compression"}
        }
        lookup[str(identifier)] = context
        pointer = f"/data/conversations/{index}" if archive else f"/conversations/{index}"
        record = {k: v for k, v in conversation.items() if k != "messages"}
        add(pointer, record, "conversation" if mode == "full" else mode, context)
        if not archive and mode == "full":
            messages = _objects(conversation.get("messages"), "messages")
            for message_index, message in enumerate(messages):
                add(f"{pointer}/messages/{message_index}", message, "message", context)
    if archive:
        for key, kind in (
            ("messages", "message"),
            ("summaries", "summary"),
            ("annotations", "annotation"),
            ("weeklyReports", "weekly_report"),
        ):
            for index, record in enumerate(_objects(body.get(key, []), key)):
                conversation_id = record.get("conversation_id", record.get("conversationId"))
                context = lookup.get(str(conversation_id))
                page_start = len(pages)
                add(f"/data/{key}/{index}", record, kind, context)
                if kind in {"message", "annotation"} and context is None:
                    for page in pages[page_start:]:
                        page.metadata["warning"] += " 本记录缺少匹配的对话，关联信息不足。"
    if not pages:
        raise StudyError("导出中没有可整理记录。")
    return pages


def text_pages(source: Source, data: bytes) -> list[Page]:
    """Keep TXT/Markdown verbatim; split only recognized exporter thread headers.

    Human-readable exports are ambiguous when message bodies imitate headers.
    Boundaries are suggestions with line references; raw bytes remain authoritative.
    """
    from .conversation_import import content_pages

    text = decode_text(data)
    vesti = bool(re.search(r"(?i)vesti.*(?:export|思想档案导出)", text[:1000]))
    lines = text.splitlines(keepends=True)
    mode_match = re.search(r"(?im)^\*{0,2}Mode:\*{0,2}\s*(full|compact|summary)\s*$", text[:1500])
    mode = mode_match[1] if mode_match else "unknown"
    starts: list[int] = []
    if vesti:
        pattern = re.compile(r"^## \[Thread \d+\]|^## Weekly Reports\s*$")
        starts = [i for i, line in enumerate(lines) if pattern.match(line)]
        if not starts:
            # Selected-thread exports use title + Platform metadata in both TXT and MD.
            starts = [
                i
                for i in range(len(lines) - 2)
                if re.match(r"^(?:## )?\d+\. .+", lines[i])
                and any(
                    re.match(r"^\s*(?:- \*\*)?Platform:", line) for line in lines[i + 1 : i + 3]
                )
            ]
    bounds = sorted(set([0, *starts, len(lines)]))
    result = []
    for index, (start, end) in enumerate(zip(bounds, bounds[1:], strict=False)):
        block = "".join(lines[start:end])
        if not block.strip():
            continue
        if len(result) >= MAX_SEGMENTS:
            raise StudyError("文本段过长或过多，请分批导入；原件没有截断。")
        metadata = {
            "format": "vesti_text" if vesti else source.kind,
            "line_start": start + 1,
            "line_end": end,
            "content_mode": mode,
            "record_kind": "transcript" if vesti or source.transcript else "text",
        }
        if vesti:
            metadata["warning"] = CAUTION + " 文本分段仅为候选，请对照原件核对边界。"
        if len(block) > MAX_TEXT:
            # Existing short-page IDs/payloads remain stable for idempotent retries.
            metadata["warning"] = metadata.get("warning", "") + (
                " 长段已按字符范围分块，可能跨句；请对照原件和相邻段核对。"
            )
            result.extend(content_pages(source, block, metadata, f"text:{index}"))
            if len(result) > MAX_SEGMENTS:
                raise StudyError("文本段过多，请分批导入；原件没有截断。")
        else:
            result.append(
                Page(f"{source.id}:t{index}", source.id, None, text=block, metadata=metadata)
            )
    if not result:
        raise StudyError("文本内容为空。")
    return result
