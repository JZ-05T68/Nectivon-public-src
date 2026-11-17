"""Bounded, inert adapters for explicitly selected conversation JSON and HTML.

These are structural adapters, not promises about every vendor export version.
Every derived segment points back to the retained original; no link is resolved.
"""

from __future__ import annotations

import hashlib
import json
from html.parser import HTMLParser
from typing import Any

from .models import Page, Source, StudyError

MAX_TEXT = 200_000
MAX_SEGMENTS = 5000
CAUTION = "历史记录待人工核对；消息角色不代表内容正确，附件描述不等于附件原件。"


def content_pages(source: Source, text: str, metadata: dict, identity: str) -> list[Page]:
    """Split without dropping characters, with offsets into the decoded record text."""
    if not text.strip():
        raise StudyError("来源记录没有可显示的文字，请核对原件。")
    chunks = range(0, len(text), MAX_TEXT)
    if len(chunks) > MAX_SEGMENTS:
        raise StudyError("来源段数量过多，请分批导入；原件已保留。")
    pages = []
    for start in chunks:
        end = min(start + MAX_TEXT, len(text))
        digest = hashlib.sha256(
            f"conversation-v1:{source.id}:{identity}:{start}".encode()
        ).hexdigest()
        pages.append(
            Page(
                digest,
                source.id,
                None,
                text=text[start:end],
                metadata={
                    **metadata,
                    "character_start": start,
                    "character_end": end,
                    "character_basis": metadata.get(
                        "character_basis", "解码后的本条正文，0 起始、右边界不包含"
                    ),
                },
            )
        )
    return pages


def _objects(value: Any) -> list[dict]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise StudyError("对话消息结构无法识别；请改用 TXT / Markdown，原件已保留。")
    if len(value) > MAX_SEGMENTS:
        raise StudyError("对话消息过多，请按会话分批导出。")
    return value


def _body(record: dict) -> str:
    """Read common text fields; preserve unsupported rich content as inert JSON."""
    value = record.get("content", record.get("content_text", record.get("text")))
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, dict) and isinstance(value.get("parts"), list):
        parts = value["parts"]
        if parts and all(isinstance(part, str) for part in parts) and any(parts):
            return "\n".join(parts)
    return json.dumps(value if value is not None else record, ensure_ascii=False, indent=2)


def generic_json_pages(source: Source, raw: Any) -> list[Page]:
    """Accept explicit messages arrays and mapping nodes, retaining all branches.

    Mapping iteration is export order, not an invented conversation chronology.
    Unknown structures fail explicitly rather than becoming guessed dialogue.
    """
    pages: list[Page] = []

    def add(record: dict, pointer: str, context: dict, **extra: Any) -> None:
        author = record.get("author")
        role = record.get("role", author.get("role") if isinstance(author, dict) else None)
        if role is not None and not isinstance(role, str):
            raise StudyError("对话角色字段无效；原件已保留，请核对导出结构。")
        metadata = {
            "format": "conversation_json_v1",
            "record_kind": "message",
            "pointer": pointer,
            "conversation": context,
            "role": role,
            "created_at": record.get(
                "created_at", record.get("create_time", record.get("timestamp"))
            ),
            "message_id": record.get("id"),
            "warning": CAUTION,
            **extra,
        }
        pages.extend(content_pages(source, _body(record), metadata, pointer))
        if len(pages) > MAX_SEGMENTS:
            raise StudyError("对话来源段过多，请按会话分批导出；原件已保留。")

    def conversation(value: dict, pointer: str) -> None:
        context = {
            key: value[key]
            for key in (
                "id",
                "conversation_id",
                "title",
                "name",
                "platform",
                "create_time",
                "current_node",
            )
            if key in value
        }
        if "messages" in value:
            for i, record in enumerate(_objects(value["messages"])):
                add(record, f"{pointer}/messages/{i}", context)
            return
        mapping = value.get("mapping")
        if not isinstance(mapping, dict) or len(mapping) > MAX_SEGMENTS:
            raise StudyError("JSON 对话结构未识别；支持 messages 数组或 mapping 节点，请改用 TXT。")
        for key, node in mapping.items():
            if not isinstance(node, dict):
                raise StudyError("对话 mapping 节点无效，原件已保留。")
            record = node.get("message")
            if record is None:
                continue  # Structural roots remain in the immutable original.
            if not isinstance(record, dict):
                raise StudyError("对话 mapping 消息无效，原件已保留。")
            escaped = key.replace("~", "~0").replace("/", "~1")
            add(
                record,
                f"{pointer}/mapping/{escaped}/message",
                context,
                branch={
                    "node": key,
                    "parent": node.get("parent"),
                    "children": node.get("children", []),
                },
                warning=CAUTION + " 已保留所有分支；显示顺序为导出顺序，未自动选择最终分支。",
            )

    if isinstance(raw, dict) and isinstance(raw.get("conversations"), list):
        for i, value in enumerate(_objects(raw["conversations"])):
            conversation(value, f"/conversations/{i}")
    elif isinstance(raw, dict):
        conversation(raw, "")
    elif isinstance(raw, list):
        values = _objects(raw)
        if all("messages" in value or "mapping" in value for value in values):
            for i, value in enumerate(values):
                conversation(value, f"/{i}")
        elif all("role" in value or isinstance(value.get("author"), dict) for value in values):
            for i, value in enumerate(values):
                add(value, f"/{i}", {})
        else:
            raise StudyError("JSON 列表未识别为带角色的消息或会话列表；请改用 TXT / Markdown。")
    else:
        raise StudyError("JSON 不是已支持的消息或会话结构。")
    if not pages:
        raise StudyError("对话导出中没有可整理的消息。")
    return pages


class _InertHTML(HTMLParser):
    """Extract local display text without rendering, scripting or resource fetching."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hidden: list[str] = []
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"head", "script", "style", "template"}:
            self.hidden.append(tag)
        if not self.hidden and tag in {"br", "p", "div", "li", "pre", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.hidden:
            if tag == self.hidden[-1]:
                self.hidden.pop()
            return
        if tag in {"p", "div", "li", "pre", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def html_pages(source: Source, data: bytes) -> list[Page]:
    """Extract HTML text; do not pretend transformed text is a byte-exact source quote."""
    from .vesti import decode_text

    parser = _InertHTML()
    try:
        parser.feed(decode_text(data))
        parser.close()
    except (ValueError, RecursionError) as exc:
        raise StudyError("HTML 文字无法安全解析；原件已保留，请改用 TXT。") from exc
    text = "".join(parser.parts)
    return content_pages(
        source,
        text,
        {
            "format": "html_text_v1",
            "record_kind": "transcript" if source.transcript else "text",
            "warning": "HTML 仅提取文字，不执行脚本、不下载附件；角色、时间及分段须人工核对。",
            "text_basis": "HTML 标签与脚本已排除、实体已解码；精确依据为可下载的原始 HTML。",
        },
        "html",
    )
