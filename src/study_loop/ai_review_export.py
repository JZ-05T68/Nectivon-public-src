"""Local exports of an exact saved AI-review revision, never the mutable working draft."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from .models import StudyError

if TYPE_CHECKING:
    from .ai_review import AIReviewService


def markdown(service: AIReviewService, version_id: str) -> str:
    """Read the selected immutable version verbatim, without changing review storage."""
    return service.version(version_id)["content"]


def docx(service: AIReviewService, version_id: str) -> bytes:
    """Reuse native Office Math rendering and embed the same exact saved Markdown."""
    from .word_math import markdown_docx

    return markdown_docx(markdown(service, version_id))


def filename(source_names: Iterable[str], number: int, extension: str) -> str:
    """Use source filenames and the visible version number instead of internal identifiers."""
    if extension not in {"md", "docx"}:
        raise StudyError("请选择 Markdown 或 Word 导出格式。")
    names = []
    for source_name in source_names:
        leaf = re.split(r"[\\/]", source_name)[-1]
        stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", Path(leaf).stem).strip(" .")
        if stem and stem not in names:
            names.append(stem)
    title = "、".join(names)[:70].rstrip(" .")
    return f"{title + '-' if title else ''}复盘-第{number}版.{extension}"
