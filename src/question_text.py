"""Pure text handling for platform page candidates; no learning persistence."""

from __future__ import annotations

import re

_FIRST_SUBQUESTION_MARKER = re.compile(
    r"(?m)(?:^|\n|(?<=[：:]))\s*"
    r"(?:\d{1,3}\s*[.．、:：]?\s*)?"
    r"(?:[（(]\s*\d+(?:\.\d+)?\s*分\s*[）)]\s*)?"
    r"[（(]\s*[0-9一二三四五六七八九十]+\s*[）)]"
)
_QUESTION_METADATA_ONLY = re.compile(
    r"(?:\d{1,3}\s*[.．、:：]?\s*)?"
    r"(?:[（(]\s*\d+(?:\.\d+)?\s*分\s*[）)]\s*)?"
)


def shared_parent_prompt(prompt: str) -> str:
    """Keep existing common-text extraction without old question-tree business code."""
    clean_prompt = str(prompt or "").strip()
    marker = _FIRST_SUBQUESTION_MARKER.search(clean_prompt)
    shared = clean_prompt if marker is None else clean_prompt[: marker.start()].strip()
    return "" if _QUESTION_METADATA_ONLY.fullmatch(shared) else shared
