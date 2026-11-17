"""Check label membership without claiming that a cited conclusion is correct."""

from __future__ import annotations

import re

_INTERNAL_KEY = r"[SMR][0-9a-fA-F]{8,64}"
_INTERNAL_REFERENCE = re.compile(
    r"\[" + _INTERNAL_KEY + r"\]|(?<![A-Za-z0-9_])" + _INTERNAL_KEY + r"(?![A-Za-z0-9_])"
)
_OPAQUE_ID = re.compile(
    r"(?:[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}|"
    r"[0-9a-fA-F]{6,16}(?:\.{3}|…)[0-9a-fA-F]{2,12}|"
    r"(?=[0-9a-fA-F]*[a-fA-F])[0-9a-fA-F]{32,64})"
)


def _protect_authored_content(text: str) -> tuple[str, list[str]]:
    """Keep genuine fenced code, math and ordinary inline code byte-for-byte intact."""
    saved = []

    def protect(value: str) -> str:
        saved.append(value)
        return f"\ue000{len(saved) - 1}\ue001"

    opening = re.compile(r"(?m)^ {0,3}(`{3,}|~{3,})([^\n]*)\n")
    pieces, position = [], 0
    while match := opening.search(text, position):
        close = re.search(
            r"(?m)^ {0,3}" + re.escape(match[1][0]) + "{" + str(len(match[1]))
            + r",}[ \t]*(?:\n|$)", text[match.end():],
        )
        end = match.end() + close.end() if close else len(text)
        body = text[match.end():match.end() + close.start()] if close else text[match.end():]
        pieces.append(text[position:match.start()])
        if match[2].strip().casefold() == "label" and not body.strip():
            pieces.append("来源信息暂不可用\n")
        else:
            pieces.append(protect(text[match.start():end]))
        position = end
    pieces.append(text[position:])
    text = "".join(pieces)
    inline_code = re.compile(r"(`+)([^`\n]+)\1")
    inline_ranges = [match.span() for match in inline_code.finditer(text)]
    math = re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|(?<![\\$])\$(?!\$)[^$\n]+\$",
                      re.DOTALL)

    def protect_math(match: re.Match[str]) -> str:
        # Math-looking literals inside authored inline code must stay code.
        if any(start <= match.start() and match.end() <= end for start, end in inline_ranges):
            return match[0]
        block = match[0]
        width = 2 if block.startswith(("$$", r"\[", r"\(")) else 1
        body = block[width:-width].strip()
        # An internal key is provenance even when the model wrapped it in math.
        # Expose only this exact shape to the existing readable-reference path;
        # genuine formulas containing a similarly named variable remain intact.
        return body if re.fullmatch(_INTERNAL_KEY, body) else protect(block)

    text = math.sub(protect_math, text)

    def inline(match: re.Match[str]) -> str:
        body = match[2].strip()
        if re.fullmatch(_INTERNAL_KEY, body) or _OPAQUE_ID.fullmatch(body):
            return body
        if re.fullmatch(
            r"(?:memory_revision|version_number)(?:\s*[:=]?\s*\d+)?|saved_review", body,
        ):
            return body
        return protect(match[0])

    text = inline_code.sub(inline, text)
    return text, saved


def visible_answer(text: str, evidence: list[dict] | None = None) -> str:
    """Hide internal reference keys on reading surfaces without rewriting saved text."""
    from .source_presentation import evidence_name

    text, protected = _protect_authored_content(text)
    # Identifier-only provenance rows carry no learning content. Their originals stay in storage.
    text = re.sub(
        r"(?m)^[ \t]*(?:[-*][ \t]+)?(?:记忆标识|会话\s*ID|对应原始对话快照|"
        r"原始对话快照|SHA[- ]?256|content_sha256|source_id|document_id|version_id)"
        r"[：:=][^\n]*(?:" + _INTERNAL_KEY + "|" + _OPAQUE_ID.pattern + r")[^\n]*(?:\n|$)",
        "", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"[（(](?:原始对话快照|来源快照|SHA[- ]?256|哈希)[：:=]?\s*"
                  + _OPAQUE_ID.pattern + r"\s*[）)]", "", text, flags=re.IGNORECASE)
    refs: dict[str, list[dict]] = {}
    for ref in evidence or []:
        if label := ref.get("label"):
            refs.setdefault(label, []).append(ref)
    for label, values in refs.items():
        names = {evidence_name(ref) for ref in values}
        name = (names.pop() if len(names) == 1 else
                "已确认知识记忆" if label.startswith("M") else "来源资料")
        token = r"(?:\[" + re.escape(label) + r"\]|(?<![A-Za-z0-9_])" + re.escape(label) \
            + r"(?![A-Za-z0-9_]))"
        text = re.sub(r"\blabel\s*(?:为|[:：=])?\s*" + token, lambda _, name=name: name, text,
                      flags=re.IGNORECASE)
        text = re.sub(
            r"(?:已确认的?知识记忆|知识记忆|已保存的?复盘|单题复盘|复盘|记忆)\s*" + token,
            lambda _, name=name: name, text,
        )
        text = re.sub(r"(提供的|引用的|来自|基于|依据|参见|参考)\s*" + token,
                      lambda match, name=name: match[1] + name, text)
    text = re.sub(r"\blabel\s*(?:为|[:：=])?\s*(?:" + _INTERNAL_KEY + r")",
                  "来源信息暂不可用", text, flags=re.IGNORECASE)
    text = re.sub(r"\blabel\s*(?:为|[:：=])?\s*\[来源未核验\]",
                  "来源信息暂不可用", text, flags=re.IGNORECASE)
    text = re.sub(r"\bmemory_revision\s*[:=]?\s*(\d+)\b", r"第\1修订", text)
    text = re.sub(r"\bversion_number\s*[:=]?\s*(\d+)\b", r"复盘第\1版", text)

    def clean_reference_parentheses(match: re.Match[str]) -> str:
        body = match[2]
        if not _INTERNAL_REFERENCE.search(body):
            return match[0]
        body = _INTERNAL_REFERENCE.sub("", body).strip(" \t,，;；、")
        return match[1] + body + match[3] if body else ""

    # Only citation-bearing parentheses are tidied; ordinary expressions such as f() stay intact.
    text = re.sub(r"([（(])([^()（）\n]*)([）)])", clean_reference_parentheses, text)
    labels = set(refs)
    labels.update(re.findall(r"\[([SMR][0-9a-f]{8,64}|[SMR]\.\.\.)\]", text))
    for label in labels - {""}:
        text = re.sub(r"(?m)^([ \t]*)\[" + re.escape(label) + r"\][ \t]*[。；，、.;,]?[ \t]*",
                      r"\1", text)
        text = text.replace("[" + label + "]", "")
    text = _INTERNAL_REFERENCE.sub("", text)
    text = re.sub(r"(?<![A-Za-z0-9_])" + _OPAQUE_ID.pattern + r"(?![A-Za-z0-9_])", "", text)
    text = re.sub(r"(?im)^\s*label\s*`{2,}\s*$", "来源信息暂不可用", text)
    text = re.sub(r"\bmemory_revision\b", "记忆修订", text)
    text = re.sub(r"\bversion_number\b", "复盘保存版本", text)
    text = re.sub(r"\bsaved_review\b", "已保存复盘", text)
    text = text.replace(
        "引用检查提示：回答含有不属于本轮证据的标签，已标为来源未核验。\n\n",
        "部分引用出处尚未核对。\n\n",
    )
    text = text.replace("[来源未核验]", "（出处待核对）")
    text = re.sub(r"(?m)^[ \t]*[。；，、.;,]+[ \t]*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    for index, original in enumerate(protected):
        text = text.replace(f"\ue000{index}\ue001", original)
    return text


def check_citations(text: str, evidence: list[dict]) -> tuple[str, list[str]]:
    """Reject unknown labels, including model placeholders such as [S...]."""
    valid = {ref["label"] for ref in evidence}
    invalid = sorted(set(re.findall(r"\[([SMR][^\]\r\n]{1,80})\]", text)) - valid)
    for label in invalid:
        text = text.replace("[" + label + "]", "[来源未核验]")
    if invalid:
        text = "引用检查提示：回答含有不属于本轮证据的标签，已标为来源未核验。\n\n" + text
    return text, invalid


def checked_messages(messages: list[dict]) -> list[dict]:
    """Recheck older answers at read time, preserving their immutable original record."""
    evidence: dict[str, dict] = {}
    checked = []
    for message in messages:
        metadata = message["metadata"]
        evidence.update({ref["label"]: ref for ref in metadata.get("evidence", [])})
        if message["role"] == "assistant" and message["status"] == "complete":
            content, invalid = check_citations(message["content"], list(evidence.values()))
            if invalid:
                message = {
                    **message,
                    "content": content,
                    "metadata": {
                        **metadata,
                        "raw_response": metadata.get("raw_response") or message["content"],
                        "unverified_citation_labels": invalid,
                    },
                }
        checked.append(message)
    return checked
