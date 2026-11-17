"""Three semantic note colours, with inert Markdown storage and safe read projection."""

from __future__ import annotations

import re
from dataclasses import dataclass

COLORS = {"black": "222222", "red": "D62828", "blue": "2563EB"}
STUDY_COLOR_POLICY = """结构化复盘按学习意义使用黑红蓝三色：一般说明保持默认黑色#222222；
少量核心公式、关键结论和步骤用<span style="color:#D62828">重点内容</span>，
适用条件、易错点与必要边界用<span style="color:#2563EB">注意内容</span>。
只使用这三种颜色，不作可信度评级；来源、版本、确认状态用普通文字说明。
普通对话无需为了颜色增加内容，不把全部公式机械标红或所有说明标蓝。
span只包行内内容或完整公式（含数学分隔符），不插入LaTeX内部，
不包表格分隔线、列表符号、标题符号或代码块，不输出其他HTML属性或标签。"""
_OPEN = re.compile(
    r'<span\s+style=["\']\s*color\s*:\s*#(222222|D62828|2563EB)\s*;?\s*["\']\s*>',
    re.IGNORECASE,
)
_TAG = re.compile(_OPEN.pattern + r"|</span\s*>", re.IGNORECASE)
_CODE = re.compile(r"(?ms)^\s*(`{3,}|~{3,})[^\n]*\n.*?^\s*\1\s*$|(`+)[^\n]*?\2")
_MATH = re.compile(r"\$\$[\s\S]*?\$\$|(?<!\\)\$[^$\n]+?\$|\\\[[\s\S]*?\\\]|\\\([^\n]*?\\\)")
_PLAIN_BREAK = re.compile(r"(<br\s*/?>)", re.IGNORECASE)


def has_plain_line_break(text: str) -> bool:
    """Recognize only an attribute-free break outside authored code examples."""
    return bool(_PLAIN_BREAK.search(_CODE.sub("", text)))


@dataclass(frozen=True)
class ColorRun:
    """A contiguous readable range, never arbitrary HTML or a trust assessment."""

    text: str
    color: str | None = None


def color_runs(text: str) -> list[ColorRun]:
    """Read only the three allowed span styles; code examples remain literal code."""
    code = [m.span() for m in _CODE.finditer(text)]
    runs, stack, cursor = [], [], 0
    for match in _TAG.finditer(text):
        if any(start <= match.start() < end for start, end in code):
            continue
        opening = _OPEN.fullmatch(match[0])
        if not opening and not stack:
            continue
        if match.start() > cursor:
            runs.append(ColorRun(text[cursor:match.start()], stack[-1] if stack else None))
        if opening:
            stack.append(opening[1].upper())
        else:
            stack.pop()
        cursor = match.end()
    if cursor < len(text):
        runs.append(ColorRun(text[cursor:], stack[-1] if stack else None))
    return runs


def _wrap(text: str, color: str | None) -> str:
    if not color or color == COLORS["black"] or not text.strip():
        return text
    return f'<span style="color:#{color}">{text}</span>'


def apply_color(text: str, start: int, end: int, color: str) -> str:
    """Colour a real source selection, expanding partial formula selections atomically.

    The JS editor implements the same operation with UTF-16 browser offsets.
    Formula operands, Markdown structure and code are not rewritten.
    """
    chosen = COLORS.get(color, color.lstrip("#").upper())
    if chosen not in COLORS.values():
        raise ValueError("请选择黑色、红色或蓝色。")
    if not 0 <= start <= end <= len(text):
        raise ValueError("文字选区已失效，请重新选择。")
    if start == end:
        return text
    runs = color_runs(text)
    plain = "".join(run.text for run in runs)
    # Map source offsets to readable offsets without treating literal code spans as markup.
    code = [m.span() for m in _CODE.finditer(text)]
    tags, depth = [], 0
    for match in _TAG.finditer(text):
        if any(a <= match.start() < b for a, b in code):
            continue
        if _OPEN.fullmatch(match[0]):
            depth += 1
        elif depth:
            depth -= 1
        else:
            continue
        tags.append(match)
    def offset(position: int) -> int:
        return position - sum(max(0, min(position, m.end()) - m.start())
                              for m in tags if m.start() < position)
    first, last = offset(start), offset(end)
    for match in _MATH.finditer(plain):
        if first < match.end() and last > match.start():
            first, last = min(first, match.start()), max(last, match.end())
    styles = [run.color for run in runs for _ in run.text]
    protected = [m.span() for m in _CODE.finditer(plain)]
    for index in range(first, last):
        if not any(a <= index < b for a, b in protected):
            styles[index] = None if chosen == COLORS["black"] else chosen
    # Keep table separators, line breaks and Markdown list/header prefixes outside spans.
    math = [m.span() for m in _MATH.finditer(plain)]
    for match in re.finditer(r"(?m)^[ \t]*(?:(?:#{1,6}|>|[-+*]|\d+[.)])[ \t]+)?", plain):
        if not any(a <= match.start() < b for a, b in math):
            styles[match.start():match.end()] = [None] * (match.end() - match.start())
    for match in re.finditer(r"(?m)^[ \t]*\|?(?:[ \t]*:?-+:?[ \t]*\|)+"
                             r"[ \t]*:?-*:?[ \t]*\|?[ \t]*$", plain):
        styles[match.start():match.end()] = [None] * (match.end() - match.start())
    for index, char in enumerate(plain):
        if (char in "\r\n" or (char == "|" and (index == 0 or plain[index-1] != "\\"))) \
                and not any(a <= index < b for a, b in math):
            styles[index] = None
    output, cursor = [], 0
    while cursor < len(plain):
        stop = cursor + 1
        while stop < len(plain) and styles[stop] == styles[cursor]:
            stop += 1
        output.append(_wrap(plain[cursor:stop], styles[cursor]))
        cursor = stop
    return "".join(output)


def _project_inner_math_color(text: str) -> str:
    """Read a sole allowed colour span inside math without passing HTML to KaTeX.

    Some model answers place the span inside the math delimiters. Move only a
    whole, unnested whitelisted wrapper outside those delimiters so the existing
    safe colour renderer can handle it. This never edits the caller's source or
    expands the set of HTML allowed on read surfaces.
    """
    code = [match.span() for match in _CODE.finditer(text)]
    closing = re.compile(r"</span\s*>$", re.IGNORECASE)
    other_markup = re.compile(r"</?[A-Za-z][\w:-]*(?:\s[^<>]*)?/?>|<!--|<!|<\?")

    def project(match: re.Match[str]) -> str:
        if any(start < match.end() and match.start() < end for start, end in code):
            return match[0]
        block = match[0]
        width = 2 if block.startswith(("$$", r"\[", r"\(")) else 1
        body = block[width:-width]
        stripped = body.strip()
        opening, end = _OPEN.match(stripped), closing.search(stripped)
        if not opening or not end or opening.end() > end.start():
            return block
        inner = stripped[opening.end():end.start()]
        if _TAG.search(inner) or other_markup.search(inner):
            return block
        # Preserve the formula's leading/trailing whitespace and every operand.
        leading = body[:len(body) - len(body.lstrip())]
        trailing = body[len(body.rstrip()):]
        formula = block[:width] + leading + inner + trailing + block[-width:]
        return opening[0] + formula + end[0]

    return _MATH.sub(project, text)


def readable_color_markdown(text: str, *, inherit_default_ink: bool = False) -> str:
    """Emit only safe generated spans and KaTeX colours, never execute stored HTML.

    This projection is for read views only; stored Markdown is left byte-for-byte intact.
    """
    from src.math_display import (
        normalize_chemistry_markdown,
        normalize_math_delimiters,
        normalize_plain_school_math,
    )

    def inert(value: str) -> str:
        # Escape individual markup delimiters, including incomplete tags. Do not
        # escape '&' again: Markdown decodes authored entities once as text, never
        # reparses the resulting '<' as executable markup. Only this exact,
        # attribute-free break is rebuilt as HTML; everything else stays inert.
        return "".join(
            "<br>" if _PLAIN_BREAK.fullmatch(part) else
            part.replace("<", "&lt;").replace(">", "&gt;").replace("![", "&#33;[")
            for part in _PLAIN_BREAK.split(value)
        )

    def render_run(run: ColorRun) -> str:
        # The read surface may use light ink on a dark skin. Black is the
        # ordinary-text meaning, while red/blue annotations keep their colours.
        if inherit_default_ink and run.color == COLORS["black"]:
            run = ColorRun(run.text)
        parts = []
        value = run.text
        cursor = 0
        for match in _MATH.finditer(value):
            parts.append(_wrap(inert(value[cursor:match.start()]), run.color))
            formula = match[0]
            if not run.color:
                parts.append(formula)
                cursor = match.end()
                continue
            delimiter = "$$" if formula.startswith("$$") else "$"
            body = formula[len(delimiter):-len(delimiter)]
            coloured = "{\\color{#" + run.color + "} " + body.strip() + "}"
            parts.append(delimiter + ("\n" if delimiter == "$$" else "") + coloured
                          + ("\n" if delimiter == "$$" else "") + delimiter)
            cursor = match.end()
        parts.append(_wrap(inert(value[cursor:]), run.color))
        return "".join(parts)

    # Normalize the complete document before splitting semantic colour runs.
    # Table rows and display fences can span several runs; formatting each
    # fragment separately loses their Markdown context. Math must also bypass
    # prose HTML escaping so '<...>' comparisons cannot swallow adjacent cells.
    value = normalize_plain_school_math(normalize_chemistry_markdown(
        normalize_math_delimiters(_project_inner_math_color(text), preserve_line_breaks=True),
    ))
    output = []
    for run in color_runs(value):
        cursor = 0
        for code in _CODE.finditer(run.text):
            output.extend((render_run(ColorRun(run.text[cursor:code.start()], run.color)), code[0]))
            cursor = code.end()
        output.append(render_run(ColorRun(run.text[cursor:], run.color)))
    return "".join(output)
