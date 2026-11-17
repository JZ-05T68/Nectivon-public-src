"""Translate three safe note spans to real Word and native Office Math font colours."""

from __future__ import annotations

import re

from .color_markup import COLORS
from .models import StudyError

_STYLE = re.compile(r"\s*color\s*:\s*#(222222|D62828|2563EB)\s*;?\s*", re.IGNORECASE)
_STYLE_NAME = b"NectivonNoteColor"


def colored_ast(ast: dict) -> tuple[dict, dict[str, list[str | None]], bool]:
    """Whitelisted spans get character styles; formula colour follows the same nesting."""
    math: dict[str, list[str | None]] = {"word/document.xml": [], "word/footnotes.xml": []}
    found = False

    def walk(value: object, color: str | None = None,
             part: str = "word/document.xml") -> object:
        nonlocal found
        if isinstance(value, list):
            return [walk(child, color, part) for child in value]
        if not isinstance(value, dict):
            return value
        if value.get("t") == "Note":
            return {"t": "Note", "c": walk(value["c"], color, "word/footnotes.xml")}
        if value.get("t") == "Span":
            attrs, children = value["c"]
            style = next((v for k, v in attrs[2] if k == "style"), "")
            if match := _STYLE.fullmatch(style):
                found = True
                color = match[1].upper()
                attrs = ["", [], [["custom-style", _STYLE_NAME.decode() + color]]]
            else:
                attrs = ["", [], []]
            return {"t": "Span", "c": [attrs, walk(children, color, part)]}
        if value.get("t") == "Math":
            math[part].append(color)
        return {key: walk(child, color, part) for key, child in value.items()}

    result = dict(ast)
    result["blocks"] = walk(ast["blocks"])
    return result, math, found


def _run_color(run: bytes, color: str, *, math: bool = False) -> bytes:
    property_tag = b"w:rPr"
    prop = b'<w:color w:val="' + color.encode() + b'" />'
    if re.search(rb"<w:rPr(?:\s[^>]*)?>", run):
        run = re.sub(rb"<w:color\b[^>]*/>", b"", run)
        return re.sub(rb"(<w:rPr(?:\s[^>]*)?>)", lambda m: m[0] + prop, run, count=1)
    run = re.sub(rb"<w:rPr\s*/>", b"", run)
    properties = b"<" + property_tag + b">" + prop + b"</" + property_tag + b">"
    if math and (match := re.search(rb"<m:rPr(?:\s[^>]*)?>.*?</m:rPr>|<m:rPr\s*/>",
                                   run, re.DOTALL)):
        return run[:match.end()] + properties + run[match.end():]
    return re.sub(rb"(<[wm]:r(?:\s[^>]*)?>)", lambda m: m[0] + properties, run, count=1)


def color_word_xml(body: bytes, math_colors: list[str | None]) -> bytes:
    """Add real run properties without serializing or altering native formula trees."""
    def ordinary(match: re.Match[bytes]) -> bytes:
        run = match[0]
        style = re.search(_STYLE_NAME + rb"(222222|D62828|2563EB)", run)
        return _run_color(run, style[1].decode() if style else COLORS["black"])

    body = re.sub(rb"<w:r(?:\s[^>]*)?>.*?</w:r>", ordinary, body, flags=re.DOTALL)
    blocks = list(re.finditer(rb"<m:oMath(?:\s[^>]*)?>.*?</m:oMath>", body, re.DOTALL))
    if len(blocks) != len(math_colors):
        raise StudyError("Word公式颜色无法完整对应，未导出部分结果；Markdown和保存稿仍保留。")
    index = 0

    def formula(match: re.Match[bytes]) -> bytes:
        nonlocal index
        color = math_colors[index] or COLORS["black"]
        index += 1
        value = re.sub(rb"<m:r(?:\s[^>]*)?>.*?</m:r>",
                       lambda m: _run_color(m[0], color, math=True), match[0], flags=re.DOTALL)
        # Radical signs, fraction rules and other operators use control properties,
        # independently of operand run colours. Do not replace their equation trees.
        structures = (b"acc|bar|borderBox|box|d|eqArr|f|func|groupChr|limLow|limUpp|m|"
                      b"nary|phant|rad|sPre|sSub|sSubSup|sSup")
        control = b'<m:ctrlPr><w:rPr><w:color w:val="' + color.encode() + b'" /></w:rPr></m:ctrlPr>'

        def properties(prop: re.Match[bytes]) -> bytes:
            block = prop[0]
            if b"<m:ctrlPr" in block:
                def existing(ctrl: re.Match[bytes]) -> bytes:
                    content = ctrl[0]
                    if content.endswith(b"/>"):
                        return control
                    if b"<w:rPr" in content:
                        return _run_color(content, color)
                    return content.replace(b"</m:ctrlPr>", control[10:-11] + b"</m:ctrlPr>")
                return re.sub(rb"<m:ctrlPr\s*/>|<m:ctrlPr(?:\s[^>]*)?>.*?</m:ctrlPr>",
                              existing, block, flags=re.DOTALL)
            if block.endswith(b"/>"):
                return block[:-2] + b">" + control + b"</m:" + prop[1] + b"Pr>"
            return block.replace(b"</m:" + prop[1] + b"Pr>",
                                 control + b"</m:" + prop[1] + b"Pr>")

        value = re.sub(rb"<m:(" + structures + rb")Pr(?:\s[^>]*)?(?:\s*/>|>.*?</m:\1Pr>)",
                       properties, value, flags=re.DOTALL)

        def missing_properties(opening: re.Match[bytes]) -> bytes:
            suffix = value[opening.end():]
            if suffix.startswith(b"<m:" + opening[1] + b"Pr"):
                return opening[0]
            return (opening[0] + b"<m:" + opening[1] + b"Pr>" + control
                    + b"</m:" + opening[1] + b"Pr>")

        return re.sub(rb"<m:(" + structures + rb")(?:\s[^>]*)?>", missing_properties, value)

    return re.sub(rb"<m:oMath(?:\s[^>]*)?>.*?</m:oMath>", formula, body, flags=re.DOTALL)
