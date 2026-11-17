"""Local Pandoc bridge: preserve Office equations, never fetch linked resources."""

from __future__ import annotations

import io
import json
import re
import subprocess
import zipfile
from xml.etree import ElementTree as ET

from .models import StudyError


def convert(data: bytes, source: str, target: str) -> bytes:
    """Run the optional, bundled local executable with bounded time and inert input."""
    try:
        import pypandoc

        executable = pypandoc.get_pandoc_path()
        result = subprocess.run(
            [
                executable,
                "--sandbox",
                "--from=" + source,
                "--to=" + target,
                "--wrap=none",
                "--output=-",
            ],
            input=data,
            capture_output=True,
            timeout=90,
            check=False,
        )
    except (ImportError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        raise StudyError(
            "Word 公式组件不可用或转换超时；请安装 requirements/study-loop-chat.txt，"
            "原件和 Markdown 仍保留。"
        ) from exc
    if result.returncode or result.stderr.strip():
        # Warnings can indicate unsupported formula conversion. Never silently lose it.
        raise StudyError(
            "Word 转换报告不兼容内容，未把部分结果当作成功；"
            "请下载 Markdown 并检查公式，原始对话完整保留。"
        )
    return result.stdout


def _inert(value: object) -> object:
    """Replace images/raw markup before writing so export cannot load files or URLs."""
    if isinstance(value, list):
        return [_inert(v) for v in value]
    if not isinstance(value, dict):
        return value
    if value.get("t") == "Image":
        return {"t": "Str", "c": "[图片引用保留在原始材料中；本对话导出不加载外链图片]"}
    if value.get("t") in {"RawInline", "RawBlock"}:
        text = value.get("c", ["", ""])[1]
        if value["t"] == "RawInline" and re.fullmatch(r"<br\s*/?>", text, re.IGNORECASE):
            return {"t": "LineBreak"}
        if value["t"] == "RawBlock":
            return {"t": "Para", "c": [{"t": "Str", "c": text}]}
        return {"t": "Str", "c": text}
    return {k: _inert(v) for k, v in value.items()}


def docx_markdown(data: bytes) -> str:
    """Extract complete document-order text, tables and native OMML as LaTeX."""
    ast = json.loads(convert(data, "docx", "json"))
    title = ast.get("meta", {}).get("title", {})
    if title.get("t") == "MetaInlines":
        ast["blocks"].insert(0, {"t": "Header", "c": [1, ["", [], []], title["c"]]})
    text = convert(
        json.dumps(_inert(ast)).encode(),
        "json",
        "markdown+pipe_tables-simple_tables-multiline_tables-grid_tables",
    )
    return text.decode("utf-8").replace("\r\n", "\n").replace("\r", "")


def _wide_math_tables(body: bytes) -> bytes:
    """Use a landscape sheet for wide formula tables, preserving native Math XML.

    Scaling only layout attributes keeps namespace declarations and all equation
    operands intact. Narrow conversations retain the default portrait layout.
    """
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
          "m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}
    root = ET.fromstring(body)
    wide = any(
        len(table.findall("./w:tblGrid/w:gridCol", ns)) >= 4
        and table.find(".//m:oMath", ns) is not None
        for table in root.findall(".//w:tbl", ns)
    )
    if not wide:
        return body
    # A4 landscape, 900-twip margins: wide math gets ~1.9x the default table width.
    layout = (
        b'<w:pgSz w:w="16838" w:h="11906" w:orient="landscape" />'
        b'<w:pgMar w:top="900" w:right="900" w:bottom="900" w:left="900" '
        b'w:header="450" w:footer="450" w:gutter="0" />'
    )
    body = re.sub(rb'<w:pg(?:Sz|Mar)\b[^>]*/>', b'', body)
    body = body.replace(b'</w:sectPr>', layout + b'</w:sectPr>')

    def grid(match: re.Match[bytes]) -> bytes:
        widths = [int(w) for w in re.findall(rb'w:w="(\d+)"', match[0])]
        total = sum(widths)
        if not total:
            return match[0]
        return re.sub(
            rb'w:w="(\d+)"',
            lambda value: f'w:w="{round(int(value[1]) * 15038 / total)}"'.encode(),
            match[0],
        )
    return re.sub(rb'<w:tblGrid>.*?</w:tblGrid>', grid, body, flags=re.DOTALL)


def markdown_docx(text: str) -> bytes:
    """Render math as editable Office Math; retain raw transcript in the DOCX package."""
    reader = "markdown+tex_math_single_backslash+pipe_tables+raw_html-raw_tex"
    ast = json.loads(convert(text.encode("utf-8"), reader, "json"))
    from .word_colors import color_word_xml, colored_ast

    ast, math_colors, uses_color = colored_ast(ast)
    data = convert(json.dumps(_inert(ast)).encode(), "json", "docx")
    output = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(data)) as source,
        zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as target,
    ):
        for info in source.infolist():
            body = source.read(info.filename)
            if uses_color and info.filename in math_colors:
                body = color_word_xml(body, math_colors[info.filename])
            if info.filename == "word/document.xml":
                body = _wide_math_tables(body)
            if info.filename == "word/styles.xml":
                # Preserve namespace declarations referenced by mc:Ignorable values.
                body = re.sub(rb'(<w:rFonts\b[^>]*?)\s+w:eastAsia="[^"]*"', rb"\1", body)
                body = body.replace(b"<w:rFonts ", b'<w:rFonts w:eastAsia="Microsoft YaHei" ')
            if info.filename == "[Content_Types].xml":
                root = ET.fromstring(body)
                ET.SubElement(
                    root,
                    "{http://schemas.openxmlformats.org/package/2006/content-types}Default",
                    {"Extension": "md", "ContentType": "text/markdown"},
                )
                body = ET.tostring(root, encoding="utf-8", xml_declaration=True)
            target.writestr(info, body)
        target.writestr("nectivon/transcript.md", text.encode("utf-8"))
    return output.getvalue()
