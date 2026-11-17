"""Read-mode rendering for AI/transcript content that may contain math.

Product contract (visual red team R2, 2026-09-27):

- READ surfaces must display math the way a student reads it — rendered
  formulas, not LaTeX source.  The transcripts produced by handwriting
  vision routinely contain ``$...$`` / ``$$...$$`` LaTeX mixed with
  markdown; Streamlit's ``st.markdown`` renders both (KaTeX) natively and
  does NOT allow raw HTML unless ``unsafe_allow_html=True``, so this is
  XSS-safe by default.
- EDIT surfaces (``st.text_area``) intentionally keep the raw source so
  the user can correct it; they must not be switched to this renderer.
- Malformed LaTeX degrades to visible source instead of breaking the
  page — an honest, readable-enough fallback.

One shared renderer, not per-page regex patches (§15 architecture rule).
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable

import streamlit as st

from src.question_recognition_rules import LATEX_SYMBOLS
from src.recognition_json import normalize_latex_command_escapes

_SUBSCRIPT_TRANSLATION = str.maketrans("0123456789+-", "₀₁₂₃₄₅₆₇₈₉₊₋")
_SUPERSCRIPT_TRANSLATION = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
_MHCHEM_COMMAND = re.compile(r"\\ce\s*\{")
_EXPLICIT_SUBSCRIPT = re.compile(r"_\{([0-9]+)\}|_([0-9]+)")
_EXPLICIT_CHARGE = re.compile(r"\^\{([0-9]*[+-])\}|\^([0-9]*[+-])")
_PLAIN_FORMULA_NUMBER = re.compile(r"(?<=[A-Za-z\)\]])([0-9]+)")
_EXISTING_MATH_BLOCK = re.compile(r"(\$\$.*?\$\$|\$.*?\$)", re.DOTALL)
_RELATION_ENTITY = re.compile(
    r"&(?:lt|gt|le|ge|leq|geq);|&#0*(?:60|62);|&#[xX]0*3[cCeE];"
)
_MATH_SPACE = re.compile(
    r"(?<!\\)(?P<slashes>\\+)mspace(?![A-Za-z])(?:"
    r"\s*\{\s*(?P<braced>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*"
    r"(?P<braced_unit>mu|em|ex|pt|mm|cm|in|bp|pc|dd|cc|nd|nc|sp)\s*\}"
    r"|\s*(?P<bare>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*"
    r"(?P<bare_unit>mu|em|ex|pt|mm|cm|in|bp|pc|dd|cc|nd|nc|sp)(?![A-Za-z]))"
)
_LINE_START_NEGATIVE_NUMBER = re.compile(
    r"(?m)^(?P<indent>[ \t]*)-(?=[ \t]+(?:[\u2212-][ \t]*)?\d)"
)
_PLAIN_SCHOOL_MATH = re.compile(
    r"(?P<mixed>(?<![A-Za-z0-9_/])[+-]?\d+\s+\d+/\d+(?![A-Za-z0-9_/]))"
    r"|(?P<fraction>(?<![A-Za-z0-9_/])[+-]?\d+/[+-]?\d+(?![A-Za-z0-9_/]))"
    r"|(?P<power>(?:\([^()\n]+\)|[A-Za-z]|[+-]?\d+)\^(?:\{[+-]?\d+\}|[+-]?\d+))"
)
_CHOICE_LABEL = re.compile(r"(?<![A-Za-z0-9\\.])([A-D])[.．、:：)）][ \t]*")
_QUESTION_EXPRESSION = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:\|[^|\n]{1,24}\||[+\-]?\([A-Za-z0-9+\-*/×÷^{}]{1,60}\)|[A-Za-z]{1,4}|[+\-]?\d+[A-Za-z]?)"
    r"(?:[ \t]*[+\-×÷*/=≤≥<>^][ \t]*"
    r"(?:\|[^|\n]{1,24}\||[+\-]?\([A-Za-z0-9+\-*/×÷^{}]{1,60}\)|"
    r"[+\-]?[A-Za-z]{1,4}|[+\-]?\d+[A-Za-z]?|\{[A-Za-z0-9+\-*/ ]{1,60}\}))+"
)
_QUESTION_ABSOLUTE = re.compile(r"\|[A-Za-z0-9+\-]{1,24}\|")
_QUESTION_COMPLEX_ABSOLUTE = re.compile(
    r"\|(?=[A-Za-z0-9(+-])[^|，,。；;：:\n]{1,100}"
    r"[()÷/][^|，,。；;：:\n]{0,100}\|"
)
_QUESTION_MONOMIAL = re.compile(r"(?<![A-Za-z0-9])[+\-]?\d+(?:\.\d+)?[A-Za-z]{1,3}(?![A-Za-z0-9])")
_QUESTION_LETTER = re.compile(r"(?<![A-Za-z\\])([A-Za-z]{1,3})(?![A-Za-z.])")
_ABSOLUTE_FRACTION = re.compile(r"\|([A-Za-z][A-Za-z0-9]*)\|/([A-Za-z][A-Za-z0-9]*)")
_PAREN_FRACTION = re.compile(r"\(([^()]{1,60})\)[÷/]([A-Za-z0-9]{1,12})")
_GROUPED_FRACTION = re.compile(
    r"\(([^()\n]{1,120})\)\s*[/÷]\s*\(([^()\n]{1,120})\)"
)
_GROUPED_DENOMINATOR = re.compile(
    r"(?<![A-Za-z0-9\\}])([A-Za-z]|[+\-]?\d+(?:\.\d+)?)\s*[/÷]\s*\(([^()\n]{1,120})\)"
)
_LETTER_FRACTION = re.compile(r"(?<![A-Za-z0-9])([A-Za-z])/([A-Za-z0-9]{1,12})")
_NUMERIC_FRACTION = re.compile(r"(?<![A-Za-z0-9])([+\-]?\d+)[÷/]([+\-]?\d+)")
_QUESTION_NUMBER = re.compile(r"(?<![A-Za-z0-9])([+\-]?\d+(?:\.\d+)?)(?![A-Za-z0-9])")
_EXPLICIT_EXPONENT = re.compile(r"\^[ \t]*(?:\(([A-Za-z0-9+\-*/ ]+)\)|([+\-]?\d+|[A-Za-z]))")
_EXPLICIT_MATH_SUBSCRIPT = re.compile(
    r"_[ \t]*(?:\(([A-Za-z0-9=+\-*/ ]+)\)|([+\-]?\d+|[A-Za-z]))"
)
_MATH_FUNCTIONS = r"sin|cos|tan|cot|sec|csc|arcsin|arccos|arctan|sinh|cosh|tanh|log|ln|exp|lim"
_ADVANCED_SYMBOLS = "".join(symbol for symbol in LATEX_SYMBOLS if symbol not in "×÷≤≥")
_ADVANCED_MATH_RUN = re.compile(
    rf"(?<![A-Za-z\\])(?:[{_ADVANCED_SYMBOLS}]|"
    rf"\\(?:{_MATH_FUNCTIONS}|int|iint|iiint|oint|sum|prod|alpha|beta|theta|sigma|pi)\b|"
    rf"(?:{_MATH_FUNCTIONS})(?![A-Za-z]))"
    rf"(?:\\[A-Za-z]+|[A-Za-z]{{1,3}}(?![A-Za-z])|"
    rf"[{''.join(LATEX_SYMBOLS)}0-9_{{}}()\[\]^+*/=<>≤≥×÷|. \t\-])*"
)


def _markdown_code_spans(text: str) -> list[tuple[int, int]]:
    """Locate fenced and inline Markdown code before notation-only rewrites."""
    spans: list[tuple[int, int]] = []
    fence = re.compile(r"(?m)^[ \t]{0,3}(`{3,}|~{3,})[^\n]*(?:\n|$)")
    cursor = 0
    while opening := fence.search(text, cursor):
        marker = opening[1]
        closing = re.compile(
            r"(?m)^[ \t]{0,3}" + re.escape(marker[0]) + "{" + str(len(marker))
            + r",}[ \t]*(?:\n|$)"
        ).search(text, opening.end())
        end = closing.end() if closing else len(text)
        spans.append((opening.start(), end))
        cursor = end
    boundaries = [(-1, 0), *spans, (len(text), len(text))]
    inline: list[tuple[int, int]] = []
    for previous, following in zip(boundaries[:-1], boundaries[1:], strict=True):
        cursor, limit = previous[1], following[0]
        while opening := re.search(r"`+", text[cursor:limit]):
            start = cursor + opening.start()
            marker = opening[0]
            closing = re.compile(
                r"(?<!`)" + re.escape(marker) + r"(?!`)"
            ).search(text, cursor + opening.end(), limit)
            if closing is None:
                cursor += opening.end()
                continue
            inline.append((start, closing.end()))
            cursor = closing.end()
    return sorted([*spans, *inline])


def _outside_code(text: str, spans: list[tuple[int, int]], transform: Callable[[str], str]) -> str:
    """Transform prose/math chunks while retaining original code bytes exactly."""
    output, cursor = [], 0
    for start, end in spans:
        output.extend((transform(text[cursor:start]), text[start:end]))
        cursor = end
    output.append(transform(text[cursor:]))
    return "".join(output)


def _advanced_math_to_latex(text: str) -> str:
    """Typeset explicit calculus/functions/Greek notation without solving it."""

    value = "".join(LATEX_SYMBOLS.get(character, character) + (
        " " if character in LATEX_SYMBOLS else ""
    ) for character in _brace_explicit_exponents(text.strip()))
    value = re.sub(
        rf"(?<![A-Za-z\\])({_MATH_FUNCTIONS})(?![A-Za-z])",
        lambda match: "\\" + match.group(1) + " ", value,
    )
    return value.strip()


def _protect_explicit_latex(text: str, save_math) -> str:
    r"""Keep arbitrary typed LaTeX commands and balanced arguments in one formula.

    This includes matrices, vectors and commands beyond a school-specific list.
    Existing delimited formulas are handled separately; ordinary prose remains text.
    """

    def environment_end(start: int) -> int | None:
        environment = re.match(r"\\begin\{([A-Za-z*]+)\}", text[start:])
        if environment:
            closing = re.search(r"\\end\{" + re.escape(environment[1]) + r"\}",
                                text[start + environment.end():])
            if closing:
                return start + environment.end() + closing.end()
        return None

    output, cursor = [], 0
    while match := re.search(r"\\[A-Za-z]+", text[cursor:]):
        start = cursor + match.start()
        end = environment_end(start) or start
        if end == start:
            while end < len(text):
                if env_end := environment_end(end):
                    end = env_end
                    continue
                token = re.match(r"\\(?:[A-Za-z]+|[,;!: ])|[A-Za-z]{1,3}(?![A-Za-z])|"
                                 r"[0-9]+|[+*/=<>_\^()\[\]|., \t\-]", text[end:])
                if text[end] == "{":
                    argument = _balanced_argument(text, end)
                    if argument is None:
                        break
                    end = argument[1]
                elif token:
                    end += token.end()
                elif text[end] in LATEX_SYMBOLS:
                    end += 1
                else:
                    break
        if end == start:
            # Never loop on a damaged command.
            end = start + len(match.group(0))
        formula = text[start:end].rstrip()
        output.extend((text[cursor:start], save_math(formula)))
        cursor = start + len(formula)
    output.append(text[cursor:])
    return "".join(output)


def _brace_explicit_exponents(text: str) -> str:
    """Recognise a typed caret as an exponent, retaining its full operand."""

    text = _EXPLICIT_MATH_SUBSCRIPT.sub(
        lambda match: "_{" + (match.group(1) or match.group(2)).strip() + "}", text
    )
    return _EXPLICIT_EXPONENT.sub(
        lambda match: "^{" + (match.group(1) or match.group(2)).strip() + "}", text
    )


def format_multiple_choice_lines(text: str) -> str:
    """Put a real A–D option set on separate lines without editing stored OCR."""

    source = str(text or "")
    math_ranges = [match.span() for match in _EXISTING_MATH_BLOCK.finditer(source)]
    matches = [
        match for match in _CHOICE_LABEL.finditer(source)
        if not any(start <= match.start() < end for start, end in math_ranges)
    ]
    if len(matches) < 2 or matches[0].group(1) != "A":
        return source
    # Require ordered labels so an incidental abbreviation in prose is not split.
    labels = [match.group(1) for match in matches]
    if labels != list("ABCD"[: len(labels)]):
        return source
    pieces = [source[: matches[0].start()].rstrip()]
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        option = source[match.start() : end].strip()
        pieces.append(option)
    return "\n\n".join(pieces)


def _question_expression_to_latex(value: str) -> str:
    """Apply notation-only rewrites; never infer a value or change an operator."""

    normalized = re.sub(r"\s+", "", _brace_explicit_exponents(value))
    normalized = _typeset_grouped_fractions(normalized)
    normalized = _ABSOLUTE_FRACTION.sub(
        lambda match: f"\\dfrac{{|{match.group(1)}|}}{{{match.group(2)}}}",
        normalized,
    )
    normalized = _PAREN_FRACTION.sub(
        lambda match: f"\\dfrac{{{match.group(1)}}}{{{match.group(2)}}}",
        normalized,
    )
    normalized = _LETTER_FRACTION.sub(
        lambda match: f"\\dfrac{{{match.group(1)}}}{{{match.group(2)}}}",
        normalized,
    )
    normalized = _NUMERIC_FRACTION.sub(
        lambda match: (
            f"-\\dfrac{{{match.group(1)[1:]}}}{{{match.group(2)}}}"
            if match.group(1).startswith("-")
            else f"\\dfrac{{{match.group(1).lstrip('+')}}}{{{match.group(2)}}}"
        ),
        normalized,
    )
    normalized = normalized.replace("×", "\\times ").replace("÷", "\\div ")
    normalized = normalized.replace("≤", "\\leq ").replace("≥", "\\geq ")
    normalized = re.sub(r"\^([+\-]?\d+)", r"^{\1}", normalized)
    return normalized


def _typeset_grouped_fractions(value: str) -> str:
    """Retain explicitly grouped numerator/denominator precedence in fractions."""

    for pattern in (_GROUPED_FRACTION, _GROUPED_DENOMINATOR):
        value = pattern.sub(
            lambda match: f"\\dfrac{{{match[1]}}}{{{match[2]}}}", value,
        )
    return value


def _normalize_existing_question_math(block: str) -> str:
    """Typeset unambiguous divisions inside already delimited math spans."""

    delimiter = "$$" if block.startswith("$$") else "$"
    value = _brace_explicit_exponents(
        block[len(delimiter) : -len(delimiter)].replace(r"\frac", r"\dfrac")
    )
    value = _typeset_grouped_fractions(value)
    value = re.sub(
        r"\(([^()]{1,60})\)\s*\\div\s*([A-Za-z0-9]{1,12})",
        lambda match: f"\\dfrac{{{match.group(1)}}}{{{match.group(2)}}}",
        value,
    )
    value = re.sub(
        r"(?<![A-Za-z0-9])([A-Za-z0-9]+)\s*\\div\s*([A-Za-z0-9]+)",
        lambda match: f"\\dfrac{{{match.group(1)}}}{{{match.group(2)}}}",
        value,
    )
    return f"{delimiter}{value}{delimiter}"


def _unicode_script_notation(text: str) -> str:
    """Translate explicit Unicode super/subscripts without inferring missing powers."""

    greek = "αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΥΦΨΩϵϑϕϱς"
    text = re.sub(
        rf"(?<=[A-Za-z0-9)）}}{greek}])[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+",
        lambda match: "^{" + match.group(0).translate(
            str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
        ) + "}", text,
    )
    return re.sub(
        rf"(?<=[A-Za-z0-9)）}}{greek}])[₀₁₂₃₄₅₆₇₈₉₊₋ₙₓₐₑₒₚᵢⱼ]+",
        lambda match: "_{" + match.group(0).translate(
            str.maketrans("₀₁₂₃₄₅₆₇₈₉₊₋ₙₓₐₑₒₚᵢⱼ", "0123456789+-nxaeopij")
        ) + "}", text,
    )


def normalize_question_math(text: str) -> str:
    """Render school-math tokens as KaTeX on question read views only.

    Original OCR and the human edit boxes remain byte-for-byte unchanged.
    Ambiguous mathematics is never solved here; unsupported spans stay visible.
    """

    source = str(text or "")
    if spans := _markdown_code_spans(source):
        return _outside_code(source, spans, normalize_question_math)

    def plain_chunk(chunk: str) -> str:
        protected: list[str] = []

        def save_math(latex: str) -> str:
            protected.append(f"${latex}$")
            return chr(0xE000 + len(protected) - 1)

        chunk = _protect_explicit_latex(chunk, save_math)

        # Unicode superscripts are mathematical notation, not OCR guesses.
        # Convert them before finding expressions so (a+1)² stays one formula.
        chunk = _unicode_script_notation(chunk)
        chunk = re.sub(
            r"（([A-Za-z0-9+\-*/×÷|^{}\s]+)）", r"(\1)", chunk
        )
        chunk = _brace_explicit_exponents(chunk)
        chunk = _ADVANCED_MATH_RUN.sub(
            lambda match: save_math(_advanced_math_to_latex(match.group(0))), chunk
        )
        chunk = re.sub(
            r"(?<![A-Za-z0-9])([A-Za-z]|[+\-]?\d+)\^\{([A-Za-z0-9+\-*/ ]{1,60})\}",
            lambda match: save_math(f"{match.group(1)}^{{{match.group(2)}}}"),
            chunk,
        )
        chunk = re.sub(
            r"√(?:\(([^()\n]+)\)|([A-Za-z0-9]+))",
            lambda match: save_math(f"\\sqrt{{{match.group(1) or match.group(2)}}}"),
            chunk,
        )
        chunk = chunk.replace("π", save_math(r"\pi")) if "π" in chunk else chunk
        chunk = re.sub(
            r"△([A-Z]{3})",
            lambda match: save_math(f"\\triangle {match.group(1)}"),
            chunk,
        )
        chunk = _QUESTION_COMPLEX_ABSOLUTE.sub(
            lambda match: save_math(_question_expression_to_latex(match.group(0))),
            chunk,
        )
        chunk = _QUESTION_EXPRESSION.sub(
            lambda match: save_math(_question_expression_to_latex(match.group(0))),
            chunk,
        )
        chunk = _QUESTION_ABSOLUTE.sub(
            lambda match: save_math(match.group(0)), chunk
        )
        chunk = _QUESTION_MONOMIAL.sub(
            lambda match: save_math(match.group(0)), chunk
        )
        chunk = re.sub(r"=(?=_{2,})", lambda _match: save_math("="), chunk)
        chunk = _QUESTION_LETTER.sub(
            lambda match: (
                match.group(0)
                if match.group(0) in {"AI", "PDF", "OCR"}
                else save_math(match.group(0))
            ),
            chunk,
        )
        chunk = _QUESTION_NUMBER.sub(
            lambda match: save_math(match.group(0)), chunk
        )
        for index, math in enumerate(protected):
            chunk = chunk.replace(chr(0xE000 + index), math)
        # Adjacent derived spans can share math mode.  Keep a separator so
        # a command followed by a letter (e.g. \pi + r) cannot become \pir.
        return chunk.replace("$$", " ")

    # A malformed AI/user fragment such as ``-2$$c &#x20;`` is not a display
    # equation.  Remove only this exact impossible in-word delimiter and the
    # observed encoded space; preserve the editable source unchanged.
    display_source = normalize_math_delimiters(text).replace("&#x20;", " ")
    display_source = re.sub(r"(?<=[0-9])\$\$(?=[A-Za-z])", "", display_source)
    choices = format_multiple_choice_lines(display_source)
    parts = _EXISTING_MATH_BLOCK.split(choices)
    return "".join(
        _normalize_existing_question_math(part)
        if _EXISTING_MATH_BLOCK.fullmatch(part) else plain_chunk(part)
        for part in parts
    )


def render_question_math_markdown(text: str) -> None:
    """Read-only first-layer renderer with option layout and mathematical type."""

    st.markdown(
        normalize_plain_school_math(normalize_question_math(normalize_chemistry_markdown(text)))
    )


def _unicode_chemical_formula(source: str) -> str:
    """Conservatively turn one mhchem payload into copy-safe plain text.

    Streamlit's bundled KaTeX build does not load the optional mhchem
    extension.  Leaving ``\\ce{...}`` in the page therefore exposes source or
    a partial parse.  This fallback intentionally handles only notation whose
    semantics can be preserved deterministically; it never balances or invents
    a chemical equation.
    """

    value = str(source).strip()
    value = (
        value.replace("\\rightleftharpoons", "⇌")
        .replace("\\leftrightharpoons", "⇌")
        .replace("\\longrightarrow", "→")
        .replace("\\rightarrow", "→")
        .replace("\\longleftarrow", "←")
        .replace("\\leftarrow", "←")
        .replace("<=>", "⇌")
        .replace("<->", "⇌")
        .replace("->", "→")
        .replace("<-", "←")
        .replace("\\cdot", "·")
        .replace("*", "·")
        .replace("\\downarrow", "↓")
        .replace("\\uparrow", "↑")
    )
    value = _EXPLICIT_CHARGE.sub(
        lambda match: (match.group(1) or match.group(2)).translate(
            _SUPERSCRIPT_TRANSLATION
        ),
        value,
    )
    value = _EXPLICIT_SUBSCRIPT.sub(
        lambda match: (match.group(1) or match.group(2)).translate(
            _SUBSCRIPT_TRANSLATION
        ),
        value,
    )
    # mhchem permits ordinary formula digits (H2O, [Cu(C2O4)2]^2-).  A
    # leading stoichiometric coefficient has no element/group immediately to
    # its left, so it stays baseline while atom/group counts become subscripts.
    value = _PLAIN_FORMULA_NUMBER.sub(
        lambda match: match.group(1).translate(_SUBSCRIPT_TRANSLATION), value
    )
    return value.replace("{", "").replace("}", "")


def _balanced_argument(text: str, opening_brace: int) -> tuple[str, int] | None:
    depth = 0
    for index in range(opening_brace, len(text)):
        character = text[index]
        if character == "{" and (index == 0 or text[index - 1] != "\\"):
            depth += 1
        elif character == "}" and (index == 0 or text[index - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[opening_brace + 1 : index], index + 1
    return None


def normalize_chemistry_markdown(text: str) -> str:
    """Remove unsupported mhchem commands without changing stored content.

    The transformation runs at display time, so returning to or reloading a
    page is stable.  Plain Unicode is deliberate: it keeps subscripts, charges,
    hydrate dots and arrows copyable even when mhchem is unavailable.  An
    unclosed command fails closed with a visible warning and never leaks a raw
    ``\\ce`` token to the page.
    """

    source = str(text or "")
    if spans := _markdown_code_spans(source):
        return _outside_code(source, spans, normalize_chemistry_markdown)
    output: list[str] = []
    cursor = 0
    while True:
        match = _MHCHEM_COMMAND.search(source, cursor)
        if match is None:
            output.append(source[cursor:].replace("\\ce", "[化学式标记无法解析]"))
            break
        opening_brace = match.end() - 1
        parsed = _balanced_argument(source, opening_brace)
        if parsed is None:
            output.append(source[cursor : match.start()])
            remainder = source[match.end() :].strip()
            fallback = _unicode_chemical_formula(remainder)
            output.append(
                f"[化学式源码不完整：{fallback}]" if fallback else "[化学式源码不完整]"
            )
            break
        payload, parsed_end = parsed
        prefix = source[cursor : match.start()]
        # A standalone ``$\\ce{...}$`` / ``$$\\ce{...}$$`` must leave math
        # mode entirely: the fallback already carries real Unicode scripts,
        # and keeping the dollar pair would make KaTeX reinterpret it.
        if prefix.endswith("$$") and source.startswith("$$", parsed_end):
            prefix = prefix[:-2]
            parsed_end += 2
        elif prefix.endswith("$") and source.startswith("$", parsed_end):
            prefix = prefix[:-1]
            parsed_end += 1
        output.append(prefix)
        output.append(_unicode_chemical_formula(payload))
        cursor = parsed_end
    return "".join(output)


def normalize_plain_school_math(text: str) -> str:
    """Turn common plain OCR math into student-facing KaTeX notation.

    The stored/editable text remains unchanged.  Only read surfaces convert
    ``3/4`` to a stacked fraction, ``-3 1/6`` to a mixed number, and
    ``(-3)^2`` to a real superscript.  Existing ``$...$`` math is skipped so
    valid LaTeX is never wrapped twice.
    """

    source = str(text or "")
    if spans := _markdown_code_spans(source):
        return _outside_code(source, spans, normalize_plain_school_math)

    def replace_plain_chunk(chunk: str) -> str:
        def replace_match(match: re.Match[str]) -> str:
            token = match.group(0)
            if match.group("mixed") is not None:
                whole, fraction = token.split(maxsplit=1)
                numerator, denominator = fraction.split("/", maxsplit=1)
                return f"${whole}\\frac{{{numerator}}}{{{denominator}}}$"
            if match.group("fraction") is not None:
                numerator, denominator = token.split("/", maxsplit=1)
                sign = ""
                if numerator.startswith(("+", "-")):
                    sign, numerator = numerator[0], numerator[1:]
                return f"${sign}\\frac{{{numerator}}}{{{denominator}}}$"
            base, exponent = token.rsplit("^", maxsplit=1)
            exponent = exponent.removeprefix("{").removesuffix("}")
            return f"${base}^{{{exponent}}}$"

        return _PLAIN_SCHOOL_MATH.sub(replace_match, chunk)

    # OCR frequently wraps a negative-number sequence onto a new line, for
    # example ``- 8, - 4``.  CommonMark interprets the first hyphen as a list
    # marker, so students see a bullet and can reasonably think the sign was
    # lost.  Escape only line-leading hyphens whose following token is numeric;
    # ordinary prose lists remain real lists.
    parts = _EXISTING_MATH_BLOCK.split(source)
    return "".join(
        part if _EXISTING_MATH_BLOCK.fullmatch(part) else replace_plain_chunk(
            _LINE_START_NEGATIVE_NUMBER.sub(
                lambda match: f"{match.group('indent')}\\-", part,
            )
        )
        for part in parts
    )


def render_math_markdown(text: str) -> None:
    """Render read-only content with real math (LaTeX via KaTeX).

    Use for every READ surface that shows AI-generated or transcribed
    content (source transcripts, secondary conclusions, method notes).
    Never use for editable fields.
    """

    chemistry_safe = normalize_chemistry_markdown(normalize_math_delimiters(text))
    st.markdown(normalize_plain_school_math(chemistry_safe))


def _normalize_markdown_math_layout(text: str, *, preserve_line_breaks: bool = False) -> str:
    """Keep equations inside table cells and render explicit Pandoc script notation.

    Display-only conversion: stored Markdown, code examples and formula operands
    are not edited. A block equation in a pipe-table cell must become inline
    display-style math, otherwise blank lines split the table into paragraphs.
    """
    # TeX delimiters are normalized later; protect their contents from prose scripts.
    protected = re.compile(
        r"(```[\s\S]*?```|`[^`\n]*`|\$\$[\s\S]*?\$\$|\$[^$\n]*\$"
        r"|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\))"
    )
    scripts = re.compile(
        r"(?P<base>\*[A-Za-zα-ωΑ-Ω]+\*|[A-Za-zα-ωΑ-Ω]+)"
        r"(?P<mark>[~^])(?P<script>[A-Za-z0-9+\-]{1,16})(?P=mark)"
    )

    def script(match: re.Match) -> str:
        # In a bare power chain, the second caret opens another exponent;
        # treating it as prose markup would merge distinct polynomial terms.
        # Require a joined variable term (3+4s, a+y, 2x), not a lone
        # exponent followed by unit text such as m^2^kg.
        if (
            match["mark"] == "^"
            and re.search(
                r"(?:[A-Za-z0-9][+-][A-Za-z0-9+-]*|\d+)[A-Za-z]+$", match["script"]
            )
            and re.match(
                r"[+-]?(?:[A-Za-z0-9{(]|\\[A-Za-z]|\.\d)", match.string[match.end():]
            )
        ):
            return match[0]
        mark = "_" if match["mark"] == "~" else "^"
        return "$" + match["base"].strip("*") + mark + "{" + match["script"] + "}$"

    parts = protected.split(text)
    value = "".join(p if protected.fullmatch(p) else scripts.sub(script, p) for p in parts)
    lines, fenced = [], False
    for line in value.splitlines(keepends=True):
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
        if not fenced and line.lstrip().startswith("|") and line.count("|") >= 2:
            line = re.sub(
                r"\$\$(.*?)\$\$|\\\[(.*?)\\\]",
                lambda m: "$\\displaystyle " + (m[1] if m[1] is not None else m[2]) + "$",
                line,
            )
            # Raw HTML stays disabled; keep line-break-separated cell content readable.
            if not preserve_line_breaks:
                line = re.sub(r"<br\s*/?>", "； ", line, flags=re.IGNORECASE)
        lines.append(line)
    return "".join(lines)


def normalize_math_delimiters(text: str, *, preserve_line_breaks: bool = False) -> str:
    """Accept standard LaTeX delimiters on every KaTeX reading surface."""

    source = str(text or "")
    if spans := _markdown_code_spans(source):
        value = _outside_code(source, spans, lambda value: normalize_math_delimiters(
            value, preserve_line_breaks=preserve_line_breaks,
        ))
        # A code span can split a row from its header in the recursive chunks.
        # Recheck the complete document, retaining the original code positions.
        return _normalize_table_math_pipes(value)
    value = _normalize_markdown_math_layout(source, preserve_line_breaks=preserve_line_breaks)
    value = re.sub(r"\\\((.*?)\\\)", lambda match: "$" + match[1] + "$", value, flags=re.DOTALL)
    value = re.sub(
        r"\\\[(.*?)\\\]", lambda match: "\n\n$$" + match[1] + "$$\n\n",
        value, flags=re.DOTALL,
    )
    # Two adjacent inline formulas have a closing and opening dollar next to
    # each other. Separate them before looking for display-math delimiters.
    value = re.sub(r"(?<!\$)(\$[^$\n]+\$)(?=\$[^$\n]+\$(?!\$))", r"\1 ", value)

    def normalized(match: re.Match) -> str:
        block = normalize_latex_command_escapes(match[0])
        # Decode only known comparison references once, inside explicit math.
        # Never turn encoded markup, quotes or dollar delimiters into active HTML.
        block = _RELATION_ENTITY.sub(lambda entity: html.unescape(entity[0]), block)
        block = _normalize_math_spacing(block)
        if block.startswith("$$") and len(block) > 4:
            # Earlier AI JSON transport repairs could retain bare \n at a
            # display block's edges. Repair only those layout escapes, never
            # commands such as \nu or matrix row separators in the formula.
            body = re.sub(r"^(?:\s|\\n(?![A-Za-z]))+|(?:\s|(?<!\\)\\n)+$", "", block[2:-2])
            # micromark treats content on an opening display fence's line as
            # metadata. A multiline formula therefore needs both fences on
            # their own lines, otherwise the first line is lost and later
            # prose can be swallowed into the unclosed math block.
            block = "$$\n" + body + "\n$$" if "\n" in body else "$$" + body + "$$"
            before = "" if value[:match.start()].endswith("\n\n") else "\n\n"
            after = "" if value[match.end():].startswith("\n\n") else "\n\n"
            return before + block + after
        return block

    return _normalize_table_math_pipes(_EXISTING_MATH_BLOCK.sub(normalized, value))


def _normalize_table_math_pipes(source: str) -> str:
    r"""Protect explicit math from GFM cell splitting on read surfaces only.

    GFM processes table separators before KaTeX. A raw ``|`` in ``$|x|$``
    therefore loses cells even when KaTeX reports no error. Recognize actual
    header/separator tables and leave code and non-table formulas untouched.
    """
    if "|" not in source:
        return source

    def mask(value: str) -> str:
        return re.sub(r"[^\r\n]", " ", value)

    masked = list(source)
    for start, end in _markdown_code_spans(source):
        masked[start:end] = mask(source[start:end])
    code_safe = "".join(masked)
    maths = list(_EXISTING_MATH_BLOCK.finditer(code_safe))
    for match in maths:
        masked[match.start():match.end()] = mask(match[0])
    structural_lines = "".join(masked).splitlines(keepends=True)

    def separator(line: str) -> bool:
        value = line.strip()
        return len(line) - len(line.lstrip(" \t")) <= 3 and "|" in value and all(
            re.fullmatch(r"[ \t]*:?-{3,}:?[ \t]*", cell)
            for cell in value.strip("|").split("|")
        )

    rows: set[int] = set()
    for index in range(1, len(structural_lines)):
        header = structural_lines[index - 1]
        if (not separator(structural_lines[index]) or "|" not in header
                or len(header) - len(header.lstrip(" \t")) > 3):
            continue
        rows.add(index - 1)
        following = index + 1
        while following < len(structural_lines):
            line = structural_lines[following]
            if "|" not in line or len(line) - len(line.lstrip(" \t")) > 3:
                break
            rows.add(following)
            following += 1

    ranges, cursor = [], 0
    for index, line in enumerate(structural_lines):
        if index in rows:
            ranges.append((cursor, cursor + len(line)))
        cursor += len(line)
    output, cursor = [], 0
    for match in maths:
        if "|" not in match[0] or not any(
            start <= match.start() and match.end() <= end for start, end in ranges
        ):
            continue
        output.extend((source[cursor:match.start()], _table_latex_pipes(match[0])))
        cursor = match.end()
    output.append(source[cursor:])
    return "".join(output)


def _table_latex_pipes(block: str) -> str:
    r"""Use equivalent KaTeX bars without changing TeX array column syntax.

    ``\|`` denotes a double bar, so ``\Vert`` retains its math meaning.
    GFM retains backslashes inside math: escaping an array rule as ``\|``
    would be an invalid KaTeX column alignment. Instead ``\char124`` creates
    the literal bar token needed by the array parser, without a Markdown pipe.
    Text arguments use the public text-mode ``\textbar`` command.
    """
    literal: list[tuple[int, int, str]] = []
    for match in re.finditer(
        r"\\(?:begin\{(?:d?array)\}|text(?:rm|sf|tt|normal|bf|it)?)\s*(\{)", block,
    ):
        if argument := _balanced_argument(block, match.start(1)):
            kind = "array" if match[0].startswith(r"\begin") else "text"
            literal.append((match.start(1), argument[1], kind))

    output, cursor = [], 0
    while cursor < len(block):
        syntax = next((kind for start, end, kind in literal if start <= cursor < end), "")
        bar = r"\char124 " if syntax == "array" else r"\textbar " if syntax else r"\vert "
        if block.startswith(r"\|", cursor):
            output.append(bar if syntax else r"\Vert ")
            cursor += 2
        elif block.startswith("\\\\", cursor):
            output.append("\\\\")
            cursor += 2
        elif block[cursor] == "|":
            output.append(bar)
            cursor += 1
        else:
            output.append(block[cursor])
            cursor += 1
    return "".join(output)


def _normalize_math_spacing(block: str) -> str:
    r"""Project Pandoc's \mspace to supported KaTeX spacing without changing operands.

    KaTeX lists \mspace as unsupported, but supports \mkern for mu and \hspace
    for the other documented TeX units. Preserve signed lengths, code outside
    this caller's math block, unknown dimensions and escaped command literals.
    See https://katex.org/docs/supported.html#overlap-and-spacing .
    """
    def compatible(match: re.Match) -> str:
        slashes = match["slashes"]
        if len(slashes) % 2 == 0:
            return match[0]
        amount = match["braced"] or match["bare"]
        unit = match["braced_unit"] or match["bare_unit"]
        command = r"\mkern" if unit == "mu" else r"\hspace"
        return slashes[:-1] + command + "{" + amount + unit + "}"

    return _MATH_SPACE.sub(compatible, block)
