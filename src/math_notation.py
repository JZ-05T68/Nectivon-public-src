"""Pure notation checks shared by independent knowledge rendering."""

import re

from src.question_recognition_rules import LATEX_SYMBOLS

_FRACTION = re.compile(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}")


def _canonical_math(text: str) -> str:
    """Compare notation conservatively, retaining fraction operand grouping."""

    value = re.sub(r"\s+", "", str(text).replace("$", ""))
    superscripts = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
    value = re.sub(
        r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+",
        lambda match: "^" + match.group(0).translate(superscripts),
        value,
    )
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = re.sub(r"\\(?:mathrm|mathit|mathbf|operatorname)\{([^{}]+)\}", r"\1", value)
    for symbol, command in sorted(LATEX_SYMBOLS.items(), key=lambda item: -len(item[1])):
        value = value.replace(command, symbol)
    value = re.sub(r"\\(?=(?:sin|cos|tan|log|ln|lim|exp)\b)", "", value)
    value = re.sub(r"\\[,;!: ]", "", value)

    def operand(token: str) -> str:
        if token.startswith("(") and token.endswith(")"):
            return token
        return f"({token})" if re.search(r"[+*/=]|(?<!^)-", token) else token

    value = _FRACTION.sub(
        lambda match: f"{operand(match.group(1))}/{operand(match.group(2))}", value
    )
    for command, symbol in (
        (r"\left", ""),
        (r"\right", ""),
        (r"\lvert", "|"),
        (r"\rvert", "|"),
        (r"\times", "×"),
        (r"\div", "÷"),
        (r"\leq", "≤"),
        (r"\geq", "≥"),
        (r"\triangle", "△"),
        (r"\pi", "π"),
        (r"\cdot", "×"),
    ):
        value = value.replace(command, symbol)
    return value.replace("{", "").replace("}", "").replace("*", "×")


def apply_verified_math_spans(source: str, spans: object) -> str:
    """Accept only source-identical, non-overlapping, meaning-preserving spans."""

    if not isinstance(spans, list):
        return source
    accepted: list[tuple[int, int, str]] = []
    last_end = 0
    for item in sorted(
        (value for value in spans if isinstance(value, dict)),
        key=lambda value: value.get("start", -1) if type(value.get("start")) is int else -1,
    ):
        start, end = item.get("start"), item.get("end")
        latex = item.get("latex")
        literal = item.get("text")
        if (
            type(start) is not int
            or type(end) is not int
            or not isinstance(latex, str)
            or not isinstance(literal, str)
            or not (last_end <= start < end <= len(source))
            or source[start:end] != literal
            or "$" in latex
            or "\n" in latex
            or "$" in literal
            or _canonical_math(latex) != _canonical_math(literal)
        ):
            continue
        accepted.append((start, end, latex))
        last_end = end
    output: list[str] = []
    cursor = 0
    for start, end, latex in accepted:
        output.extend((source[cursor:start], f"${latex}$"))
        cursor = end
    output.append(source[cursor:])
    return "".join(output)
