"""General Markdown layout regressions; no stored user answers or formula-specific repairs."""

from __future__ import annotations

import pytest

from src import math_display
from src.study_loop.color_markup import readable_color_markdown


@pytest.mark.parametrize("delimiters", [("$$", "$$"), (r"\[", r"\]")])
def test_multiline_display_keeps_standalone_fences_and_all_operands(delimiters):
    opening, closing = delimiters
    body = "a+b\n= c+d\n= e"
    source = "前文\n\n" + opening + "\n" + body + "\n" + closing + "\n\n后文保留。"
    result = math_display.normalize_math_delimiters(source)
    assert "$$\n" + body + "\n$$" in result
    assert result.endswith("后文保留。")
    assert math_display.normalize_math_delimiters(result) == result


def test_single_line_inline_and_display_spans_remain_renderable():
    source = r"行内 \(a+b\)，已有 $c+d$，独立 \[e+f\]。"
    result = math_display.normalize_math_delimiters(source)
    assert "$a+b$" in result and "$c+d$" in result and "$$e+f$$" in result
    assert math_display.normalize_math_delimiters(result) == result


def test_table_equation_stays_inside_its_cell_and_does_not_gain_block_fences():
    source = "|名称|公式|\n|---|---|\n" + r"|示例|\[\frac{a}{b}\]|"
    result = math_display.normalize_math_delimiters(source)
    assert len(result.splitlines()) == 3
    assert r"$\displaystyle \frac{a}{b}$" in result
    assert "$$" not in result


@pytest.mark.parametrize("fence", ["```", "~~~", "````"])
def test_fenced_code_keeps_raw_math_and_plain_fraction_notation(fence, monkeypatch):
    source = (
        fence + "latex\n" + r"\[a+b\] $x^2$ 3/4 \ce{H2O}" + "\n" + fence
        + "\n\n正文 \\(y\\) 和 5/6。"
    )
    calls = []
    monkeypatch.setattr(math_display.st, "markdown", lambda value: calls.append(value))
    math_display.render_math_markdown(source)
    original_code = source.split("\n\n")[0]
    assert calls[0].startswith(original_code)
    assert "$y$" in calls[0]
    assert r"$\frac{5}{6}$" in calls[0]


@pytest.mark.parametrize("ticks", ["`", "``"])
def test_inline_code_preserves_delimiters_commands_and_scripts(ticks, monkeypatch):
    code = ticks + r"\[a\] $b^2$ m^2^ 3/4 \ce{H2O}" + ticks
    calls = []
    monkeypatch.setattr(math_display.st, "markdown", lambda value: calls.append(value))
    math_display.render_math_markdown("示例 " + code + r"；正文 \(c\)。")
    assert code in calls[0]
    assert "$c$" in calls[0]


def test_unclosed_fenced_code_is_preserved_without_guessing_a_closing_fence(monkeypatch):
    source = "~~~latex\n" + r"\[a+b\] 3/4"
    calls = []
    monkeypatch.setattr(math_display.st, "markdown", lambda value: calls.append(value))
    math_display.render_math_markdown(source)
    assert calls == [source]


def test_math_within_list_keeps_following_list_item_outside_the_equation():
    source = "- 第一项：\n\n\\[\na+b\n=c\n\\]\n\n- 第二项。"
    result = math_display.normalize_math_delimiters(source)
    assert "$$\na+b\n=c\n$$" in result
    assert result.endswith("- 第二项。")


def test_plain_number_list_repair_never_escapes_a_negative_sign_in_explicit_math():
    source = "$$\na\n- 1\n$$\n\n- 2，- 3"
    result = math_display.normalize_plain_school_math(source)
    assert "$$\na\n- 1\n$$" in result
    assert result.endswith(r"\- 2，- 3")


@pytest.mark.parametrize("delimiters", [(r"\[", r"\]"), (r"\(", r"\)")])
def test_standard_latex_power_chains_are_not_pandoc_scripts(delimiters):
    """Repeated TeX carets must not become a paired prose superscript."""
    opening, closing = delimiters
    polynomial = r"s^3+6s^2+8s+K_r=0"
    other_formula = r"a^2+b^2=c^2"
    source = (
        "前文 " + opening + polynomial + closing
        + "；第二式 " + opening + other_formula + closing
        + r"；已有 $u^2+v^2$；普通记号 m^2^。"
    )
    before = source
    result = math_display.normalize_math_delimiters(source)
    delimiter = "$$" if opening == r"\[" else "$"
    assert delimiter + polynomial + delimiter in result
    assert delimiter + other_formula + delimiter in result
    assert r"$u^2+v^2$" in result
    assert r"$m^{2}$" in result  # Keep the existing prose-script feature.
    assert "$$$" not in result and r"s^{3+6s}" not in result
    assert math_display.normalize_math_delimiters(result) == result
    assert source == before


def test_colored_latex_polynomial_keeps_operands_and_neighboring_math():
    """The saved reader's color path shares the same delimiter protection."""
    polynomial = r"s^3+6s^2+8s+K_r=0"
    source = (
        r"前式 \(a^2+b^2=c^2\)。"
        + '\n\n<span style="color:#D62828">\\[\n'
        + polynomial + "\n\\]</span>\n\n"
        + r"后式 $u^2+v^2$，说明保留。"
    )
    before = source
    result = readable_color_markdown(source)
    assert r"$a^2+b^2=c^2$" in result
    assert "{\\color{#D62828} " + polynomial + "}" in result
    assert r"$u^2+v^2$" in result and result.endswith("，说明保留。")
    assert "<span" not in result and "$$$" not in result
    assert source == before


@pytest.mark.parametrize("expression", [
    r"s^3+4s^2+3s+K_r=0",
    r"x^2+y^2=z^2",
    r"s^3+s^-2=1",
    r"x^a+y^b=c",
])
def test_bare_power_chains_do_not_close_a_pandoc_superscript(expression):
    source = expression + r"；已有 \(u^2+v^2\)；单位 m^2^；化学 H~2~O。"
    before = source
    result = math_display.normalize_math_delimiters(source)
    assert result.startswith(expression + "；")
    assert r"$u^2+v^2$" in result
    assert r"$m^{2}$" in result and r"$H_{2}$O" in result
    assert "$$$" not in result
    assert math_display.normalize_math_delimiters(result) == result
    assert source == before


def test_bare_candidate_polynomial_keeps_separate_exponents_on_read():
    source = "闭环特征方程为 s^3+4s^2+3s+K_r=0；单位 m^2^，分子式 H~2~O。"
    before = source
    result = readable_color_markdown(source)
    assert r"$s^{3}$+4$s^{2}$+3s+K_r=0" in result
    assert r"s^{3+4s}" not in result
    assert r"$m^{2}$" in result and r"$H_{2}$O" in result
    assert "$$$" not in result and source == before


@pytest.mark.parametrize(("literal", "expected"), [
    ("m^2^kg", "$m^{2}$kg"),
    ("m^-2^kg", "$m^{-2}$kg"),
    ("m^n+1^kg", "$m^{n+1}$kg"),
    ("m^2^次", "$m^{2}$次"),
    ("H~2~O", "$H_{2}$O"),
])
def test_pandoc_scripts_can_touch_unit_suffixes(literal, expected):
    source = literal + "；裸式 s^3+4s^2+3s+K_r=0。"
    before = source
    result = math_display.normalize_math_delimiters(source)
    assert result.startswith(expected + "；")
    assert "s^3+4s^2+3s+K_r=0" in result
    assert readable_color_markdown(source).startswith(expected + "；")
    assert math_display.normalize_math_delimiters(result) == result
    assert source == before
