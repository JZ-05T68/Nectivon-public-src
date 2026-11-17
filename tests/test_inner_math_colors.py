"""Read misplaced model colour markup without modifying formulas or stored sources."""

from __future__ import annotations

import hashlib

import pytest

from src.study_loop.color_markup import _project_inner_math_color, readable_color_markdown


@pytest.mark.parametrize("delimiters", [("$", "$"), ("$$", "$$"), (r"\(", r"\)"),
                                       (r"\[", r"\]")])
@pytest.mark.parametrize("color", ["222222", "D62828", "2563EB"])
def test_entire_inner_span_uses_existing_safe_color_projection(delimiters, color):
    start, end = delimiters
    formula = r"x<y,\quad y>0,\quad q=\frac{a}{b}"
    source = start + f'<span style="color:#{color}">' + formula + "</span>" + end
    original_hash = hashlib.sha256(source.encode()).hexdigest()
    projected = _project_inner_math_color(source)
    rendered = readable_color_markdown(source)

    assert projected == f'<span style="color:#{color}">' + start + formula + end + "</span>"
    assert formula in rendered
    assert "\\color{#" + color + "}" in rendered
    assert "<span" not in rendered and "</span>" not in rendered
    assert hashlib.sha256(source.encode()).hexdigest() == original_hash
    assert _project_inner_math_color(projected) == projected


def test_multiline_matrix_row_separators_and_whitespace_are_preserved():
    formula = r"\begin{matrix}a&b\\c&d\end{matrix}"
    source = '$$\n  <span style="color:#D62828">' + formula + "</span>\r\n $$"
    projected = _project_inner_math_color(source)

    assert projected == '<span style="color:#D62828">$$\n  ' + formula + "\r\n $$</span>"
    rendered = readable_color_markdown(source)
    assert formula in rendered and "\\color{#D62828}" in rendered
    assert "<span" not in rendered


@pytest.mark.parametrize("opening", [
    '<span style="color:#00FF00">',
    '<span style="color:#D62828" onclick="alert(1)">',
    '<span style="color:#D62828;background:url(https://invalid.example)">',
    '<span class="red">',
])
def test_unknown_colors_or_extra_attributes_are_not_promoted(opening):
    source = "$$" + opening + r"q=\frac{a}{b}" + "</span>$$"
    assert _project_inner_math_color(source) == source
    rendered = readable_color_markdown(source)
    assert r"\color{#D62828}" not in rendered
    assert r"\color{#00FF00}" not in rendered


@pytest.mark.parametrize("body", [
    '<span style="color:#2563EB">a</span>',
    'a</span><span style="color:#2563EB">b',
    '<img src="https://invalid.example" onerror="alert(1)">',
    '<script>alert(1)</script>',
    '<!-- hidden -->a',
])
def test_nested_multiple_or_active_html_is_not_unwrapped(body):
    source = '$$<span style="color:#D62828">' + body + "</span>$$"
    assert _project_inner_math_color(source) == source


def test_code_examples_and_correct_outer_spans_are_not_rewritten():
    inner = '$$<span style="color:#D62828">a+b=c</span>$$'
    code = "```latex\n" + inner + "\n```\n`" + inner + "`"
    outer = '<span style="color:#2563EB">$$a=b$$</span>'
    source = code + "\n\n" + outer

    assert _project_inner_math_color(source) == source
    rendered = readable_color_markdown(source)
    assert code in rendered
    assert r"\color{#2563EB}" in rendered


def test_later_correct_formula_and_color_annotation_survive_same_read_document():
    source = (
        '$$<span style="color:#D62828">q=a+b</span>$$\n\n'
        '后续 $q-c=d$；<span style="color:#2563EB">仅当 $a>0$</span>'
    )
    rendered = readable_color_markdown(source)

    assert r"\color{#D62828} q=a+b" in rendered
    assert "$q-c=d$" in rendered
    assert r"\color{#2563EB} a>0" in rendered
    assert "后续" in rendered and "仅当" in rendered


def test_table_math_keeps_its_tail_cell_and_absolute_value_after_color_projection():
    source = (
        "| 关系 | 用途 |\n|---|---|\n"
        '| $$<span style="color:#D62828">|x|=q</span>$$ | 完整尾列 |'
    )
    rendered = readable_color_markdown(source)

    assert r"\color{#D62828}" in rendered
    assert r"\vert x\vert =q" in rendered
    assert "| 完整尾列 |" in rendered
    assert "<span" not in rendered


def test_read_skin_uses_default_ink_for_black_formulas_without_rewriting_source():
    source = ('<span style="color:#222222">普通 $a=b$</span>\n'
              '<span style="color:#D62828">重点 $c=d$</span>\n'
              '<span style="color:#2563EB">条件 $e>0$</span>')
    original_hash = hashlib.sha256(source.encode()).hexdigest()
    rendered = readable_color_markdown(source, inherit_default_ink=True)
    assert "$a=b$" in rendered and r"\color{#222222}" not in rendered
    assert r"\color{#D62828}" in rendered and r"\color{#2563EB}" in rendered
    assert hashlib.sha256(source.encode()).hexdigest() == original_hash
