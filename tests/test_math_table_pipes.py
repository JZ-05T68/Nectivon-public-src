"""Keep complete math operands and following cells at the Markdown/KaTeX boundary."""

from __future__ import annotations

import hashlib

import pytest

from src.math_display import normalize_math_delimiters, render_math_markdown
from src.study_loop.color_markup import readable_color_markdown


def table(formula: str) -> str:
    return "| 关系 | 条件 | 用途 |\n|---|---|---|\n| 频率 | " + formula + " | 计算频率 |"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"$|s|f_1$", r"$\vert s\vert f_1$"),
        (r"$\left|x+1\right|$", r"$\left\vert x+1\right\vert $"),
        (r"$\|x\|$", r"$\Vert x\Vert $"),
        (r"$\left\|x\right\|$", r"$\left\Vert x\right\Vert $"),
        (r"$\lVert x\rVert$", r"$\lVert x\rVert$"),
        (r"$P(A|B)$", r"$P(A\vert B)$"),
        (r"$a\mid b$", r"$a\mid b$"),
        (r"$\{x|x>0\}$", r"$\{x\vert x>0\}$"),
        (r"$\big|_0^1$", r"$\big\vert _0^1$"),
        (r"$||x||$", r"$\vert \vert x\vert \vert $"),
        (r"\(|x|\)", r"$\vert x\vert $"),
        (r"$$|x|$$", r"$\displaystyle \vert x\vert $"),
    ],
)
def test_table_math_preserves_bar_semantics_and_following_cells(
    source: str, expected: str,
) -> None:
    original = table(source)
    before = hashlib.sha256(original.encode()).hexdigest()
    rendered = normalize_math_delimiters(original)

    assert rendered == table(expected)
    assert rendered.count("|") == 12  # Only the three-column Markdown structure remains.
    assert hashlib.sha256(original.encode()).hexdigest() == before
    assert normalize_math_delimiters(rendered) == rendered


@pytest.mark.parametrize("environment", ["array", "darray"])
def test_array_column_rules_keep_their_tex_syntax_and_body_operands(environment: str) -> None:
    formula = r"$\begin{" + environment + r"}{c|c}|x| & y\\z & |w|\end{" + environment + "}$"
    expected = (
        r"$\begin{" + environment + r"}{c\char124 c}\vert x\vert  & y\\z & \vert w\vert \end{"
        + environment + "}$"
    )
    rendered = normalize_math_delimiters(table(formula))

    assert rendered == table(expected)
    assert r"{c\vert" not in rendered  # That would be an invalid KaTeX column specifier.
    assert normalize_math_delimiters(rendered) == rendered


def test_preescaped_array_rule_becomes_a_literal_rule_not_double_norm() -> None:
    source = table(r"$\begin{array}{c\|c}a&b\end{array}$")
    assert normalize_math_delimiters(source) == table(
        r"$\begin{array}{c\char124 c}a&b\end{array}$"
    )
    assert r"{c\|c}" in source


def test_pipe_in_math_text_uses_a_public_text_mode_bar() -> None:
    assert normalize_math_delimiters(table(r"$\text{a|b}+|x|$")) == table(
        r"$\text{a\textbar b}+\vert x\vert $"
    )


def test_inline_code_does_not_hide_table_context_for_math_after_it() -> None:
    source = (
        "| `a|b` | 条件 | 用途 |\n|---|---|---|\n"
        r"| `c|d` | `literal $|x|$` 实际 $|s|f_1$ | 计算频率 |"
        "\n| 后续 | $|z|$ | 保留尾列 |"
    )
    rendered = normalize_math_delimiters(source)

    assert "| `a|b` | 条件 | 用途 |" in rendered
    assert r"`literal $|x|$` 实际 $\vert s\vert f_1$ | 计算频率 |" in rendered
    assert r"| 后续 | $\vert z\vert $ | 保留尾列 |" in rendered
    assert "`c|d`" in rendered


def test_fenced_code_prose_and_non_table_math_are_byte_for_byte_unchanged() -> None:
    source = (
        r"普通 $|s|f_1$，概率 $P(A|B)$，代码 `x|y`。"
        "\n\n```markdown\n" + table(r"$|x|$") + "\n```\n\n"
        + "~~~text\n" + table(r"$\|x\|$") + "\n~~~\n\n"
        + r"仅看起来像行 | $|x|$ | 不是表格"
    )
    assert normalize_math_delimiters(source) == source


def test_indented_table_example_is_code_not_an_active_table() -> None:
    source = "\n".join("    " + line for line in table(r"$|x|$").splitlines())
    assert normalize_math_delimiters(source) == source


def test_tables_without_outer_pipes_keep_same_cells() -> None:
    source = "关系 | 条件 | 用途\n---|---|---\n频率 | $|s|f_1$ | 完整尾列"
    assert normalize_math_delimiters(source) == (
        "关系 | 条件 | 用途\n---|---|---\n频率 | $\\vert s\\vert f_1$ | 完整尾列"
    )


def test_one_column_table_still_protects_its_math() -> None:
    source = "| 内容 |\n|---|\n| 实际 $|s|f_1$ 完整尾句 |"
    assert normalize_math_delimiters(source) == (
        "| 内容 |\n|---|\n| 实际 $\\vert s\\vert f_1$ 完整尾句 |"
    )


def test_color_read_projection_preserves_full_real_failure_condition_and_tail() -> None:
    source = table(
        '<span style="color:#2563EB">转子电气频率；实际频率大小取 $|s|f_1$；完整条件</span>'
    )
    rendered = readable_color_markdown(source)

    assert r"\vert s\vert f_1" in rendered
    assert r"\color{#2563EB}" in rendered
    assert "完整条件" in rendered
    assert "| 计算频率 |" in rendered
    assert "$|s|f_1$" in source


def test_regular_read_renderer_uses_shared_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("src.math_display.st.markdown", calls.append)
    source = table(r"实际频率 $|s|f_1$，完整条件")

    render_math_markdown(source)

    assert calls == [table(r"实际频率 $\vert s\vert f_1$，完整条件")]
    assert "$|s|f_1$" in source
