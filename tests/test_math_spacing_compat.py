"""Notation-only Word/Pandoc spacing compatibility, never changing saved note bodies."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import streamlit

from src.math_display import normalize_math_delimiters, render_math_markdown
from src.study_loop.color_markup import readable_color_markdown


@pytest.mark.parametrize("dimension", ["6mu", "-3mu", "+0mu", ".25em", "1.5ex", "2pt", "1cm"])
@pytest.mark.parametrize("braced", [True, False])
def test_spacing_projects_exact_supported_length_and_preserves_formula(dimension, braced):
    command = r"\mspace" + ("{" + dimension + "}" if braced else dimension)
    source = "$a+b" + command + r"\Rightarrow\frac{x}{y}=z$"
    target = r"\mkern" if dimension.endswith("mu") else r"\hspace"
    expected = source.replace(command, target + "{" + dimension + "}")
    assert normalize_math_delimiters(source) == expected
    assert normalize_math_delimiters(expected) == expected
    assert command in source  # Stored/editable caller text is not replaced.


@pytest.mark.parametrize("command", [r"\mspace{6pixels}", r"\mspace{\value}",
                                    r"\mspace{em}", r"\mspace6mutex", r"\\mspace{6mu}"])
def test_invalid_or_literal_spacing_is_not_guessed(command):
    source = "$a" + command + "b$"
    assert normalize_math_delimiters(source) == source


def test_only_math_changes_and_code_and_matrix_row_breaks_remain_exact():
    code = "````latex\n" + r"$a\mspace{6mu}b$" + "\n````"
    source = code + "\n " + r"`\mspace6mu` and \mspace{6mu}; \[a\mspace6mu b\]"
    result = normalize_math_delimiters(source)
    assert result.startswith(code)
    assert r"`\mspace6mu` and \mspace{6mu};" in result
    assert r"a\mkern{6mu} b" in result
    matrix = r"$\begin{matrix}a&b\\\mspace6mu c&d\end{matrix}$"
    assert normalize_math_delimiters(matrix) == matrix.replace(r"\mspace6mu", r"\mkern{6mu}")


def test_table_display_and_colored_formula_share_the_same_projection(monkeypatch):
    source = '|内容|公式|\n|---|---|\n' + r'|合成|\[a\mspace{6mu}=b\]|'
    calls = []
    monkeypatch.setattr("src.math_display.st.markdown", lambda value: calls.append(value))
    render_math_markdown(source)
    assert len(calls[0].splitlines()) == 3
    assert r"$\displaystyle a\mkern{6mu}=b$" in calls[0]
    colored = '<span style="color:#2563EB">' + r'$a\mspace6mu=b$' + '</span>'
    rendered = readable_color_markdown(colored)
    assert r"\color{#2563EB}" in rendered and r"\mkern{6mu}" in rendered
    assert r"\mspace" in colored and r"\mspace" not in rendered


def test_exact_bundled_katex_rejects_original_command_and_accepts_projection():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is unavailable; browser rendering is validated separately")
    assets = Path(streamlit.__file__).parent / "static/static/js"
    candidates = [path for path in assets.glob("katex.*.js")
                  if "renderToString" in path.read_text(encoding="utf-8")]
    if not candidates:
        pytest.skip("Bundled KaTeX module was not located; browser validation remains required")
    source = r"a+b\mspace{6mu}\Rightarrow\frac{x}{y}=z"
    examples = [normalize_math_delimiters("$" + source + "$")[1:-1],
                normalize_math_delimiters(r"$a\mspace-.5em=b$")[1:-1]]
    script = (
        "const k=(await import(process.argv[1])).default;let rejected=false;"
        "try{k.renderToString(process.argv[2],{throwOnError:true})}catch(e){rejected=true}"
        "const results=JSON.parse(process.argv[3]).map(s=>{const html=k.renderToString(s,"
        "{throwOnError:true});return !html.includes('katex-error')});"
        "process.stdout.write(JSON.stringify({version:k.version,rejected,results}));"
    )
    completed = subprocess.run(
        [node, "--input-type=module", "-e", script, candidates[0].as_uri(), source,
         json.dumps(examples)], capture_output=True, text=True, encoding="utf-8",
        check=True, timeout=20,
    )
    result = json.loads(completed.stdout)
    assert result["rejected"] and result["results"] == [True, True]
