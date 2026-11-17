"""Real colour selection, safe rendering and editable native Word equations, using only fixtures."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import zipfile
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from src.study_loop.color_markup import apply_color, color_runs, readable_color_markdown
from src.study_loop.word_math import markdown_docx


@pytest.mark.parametrize("color", ["red", "blue", "black"])
def test_real_selection_recolors_existing_text_and_keeps_operands(color):
    source = '普通 <span style="color:#2563EB">重点 $a+b=c$</span> 末尾\n'
    start = source.index("b=c")
    result = apply_color(source, start, start + 1, color)
    assert "".join(run.text for run in color_runs(result)) == "普通 重点 $a+b=c$ 末尾\n"
    formula = next(run for run in color_runs(result) if "$a+b=c$" in run.text)
    assert formula.color == {"red": "D62828", "blue": "2563EB", "black": None}[color]
    assert source.endswith("</span> 末尾\n")  # Caller-owned original was not edited.


def test_multiline_formula_and_table_list_structure_survive_real_selection():
    source = ("# 标题\n\n- 结论\n\n$$\n\\sqrt{x}+y=z\n$$\n\n"
              "| 公式 | 条件 |\n|---|---|\n| $a=b$ | 正数 |\n")
    result = apply_color(source, 0, len(source), "red")
    assert "".join(run.text for run in color_runs(result)) == source
    assert result.startswith('# <span style="color:#D62828">标题</span>')
    assert '\n- <span style="color:#D62828">结论</span>' in result
    assert '<span style="color:#D62828">$$\n\\sqrt{x}+y=z\n$$</span>' in result
    assert result.count("|") == source.count("|")
    assert '\n|---|---|\n' in result
    # No span may contain a table separator or the header separator row.
    assert "|" not in "".join(run.text for run in color_runs(result) if run.color)


def test_codes_and_unrelated_html_are_inert_during_selection_and_reading():
    source = ('<span style="color:#D62828">重点 $x$</span>\n\n'
              '```html\n<span style="color:#2563EB">$y$</span>\n```\n'
              '`<span style="color:#2563EB">$z$</span>`\n'
              '<img src="https://invalid.example/track" onerror="alert(1)">')
    result = readable_color_markdown(source)
    assert '{\\color{#D62828} x}' in result
    assert '```html\n<span style="color:#2563EB">$y$</span>\n```' in result
    assert '`<span style="color:#2563EB">$z$</span>`' in result
    assert '<img src=' not in result and '&lt;img src=' in result
    protected = source[source.index('```'):source.index('```', source.index('```')+3)+3]
    assert protected in apply_color(source, 0, len(source), "blue")


def test_unknown_color_is_rejected_and_empty_selection_does_not_rewrite():
    source = '<span style="color:#D62828">重点</span>\n'
    assert apply_color(source, 1, 1, "black") == source
    with pytest.raises(ValueError, match="黑色"):
        apply_color(source, 0, len(source), "00FF00")
    malformed = '<span onmouseover="x">原文</span>'
    assert "原文" in apply_color(malformed, 0, len(malformed), "red")


def test_colored_display_formula_has_own_fences_not_whitespace_html_wrappers():
    source = '<span style="color:#D62828">$$\n\\sqrt{x}=y\n$$</span>'
    rendered = readable_color_markdown(source)
    assert rendered.startswith('\n\n$$\n') and rendered.endswith('\n$$\n\n')
    assert '<span' not in rendered


def test_browser_selection_engine_matches_python_for_real_source_ranges():
    node = shutil.which("node")
    if not node:
        pytest.skip("本地Node未安装；真实浏览器选区另作验收")
    module = Path(__file__).parents[1] / "src/study_loop/color_editor_component/selection.js"
    examples = [
        ('前 $a+b=c$ 后\n', 5, 6, 'D62828'),
        ('<span style="color:#D62828">$x$提示</span>', 31, 34, '2563EB'),
        ('$$\n\\sqrt{x}\n$$\n| A | B |\n', 0, 23, 'D62828'),
        ('```js\nconst x="$a$";\n```\n普通', 0, 26, '2563EB'),
    ]
    script = ("const c=require(process.argv[1]);const xs=JSON.parse(process.argv[2]);"
              "process.stdout.write(JSON.stringify(xs.map(x=>c.applyColor(...x))));")
    result = subprocess.run([node, "-e", script, str(module), json.dumps(examples)],
                            capture_output=True, text=True, encoding="utf-8",
                            timeout=15, check=True)
    assert json.loads(result.stdout) == [apply_color(*example) for example in examples]


def test_word_real_font_colors_cover_inline_display_radical_and_table_equations():
    source = ('普通黑色\n\n<span style="color:#D62828">核心 $x=\\sqrt{a^2+b^2}$</span>\n\n'
              '<span style="color:#2563EB">$$\n\\frac{a}{b}=c\n$$</span>\n\n'
              '| 重点 | 条件 |\n|---|---|\n'
              '| <span style="color:#D62828">$u=v$</span> | '
              '<span style="color:#2563EB">$b\\ne0$</span> |\n')
    data = markdown_docx(source)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
          "m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        root = ET.fromstring(archive.read('word/document.xml'))
        assert archive.read('nectivon/transcript.md').decode() == source
        assert root.find('.//m:rad', ns) is not None
        assert root.find('.//m:f', ns) is not None
        equations = root.findall('.//m:oMath', ns)
        assert len(equations) == 4
        expected_colors = ['D62828', '2563EB', 'D62828', '2563EB']
        for equation, expected in zip(equations, expected_colors, strict=True):
            runs = equation.findall('.//m:r', ns)
            assert runs
            assert all(run.find('w:rPr/w:color', ns).get('{'+ns['w']+'}val') == expected
                       for run in runs)
            controls = equation.findall('.//m:ctrlPr/w:rPr/w:color', ns)
            assert all(control.get('{'+ns['w']+'}val') == expected for control in controls)
        assert root.find('.//m:radPr/m:ctrlPr/w:rPr/w:color', ns) is not None
        assert root.find('.//m:fPr/m:ctrlPr/w:rPr/w:color', ns) is not None
        assert {e.get('{'+ns['w']+'}val') for e in root.findall('.//w:color', ns)} == {
            '222222', 'D62828', '2563EB',
        }
        assert not any('<span' in (t.text or '') for t in root.findall('.//w:t', ns))


def test_saved_working_versions_and_markdown_reimport_preserve_colors(tmp_path):
    from test_ai_review_support import RecordingProvider, review_case

    from src.study_loop.ai_review import AIReviewService
    from src.study_loop.ai_review_export import markdown
    from src.study_loop.components import import_source

    case = review_case(tmp_path / 'source')
    service = AIReviewService(case.store)
    original = '<span style="color:#D62828">核心 $a=b$</span>\n'
    provider = RecordingProvider(original)
    document = service.generate([case.selected.id], provider)
    first = service.save_version(document['id'])
    changed = original + '<span style="color:#2563EB">仅在 $a>0$ 时适用</span>\n'
    service.update_working(document['id'], changed, expected_revision=document['working_revision'])
    second = service.save_version(document['id'])
    reopened = AIReviewService(case.store)
    assert reopened.document(document['id'])['working'] == changed
    assert reopened.version(first['id'])['content'] == original
    assert markdown(reopened, second['id']) == changed
    source = case.store.receive('三色复盘.md', changed.encode(), 'md')
    import_source(case.store, source.id)
    assert case.store.pages(source.id)[0].text == changed
    rendered = readable_color_markdown(case.store.pages(source.id)[0].text)
    assert '#D62828' in rendered and '#2563EB' in rendered
    assert '<span style=' in rendered and len(provider.prompts) == 1


def test_dialogue_export_uses_each_turns_memory_revision_without_future_projection():
    from src.study_loop.chat_export import study_markdown

    label = 'Mabcdef123456'
    def ref(revision: int) -> dict:
        return {'label': label, 'memory_id': 'fixture-memory', 'memory_revision': revision,
                'text': '# 同一记忆', 'name': '记忆', 'sha256': str(revision)}
    def message(role: str, content: str, refs: list[dict]) -> dict:
        return {'role': role, 'content': content, 'status': 'complete',
                'created_at': '2026-11-14T00:00:00Z', 'metadata': {'evidence': refs}}
    snapshot = {'conversation': {'kind': 'course', 'subject': '合成课程', 'context': {},
                                 'exam_notes': ''}, 'messages': [
        message('user', '第一问', [ref(2)]),
        message('assistant', '依据知识记忆['+label+']，第一轮结论。', []),
        message('user', '第二问', [ref(3)]),
        message('assistant', '依据知识记忆['+label+']，第二轮结论。', []),
    ]}
    before = deepcopy(snapshot)
    result = study_markdown(snapshot)
    first = result[result.index('## 2. AI'):result.index('## 3. 用户')]
    second = result[result.index('## 4. AI'):result.index('## 来源')]
    assert '第2修订' in first and '第3修订' not in first
    assert '第3修订' in second and '第2修订' not in second
    assert label not in result
    assert snapshot == before
