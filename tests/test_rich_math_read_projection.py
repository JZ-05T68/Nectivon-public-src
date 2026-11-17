"""Keep table math, entities and safe breaks readable without changing stored bodies."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src.math_display import normalize_math_delimiters
from src.study_loop.color_markup import has_plain_line_break, readable_color_markdown


def test_colored_table_math_cannot_be_misread_as_a_cross_formula_html_tag():
    source = ('|公式|单位|条件|\n|---|---|---|\n'
              '|<span style="color:#D62828">$Q=(1-r)H$</span>|kW|'
              '同上；$0<r<1$ 时 $Q>0$|')
    result = readable_color_markdown(source)
    assert len(result.splitlines()) == 3 and result.count('|') == source.count('|')
    assert r'$0<r<1$ 时 $Q>0$' in result
    assert '&lt;' not in result and '&gt;' not in result
    assert r'$Q=(1-r)H$' in source
    assert r'\color{#D62828}' in result


@pytest.mark.parametrize('source,expected', [
    ('$0&lt;r&lt;1$', '$0<r<1$'),
    ('$Q&gt;0$', '$Q>0$'),
    ('$r&#60;1$', '$r<1$'),
    ('$r&#x3e;0$', '$r>0$'),
    ('$r&le;1$', '$r≤1$'),
])
def test_only_comparison_entities_inside_explicit_math_are_decoded_once(source, expected):
    assert normalize_math_delimiters(source) == expected
    assert normalize_math_delimiters(expected) == expected
    assert source != expected


def test_entities_cannot_decode_into_active_markup_or_new_math_delimiters():
    source = ('<span style="color:#2563EB">$0&lt;r&lt;1$</span>'
              '&lt;script&gt;alert(1)&lt;/script&gt;'
              '&amp;lt;img src=x&amp;gt;'
              '$a&dollar;b&dollar; &quot; &amp;lt;c$')
    result = readable_color_markdown(source)
    assert '<script>' not in result and '<img' not in result
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in result
    assert '&amp;lt;img src=x&amp;gt;' in result
    assert '$a&dollar;b&dollar; &quot; &amp;lt;c$' in result


def test_no_attribute_breaks_are_rebuilt_and_other_html_stays_inert():
    source = ('|公式|条件|\n|---|---|\n'
              '|<span style="color:#D62828">$x$</span>|<br>首行<br />次行|\n'
              '<br onclick="alert(1)"><img src="https://invalid.example/a" onerror="x">'
              '<script>bad()</script><img src="unterminated')
    result = readable_color_markdown(source)
    assert result.count('<br>') == 2 and len(result.splitlines()) == 4
    assert '<img' not in result and '<script>' not in result and '<br onclick' not in result
    assert '&lt;br onclick="alert(1)"&gt;' in result
    assert '&lt;img src="unterminated' in result


def test_code_links_and_undelimited_escaped_text_are_not_guessed_as_math():
    code = '````markdown\n' + r'| <br> $0&lt;r&lt;1$ Q\_{test}&gt;0 |' + '\n````'
    literal = r'`<br> $0&lt;r&lt;1$`'
    undelimited = r'Q\_{test}&gt;0'
    source = code + '\n\n' + literal + '\n\n' + undelimited
    result = readable_color_markdown(source)
    assert code in result and literal in result and undelimited in result
    assert not has_plain_line_break(source)
    assert '<br>' not in readable_color_markdown(undelimited)
    link = '[site](https://example.invalid/Q_test?x=1&y=2)'
    assert readable_color_markdown(link) == link


def test_shared_saved_reader_allows_only_safe_break_and_keeps_actual_table_math():
    app = AppTest.from_string(r'''
import streamlit as st
from src.study_loop.display import render_saved_text
source = ('|公式|条件|\n|---|---|\n'
          '|<span style="color:#D62828">$Q=(1-r)H$</span>|'
          '<br>同上；$0<r<1$ 时 $Q>0$|\n'
          '<img src="https://invalid.example/no-fetch" onerror="x">')
before = source
render_saved_text(source)
st.session_state['source_unchanged'] = source == before
''').run(timeout=30)
    assert not app.exception and not app.error
    body = app.markdown[0].value
    assert app.markdown[0].proto.allow_html
    assert '<br>同上；$0<r<1$ 时 $Q>0$' in body
    assert '<img' not in body and '&lt;img' in body
    assert app.session_state['source_unchanged']
