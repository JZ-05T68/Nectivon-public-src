"""Non-learning UI glyph fixtures; these are not study or model-quality evidence."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_reference_fields_render_separately_and_preserve_source_values() -> None:
    """Exercise the actual read panel with display-only glyphs and no database."""
    app = AppTest.from_string(
        r'''
from src.study_loop.models import Question
from src.study_loop.review_ui import _render_reference_content
import streamlit as st

unit = Question(
    answer_status="source_confirmed",
    reference_answer=r"[UI-only answer glyph] \(x\)",
    reference_analysis=r"[UI-only analysis glyph] \[y\]",
)
before = (unit.reference_answer, unit.reference_analysis)
_render_reference_content(unit)
st.session_state["source_unchanged"] = before == (unit.reference_answer, unit.reference_analysis)
'''
    ).run()
    assert not app.exception
    paragraphs = [element.value for element in app.markdown]
    assert paragraphs[0] == "**已核对的参考答案**"
    assert paragraphs[1] == "[UI-only answer glyph] $x$"
    assert paragraphs[2] == "**已核对的参考解析**"
    assert "$$y$$" in paragraphs[3]
    assert app.session_state["source_unchanged"] is True


def test_history_display_does_not_activate_markdown_image_references() -> None:
    """A display change must not make browser image fetches from imported text."""
    app = AppTest.from_string(
        r'''
from src.study_loop.display import render_saved_text
render_saved_text(r"[UI-only glyph] \(x\) ![external](https://example.invalid/image.png)")
'''
    ).run()
    assert not app.exception
    assert "$x$" in app.markdown[0].value
    assert "![external]" not in app.markdown[0].value
    assert "&#33;[external]" in app.markdown[0].value
    assert app.markdown[0].proto.allow_html is False


def test_verification_display_creates_revision_only_after_user_action() -> None:
    """Use UI-only record labels; never create materials or invent learning results."""
    app = AppTest.from_string(
        '''
import streamlit as st
from src.study_loop.models import Draft, Question
from src.study_loop.review_ui import _render_verification

class UiOnlyStore:
    def create_revision(self, item_id, *, user_requested):
        assert item_id == "ui-only-record" and user_requested is True
        st.session_state["revision_requests"] += 1

st.session_state.setdefault("revision_requests", 0)
item = Draft(
    id="ui-only-record", source_id="ui-only-source", page_ids=[],
    question=Question(content="[UI control fixture; no learning material]"),
    verification_status="error", verification_note="[UI status note fixture]",
)
_render_verification(UiOnlyStore(), item)
'''
    ).run()
    assert not app.exception
    assert app.caption[0].value == "知识验证状态：存在错误"
    assert app.session_state["revision_requests"] == 0
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["revision_requests"] == 1
    assert "原版本未改动" in app.success[0].value
    app.run()
    assert app.session_state["revision_requests"] == 1
