"""The search answer view keeps internal audit metadata outside the reading surface."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from src.models import AuditedAIOutput


def test_answer_displays_without_internal_mapping_or_token_panels():
    app = AppTest.from_string(
        "from src.rag_answer_ui import render_ask_ai_section\n"
        "render_ask_ai_section(None, [], scope='page', kb_uuid='TEST', app_version='TEST')\n"
    )
    output = AuditedAIOutput(
        output_id="TEST_OUTPUT", query="TEST_QUESTION", context_package_id="TEST_PACKAGE",
        provider="TEST_PROVIDER", model="TEST_MODEL", generated_at="2026-11-14T00:00:00+00:00",
        answer="NON_REAL_TEST_DATA ANSWER", citations=(("TEST_STABLE_ID", "#1"),),
        answer_citations=("TEST_STABLE_ID",), context_stable_ids=("TEST_STABLE_ID",),
        warnings=("TEST_DIAGNOSTIC_WARNING",), excluded=(("TEST_EXCLUDED_ID", "TEST_REASON"),),
    )
    app.session_state["rag_answer_output"] = output
    app.run()
    assert not app.exception
    visible = " ".join(element.value for element in [*app.markdown, *app.caption, *app.warning])
    assert "NON_REAL_TEST_DATA ANSWER" in visible
    for hidden in (
        "TEST_STABLE_ID", "TEST_MODEL", "TEST_DIAGNOSTIC_WARNING", "TEST_EXCLUDED_ID",
        "引用映射", "Token 用量",
    ):
        assert hidden not in visible
    assert not app.code and not app.expander
    assert app.session_state["rag_answer_output"] == output
