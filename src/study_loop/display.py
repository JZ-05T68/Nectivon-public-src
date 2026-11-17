"""Read-only math display for saved material, without loading linked images."""

from __future__ import annotations

import streamlit as st

from src.math_display import render_math_markdown


def render_saved_text(text: str) -> None:
    """Reuse math rendering while leaving stored text and external images untouched.

    Markdown image syntax can load remote URLs merely by displaying a history
    record. Escaping its opener keeps such references visible as text; actual
    source images are shown separately after their local hashes are verified.
    Only the three whitelisted note spans are projected to safe formatting;
    arbitrary stored HTML remains inert.
    """
    from .color_markup import color_runs, has_plain_line_break, readable_color_markdown

    value = str(text)
    with st.container():
        # Direct-child marker scopes default ink to this read body, never navigation
        # or status UI. Inline note spans and KaTeX retain their explicit red/blue.
        st.html('''<span data-nectivon-study-read></span><style>
        [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"]
          [data-nectivon-study-read]) {
            background:var(--ekb-surface,#ffffff);
        }
        [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"]
          [data-nectivon-study-read]) > [data-testid="stElementContainer"]
          [data-testid="stMarkdownContainer"] { color: var(--ekb-reading-ink, #222222); }
        [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"]
          [data-nectivon-study-read]) > [data-testid="stElementContainer"]
          [data-testid="stMarkdownContainer"] span:is(
            [style*="color:#D62828" i], [style*="color: #D62828" i],
            [style*="color: rgb(214, 40, 40)" i], [style*="color:rgb(214,40,40)" i]) {
            color:var(--ekb-note-red,#D62828) !important;
        }
        [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"]
          [data-nectivon-study-read]) > [data-testid="stElementContainer"]
          [data-testid="stMarkdownContainer"] span:is(
            [style*="color:#2563EB" i], [style*="color: #2563EB" i],
            [style*="color: rgb(37, 99, 235)" i], [style*="color:rgb(37,99,235)" i]) {
            color:var(--ekb-note-blue,#2563EB) !important;
        }
        [data-testid="stElementContainer"]:has(> [data-testid="stHtml"]
          [data-nectivon-study-read]) { display: none; }
        </style>''')
        if any(run.color for run in color_runs(value)) or has_plain_line_break(value):
            st.markdown(readable_color_markdown(value, inherit_default_ink=True),
                        unsafe_allow_html=True)
        else:
            render_math_markdown(value.replace("![", "&#33;["))
