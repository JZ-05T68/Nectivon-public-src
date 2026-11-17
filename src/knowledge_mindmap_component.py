"""Offline Streamlit bridge for the local SVG mind map."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

from src.appearance import active_theme, theme_tokens

_COMPONENT = components.declare_component(
    "nectivon_knowledge_mindmap",
    path=str(Path(__file__).parent / "components" / "knowledge_mindmap"),
)


def render_mindmap(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return selection/layout events only; no graph-edit command is supported."""
    # Presentation travels outside the graph contract and its saved layout identity.
    return _COMPONENT(payload=payload, theme=theme_tokens(active_theme()),
                      key=f"knowledge_mindmap_{payload['scope']}", default=None)
