"""A local textarea with real selection colours and the existing persistence callbacks."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from src.appearance import active_theme, theme_tokens

from .models import StudyError

_COMPONENT = components.declare_component(
    "nectivon_note_colors", path=str(Path(__file__).parent / "color_editor_component"),
)


def _changed(key: str, component_key: str, limit: int,
             callback: Callable | None, args: tuple) -> None:
    event = st.session_state.get(component_key)
    if not isinstance(event, dict) or not isinstance(event.get("text"), str):
        return
    serial = event.get("serial")
    if (not isinstance(serial, int) or isinstance(serial, bool)
            or serial <= st.session_state.get(key + "_color_ack", 0)):
        return
    if event.get("external_revision", 0) != st.session_state.get(key + "_color_external", 0):
        return
    if len(event["text"]) > limit:
        st.session_state[key + "_color_error"] = "正文超过长度上限，输入仍保留在编辑框中。"
        return
    changed = event["text"] != st.session_state.get(key)
    st.session_state[key] = event["text"]
    st.session_state[key + "_color_ack"] = serial
    st.session_state[key + "_color_ready"] = True
    if changed and callback is not None:
        callback(*args)
    pending = st.session_state.get(key + "_color_pending")
    if pending and event.get("flush_token") == pending["token"]:
        st.session_state[key + "_color_flushed"] = {
            "text": event["text"], "serial": serial,
            "external_revision": event.get("external_revision", 0), "token": pending["token"],
        }
        st.session_state.pop(key + "_color_pending", None)
        deactivate_color_editor(key)
        pending["callback"](*pending["args"], **pending["kwargs"])


def confirmed_color_body(key: str, *, flushed: bool = False) -> str:
    """Return an action's exact acknowledged payload, never a native-widget replay."""
    if not flushed:
        return str(st.session_state.get(key, ""))
    receipt = st.session_state.get(key + "_color_flushed")
    current_external = st.session_state.get(key + "_color_external", 0)
    if (not isinstance(receipt, dict) or not isinstance(receipt.get("text"), str)
            or receipt.get("serial") != st.session_state.get(key + "_color_ack")
            or receipt.get("external_revision") != current_external):
        raise StudyError("编辑器最新输入尚未确认，未保存旧稿。当前输入仍保留，请完成同步后保存。")
    st.session_state[key] = receipt["text"]
    return receipt["text"]


def defer_color_action(key: str, callback: Callable, *, args: tuple = (),
                       kwargs: dict | None = None) -> bool:
    """Flush text and action atomically before a currently visible editor is unmounted.

    The single local editor must acknowledge its current body even before
    its first ready event. Legacy auxiliary-widget flags never authorize a save.
    """
    if (not st.session_state.get(key + "_color_ready")
            and not st.session_state.get(key + "_color_mounted")):
        return False
    if key + "_color_pending" not in st.session_state:
        token = st.session_state.get(key + "_color_next_action", 0) + 1
        st.session_state[key + "_color_next_action"] = token
        st.session_state[key + "_color_pending"] = {
            "token": token, "callback": callback, "args": args, "kwargs": kwargs or {},
        }
    return True


def deactivate_color_editor(key: str) -> None:
    """Clear a route's mounted-editor marker without deleting its input or acknowledgements."""
    if st.session_state.get("sl_active_color_editor") == key:
        st.session_state.pop("sl_active_color_editor", None)
    st.session_state[key + "_color_mounted"] = False


def defer_active_color_action(callback: Callable, *, args: tuple = (),
                              kwargs: dict | None = None) -> bool:
    """Let a route-level navigation wait for the last actually rendered editor."""
    key = st.session_state.get("sl_active_color_editor")
    return bool(key and defer_color_action(key, callback, args=args, kwargs=kwargs))


def render_color_editor(
    label: str, *, key: str, height: int = 260, max_chars: int = 30000,
    on_change: Callable | None = None, args: tuple = (),
) -> Any:
    """Edit the caller's same body key; preview/save keep their existing local semantics."""
    component_key = key + "_color_component"
    value = str(st.session_state.get(key, ""))
    render_serial = st.session_state.get(key + "_color_render", 0) + 1
    st.session_state[key + "_color_render"] = render_serial
    pending = st.session_state.get(key + "_color_pending")
    st.session_state["sl_active_color_editor"] = key
    st.session_state[key + "_color_mounted"] = True
    result = _COMPONENT(
        label=label, value=value, height=height, max_chars=max_chars,
        palette=theme_tokens(active_theme()),
        ack_serial=st.session_state.get(key + "_color_ack", 0), render_serial=render_serial,
        external_revision=st.session_state.get(key + "_color_external", 0),
        flush_token=pending["token"] if pending else None,
        key=component_key, default=None,
        on_change=lambda: _changed(key, component_key, max_chars, on_change, args),
    )
    if error := st.session_state.pop(key + "_color_error", None):
        st.warning(error)
    if pending:
        st.caption("正在同步最新输入…")
    return result
