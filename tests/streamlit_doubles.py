"""Shared Streamlit test doubles for pages that run outside ``streamlit run``.

``tests/test_import_ui.py`` drives ``pages/1_导入资料.py`` with ``runpy``, i.e.
in Streamlit "bare mode". Bare mode silently no-ops most widgets, so a test
cannot assert on what the user would actually see unless the widgets are
replaced by a recorder.

The recorder is deliberately generic: the v0.8.8 import panel renders metrics,
progress bars, containers, expanders and per-file control buttons, and a
hand-written double that only knows ``button`` breaks the moment the panel
grows one more widget. ``Node`` therefore records *any* widget call and
returns a sensible value for the few that matter (``button`` -> the click
decision, ``columns`` -> one node per requested column, containers -> a node
that also works as a context manager).

Nothing here touches the real runtime: no ``application_settings()``, no
production data directory, no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
import streamlit as st

#: Widgets whose first positional argument is user-visible text.
_TEXT_WIDGETS = frozenset(
    {
        "markdown",
        "caption",
        "info",
        "warning",
        "error",
        "success",
        "title",
        "subheader",
        "header",
        "text",
        "write",
        "expander",
        "help",
    }
)


@dataclass(slots=True)
class Call:
    """One recorded widget invocation."""

    widget: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass(slots=True)
class StreamlitLog:
    """Everything a bare-mode page run rendered, in call order."""

    calls: list[Call] = field(default_factory=list)
    click_labels: set[str] = field(default_factory=set)
    uploads: list[Any] = field(default_factory=list)
    uploads_by_key: dict[str, list[Any]] = field(default_factory=dict)
    exceptions: list[str] = field(default_factory=list)

    # -------------------------------------------------------------- recording
    def record(self, widget: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.calls.append(Call(widget=widget, args=tuple(args), kwargs=dict(kwargs)))

    def clicked(self, label: str) -> bool:
        """Whether this run should treat ``label`` as a pressed button."""

        return label in self.click_labels

    # ------------------------------------------------------------- assertions
    def calls_of(self, widget: str) -> list[Call]:
        return [call for call in self.calls if call.widget == widget]

    def text_of(self, widget: str) -> list[str]:
        """Visible text rendered through one widget, in call order."""

        texts: list[str] = []
        for call in self.calls_of(widget):
            if call.args:
                texts.append(str(call.args[0]))
            elif "body" in call.kwargs:
                texts.append(str(call.kwargs["body"]))
        return texts

    def all_text(self, widgets: frozenset[str] = _TEXT_WIDGETS, *,
                 include_html: bool = False) -> str:
        """All user-visible text, for substring / forbidden-term assertions.

        Raw-HTML ``st.markdown`` bodies (the shared workspace shell injects a
        ``<style>`` block and a ``<div class="ekb-topbar">``) are skipped
        unless ``include_html`` is set, so an assertion about wording is never
        satisfied or broken by page chrome.
        """

        parts: list[str] = []
        for call in self.calls:
            texts: list[str] = []
            if call.widget in widgets and call.args:
                texts.append(str(call.args[0]))
            for key in ("text", "body", "label", "help"):
                if call.kwargs.get(key):
                    texts.append(str(call.kwargs[key]))
            for text in texts:
                if not include_html and text.lstrip().startswith("<"):
                    continue
                parts.append(text)
        for call in self.calls_of("metric"):
            if len(call.args) >= 2:
                parts.append(f"{call.args[0]}={call.args[1]}")
        for call in self.calls_of("progress"):
            if call.kwargs.get("text"):
                parts.append(str(call.kwargs["text"]))
        return "\n".join(parts)

    def metrics(self) -> dict[str, str]:
        """Rendered ``st.metric`` values keyed by label (last one wins)."""

        values: dict[str, str] = {}
        for call in self.calls_of("metric"):
            if len(call.args) >= 2:
                values[str(call.args[0])] = str(call.args[1])
        return values

    def progress_calls(self) -> list[tuple[Any, str]]:
        """Rendered ``st.progress`` calls as ``(value, text)``."""

        result: list[tuple[Any, str]] = []
        for call in self.calls_of("progress"):
            value = call.args[0] if call.args else call.kwargs.get("value")
            text = call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "")
            result.append((value, str(text or "")))
        return result

    def buttons(self) -> list[tuple[str, bool, str | None]]:
        """Rendered buttons as ``(label, disabled, key)`` — visibility proof."""

        result: list[tuple[str, bool, str | None]] = []
        for call in self.calls_of("button"):
            label = str(call.args[0]) if call.args else str(call.kwargs.get("label", ""))
            disabled = bool(call.kwargs.get("disabled", False))
            key = call.kwargs.get("key")
            result.append((label, disabled, key))
        return result

    def button_labels(self) -> list[str]:
        return [label for label, _, _ in self.buttons()]


class Node:
    """A container/column double that records any widget called on it."""

    def __init__(self, log: StreamlitLog) -> None:
        # ``object.__setattr__`` is not needed: slots are not used here so the
        # private attributes are found before ``__getattr__`` is consulted.
        self._log = log

    # ---------------------------------------------------- container protocol
    def __enter__(self) -> Node:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def __iter__(self):
        return iter(())

    # ---------------------------------------------------------- widget proxy
    def __getattr__(self, name: str):
        log = self.__dict__["_log"]

        def widget(*args: Any, **kwargs: Any) -> Any:
            log.record(name, args, kwargs)
            if name == "button":
                label = str(args[0]) if args else str(kwargs.get("label", ""))
                return log.clicked(label)
            if name == "columns":
                return _columns(log, args, kwargs)
            if name in ("container", "expander", "form", "sidebar", "popover"):
                return Node(log)
            if name == "progress":
                return Node(log)
            if name == "file_uploader":
                # A keyed uploader (the page's image-only picker) must not be
                # handed the general picker's files.
                key = kwargs.get("key")
                if key is None:
                    return list(log.uploads)
                return list(log.uploads_by_key.get(str(key), []))
            if name == "stop":
                raise SystemExit
            return None

        return widget


def _columns(log: StreamlitLog, args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[Node]:
    spec = args[0] if args else kwargs.get("spec", 1)
    if isinstance(spec, (list, tuple)):
        count = len(spec)
    elif isinstance(spec, int):
        count = spec
    else:
        count = 1
    return [Node(log) for _ in range(max(1, count))]


def install(
    monkeypatch: pytest.MonkeyPatch,
    log: StreamlitLog,
    *,
    patch: tuple[str, ...] = (
        "set_page_config",
        "title",
        "header",
        "subheader",
        "caption",
        "markdown",
        "write",
        "text",
        "info",
        "warning",
        "error",
        "success",
        "metric",
        "progress",
        "button",
        "columns",
        "container",
        "expander",
        "popover",
        "file_uploader",
        "toggle",
        "checkbox",
        "image",
        "switch_page",
        "rerun",
    ),
) -> StreamlitLog:
    """Route the listed ``st`` widgets into ``log`` for one test.

    ``st.stop`` is left alone unless the caller patches it, and
    ``st.session_state`` is left alone because bare mode already provides a
    non-functional stand-in that the pages tolerate.
    """

    root = Node(log)
    for name in patch:
        monkeypatch.setattr(st, name, getattr(root, name), raising=False)
    return log


__all__ = ["Call", "Node", "StreamlitLog", "install"]
