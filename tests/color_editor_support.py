"""Bounded local component transport for AppTest; no model, service or DB substitutions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from streamlit.proto.WidgetStates_pb2 import WidgetStates
from streamlit.testing.v1 import AppTest


def _components(app: AppTest) -> list[tuple[Any, dict]]:
    return [(item, json.loads(item.proto.json_args)) for item in app.get("component_instance")
            if "nectivon_note_colors" in item.proto.component_name]


def _body_key(arguments: dict) -> str:
    return arguments["key"].removesuffix("_color_component")


def _packet(states: WidgetStates, identity: str, event: dict) -> None:
    value = next((item for item in states.widgets if item.id == identity), None)
    if value is None:
        value = states.widgets.add()
        value.id = identity
    value.json_value = json.dumps(event)


def _install_component_transport(app: AppTest) -> None:
    """Model the iframe's ready/input/flush replies, never a second native writer.

    A maximum of three local render/reply rounds handles one requested action;
    unresolved pending actions remain visible to assertions. No outcome is read
    from or written to a business store by this helper.
    """
    if vars(app).get("_color_transport_installed"):
        return
    original = app._run
    clients: dict[str, dict] = {}
    queued: dict[str, str] = {}
    app._color_client_values = clients
    app._color_queued_values = queued

    def client(arguments: dict) -> dict:
        key = _body_key(arguments)
        existing = clients.setdefault(key, {
            "text": arguments["value"], "serial": arguments.get("ack_serial", 0),
        })
        if key not in queued and arguments.get("ack_serial", 0) >= existing["serial"]:
            existing["text"] = arguments["value"]
        return existing

    def event(arguments: dict, content: str, token=None) -> dict:
        value = clients.setdefault(_body_key(arguments), {
            "text": content, "serial": arguments.get("ack_serial", 0),
        })
        value["serial"] = max(value["serial"], arguments.get("ack_serial", 0)) + 1
        value["text"] = content
        return {"text": content, "serial": value["serial"],
                "external_revision": arguments.get("external_revision", 0), "flush_token": token}

    def with_transport(widget_state: WidgetStates | None = None,
                       timeout: float | None = None) -> AppTest:
        if queued:
            if widget_state is None:
                widget_state = app._tree.get_widget_states()
            for component, arguments in _components(app):
                key = _body_key(arguments)
                if key in queued:
                    _packet(widget_state, component.proto.id, event(arguments, queued.pop(key)))
        result = original(widget_state=widget_state, timeout=timeout)
        for _ in range(3):
            states = app._tree.get_widget_states()
            reply = False
            for component, arguments in _components(app):
                key = _body_key(arguments)
                value = client(arguments)
                ready = app.session_state.filtered_state.get(key + "_color_ready", False)
                if arguments.get("flush_token") is not None or not ready:
                    _packet(states, component.proto.id,
                            event(arguments, value["text"], arguments.get("flush_token")))
                    reply = True
            if not reply:
                break
            result = original(widget_state=states, timeout=timeout)
        return result

    app._run = with_transport
    app._color_transport_installed = True


@dataclass
class ColorBody:
    """The visible component textarea, represented by its client input and render props."""

    app: AppTest
    label: str

    def _arguments(self) -> dict:
        return next(arguments for _, arguments in _components(self.app)
                    if arguments["label"] == self.label)

    @property
    def value(self) -> str:
        arguments = self._arguments()
        key = _body_key(arguments)
        return vars(self.app)["_color_queued_values"].get(key, arguments["value"])

    def set_value(self, content: str) -> ColorBody:
        arguments = self._arguments()
        vars(self.app)["_color_queued_values"][_body_key(arguments)] = content
        return self

    def run(self, *, timeout: float = 30) -> AppTest:
        return self.app.run(timeout=timeout)


def color_body(app: AppTest, label: str) -> ColorBody:
    _install_component_transport(app)
    value = ColorBody(app, label)
    value._arguments()  # Fail if the application did not render its actual editor.
    return value


def edit_color_body(app: AppTest, label: str, content: str) -> AppTest:
    return color_body(app, label).set_value(content).run()
