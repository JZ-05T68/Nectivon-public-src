"""Explicit offline seam for preserved, retired wire-format regression tests.

This is NOT an application-policy fixture. Only codec test modules opt in.
Factories, settings, selectors and test_ai_current_policy use the real policy.
Every legacy request still runs the real serialization, parsing and retry loop;
the fixture requires a synthetic transport and a finite historical route.
"""

import urllib.request
from collections.abc import Mapping
from typing import Any

import pytest

from src.ai.model_registry import ProviderId, get_provider_definition
from src.ai.openai_compatible import OpenAICompatibleClient, urllib_transport
from src.ai.provider_scope import ACTIVE_MODELS

_ARCHIVED_MODELS = {
    "deepseek": {
        "deepseek-user-model", "deepseek-custom-model", "deepseek-v4-flash",
        "deepseek-custom-future",
    },
    "qwen": {"future-custom-model", "qwen3.7-plus", "qwen-custom"},
    "kimi": {"kimi-k3", "kimi-k2.6", "future-custom-model"},
    "hunyuan": {"hy4-preview", "hy3", "hunyuan-future"},
    "glm": {"glm-5.3", "glm-5.3-flash", "future-custom-glm"},
}


@pytest.fixture
def archived_wire_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Supply an offline historical policy only for codec tests that request it."""
    original = OpenAICompatibleClient._validate_request_policy

    def validate(
        client: OpenAICompatibleClient, path: str, payload: Mapping[str, Any], provider_id: str,
    ) -> None:
        model = str(payload.get("model", ""))
        if model in ACTIVE_MODELS.get(ProviderId(provider_id), ()):
            original(client, path, payload, provider_id)
            return
        assert path == "/chat/completions"
        assert model in _ARCHIVED_MODELS[provider_id], "Unreviewed historical model fixture"
        assert client._base_url == get_provider_definition(provider_id).default_base_url
        transport = client._transport
        if transport is urllib_transport:
            assert urllib.request.urlopen.__module__ != "urllib.request", "Real network forbidden"
        else:
            module = getattr(transport, "__module__", type(transport).__module__)
            assert module.startswith(("tests.", "test_")) or (
                module.startswith("src.ai.")
                and getattr(transport, "__name__", "") == "_unconfigured_transport"
            ), "Archived codec test must use a fake transport"

    monkeypatch.setattr(OpenAICompatibleClient, "_validate_request_policy", validate)
