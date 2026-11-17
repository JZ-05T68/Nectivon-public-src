"""Chat follows instance text settings without real credentials, databases or I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import src.ai.credential_store as credential_module
import src.ai.provider_config as config_module
import src.ai.study_review_adapter as review_adapter
import src.config as config
import src.runtime as runtime
from src.ai.credential_store import MemoryCredentialStore, SecretCredential
from src.ai.model_registry import ProviderId
from src.ai.provider import AIExecutionError, AIUnavailableError
from src.ai.provider_config import (
    AIProviderConfig,
    ModelSelectionKind,
    ProviderConfigStore,
    ProviderSettings,
)
from src.ai.study_chat_adapter import StudyChatProvider, configured_chat_model
from src.config import Settings


def _selection(provider: str, model: str) -> ProviderSettings:
    return ProviderSettings(
        provider_id=ProviderId(provider),
        base_url=(
            "https://api.deepseek.com"
            if provider == "deepseek"
            else "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        model_id=model,
        model_selection=ModelSelectionKind.PRESET,
    )


@pytest.fixture
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Point every setting read at temporary files and forbid real runtime access."""
    import nectivon_model_adapter

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Chat model test reached a real database, credential chain or transport")

    monkeypatch.setattr(config_module, "runtime_private_root", lambda: tmp_path)
    monkeypatch.setattr(config, "settings_env_path", lambda: tmp_path / "absent.env")
    monkeypatch.setattr(runtime, "application_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(runtime, "application_database", forbidden)
    monkeypatch.setattr(credential_module, "build_default_credential_store", forbidden)
    monkeypatch.setattr(review_adapter, "build_default_credential_store", forbidden)
    monkeypatch.setattr(nectivon_model_adapter, "HttpsTransport", forbidden)
    credentials = MemoryCredentialStore()
    monkeypatch.setattr(runtime, "application_credential_store", lambda: credentials)
    return ProviderConfigStore(), credentials


@pytest.mark.parametrize(
    ("provider", "model"),
    [("deepseek", "deepseek-v4-pro"), ("qwen", "qwen3.8-max")],
)
def test_chat_uses_active_text_selection_and_same_cached_credentials(
    isolated_settings, monkeypatch: pytest.MonkeyPatch, provider: str, model: str,
) -> None:
    store, credentials = isolated_settings
    settings = _selection(provider, model)
    document = AIProviderConfig().with_provider(settings, make_active=True)
    # A distinct image model must not take over the text chat.
    document = document.with_image_provider(_selection("qwen", "qwen3.8-flash"))
    store.save(document)
    before = store.path.read_bytes()
    credentials.set(provider, SecretCredential("NON_REAL_TEST_DATA_CHAT_KEY"))
    if provider == "deepseek":
        assert credentials.get("qwen") is None
    calls: list[tuple[str, str, dict[str, Any]]] = []
    prompts: list[str] = []

    class Response:
        def review(self, prompt: str) -> dict[str, Any]:
            prompts.append(prompt)
            return {
                "status": "ok", "text": "NON_REAL_TEST_DATA_ANSWER",
                "reported_model": "NON_REAL_TEST_DATA_SERVER_ALIAS", "finish_reason": "stop",
            }

    def bind(selected_provider: str, selected_model: str, **kwargs: Any) -> Response:
        calls.append((selected_provider, selected_model, kwargs))
        assert kwargs["credential_store"] is credentials
        assert kwargs["credential_store"].get(selected_provider) is not None
        assert kwargs["max_extra_attempts"] == 0
        assert kwargs["budget"].remaining == 1
        assert kwargs["budget"].consume()
        assert not kwargs["budget"].consume()
        kwargs["control"]()
        return Response()

    monkeypatch.setattr(review_adapter, "configured_review_provider", bind)
    selected = configured_chat_model()
    assert selected == (provider, model)
    chat_provider = StudyChatProvider(selected)
    assert calls == []
    result = chat_provider.complete("NON_REAL_TEST_DATA_PROMPT", max_completion_tokens=1024)
    assert len(calls) == 1 and calls[0][:2] == (provider, model)
    assert calls[0][2]["max_output_tokens"] == 1024
    assert prompts == ["NON_REAL_TEST_DATA_PROMPT"]
    assert result.model == model
    assert result.resolved_model == "NON_REAL_TEST_DATA_SERVER_ALIAS"
    assert result.finish_reason == "stop"
    assert result.text == "NON_REAL_TEST_DATA_ANSWER"
    assert store.path.read_bytes() == before


def test_chat_selection_reloads_changed_settings_without_mutating_them(isolated_settings) -> None:
    store, _ = isolated_settings
    document = AIProviderConfig().with_provider(
        _selection("deepseek", "deepseek-v4-pro"), make_active=True,
    ).with_provider(_selection("qwen", "qwen3.8-max"))
    store.save(document)
    assert configured_chat_model() == ("deepseek", "deepseek-v4-pro")
    store.save(document.with_provider(_selection("qwen", "qwen3.8-max"), make_active=True))
    before = store.path.read_bytes()
    assert configured_chat_model() == ("qwen", "qwen3.8-max")
    assert store.path.read_bytes() == before


@pytest.mark.parametrize("stored_document", [False, True])
def test_missing_or_unselected_config_does_not_guess_qwen(
    isolated_settings, stored_document: bool,
) -> None:
    store, _ = isolated_settings
    if stored_document:
        store.save(AIProviderConfig().with_provider(_selection("qwen", "qwen3.8-max")))
    with pytest.raises(AIUnavailableError):
        configured_chat_model()


@pytest.mark.parametrize("disabled_provider", [False, True])
def test_disabled_selection_never_falls_back_to_other_saved_provider(
    isolated_settings, disabled_provider: bool,
) -> None:
    store, _ = isolated_settings
    selected = (
        ProviderSettings.default_for(ProviderId.KIMI)
        if disabled_provider
        else ProviderSettings(
            provider_id=ProviderId.DEEPSEEK,
            base_url="https://api.deepseek.com",
            model_id="NON_REAL_TEST_DATA_RETIRED_MODEL",
            model_selection=ModelSelectionKind.CUSTOM,
        )
    )
    document = AIProviderConfig().with_provider(selected, make_active=True).with_provider(
        _selection("qwen", "qwen3.8-max"),
    )
    store.save(document)
    before = store.path.read_bytes()
    with pytest.raises(AIUnavailableError):
        configured_chat_model()
    assert store.path.read_bytes() == before


def test_invalid_config_does_not_fall_back_to_legacy(
    isolated_settings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _ = isolated_settings
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{NON_REAL_TEST_DATA_BROKEN_JSON", encoding="utf-8")
    monkeypatch.setenv("EKB_AI_MODE", "api")
    monkeypatch.setenv("EKB_AI_API_KEY", "NON_REAL_TEST_DATA_LEGACY_KEY")
    with pytest.raises(AIUnavailableError):
        configured_chat_model()


def test_legacy_qwen_requires_explicit_api_mode_and_no_new_config(
    isolated_settings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _ = isolated_settings
    monkeypatch.setenv("EKB_AI_API_KEY", "NON_REAL_TEST_DATA_LEGACY_KEY")
    monkeypatch.setenv("EKB_AI_LLM_MODEL", "qwen3.8-flash")
    monkeypatch.setenv("EKB_AI_MODE", "manual")
    with pytest.raises(AIUnavailableError):
        configured_chat_model()
    monkeypatch.setenv("EKB_AI_MODE", "api")
    assert configured_chat_model() == ("qwen", "qwen3.8-flash")
    # An explicitly saved but unselected new configuration supersedes legacy.
    store.save(AIProviderConfig())
    with pytest.raises(AIUnavailableError):
        configured_chat_model()


@pytest.mark.parametrize("failure", ["result", "configuration"])
def test_failed_deepseek_request_does_not_retry_or_switch_provider(
    isolated_settings, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    calls: list[tuple[str, str]] = []
    prompts: list[str] = []

    class FailedResponse:
        def review(self, prompt: str) -> dict[str, str]:
            prompts.append(prompt)
            return {"status": "failed", "text": ""}

    def bind(provider: str, model: str, **kwargs: Any) -> FailedResponse:
        calls.append((provider, model))
        assert kwargs["max_extra_attempts"] == 0
        if failure == "configuration":
            raise AIUnavailableError("NON_REAL_TEST_DATA_MISSING_SELECTED_CREDENTIAL")
        return FailedResponse()

    monkeypatch.setattr(review_adapter, "configured_review_provider", bind)
    error = AIUnavailableError if failure == "configuration" else AIExecutionError
    with pytest.raises(error):
        StudyChatProvider(("deepseek", "deepseek-v4-pro")).complete("NON_REAL_TEST_DATA")
    assert calls == [("deepseek", "deepseek-v4-pro")]
    assert len(prompts) == (0 if failure == "configuration" else 1)


def test_real_review_binding_uses_injected_memory_credential_without_default_store(
    isolated_settings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the actual host binding; only its transport and adapter are replaced."""
    import nectivon_model_adapter

    store, credentials = isolated_settings
    store.save(AIProviderConfig().with_provider(
        _selection("deepseek", "deepseek-v4-pro"), make_active=True,
    ))
    credentials.set("deepseek", SecretCredential("NON_REAL_TEST_DATA_INJECTED_KEY"))
    read_providers: list[str] = []
    transport = object()
    adapter = object()
    seen: dict[str, Any] = {}
    budget = nectivon_model_adapter.RequestBudget(1)

    def get_credential(provider: str) -> SecretCredential | None:
        read_providers.append(provider)
        return MemoryCredentialStore.get(credentials, provider)

    def transport_factory(control, *, connectivity: bool):
        seen["connectivity"] = connectivity
        seen["control"] = control
        return transport

    def adapter_factory(selection, injected_credentials, selected_transport, **kwargs):
        seen.update(
            selection=selection, credentials=injected_credentials,
            transport=selected_transport, **kwargs,
        )
        return adapter

    monkeypatch.setattr(credentials, "get", get_credential)
    monkeypatch.setattr(review_adapter, "ReviewTransport", transport_factory)
    monkeypatch.setattr(nectivon_model_adapter, "ModelAdapter", adapter_factory)
    bound = review_adapter.configured_review_provider(
        "deepseek", "deepseek-v4-pro", control=lambda: None, budget=budget,
        max_extra_attempts=0, max_output_tokens=1024, credential_store=credentials,
    )
    assert read_providers == ["deepseek"]
    assert seen["selection"].provider.value == "deepseek"
    assert seen["selection"].model == "deepseek-v4-pro"
    supplied = seen["credentials"].get("deepseek")
    assert supplied is not None and supplied.reveal() == "NON_REAL_TEST_DATA_INJECTED_KEY"
    assert seen["credentials"].get("qwen") is None
    assert seen["transport"] is transport
    assert seen["budget"] is budget and budget.remaining == 1
    assert seen["policy"].max_extra_attempts == 0
    assert seen["policy"].max_output_tokens == 1024
    assert seen["connectivity"] is False
    assert bound.adapter is adapter and bound.transport is transport
    assert (bound.provider, bound.model) == ("deepseek", "deepseek-v4-pro")


def test_real_review_binding_rejects_missing_deepseek_key_without_qwen_fallback(
    isolated_settings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nectivon_model_adapter

    store, credentials = isolated_settings
    store.save(AIProviderConfig().with_provider(
        _selection("deepseek", "deepseek-v4-pro"), make_active=True,
    ).with_provider(_selection("qwen", "qwen3.8-max")))
    credentials.set("qwen", SecretCredential("NON_REAL_TEST_DATA_OTHER_PROVIDER_KEY"))
    monkeypatch.setenv("EKB_AI_MODE", "api")
    monkeypatch.setenv("EKB_AI_API_KEY", "NON_REAL_TEST_DATA_LEGACY_QWEN_KEY")
    reads: list[str] = []

    def get_credential(provider: str) -> SecretCredential | None:
        reads.append(provider)
        return MemoryCredentialStore.get(credentials, provider)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Missing selected-provider key must fail before transport or adapter creation")

    monkeypatch.setattr(credentials, "get", get_credential)
    monkeypatch.setattr(review_adapter, "ReviewTransport", forbidden)
    monkeypatch.setattr(nectivon_model_adapter, "ModelAdapter", forbidden)
    with pytest.raises(AIUnavailableError, match="API Key"):
        review_adapter.configured_review_provider(
            "deepseek", "deepseek-v4-pro", control=lambda: None,
            budget=nectivon_model_adapter.RequestBudget(1), credential_store=credentials,
        )
    assert reads == ["deepseek"]


def test_host_conversation_envelope_reaches_chat_review_as_exact_text_roles(
    isolated_settings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.ai.conversation import encode_conversation

    seen = {}
    messages = [
        {"role": "system", "content": "NON_REAL_POLICY"},
        {"role": "user", "content": "NON_REAL_SOURCE"},
        {"role": "user", "content": "NON_REAL_FIRST_QUESTION"},
        {"role": "assistant", "content": "NON_REAL_FIRST_ANSWER"},
        {"role": "user", "content": "NON_REAL_CURRENT_QUESTION"},
    ]
    envelope = encode_conversation(messages)

    class Response:
        def review(self, prompt, *, chat_messages):
            seen.update(prompt=prompt, messages=chat_messages)
            return {"status": "ok", "text": "NON_REAL_ANSWER", "finish_reason": "stop"}

    def bind(provider, model, **kwargs):
        seen.update(provider=provider, model=model, options=kwargs)
        return Response()

    monkeypatch.setattr(review_adapter, "configured_review_provider", bind)
    result = StudyChatProvider(("deepseek", "deepseek-v4-pro")).complete(envelope)
    assert seen["prompt"] == envelope and seen["messages"] == messages
    assert seen["provider"] == "deepseek" and seen["model"] == "deepseek-v4-pro"
    assert seen["options"]["max_output_tokens"] == 32768
    assert seen["options"]["reasoning_effort"] == "max"
    assert seen["options"]["timeout_seconds"] == 300
    assert seen["options"]["max_extra_attempts"] == 0
    assert result.text == "NON_REAL_ANSWER"


def test_malformed_host_conversation_is_rejected_without_transport(
    isolated_settings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("Malformed conversation reached model binding")

    monkeypatch.setattr(review_adapter, "configured_review_provider", forbidden)
    with pytest.raises(AIUnavailableError):
        StudyChatProvider(("deepseek", "deepseek-v4-pro")).complete(
            '{"schema":"nectivon.study-chat.v2","messages":[]}',
        )
