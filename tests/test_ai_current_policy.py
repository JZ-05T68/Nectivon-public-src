"""Live policy regressions; deliberately NEVER use the archived codec fixture."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from src.ai.credential_store import MemoryCredentialStore, SecretCredential
from src.ai.model_registry import ProviderId, get_provider_definition
from src.ai.openai_compatible import OpenAICompatibleClient
from src.ai.provider import AIUnavailableError
from src.ai.provider_config import (
    AIProviderConfig,
    ModelSelectionKind,
    ProviderConfigStore,
    ProviderSettings,
)
from src.ai.provider_factory import resolve_active_provider_runtime
from src.ai.provider_scope import ACTIVE_MODELS
from src.ai.provider_settings_service import ProviderSettingsService
from src.config import Settings

RETIRED = (ProviderId.KIMI, ProviderId.HUNYUAN, ProviderId.GLM)
BLOCKED_MODELS = (
    [
        (provider, preset.model_id)
        for provider in RETIRED
        for preset in get_provider_definition(provider).presets
    ]
    + [
        (ProviderId.DEEPSEEK, model)
        for model in (
            "deepseek-v4-flash",
            "deepseek-user-model",
            "deepseek-custom-model",
            "deepseek-custom-future",
            "console-confirmed-custom",
        )
    ]
    + [(ProviderId.QWEN, model) for model in ("qwen3.7-plus", "qwen-custom", "future-custom-model")]
)


@pytest.mark.parametrize("provider,model", BLOCKED_MODELS)
def test_live_client_blocks_retired_models_before_secret_or_transport(provider, model, monkeypatch):
    transport, reveal = Mock(), Mock(side_effect=AssertionError("Secret must stay wrapped"))
    monkeypatch.setattr(SecretCredential, "reveal", reveal)
    client = OpenAICompatibleClient(
        base_url=get_provider_definition(provider).default_base_url,
        timeout_seconds=1,
        max_extra_attempts=2,
        transport=transport,
    )
    with pytest.raises(AIUnavailableError, match="停用"):
        client.post(
            "/chat/completions",
            {"model": model},
            credential=SecretCredential("synthetic-current-policy"),
            provider_id=provider.value,
        )
    transport.assert_not_called()
    reveal.assert_not_called()


@pytest.mark.parametrize(
    "provider,model",
    [(provider, model) for provider, models in ACTIVE_MODELS.items() for model in models],
)
def test_live_client_allows_only_current_official_routes(provider, model):
    transport = Mock(return_value={"ok": True})
    client = OpenAICompatibleClient(
        base_url=get_provider_definition(provider).default_base_url,
        timeout_seconds=1,
        max_extra_attempts=2,
        transport=transport,
    )
    result, retries = client.post(
        "/chat/completions",
        {"model": model},
        credential=SecretCredential("synthetic-current-policy"),
        provider_id=provider.value,
    )
    assert result == {"ok": True} and retries == 0
    transport.assert_called_once()
    assert transport.call_args.args[2]["model"] == model


@pytest.mark.parametrize(
    "provider,url",
    [
        (provider, url)
        for provider in ACTIVE_MODELS
        for url in (
            "https://example.invalid/v1",
            "https://api.deepseek.com.evil.invalid/v1",
            "https://dashscope.aliyuncs.com.evil.invalid/compatible-mode/v1",
            "https://api.deepseek.com:444/v1",
            "https://api.deepseek.com/wrong",
        )
    ],
)
def test_unofficial_endpoint_never_receives_credentials(provider, url, monkeypatch):
    transport, reveal = Mock(), Mock(side_effect=AssertionError("Secret must stay wrapped"))
    monkeypatch.setattr(SecretCredential, "reveal", reveal)
    client = OpenAICompatibleClient(
        base_url=url,
        timeout_seconds=1,
        max_extra_attempts=2,
        transport=transport,
    )
    with pytest.raises(AIUnavailableError, match="官方 HTTPS"):
        client.post(
            "/chat/completions",
            {"model": ACTIVE_MODELS[provider][0]},
            credential=SecretCredential("synthetic-current-policy"),
            provider_id=provider.value,
        )
    transport.assert_not_called()
    reveal.assert_not_called()


@pytest.mark.parametrize("provider,model", BLOCKED_MODELS)
def test_settings_reject_retired_selection_without_changing_config_or_credential(
    tmp_path, provider, model
):
    store = ProviderConfigStore(tmp_path / "providers.json")
    credentials = MemoryCredentialStore()
    credentials.set(provider, SecretCredential("synthetic-original-key"))
    legacy = replace(
        ProviderSettings.default_for(provider),
        model_id=model,
        model_selection=(
            ModelSelectionKind.PRESET if provider in RETIRED else ModelSelectionKind.CUSTOM
        ),
    )
    store.save(AIProviderConfig().with_provider(legacy, make_active=True))
    before = store.path.read_bytes()
    transport, invalidate = Mock(), Mock()
    service = ProviderSettingsService(
        config_store=store,
        credential_store=credentials,
        application_settings=Settings(_env_file=None),
        transport=transport,
        invalidate_runtime=invalidate,
    )
    with pytest.raises(AIUnavailableError, match="停用"):
        service.save(legacy, new_api_key="synthetic-replacement-key", make_active=True)
    with pytest.raises(AIUnavailableError, match="停用"):
        service.test_connection(legacy, api_key_override="synthetic-override-key")
    assert store.path.read_bytes() == before
    assert credentials.get(provider).reveal() == "synthetic-original-key"
    assert store.load().get(provider) == legacy
    transport.assert_not_called()
    invalidate.assert_not_called()


@pytest.mark.parametrize("provider", RETIRED)
def test_retired_factory_has_no_fallback_and_preserves_history(tmp_path, provider):
    store = ProviderConfigStore(tmp_path / "providers.json")
    selected = ProviderSettings.default_for(provider)
    store.save(AIProviderConfig().with_provider(selected, make_active=True))
    credentials = MemoryCredentialStore()
    credentials.set(provider, SecretCredential("synthetic-retired-key"))
    before = store.path.read_bytes()
    resolved = resolve_active_provider_runtime(
        Settings(_env_file=None, ai_mode="api", ai_api_key="synthetic-fallback-must-not-win"),
        config_store=store,
        credential_store=credentials,
    )
    assert resolved is None
    assert store.path.read_bytes() == before
    assert credentials.get(provider).reveal() == "synthetic-retired-key"
