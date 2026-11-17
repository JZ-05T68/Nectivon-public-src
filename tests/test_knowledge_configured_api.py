"""Explicit domestic API authorization; synthetic transports never send HTTP."""

import json
from dataclasses import replace
from unittest.mock import Mock

import pytest

from src.ai.knowledge_generation import KnowledgeTaskBudget, KnowledgeWorkflowProvider
from src.ai.knowledge_generation_settings import (
    KnowledgeGenerationSettings,
    load_knowledge_generation_settings,
    save_knowledge_generation_settings,
)
from src.ai.provider import AIBudgetExceededError, AIUnavailableError


@pytest.mark.parametrize("provider,model", [
    ("deepseek", "deepseek-v4-pro"), ("qwen", "qwen3.8-max"),
])
def test_explicit_api_roundtrip_and_old_config_remains_fail_closed(
    tmp_path, monkeypatch, provider, model,
):
    import src.ai.knowledge_generation_settings as config

    monkeypatch.setattr(config, "runtime_private_root", lambda: tmp_path)
    settings = KnowledgeGenerationSettings(provider, model, configured_api=True)
    save_knowledge_generation_settings(settings)
    assert load_knowledge_generation_settings() == settings
    path = tmp_path / "knowledge_generation_settings.json"
    raw = json.loads(path.read_text())
    assert raw["grant"] is None and raw.pop("configured_api") is True
    assert not any("key" in field or "secret" in field for field in raw)
    path.write_text(json.dumps(raw))  # Existing version-one files lack this new field.
    legacy = load_knowledge_generation_settings()
    assert legacy is not None and legacy.configured_api is False
    with pytest.raises(AIBudgetExceededError):
        KnowledgeTaskBudget(None, provider=provider, model=model).reserve("JSON", 8)


@pytest.mark.parametrize("provider,model,authorization", [
    ("openai", "gpt-6.1-sol", True), ("deepseek", "deepseek-v4-pro", 1),
    ("qwen", "qwen3.8-max", "yes"),
])
def test_api_authorization_does_not_bypass_gpt_or_accept_truthy_values(
    provider, model, authorization,
):
    with pytest.raises(ValueError):
        KnowledgeGenerationSettings(provider, model, configured_api=authorization)
    with pytest.raises(ValueError):
        KnowledgeTaskBudget(None, provider=provider, model=model, configured_api=authorization)


def test_explicit_api_keeps_failed_call_reservation_and_token_limit():
    base = Mock()
    base.complete.side_effect = AIUnavailableError("synthetic failure")
    provider = KnowledgeWorkflowProvider(base, KnowledgeTaskBudget(
        None, provider="deepseek", model="deepseek-v4-pro", configured_api=True,
    ))
    with pytest.raises(AIUnavailableError):
        provider.complete("JSON", max_completion_tokens=8)
    with pytest.raises(AIBudgetExceededError):
        provider.complete("JSON", max_completion_tokens=8)
    assert base.complete.call_count == 1
    durable = Mock()
    budget = KnowledgeTaskBudget(None, provider="qwen", model="qwen3.8-max",
                                configured_api=True, max_calls=3, max_total_tokens=1048,
                                reserve_durable=durable)
    budget.reserve("JSON", 8)  # 4 bytes * 4 + 8 output + 1024 overhead = 1048.
    with pytest.raises(AIBudgetExceededError):
        budget.reserve("JSON", 8)
    durable.assert_not_called()  # Ordinary API use must not invent free credits.


def test_switching_model_requires_fresh_free_credit_verification(tmp_path, monkeypatch):
    from datetime import UTC, datetime, timedelta

    from streamlit.testing.v1 import AppTest

    import src.ai.knowledge_generation_settings as config
    from src.ai.knowledge_generation import VerifiedFreeCreditGrant

    monkeypatch.setattr(config, "runtime_private_root", lambda: tmp_path)
    now = datetime.now(UTC)
    grant = VerifiedFreeCreditGrant(
        "deepseek", "SYNTHETIC-PRO-ONLY", "deepseek-v4-pro",
        now - timedelta(days=1), now + timedelta(days=1), 200000, "SYNTHETIC-EVIDENCE",
        free_only_enforced=True, model_access_verified=True,
    )
    save_knowledge_generation_settings(KnowledgeGenerationSettings(
        "deepseek", "deepseek-v4-pro", grant,
    ))
    app = AppTest.from_string(
        "from src.provider_settings_ui import _render_knowledge_generation_settings\n"
        "_render_knowledge_generation_settings()\n",
    ).run()
    assert not app.exception
    assert next(item for item in app.checkbox if "访问权限" in item.label).value is True
    app.selectbox(key="knowledge_generation_model_deepseek").select("deepseek-flash").run()
    assert not app.exception
    assert next(item for item in app.checkbox if "访问权限" in item.label).value is False
    organization = next(item for item in app.text_input if item.label == "已核对的组织 / 业务空间")
    assert organization.value == ""
    next(item for item in app.button if item.label == "保存知识生成设置").click().run()
    saved = load_knowledge_generation_settings()
    assert saved is not None and saved.model == "deepseek-flash" and saved.grant is None


@pytest.mark.parametrize("provider,model,active", [
    ("deepseek", "deepseek-v4-pro", "qwen"), ("qwen", "qwen3.8-max", "deepseek"),
])
def test_factory_reads_explicit_route_preserves_global_audit_and_ordinary_selection(
    tmp_path, monkeypatch, provider, model, active,
):
    import src.ai.knowledge_generation as generation
    import src.ai.knowledge_generation_settings as config
    import src.ai.provider_config as provider_config
    from src import runtime
    from src.ai.credential_store import MemoryCredentialStore, SecretCredential
    from src.ai.model_registry import ProviderId
    from src.ai.provider_config import AIProviderConfig, ProviderConfigStore, ProviderSettings
    from src.config import Settings

    monkeypatch.setattr(config, "runtime_private_root", lambda: tmp_path)
    store = ProviderConfigStore(tmp_path / "providers.json")
    store.save(AIProviderConfig().with_provider(
        replace(ProviderSettings.default_for(provider), model_id=model), make_active=False,
    ).with_provider(ProviderSettings.default_for(active), make_active=True))
    credentials = MemoryCredentialStore()
    credentials.set(provider, SecretCredential("NON_REAL_TEST_KEY"))
    monkeypatch.setattr(provider_config, "ProviderConfigStore", lambda: store)
    monkeypatch.setattr(runtime, "application_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(runtime, "application_credential_store", lambda: credentials)
    ledger, guard, records = Mock(), Mock(), []
    monkeypatch.setattr(runtime, "_LazyDatabaseAiCallLedger", lambda: ledger)
    monkeypatch.setattr(runtime, "_LazyTokenBudgetGuard", lambda _: guard)
    wire = Mock(return_value={"model": model, "choices": [{
        "message": {"content": '{"points":[]}'}, "finish_reason": "stop",
    }], "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}})
    monkeypatch.setattr(generation, "urllib_transport", wire)
    monkeypatch.setattr(generation, "_record_local_options", records.append)
    save_knowledge_generation_settings(KnowledgeGenerationSettings(
        provider, model, configured_api=True,
    ))
    wrapped = generation.configured_knowledge_generation_provider()
    assert wrapped is not None and wrapped.provider_id == provider
    assert wrapped.complete("Return JSON", max_completion_tokens=8192).text == '{"points":[]}'
    assert wire.call_count == 1 and ledger.record.call_count == 1 and guard.mock_calls
    payload = wire.call_args.args[2]
    assert payload["model"] == model and payload["reasoning_effort"] == (
        "max" if provider == "deepseek" else "xhigh"
    )
    assert records[0]["requested_model"] == model
    assert store.load().active_provider_id is ProviderId(active)
    assert not (tmp_path / "knowledge_credit_reservations.json").exists()
    assert "NON_REAL_TEST_KEY" not in str(records)
