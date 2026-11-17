"""R3 model verification uses fake HTTP, isolated config/DB, and synthetic keys only."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from src.ai.completion_stage import CompletionStage, completion_stage_scope
from src.ai.credential_store import SecretCredential
from src.ai.deepseek_client import DeepSeekAdapter
from src.ai.knowledge_generation import (
    KnowledgeTaskBudget,
    KnowledgeWorkflowProvider,
    OpenAIKnowledgeProvider,
    VerifiedFreeCreditGrant,
    audited_knowledge_transport,
    require_openai_knowledge_request,
)
from src.ai.knowledge_generation_settings import (
    KnowledgeGenerationSettings,
    load_knowledge_generation_settings,
    reserve_credit_tokens,
    save_knowledge_generation_settings,
    set_knowledge_openai_credential,
)
from src.ai.model_registry import ProviderId, list_model_presets
from src.ai.provider import (
    AIBudgetExceededError,
    AIUnavailableError,
    ProviderCallError,
    build_production_audited_provider,
)
from src.ai.qwen_client import QwenProvider
from src.ai_ledger_service import AILedgerService
from src.database import Database
from src.models import AICallLedgerQuery


def _grant(**changes):
    now = datetime.now(UTC)
    return replace(
        VerifiedFreeCreditGrant(
            "deepseek",
            "synthetic-org",
            "deepseek-v4-pro",
            now - timedelta(days=1),
            now + timedelta(days=1),
            500000,
            "synthetic-control-check",
            True,
            True,
        ),
        **changes,
    )


def _chat(model):
    return {
        "model": model,
        "choices": [
            {
                "message": {"content": '{"points":[]}', "reasoning_content": "hidden"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 10,
            "total_tokens": 14,
            "completion_tokens_details": {"reasoning_tokens": 7},
        },
    }


@pytest.mark.parametrize("model", ["deepseek-flash", "deepseek-v4-pro"])
def test_deepseek_highest_effort_actual_wire_options(model):
    wire = Mock(return_value=_chat(model))
    provider = DeepSeekAdapter(
        api_key=SecretCredential("synthetic-only"), model=model, transport=wire
    )
    with completion_stage_scope(CompletionStage.KNOWLEDGE_GENERATION):
        result = provider.complete("Return JSON", max_completion_tokens=8192)
    options = wire.call_args.args[2]
    assert options["thinking"] == {"type": "enabled"}
    assert options["reasoning_effort"] == "max"
    assert options["response_format"] == {"type": "json_object"}
    assert options["max_tokens"] == 8192
    assert wire.call_args.args[3] >= 300
    assert result.diagnostics.reasoning_tokens == 7
    assert result.diagnostics.reasoning_effort == "max"


@pytest.mark.parametrize("model", ["qwen3.8-flash", "qwen3.8-max"])
def test_qwen_highest_effort_actual_wire_options(model):
    wire = Mock(return_value=_chat(model))
    provider = QwenProvider(
        api_key="synthetic-only",
        llm_model=model,
        llm_model_hard=model,
        embedding_model="unused",
        rerank_model="unused",
        transport=wire,
    )
    with completion_stage_scope(CompletionStage.KNOWLEDGE_GENERATION):
        result = provider.complete("Return JSON", max_completion_tokens=8192)
    options = wire.call_args.args[2]
    assert options["enable_thinking"] is True
    assert options["reasoning_effort"] == "xhigh"
    assert "thinking_budget" not in options
    assert wire.call_args.args[3] >= 300
    assert result.diagnostics.reasoning_tokens == 7
    assert result.diagnostics.reasoning_effort == "xhigh"


def test_ordinary_qwen_requests_do_not_acquire_high_effort():
    wire = Mock(return_value=_chat("qwen3.8-max"))
    provider = QwenProvider(
        api_key="synthetic-only",
        llm_model="qwen3.8-max",
        llm_model_hard="qwen3.8-max",
        embedding_model="unused",
        rerank_model="unused",
        transport=wire,
    )
    with completion_stage_scope(CompletionStage.PAGE_READING):
        provider.complete("Return JSON", max_completion_tokens=100)
    assert wire.call_args.args[2]["enable_thinking"] is False
    assert "reasoning_effort" not in wire.call_args.args[2]
    assert wire.call_args.args[3] < 300


def test_knowledge_default_output_budget_covers_reasoning_and_visible_json():
    body = _chat("deepseek-flash")
    body["usage"].update(completion_tokens=33000, total_tokens=33004)
    body["usage"]["completion_tokens_details"]["reasoning_tokens"] = 32000
    wire = Mock(return_value=body)
    adapter = DeepSeekAdapter(
        api_key=SecretCredential("synthetic-only"), model="deepseek-flash", transport=wire,
    )
    budget = KnowledgeTaskBudget(
        _grant(model="deepseek-flash"), provider="deepseek", model="deepseek-flash",
    )
    result = KnowledgeWorkflowProvider(adapter, budget).complete("Return JSON")
    assert wire.call_args.args[2]["max_tokens"] == 65536
    assert budget.max_total_tokens == 150000
    assert result.text and result.diagnostics.reasoning_tokens == 32000
    assert result.diagnostics.output_limit == 65536
    assert wire.call_count == 1


def test_knowledge_explicit_smaller_output_budget_is_preserved():
    base = Mock(return_value=None)
    base.complete.return_value = _chat("deepseek-flash")
    budget = KnowledgeTaskBudget(
        _grant(model="deepseek-flash"), provider="deepseek", model="deepseek-flash",
    )
    KnowledgeWorkflowProvider(base, budget).complete("Return JSON", max_completion_tokens=8192)
    assert base.complete.call_args.kwargs["max_completion_tokens"] == 8192


def _response(**changes):
    return {
        "model": "gpt-6.1-sol",
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "content": [{"type": "output_text", "text": "{}"}]},
        ],
        "usage": {
            "input_tokens": 4,
            "output_tokens": 12,
            "total_tokens": 16,
            "output_tokens_details": {"reasoning_tokens": 9},
        },
        **changes,
    }


def test_openai_responses_xhigh_is_not_chat_or_alias():
    wire = Mock(return_value=_response())
    provider = OpenAIKnowledgeProvider(SecretCredential("synthetic-only"), transport=wire)
    result = provider.complete("Return JSON", max_completion_tokens=8192)
    assert wire.call_args.args[0] == "https://api.openai.com/v1/responses"
    options = wire.call_args.args[2]
    assert options["model"] == "gpt-6.1-sol"
    assert options["reasoning"] == {"effort": "xhigh"}
    assert options["text"] == {"format": {"type": "json_object"}}
    assert options["store"] is False
    assert "messages" not in options and "temperature" not in options
    assert result.text == "{}" and result.resolved_model == "gpt-6.1-sol"
    assert result.diagnostics.reasoning_tokens == 9


def test_incomplete_openai_response_is_not_reported_complete():
    wire = Mock(
        return_value=_response(
            status="incomplete",
            incomplete_details={"reason": "max_output_tokens"},
        )
    )
    result = OpenAIKnowledgeProvider(SecretCredential("synthetic-only"), transport=wire).complete(
        "JSON"
    )
    assert result.finish_reason == "length"


def test_malformed_openai_response_reports_safe_error():
    wire = Mock(return_value={"status": "completed", "output": "not-a-list"})
    with pytest.raises(ProviderCallError):
        OpenAIKnowledgeProvider(SecretCredential("synthetic-only"), transport=wire).complete("JSON")


@pytest.mark.parametrize(
    "base,path,model",
    [
        ("https://evil.invalid/v1", "/responses", "gpt-6.1-sol"),
        ("https://api.openai.com/v1", "/chat/completions", "gpt-6.1-sol"),
        ("https://api.openai.com/v1", "/responses", "gpt-other"),
    ],
)
def test_openai_invalid_routes_fail_before_credential_or_transport(base, path, model):
    with pytest.raises(AIUnavailableError):
        require_openai_knowledge_request(
            base,
            path,
            {
                "model": model,
                "reasoning": {"effort": "xhigh"},
                "store": False,
            },
        )


@pytest.mark.parametrize(
    "grant",
    [
        None,
        _grant(free_only_enforced=False),
        _grant(model_access_verified=False),
        _grant(remaining_tokens=0),
        _grant(provider="qwen"),
        _grant(model="deepseek-flash"),
        _grant(expires_at=datetime.now(UTC) - timedelta(seconds=1)),
    ],
)
def test_unknown_or_inapplicable_credit_grants_never_call_provider(grant):
    base = Mock()
    provider = KnowledgeWorkflowProvider(
        base,
        KnowledgeTaskBudget(
            grant,
            provider="deepseek",
            model="deepseek-v4-pro",
        ),
    )
    with pytest.raises(AIBudgetExceededError):
        provider.complete("JSON")
    base.complete.assert_not_called()


def test_task_request_limit_shared_including_failed_calls():
    base = Mock()
    base.complete.side_effect = AIUnavailableError("synthetic failure")
    provider = KnowledgeWorkflowProvider(
        base,
        KnowledgeTaskBudget(
            _grant(),
            provider="deepseek",
            model="deepseek-v4-pro",
        ),
    )
    with pytest.raises(AIUnavailableError):
        provider.complete("JSON")
    with pytest.raises(AIBudgetExceededError):
        provider.complete("JSON")
    assert base.complete.call_count == 1


def test_credit_rejection_enters_existing_audit_ledger(tmp_path):
    database = Database(tmp_path / "isolated.db")
    wrapped = KnowledgeWorkflowProvider(
        Mock(),
        KnowledgeTaskBudget(
            None,
            provider="openai",
            model="gpt-6.1-sol",
        ),
    )
    ledger = Mock()
    ledger.record.side_effect = database.insert_ai_call
    provider = build_production_audited_provider(
        wrapped,
        default_model="gpt-6.1-sol",
        default_embedding_model="gpt-6.1-sol",
        source_feature="knowledge_generation",
        ledger=ledger,
        budget_guard=Mock(),
    )
    with pytest.raises(AIBudgetExceededError):
        provider.complete("JSON")
    rows = AILedgerService(database).query(AICallLedgerQuery(provider="openai"))
    assert rows.total == 1 and rows.entries[0].provider == "openai"
    assert rows.entries[0].status == "rejected"


def test_actual_wire_metadata_audit_has_unknown_cost_and_no_content_or_key():
    records = []
    wire = audited_knowledge_transport(Mock(return_value=_chat("deepseek-v4-pro")), records.append)
    wire(
        "https://api.deepseek.com/chat/completions",
        {"Authorization": "secret"},
        {
            "model": "deepseek-v4-pro",
            "messages": [{"content": "private prompt"}],
            "reasoning_effort": "max",
            "thinking": {"type": "enabled"},
        },
        10,
    )
    record = records[0]
    assert record["cost"] == "UNKNOWN" and record["reasoning_tokens"] == 7
    assert record["request_options"]["reasoning_effort"] == "max"
    assert "private prompt" not in str(records) and "secret" not in str(records)
    assert "hidden" not in str(records) and '{"points":[]}' not in str(records)


def test_settings_and_grant_reservations_are_local_nonsecret_and_cumulative(tmp_path, monkeypatch):
    import src.ai.knowledge_generation_settings as module

    monkeypatch.setattr(module, "runtime_private_root", lambda: tmp_path)
    grant = _grant(remaining_tokens=10000)
    settings = KnowledgeGenerationSettings("deepseek", "deepseek-v4-pro", grant)
    set_knowledge_openai_credential("synthetic-never-write-this")
    save_knowledge_generation_settings(settings)
    assert load_knowledge_generation_settings() == settings
    assert (
        "synthetic-never-write-this"
        not in (tmp_path / "knowledge_generation_settings.json").read_text()
    )
    reserve_credit_tokens(grant, 6000)
    with pytest.raises(AIBudgetExceededError):
        reserve_credit_tokens(replace(grant, evidence_ref="changed-model-check"), 6000)
    reserve_credit_tokens(grant, 1000)
    set_knowledge_openai_credential("")


@pytest.mark.parametrize("provider", ["kimi", "hunyuan", "glm"])
def test_retired_providers_never_appear_in_domestic_or_knowledge_workflow(provider):
    assert list_model_presets(ProviderId(provider)) == ()
    with pytest.raises(ValueError):
        KnowledgeGenerationSettings(provider, "historical")


def test_provider_specific_safe_errors_keep_openai_identity():
    from src.ai.openai_compatible import OpenAICompatibleTransportError

    wire = Mock(side_effect=OpenAICompatibleTransportError(status_code=403))
    with pytest.raises(ProviderCallError) as raised:
        OpenAIKnowledgeProvider(SecretCredential("synthetic-only"), transport=wire).complete("JSON")
    assert raised.value.detail.provider_id == "openai"


def test_configured_knowledge_factory_is_independent_audited_and_local(tmp_path, monkeypatch):
    import src.ai.knowledge_generation as module
    import src.ai.knowledge_generation_settings as config_module
    import src.ai.provider_config as provider_config
    from src import runtime
    from src.ai.credential_store import MemoryCredentialStore
    from src.ai.provider_config import AIProviderConfig, ProviderConfigStore, ProviderSettings
    from src.config import Settings

    monkeypatch.setattr(config_module, "runtime_private_root", lambda: tmp_path)
    store = ProviderConfigStore(tmp_path / "providers.json")
    store.save(
        AIProviderConfig()
        .with_provider(
            replace(ProviderSettings.default_for("deepseek"), model_id="deepseek-v4-pro"),
            make_active=False,
        )
        .with_provider(ProviderSettings.default_for("qwen"), make_active=True)
    )
    credentials = MemoryCredentialStore()
    credentials.set("deepseek", SecretCredential("synthetic-only"))
    monkeypatch.setattr(provider_config, "ProviderConfigStore", lambda: store)
    monkeypatch.setattr(runtime, "application_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(runtime, "application_credential_store", lambda: credentials)
    ledger = Mock()
    monkeypatch.setattr(runtime, "_LazyDatabaseAiCallLedger", lambda: ledger)
    monkeypatch.setattr(runtime, "_LazyTokenBudgetGuard", lambda _: Mock())
    wire = Mock(return_value=_chat("deepseek-v4-pro"))
    monkeypatch.setattr(module, "urllib_transport", wire)
    detail_records = []
    monkeypatch.setattr(module, "_record_local_options", detail_records.append)
    provider = module.configured_knowledge_generation_provider(
        provider="deepseek",
        model="deepseek-v4-pro",
        grant=_grant(),
    )
    assert provider is not None
    provider.complete("Return JSON", max_completion_tokens=8192)
    assert provider.provider_id == "deepseek" and provider.default_model == "deepseek-v4-pro"
    assert wire.call_args.args[2]["reasoning_effort"] == "max"
    assert detail_records and ledger.record.call_count == 1
    assert store.load().active_provider_id is ProviderId.QWEN
    assert (tmp_path / "knowledge_credit_reservations.json").is_file()
