"""Synthetic transport regressions; never contact a model or use a real library."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.ai.credential_store import MemoryCredentialStore, SecretCredential
from src.ai.model_registry import ProviderId
from src.ai.provider import AIExecutionError, CompletionDiagnostics
from src.ai.provider_config import (
    AIProviderConfig,
    ModelSelectionKind,
    ProviderConfigStore,
    ProviderSettings,
)
from src.ai.study_chat_adapter import CHAT_TIMEOUT_SECONDS, StudyChatProvider
from src.study_loop import chat
from src.study_loop.chat_store import ChatStore
from src.study_loop.components import import_source
from src.study_loop.models import StudyError
from src.study_loop.storage import StudyStore


@pytest.fixture
def local_chat(tmp_path: Path):
    study = StudyStore(tmp_path / "study", create=True)
    source = study.receive("NON_REAL.txt", b"1. NON_REAL question", "txt")
    import_source(study, source.id)
    chats = ChatStore(study)
    chats.set_course("NON_REAL_COURSE", [source.id], "")
    identity = chats.open_course("NON_REAL_COURSE")
    return study, source, chats, identity


@pytest.fixture
def configured_transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Exercise the actual adapter with one explicitly mocked HTTP boundary."""
    import nectivon_model_adapter

    import src.ai.provider_config as config_module
    import src.ai.study_review_adapter as review_module
    import src.runtime as runtime

    monkeypatch.setattr(config_module, "runtime_private_root", lambda: tmp_path)
    selected = ProviderSettings(
        provider_id=ProviderId.DEEPSEEK,
        model_id="deepseek-v4-pro",
        base_url="https://api.deepseek.com",
        model_selection=ModelSelectionKind.PRESET,
    )
    ProviderConfigStore().save(AIProviderConfig().with_provider(selected, make_active=True))
    credentials = MemoryCredentialStore()
    credentials.set("deepseek", SecretCredential("NON_REAL_TEST_CREDENTIAL"))
    monkeypatch.setattr(runtime, "application_credential_store", lambda: credentials)

    def forbidden():
        pytest.fail("Unexpected credential fallback")

    monkeypatch.setattr(review_module, "build_default_credential_store", forbidden)
    calls = []
    response = {"status": 200, "body": b""}

    class MockHttp:
        def __call__(self, request):
            calls.append(request)
            if "error" in response:
                raise nectivon_model_adapter.AdapterError(response["error"])
            return nectivon_model_adapter.HttpResponse(response["status"], response["body"])

    monkeypatch.setattr(nectivon_model_adapter, "HttpsTransport", MockHttp)
    return response, calls


@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, "authentication_failed"), (429, "rate_limited"), (503, "server_error")],
)
def test_http_failure_category_survives_adapter_without_retry(
    configured_transport, status: int, expected: str,
):
    response, calls = configured_transport
    response.update(status=status, body=b'{"error":{"message":"DO_NOT_EXPOSE"}}')
    with pytest.raises(AIExecutionError) as failure:
        StudyChatProvider(("deepseek", "deepseek-v4-pro")).complete("NON_REAL_QUESTION")
    assert failure.value.error_class == expected
    assert "DO_NOT_EXPOSE" not in str(failure.value)
    assert len(calls) == 1
    assert calls[0].timeout_seconds == CHAT_TIMEOUT_SECONDS == 300


def test_timeout_category_survives_transport_and_is_saved_safely(
    configured_transport, local_chat, caplog: pytest.LogCaptureFixture,
):
    from nectivon_model_adapter import ErrorCode

    response, calls = configured_transport
    response["error"] = ErrorCode.TIMEOUT
    _, _, chats, identity = local_chat
    with caplog.at_level(logging.ERROR):
        chat.send(
            chats, identity, "NON_REAL_QUESTION", "timeout-request",
            StudyChatProvider(("deepseek", "deepseek-v4-pro")), allow_remote=True,
        )
    message = chats.messages(identity)[-1]
    assert len(calls) == 1
    assert message["status"] == "error"
    assert "超时" in message["content"]
    assert "可能已计费" not in message["content"]
    assert message["metadata"]["error_code"] == "timeout"
    assert message["metadata"]["duration_ms"] >= 0
    assert "code=timeout" in caplog.text
    assert "NON_REAL_TEST_CREDENTIAL" not in caplog.text


def test_chat_deepseek_has_explicit_high_thinking_and_bounded_shared_output_budget(
    configured_transport,
):
    response, calls = configured_transport
    response["body"] = json.dumps({
        "model": "deepseek-v4-pro",
        "choices": [{"message": {"content": "NON_REAL_ANSWER"}, "finish_reason": "stop"}],
    }).encode()
    result = StudyChatProvider(("deepseek", "deepseek-v4-pro")).complete("NON_REAL_QUESTION")
    payload = json.loads(calls[0].body)
    assert payload["model"] == "deepseek-v4-pro"
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "max"
    assert payload["max_tokens"] == 32768
    assert len(calls) == 1 and calls[0].timeout_seconds == 300
    assert result.diagnostics.reasoning_effort == "max"
    assert result.diagnostics.output_limit == 32768


def test_full_chat_binding_sends_conversation_messages_through_frozen_adapter(configured_transport):
    from src.ai.conversation import encode_conversation

    response, calls = configured_transport
    response["body"] = json.dumps({
        "model": "deepseek-v4-pro",
        "choices": [{"message": {"content": "NON_REAL_ANSWER"}, "finish_reason": "stop"}],
    }).encode()
    messages = [
        {"role": "system", "content": "NON_REAL_POLICY"},
        {"role": "user", "content": "NON_REAL_MATERIAL"},
        {"role": "user", "content": "NON_REAL_PRIOR_QUESTION"},
        {"role": "assistant", "content": "NON_REAL_PRIOR_ANSWER"},
        {"role": "user", "content": "NON_REAL_CURRENT_QUESTION"},
    ]
    result = StudyChatProvider(("deepseek", "deepseek-v4-pro")).complete(
        encode_conversation(messages),
    )
    assert len(calls) == 1 and result.text == "NON_REAL_ANSWER"
    assert json.loads(calls[0].body)["messages"] == messages


def test_qwen_chat_does_not_receive_deepseek_thinking_options(configured_transport):
    import src.runtime as runtime

    response, calls = configured_transport
    runtime.application_credential_store().set("qwen", SecretCredential("NON_REAL_QWEN_KEY"))
    response["body"] = json.dumps({
        "model": "qwen3.8-max",
        "choices": [{"message": {"content": "NON_REAL_ANSWER"}, "finish_reason": "stop"}],
    }).encode()
    StudyChatProvider(("qwen", "qwen3.8-max")).complete("NON_REAL_QUESTION")
    payload = json.loads(calls[0].body)
    assert "thinking" not in payload and "reasoning_effort" not in payload
    assert payload["max_completion_tokens"] == 8192
    assert payload["enable_thinking"] is False


@pytest.mark.parametrize("connectivity", [False, True])
def test_regular_review_and_connectivity_do_not_inherit_chat_thinking_policy(
    configured_transport, connectivity,
):
    from nectivon_model_adapter import RequestBudget

    import src.runtime as runtime
    from src.ai.study_review_adapter import configured_review_provider

    response, calls = configured_transport
    response["body"] = json.dumps({
        "model": "deepseek-v4-pro",
        "choices": [{"message": {"content": "NON_REAL_ANSWER"}, "finish_reason": "stop"}],
    }).encode()
    provider = configured_review_provider(
        "deepseek", "deepseek-v4-pro", control=lambda: None, budget=RequestBudget(1),
        connectivity=connectivity, credential_store=runtime.application_credential_store(),
    )
    provider.review("NON_REAL_QUESTION")
    payload = json.loads(calls[0].body)
    assert "reasoning_effort" not in payload
    if connectivity:
        assert payload["thinking"] == {"type": "disabled"}
    else:
        assert "thinking" not in payload
    assert payload["max_tokens"] == 8192


def test_empty_length_limited_response_saves_numeric_diagnostics_without_reasoning(
    configured_transport, local_chat, caplog,
):
    response, calls = configured_transport
    reasoning = "NON_REAL_PRIVATE_REASONING_MUST_NOT_BE_SAVED"
    response["body"] = json.dumps({
        "model": "deepseek-v4-pro",
        "choices": [{"message": {"content": "", "reasoning_content": reasoning},
                     "finish_reason": "length"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 32768, "total_tokens": 32868,
                  "completion_tokens_details": {"reasoning_tokens": 32768}},
    }).encode()
    _, _, chats, identity = local_chat
    chat.send(
        chats, identity, "NON_REAL_QUESTION", "empty-reasoning-budget",
        StudyChatProvider(("deepseek", "deepseek-v4-pro")), allow_remote=True,
    )
    message = chats.messages(identity)[-1]
    assert message["status"] == "error"
    assert message["metadata"]["error_code"] == "output_truncated"
    diagnostic = message["metadata"]["diagnostics"]
    assert diagnostic["visible_output_chars"] == 0
    assert diagnostic["reasoning_output_chars"] == len(reasoning)
    assert diagnostic["completion_tokens"] == diagnostic["reasoning_tokens"] == 32768
    assert diagnostic["finish_reason"] == "length"
    assert diagnostic["http_status"] == 200
    assert reasoning not in json.dumps(chats.snapshot(identity)) + caplog.text
    assert len(calls) == 1


def test_neutral_diagnostics_cannot_contain_free_form_model_text():
    with pytest.raises(ValueError):
        CompletionDiagnostics(reasoning_tokens="NON_REAL_PRIVATE_REASONING")
    with pytest.raises(ValueError):
        CompletionDiagnostics(reasoning_effort="NON_REAL_PRIVATE_REASONING")


def test_usable_truncated_answer_is_retained_instead_of_becoming_generic_failure(
    configured_transport, local_chat,
):
    response, calls = configured_transport
    response["body"] = json.dumps({
        "model": "deepseek-v4-pro",
        "choices": [{
            "message": {"role": "assistant", "content": "NON_REAL_PARTIAL_ANSWER"},
            "finish_reason": "length",
        }],
    }).encode()
    _, _, chats, identity = local_chat
    chat.send(
        chats, identity, "NON_REAL_QUESTION", "partial-request",
        StudyChatProvider(("deepseek", "deepseek-v4-pro")), allow_remote=True,
    )
    message = chats.messages(identity)[-1]
    assert len(calls) == 1
    assert message["status"] != "error"
    assert "NON_REAL_PARTIAL_ANSWER" in message["content"]
    assert "继续" in message["content"]
    assert message["metadata"]["finish_reason"] == "length"


def test_empty_truncated_answer_stays_a_categorized_failure(configured_transport):
    response, calls = configured_transport
    response["body"] = json.dumps({
        "model": "deepseek-v4-pro",
        "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
    }).encode()
    with pytest.raises(AIExecutionError) as failure:
        StudyChatProvider(("deepseek", "deepseek-v4-pro")).complete("NON_REAL_QUESTION")
    assert failure.value.error_class == "output_truncated"
    assert len(calls) == 1


def test_unknown_exception_never_exposes_raw_message(local_chat, caplog):
    _, _, chats, identity = local_chat

    class Broken:
        def complete(self, *args, **kwargs):
            raise RuntimeError("SENSITIVE_RAW_ERROR")

    chat.send(chats, identity, "NON_REAL_QUESTION", "unknown", Broken(), allow_remote=True)
    message = chats.messages(identity)[-1]
    assert message["metadata"]["error_code"] == "execution"
    assert "SENSITIVE_RAW_ERROR" not in json.dumps(chats.snapshot(identity)) + caplog.text


@pytest.mark.parametrize("operation", ["send", "propose"])
def test_deleted_source_cannot_be_sent_from_a_saved_conversation(local_chat, operation: str):
    study, source, chats, identity = local_chat
    study.delete_source(source.id, user_confirmed=True)

    class Forbidden:
        def complete(self, *args, **kwargs):
            pytest.fail("Deleted source reached model")

    with pytest.raises(StudyError, match="已删除"):
        if operation == "send":
            chat.send(
                chats, identity, "NON_REAL_QUESTION", "deleted", Forbidden(), allow_remote=True,
            )
        else:
            chat.propose(chats, identity, Forbidden(), allow_remote=True)
    assert chats.messages(identity) == []
    assert study.read_source(source) == b"1. NON_REAL question"


def test_deleted_source_cannot_reach_direct_extraction(local_chat):
    from src.study_loop.extraction import extract_page

    study, source, _, _ = local_chat
    page = study.pages(source.id)[0]
    study.delete_source(source.id, user_confirmed=True)

    class Forbidden:
        def extract(self, *args, **kwargs):
            pytest.fail("Deleted source reached extraction model")

    with pytest.raises(StudyError, match="已删除"):
        extract_page(study, source.id, page.id, Forbidden(), allow_remote=True, use_image=False)


def test_deleted_source_cannot_reach_model_review_image_step(local_chat):
    from src.study_loop.model_review import _image

    study, source, _, _ = local_chat
    study.delete_source(source.id, user_confirmed=True)
    snapshot = {"source": {"id": source.id}, "input_mode": "text"}
    with pytest.raises(StudyError, match="已删除"):
        _image(study, snapshot)
