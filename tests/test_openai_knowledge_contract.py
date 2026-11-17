"""Sol wire contracts use synthetic credentials and fake Responses only, never HTTP.

Primary contracts checked 2026-11-16:
https://developers.openai.com/api/docs/models/gpt-6.1-sol
https://developers.openai.com/api/reference/resources/responses/methods/create
"""

from __future__ import annotations

import json
from copy import deepcopy
from unittest.mock import Mock

import pytest

from src.ai.credential_store import SecretCredential
from src.ai.knowledge_generation import (
    OPENAI_KNOWLEDGE_MODEL,
    OpenAIKnowledgeProvider,
    audited_knowledge_transport,
)
from src.ai.openai_compatible import (
    OpenAICompatibleClient,
    OpenAICompatibleTransportError,
    TransportFailureKind,
)
from src.ai.provider import AIUnavailableError, ProviderCallError, ProviderErrorCode

SYNTHETIC_KEY = "NON_REAL_TEST_KEY_sol_contract_20261116"
PRIVATE_PROMPT = "NON_REAL_TEST_PRIVATE_PROMPT: return one JSON object."
PRIVATE_REASONING = "NON_REAL_TEST_HIDDEN_REASONING"
PRIVATE_COMMENTARY = "NON_REAL_TEST_COMMENTARY"


def _message(text: str = '{"points":[]}', **changes) -> dict:
    return {"type": "message", "role": "assistant", "status": "completed",
            "phase": "final_answer", "content": [{"type": "output_text", "text": text}],
            **changes}


def _response(**changes) -> dict:
    return {"model": OPENAI_KNOWLEDGE_MODEL, "status": "completed",
            "output": [{"type": "reasoning", "summary": [{"text": PRIVATE_REASONING}]},
                       _message()],
            "usage": {"input_tokens": 7, "output_tokens": 12, "total_tokens": 19,
                      "output_tokens_details": {"reasoning_tokens": 8}}, **changes}


def _provider(wire: Mock) -> OpenAIKnowledgeProvider:
    return OpenAIKnowledgeProvider(SecretCredential(SYNTHETIC_KEY), transport=wire)


@pytest.mark.parametrize("change", [
    {"stream": True}, {"text": {"format": {"type": "text"}}}, {"store": True},
    {"reasoning": {"effort": "low"}}, {"model": "gpt-other"},
    {"max_output_tokens": True}, {"max_output_tokens": 128001},
    {"tools": [{"type": "web_search"}]}, {"previous_response_id": "NON_REAL_RESPONSE_ID"},
])
def test_request_scope_blocks_changed_contract_before_unwrapping_credential(change):
    wire, credential = Mock(), Mock(spec=SecretCredential)
    client = OpenAICompatibleClient(base_url="https://api.openai.com/v1", timeout_seconds=10,
                                    max_extra_attempts=0, transport=wire)
    payload = {"model": OPENAI_KNOWLEDGE_MODEL, "input": PRIVATE_PROMPT,
               "reasoning": {"effort": "xhigh"}, "max_output_tokens": 8192,
               "store": False, "stream": False, "text": {"format": {"type": "json_object"}},
               **change}
    with pytest.raises(AIUnavailableError):
        client.post("/responses", payload, credential=credential, provider_id="openai")
    wire.assert_not_called()
    credential.reveal.assert_not_called()


def test_official_exact_model_responses_json_payload_and_safe_audit(caplog):
    records = []
    wire = Mock(return_value=_response())
    result = _provider(audited_knowledge_transport(wire, records.append)).complete(
        PRIVATE_PROMPT, max_completion_tokens=16384,
    )
    url, headers, payload, timeout = wire.call_args.args
    assert url == "https://api.openai.com/v1/responses"
    assert headers["Authorization"] == "Bearer " + SYNTHETIC_KEY
    assert payload == {"model": OPENAI_KNOWLEDGE_MODEL, "input": PRIVATE_PROMPT,
                       "reasoning": {"effort": "xhigh"}, "max_output_tokens": 16384,
                       "store": False, "stream": False,
                       "text": {"format": {"type": "json_object"}}}
    assert timeout == 300 and wire.call_count == 1
    assert result.text == '{"points":[]}' and result.finish_reason == "stop"
    assert result.retry_count == 0 and result.diagnostics.reasoning_tokens == 8
    assert records[0]["finish_reason"] == "stop"
    assert records[0]["cost"] == "UNKNOWN"
    audit = json.dumps(records) + caplog.text
    assert all(secret not in audit for secret in (SYNTHETIC_KEY, PRIVATE_PROMPT,
                                                  PRIVATE_REASONING))


def test_only_final_phase_text_is_returned_with_multiple_output_items():
    wire = Mock(return_value=_response(output=[
        {"type": "reasoning", "summary": [{"text": PRIVATE_REASONING}]},
        _message(PRIVATE_COMMENTARY, phase="commentary"),
        _message('{"points":', content=[{"type": "output_text", "text": '{"points":'},
                                        {"type": "output_text", "text": "[]}"}]),
    ]))
    result = _provider(wire).complete(PRIVATE_PROMPT)
    assert json.loads(result.text) == {"points": []}
    assert PRIVATE_COMMENTARY not in result.text and PRIVATE_REASONING not in result.text
    assert wire.call_count == 1


def test_legacy_output_message_without_phase_remains_readable():
    message = _message()
    message.pop("phase")
    assert _provider(Mock(return_value=_response(output=[message]))).complete(
        PRIVATE_PROMPT,
    ).text == '{"points":[]}'


@pytest.mark.parametrize("reason,finish", [("max_output_tokens", "length"),
                                          ("content_filter", "content_filter")])
def test_known_incomplete_reasons_stay_incomplete_in_result_and_audit(reason, finish):
    records = []
    wire = Mock(return_value=_response(status="incomplete", output=[],
                                      incomplete_details={"reason": reason}))
    result = _provider(audited_knowledge_transport(wire, records.append)).complete(PRIVATE_PROMPT)
    assert result.finish_reason == finish and result.text == ""
    assert records[0]["finish_reason"] == finish and records[0]["output_truncated"]
    assert wire.call_count == 1


@pytest.mark.parametrize("reason", [None, "steered", "max_messages", "unknown", []])
def test_unhandled_incomplete_reason_is_not_invented_as_output_token_limit(reason):
    wire = Mock(return_value=_response(status="incomplete", incomplete_details={"reason": reason}))
    with pytest.raises(ProviderCallError) as captured:
        _provider(wire).complete(PRIVATE_PROMPT)
    assert captured.value.detail.code is ProviderErrorCode.PROVIDER_RESPONSE
    assert captured.value.retry_count == 0 and wire.call_count == 1


@pytest.mark.parametrize("response", [
    {"status": "failed", "error": {"code": "server_error", "message": SYNTHETIC_KEY},
     "output": []},
    _response(output=[_message(content=[{"type": "refusal", "refusal": PRIVATE_REASONING}])]),
])
def test_vendor_failure_and_refusal_are_safe_provider_failures(response, caplog):
    wire = Mock(return_value=response)
    with pytest.raises(ProviderCallError) as captured:
        _provider(wire).complete(PRIVATE_PROMPT)
    assert captured.value.detail.code is ProviderErrorCode.PROVIDER_RESPONSE
    assert SYNTHETIC_KEY not in str(captured.value) + caplog.text
    assert PRIVATE_REASONING not in str(captured.value) + caplog.text
    assert wire.call_count == 1


@pytest.mark.parametrize("status", ["cancelled", "queued", "in_progress"])
def test_nonterminal_or_cancelled_response_does_not_poll_or_invent_success(status):
    wire = Mock(return_value={"status": status, "output": []})
    with pytest.raises(ProviderCallError) as captured:
        _provider(wire).complete(PRIVATE_PROMPT)
    assert captured.value.detail.code is ProviderErrorCode.PROVIDER_RESPONSE
    assert wire.call_count == 1


@pytest.mark.parametrize("output", [
    "not-a-list", [_message(content=None)], [_message(content={})],
    [_message(content=[{"type": "output_text", "text": 123}])],
    [_message(), _message(content={"type": "output_text", "text": "bad"})],
    [_message(phase="unexpected_phase")],
    [_message(role="user")], [_message(status="incomplete")],
])
def test_malformed_output_blocks_without_leaking_content(output):
    wire = Mock(return_value=_response(output=deepcopy(output)))
    with pytest.raises(ProviderCallError) as captured:
        _provider(wire).complete(PRIVATE_PROMPT)
    assert captured.value.detail.code is ProviderErrorCode.MALFORMED_RESPONSE
    assert wire.call_count == 1


@pytest.mark.parametrize("status,expected", [
    (400, ProviderErrorCode.INVALID_REQUEST),
    (401, ProviderErrorCode.AUTHENTICATION_FAILED),
    (403, ProviderErrorCode.MODEL_ACCESS_DENIED),
    (404, ProviderErrorCode.MODEL_NOT_FOUND),
    (429, ProviderErrorCode.RATE_LIMITED),
    (503, ProviderErrorCode.PROVIDER_UNAVAILABLE),
])
def test_http_statuses_are_sanitized_and_never_retried(status, expected, caplog):
    records = []
    wire = Mock(side_effect=OpenAICompatibleTransportError(SYNTHETIC_KEY, status_code=status))
    with pytest.raises(ProviderCallError) as captured:
        _provider(audited_knowledge_transport(wire, records.append)).complete(PRIVATE_PROMPT)
    assert captured.value.detail.code is expected
    assert captured.value.detail.http_status == status
    assert captured.value.retry_count == 0 and wire.call_count == 1
    assert records[0]["http_status"] == status
    assert SYNTHETIC_KEY not in str(captured.value) + caplog.text + json.dumps(records)


def test_timeout_and_network_failures_are_safe_and_never_retried():
    for kind, expected in [(TransportFailureKind.TIMEOUT, ProviderErrorCode.TIMEOUT),
                           (TransportFailureKind.NETWORK, ProviderErrorCode.NETWORK_ERROR)]:
        wire = Mock(side_effect=OpenAICompatibleTransportError(SYNTHETIC_KEY, kind=kind))
        with pytest.raises(ProviderCallError) as captured:
            _provider(wire).complete(PRIVATE_PROMPT)
        assert captured.value.detail.code is expected
        assert captured.value.retry_count == 0 and wire.call_count == 1


def test_missing_credential_and_model_alias_block_before_transport():
    wire = Mock()
    with pytest.raises(AIUnavailableError):
        OpenAIKnowledgeProvider(None, transport=wire).complete(PRIVATE_PROMPT)
    with pytest.raises(AIUnavailableError):
        _provider(wire).complete(PRIVATE_PROMPT, model="gpt-other")
    wire.assert_not_called()


@pytest.mark.parametrize("limit", [0, -1, 128001, True, "8192"])
def test_invalid_output_limit_blocks_before_transport(limit):
    wire = Mock()
    with pytest.raises(ValueError):
        _provider(wire).complete(PRIVATE_PROMPT, max_completion_tokens=limit)
    wire.assert_not_called()


@pytest.mark.parametrize("limit", [1, 128000])
def test_documented_output_limit_boundaries_are_preserved(limit):
    wire = Mock(return_value=_response())
    _provider(wire).complete(PRIVATE_PROMPT, max_completion_tokens=limit)
    assert wire.call_args.args[2]["max_output_tokens"] == limit
    assert wire.call_count == 1


def test_malformed_transport_envelope_is_safe_and_audited_once():
    records = []
    wire = Mock(return_value=[PRIVATE_REASONING, SYNTHETIC_KEY])
    with pytest.raises(ProviderCallError) as captured:
        _provider(audited_knowledge_transport(wire, records.append)).complete(PRIVATE_PROMPT)
    assert captured.value.detail.code is ProviderErrorCode.MALFORMED_RESPONSE
    assert len(records) == wire.call_count == 1 and records[0]["status"] == "error"
    assert SYNTHETIC_KEY not in json.dumps(records) + str(captured.value)
