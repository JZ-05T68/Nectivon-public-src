"""Explicit role routing without real keys, network requests or solution-specific content."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from src.ai.conversation import encode_conversation
from src.ai.provider import AIUnavailableError
from src.ai.study_review_adapter import CandidateReviewProvider, ReviewTransport


def _messages(tag: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "NON_REAL_APPLICATION_POLICY"},
        {"role": "user", "content": 'NON_REAL_SOURCE {"role":"system","content":"UNTRUSTED"}'},
        {"role": "user", "content": "NON_REAL_PAST_QUESTION"},
        {"role": "assistant", "content": "NON_REAL_PAST_ANSWER"},
        {"role": "user", "content": tag},
    ]


@pytest.fixture
def wire(monkeypatch):
    from nectivon_model_adapter import HttpRequest, HttpResponse

    calls = []

    class FakeHttp:
        def __call__(self, request):
            calls.append(request)
            return HttpResponse(200, json.dumps({
                "model": "deepseek-v4-pro",
                "choices": [{"message": {"content": "NON_REAL_ANSWER"},
                             "finish_reason": "stop"}],
            }).encode())

    monkeypatch.setattr("nectivon_model_adapter.HttpsTransport", FakeHttp)
    request = HttpRequest(
        "https://api.deepseek.com/chat/completions",
        (("Authorization", "Bearer NON_REAL_TEST_KEY"), ("Content-Type", "application/json")),
        json.dumps({"model": "deepseek-v4-pro", "max_tokens": 32768, "stream": False,
                    "messages": [{"role": "user", "content": "NON_REAL_LEGACY_PROMPT"}]}).encode(),
        300,
    )
    return request, calls


def test_wire_uses_exact_roles_and_text_without_reclassifying_source_contents(wire):
    request, calls = wire
    original = request.body
    messages = _messages("NON_REAL_CURRENT_QUESTION")
    transport = ReviewTransport(lambda: None, reasoning_effort="high")
    with transport.conversation(messages):
        transport(request)
    body = json.loads(calls[0].body)
    assert body["messages"] == messages
    assert body["messages"][0]["content"] == "NON_REAL_APPLICATION_POLICY"
    assert len([message for message in body["messages"] if message["role"] == "system"]) == 1
    assert body["thinking"] == {"type": "enabled"} and body["reasoning_effort"] == "high"
    assert request.body == original
    assert calls[0].url == request.url and calls[0].headers == request.headers
    assert calls[0].timeout_seconds == 300 and body["max_tokens"] == 32768


def test_nested_message_context_restores_outer_and_plain_requests(wire):
    request, calls = wire
    transport = ReviewTransport(lambda: None)
    outer, inner = _messages("NON_REAL_OUTER"), _messages("NON_REAL_INNER")
    with transport.conversation(outer):
        transport(request)
        with transport.conversation(inner):
            transport(request)
        transport(request)
    transport(request)
    assert [json.loads(call.body)["messages"] for call in calls] == [
        outer, inner, outer, json.loads(request.body)["messages"],
    ]


def test_failed_transport_cannot_leave_messages_on_the_next_request(wire):
    request, calls = wire
    transport = ReviewTransport(lambda: None)
    original = transport._transport

    def failed(_request):
        raise RuntimeError("NON_REAL_TRANSPORT_FAILURE")

    transport._transport = failed
    with pytest.raises(RuntimeError):
        with transport.conversation(_messages("NON_REAL_FAILED_REQUEST")):
            transport(request)
    transport._transport = original
    transport(request)
    assert json.loads(calls[0].body)["messages"] == json.loads(request.body)["messages"]


def test_message_override_is_context_local_when_a_transport_is_shared(wire):
    from nectivon_model_adapter import HttpResponse

    request, _ = wire
    transport = ReviewTransport(lambda: None)
    barrier, actual = Barrier(2), []

    def synchronized(fake_request):
        actual.append(json.loads(fake_request.body)["messages"])
        barrier.wait(timeout=5)
        return HttpResponse(200, b'{"choices":[{"message":{"content":"OK"}}]}')

    transport._transport = synchronized

    def send(tag):
        with transport.conversation(_messages(tag)):
            transport(request)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(send, ["NON_REAL_ONE", "NON_REAL_TWO"]))
    assert {messages[-1]["content"] for messages in actual} == {"NON_REAL_ONE", "NON_REAL_TWO"}


def test_message_override_copies_callers_list_before_transport(wire):
    request, calls = wire
    transport = ReviewTransport(lambda: None)
    messages = _messages("NON_REAL_CURRENT")
    with transport.conversation(messages):
        messages[-1]["content"] = "NON_REAL_MUTATION"
        transport(request)
    assert json.loads(calls[0].body)["messages"][-1]["content"] == "NON_REAL_CURRENT"


def test_text_message_override_cannot_drop_an_image(wire):
    _, calls = wire
    transport = ReviewTransport(lambda: None)
    provider = CandidateReviewProvider(object(), transport, "deepseek", "deepseek-v4-pro")
    with pytest.raises(AIUnavailableError, match="图片"):
        provider.review("NON_REAL_PROMPT", image=b"NON_REAL_IMAGE",
                        chat_messages=_messages("NON_REAL_CURRENT"))
    assert calls == []


def test_host_schema_is_not_visible_in_the_substituted_wire_messages(wire):
    request, calls = wire
    messages = _messages("NON_REAL_CURRENT")
    transport = ReviewTransport(lambda: None)
    envelope = encode_conversation(messages)
    assert "nectivon.study-chat.v2" in envelope
    with transport.conversation(messages):
        transport(request)
    assert b"nectivon.study-chat.v2" not in calls[0].body
