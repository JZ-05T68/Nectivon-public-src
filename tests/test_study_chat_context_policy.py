"""Long-context calls keep the same model, evidence, finite deadline and one-call budget."""
from __future__ import annotations

import pytest

from src.ai.conversation import encode_conversation
from src.ai.study_chat_adapter import (
    CHAT_TIMEOUT_SECONDS,
    DEEPSEEK_CHAT_OUTPUT_TOKENS,
    LONG_CONTEXT_THINKING_CHARS,
    StudyChatProvider,
)


@pytest.mark.parametrize('size,effort', [
    (LONG_CONTEXT_THINKING_CHARS - 1, 'max'),
    (LONG_CONTEXT_THINKING_CHARS, 'high'),
    (LONG_CONTEXT_THINKING_CHARS + 9000, 'high'),
])
@pytest.mark.parametrize('encoded', [True, False])
@pytest.mark.parametrize('verification', [True, False])
def test_context_policy_preserves_entire_input_and_never_retries(
    monkeypatch, size, effort, encoded, verification,
):
    import src.ai.study_review_adapter as adapter
    import src.runtime as runtime

    question = '请核验这些公式。' if verification else ''
    messages = [{'role': 'system', 'content': 'NON_REAL_POLICY'},
                {'role': 'user', 'content': question + 'x' * (
                    size - len('NON_REAL_POLICY') - len(question))}]
    prompt = encode_conversation(messages) if encoded else 'x' * size
    credentials = object()
    calls, inputs = [], []
    monkeypatch.setattr(runtime, 'application_credential_store', lambda: credentials)

    class Response:
        def review(self, value, **kwargs):
            inputs.append((value, kwargs))
            return {'status': 'ok', 'text': 'NON_REAL_ANSWER', 'finish_reason': 'stop'}

    def bind(provider, model, **kwargs):
        calls.append((provider, model, kwargs))
        return Response()

    monkeypatch.setattr(adapter, 'configured_review_provider', bind)
    result = StudyChatProvider(('deepseek', 'deepseek-v4-pro')).complete(prompt)
    assert len(calls) == 1
    provider, model, options = calls[0]
    assert (provider, model) == ('deepseek', 'deepseek-v4-pro')
    assert options['reasoning_effort'] == effort
    assert options['credential_store'] is credentials
    assert options['max_extra_attempts'] == 0
    assert options['budget'].remaining == 1
    assert options['timeout_seconds'] == CHAT_TIMEOUT_SECONDS == 300
    assert options['max_output_tokens'] == DEEPSEEK_CHAT_OUTPUT_TOKENS == 32768
    assert inputs == [(prompt, {'chat_messages': messages} if encoded else {})]
    assert result.diagnostics.reasoning_effort == effort


def test_non_deepseek_call_keeps_its_existing_options(monkeypatch):
    import src.ai.study_review_adapter as adapter
    import src.runtime as runtime

    calls = []
    monkeypatch.setattr(runtime, 'application_credential_store', object)

    class Response:
        def review(self, value):
            return {'status': 'ok', 'text': 'NON_REAL', 'finish_reason': 'stop'}

    def bind(provider, model, **kwargs):
        calls.append(kwargs)
        return Response()

    monkeypatch.setattr(adapter, 'configured_review_provider', bind)
    StudyChatProvider(('qwen', 'qwen3.8-max')).complete('x' * LONG_CONTEXT_THINKING_CHARS)
    assert len(calls) == 1 and 'reasoning_effort' not in calls[0]
    assert calls[0]['max_output_tokens'] == 8192
