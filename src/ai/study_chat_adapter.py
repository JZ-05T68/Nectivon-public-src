"""Reuse the current model binding behind the neutral CompletionProvider contract."""

from __future__ import annotations

from src.ai.conversation import decode_conversation
from src.ai.provider import (
    AIExecutionError,
    AIUnavailableError,
    CompletionDiagnostics,
    CompletionResult,
    CompletionUsage,
)
from src.ai.provider_config import ProviderConfigStore
from src.ai.provider_scope import require_active_model

# A complete multi-turn answer can take longer than a connectivity probe. Keep
# one bounded request; a longer deadline must never become an automatic retry.
CHAT_TIMEOUT_SECONDS = 300
DEEPSEEK_CHAT_OUTPUT_TOKENS = 32768
# A character threshold is an operational latency policy, not a token estimate
# or a proof about the provider's timeout phase. Keep the established maximum
# effort for ordinary question/review inputs; full-document contexts use the
# provider's documented high effort under the same finite deadline. An explicit
# verification must not override this latency bound: max on a full historical
# document timed out in a real call. Never retry or reduce effort automatically.
LONG_CONTEXT_THINKING_CHARS = 16000

CHAT_FAILURE_CODES = frozenset({
    "timeout", "network_error", "rate_limited", "server_error", "authentication_failed",
    "credential_missing", "credential_store", "model_not_found", "model_access_denied",
    "quota_exhausted", "invalid_request", "invalid_configuration", "malformed_response",
    "empty_response", "output_truncated", "output_blocked",
})


def _usable_partial(text: object) -> bool:
    """Apply the text boundary before retaining a codec-rejected partial answer."""
    if not isinstance(text, str) or not text.strip() or len(text) > 1_048_576:
        return False
    if any(
        (ord(c) < 32 and c not in "\t\n\r") or 127 <= ord(c) <= 159 or 0xD800 <= ord(c) <= 0xDFFF
        for c in text
    ):
        return False
    return len(text.encode("utf-8")) <= 1_048_576


def _diagnostics(result: dict, limit: int, effort: str | None) -> CompletionDiagnostics:
    """Keep only bounded counters/settings; never carry model reasoning text."""
    def counter(value: object) -> int | None:
        return value if type(value) is int and 0 <= value <= 2**53 else None

    usage = result.get("usage") or {}
    usage = usage if isinstance(usage, dict) else {}
    reason = result.get("finish_reason")
    return CompletionDiagnostics(
        output_limit=limit,
        visible_output_chars=counter(result.get("visible_output_chars")),
        reasoning_output_chars=counter(result.get("reasoning_output_chars")),
        reasoning_tokens=counter(result.get("reasoning_tokens")),
        prompt_tokens=counter(usage.get("prompt_tokens")),
        completion_tokens=counter(usage.get("completion_tokens")),
        total_tokens=counter(usage.get("total_tokens")),
        duration_ms=counter(result.get("duration_ms")),
        http_status=counter(result.get("http_status")),
        finish_reason=(
            reason if reason in {"stop", "length", "content_filter", "sensitive"} else None
        ),
        thinking_mode="enabled" if effort and effort != "none" else None,
        reasoning_effort=effort,
    )


def configured_chat_model() -> tuple[str, str]:
    """Read the same active text selection as Settings, without opening a library.

    An explicit config (including an inactive/disabled selection) always wins
    over legacy environment settings. Reading this selection never calls a model.
    """
    store = ProviderConfigStore()
    try:
        config = store.load()
    except ValueError:
        raise AIUnavailableError("模型设置无法读取，请在「设置 → AI / 模型服务」中检查。") from None
    selected = config.get(config.active_provider_id) if config.active_provider_id else None
    if store.path.exists():
        if selected is None:
            raise AIUnavailableError("请在「设置 → 讲题 / 知识问答」中选择当前模型。")
        provider, model = selected.provider_id.value, selected.model_id
    else:
        from src.config import Settings, settings_env_path

        settings = Settings(_env_file=settings_env_path())
        if (
            settings.ai_mode != "api" or settings.ai_provider != "qwen"
            or not settings.ai_api_key.get_secret_value()
        ):
            raise AIUnavailableError("尚未配置对话模型，请前往「设置 → AI / 模型服务」。")
        provider, model = "qwen", settings.ai_llm_model
    require_active_model(provider, model)
    return provider, model


class StudyChatProvider:
    """One explicit request, no semantic or transport retries, no application DB."""

    def __init__(self, selection: tuple[str, str]) -> None:
        self.provider_id, self.model = selection
        require_active_model(self.provider_id, self.model)

    def complete(
        self, prompt: str, *, model: str | None = None, max_completion_tokens: int | None = None
    ) -> CompletionResult:
        from nectivon_model_adapter import RequestBudget

        from src.ai.study_review_adapter import configured_review_provider
        from src.runtime import application_credential_store

        try:
            messages = decode_conversation(prompt)
        except ValueError:
            raise AIUnavailableError("对话请求格式无效，未发送。") from None
        # Both modes retain thinking, the selected model, the full input and the
        # output allowance. This is a single-call policy, never a retry/fallback.
        input_chars = (sum(len(message["content"]) for message in messages)
                       if messages is not None else len(prompt))
        effort = ("high" if input_chars >= LONG_CONTEXT_THINKING_CHARS else "max"
                  ) if self.provider_id == "deepseek" else None
        output_limit = max_completion_tokens or (
            DEEPSEEK_CHAT_OUTPUT_TOKENS if self.provider_id == "deepseek" else 8192
        )
        options = {"reasoning_effort": effort} if effort else {}
        provider = configured_review_provider(
            self.provider_id,
            model or self.model,
            control=lambda: None,
            budget=RequestBudget(1),
            max_extra_attempts=0,
            timeout_seconds=CHAT_TIMEOUT_SECONDS,
            max_output_tokens=output_limit,
            credential_store=application_credential_store(),
            **options,
        )
        result = (
            provider.review(prompt, chat_messages=messages)
            if messages is not None else provider.review(prompt)
        )
        diagnostics = _diagnostics(result, output_limit, effort)
        # The frozen review codec deliberately rejects truncated answers, but a
        # dialogue should retain usable partial text and let the user continue.
        partial = (
            result.get("error_code") == "output_truncated"
            and result.get("finish_reason") == "length"
            and _usable_partial(result.get("text"))
        )
        if result["status"] != "ok" and not partial:
            code = result.get("error_code")
            code = code if code in CHAT_FAILURE_CODES else "execution"
            raise AIExecutionError(
                "模型请求未完成。",
                error_class=code,
                finish_reason=diagnostics.finish_reason,
                output_chars=diagnostics.visible_output_chars,
                output_tokens=diagnostics.completion_tokens,
                diagnostics=diagnostics,
            )
        usage = None
        if all(value is not None for value in (
            diagnostics.prompt_tokens, diagnostics.completion_tokens, diagnostics.total_tokens,
        )):
            usage = CompletionUsage(
                diagnostics.prompt_tokens, diagnostics.completion_tokens, diagnostics.total_tokens,
            )
        return CompletionResult(
            text=result["text"],
            model=model or self.model,
            resolved_model=result.get("reported_model"),
            finish_reason=result.get("finish_reason"),
            usage=usage,
            diagnostics=diagnostics,
        )
