"""Map the frozen Model Adapter public contract to Nectivon's neutral extraction protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.ai.provider import AIExecutionError, AIUnavailableError, CompletionResult, CompletionUsage

if TYPE_CHECKING:
    from nectivon_model_adapter import ModelAdapter


class CandidateExtractionProvider:
    """An explicit Qwen-only host binding; endpoint/payload/parsing remain in the component."""

    def __init__(self, adapter: ModelAdapter) -> None:
        self._adapter = adapter

    def extract(self, prompt: str, *, image: bytes | None = None) -> CompletionResult:
        from nectivon_model_adapter import AdapterError

        try:
            result = self._adapter.complete(prompt, image=image, retry=False)
        except AdapterError as exc:
            raise AIExecutionError("识别调用失败，未自动重试；请人工整理或核对模型配置。") from exc
        usage = result.record.usage
        return CompletionResult(
            text=result.text,
            model=result.record.requested_model,
            usage=CompletionUsage(usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
            if usage
            else None,
            finish_reason=result.finish_reason,
            retry_count=max(0, result.record.attempts - 1),
            resolved_model=result.record.reported_model,
        )


def configured_extractor(model: str) -> CandidateExtractionProvider:
    """Read SecretStr only on an authorized worker action, with a one-request budget."""
    from nectivon_model_adapter import (
        CallPolicy,
        HttpsTransport,
        MemoryCredentialStore,
        ModelAdapter,
        ModelSelection,
        RequestBudget,
        SecretCredential,
    )

    from src.ai.provider_scope import require_active_model
    from src.config import Settings

    require_active_model("qwen", model)
    settings = Settings(_env_file=None)
    if settings.ai_mode == "manual" or not settings.ai_api_key.get_secret_value():
        raise AIUnavailableError("当前为手动模式或未配置密钥，请使用人工整理。")
    credentials = MemoryCredentialStore()
    credentials.set("qwen", SecretCredential(settings.ai_api_key.get_secret_value()))
    adapter = ModelAdapter(
        ModelSelection("qwen", model),
        credentials,
        HttpsTransport(),
        policy=CallPolicy(max_extra_attempts=0, max_output_tokens=8192),
        budget=RequestBudget(1),
    )
    return CandidateExtractionProvider(adapter)
