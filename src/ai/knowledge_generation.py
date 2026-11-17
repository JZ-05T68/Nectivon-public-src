"""Optional high-reasoning knowledge composition with explicit, bounded authorization.

No provider is contacted while constructing this module or a provider. Account
entitlements cannot be inferred from a ChatGPT subscription or from an API key.
Free-credit routes require a verified, provider-enforced free-only grant.
Domestic providers may use the explicitly authorized, configured API billing route.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.ai.completion_stage import CompletionStage, completion_stage_scope
from src.ai.credential_store import SecretCredential
from src.ai.openai_compatible import (
    OpenAICompatibleClient,
    OpenAICompatibleTransportError,
    Transport,
    TransportFailureKind,
    urllib_transport,
)
from src.ai.provider import (
    AIBudgetExceededError,
    AIUnavailableError,
    AuditedAIProvider,
    CompletionDiagnostics,
    CompletionResult,
    CompletionUsage,
    ProviderCallError,
    ProviderErrorCode,
    SafeProviderError,
    build_production_audited_provider,
)
from src.ai.provider_scope import require_active_model

OPENAI_KNOWLEDGE_MODEL = "gpt-6.1-sol"
KNOWLEDGE_MODEL_ROLES = {
    "openai": (OPENAI_KNOWLEDGE_MODEL, "主要提炼与复杂关系候选", "xhigh"),
    "deepseek": ("deepseek-v4-pro", "独立审查遗漏、矛盾与关系", "max"),
    "qwen": ("qwen3.8-max", "来源覆盖、分类与结构审查", "xhigh"),
}
CREDIT_STATUS = "UNKNOWN"
DEFAULT_KNOWLEDGE_OUTPUT_TOKENS = 65536
DEFAULT_KNOWLEDGE_TASK_TOKENS = 150000
KNOWLEDGE_TIMEOUT_SECONDS = 300


def require_openai_knowledge_request(
    base_url: str,
    path: str,
    payload: Mapping[str, Any],
) -> None:
    """Permit the bounded, non-streaming JSON contract before credentials are revealed."""
    output_limit = payload.get("max_output_tokens")
    if (
        base_url != "https://api.openai.com/v1"
        or path != "/responses"
        or payload.get("model") != OPENAI_KNOWLEDGE_MODEL
        or payload.get("reasoning") != {"effort": "xhigh"}
        or payload.get("store") is not False
        or payload.get("stream") is not False
        or payload.get("text") != {"format": {"type": "json_object"}}
        or type(output_limit) is not int
        or not 1 <= output_limit <= 128000
        or not isinstance(payload.get("input"), str)
        or set(payload) - {
            "model", "input", "reasoning", "max_output_tokens", "store", "stream", "text",
        }
    ):
        raise AIUnavailableError("OpenAI 知识生成只允许官方 GPT-6.1 Sol Responses/xhigh 接口。")


def _openai_knowledge_error(
    error: OpenAICompatibleTransportError, provider_id: str,
) -> SafeProviderError:
    """Classify access, request and transport failures without vendor messages."""
    status = error.status_code
    if status is None:
        code = (ProviderErrorCode.TIMEOUT if error.kind is TransportFailureKind.TIMEOUT
                else ProviderErrorCode.NETWORK_ERROR)
    else:
        code = {
            400: ProviderErrorCode.INVALID_REQUEST,
            401: ProviderErrorCode.AUTHENTICATION_FAILED,
            403: ProviderErrorCode.MODEL_ACCESS_DENIED,
            404: ProviderErrorCode.MODEL_NOT_FOUND,
            429: ProviderErrorCode.RATE_LIMITED,
        }.get(status, ProviderErrorCode.PROVIDER_UNAVAILABLE
              if status >= 500 else ProviderErrorCode.PROVIDER_RESPONSE)
    return SafeProviderError(code=code, provider_id=provider_id, http_status=status)


def _counter(value: object) -> int | None:
    return value if type(value) is int and 0 <= value <= 2**53 else None


def chat_diagnostics(response: Mapping[str, Any], result: CompletionResult) -> dict[str, Any]:
    """Retain only safe counters; hidden reasoning and final content never enter audit."""
    usage = response.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    details = usage.get("completion_tokens_details")
    details = details if isinstance(details, Mapping) else {}
    choices = response.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else {}
    message = first.get("message", {}) if isinstance(first, Mapping) else {}
    reasoning = message.get("reasoning_content") if isinstance(message, Mapping) else None
    return {
        "prompt_tokens": _counter(usage.get("prompt_tokens")),
        "completion_tokens": _counter(usage.get("completion_tokens")),
        "total_tokens": _counter(usage.get("total_tokens")),
        "reasoning_tokens": _counter(details.get("reasoning_tokens")),
        "reasoning_output_chars": len(reasoning) if isinstance(reasoning, str) else None,
        "visible_output_chars": len(result.text),
        "finish_reason": result.finish_reason
        if result.finish_reason in {None, "stop", "length", "content_filter", "sensitive"}
        else None,
        "http_status": 200,
    }


class OpenAIKnowledgeProvider:
    """Responses API adapter for the verified exact model; contains no account guesses."""

    provider_id = "openai"
    supports_embedding = False

    def __init__(
        self,
        credential: SecretCredential | None,
        *,
        transport: Transport = urllib_transport,
        timeout_seconds: float = KNOWLEDGE_TIMEOUT_SECONDS,
    ) -> None:
        self._credential = credential
        self._client = OpenAICompatibleClient(
            base_url="https://api.openai.com/v1",
            timeout_seconds=timeout_seconds,
            max_extra_attempts=0,
            transport=transport,
        )

    @property
    def is_configured(self) -> bool:
        return self._credential is not None

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_completion_tokens: int | None = None,
    ) -> CompletionResult:
        """Return final text only; incomplete Responses are explicitly length-marked."""
        if self._credential is None:
            raise AIUnavailableError(
                "GPT-6.1 Sol 尚未配置 API 凭据，组织访问权限未核验（BLOCKED）。"
            )
        chosen = model or OPENAI_KNOWLEDGE_MODEL
        limit = (
            max_completion_tokens if max_completion_tokens is not None
            else DEFAULT_KNOWLEDGE_OUTPUT_TOKENS
        )
        if type(limit) is not int or not 1 <= limit <= 128000:
            raise ValueError("GPT-6.1 Sol 输出 Token 上限须为 1～128000。")
        payload = {
            "model": chosen,
            "input": prompt,
            "reasoning": {"effort": "xhigh"},
            "max_output_tokens": limit,
            "store": False,
            "stream": False,
            "text": {"format": {"type": "json_object"}},
        }
        response, retries = self._client.post(
            "/responses",
            payload,
            credential=self._credential,
            provider_id=self.provider_id,
            error_mapper=_openai_knowledge_error,
        )
        try:
            if not isinstance(response, Mapping):
                raise TypeError
            status = response["status"]
            if status in {"failed", "cancelled", "queued", "in_progress"}:
                raise ProviderCallError(
                    SafeProviderError(code=ProviderErrorCode.PROVIDER_RESPONSE,
                                      provider_id="openai", http_status=200)
                )
            if status not in {"completed", "incomplete"}:
                raise ValueError
            output = response["output"]
            if not isinstance(output, list):
                raise TypeError
            parts: list[str] = []
            for message in output:
                if not isinstance(message, Mapping):
                    raise TypeError
                if message.get("type") != "message":
                    continue  # Reasoning/tool items never become candidate JSON or audit text.
                if message.get("role") not in {None, "assistant"}:
                    raise ValueError
                message_status = message.get("status")
                if message_status not in {None, "completed", "incomplete"} or (
                    status == "completed" and message_status == "incomplete"
                ):
                    raise ValueError
                phase = message.get("phase")
                if phase == "commentary":
                    continue
                if phase not in {None, "final_answer"}:
                    raise ValueError
                content = message["content"]
                if not isinstance(content, list):
                    raise TypeError
                for item in content:
                    if not isinstance(item, Mapping):
                        raise TypeError
                    if item.get("type") == "refusal":
                        raise ProviderCallError(
                            SafeProviderError(code=ProviderErrorCode.PROVIDER_RESPONSE,
                                              provider_id="openai", http_status=200)
                        )
                    if item.get("type") != "output_text" or not isinstance(item.get("text"), str):
                        raise TypeError
                    parts.append(item["text"])
            text = "".join(parts)
            if status == "completed" and not text.strip():
                raise ValueError
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderCallError(
                SafeProviderError(
                    code=ProviderErrorCode.MALFORMED_RESPONSE,
                    provider_id="openai",
                )
            ) from exc
        raw = response.get("usage")
        raw = raw if isinstance(raw, Mapping) else {}
        counts = [
            _counter(raw.get(key)) for key in ("input_tokens", "output_tokens", "total_tokens")
        ]
        usage = CompletionUsage(*counts) if all(value is not None for value in counts) else None
        details = raw.get("output_tokens_details")
        reasoning = (
            _counter(details.get("reasoning_tokens")) if isinstance(details, Mapping) else None
        )
        reported = response.get("model")
        reported = reported if isinstance(reported, str) and reported.strip() else None
        incomplete = response.get("incomplete_details")
        incomplete_reason = incomplete.get("reason") if isinstance(incomplete, Mapping) else None
        if status == "completed":
            finish = "stop"
        elif incomplete_reason == "max_output_tokens":
            finish = "length"
        elif incomplete_reason == "content_filter":
            finish = "content_filter"
        else:
            # An unknown/unsupported interruption is not evidence of a token cap.
            raise ProviderCallError(
                SafeProviderError(code=ProviderErrorCode.PROVIDER_RESPONSE,
                                  provider_id="openai", http_status=200)
            )
        return CompletionResult(
            text=text,
            model=reported or chosen,
            usage=usage,
            finish_reason=finish,
            retry_count=retries,
            resolved_model=reported,
            diagnostics=CompletionDiagnostics(
                output_limit=limit,
                visible_output_chars=len(text),
                reasoning_tokens=reasoning,
                prompt_tokens=counts[0],
                completion_tokens=counts[1],
                total_tokens=counts[2],
                http_status=200,
                finish_reason=finish,
                thinking_mode="enabled",
                reasoning_effort="xhigh",
            ),
        )


@dataclass(frozen=True, slots=True)
class VerifiedFreeCreditGrant:
    """Evidence supplied by a verified account check, never inferred from key presence.

    `free_only_enforced` means the supplier/project rejects calls before any
    paid fallback. Token counts cover input, hidden reasoning and visible output.
    A local balance estimate alone does not satisfy this requirement.
    """

    provider: str
    organization: str
    model: str
    period_start: datetime
    expires_at: datetime
    remaining_tokens: int
    evidence_ref: str
    free_only_enforced: bool = False
    model_access_verified: bool = False


class KnowledgeTaskBudget:
    """Share one finite allowance; never silently fall back from free to paid API."""

    def __init__(
        self,
        grant: VerifiedFreeCreditGrant | None,
        *,
        provider: str,
        model: str,
        max_calls: int = 1,
        max_total_tokens: int = DEFAULT_KNOWLEDGE_TASK_TOKENS,
        reserve_durable: Callable[[VerifiedFreeCreditGrant, int], None] | None = None,
        configured_api: bool = False,
    ) -> None:
        if type(max_calls) is not int or not 1 <= max_calls <= 3:
            raise ValueError("每个知识任务只允许 1～3 次调用。")
        if type(max_total_tokens) is not int or max_total_tokens <= 0:
            raise ValueError("知识任务 Token 预算必须为正整数。")
        if type(configured_api) is not bool or (
            configured_api and provider not in {"deepseek", "qwen"}
        ):
            raise ValueError("已配置 API 授权仅适用于 DeepSeek / Qwen；GPT 仍须核验免费额度。")
        self.grant, self.provider, self.model = grant, provider, model
        self.configured_api = configured_api
        self.max_calls, self.max_total_tokens = max_calls, max_total_tokens
        self._calls, self._reserved = 0, 0
        self._reserve_durable = reserve_durable
        self._lock = threading.Lock()

    def reserve(self, prompt: str, output_limit: int) -> None:
        """Reserve a conservative token bound before transport, including failed attempts."""
        # Byte fallback tokenizers use no more than this bound; overhead is reserved too.
        required = len(prompt.encode("utf-8")) * 4 + output_limit + 1024
        with self._lock:
            if self.configured_api:
                if (
                    self._calls >= self.max_calls
                    or self._reserved + required > self.max_total_tokens
                ):
                    raise AIBudgetExceededError("已达到本任务 Token 或调用上限，未发送请求。")
                self._calls += 1
                self._reserved += required
                return
            grant = self.grant
            now = datetime.now(UTC)
            if (
                grant is None
                or grant.provider != self.provider
                or grant.model != self.model
                or not grant.organization
                or not grant.evidence_ref
                or not grant.free_only_enforced
                or not grant.model_access_verified
                or grant.period_start.tzinfo is None
                or grant.expires_at.tzinfo is None
                or not grant.period_start <= now < grant.expires_at
                or type(grant.remaining_tokens) is not int
                or grant.remaining_tokens <= 0
            ):
                raise AIBudgetExceededError(
                    "免费额度、账期、组织访问权限或禁止额外扣费状态尚未核验"
                    "（UNKNOWN / BLOCKED）；未发送请求。"
                )
            if self._calls >= self.max_calls or self._reserved + required > min(
                self.max_total_tokens,
                grant.remaining_tokens,
            ):
                raise AIBudgetExceededError("已达到本任务免费额度或调用上限，未发送请求。")
            if self._reserve_durable is not None:
                self._reserve_durable(grant, required)
            self._calls += 1
            self._reserved += required


class KnowledgeWorkflowProvider:
    """Opt into high reasoning for one task while preserving the existing audit wrapper."""

    def __init__(self, provider: object, budget: KnowledgeTaskBudget) -> None:
        self._provider, self._budget = provider, budget
        self.provider_id = budget.provider

    @property
    def is_configured(self) -> bool:
        return bool(getattr(self._provider, "is_configured", False))

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_completion_tokens: int | None = None,
    ) -> CompletionResult:
        chosen = model or self._budget.model
        if chosen != self._budget.model:
            raise AIUnavailableError("知识任务不能切换到预算授权以外的模型。")
        # Reasoning and visible JSON share this cap; 8K may leave no visible output.
        limit = (
            max_completion_tokens if max_completion_tokens is not None
            else DEFAULT_KNOWLEDGE_OUTPUT_TOKENS
        )
        if type(limit) is not int or not 1 <= limit <= 128000:
            raise ValueError("知识生成输出预算须为 1～128000 Token。")
        self._budget.reserve(prompt, limit)
        with completion_stage_scope(CompletionStage.KNOWLEDGE_GENERATION):
            return self._provider.complete(
                prompt,
                model=chosen,
                max_completion_tokens=limit,
            )


def audited_knowledge_transport(
    transport: Transport,
    record: Callable[[dict[str, Any]], None],
) -> Transport:
    """Record the actual wire options and token metadata without headers/content."""

    def send(
        url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout_seconds: float,
    ) -> Mapping[str, Any]:
        entry = {
            "created_at": datetime.now(UTC).isoformat(),
            "requested_model": payload.get("model"),
            "wire_payload_sha256": hashlib.sha256(
                json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "request_options": {
                key: payload[key]
                for key in (
                    "stream",
                    "max_tokens",
                    "max_completion_tokens",
                    "max_output_tokens",
                    "reasoning",
                    "thinking",
                    "enable_thinking",
                    "reasoning_effort",
                    "response_format",
                    "text",
                    "store",
                )
                if key in payload
            },
            "cost": "UNKNOWN",
            "status": "error",
        }
        try:
            response = transport(url, headers, payload, timeout_seconds)
            if not isinstance(response, Mapping):
                raise ProviderCallError(
                    SafeProviderError(code=ProviderErrorCode.MALFORMED_RESPONSE)
                )
        except Exception as exc:
            entry["error_type"] = type(exc).__name__
            if isinstance(exc, OpenAICompatibleTransportError):
                entry["http_status"] = exc.status_code
            record(entry)
            raise
        entry.update(status="response_received", http_status=200)
        reported = response.get("model")
        entry["reported_model"] = reported if isinstance(reported, str) else None
        raw_usage = response.get("usage")
        if isinstance(raw_usage, Mapping):
            entry["usage"] = {
                key: _counter(raw_usage.get(key))
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                )
            }
            for key in ("output_tokens_details", "completion_tokens_details"):
                details = raw_usage.get(key)
                if isinstance(details, Mapping):
                    entry["reasoning_tokens"] = _counter(details.get("reasoning_tokens"))
        choices = response.get("choices")
        first = choices[0] if isinstance(choices, list) and choices else {}
        reason = first.get("finish_reason") if isinstance(first, Mapping) else None
        if response.get("status") == "completed":
            reason = "stop"
        elif response.get("status") == "incomplete":
            incomplete = response.get("incomplete_details")
            why = incomplete.get("reason") if isinstance(incomplete, Mapping) else None
            reason = ("length" if why == "max_output_tokens"
                      else "content_filter" if why == "content_filter" else None)
        entry["finish_reason"] = (
            reason if isinstance(reason, str) and reason in {"stop", "length", "content_filter"}
            else None
        )
        entry["output_truncated"] = reason == "length" or response.get("status") == "incomplete"
        record(entry)
        return response

    return send


_AUDIT_LOCK = threading.Lock()


def _record_local_options(entry: dict[str, Any]) -> None:
    from src.config import runtime_private_root

    path: Path = runtime_private_root() / "knowledge_generation_calls.jsonl"
    with _AUDIT_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")


def configured_knowledge_generation_provider(
    *,
    provider: str | None = None,
    model: str | None = None,
    grant: VerifiedFreeCreditGrant | None = None,
    max_calls: int = 1,
    max_total_tokens: int = DEFAULT_KNOWLEDGE_TASK_TOKENS,
    configured_api: bool = False,
) -> AuditedAIProvider | None:
    """Build the existing audited composition; default account evidence is UNKNOWN.

    OpenAI is knowledge-only and never appears in domestic/default selectors.
    Its optional key is read from EKB_OPENAI_API_KEY; it is never copied into
    configuration, diagnostics or backups by this module.
    """
    import os

    from dotenv import dotenv_values

    from src import runtime
    from src.ai.knowledge_generation_settings import (
        knowledge_openai_credential,
        load_knowledge_generation_settings,
        reserve_credit_tokens,
    )
    from src.ai.provider_config import ProviderConfigStore
    from src.ai.provider_factory import (
        build_provider_adapter,
        resolve_active_provider_runtime,
        resolve_qwen_runtime,
        resolved_runtime_from_settings,
    )
    from src.config import settings_env_path

    settings = runtime.application_settings()
    saved = load_knowledge_generation_settings()
    if provider is None and saved is not None:
        provider, model = saved.provider, saved.model
        grant = grant or saved.grant
        max_calls, max_total_tokens = saved.max_calls, saved.task_tokens
        configured_api = saved.configured_api
    wire = audited_knowledge_transport(urllib_transport, _record_local_options)
    if provider == "openai":
        selected_model = model or OPENAI_KNOWLEDGE_MODEL
        if selected_model != OPENAI_KNOWLEDGE_MODEL:
            raise AIUnavailableError("未核验的 OpenAI Model ID，未发送请求。")
        key = os.environ.get("EKB_OPENAI_API_KEY") or dotenv_values(settings_env_path()).get(
            "EKB_OPENAI_API_KEY",
            "",
        )
        credential = knowledge_openai_credential() or (SecretCredential(key) if key else None)
        adapter = OpenAIKnowledgeProvider(credential, transport=wire)
    else:
        credentials = runtime.application_credential_store()
        if provider is None:
            resolved = resolve_active_provider_runtime(settings, credential_store=credentials)
        elif provider in {"deepseek", "qwen"}:
            selected = ProviderConfigStore().load().get(provider)
            credential = credentials.get(provider)
            resolved = (
                resolved_runtime_from_settings(selected, credential, settings)
                if (selected is not None and credential is not None)
                else (
                    resolve_qwen_runtime(
                        settings, credential_store=credentials, allow_inactive=True
                    )
                    if provider == "qwen"
                    else None
                )
            )
        else:
            raise AIUnavailableError("该供应商已退出知识生成工作流。")
        if resolved is None:
            return None
        selected_provider = resolved.provider_id.value
        provider = selected_provider
        selected_model = model or resolved.default_model
        require_active_model(provider, selected_model)
        # One task slot is one transmission; no hidden retry spends a second slot.
        resolved = replace(
            resolved, max_extra_attempts=0,
            timeout_seconds=max(resolved.timeout_seconds, KNOWLEDGE_TIMEOUT_SECONDS),
        )
        adapter = build_provider_adapter(resolved, transport=wire)
    budget = KnowledgeTaskBudget(
        grant,
        provider=provider,
        model=selected_model,
        max_calls=max_calls,
        max_total_tokens=max_total_tokens,
        reserve_durable=reserve_credit_tokens,
        configured_api=configured_api,
    )
    return build_production_audited_provider(
        KnowledgeWorkflowProvider(adapter, budget),
        default_model=selected_model,
        default_embedding_model=selected_model,
        source_feature="knowledge_generation",
        ledger=runtime._LazyDatabaseAiCallLedger(),  # noqa: SLF001 - composition root
        budget_guard=runtime._LazyTokenBudgetGuard(settings),  # noqa: SLF001 - composition root
    )


def knowledge_model_status() -> tuple[dict[str, str], ...]:
    """Read-only display information; documentation support is not organization access."""
    return tuple(
        {
            "provider": provider,
            "model": model,
            "role": role,
            "effort": effort,
            "documentation": "VERIFIED 2026-11-15",
            "organization_access": "UNKNOWN",
            "credits": CREDIT_STATUS,
            "live_status": "BLOCKED",
        }
        for provider, (model, role, effort) in KNOWLEDGE_MODEL_ROLES.items()
    )
