"""Host binding of the frozen Model Adapter; never opens an application database."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from datetime import UTC, datetime

from src.ai.credential_store import CredentialStore, build_default_credential_store
from src.ai.provider import AIUnavailableError
from src.ai.provider_config import ProviderConfigStore
from src.ai.provider_scope import (
    DOCUMENTATION_CHECKED,
    DOCUMENTED_VERSIONS,
    require_active_model,
)


class ReviewTransport:
    """Capture visible answer metadata, not keys, headers, errors or hidden reasoning."""

    def __init__(
        self, control: Callable[[], None], *, connectivity: bool = False,
        reasoning_effort: str | None = None,
    ) -> None:
        from nectivon_model_adapter import HttpsTransport

        self._transport = HttpsTransport()
        self._control = control
        self._connectivity = connectivity
        self._reasoning_effort = reasoning_effort
        self._chat_messages: ContextVar[tuple[dict[str, str], ...] | None] = ContextVar(
            "study_chat_messages", default=None,
        )
        self.response: dict = {}
        self.request_info: dict = {}
        self.control_state = ""
        self.transport_attempts = 0

    @contextmanager
    def conversation(self, messages: list[dict[str, str]] | None):
        """Bind validated messages to this call/context; restore after success or failure."""
        from src.ai.conversation import decode_conversation, encode_conversation

        copied = None
        if messages is not None:
            validated = decode_conversation(encode_conversation(messages))
            copied = tuple(dict(message) for message in validated)
        token = self._chat_messages.set(copied)
        try:
            yield
        finally:
            self._chat_messages.reset(token)

    def __call__(self, request):
        from nectivon_model_adapter import AdapterError, ErrorCode

        try:
            self._control()
        except Exception as exc:
            self.control_state = (
                "control_interrupted"
                if getattr(exc, "code", None) == "CONTROL"
                else "control_failed"
            )
            # Non-retryable, secret-free transport signal; caller records cancellation.
            raise AdapterError(ErrorCode.RESOURCE_CLEANUP) from None
        body = json.loads(request.body)
        chat_messages = self._chat_messages.get()
        if chat_messages is not None:
            body["messages"] = [dict(message) for message in chat_messages]
        if self._connectivity or self._reasoning_effort is not None:
            if body["model"].startswith("deepseek"):
                if self._connectivity:
                    body["thinking"] = {"type": "disabled"}
                elif self._reasoning_effort is not None:
                    body["thinking"] = {
                        "type": "disabled" if self._reasoning_effort == "none" else "enabled",
                    }
                    body["reasoning_effort"] = self._reasoning_effort
        if chat_messages is not None or self._connectivity or self._reasoning_effort is not None:
            request = replace(request, body=json.dumps(body).encode("utf-8"))
        self.request_info = {
            "wire_payload_sha256": hashlib.sha256(request.body).hexdigest(),
            "request_options": {
                key: body[key]
                for key in (
                    "stream",
                    "max_tokens",
                    "max_completion_tokens",
                    "thinking",
                    "enable_thinking",
                    "reasoning_effort",
                )
                if key in body
            },
        }
        self.transport_attempts += 1
        response = self._transport(request)
        if response.status == 200:
            try:
                value = json.loads(response.body)
                message = value["choices"][0]["message"]
                content = message.get("content")
                model = value.get("model")
                fingerprint = value.get("system_fingerprint")
                finish_reason = value["choices"][0].get("finish_reason")
                reasoning = message.get("reasoning_content")
                usage = value.get("usage")
                details = (
                    usage.get("completion_tokens_details") if isinstance(usage, dict) else None
                )
                reasoning_tokens = (
                    details.get("reasoning_tokens") if isinstance(details, dict) else None
                )
                self.response = {
                    "text": content if isinstance(content, str) else "",
                    "reported_model": model if isinstance(model, str) else None,
                    "system_fingerprint": fingerprint if isinstance(fingerprint, str) else None,
                    "finish_reason": finish_reason if isinstance(finish_reason, str) else None,
                    "visible_output_chars": len(content) if isinstance(content, str) else 0,
                    "reasoning_output_chars": len(reasoning) if isinstance(reasoning, str) else 0,
                    "reasoning_tokens": reasoning_tokens
                    if type(reasoning_tokens) is int and 0 <= reasoning_tokens <= 2**53 else None,
                    "http_status": response.status,
                }
            except (ValueError, TypeError, KeyError, IndexError, AttributeError, RecursionError):
                self.response = {}
        return response


class CandidateReviewProvider:
    """One explicitly selected model, a shared request budget and cooperative cancellation."""

    def __init__(self, adapter, transport: ReviewTransport, provider: str, model: str) -> None:
        self.adapter, self.transport = adapter, transport
        self.provider, self.model = provider, model

    def review(
        self, prompt: str, *, image: bytes | None = None,
        chat_messages: list[dict[str, str]] | None = None,
    ) -> dict:
        from nectivon_model_adapter import AdapterError, __version__

        if chat_messages is not None and image is not None:
            raise AIUnavailableError("文本对话不能覆盖图片输入，未发送。")
        self.transport.response = {}
        self.transport.request_info = {}
        self.transport.control_state = ""
        self.transport.transport_attempts = 0
        started = time.monotonic()
        record = {
            "provider": self.provider,
            "workflow_version": "study-model-review-v1.1",
            "adapter_version": __version__,
            "requested_model": self.model,
            "documented_version": DOCUMENTED_VERSIONS[self.model],
            "documentation_checked": DOCUMENTATION_CHECKED,
            "started_at": datetime.now(UTC).isoformat(),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "status": "failed",
            "text": "",
            "reported_model": None,
            "attempts": 0,
            "usage": None,
            "timeout_seconds": self.adapter.policy.timeout_seconds,
            "max_extra_attempts": self.adapter.policy.max_extra_attempts,
            "error_code": "",
            "error_message": "",
        }
        try:
            with self.transport.conversation(chat_messages):
                result = self.adapter.complete(prompt, image=image, retry=True)
        except AdapterError as exc:
            record.update(self.transport.response)
            record.update(
                status=self.transport.control_state or "failed",
                attempts=exc.attempts,
                http_status=exc.http_status or self.transport.response.get("http_status"),
                error_code=exc.code.value,
                error_message=str(exc),
            )
            if exc.record is not None:
                # A changed server version can fail the frozen codec. Preserve
                # its actual unverified identifier instead of replacing it by None.
                if exc.record.reported_model is not None:
                    record["reported_model"] = exc.record.reported_model
                if exc.record.usage is not None:
                    from dataclasses import asdict

                    record["usage"] = asdict(exc.record.usage)
            if self.transport.control_state:
                record["error_message"] = (
                    "暂停或取消已阻止后续传输；在途结果需人工核对。"
                    if self.transport.control_state == "control_interrupted"
                    else "无法确认任务控制状态，已停止传输；不能认定为用户取消。"
                )
        else:
            from dataclasses import asdict

            record.update(self.transport.response)
            record.update(
                status="ok",
                text=result.text,
                reported_model=result.record.reported_model,
                attempts=result.record.attempts,
                usage=asdict(result.record.usage) if result.record.usage else None,
                finish_reason=result.finish_reason,
            )
        record["duration_ms"] = round((time.monotonic() - started) * 1000)
        record["transport_attempts"] = self.transport.transport_attempts
        record.update(self.transport.request_info)
        record["version_verified"] = False
        record["version_note"] = "响应标识如为可变别名，不能证明精确后端快照；文档版本仅供对照。"
        return record


def configured_review_provider(
    provider: str,
    model: str,
    *,
    control: Callable[[], None],
    budget,
    max_extra_attempts: int = 0,
    timeout_seconds: float = 90,
    max_output_tokens: int = 8192,
    connectivity: bool = False,
    credential_store: CredentialStore | None = None,
    reasoning_effort: str | None = None,
) -> CandidateReviewProvider:
    """Read configured credentials; never silently replace a region or endpoint."""
    from nectivon_model_adapter import (
        CallPolicy,
        MemoryCredentialStore,
        ModelAdapter,
        ModelSelection,
        SecretCredential,
    )

    require_active_model(provider, model)
    if reasoning_effort not in {None, "none", "low", "high", "max"}:
        raise AIUnavailableError("模型推理设置无效。")
    if reasoning_effort is not None and provider != "deepseek":
        raise AIUnavailableError("当前推理设置不适用于所选模型。")
    selection = ModelSelection(provider, model)
    config = ProviderConfigStore().load()
    selected = config.get(provider)
    if selected is not None and selected.base_url.rstrip("/") != selection.endpoint:
        raise AIUnavailableError(
            "本轮封存适配器尚不支持该地域/自定义端点，未替换已配置地址；请保留配置并人工核对。"
        )
    credentials_source = (
        credential_store if credential_store is not None else build_default_credential_store()
    )
    credential = credentials_source.get(provider)
    if credential is None and provider == "qwen":
        from src.config import Settings, settings_env_path

        settings = Settings(_env_file=settings_env_path())
        if settings.ai_mode == "api":
            secret = settings.ai_api_key.get_secret_value()
            if secret:
                from src.ai.credential_store import SecretCredential as HostSecret

                credential = HostSecret(secret)
    if credential is None:
        raise AIUnavailableError("该供应商缺少安全配置的 API Key，请在模型设置中保存。")
    credentials = MemoryCredentialStore()
    credentials.set(provider, SecretCredential(credential.reveal()))
    transport = (
        ReviewTransport(control, connectivity=connectivity, reasoning_effort=reasoning_effort)
        if reasoning_effort is not None else ReviewTransport(control, connectivity=connectivity)
    )
    adapter = ModelAdapter(
        selection,
        credentials,
        transport,
        policy=CallPolicy(timeout_seconds, max_extra_attempts, max_output_tokens),
        budget=budget,
    )
    return CandidateReviewProvider(adapter, transport, provider, model)
