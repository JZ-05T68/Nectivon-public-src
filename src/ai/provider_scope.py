"""Current product allowlist, separate from historical provider/model decoding."""

import re
from urllib.parse import urlsplit

from src.ai.model_registry import ProviderId
from src.ai.provider import AIUnavailableError

ACTIVE_MODELS: dict[ProviderId, tuple[str, ...]] = {
    ProviderId.DEEPSEEK: ("deepseek-flash", "deepseek-v4-pro"),
    ProviderId.QWEN: ("qwen3.8-flash", "qwen3.8-max"),
}
DOCUMENTATION_CHECKED = "2026-11-15"
DOCUMENTED_VERSIONS = {
    "deepseek-flash": "DeepSeek-V4.1-Flash-1017",
    "deepseek-v4-pro": "DeepSeek-V4-Pro-0813",
    "qwen3.8-flash": "官方可变别名；精确后端快照未确认",
    "qwen3.8-max": "官方可变别名；精确后端快照未确认",
}


def require_active_provider(provider_id: ProviderId | str) -> ProviderId:
    """Disable calls without deleting stored credentials, config or old review records."""
    try:
        normalized = ProviderId(provider_id)
    except ValueError:
        raise AIUnavailableError("本轮仅接入 DeepSeek 和通义千问。") from None
    if normalized not in ACTIVE_MODELS:
        raise AIUnavailableError("该供应商已停用；历史记录仍可读取，请选择 DeepSeek 或通义千问。")
    return normalized


def require_active_model(provider_id: ProviderId | str, model_id: str) -> None:
    provider = require_active_provider(provider_id)
    if model_id not in ACTIVE_MODELS[provider]:
        raise AIUnavailableError("该对话模型已停用，请从本轮四个目标型号中重新选择。")


def require_official_endpoint(provider_id: ProviderId | str, base_url: str) -> None:
    """Keep retired/general-purpose endpoints from being reintroduced through a custom URL."""
    provider = require_active_provider(provider_id)
    value = urlsplit(base_url)
    host = value.hostname or ""
    valid = (
        host == "api.deepseek.com" and value.path.rstrip("/") in ("", "/v1")
        if provider is ProviderId.DEEPSEEK
        else (
            host
            in {
                "dashscope.aliyuncs.com",
                "dashscope-intl.aliyuncs.com",
                "dashscope-us.aliyuncs.com",
            }
            or re.fullmatch(
                r"[a-zA-Z0-9-]+\.(?:cn-beijing|ap-southeast-1|us-east-1|ap-northeast-1)"
                r"\.maas\.aliyuncs\.com",
                host,
            )
            is not None
        )
        and value.path.rstrip("/") == "/compatible-mode/v1"
    )
    if (
        not valid
        or value.scheme != "https"
        or value.username
        or value.password
        or value.query
        or value.fragment
        or value.port not in (None, 443)
    ):
        raise AIUnavailableError("本轮模型调用仅允许对应供应商的官方 HTTPS 接口。")
