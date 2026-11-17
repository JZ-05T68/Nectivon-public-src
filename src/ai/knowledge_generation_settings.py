"""Local, non-secret knowledge-task configuration and cumulative credit reservations.

Manual evidence is explicitly labeled as user-verified, never an API balance
query. OpenAI keys stay in process memory (or the user's environment), outside
the configuration, audit, backups and Git.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from src.ai.credential_store import SecretCredential
from src.ai.knowledge_generation import (
    DEFAULT_KNOWLEDGE_TASK_TOKENS,
    KNOWLEDGE_MODEL_ROLES,
    VerifiedFreeCreditGrant,
)
from src.ai.provider import AIBudgetExceededError, AIUnavailableError
from src.ai.provider_scope import require_active_model
from src.config import runtime_private_root

_LOCK = threading.RLock()
_OPENAI_CREDENTIAL: SecretCredential | None = None


def knowledge_openai_credential() -> SecretCredential | None:
    """Return the redacted process-session key; callers unwrap only at transport."""
    return _OPENAI_CREDENTIAL


def set_knowledge_openai_credential(value: str) -> None:
    """Keep a manually supplied OpenAI key in memory for this app process only."""
    global _OPENAI_CREDENTIAL
    _OPENAI_CREDENTIAL = SecretCredential(value) if value.strip() else None


@dataclass(frozen=True, slots=True)
class KnowledgeGenerationSettings:
    """A separate knowledge-only route, independent of ordinary AI selection."""

    provider: str
    model: str
    grant: VerifiedFreeCreditGrant | None = None
    max_calls: int = 1
    task_tokens: int = DEFAULT_KNOWLEDGE_TASK_TOKENS
    configured_api: bool = False

    def __post_init__(self) -> None:
        if self.provider not in KNOWLEDGE_MODEL_ROLES:
            raise ValueError("知识工作流仅支持 OpenAI、DeepSeek 和 Qwen。")
        if type(self.configured_api) is not bool or (
            self.configured_api and self.provider not in {"deepseek", "qwen"}
        ):
            raise ValueError("已配置 API 授权仅适用于 DeepSeek / Qwen；GPT 仍须核验免费额度。")
        if self.provider == "openai":
            if self.model != KNOWLEDGE_MODEL_ROLES["openai"][0]:
                raise ValueError("OpenAI 知识模型必须为已核验的 GPT-6.1 Sol。")
        else:
            require_active_model(self.provider, self.model)
        if type(self.max_calls) is not int or not 1 <= self.max_calls <= 3:
            raise ValueError("每任务调用上限须为 1～3。")
        if type(self.task_tokens) is not int or not 1024 <= self.task_tokens <= 1000000:
            raise ValueError("任务 Token 上限须为 1024～1000000。")


def _config_path() -> Path:
    return runtime_private_root() / "knowledge_generation_settings.json"


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def save_knowledge_generation_settings(settings: KnowledgeGenerationSettings) -> None:
    """Persist only explicit non-secret fields; no key can enter this schema."""
    value = asdict(settings)
    if settings.grant is not None:
        value["grant"]["period_start"] = settings.grant.period_start.isoformat()
        value["grant"]["expires_at"] = settings.grant.expires_at.isoformat()
    _atomic_json(_config_path(), {"version": 1, **value})


def load_knowledge_generation_settings() -> KnowledgeGenerationSettings | None:
    """Bad/unknown config blocks generation without blocking application startup."""
    path = _config_path()
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.pop("version") != 1:
            raise ValueError
        if value.get("grant") is not None:
            grant = value["grant"]
            grant["period_start"] = datetime.fromisoformat(grant["period_start"])
            grant["expires_at"] = datetime.fromisoformat(grant["expires_at"])
            value["grant"] = VerifiedFreeCreditGrant(**grant)
        return KnowledgeGenerationSettings(**value)
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
        raise AIUnavailableError("知识生成额度配置无效（BLOCKED），请在高级设置重新保存。") from exc


def reserve_credit_tokens(grant: VerifiedFreeCreditGrant, amount: int) -> None:
    """Reserve cumulatively across tasks/reruns; a lock collision fails closed.

    Failed requests keep their reservations, so uncertain billing never returns
    spending power. A provider-enforced free-only limit is still mandatory.
    """
    root = runtime_private_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "knowledge_credit_reservations.json"
    lock_path = root / "knowledge_credit_reservations.lock"
    # Same organization/provider/period shares a pot across model/evidence edits.
    identity = hashlib.sha256(
        json.dumps(
            [
                grant.provider,
                grant.organization,
                grant.period_start.isoformat(),
                grant.expires_at.isoformat(),
            ]
        ).encode("utf-8")
    ).hexdigest()
    with _LOCK:
        try:
            handle = lock_path.open("x", encoding="utf-8")
        except FileExistsError as exc:
            raise AIBudgetExceededError(
                "额度预留正在处理或上次未完成，请核对本地锁后重试；未发送请求。"
            ) from exc
        try:
            handle.close()
            value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            used = value.get(identity, 0)
            if type(used) is not int or used < 0 or used + amount > grant.remaining_tokens:
                raise AIBudgetExceededError("该账期免费额度的本地累计预留已达上限；未发送请求。")
            value[identity] = used + amount
            _atomic_json(path, value)
        except (ValueError, TypeError, OSError) as exc:
            raise AIBudgetExceededError("无法安全预留已授权免费额度；未发送请求。") from exc
        finally:
            lock_path.unlink()
