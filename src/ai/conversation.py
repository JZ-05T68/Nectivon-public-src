"""Text-only conversation envelope for the neutral completion boundary.

The host adapter may translate this envelope into its vendor's message format.
Other completion clients still receive a self-contained, auditable text prompt.
Only application code assigns roles; user/source text remains string content.
"""

from __future__ import annotations

import json
import re

SCHEMA = "nectivon.study-chat.v2"
MAX_CONVERSATION_CHARS = 160_000


def requests_formula_verification(text: str) -> bool:
    """Recognize an explicit verification task, never instructions inside evidence.

    Callers supply only the current user's question. This changes preparation
    effort, not the selected model, paid request count, or stored question.
    """
    formula = re.search(r"公式|关系式|\b(?:formula(?:s)?|equation(?:s)?)\b", text, re.I)
    verify = re.search(
        r"订正|纠错|核验|校验|校核|核对|检查|审查|审核|审校|修正|"
        r"\b(?:audit|verify|verification|correct|corrected|correction|check)\b", text, re.I,
    )
    return bool(formula and verify)


def _validate(messages: object) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("对话消息格式无效。")
    result = []
    for index, message in enumerate(messages):
        if (
            not isinstance(message, dict) or set(message) != {"role", "content"}
            or not isinstance(message["role"], str)
            or message["role"] not in {"system", "user", "assistant"}
            or not isinstance(message["content"], str) or not message["content"].strip()
            or (message["role"] == "system" and index != 0)
        ):
            raise ValueError("对话消息格式无效。")
        result.append(dict(message))
    if result[0]["role"] != "system" or result[-1]["role"] != "user":
        raise ValueError("对话消息角色顺序无效。")
    return result


def encode_conversation(messages: list[dict[str, str]]) -> str:
    """Serialize explicit roles without interpreting any content as instructions."""
    return json.dumps(
        {"schema": SCHEMA, "messages": _validate(messages)}, ensure_ascii=False,
        separators=(",", ":"),
    )


def decode_conversation(prompt: str) -> list[dict[str, str]] | None:
    """Decode only this host schema; leave ordinary completion prompts alone."""
    duplicates = False

    def object_pairs(pairs: list[tuple[str, object]]) -> dict:
        nonlocal duplicates
        value = dict(pairs)
        duplicates |= len(value) != len(pairs)
        return value

    try:
        value = json.loads(prompt, object_pairs_hook=object_pairs)
    except (ValueError, RecursionError):
        return None
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        return None
    if duplicates or len(prompt) > MAX_CONVERSATION_CHARS or set(value) != {"schema", "messages"}:
        raise ValueError("对话请求超出范围或格式无效。")
    return _validate(value["messages"])
