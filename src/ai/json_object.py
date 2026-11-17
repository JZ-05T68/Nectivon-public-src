"""Small JSON response parser shared by the independent knowledge tools."""

import json


class JSONResponseError(ValueError):
    """The optional provider did not return a usable object."""


def parse_json_object(text: str) -> dict:
    """Unwrap a fenced object without evaluating or interpreting its contents."""
    cleaned = text.strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise JSONResponseError("AI 未返回有效的 JSON 对象。")
    try:
        value = json.loads(cleaned[start : end + 1])
    except (ValueError, RecursionError) as exc:
        raise JSONResponseError("AI 返回的 JSON 无法解析，请人工核对。") from exc
    if not isinstance(value, dict):
        raise JSONResponseError("AI 返回内容不是 JSON 对象。")
    return value
