"""Complete dialogue exports, independent of model availability and context windows."""

from __future__ import annotations

import io
import json
import zipfile
from xml.etree import ElementTree as ET

from .chat_citations import checked_messages, visible_answer
from .models import StudyError
from .source_presentation import evidence_name


def markdown(snapshot: dict) -> str:
    """Do not shorten or select rounds; include provenance attached to every request."""
    conversation = snapshot["conversation"]
    messages = checked_messages(snapshot["messages"])
    if not messages:
        raise StudyError("当前没有可导出的对话。")
    if any(m["status"] == "pending" for m in messages):
        raise StudyError("当前回答尚未结束，请等待或明确标记中断后导出。")
    lines = [
        "# 学习对话记录",
        "",
        "类型：" + ("题目追问" if conversation["kind"] == "question" else "课程自由复盘"),
        "",
        "课程：" + (conversation["subject"] or "未指定"),
        "",
        "会话 ID：" + conversation["id"],
        "",
        "说明：原始材料、用户说明与 AI 推导应分别核对；保存对话不代表知识已经验证。",
        "",
    ]
    if conversation["exam_notes"]:
        lines += ["## 用户提供的考试范围", "", conversation["exam_notes"], ""]
    context = conversation["context"]
    if context:
        question = context["item"]["question"]
        lines += [
            "## 关联原题",
            "",
            question["content"],
            "",
            "来源：" + context["source"]["name"],
            "",
            "SHA-256：" + context["source"]["sha256"],
            "",
        ]
        pending = [question]
        while pending:
            node = pending.pop(0)
            if node is not question:
                lines += ["### 小问 " + node["number"], "", node["content"], ""]
            for field, label in (
                ("reference_answer", "参考答案"),
                ("reference_analysis", "参考解析"),
                ("observed_answer", "材料中的答案（待核对）"),
                ("observed_analysis", "材料中的解析（待核对）"),
            ):
                if node[field]:
                    lines += ["**" + label + "**", "", node[field], ""]
            pending.extend(node["children"])
    references = {}
    for index, message in enumerate(messages, 1):
        role = "用户" if message["role"] == "user" else "AI"
        lines += [
            f"## {index}. {role}",
            "",
            f"时间：{message['created_at']} · 状态：{message['status']}",
            "",
            message["content"],
            "",
        ]
        metadata = message["metadata"]
        if metadata.get("model"):
            lines += ["模型：" + metadata["model"], ""]
        if metadata.get("raw_response") and metadata["raw_response"] != message["content"]:
            lines += [
                "### 原始模型正文（含未核验引用或截断内容，保留供追溯）",
                "",
                metadata["raw_response"],
                "",
            ]
        if metadata.get("evidence"):
            lines += [
                "本轮来源：" + "、".join("[" + r["label"] + "]" for r in metadata["evidence"]),
                "",
            ]
            for ref in metadata["evidence"]:
                references[ref["label"]] = ref
    if references:
        lines += ["## 来源快照附录（相同片段仅列一次）", ""]
        for ref in references.values():
            lines += [
                f"### [{ref['label']}] {ref['name']}",
                "",
                f"定位：{ref['location']}；SHA-256：{ref['sha256']}",
                "",
                ref["text"],
                "",
            ]
    return "\n".join(lines)


def docx(snapshot: dict) -> bytes:
    from .word_math import markdown_docx

    return markdown_docx(markdown(snapshot))


def study_markdown(snapshot: dict) -> str:
    """Export every dialogue turn for reading, keeping technical provenance out of the body."""
    conversation = snapshot["conversation"]
    messages = checked_messages(snapshot["messages"])
    if not messages:
        raise StudyError("当前没有可导出的对话。")
    if any(message["status"] == "pending" for message in messages):
        raise StudyError("当前回答尚未结束，请等待或结束等待后导出。")
    evidence: dict[str, dict] = {}
    source_names: list[str] = []
    lines = ["# 学习对话", ""]
    if conversation.get("subject"):
        lines += ["课程：" + conversation["subject"], ""]
    context = conversation.get("context", {})
    if context:
        lines += ["## 原题", ""]
        pending = [context["item"]["question"]]
        while pending:
            question = pending.pop(0)
            lines += [visible_answer(question["content"]), ""]
            pending[0:0] = question.get("children", [])
        source_name = context.get("source", {}).get("name")
        if source_name:
            lines += ["来源：" + source_name, ""]
    if conversation.get("exam_notes"):
        lines += ["## 考试范围", "", conversation["exam_notes"], ""]
    for index, message in enumerate(messages, 1):
        for ref in message["metadata"].get("evidence", []):
            evidence[ref["label"]] = ref
            name = evidence_name(ref)
            if name not in source_names:
                source_names.append(name)
        role = "用户" if message["role"] == "user" else "AI"
        content = visible_answer(message["content"], list(evidence.values()))
        if (message["status"] == "error" and not message["metadata"].get("error_code")
                and content.startswith("模型调用未完成，原消息和来源已保存。")):
            content = "这次未取得回答，请重新发送。"
        lines += [f"## {index}. {role}", "", content, ""]
    if source_names:
        lines += ["## 来源", "", *["- " + name for name in source_names], ""]
    return "\n".join(lines)


def study_docx(snapshot: dict) -> bytes:
    """Render the readable dialogue and embed its unmodified local source snapshot separately."""
    from .word_math import markdown_docx

    data = markdown_docx(study_markdown(snapshot))
    output = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(data)) as original,
        zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as target,
    ):
        for info in original.infolist():
            body = original.read(info.filename)
            if info.filename == "[Content_Types].xml":
                root = ET.fromstring(body)
                namespace = "{http://schemas.openxmlformats.org/package/2006/content-types}"
                if not any(element.get("Extension") == "json" for element in root):
                    ET.SubElement(root, namespace + "Default",
                                  {"Extension": "json", "ContentType": "application/json"})
                body = ET.tostring(root, encoding="utf-8", xml_declaration=True)
            target.writestr(info, body)
        target.writestr("nectivon/source-snapshot.json",
                        json.dumps(snapshot, ensure_ascii=False, indent=2).encode("utf-8"))
    return output.getvalue()
