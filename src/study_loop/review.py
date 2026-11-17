"""Evidence-only mastery and summaries; there is no review scheduler."""

from __future__ import annotations

from collections import Counter, defaultdict

from src.time_display import format_beijing_time

from .models import OUTCOME_LABELS, VERIFICATION_LABELS, Draft
from .storage import StudyStore


def unit_states(items: list[Draft], attempts: list[dict]) -> dict[tuple[str, str], str]:
    """Latest valid user attempt wins, independently for each subquestion."""
    states = {(item.id, unit.id): "unknown" for item in items for unit in item.question.units()}
    for attempt in attempts:
        key = (attempt.get("item_id"), attempt.get("unit_id"))
        if key not in states or attempt.get("source") != "user_attempt":
            continue
        result = attempt.get("outcome")
        if result not in {"not_yet", "partial", "correct"}:
            continue
        if not (attempt.get("answer", "").strip() or attempt.get("note", "").strip()):
            continue
        if result == "correct" and (
            attempt.get("independent") is not True or not attempt.get("answer", "").strip()
        ):
            continue
        states[key] = result
    return states


def aggregate(values: list[str]) -> str:
    """Do not generalize a correct subquestion to unattempted siblings."""
    if not values or all(value == "unknown" for value in values):
        return "unknown"
    if "not_yet" in values:
        return "not_yet"
    if all(value == "correct" for value in values):
        return "correct"
    return "partial"


def item_state(item: Draft, states: dict[tuple[str, str], str]) -> str:
    return aggregate([states.get((item.id, unit.id), "unknown") for unit in item.question.units()])


def knowledge_states(items: list[Draft], attempts: list[dict]) -> dict[tuple[str, str, str], str]:
    """Keep same-name knowledge in different courses/chapters distinct."""
    states = unit_states(items, attempts)
    grouped: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for item in items:
        for unit in item.question.units():
            for name in unit.knowledge or item.question.knowledge:
                grouped[(item.subject, item.chapter, name)].append(states[(item.id, unit.id)])
    return {key: aggregate(values) for key, values in grouped.items()}


def select_items(
    items: list[Draft],
    attempts: list[dict],
    *,
    subject: str = "",
    chapter: str = "",
    knowledge: str = "",
    attribute: str = "",
    mastery: str = "",
) -> list[Draft]:
    """Pure filtering does not create sessions, records, reminders or model calls."""
    states = unit_states(items, attempts)
    return [
        item
        for item in items
        if (
            (not subject or item.subject == subject)
            and (not chapter or item.chapter == chapter)
            and (
                not knowledge
                or any(
                    knowledge in (unit.knowledge or item.question.knowledge)
                    for unit in item.question.units()
                )
            )
            and (
                not attribute
                or item.question.attribute == attribute
                or any(unit.attribute == attribute for unit in item.question.units())
            )
            and (not mastery or item_state(item, states) == mastery)
        )
    ]


def summary(store: StudyStore, item_ids: list[str]) -> str:
    """Generate only from stored user work and immutable source references, on request."""
    items = [item for item in store.items() if item.id in item_ids]
    attempts = [attempt for attempt in store.attempts() if attempt["item_id"] in item_ids]
    states = unit_states(items, attempts)
    knowledge = knowledge_states(items, attempts)
    lines = [
        "# 学习复盘总结",
        "",
        "本总结来自主动提交的作答与自评，不代表系统自动判卷。",
        "仅反映所选材料，不推断整门课程已经掌握。",
        "",
        "## 复盘题目概况",
        "",
        f"- 所选材料：{len(items)} 条；可作答小问：{len(states)} 个。",
        f"- 实际作答记录：{len(attempts)} 次；主动复盘："
        f"{len({a['session_id'] for a in attempts})} 轮。",
    ]
    counts = Counter(states.values())
    lines.extend(f"- {label}：{counts[state]} 个小问。" for state, label in OUTCOME_LABELS.items())
    lines.extend(["", "## 薄弱知识点", ""])
    if not knowledge:
        lines.append("信息不足：尚未建立可关联的知识点。")
    for (subject, chapter, name), state in sorted(knowledge.items()):
        lines.append(
            f"- {subject or '科目未确认'} / {chapter or '章节未确认'} / {name}："
            f"{OUTCOME_LABELS[state]}（仅限所选题目）。"
        )
    lines.extend(["", "## 反复出现的错误", ""])
    errors: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for attempt in attempts:
        if attempt["outcome"] != "correct":
            for tag in set(attempt["error_tags"]):
                errors[(attempt["subject"], attempt["chapter"], tag)].append(attempt)
    recurring = {key: values for key, values in errors.items() if len(values) >= 2}
    if not recurring:
        lines.append("信息不足：没有至少两次实际作答支持的重复错因；不会据 AI 讲解推测。")
    for (subject, chapter, tag), values in recurring.items():
        lines.append(
            f"- {subject or '科目未确认'} / {chapter or '章节未确认'} / {tag}："
            f"{len(values)} 次；作答时间（北京时间）："
            + "、".join(format_beijing_time(a["created_at"]) for a in values)
        )
    lines.extend(["", "## 掌握情况变化", ""])
    for item in items:
        for unit in item.question.units():
            history = [a for a in attempts if a["item_id"] == item.id and a["unit_id"] == unit.id]
            label = f"{item.subject or '待确认科目'} · {item.question.number or '未编号'}"
            if unit.id != item.question.id:
                label += f" · 小问 {unit.number or '未编号'}"
            if len(history) < 2:
                lines.append(f"- {label}：信息不足，尚无两次作答可供比较。")
            else:
                first, last = history[0], history[-1]
                lines.append(
                    f"- {label}：{OUTCOME_LABELS[first['outcome']]} → "
                    f"{OUTCOME_LABELS[last['outcome']]}；"
                    f"{format_beijing_time(first['created_at'])} → "
                    f"{format_beijing_time(last['created_at'])}（北京时间）。"
                )
    lines.extend(["", "## 后续优先复习建议", ""])
    priorities = {"not_yet": 0, "partial": 1, "unknown": 2, "correct": 3}
    for item in sorted(items, key=lambda x: priorities[item_state(x, states)]):
        state = item_state(item, states)
        advice = {
            "not_yet": "先补足已记录的理解障碍，再独立重做。",
            "partial": "重点重做尚未独立完成的小问。",
            "unknown": "信息不足，先做一次独立尝试以建立基线。",
            "correct": "已有独立正确记录，可自主选择其他题检验迁移；不外推课程掌握。",
        }[state]
        lines.append(
            f"- {item.subject or '待确认科目'} · {item.question.number or '未编号'}：{advice}"
        )
    if not items:
        lines.append("信息不足：尚未选择已确认的学习材料。")
    lines.extend(["", "## 原始材料引用", ""])
    for item in items:
        source = store.source(item.source_id)
        lines.append(
            f"- 材料 `{item.id}`：{VERIFICATION_LABELS[item.verification_status]}；"
            "该状态是人工核对记录，与作答自评及文件完整性核验不同。"
        )
        if item.verification_note:
            lines.append(f"  - 人工核对依据：{item.verification_note}")
        if item.revision_of:
            lines.append(f"  - 修订自材料 `{item.revision_of}`；旧版本保留。")
        for other_id in item.conflict_item_ids:
            other = store.item(other_id)
            other_source = store.source(other.source_id)
            lines.append(
                f"  - 人工标记冲突：材料 `{other.id}` · 来源 {other_source.name} · "
                f"SHA-256 `{other_source.sha256}`；未自动裁定哪一方正确。"
            )
        pages = {page.id: page for page in store.pages(item.source_id)}
        for page_id in item.page_ids:
            page = pages[page_id]
            position = (
                f"物理页 {page.number}"
                if page.number is not None and source.kind == "pdf"
                else f"来源渲染页 / 图片 {page.number}"
                if page.number is not None
                else "文本/录音/图片段（无物理页码）"
            )
            if page.metadata.get("source_frame"):
                position += f" · 相机照片帧 {page.metadata['source_frame']}"
            if page.metadata.get("pointer"):
                position += f" · JSON 来源位置 `{page.metadata['pointer']}`"
            elif page.metadata.get("line_start"):
                position += f" · 原文行 {page.metadata['line_start']}–{page.metadata['line_end']}"
            lines.append(
                f"- {source.name} · {position} · SHA-256 `{source.sha256}` · "
                f"来源段 `{page.id}`。题号：{item.question.number or '未编号'}。"
            )
    lines.extend(
        [
            "",
            "引用表示已保存的来源记录；原件是否仍一致须主动核验。",
            "没有足够证据的条目保持信息不足；没有自动安排时间或发送提醒。",
            "",
        ]
    )
    return "\n".join(lines)
