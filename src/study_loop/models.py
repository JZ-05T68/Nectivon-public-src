"""Shared, versioned records for every input and human-led review."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

InputKind = Literal["image", "docx", "pdf", "txt", "md", "json", "html", "manual", "voice"]
Outcome = Literal["not_yet", "partial", "correct"]
OUTCOME_LABELS = {
    "unknown": "信息不足",
    "not_yet": "不会",
    "partial": "部分掌握",
    "correct": "独立正确",
}
KINDS = {"image", "docx", "pdf", "txt", "md", "json", "html", "manual", "voice"}
ATTRIBUTES = {"normal", "error", "good"}
VERIFICATION_LABELS = {
    "pending": "待验证",
    "verified": "已验证（人工记录）",
    "error": "存在错误",
}


class StudyError(ValueError):
    """Safe, user-facing study workflow failure."""


def now() -> str:
    """Use UTC for persistence; the UI is free to display Beijing time."""
    return datetime.now(UTC).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class Source:
    """An immutable, user-selected original; paths are relative to the private root."""

    id: str
    kind: InputKind
    name: str
    sha256: str
    blob: str
    created_at: str = field(default_factory=now)
    transcript: bool = False


@dataclass(frozen=True)
class Page:
    """A logical source segment; number=None means there is no physical page."""

    id: str
    source_id: str
    number: int | None
    text: str = ""
    image: str = ""
    image_sha256: str = ""
    needs_review: bool = False
    import_job: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Question:
    """A parent or subquestion; observed answers and human decisions stay separate."""

    id: str = field(default_factory=lambda: uuid4().hex)
    number: str = ""
    content: str = ""
    question_type: str = "待确认"
    attribute: str = "normal"
    knowledge: list[str] = field(default_factory=list)
    observed_answer: str = ""
    observed_analysis: str = ""
    reference_answer: str = ""
    reference_analysis: str = ""
    answer_status: str = "unconfirmed"
    obstacle: str = ""
    obstacle_evidence: str = ""
    children: list[Question] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Question:
        data = dict(value)
        data["children"] = [cls.from_dict(child) for child in data.get("children", [])]
        return cls(**data)

    def units(self) -> list[Question]:
        """Only leaf questions receive outcomes; parent text is shared context."""
        return [leaf for child in self.children for leaf in child.units()] or [self]


@dataclass
class Draft:
    """All import routes converge here, before explicit human confirmation."""

    id: str
    source_id: str
    page_ids: list[str]
    question: Question
    subject: str = ""
    chapter: str = ""
    material_kind: str = "question"
    extraction: str = "local_rules"
    warnings: list[str] = field(default_factory=list)
    schema_version: int = 1
    verification_status: str = "pending"
    verification_note: str = ""
    conflict_item_ids: list[str] = field(default_factory=list)
    revision_of: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Draft:
        data = dict(value)
        data["question"] = Question.from_dict(data["question"])
        return cls(**data)


def validate_page_ids(page_ids: list[str]) -> None:
    """Reject malformed or repeated source references before persistence or display."""
    if (
        not isinstance(page_ids, list)
        or not 1 <= len(page_ids) <= 200
        or any(not isinstance(value, str) or not value.strip() for value in page_ids)
        or len(set(page_ids)) != len(page_ids)
    ):
        raise StudyError("来源范围无效、重复或超过 200 段。")


def validate_draft(draft: Draft) -> None:
    """Reject ambiguous identity, oversized trees and unsupported confirmation states."""
    if draft.schema_version != 1:
        raise StudyError("草稿版本或来源范围无效。")
    validate_page_ids(draft.page_ids)
    if draft.material_kind not in {"question", "knowledge", "note"}:
        raise StudyError("材料类型无效。")
    if draft.verification_status not in VERIFICATION_LABELS:
        raise StudyError("知识验证状态无效。")
    if (draft.verification_status != "pending" or draft.conflict_item_ids) and not (
        draft.verification_note.strip()
    ):
        raise StudyError("标记已验证、存在错误或来源冲突时，请记录人工核对依据。")
    if len(draft.conflict_item_ids) > 100 or len(set(draft.conflict_item_ids)) != len(
        draft.conflict_item_ids
    ):
        raise StudyError("冲突来源重复或数量过多。")
    ids: set[str] = set()

    def visit(question: Question, depth: int) -> None:
        if depth > 4 or len(ids) >= 100 or not question.id or question.id in ids:
            raise StudyError("题目层级过深、数量过多或小问标识重复。")
        ids.add(question.id)
        if not question.content.strip() or len(question.content) > 200_000:
            raise StudyError("请补充题目、知识点或笔记内容，并控制单条长度。")
        if question.attribute not in ATTRIBUTES:
            raise StudyError("请选择普通题、错题或好题。")
        if question.answer_status not in {"unconfirmed", "source_confirmed", "user_note"}:
            raise StudyError("答案来源状态无效。")
        if question.answer_status == "unconfirmed" and (
            question.reference_answer or question.reference_analysis
        ):
            raise StudyError("请先核对答案来源；未确认的识别结果不能成为参考答案。")
        if question.obstacle and not question.obstacle_evidence.strip():
            raise StudyError("记录错因或理解障碍时，请填写实际作答或原文依据。")
        if len(question.knowledge) > 30 or any(not k.strip() for k in question.knowledge):
            raise StudyError("知识点标签为空或过多。")
        for child in question.children:
            visit(child, depth + 1)

    visit(draft.question, 0)
