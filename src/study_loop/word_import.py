"""Supplement rendered Word pages with complete, equation-preserving text."""

from __future__ import annotations

import re

from .models import Draft, Page, Question, Source
from .storage import StudyStore
from .word_math import docx_markdown

SECTION = re.compile(r"(?m)^#{1,6}\s+(?:[一二三四五六七八九十]+\s+)?(.+?)\s*$")
QUESTION_HEADERS = {"原题", "原始题目", "题目"}
ANSWER_HEADERS = {"答案", "标准答案", "参考答案", "计算结果", "课程平台公布的正确答案"}
ANALYSIS_HEADERS = {"解析", "完整解析", "计算解析", "标准解析", "逐步推导", "计算依据"}


def structured_word(source: Source, page: Page, page_ids: list[str]) -> list[Draft]:
    """Only explicit headings separate question/answer; never infer correctness or roles."""
    headings = list(SECTION.finditer(page.text))
    starts = [m for m in headings if m[1] in QUESTION_HEADERS]
    if len(starts) != 1:
        from .structure import local_drafts

        return local_drafts(source, page)
    sections = {
        m[1]: page.text[
            m.end() : headings[i + 1].start() if i + 1 < len(headings) else len(page.text)
        ].strip()
        for i, m in enumerate(headings)
    }
    question = Question(
        id=page.id + ":question",
        number=source.name.rsplit(".", 1)[0],
        content=sections[starts[0][1]],
        observed_answer="\n\n".join(v for k, v in sections.items() if k in ANSWER_HEADERS),
    )
    # Retain the whole analytical tail, including deeper subheadings and follow-ups.
    analysis = next((m for m in headings if m[1] in ANALYSIS_HEADERS), None)
    question.observed_analysis = page.text[analysis.start() :] if analysis else ""
    return [
        Draft(
            id=question.id,
            source_id=source.id,
            page_ids=page_ids,
            question=question,
            extraction="word_headings_v1",
            warnings=[
                "按 Word 明确标题提取；答案、解析与追问仍待本人核对。",
                "公式文字段对应完整 Word，不伪造物理页码；原页图与原件均保留。",
                "文档内追问是材料记录，不冒充本次用户/AI 实时会话。",
            ],
        )
    ]


def supplement(store: StudyStore, source: Source, data: bytes) -> Page:
    """New immutable parser identity; previous rendered snapshots are never overwritten."""
    identity = source.id + ":word-math-v1"
    prior = next((p for p in store.pages(source.id) if p.id == identity), None)
    if prior is not None:
        return prior
    page = Page(
        identity,
        source.id,
        None,
        text=docx_markdown(data),
        needs_review=True,
        metadata={
            "format": "word_math_v1",
            "pointer": "word/document.xml（全文）",
            "warning": "原生公式转为 LaTeX；图片只保留引用，请对照 Word 渲染页。",
        },
    )
    store.put_page(page)
    return page
