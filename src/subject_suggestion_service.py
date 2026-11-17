"""AI 建议学科 (v0.8.8 AI-First offload #2) — suggest, never auto-decide.

Turns the mandatory manual 学科 pick into "AI suggests + one-click user
confirmation" while preserving the human right to override.  The subject
evidence is already stored locally with the material: the document title and
each page's native text layer (the same source the AI reading runs on), plus
any reading-result keywords/summaries a caller injects.  This module only
reads that evidence — it never calls AI again, never becomes a startup
dependency, and never writes a confirmed subject on its own.

Design notes:
- Pure :func:`suggest_subjects` takes evidence text + candidate subjects and
  returns ranked ``(subject, reason)`` pairs, so it is unit-testable without
  a database or runtime.
- No rapidfuzz dependency here; matching is substring + jieba
  token containment, deterministic and offline.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import jieba

LOGGER = logging.getLogger(__name__)

#: How many suggestions the UI should offer at most.
MAX_SUGGESTIONS = 3
#: Characters of page text sampled per page when building evidence.
_PAGE_TEXT_SAMPLE = 400
#: Pages sampled per document when building evidence.
_MAX_SAMPLED_PAGES = 12


def build_document_evidence(
    database,
    document_id: int,
    *,
    reading_store=None,
) -> str:
    """Concatenate the locally stored subject evidence for one document.

    Uses the document title and the native text layer of its first pages
    (offline, always available).  When a reading ``store`` is injected, the
    AI reading keywords/summary for each sampled page are merged in too —
    the reading results are the richest signal, but they are optional so the
    suggestion degrades gracefully to title + text when absent.
    """

    parts: list[str] = []
    try:
        document = database.get_document(document_id)
    except Exception:  # noqa: BLE001 - a missing document just yields no evidence
        LOGGER.debug("读取文档标题失败：document_id=%s", document_id, exc_info=True)
        document = None
    if document is not None:
        parts.append(str(getattr(document, "title", "") or ""))

    try:
        pages = database.list_pages(document_id)[:_MAX_SAMPLED_PAGES]
    except Exception:  # noqa: BLE001 - text evidence is best-effort
        LOGGER.debug("读取页面文字失败：document_id=%s", document_id, exc_info=True)
        pages = []
    for page in pages:
        text = str(getattr(page, "extracted_text", "") or "").strip()
        if text:
            parts.append(text[:_PAGE_TEXT_SAMPLE])
        if reading_store is not None:
            try:
                reading = reading_store.page_reading(page.id)
            except Exception:  # noqa: BLE001 - reading results are optional
                LOGGER.debug("读取 AI 阅读结果失败：page_id=%s", page.id, exc_info=True)
                reading = None
            if reading is not None:
                parts.append(str(getattr(reading, "summary", "") or ""))
                parts.extend(str(keyword) for keyword in getattr(reading, "keywords", ()))
    return "\n".join(part for part in parts if part)


def _normalize(text: str) -> str:
    return "".join(str(text or "").split()).casefold()


def suggest_subjects(
    evidence_text: str,
    candidate_subjects: Iterable[str],
    *,
    limit: int = MAX_SUGGESTIONS,
) -> list[tuple[str, str]]:
    """Rank candidate subjects whose names appear in the evidence.

    Returns at most ``limit`` ``(subject, reason)`` pairs, most specific
    first (longer subject names win, then more token hits).  Deduplicates by
    normalized subject name.  An empty evidence or empty candidate list
    yields no suggestions — never a fabricated subject.
    """

    evidence = _normalize(evidence_text)
    if not evidence:
        return []
    evidence_tokens = {token for token in jieba.lcut(evidence_text) if token.strip()}

    seen: set[str] = set()
    scored: list[tuple[int, int, str, str]] = []
    for raw in candidate_subjects:
        subject = str(raw or "").strip()
        if not subject:
            continue
        key = _normalize(subject)
        if not key or key in seen:
            continue
        seen.add(key)
        reason = ""
        specificity = 0
        if key in evidence:
            specificity = len(key)
            reason = f"资料内容里直接出现了「{subject}」"
        else:
            subject_tokens = [
                token for token in jieba.lcut(subject) if token.strip() and len(token) > 1
            ]
            hits = [token for token in subject_tokens if token in evidence_tokens]
            if subject_tokens and len(hits) == len(subject_tokens):
                specificity = sum(len(token) for token in hits)
                reason = "资料关键词命中：" + "、".join(hits)
        if specificity:
            scored.append((specificity, len(reason), subject, reason))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [(subject, reason) for _, _, subject, reason in scored[: max(0, limit)]]


def ordered_candidates(
    policy_subjects: Sequence[str],
    *,
    saved_subject: str = "",
    existing_subjects: Sequence[str] = (),
) -> list[str]:
    """Build the candidate subject list, most-likely-first.

    The saved subject and subjects already used on this platform come first
    (so 液压与气动 that the student confirmed before is re-suggested), then
    the policy's allowed subjects.  Duplicates are removed, order preserved.
    """

    candidates: list[str] = []
    for group in (
        [saved_subject] if saved_subject else [],
        list(existing_subjects),
        list(policy_subjects),
    ):
        for subject in group:
            cleaned = str(subject or "").strip()
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned)
    return candidates
