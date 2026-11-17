"""Read-only projections of explicitly recorded errors and human review sessions."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from src.time_display import to_beijing_time

from .models import Draft, Question


@dataclass(frozen=True)
class ErrorEvidence:
    """One existing event or explicit material marker, never an inferred weakness."""

    id: str
    item_id: str
    unit_id: str
    origin: str
    created_at: str | None
    subject: str
    chapter: str
    knowledge: tuple[str, ...]
    tags: tuple[str, ...]
    answer: str = ""
    note: str = ""

    @property
    def question_key(self) -> tuple[str, str]:
        """Revision identity is retained, matching the existing attempt contract."""
        return self.item_id, self.unit_id


@dataclass(frozen=True)
class ReviewEvent:
    """A single existing human-started session, including separate later rounds."""

    id: str
    item_ids: tuple[str, ...]
    started_at: str | None
    ended_at: str | None
    status: str
    attempts: tuple[dict, ...]
    unavailable_items: tuple[str, ...]
    invalid_reference: bool = False


def question_nodes(question: Question) -> list[Question]:
    """Keep explicit parent markers distinct from leaf-question attempts."""
    return [question, *(node for child in question.children for node in question_nodes(child))]


def _labels(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict.fromkeys(v.strip() for v in value if isinstance(v, str) and v.strip()))


def error_evidence(items: list[Draft], attempts: list[dict]) -> list[ErrorEvidence]:
    """Count only explicit human markers and explicitly confirmed actual answer errors.

    Material verification, model suggestions, partial completion and historical
    error tags alone are insufficient. Marker dates are unknown: confirmation or
    import dates do not establish when the learner marked the question wrong.
    """
    questions = {item.id: item for item in items if item.material_kind == "question"}
    result: list[ErrorEvidence] = []
    for item in questions.values():
        for unit in question_nodes(item.question):
            if unit.attribute != "error":
                continue
            result.append(ErrorEvidence(
                id=f"marker:{item.id}:{unit.id}", item_id=item.id, unit_id=unit.id,
                origin="manual_marker", created_at=None,
                subject=item.subject, chapter=item.chapter,
                knowledge=_labels(unit.knowledge or item.question.knowledge),
                tags=(unit.obstacle.strip(),) if (
                    unit.obstacle.strip() and unit.obstacle_evidence.strip()
                ) else (),
                note=unit.obstacle_evidence,
            ))
    seen: set[str] = set()
    for attempt in attempts:
        identity = attempt.get("id")
        item = questions.get(attempt.get("item_id"))
        if (
            not isinstance(identity, str) or not identity or identity in seen
            or item is None or attempt.get("identity_valid") is not True
            or attempt.get("source") != "user_attempt"
            or attempt.get("error_confirmed") is not True
            or attempt.get("outcome") not in {"not_yet", "partial"}
            or not isinstance(attempt.get("answer"), str) or not attempt["answer"].strip()
        ):
            continue
        unit = next((u for u in item.question.units() if u.id == attempt.get("unit_id")), None)
        if unit is None:
            continue
        seen.add(identity)
        result.append(ErrorEvidence(
            id=identity, item_id=item.id, unit_id=unit.id, origin="user_attempt",
            created_at=attempt.get("created_at"),
            subject=item.subject, chapter=item.chapter,
            knowledge=_labels(attempt.get("knowledge")), tags=_labels(attempt.get("error_tags")),
            answer=attempt["answer"], note=str(attempt.get("note") or ""),
        ))
    return result


def evidence_counts(records: list[ErrorEvidence]) -> dict[str, int]:
    """Keep distinct questions, repeat answer errors and manual markers separate."""
    return {
        "questions": len({r.question_key for r in records}),
        "attempts": sum(r.origin == "user_attempt" for r in records),
        "markers": len({r.question_key for r in records if r.origin == "manual_marker"}),
        "unknown_time": sum(_instant(r.created_at) is None for r in records),
        "unclassified": len({r.question_key for r in records if not r.knowledge}),
        "untagged": len({r.question_key for r in records if not r.tags}),
        "tag_assignments": sum(len(r.tags) for r in records),
    }


def distribution(records: list[ErrorEvidence], dimension: str) -> dict[str, list[ErrorEvidence]]:
    """Return the exact evidence behind every selectable distribution entry.

    Groups may overlap. Charts count distinct version/question identities inside
    each group rather than pretending overlapping labels are exclusive shares.
    """
    grouped: dict[str, list[ErrorEvidence]] = defaultdict(list)
    for record in records:
        course = record.subject or "未分类科目"
        chapter = record.chapter or "未分类章节"
        if dimension == "knowledge":
            labels = [f"{course} / {chapter} / {name}" for name in record.knowledge]
            labels = labels or ["未分类知识点"]
        elif dimension == "chapter":
            labels = [f"{course} / {chapter}"]
        elif dimension == "tags":
            labels = list(record.tags) or ["未记录错因"]
        else:
            raise ValueError("Unsupported error distribution dimension")
        for label in labels:
            grouped[label].append(record)
    return dict(sorted(grouped.items(), key=lambda pair: (
        -len({r.question_key for r in pair[1]}), pair[0],
    )))


def _instant(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return to_beijing_time(value)
    except (ValueError, TypeError, OverflowError):
        return None


def recent_reviews(
    items: list[Draft], sessions: list[dict], attempts: list[dict], *, limit: int | None = 10,
) -> list[ReviewEvent]:
    """Stable ordering by actual start time, with unknown times last and ID tie-breaks.

    Only session records establish a review event. Rendering, import jobs and
    model completion never add events. A retry of one submission cannot add a
    second attempt; separate session IDs are always preserved.
    """
    known = {item.id for item in items}
    result: list[ReviewEvent] = []
    seen: set[str] = set()
    for session in sessions:
        identity = session.get("id")
        if not isinstance(identity, str) or not identity or identity in seen:
            continue
        seen.add(identity)
        invalid_reference = False
        try:
            raw = json.loads(session.get("item_ids", ""))
            if not isinstance(raw, list) or any(not isinstance(v, str) or not v for v in raw):
                raise ValueError("Invalid item identities")
            item_ids = tuple(dict.fromkeys(raw))
        except (ValueError, TypeError):
            item_ids, invalid_reference = (), True
        valid_attempts = {
            a["id"]: a for a in attempts
            if a.get("identity_valid") is True and isinstance(a.get("id"), str)
            and a.get("session_id") == identity and a.get("item_id") in item_ids
            and a.get("source") == "user_attempt"
        }
        ended = session.get("ended_at")
        result.append(ReviewEvent(
            id=identity, item_ids=item_ids, started_at=session.get("started_at"),
            ended_at=ended, status="completed" if ended is not None else "in_progress",
            attempts=tuple(valid_attempts.values()),
            unavailable_items=tuple(value for value in item_ids if value not in known),
            invalid_reference=invalid_reference,
        ))
    result.sort(key=lambda value: (
        -(_instant(value.started_at).timestamp()) if _instant(value.started_at) else float("inf"),
        value.id,
    ))
    return result if limit is None else result[:limit]
