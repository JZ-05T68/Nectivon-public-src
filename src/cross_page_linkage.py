"""Deterministic cross-page linkage for photographed exercise sets (P0-1).

Merged multi-photo documents make one exercise span two photos.  The model
is instructed never to drop a leading continuation line (CONTINUATION_RULES);
this module then closes the loop locally, without any AI call:

- :func:`merge_cross_page_continuations` — attach a page's leading
  「（接上页）」candidates to the previous page's last incomplete leaf.
- :func:`link_document_figures` — when a candidate references 图 X that is
  registered (page_figures) on a sibling page, record where the figure
  lives so the student can find the original drawing.

Both functions only append information (stem text / visual notes) and log a
stem revision; they never delete user content and never re-bill the AI.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.figure_index_service import figure_key as _figure_key

if TYPE_CHECKING:
    from src.database import Database
    from src.question_candidate_service import QuestionCandidateStore

LOGGER = logging.getLogger(__name__)

CONTINUATION_PREFIX = "（接上页）"
_CROSS_PAGE_NOTE = re.compile(r"位于本资料第 \d+ 页")
_EDITORIAL_RESIDUE = re.compile(r"（?注[:：].*$")


def sanitize_region_description(description: str) -> str:
    """Strip model editorial residue; keep only the caption itself."""

    return _EDITORIAL_RESIDUE.sub("", str(description or "")).strip()


def clip_regions_to_siblings(
    candidates: list, page_figures: list[dict], text_blocks: list[dict] | None = None,
) -> int:
    """Bound each figure region between its same-page neighbours (P0-2/P1-3).

    The model's boxes separate neighbouring figures and prose columns
    correctly, but local ink completion can walk across the narrow gutter
    between them.  When a neighbour shares the row band on one side, that
    side gets a hard horizontal stop in the gutter between the two boxes.
    Model edges can clip detached labels and output arrows; the midpoint
    allows recovery without reaching the neighbouring figure. Prose still
    clips at the figure's own edge, since text can occupy the whole gutter.
    Siblings are same-page figure registry entries, other candidates'
    stem/shared regions, and text blocks from the page's content blocks.
    """

    from src.question_candidate_service import iter_question_nodes

    def boxes_overlap_y(a: list, b: list) -> bool:
        overlap = min(a[3], b[3]) - max(a[1], b[1])
        return overlap > 0.3 * min(a[3] - a[1], b[3] - b[1])

    def owns(region: dict, node, figure: dict) -> bool:
        name = _figure_key(figure.get("name", ""))
        description = _figure_key(region.get("description", ""))
        refs = {_figure_key(ref) for ref in (getattr(node, "figure_refs", []) or [])}
        return (
            name in description
            or description in name
            or name in refs
            or not figure.get("bbox")
        )

    def valid(box: object) -> list[float] | None:
        if (isinstance(box, list) and len(box) == 4
                and all(isinstance(v, (int, float)) for v in box)
                and 0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000):
            return [float(v) for v in box]
        return None

    regions_with_owner: list[tuple[dict, object]] = []
    for node in iter_question_nodes(candidates):
        for region in getattr(node, "visual_regions", []) or []:
            if region.get("role") in ("stem", "shared") and region.get("bbox"):
                regions_with_owner.append((region, node))
    changed = 0
    for region, node in regions_with_owner:
        bbox = valid(region.get("bbox"))
        if bbox is None:
            continue
        left_cut, right_cut = 0.0, 1000.0
        for provider, prose in ((page_figures or [], False), (text_blocks or [], True)):
            for item in provider:
                if item in (page_figures or []) and owns(region, node, item):
                    continue
                sibling = valid(item.get("bbox"))
                if sibling is None or not boxes_overlap_y(bbox, sibling):
                    continue
                if sibling[2] <= bbox[0]:
                    left_cut = max(left_cut, bbox[0] if prose else (sibling[2] + bbox[0]) / 2)
                elif sibling[0] >= bbox[2]:
                    right_cut = min(right_cut, bbox[2] if prose else (bbox[2] + sibling[0]) / 2)
        for other, _owner in regions_with_owner:
            if other is region:
                continue
            if _figure_key(other.get("description", "")) == _figure_key(
                region.get("description", "")
            ) and other.get("description"):
                continue
            sibling = valid(other.get("bbox"))
            if sibling is None or not boxes_overlap_y(bbox, sibling):
                continue
            if sibling[2] <= bbox[0]:
                left_cut = max(left_cut, (sibling[2] + bbox[0]) / 2)
            elif sibling[0] >= bbox[2]:
                right_cut = min(right_cut, (bbox[2] + sibling[0]) / 2)
        if left_cut > 0 or right_cut < 1000:
            region["sibling_clip"] = [left_cut, bbox[1], right_cut, bbox[3]]
            changed += 1
    return changed


def _is_continuation(candidate) -> bool:
    return (
        not candidate.children
        and candidate.status == "pending"
        and (not candidate.number.strip() or candidate.stem.startswith(CONTINUATION_PREFIX))
    )


_BARE_SUBQUESTION = re.compile(r"^[（(]\s*([0-9一二三四五六七八九十]+)\s*[）)]")


def merge_cross_page_continuations(
    store: QuestionCandidateStore, database: Database, page_id: int
) -> list[dict[str, str]]:
    """Fold this page's leading continuation candidates into the prior page.

    A continuation that opens with a bare sub-question label (``(2) …``)
    joins the previous page's last incomplete composite ROOT as a new
    child; plain trailing text appends to that root's / the last
    incomplete leaf's stem.  Returns one record per merge.  No merge
    happens without an incomplete receiver — an orphan continuation stays
    visible on its own page instead of being glued to an unrelated
    question.
    """

    from src.question_candidate_service import (
        QuestionCandidate,
        _append_stem_revision,
        _utc_now,
        canonical_question_number,
        iter_atomic_leaves,
    )

    page = database.get_page(page_id)
    if page is None or page.page_number <= 1:
        return []
    previous_pages = [
        item for item in database.list_pages(page.document_id)
        if item.page_number < page.page_number
    ]
    if not previous_pages:
        return []
    previous = max(previous_pages, key=lambda item: item.page_number)
    current = store.page_candidates(page_id) or []
    leading: list = []
    rest = list(current)
    while rest and _is_continuation(rest[0]):
        leading.append(rest.pop(0))
    if not leading:
        return []
    previous_candidates = store.page_candidates(previous.id) or []
    added_text = "\n".join(
        node.stem.removeprefix(CONTINUATION_PREFIX).strip() for node in leading
    ).strip()
    if not added_text:
        return []

    # A leading continuation can only belong to the LAST question of the
    # previous page; an incomplete question in the middle stays untouched.
    last_root = previous_candidates[-1] if previous_candidates else None
    incomplete_root = (
        last_root if last_root is not None and last_root.completeness == "incomplete" else None
    )
    label = _BARE_SUBQUESTION.match(added_text.lstrip())
    merged_number = ""
    if incomplete_root is not None and incomplete_root.children and label:
        parent_number = canonical_question_number(incomplete_root.number)
        child = QuestionCandidate(
            number=f"{parent_number}({label.group(1)})",
            stem=added_text,
            completeness=leading[-1].completeness,
            incomplete_reason=leading[-1].incomplete_reason,
            extracted_at=_utc_now(),
            split_source="ai_inference",
            split_confidence="medium",
        )
        child.validate()
        incomplete_root.children.append(child)
        if (
            leading[-1].completeness == "complete"
            and all(node.completeness == "complete" for node in incomplete_root.children)
        ):
            incomplete_root.completeness = "complete"
            incomplete_root.incomplete_reason = ""
        receiver = incomplete_root
        merged_number = child.number
    elif incomplete_root is not None and not incomplete_root.children:
        incomplete_root.stem = f"{incomplete_root.stem}\n{added_text}".strip()
        if leading[-1].completeness == "complete":
            incomplete_root.completeness = "complete"
            incomplete_root.incomplete_reason = ""
        receiver = incomplete_root
        merged_number = incomplete_root.number
    else:
        receiver = next(
            (
                leaf
                for _root, leaf, _path, _ancestors in reversed(
                    list(iter_atomic_leaves(previous_candidates))
                )
                if leaf.completeness == "incomplete"
            ),
            None,
        )
        if receiver is None:
            return []
        receiver.stem = f"{receiver.stem}\n{added_text}".strip()
        if leading[-1].completeness == "complete":
            receiver.completeness = "complete"
            receiver.incomplete_reason = ""
        merged_number = receiver.number
    _append_stem_revision(
        receiver,
        kind="cross_page_merge",
        detail=f"并入第 {page.page_number} 页开头的延续内容：{added_text[:180]}",
    )
    store.save_page_candidates(previous.id, previous_candidates)
    store.save_page_candidates(page_id, rest)
    LOGGER.info(
        "跨页延续已合并：page %s → page %s 题号 %s",
        page.page_number, previous.page_number, merged_number or "（无题号）",
    )
    return [
        {
            "number": merged_number or "（无题号）",
            "from_page": str(page.page_number),
            "added": added_text[:120],
        }
    ]


_FIGURE_MENTION = re.compile(r"图\s*[0-9０-９]+[-－—–][0-9０-９]+[A-Za-z]?[0-9]?")

_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def _referenced_figures(node) -> list[str]:
    """Figure references from BOTH figure_refs and visual_notes mentions.

    Round-2 evidence: the model often writes 「本题所需图 6-8a 不在本页」only
    in visual_notes while leaving figure_refs empty; harvesting both keeps
    the cross-page link deterministic.
    """

    references = [str(item) for item in (getattr(node, "figure_refs", []) or [])]
    for match in _FIGURE_MENTION.finditer(str(getattr(node, "visual_notes", "") or "")):
        mention = match.group(0).translate(_FULLWIDTH_DIGITS)
        if mention not in references:
            references.append(mention)
    return references


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sibling_candidate_label(figure: dict, page_number: int) -> str:
    name = str(figure.get("name", "")).strip()
    return f"本资料第 {page_number} 页 · {name or '（无图注）'}"


def _global_candidate_label(entry: dict) -> str:
    name = str(entry.get("name", "")).strip()
    description = str(entry.get("description", "")).strip()
    label = name or (description[:40] if description else "（无图注）")
    return f"《{entry.get('document_title', '')}》第 {entry.get('page_number', '?')} 页 · {label}"


def _candidate_record(
    *, name: str, description: str, document_id: int, page_id: int,
    page_number: int, document_title: str, bbox: list, match_basis: str,
    label: str,
) -> dict:
    """One surfaced hit for the manual-confirmation panel (never auto-bound)."""

    return {
        "label": label[:160],
        "name": str(name)[:80],
        "description": str(description)[:200],
        "document_id": int(document_id),
        "page_id": int(page_id),
        "page_number": int(page_number) if str(page_number).isdigit() else page_number,
        "document_title": str(document_title)[:120],
        "bbox": [float(v) for v in bbox],
        "match_basis": match_basis,
    }


def _resolve_figure_reference(
    reference: str,
    reference_key: str,
    *,
    page,  # noqa: ANN001 - Page model
    pages: list,
    registry: dict[int, list[dict]],
    store_root,
    document_id: int,
    document_title: str,
) -> tuple[str, dict | None, list[dict]]:
    """Resolve one figure reference deterministically (round-1 P1-1 fix).

    Returns ``(state, region, surfaced_candidates)`` where state is one of
    ``linked`` / ``ambiguous`` / ``unresolved``. Exact normalized-name hits
    outrank boundary-safe substring hits, same-document hits outrank the
    global index, and a tier with MORE THAN ONE hit is surfaced for manual
    confirmation instead of silently binding the first one.
    """

    from src.figure_index_service import (
        LINK_BASIS_EXACT_GLOBAL_INDEX,
        LINK_BASIS_EXACT_SAME_DOCUMENT,
        LINK_BASIS_SUBSTRING_GLOBAL_INDEX,
        LINK_BASIS_SUBSTRING_SAME_DOCUMENT,
        boundary_safe_contains,
        find_figure_hits,
        virtual_region,
    )

    sibling_exact: list[tuple[dict, object]] = []
    sibling_substring: list[tuple[dict, object]] = []
    for other in pages:
        if other.id == page.id:
            continue
        for figure in registry.get(other.id, []):
            name_key = _figure_key(figure.get("name", ""))
            if not name_key:
                continue
            if name_key == reference_key:
                sibling_exact.append((figure, other))
            elif boundary_safe_contains(
                name_key, reference_key
            ) or boundary_safe_contains(reference_key, name_key):
                sibling_substring.append((figure, other))

    sibling_identities = {
        (int(other.id), tuple(float(v) for v in figure.get("bbox", [])))
        for figure, other in (*sibling_exact, *sibling_substring)
    }
    global_hits = [
        hit
        for hit in find_figure_hits(store_root, reference)
        if int(hit.entry.get("page_id", 0)) != int(page.id)
        and (
            int(hit.entry.get("page_id", 0)),
            tuple(float(v) for v in hit.entry.get("bbox", [])),
        )
        not in sibling_identities
    ]
    global_exact = [hit for hit in global_hits if hit.match_basis == "exact"]
    global_substring = [hit for hit in global_hits if hit.match_basis == "substring"]

    def sibling_region(figure: dict, other, basis: str, confidence: str) -> dict:
        where = f"本资料第 {other.page_number} 页"
        return {
            "role": "shared",
            "bbox": [float(v) for v in figure.get("bbox", [])],
            "description": f"{reference}（{where}，跨页插图）",
            "source_page_id": int(other.id),
            "source_document_id": document_id,
            "source_label": where,
            "figure_number": reference,
            "link_basis": basis,
            "link_confidence": confidence,
            "link_confirmed": False,
        }

    if len(sibling_exact) == 1:
        figure, other = sibling_exact[0]
        return (
            "linked",
            sibling_region(
                figure, other, LINK_BASIS_EXACT_SAME_DOCUMENT, "high"
            ),
            [],
        )
    if len(sibling_exact) > 1:
        return "ambiguous", None, [
            _candidate_record(
                name=figure.get("name", ""),
                description=figure.get("description", ""),
                document_id=document_id,
                page_id=other.id,
                page_number=other.page_number,
                document_title=document_title,
                bbox=figure.get("bbox", []),
                match_basis="exact",
                label=_sibling_candidate_label(figure, other.page_number),
            )
            for figure, other in sibling_exact
        ]
    if len(global_exact) == 1:
        entry = global_exact[0].entry
        return (
            "linked",
            virtual_region(
                entry,
                description=reference,
                figure_number=reference,
                link_basis=LINK_BASIS_EXACT_GLOBAL_INDEX,
                link_confidence="high",
            ),
            [],
        )
    if len(global_exact) > 1:
        return "ambiguous", None, [
            _candidate_record(
                name=hit.entry.get("name", ""),
                description=hit.entry.get("description", ""),
                document_id=hit.entry.get("document_id", 0),
                page_id=hit.entry.get("page_id", 0),
                page_number=hit.entry.get("page_number", "?"),
                document_title=hit.entry.get("document_title", ""),
                bbox=hit.entry.get("bbox", []),
                match_basis="exact",
                label=_global_candidate_label(hit.entry),
            )
            for hit in global_exact
        ]
    if len(sibling_substring) + len(global_substring) == 1:
        if sibling_substring:
            figure, other = sibling_substring[0]
            return (
                "linked",
                sibling_region(
                    figure, other, LINK_BASIS_SUBSTRING_SAME_DOCUMENT, "medium"
                ),
                [],
            )
        entry = global_substring[0].entry
        return (
            "linked",
            virtual_region(
                entry,
                description=reference,
                figure_number=reference,
                link_basis=LINK_BASIS_SUBSTRING_GLOBAL_INDEX,
                link_confidence="medium",
            ),
            [],
        )
    if sibling_substring or global_substring:
        surfaced = [
            _candidate_record(
                name=figure.get("name", ""),
                description=figure.get("description", ""),
                document_id=document_id,
                page_id=other.id,
                page_number=other.page_number,
                document_title=document_title,
                bbox=figure.get("bbox", []),
                match_basis="substring",
                label=_sibling_candidate_label(figure, other.page_number),
            )
            for figure, other in sibling_substring
        ] + [
            _candidate_record(
                name=hit.entry.get("name", ""),
                description=hit.entry.get("description", ""),
                document_id=hit.entry.get("document_id", 0),
                page_id=hit.entry.get("page_id", 0),
                page_number=hit.entry.get("page_number", "?"),
                document_title=hit.entry.get("document_title", ""),
                bbox=hit.entry.get("bbox", []),
                match_basis="substring",
                label=_global_candidate_label(hit.entry),
            )
            for hit in global_substring
        ]
        return "ambiguous", None, surfaced[:8]
    return "unresolved", None, []


def link_document_figures(
    store: QuestionCandidateStore, database: Database, document_id: int
) -> list[str]:
    """Attach referenced figures to questions — same page, same document,
    or ANY imported photo via the global figure index (round-2 issue 2).

    A cross-page/global hit both records a visual note and injects a
    cross-page virtual region carrying structured provenance (figure
    number, link basis, confidence, per-link confirmation). Round-1 P1-1:
    matching is exact-first with a boundary-safe substring fallback, and a
    tier with multiple hits is NEVER auto-bound — it is surfaced in
    ``unresolved_figure_refs`` for the manual-confirmation panel instead.
    Deterministic caption matching only — no AI call, no OCR, no adjacency
    assumption.
    """

    from src.question_candidate_service import iter_question_nodes

    document = database.get_document(document_id)
    document_title = document.title if document is not None else ""
    pages = sorted(database.list_pages(document_id), key=lambda item: item.page_number)
    registry: dict[int, list[dict]] = {}
    for page in pages:
        registry[page.id] = store.page_figures(page.id) or []
    notes: list[str] = []
    for page in pages:
        local_keys = {_figure_key(figure.get("name", "")) for figure in registry[page.id]}
        candidates = store.page_candidates(page.id)
        if not candidates:
            continue
        changed = False
        for node in iter_question_nodes(candidates):
            if getattr(node, "binding_confirmed", False):
                continue
            for reference in _referenced_figures(node):
                reference_key = _figure_key(reference)
                if not reference_key or reference_key in local_keys:
                    continue
                if any(
                    str(region.get("figure_number", "")) == reference
                    for region in node.visual_regions
                ):
                    continue  # already linked with provenance (idempotent)
                state, region, surfaced = _resolve_figure_reference(
                    reference,
                    reference_key,
                    page=page,
                    pages=pages,
                    registry=registry,
                    store_root=store.root,
                    document_id=document_id,
                    document_title=document_title,
                )
                if region is not None and state == "linked":
                    note = f"{reference} 位于{region['source_label']}（已跨照片关联）"
                    if reference in (node.visual_notes or "") and any(
                        r.get("source_page_id") == region.get("source_page_id")
                        for r in node.visual_regions
                    ):
                        continue
                    region["question_number"] = node.number or ""
                    node.visual_notes = (
                        f"{node.visual_notes}；{note}" if node.visual_notes else note
                    )
                    if node.visual_dependency == "none":
                        node.visual_dependency = "required"
                    node.visual_regions.append(region)
                    notes.append(
                        f"第 {page.page_number} 页 {node.number or '（无题号）'}：{note}"
                    )
                    changed = True
                    continue
                # Ambiguous or unresolved: never bind, surface for the
                # manual-confirmation panel (round-1: explicit repair path
                # instead of a silent wrong binding or a dead end).
                record = {
                    "reference": reference,
                    "state": state,
                    "at": _utc_now(),
                    "candidates": surfaced,
                }
                existing = [
                    item
                    for item in getattr(node, "unresolved_figure_refs", [])
                    if isinstance(item, dict)
                    and item.get("reference") == reference
                ]
                if existing and (
                    existing[0].get("state") == state
                    and [
                        (c.get("label"), c.get("match_basis"))
                        for c in existing[0].get("candidates", [])
                    ]
                    == [(c.get("label"), c.get("match_basis")) for c in surfaced]
                ):
                    continue  # unchanged surfacing; stay idempotent
                node.unresolved_figure_refs = [
                    item
                    for item in getattr(node, "unresolved_figure_refs", [])
                    if not (
                        isinstance(item, dict) and item.get("reference") == reference
                    )
                ]
                node.unresolved_figure_refs.append(record)
                reason = (
                    "命中多个图形候选，为避免误配未自动关联"
                    if state == "ambiguous"
                    else "未在已阅读页面中命中该图号，未自动关联"
                )
                notes.append(
                    f"第 {page.page_number} 页 {node.number or '（无题号）'}："
                    f"{reference} {reason}（可在图号关联面板中人工确认）"
                )
                changed = True
        if changed:
            store.save_page_candidates(page.id, candidates)
    return notes
