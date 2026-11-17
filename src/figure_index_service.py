"""Global figure index across every imported photo (round-2 issue 2).

A question's 原理图 often lives in the textbook body, photographed as a
separate page in a separate import batch.  This module keeps one global,
deterministic index — {figure name → where it physically lives} — updated
whenever a page is read, and hands back cross-page "virtual regions" so any
question view can crop and display the figure from ITS page.  No adjacency
assumption, no per-subject rules; one figure may serve many questions.

Storage is application metadata (``figures-index.json`` beside the
candidate store), never user material.

v0.8.8 round-1 hardening (P1-1 / P2-1 / P2-3):

- :func:`figure_key` normalizes full-width digits and dash variants, so
  「图 6－33」and「图6-33」compare equal;
- matching is EXACT-FIRST and every hit carries its ``match_basis``
  (``exact`` or boundary-safe ``substring``) — the 图6-3 ↔ 图6-33
  substring mis-match class is rejected by a digit/letter boundary rule;
- multiple hits are ALL returned; callers must not silently pick the first;
- figures without a printed caption are indexed too, keyed by a stable
  synthetic key and surfaced with the vision model's description, so the
  manual picker (:func:`list_all_figures`) and
  :func:`find_figures_by_description` can still reach them;
- linked regions carry structured provenance (figure number, question
  number, link basis, confidence, per-link confirmation).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

INDEX_FILENAME = "figures-index.json"
_MAX_DESCRIPTION = 200

#: Structured linkage provenance vocabulary (round-1 P2-1). These values
#: are stored inside candidate JSON region dicts; the UI maps them to
#: Chinese labels — internal enums never reach the screen raw.
LINK_BASIS_EXACT_SAME_DOCUMENT = "exact_same_document"
LINK_BASIS_EXACT_GLOBAL_INDEX = "exact_global_index"
LINK_BASIS_SUBSTRING_SAME_DOCUMENT = "substring_same_document"
LINK_BASIS_SUBSTRING_GLOBAL_INDEX = "substring_global_index"
LINK_BASIS_MANUAL_PICKER = "manual_picker"
LINK_BASES = frozenset(
    {
        LINK_BASIS_EXACT_SAME_DOCUMENT,
        LINK_BASIS_EXACT_GLOBAL_INDEX,
        LINK_BASIS_SUBSTRING_SAME_DOCUMENT,
        LINK_BASIS_SUBSTRING_GLOBAL_INDEX,
        LINK_BASIS_MANUAL_PICKER,
    }
)
LINK_CONFIDENCES = ("high", "medium", "low")

_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
_DASH_VARIANTS = re.compile(r"[－—–‐‑−]")
_KEY_NOISE = re.compile(r"[\s，。、．.·()（）:：]")
_TRAILING_FIGURE_SUFFIX = re.compile(r"[0-9a-z]")


def figure_key(value: str) -> str:
    """Normalize one figure name/reference for deterministic comparison.

    Full-width digits fold to ASCII, every dash variant folds to ``-``,
    spacing and Chinese punctuation are dropped, and the result is
    lower-cased.  The figure NUMBER itself (digits, dashes, letter
    suffixes such as 图6-8a) is never folded away.
    """

    text = str(value or "").translate(_FULLWIDTH_DIGITS)
    text = _DASH_VARIANTS.sub("-", text)
    text = _KEY_NOISE.sub("", text)
    return text.lower()


# Backwards-compatible private alias used by earlier internal callers.
_figure_key = figure_key


def boundary_safe_contains(longer: str, shorter: str) -> bool:
    """Substring test that refuses figure-number continuation matches.

    ``图6-3`` must NOT match ``图6-33`` or ``图6-3a``: an occurrence only
    counts when the character right after it is neither a digit nor a
    letter (caption words like 「回路」are fine, number continuations are
    not).  This is the deterministic guard against the round-1 P1-1
    mis-binding class in figure-dense textbooks.
    """

    if not shorter or not longer or shorter == longer:
        return False
    start = 0
    while True:
        position = longer.find(shorter, start)
        if position < 0:
            return False
        end = position + len(shorter)
        if end == len(longer) or not _TRAILING_FIGURE_SUFFIX.match(longer[end]):
            return True
        start = position + 1


@dataclass(frozen=True, slots=True)
class FigureHit:
    """One indexed figure entry plus HOW it matched a reference."""

    entry: dict
    match_basis: str  # "exact" | "substring"
    matched_key: str  # the normalized index key that matched


def index_path(store_root: Path | str) -> Path:
    return Path(store_root) / INDEX_FILENAME


def read_index(store_root: Path | str) -> dict[str, list[dict]]:
    path = index_path(store_root)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("全局图形索引无法读取，按空索引处理：%s", path)
        return {}
    figures = payload.get("figures") if isinstance(payload, dict) else None
    if not isinstance(figures, dict):
        return {}
    cleaned: dict[str, list[dict]] = {}
    for key, entries in figures.items():
        if isinstance(key, str) and isinstance(entries, list):
            cleaned[key] = [e for e in entries if isinstance(e, dict)]
    return cleaned


def _write_index(store_root: Path | str, figures: dict[str, list[dict]]) -> None:
    path = index_path(store_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"figures": figures}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    temporary.replace(path)


def update_page_figures(
    store_root: Path | str,
    *,
    document_id: int,
    page_id: int,
    page_number: int,
    document_title: str,
    figures: list[dict],
) -> int:
    """Refresh one page's entries in the global index; returns entry count.

    Figures WITH a printed caption are keyed by their normalized name.
    Figures WITHOUT one (a standalone photographed circuit diagram, round-1
    P2-3) are no longer dropped: they are keyed by a stable synthetic key
    and carry the vision model's description, so the manual picker and the
    description-similarity suggestion path can still reach them. Entries
    with neither name nor description are skipped — there is nothing
    honest to key on, and nothing is ever invented.
    """

    index = read_index(store_root)
    for key, entries in list(index.items()):
        index[key] = [e for e in entries if e.get("page_id") != page_id]
        if not index[key]:
            del index[key]
    added = 0
    for order, figure in enumerate(figures or []):
        name = str(figure.get("name", "")).strip()
        description = str(figure.get("description", "")).strip()
        bbox = figure.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        captioned = bool(name)
        if captioned:
            key = figure_key(name)
        elif description:
            key = f"无图注-{page_id}-{order}"
        else:
            continue
        if not key:
            continue
        index.setdefault(key, []).append(
            {
                "name": name[:80],
                "description": description[:_MAX_DESCRIPTION],
                "captioned": captioned,
                "document_id": document_id,
                "page_id": page_id,
                "page_number": page_number,
                "document_title": str(document_title)[:120],
                "bbox": [float(v) for v in bbox],
            }
        )
        added += 1
    _write_index(store_root, index)
    return added


def find_figure_hits(store_root: Path | str, reference: str) -> list[FigureHit]:
    """All entries matching a figure reference, exact matches first.

    Returns every hit with its ``match_basis`` so the caller can refuse to
    auto-bind on multiple hits. When any exact (normalized-key) match
    exists, substring hits are NOT returned: exactness wins. Stored keys
    are re-normalized at lookup time so indexes written by older key
    normalizations still match.
    """

    key = figure_key(reference)
    if not key:
        return []
    index = read_index(store_root)
    exact: list[FigureHit] = []
    substring: list[FigureHit] = []
    for name_key, entries in index.items():
        normalized = figure_key(name_key)
        if not normalized:
            continue
        if normalized == key:
            basis = "exact"
        elif boundary_safe_contains(normalized, key) or boundary_safe_contains(
            key, normalized
        ):
            basis = "substring"
        else:
            continue
        for entry in entries:
            hit = FigureHit(entry=entry, match_basis=basis, matched_key=normalized)
            (exact if basis == "exact" else substring).append(hit)
    hits = exact or substring
    return sorted(
        hits,
        key=lambda hit: (
            str(hit.entry.get("document_title", "")),
            int(hit.entry.get("page_number") or 0),
        ),
    )


def find_figures(store_root: Path | str, reference: str) -> list[dict]:
    """Entries whose indexed name matches a figure reference like 图 6-8a.

    Compatibility facade over :func:`find_figure_hits`; callers that must
    not silently pick among several hits should use the hits API instead.
    """

    return [hit.entry for hit in find_figure_hits(store_root, reference)]


def list_all_figures(store_root: Path | str) -> list[dict]:
    """Flat list for the manual picker, ordered by document/page.

    Wired into the learning-entry UI figure-linkage panel (round-1 P2-3):
    the genuine fallback for references the deterministic matcher could
    not resolve, including uncaptioned figures that only carry the vision
    model's description.
    """

    flat = [
        {**entry, "key": key}
        for key, entries in read_index(store_root).items()
        for entry in entries
    ]
    return sorted(
        flat, key=lambda e: (str(e.get("document_title")), e.get("page_number", 0))
    )


def find_figures_by_description(
    store_root: Path | str, text: str, *, limit: int = 5
) -> list[tuple[dict, float]]:
    """Suggest indexed figures whose description overlaps ``text``.

    Manual-confirmation aid only — never an automatic binding. Scoring is
    a deterministic jieba token Dice coefficient (the packaged product
    runtime deliberately stays free of rapidfuzz); only suggestions with a
    real overlap are returned, best first.
    """

    import jieba

    def tokens(value: str) -> set[str]:
        return {
            token.strip().lower()
            for token in jieba.cut_for_search(str(value or ""))
            if len(token.strip()) > 1
        }

    probe = tokens(text)
    if not probe:
        return []
    scored: list[tuple[dict, float]] = []
    for key, entries in read_index(store_root).items():
        for entry in entries:
            entry_tokens = tokens(
                f"{entry.get('name', '')} {entry.get('description', '')}"
            )
            shared = probe & entry_tokens
            if not shared or not entry_tokens:
                continue
            dice = 2.0 * len(shared) / (len(probe) + len(entry_tokens))
            if dice >= 0.2:
                scored.append(({**entry, "key": key}, round(dice, 4)))
    scored.sort(key=lambda item: (-item[1], str(item[0].get("document_title"))))
    return scored[: max(1, int(limit))]


def virtual_region(
    entry: dict,
    *,
    description: str = "",
    figure_number: str = "",
    question_number: str = "",
    link_basis: str = "",
    link_confidence: str = "",
) -> dict:
    """Build a cross-page region dict that crops from the figure's own page.

    The optional provenance arguments give every automatic link its
    structured audit trail (round-1 P2-1): figure number, question number,
    how the match was made and with which confidence. ``link_confirmed``
    always starts ``False`` — only an explicit user action confirms it.
    """

    label = description or str(entry.get("name", ""))
    source = str(entry.get("document_title", ""))
    page_number = entry.get("page_number", "?")
    region = {
        "role": "shared",
        "bbox": [float(v) for v in entry.get("bbox", [0, 0, 1000, 1000])],
        "description": f"{label}（来自《{source}》第 {page_number} 页照片）",
        "source_page_id": int(entry.get("page_id", 0)),
        "source_document_id": int(entry.get("document_id", 0)),
        "source_label": f"《{source}》第 {page_number} 页",
        "link_confirmed": False,
    }
    if figure_number:
        region["figure_number"] = str(figure_number)[:40]
    if question_number:
        region["question_number"] = str(question_number)[:40]
    if link_basis in LINK_BASES:
        region["link_basis"] = link_basis
    if link_confidence in LINK_CONFIDENCES:
        region["link_confidence"] = link_confidence
    return region


def prune_missing_pages(store_root: Path | str, valid_page_ids: set[int]) -> int:
    """Drop index entries whose page no longer exists; return removed count.

    Deletion residue cleanup (v0.8.8 round-1): after documents are deleted
    the global figure index can keep entries pointing at removed pages, so
    linkage hits would reference files that no longer exist. Only entries
    with a page id absent from ``valid_page_ids`` are removed; the rewrite
    goes through the same atomic writer as every other index update.
    """

    if not index_path(store_root).is_file():
        return 0
    index = read_index(store_root)
    pruned: dict[str, list[dict]] = {}
    removed = 0
    for key, entries in index.items():
        kept = [entry for entry in entries if _entry_page_id(entry) in valid_page_ids]
        removed += len(entries) - len(kept)
        if kept:
            pruned[key] = kept
    if removed:
        _write_index(store_root, pruned)
        LOGGER.info("全局图形索引已清理失效页面条目：%s 条", removed)
    return removed


def _entry_page_id(entry: dict) -> int:
    try:
        return int(entry.get("page_id", 0))
    except (TypeError, ValueError):
        return 0


def refresh_automatic_figure_clips(
    store_root: Path, regions: list[dict], *, page_id: int,
) -> list[dict]:
    """Recompute old automatic hard stops from the actual source page registry.

    This is a read-only compatibility repair for already imported figures;
    saved human pixel boxes are never expanded. No recognition is repeated.
    """

    from types import SimpleNamespace

    from src.cross_page_linkage import clip_regions_to_siblings

    result = []
    for supplied in regions:
        region = dict(supplied)
        if region.get("manual_bounds") or region.get("region_id"):
            result.append(region)
            continue
        physical_page = region.get("source_page_id", region.get("page_id", page_id))
        path = store_root / f"figures_page_{physical_page}.json"
        if region.get("role") in ("stem", "shared") and path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("page_id") != physical_page:
                    raise ValueError("Figure registry source mismatch")
                figures = payload["figures"]
                if not isinstance(figures, list):
                    raise ValueError("Invalid figure registry")
                region.pop("sibling_clip", None)
                node = SimpleNamespace(visual_regions=[region], figure_refs=[], children=[])
                clip_regions_to_siblings([node], figures)
            except (OSError, ValueError, KeyError, TypeError):
                LOGGER.exception("无法核对图形间距，保留已记录边界：page=%s", physical_page)
                region = dict(supplied)
        result.append(region)
    return result
