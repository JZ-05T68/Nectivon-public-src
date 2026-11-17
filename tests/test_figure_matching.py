"""Figure-number matching hardening (v0.8.8 round-1 P1-1 / P2-1 / P2-3).

Covers: normalized exact-first matching, the 图6-3 vs 图6-33 substring
mis-match class, multi-hit ambiguity that must NOT silently bind,
structured per-link provenance (figure number, question number, link
basis, confidence, confirmation), uncaptioned-figure indexing, and the
manual-picker store path.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.cross_page_linkage import link_document_figures
from src.database import Database
from src.figure_index_service import (
    boundary_safe_contains,
    figure_key,
    find_figure_hits,
    find_figures_by_description,
    list_all_figures,
    prune_missing_pages,
    update_page_figures,
)
from src.question_candidate_service import QuestionCandidate, QuestionCandidateStore


# ------------------------------------------------------- key normalization
def test_figure_key_normalizes_fullwidth_digits_dashes_and_spacing() -> None:
    assert figure_key("图 6－33") == figure_key("图6-33") == "图6-33"
    assert figure_key("图 6—9") == "图6-9"
    assert figure_key("图6-8A") == "图6-8a"


def test_boundary_safe_contains_rejects_number_continuations() -> None:
    # The core P1-1 class: 图6-3 must not match 图6-33 / 图6-3a, in either
    # direction, but caption suffixes after the figure number are fine.
    assert not boundary_safe_contains("图6-33", "图6-3")
    assert not boundary_safe_contains("图6-3", "图6-33")
    assert not boundary_safe_contains("图6-3a", "图6-3")
    assert boundary_safe_contains("图6-3节流调速回路", "图6-3")
    assert boundary_safe_contains("图6-33", "图6-33回路") is False
    assert boundary_safe_contains("图6-33回路", "图6-33")


# ----------------------------------------------------------- index lookup
def test_find_figure_hits_prefers_exact_and_never_crosses_number_boundary(
    tmp_path: Path,
) -> None:
    root = tmp_path / "candidates"
    update_page_figures(
        root, document_id=1, page_id=11, page_number=5, document_title="第六章",
        figures=[{"name": "图6-33", "description": "调速回路", "bbox": [0, 0, 10, 10]}],
    )
    update_page_figures(
        root, document_id=1, page_id=12, page_number=6, document_title="第六章",
        figures=[{"name": "图6-3", "description": "换向回路", "bbox": [0, 0, 10, 10]}],
    )

    hits_33 = find_figure_hits(root, "图6-33")
    assert [h.entry["page_id"] for h in hits_33] == [11]
    assert all(h.match_basis == "exact" for h in hits_33)

    hits_3 = find_figure_hits(root, "图6-3")
    assert [h.entry["page_id"] for h in hits_3] == [12]
    assert all(h.match_basis == "exact" for h in hits_3)

    # A caption-suffixed registry name still matches by boundary-safe
    # substring and is labelled as such.
    update_page_figures(
        root, document_id=1, page_id=13, page_number=7, document_title="第六章",
        figures=[
            {"name": "图6-9 快速运动回路", "description": "", "bbox": [0, 0, 10, 10]}
        ],
    )
    hits_9 = find_figure_hits(root, "图6-9")
    assert [h.entry["page_id"] for h in hits_9] == [13]
    assert hits_9[0].match_basis == "substring"


def test_ambiguous_global_exact_hits_are_all_returned(tmp_path: Path) -> None:
    root = tmp_path / "candidates"
    update_page_figures(
        root, document_id=1, page_id=21, page_number=3, document_title="教材A",
        figures=[{"name": "图6-33", "description": "A版", "bbox": [0, 0, 10, 10]}],
    )
    update_page_figures(
        root, document_id=2, page_id=22, page_number=4, document_title="教材B",
        figures=[{"name": "图6-33", "description": "B版", "bbox": [0, 0, 10, 10]}],
    )
    hits = find_figure_hits(root, "图6-33")
    assert len(hits) == 2
    assert {h.entry["document_title"] for h in hits} == {"教材A", "教材B"}


def test_uncaptioned_figures_are_indexed_and_pickable(tmp_path: Path) -> None:
    root = tmp_path / "candidates"
    added = update_page_figures(
        root, document_id=3, page_id=31, page_number=1, document_title="独立照片",
        figures=[
            {"name": "", "description": "拍摄液压回路实物图，含溢流阀与换向阀",
             "bbox": [0, 0, 500, 400]},
            {"name": "", "description": "", "bbox": [0, 0, 10, 10]},  # nothing to key on
        ],
    )
    assert added == 1
    all_figures = list_all_figures(root)
    assert len(all_figures) == 1
    assert all_figures[0]["captioned"] is False
    assert "溢流阀" in all_figures[0]["description"]
    # Name matching cannot hit it (no caption — nothing invented), but the
    # description-similarity suggestion path can surface it for a human.
    assert find_figure_hits(root, "图6-33") == []
    suggestions = find_figures_by_description(root, "液压回路 溢流阀", limit=5)
    assert suggestions and suggestions[0][0]["page_id"] == 31


def test_prune_missing_pages_only_drops_dead_pages(tmp_path: Path) -> None:
    root = tmp_path / "candidates"
    update_page_figures(
        root, document_id=1, page_id=41, page_number=1, document_title="甲",
        figures=[{"name": "图1-1", "description": "", "bbox": [0, 0, 10, 10]}],
    )
    update_page_figures(
        root, document_id=2, page_id=42, page_number=1, document_title="乙",
        figures=[{"name": "图2-1", "description": "", "bbox": [0, 0, 10, 10]}],
    )
    removed = prune_missing_pages(root, {42})
    assert removed == 1
    remaining = list_all_figures(root)
    assert [entry["page_id"] for entry in remaining] == [42]
    assert prune_missing_pages(root, {42}) == 0  # idempotent


# ----------------------------------------------------- linkage integration
@pytest.fixture()
def textbook(tmp_path: Path):
    """One 3-page document; page 3 carries 图6-33 AND 图6-3 side by side."""

    database = Database(tmp_path / "data" / "database" / "knowledge.db")
    (tmp_path / "data" / "raw").mkdir(parents=True)
    raw = tmp_path / "data" / "raw" / "液压.jpg"
    raw.write_bytes(b"jpg")
    document = database.create_document(
        title="液压与气动",
        filename="液压.jpg",
        source_path=raw,
        sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        page_count=3,
        import_status="completed",
    )
    pages = [
        database.create_page(
            document_id=document.id, page_number=n, image_path=raw,
            extracted_text="", status="pending",
        )
        for n in (1, 2, 3)
    ]
    store = QuestionCandidateStore(tmp_path / "data" / "question-candidates")
    return database, document, pages, store


def _save_candidate(store, page, number: str, refs: list[str]) -> None:
    store.save_page_candidates(
        page.id,
        [
            QuestionCandidate(
                number=number, stem="如题图所示，试分析回路。",
                completeness="complete", figure_refs=refs,
                extracted_at="2026-11-04T00:00:00+00:00",
            )
        ],
    )


def test_question_6_2_links_exactly_to_figure_6_33_not_6_3(textbook) -> None:
    database, document, pages, store = textbook
    store.save_page_figures(
        pages[2].id,
        [
            {"name": "图6-33", "description": "调速回路", "bbox": [0, 0, 100, 100]},
            {"name": "图6-3", "description": "换向回路", "bbox": [0, 200, 100, 300]},
        ],
    )
    _save_candidate(store, pages[0], "6-2", ["图6-33"])
    _save_candidate(store, pages[1], "6-4", ["图6-9"])  # not registered anywhere

    notes = link_document_figures(store, database, document.id)

    q62 = store.page_candidates(pages[0].id)[0]
    assert len(q62.visual_regions) == 1
    region = q62.visual_regions[0]
    assert region["source_page_id"] == pages[2].id
    assert region["bbox"] == [0.0, 0.0, 100.0, 100.0]  # 图6-33's box, not 图6-3's
    # Structured provenance (round-1 P2-1 seven-field audit).
    assert region["figure_number"] == "图6-33"
    assert region["question_number"] == "6-2"
    assert region["link_basis"] == "exact_same_document"
    assert region["link_confidence"] == "high"
    assert region["link_confirmed"] is False
    assert any("图6-33" in note for note in notes)

    # 6-4's 图6-9 is NOT silently bound to 图6-3 or 图6-33 (the old
    # substring fallback would have picked the first global hit).
    q64 = store.page_candidates(pages[1].id)[0]
    assert q64.visual_regions == []
    assert q64.unresolved_figure_refs[0]["reference"] == "图6-9"
    assert q64.unresolved_figure_refs[0]["state"] == "unresolved"
    assert "图6-9" not in (q64.visual_notes or "")


def test_ambiguous_reference_is_surfaced_not_bound(textbook) -> None:
    database, document, pages, store = textbook
    # Same figure name registered on two sibling pages → two exact hits.
    store.save_page_figures(
        pages[1].id,
        [{"name": "图6-33", "description": "第2页版本", "bbox": [0, 0, 100, 100]}],
    )
    store.save_page_figures(
        pages[2].id,
        [{"name": "图6-33", "description": "第3页版本", "bbox": [0, 200, 100, 300]}],
    )
    _save_candidate(store, pages[0], "6-2", ["图6-33"])

    link_document_figures(store, database, document.id)

    node = store.page_candidates(pages[0].id)[0]
    assert node.visual_regions == []  # NEVER auto-picks the first hit
    record = node.unresolved_figure_refs[0]
    assert record["state"] == "ambiguous"
    assert len(record["candidates"]) == 2
    assert {c["page_id"] for c in record["candidates"]} == {pages[1].id, pages[2].id}
    assert all(c["match_basis"] == "exact" for c in record["candidates"])


def test_substring_caption_hit_binds_with_medium_confidence(textbook) -> None:
    database, document, pages, store = textbook
    store.save_page_figures(
        pages[2].id,
        [{"name": "图6-9 快速运动回路", "description": "", "bbox": [0, 0, 100, 100]}],
    )
    _save_candidate(store, pages[0], "6-4", ["图6-9"])

    link_document_figures(store, database, document.id)

    node = store.page_candidates(pages[0].id)[0]
    assert len(node.visual_regions) == 1
    region = node.visual_regions[0]
    assert region["figure_number"] == "图6-9"
    assert region["link_basis"] == "substring_same_document"
    assert region["link_confidence"] == "medium"
    assert region["link_confirmed"] is False


def test_relink_is_idempotent_and_respects_binding_confirmed(textbook) -> None:
    database, document, pages, store = textbook
    store.save_page_figures(
        pages[2].id,
        [{"name": "图6-33", "description": "调速回路", "bbox": [0, 0, 100, 100]}],
    )
    _save_candidate(store, pages[0], "6-2", ["图6-33"])
    link_document_figures(store, database, document.id)
    link_document_figures(store, database, document.id)
    node = store.page_candidates(pages[0].id)[0]
    assert len(node.visual_regions) == 1
    assert len(node.unresolved_figure_refs) == 0

    node.binding_confirmed = True
    node.visual_regions = []
    store.save_page_candidates(pages[0].id, [node])
    link_document_figures(store, database, document.id)
    frozen = store.page_candidates(pages[0].id)[0]
    assert frozen.visual_regions == []  # confirmed bindings are never relinked


def test_manual_picker_attaches_confirmed_link_and_clears_unresolved(
    textbook,
) -> None:
    from src.figure_index_service import virtual_region

    database, document, pages, store = textbook
    # An uncaptioned photo in the global index cannot be name-matched —
    # the wired picker is the genuine fallback (P2-3), no re-crop needed.
    update_page_figures(
        store.root, document_id=99, page_id=999, page_number=8,
        document_title="另一本教材",
        figures=[
            {"name": "", "description": "他书调速回路（无印刷图注）",
             "bbox": [10, 10, 90, 90]}
        ],
    )
    _save_candidate(store, pages[0], "6-2", ["图6-33"])
    link_document_figures(store, database, document.id)
    node = store.page_candidates(pages[0].id)[0]
    assert node.visual_regions == []
    assert node.unresolved_figure_refs[0]["state"] == "unresolved"

    entry = list_all_figures(store.root)[0]
    region = virtual_region(
        entry, description="图6-33", figure_number="图6-33", question_number="6-2",
    )
    store.attach_manual_figure_at_path(pages[0].id, "0", region)

    node = store.page_candidates(pages[0].id)[0]
    assert node.unresolved_figure_refs == []
    attached = node.visual_regions[-1]
    assert attached["link_basis"] == "manual_picker"
    assert attached["link_confidence"] == "high"
    assert attached["link_confirmed"] is True
    assert attached["figure_number"] == "图6-33"
    assert attached["question_number"] == "6-2"
    assert attached["source_page_id"] == 999


def test_per_link_confirmation_flips_only_that_link(textbook) -> None:
    database, document, pages, store = textbook
    store.save_page_figures(
        pages[2].id,
        [{"name": "图6-33", "description": "调速回路", "bbox": [0, 0, 100, 100]}],
    )
    _save_candidate(store, pages[0], "6-2", ["图6-33"])
    link_document_figures(store, database, document.id)
    node = store.page_candidates(pages[0].id)[0]
    assert node.visual_regions[0]["link_confirmed"] is False
    assert node.binding_confirmed is False

    store.confirm_figure_link_at_path(pages[0].id, "0", 0)
    node = store.page_candidates(pages[0].id)[0]
    assert node.visual_regions[0]["link_confirmed"] is True
    # Per-link confirmation never touches the question-level freeze flag.
    assert node.binding_confirmed is False


def test_provenance_survives_candidate_persistence_roundtrip(textbook) -> None:
    database, document, pages, store = textbook
    store.save_page_figures(
        pages[2].id,
        [{"name": "图6-33", "description": "调速回路", "bbox": [0, 0, 100, 100]}],
    )
    _save_candidate(store, pages[0], "6-2", ["图6-33"])
    link_document_figures(store, database, document.id)
    reloaded = store.page_candidates(pages[0].id)[0]
    region = reloaded.visual_regions[0]
    assert region["figure_number"] == "图6-33"
    assert region["link_basis"] == "exact_same_document"
    assert region["link_confidence"] == "high"
    assert region["link_confirmed"] is False
