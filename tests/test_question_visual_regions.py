"""Original-pixel option crops survive first-pass recognition and learning."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import src.question_content_ui as content_ui
import src.runtime as runtime
from src.ai.credential_store import MemoryCredentialStore, SecretCredential
from src.ai.model_registry import ProviderId
from src.ai.provider_config import AIProviderConfig, ProviderConfigStore, ProviderSettings
from src.ai.provider_factory import resolve_qwen_runtime
from src.config import Settings
from src.database import Database
from src.question_candidate_service import (
    extract_candidates,
    parse_candidates_payload,
)
from src.question_visual_regions import bind_regions, crop_region, normalize_regions


def test_overlapping_figures_share_one_original_crop_and_options_stay_separate() -> None:
    regions = [
        {"role": "stem", "bbox": [100, 200, 400, 300]},
        {"role": "stem", "bbox": [250, 100, 500, 300]},
        {"role": "option", "option_label": "A", "bbox": [100, 400, 400, 500]},
        {"role": "option", "option_label": "B", "bbox": [250, 400, 500, 500]},
    ]
    grouped = content_ui._coalesce_overlapping_figures(regions)
    assert len(grouped) == 3
    assert grouped[0]["bbox"] == [100, 100, 500, 300]
    assert [region["option_label"] for region in grouped[1:]] == ["A", "B"]
    assert regions[0]["bbox"] == [100, 200, 400, 300]


def _seed_verified_replica(image_path: Path, region: dict) -> None:
    """Write a verified replica record for exactly the pixels a view will crop.

    Round 4 replaced the scan-cleanup default with replica display; UI tests
    seed a minimal verified replica so the real display path is exercised
    without any cloud call.
    """

    from src.figure_replica_render import render_replica
    from src.figure_replica_schema import parse_replica_spec
    from src.figure_replica_service import replica_key, replica_store_root

    data = crop_region(image_path, region)
    assert data is not None
    spec = parse_replica_spec(
        {
            "schema": "nectivon-replica/1",
            "figure_type": "node_link",
            "canvas": {"width": 480, "height": 360},
            "elements": [
                {
                    "id": "n1",
                    "kind": "box",
                    "x": 240,
                    "y": 180,
                    "w": 260,
                    "h": 160,
                    "text": "复刻测试图",
                }
            ],
            "connections": [],
            "annotations": [],
        }
    )
    svg, png = render_replica(spec, px_width=480)
    root = replica_store_root(image_path) / replica_key(data)
    root.mkdir(parents=True, exist_ok=True)
    (root / "spec.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    (root / "replica.svg").write_bytes(svg)
    (root / "replica.png").write_bytes(png)
    (root / "status.json").write_text(
        json.dumps(
            {
                "status": "verified",
                "figure_type": "node_link",
                "verification": {
                    "verdict": "pass",
                    "missing": [],
                    "extra": [],
                    "wrong": [],
                    "text_errors": [],
                    "style_notes": [],
                },
                "render_version": "replica-render-1",
                "model": "test-seed",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_visual_grouping_never_merges_regions_from_different_original_pages() -> None:
    regions = [
        {"role": "shared", "bbox": [100, 200, 400, 300], "page_id": page_id} for page_id in (1, 2)
    ]
    assert len(content_ui._coalesce_overlapping_figures(regions)) == 2


def test_duplicate_option_boxes_and_neighbor_crop_boundaries() -> None:
    boxes = [
        {"role": "option", "option_label": "A", "bbox": [100, 100, 400, 200]},
        {"role": "option", "option_label": "B", "bbox": [600, 100, 900, 200]},
        {"role": "option", "option_label": "C", "bbox": [100, 230, 400, 330]},
        {"role": "option", "option_label": "D", "bbox": [600, 230, 900, 330]},
    ]
    regions = normalize_regions([*boxes, {**boxes[0], "description": "another description"}])
    assert len(regions) == 4
    assert regions[0]["clip_bbox"] == [0, 0, 500, 215]
    assert regions[3]["clip_bbox"] == [500, 215, 1000, 1000]


def _payload() -> dict:
    return {
        "candidates": [
            {
                "number": "3",
                "stem": r"已知 $a+b+c=0$，位置不可能是（ ）",
                "completeness": "complete",
                "visual_dependency": "required",
                "is_multiple_choice": True,
                "options": [
                    {
                        "label": label,
                        "text": "",
                        "requires_image": True,
                        "image_bbox": [100 + i * 200, 100, 250 + i * 200, 250],
                    }
                    for i, label in enumerate("ABCD")
                ],
                "visual_regions": [{"role": "stem", "bbox": [95, 95, 860, 260]}],
                "question_bbox": [80, 50, 900, 300],
            }
        ]
    }


@pytest.fixture()
def source_image(tmp_path: Path) -> Path:
    path = tmp_path / "original.png"
    image = Image.new("RGB", (1000, 1000), "white")
    for i, color in enumerate(["red", "green", "blue", "black"]):
        image.paste(color, (100 + i * 200, 100, 250 + i * 200, 250))
    image.save(path)
    return path


def test_graph_choices_are_complete_without_invented_text() -> None:
    candidate = parse_candidates_payload(json.dumps(_payload()))[0]
    assert candidate.completeness == "complete"
    assert candidate.stem.endswith("D. （见原图）")
    assert [r["option_label"] for r in candidate.visual_regions] == list("ABCD")
    assert all(r["role"] == "option" for r in candidate.visual_regions)


def test_missing_graph_box_preserves_original_question_fallback() -> None:
    payload = _payload()
    payload["candidates"][0]["visual_regions"] = []
    payload["candidates"][0]["options"][3]["image_bbox"] = [999, 300, 200, 300]
    candidate = parse_candidates_payload(json.dumps(payload))[0]
    assert candidate.completeness == "incomplete"
    assert "D" in candidate.incomplete_reason
    assert candidate.visual_regions[-1]["role"] == "question"
    assert candidate.visual_regions[-1]["bbox"] == [80, 50, 900, 300]


@pytest.mark.parametrize(
    "bbox",
    [
        [float("nan"), 0, 20, 20],
        [0, 0, float("inf"), 20],
        [-1, 0, 50, 50],
        [1, 1, 1001, 10],
        [80, 0, 10, 20],
        [0, 0, 2, 50],
        [True, 0, 50, 50],
        "bad",
        None,
    ],
)
def test_unusable_regions_cannot_crop(bbox) -> None:
    assert normalize_regions([{"role": "stem", "bbox": bbox}]) == []


def test_single_call_binds_only_actual_image_and_page(source_image: Path) -> None:
    class Provider:
        provider_id = "qwen"
        default_model = "qwen3.8-max"
        calls = 0

        def complete_vision(self, prompt, image, **kwargs):
            self.calls += 1
            assert "0～1000" in prompt and "四个不同选项" in prompt
            assert "OCR噪声" not in prompt
            payload = _payload()
            payload["candidates"][0]["visual_regions"][0].update(
                page_id=999,
                image_sha256="a" * 64,
                path="arbitrary-file.png",
            )
            return SimpleNamespace(text=json.dumps(payload))

    provider = Provider()
    candidates = extract_candidates(
        provider,
        page_text="OCR噪声",
        diagram_text="",
        page_id=12,
        image_path=source_image,
    )
    assert provider.calls == 1
    assert all(r["page_id"] == 12 for r in candidates[0].visual_regions)
    assert all("path" not in r for r in candidates[0].visual_regions)
    assert crop_region(source_image, candidates[0].visual_regions[0])


def test_crop_uses_original_pixels_and_rejects_changed_source(source_image: Path) -> None:
    original = source_image.read_bytes()
    region = bind_regions(
        [{"role": "stem", "bbox": [100, 100, 250, 250]}], page_id=1, image_bytes=original
    )[0]
    data = crop_region(source_image, region)
    with Image.open(io.BytesIO(data)) as crop:
        from src.question_visual_regions import region_pixel_bounds

        with Image.open(source_image) as source:
            expected = source.crop(region_pixel_bounds(source_image, region)).convert("RGB")
        assert crop.tobytes() == expected.tobytes()
    assert source_image.read_bytes() == original
    source_image.write_bytes(original + b"changed")
    assert crop_region(source_image, region) is None


def test_current_compound_roundtrip_preserves_shared_source(tmp_path: Path) -> None:
    from src.study_loop.models import Draft
    from src.study_loop.storage import StudyStore
    from tests.study_migration_support import compound_image_draft, image_case

    store, source, page, _, raw = image_case(tmp_path / "study")
    draft = compound_image_draft(store, source.id, page)
    restored = Draft.from_dict(draft.to_dict())
    assert restored.to_dict() == draft.to_dict()
    assert restored.page_ids == [page.id]
    assert [q.content for q in restored.question.units()] == [
        "NON_REAL_TEST_DATA OPTION_A",
        "NON_REAL_TEST_DATA OPTION_B",
    ]
    assert all(
        q.answer_status == "unconfirmed" and not q.reference_answer
        for q in restored.question.units()
    )
    store.confirm(restored, user_confirmed=True)
    reopened = StudyStore(store.root)
    assert reopened.item(draft.id).to_dict() == restored.to_dict()
    assert reopened.read_source(source) == raw
    assert reopened.pages(source.id) == [page]


def test_tinted_paper_does_not_expand_figure_crop_to_the_page_frame(tmp_path: Path) -> None:
    from PIL import ImageDraw

    from src.question_visual_regions import region_pixel_bounds

    source = tmp_path / "tinted.png"
    photo = Image.new("RGB", (1000, 1000), (80, 130, 170))
    pen = ImageDraw.Draw(photo)
    pen.rectangle((8, 8, 992, 992), outline="black", width=2)
    pen.rectangle((300, 250, 600, 650), outline="black", width=3)
    pen.line((300, 450, 600, 450), fill="black", width=2)
    photo.save(source)
    before = source.read_bytes()
    bounds = region_pixel_bounds(source, {"role": "stem", "bbox": [280, 230, 620, 700]})
    assert bounds is not None
    left, top, right, bottom = bounds
    assert 150 < left <= 300 < 600 <= right < 850
    assert top <= 250 < 650 <= bottom
    assert source.read_bytes() == before


def test_figures_follow_their_option_after_manual_text_edit(
    source_image: Path,
    monkeypatch,
) -> None:
    candidate = parse_candidates_payload(json.dumps(_payload()))[0]
    for region in candidate.visual_regions:
        _seed_verified_replica(source_image, region)
    calls: list = []
    monkeypatch.setattr(
        content_ui,
        "render_question_math_markdown",
        lambda text: calls.append(text),
    )
    monkeypatch.setattr(content_ui.st, "image", lambda data, **kwargs: calls.append(data))
    edited = "人工修订 $x^{2}$。\n\n" + "\n\n".join(f"{x}. 选项文字已校正" for x in "ABCD")
    assert content_ui.render_question_content(
        edited,
        image_path=source_image,
        regions=candidate.visual_regions,
    )
    # Round 4.1: each option renders its replica AND cleanup views (both
    # collapsed expanders still render): 1 stem + 4 x (option text + 2 images).
    assert len(calls) == 13
    assert calls[0] == "人工修订 $x^{2}$。"
    for i in range(4):
        assert calls[1 + i * 3].startswith("ABCD"[i] + ".")
        assert isinstance(calls[2 + i * 3], bytes)  # replica render
        assert isinstance(calls[3 + i * 3], bytes)  # binary cleanup
    calls.clear()
    assert content_ui.render_question_content(
        "人工只修改了题干。",
        image_path=source_image,
        regions=candidate.visual_regions,
    )
    assert len([c for c in calls if isinstance(c, bytes)]) == 8
    assert "D. （见原图）" in calls


def _database(tmp_path: Path, image: Path) -> tuple[Database, object]:
    db = Database(tmp_path / "knowledge.db")
    raw = tmp_path / "original.pdf"
    raw.write_bytes(b"%PDF fixture")
    doc = db.create_document(
        title="数轴选择测试卷",
        filename="original.pdf",
        source_path=raw,
        sha256="b" * 64,
        import_status="completed",
    )
    page = db.create_page(document_id=doc.id, page_number=1, image_path=image)
    return db, page


def test_new_extraction_preserves_human_revision_source_scope_and_attempts(tmp_path: Path) -> None:
    from src.study_loop.extraction import save_extraction
    from src.study_loop.models import StudyError
    from tests.study_migration_support import pdf_case, submit_attempt

    store, source, pages, draft, raw = pdf_case(tmp_path / "study")
    draft.question.content = "NON_REAL_TEST_DATA HUMAN_CORRECTED"
    draft.question.reference_analysis = "NON_REAL_TEST_DATA HUMAN_NOTE"
    draft.question.answer_status = "user_note"
    draft.page_ids = [pages[1].id]
    store.confirm(draft, user_confirmed=True)
    session = store.start_review([draft.id], user_started=True)
    submit_attempt(store, session, draft)
    saved, history = store.item(draft.id).to_dict(), store.attempt_records()
    response = {
        "text": json.dumps(
            {
                "questions": [
                    {
                        "content": "NON_REAL_TEST_DATA NEW_MODEL_DRAFT",
                        "observed_answer": "NON_REAL_TEST_DATA UNCONFIRMED",
                        "reference_answer": "MUST_NOT_PROMOTE",
                        "answer_status": "source_confirmed",
                    }
                ]
            }
        ),
        "model": "OFFLINE_TEST_RESPONSE",
    }
    save_extraction(store, source.id, pages[0].id, response, "TEST_NEW_EXTRACTION")
    assert store.item(draft.id).to_dict() == saved
    assert store.attempt_records() == history
    new = next(d for d in store.drafts() if d.extraction == "model_draft")
    assert new.page_ids == [pages[0].id]
    assert new.question.observed_answer == "NON_REAL_TEST_DATA UNCONFIRMED"
    assert new.question.reference_answer == ""
    assert new.question.answer_status == "unconfirmed"
    assert len(store.items()) == 1
    draft.question.content = "UNAUTHORIZED_OVERWRITE"
    with pytest.raises(StudyError, match="不会被覆盖"):
        store.confirm(draft, user_confirmed=True)
    assert store.item(draft.id).to_dict() == saved
    assert store.read_source(source) == raw


def test_current_draft_and_review_render_identical_verified_image(
    tmp_path: Path, monkeypatch
) -> None:
    from tests.study_migration_support import (
        capture_images,
        compound_image_draft,
        image_case,
        panel,
    )

    store, source, page, _, raw = image_case(tmp_path / "study")
    draft = compound_image_draft(store, source.id, page)
    images = capture_images(monkeypatch)
    app = panel(store, "drafts")
    assert not app.exception
    assert images == []
    app = panel(store, "drafts", advanced=True)
    assert not app.exception
    assert images == [store.path(page.image).read_bytes()]
    store.confirm(draft, user_confirmed=True)
    store.start_review([draft.id], user_started=True)
    images.clear()
    app = panel(store, "review", advanced=True)
    assert not app.exception
    assert images == [store.path(page.image).read_bytes()]
    assert len(app.get("image")) == 1
    assert "NON_REAL_TEST_DATA OPTION_A" in " ".join(m.value for m in app.markdown)
    assert not store.attempts()
    assert store.item(draft.id).question.units()[0].answer_status == "unconfirmed"
    assert store.read_source(source) == raw


def test_inactive_qwen_is_used_only_for_explicit_vision_route(tmp_path: Path) -> None:
    store = ProviderConfigStore(tmp_path / "providers.json")
    config = AIProviderConfig().with_provider(ProviderSettings.default_for(ProviderId.QWEN))
    config = config.with_provider(
        ProviderSettings.default_for(ProviderId.DEEPSEEK),
        make_active=True,
    )
    store.save(config)
    credentials = MemoryCredentialStore()
    credentials.set(ProviderId.QWEN, SecretCredential("synthetic-vision-credential"))
    settings = Settings(_env_file=None)
    assert resolve_qwen_runtime(settings, config_store=store, credential_store=credentials) is None
    selected = resolve_qwen_runtime(
        settings,
        config_store=store,
        credential_store=credentials,
        allow_inactive=True,
    )
    assert selected.default_model == "qwen3.8-max"
    assert store.load().active_provider_id is ProviderId.DEEPSEEK
    store.save(AIProviderConfig(providers=config.providers))
    assert (
        resolve_qwen_runtime(
            settings,
            config_store=store,
            credential_store=credentials,
            allow_inactive=True,
        )
        is None
    )


def test_question_route_uses_explicit_audited_vision_model(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        ai_mode="api",
        ai_api_key="synthetic-test-key",
        ai_vision_model="qwen3-vl-plus",
    )
    resolved = resolve_qwen_runtime(settings, config_store=ProviderConfigStore(tmp_path / "absent"))
    monkeypatch.setattr(
        runtime,
        "application_ai_provider",
        lambda: SimpleNamespace(
            provider_id="deepseek",
            default_model="deepseek-v4-pro",
        ),
    )
    monkeypatch.setattr(runtime, "application_settings", lambda: settings)
    monkeypatch.setattr(runtime, "application_credential_store", lambda: MemoryCredentialStore())
    resolver_calls = []

    def resolve(*args, **kwargs):
        resolver_calls.append(kwargs)
        return resolved

    monkeypatch.setattr(runtime, "resolve_image_provider_runtime", resolve)
    selected = runtime.application_question_vision_provider()
    assert selected.provider_id == "qwen"
    assert selected.default_model == "qwen3.8-max"
    assert "credential_store" in resolver_calls[0]
    monkeypatch.setattr(runtime, "application_ai_provider", lambda: None)
    assert runtime.application_question_vision_provider() is not None
    assert len(resolver_calls) == 2
    monkeypatch.setattr(runtime, "resolve_image_provider_runtime", lambda *args, **kwargs: None)
    assert runtime.application_question_vision_provider() is None


def test_source_view_rejects_wrong_missing_and_changed_pages_before_display(tmp_path: Path) -> None:
    from src.study_loop.components import import_source
    from tests.study_migration_support import pdf_case, source_panel

    store, source, pages, _, raw = pdf_case(tmp_path / "study")
    other = store.receive("NON_REAL_TEST_DATA.txt", b"NON_REAL_TEST_DATA OTHER", "txt")
    import_source(store, other.id)
    for source_id, refs in [
        (other.id, [pages[0].id]),
        (source.id, ["MISSING"]),
        (source.id, [pages[0].id, pages[0].id]),
    ]:
        app = source_panel(store, source_id, refs)
        assert not app.exception
        assert app.warning
        assert not app.get("image")
        assert not app.get("download_button")
    # A late corrupt page must not leak earlier images or an original download.
    store.path(pages[1].image).write_bytes(b"NON_REAL_TEST_DATA TAMPERED")
    app = source_panel(store, source.id, [p.id for p in pages])
    assert not app.exception
    assert app.warning
    assert not app.get("image")
    assert not app.get("download_button")
    assert store.read_source(source) == raw


def test_review_siblings_share_one_source_image_without_duplicate_display(
    tmp_path: Path, monkeypatch
) -> None:
    from tests.study_migration_support import (
        capture_images,
        compound_image_draft,
        image_case,
        panel,
    )

    store, source, page, _, raw = image_case(tmp_path / "study")
    draft = compound_image_draft(store, source.id, page)
    store.confirm(draft, user_confirmed=True)
    session = store.start_review([draft.id], user_started=True)
    images = capture_images(monkeypatch)
    app = panel(store, "review", advanced=True)
    assert not app.exception
    assert images == [store.path(page.image).read_bytes()]
    assert len(app.get("image")) == 1
    images.clear()
    app.selectbox(key=f"sl_unit_{session}_{draft.id}").select(draft.question.units()[1]).run()
    assert not app.exception
    assert images == [store.path(page.image).read_bytes()]
    assert len(app.get("image")) == 1
    assert "NON_REAL_TEST_DATA OPTION_B" in " ".join(m.value for m in app.markdown)
    assert not store.attempts()
    assert store.read_source(source) == raw
