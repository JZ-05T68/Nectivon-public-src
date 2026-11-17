"""Readable source projections preserve historical evidence and ordinary authored content."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from test_ai_review_support import RecordingProvider, review_case

from src.study_loop.ai_review import AIReviewService
from src.study_loop.chat_citations import visible_answer
from src.study_loop.conversation_titles import ConversationTitles
from src.study_loop.knowledge_memory import KnowledgeMemoryService
from src.study_loop.source_presentation import (
    UNAVAILABLE,
    evidence_name,
    human_name,
    memory_source_details,
)


def test_real_parenthesized_titles_are_preserved_but_empty_identity_wrappers_disappear():
    title = "三相异步电动机额定计算（六极 28 kW 题）"
    assert human_name(title) == title
    assert human_name("（M0123456789ab）") == ""


@pytest.fixture
def case(tmp_path):
    source = review_case(tmp_path / "NON_REAL_READABLE_SOURCES")
    service = AIReviewService(source.store)
    document = service.generate([source.selected.id], RecordingProvider("NON_REAL_AI"))
    for number in range(1, 5):
        service.update_working(document["id"], f"NON_REAL_REVIEW_SAVED_{number}")
        version = service.save_version(document["id"])
    memories = KnowledgeMemoryService(source.store)
    extraction = memories.extract_review(document["id"], RecordingProvider("额定功率计算"))
    identity = memories.confirm(extraction["id"], "额定功率计算", confirmed=True)
    memory = source.chats.update_memory(identity, "三相异步电动机额定计算", expected_revision=1)
    return SimpleNamespace(**vars(source), service=service, document=document, version=version,
                           memory=memory)


def test_memory_revision_and_saved_review_version_are_independent_read_only_names(case):
    before = case.chats.db.read_bytes()
    source_before = copy.deepcopy(case.memory["sources"])
    detail = memory_source_details(case.store, case.memory)[0]
    assert detail["review_version"] == 4 and detail["memory_revision"] == 2
    assert detail["filename"] == case.source.name
    assert detail["available"] and detail["type_name"] == "单题复盘记忆"
    review = {"kind": "saved_review", "label": "R0123456789ab",
              "name": case.source.name, "version_number": 4}
    memory = {"label": "M0123456789ab", "memory_revision": 2, "text": case.memory["content"]}
    assert evidence_name(review) == case.source.name + " · 复盘第4版"
    assert evidence_name(memory) == "知识记忆：三相异步电动机额定计算 · 第2修订"
    assert case.memory["sources"] == source_before and case.chats.db.read_bytes() == before


def test_source_dialogue_uses_manual_chinese_title_and_deleted_origin_is_unavailable(case):
    identity = case.chats.open_course("NON_REAL_COURSE_A", new=True)
    title = ConversationTitles(case.store)
    title.rename(identity, "电拖常考公式整理")
    memory = {"revision": 2, "sources": [{"type": "course_dialogue_memory",
              "subject": "NON_REAL_COURSE_A", "conversation_id": identity}]}
    detail = memory_source_details(case.store, memory)[0]
    assert detail["dialogue_title"] == "电拖常考公式整理" and detail["available"]
    assert detail["review_version"] is None and detail["memory_revision"] == 2
    case.chats.delete("conversation", identity, confirmed=True)
    detail = memory_source_details(case.store, memory)[0]
    assert not detail["available"] and detail["dialogue_title"] == UNAVAILABLE
    assert "回收站" in detail["unavailable_reason"]
    assert evidence_name({"name": "0123456789abcdef0123456789abcdef"}) == UNAVAILABLE
    assert evidence_name({}) == UNAVAILABLE


def test_unresolvable_review_version_never_invents_a_number_or_an_active_source(case):
    memory = copy.deepcopy(case.memory)
    memory["sources"][0]["version_id"] = "NON_REAL_MISSING_VERSION"
    detail = memory_source_details(case.store, memory)[0]
    assert detail["review_version"] is None and not detail["available"]
    assert detail["unavailable_reason"] == UNAVAILABLE and detail["memory_revision"] == 2


def test_actual_label_inline_code_shape_is_projected_without_altering_the_stored_raw(case):
    evidence = [{"label": "M0123456789ab", "memory_revision": 2, "text": "额定计算"},
                {"label": "Rabcdef123456", "kind": "saved_review", "name": "作业.docx",
                 "version_number": 4}]
    raw = (
        "- 已确认记忆：label `M0123456789ab`，memory_revision 2。\n"
        "- 已保存复盘：label `Rabcdef123456`，version_number 4，类型 saved_review。\n"
        "- 对应原始对话快照：`50e90ab8aaf0f34bead410809e1afc6c6a2d0bc3`\n"
        "已确认知识记忆 [M0123456789ab]（原始对话快照 `50e90ab8…71b`）。\n"
        "正常结论 $P=UI$。"
    )
    identity = case.chats.open_course("NON_REAL_COURSE_A", new=True)
    case.chats.begin(identity, "NON_REAL_PROJECTION", "请复习", {"evidence": evidence})
    case.chats.finish(identity, "NON_REAL_PROJECTION", raw, {"raw_response": raw})
    snapshot = copy.deepcopy(case.chats.snapshot(identity))
    projected = visible_answer(raw, evidence)
    assert "额定计算 · 第2修订" in projected and "作业.docx · 复盘第4版" in projected
    assert "第2修订" in projected and "复盘第4版" in projected
    for value in ("label", "memory_revision", "version_number", "saved_review", "M0123456789ab",
                  "Rabcdef123456", "50e90ab8aaf0f34bead410809e1afc6c6a2d0bc3", "``"):
        assert value not in projected
    assert "50e90ab8" not in projected and "原始对话快照" not in projected
    assert "已确认知识知识记忆" not in projected
    assert "正常结论 $P=UI$。" in projected
    assert case.chats.snapshot(identity) == snapshot


def test_citation_cleanup_preserves_real_code_fences_math_inline_code_and_markdown():
    code = ('```python\nlabel = "R0123456789ab"\nmemory_revision = 2\n\n\nprint(label)\n```\n')
    math = r"\[R0123456789ab + S0123456789ab = 2\]"
    inline = '`print("label", memory_revision)`'
    markdown = "| 知识 | 条件 |\n|---|---|\n| 转矩 | $s>0$ |"
    raw = code + "\n" + math + "\n" + inline + "\n" + markdown + "\n结论[S0123456789ab]。"
    actual = visible_answer(raw)
    assert code in actual and math in actual and inline in actual and markdown in actual
    assert actual.endswith("结论。")


@pytest.mark.parametrize("artifact", ["label ``", "```label\n\n```\n"])
def test_empty_label_artifact_has_an_explicit_human_fallback(artifact):
    assert visible_answer(artifact).strip() == UNAVAILABLE


def test_stable_memory_label_does_not_replace_old_revision_with_a_new_revision():
    refs = [{"label": "M0123456789ab", "memory_revision": revision, "text": "额定计算"}
            for revision in (1, 2)]
    actual = visible_answer("label `M0123456789ab`，memory_revision 1。", refs)
    assert "第1修订" in actual and "第2修订" not in actual


@pytest.mark.parametrize("math", [
    "$S3058e1520425$", "$ S3058e1520425 $", "$$\nS3058e1520425\n$$",
    r"\( S3058e1520425 \)", r"\[S3058e1520425\]",
])
def test_id_only_math_reference_uses_existing_readable_cleanup(math: str) -> None:
    raw = "正文[" + math + "]。"
    assert visible_answer(raw) == "正文。"
    assert math in raw


def test_math_wrapped_citation_with_annotation_keeps_the_annotation() -> None:
    raw = "结论[$S3058e1520425$ 原稿此处不充分，已补全]，公式 $P=UI$。"
    result = visible_answer(raw)
    assert "S3058e1520425" not in result
    assert "原稿此处不充分，已补全" in result
    assert "$P=UI$" in result
    assert "$S3058e1520425$" in raw


@pytest.mark.parametrize("label", ["S3058e1520425", "M0123456789ab", "Rabcdef123456"])
def test_id_only_math_uses_source_names_when_the_existing_context_requires_it(label: str) -> None:
    # Memories derive their visible title from content, while sources/reviews
    # use the filename. Supply both without changing the existing naming rules.
    ref = {"label": label, "name": "作业.docx", "text": "作业.docx"}
    result = visible_answer("依据 $" + label + "$ 作答。", [ref])
    assert label not in result
    assert "依据" in result and "作业.docx" in result


def test_actual_math_using_a_key_shaped_variable_and_code_are_unchanged() -> None:
    maths = [r"$S3058e1520425+x$", r"\[M0123456789ab=2\]", r"$S3058e1520425_{1}$"]
    code = '`[$S3058e1520425$]`；`value = "$Rabcdef123456$"`'
    fence = '```latex\n[$S3058e1520425$]\n```\n'
    raw = "\n".join(maths) + "\n" + code + "\n" + fence
    assert visible_answer(raw) == raw


def test_math_reference_read_projection_does_not_change_saved_message(case) -> None:
    raw = "频率条件[$S3058e1520425$ 原稿注明]，实际 $|s|f_1$。"
    identity = case.chats.open_course("NON_REAL_COURSE_A", new=True)
    case.chats.begin(identity, "NON_REAL_MATH_REFERENCE", "请核对", {"evidence": []})
    case.chats.finish(identity, "NON_REAL_MATH_REFERENCE", raw, {"raw_response": raw})
    snapshot = copy.deepcopy(case.chats.snapshot(identity))

    assert "S3058e1520425" not in visible_answer(raw)
    assert case.chats.snapshot(identity) == snapshot
