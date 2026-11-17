"""Course chat renders versioned review memory and friendly citations without changing history."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest
from test_ai_review_support import RecordingProvider, review_case

from src.study_loop import chat
from src.study_loop.ai_review import AIReviewService
from src.study_loop.chat_citations import visible_answer
from src.study_loop.knowledge_memory import KnowledgeMemoryService
from src.study_loop.source_presentation import evidence_name


@pytest.fixture
def case(tmp_path, monkeypatch):
    source = review_case(tmp_path / "NON_REAL_COURSE_RENDER")
    reviews = AIReviewService(source.store)
    document = reviews.generate([source.selected.id], RecordingProvider("NON_REAL_REVIEW_AI"))
    reviews.update_working(document["id"], r"NON_REAL_SAVED_REVIEW $q=\frac{a}{b}$。")
    version = reviews.save_version(document["id"])
    memories = KnowledgeMemoryService(source.store)
    content = "NON_REAL_SHARED_CONFIRMED_KNOWLEDGE"
    extraction = memories.extract_review(document["id"], RecordingProvider(content))
    memory = memories.confirm(extraction["id"], content, confirmed=True)
    identity = source.chats.open_course("NON_REAL_COURSE_A", new=True)
    refs = chat.retrieve(source.chats, source.chats.conversation(identity), content)
    review_ref = next(ref for ref in refs if ref.get("kind") == "saved_review")
    memory_ref = next(ref for ref in refs if ref["label"].startswith("M"))
    provider = RecordingProvider(f"NON_REAL_FIRST_REPLY（{review_ref['label']}，第1版）。")
    chat.send(source.chats, identity, content, "FIRST", provider, allow_remote=True)
    provider.content = f"NON_REAL_SECOND_REPLY [{memory_ref['label']}]。"
    chat.send(source.chats, identity, content, "SECOND", provider, allow_remote=True)
    monkeypatch.setattr(
        "src.study_loop.chat_ui.configured_chat_model", lambda: ("qwen", "NON_REAL_MODEL")
    )
    monkeypatch.setattr("src.study_loop.chat_ui.StudyChatProvider", lambda _: provider)
    return SimpleNamespace(
        **vars(source), reviews=reviews, document=document, version=version,
        memories=memories, extraction=extraction, memory=memory, identity=identity,
        provider=provider, review_ref=review_ref, memory_ref=memory_ref, content=content,
    )


def _app(case):
    app = AppTest.from_string(
        "from pathlib import Path\nimport streamlit as st\n"
        "from src.study_loop.storage import StudyStore\n"
        "from src.study_loop.chat_ui import render_chat\n"
        "render_chat(StudyStore(Path(st.session_state['test_root'])), "
        "st.session_state['test_conversation'], embedded=True)"
    )
    app.session_state["test_root"] = str(case.store.root)
    app.session_state["test_conversation"] = case.identity
    return app.run(timeout=30)


def test_course_chat_renders_confirmed_review_memory_without_treating_it_as_chat(case):
    before = copy.deepcopy(case.chats.snapshot(case.identity))
    archive = case.chats.archived(case.extraction["archive_id"])
    assert "review_items" in archive["conversation"]["context"]
    assert "item" not in archive["conversation"]["context"]
    app = _app(case)
    assert not app.exception and not app.error
    assert any(button.label == "打开来源复盘" for button in app.button)
    downloads = app.get("download_button")
    assert any(button.label == "下载完整 Markdown" for button in downloads)
    assert not any(button.label == "下载关联的原始对话" for button in downloads)
    captions = [caption.value for caption in app.caption if caption.value.startswith("引用资料：")]
    assert captions == ["引用资料：" + evidence_name(case.review_ref, case.store),
                        "引用资料：" + evidence_name(case.memory_ref, case.store)]
    visible = "\n".join(value.value for value in app.markdown)
    assert case.review_ref["label"] not in visible and case.memory_ref["label"] not in visible
    assert "NON_REAL_FIRST_REPLY（第1版）" in visible
    assert case.chats.snapshot(case.identity) == before
    assert case.review_ref["label"] in before["messages"][1]["metadata"]["raw_response"]
    assert len(case.provider.prompts) == 2
    next(button for button in app.button if button.label == "打开来源复盘").click().run(timeout=30)
    assert not app.exception and not app.error
    assert app.session_state["sl_ai_review_requested"] == {
        "kind": "document", "id": case.document["id"], "version_id": case.version["id"],
    }
    assert case.chats.snapshot(case.identity) == before and len(case.provider.prompts) == 2


def test_mixed_memory_origins_keep_review_navigation_course_navigation_and_legacy_downloads(case):
    extraction = case.memories.extract_course(case.identity, RecordingProvider(case.content))
    assert case.memories.confirm(extraction["id"], case.content, confirmed=True) == case.memory
    legacy = case.chats.add_candidate(case.chats.archive(case.identity), "NON_REAL_LEGACY_MEMORY")
    case.chats.confirm(legacy, "NON_REAL_LEGACY_MEMORY", confirmed=True)
    before = copy.deepcopy(case.chats.snapshot(case.identity))
    memory = next(
        row for row in case.chats.memories("NON_REAL_COURSE_A") if row["id"] == case.memory
    )
    assert {source["type"] for source in memory["sources"]} == {
        "review_memory", "course_dialogue_memory",
    }
    app = _app(case)
    assert not app.exception and not app.error
    assert sum(button.label == "打开来源复盘" for button in app.button) == 1
    assert sum(button.label == "打开来源对话" for button in app.button) == 1
    downloads = app.get("download_button")
    assert sum(button.label == "下载关联的原始对话" for button in downloads) == 2
    assert any(button.label == "下载完整 Markdown" for button in downloads)
    assert case.chats.snapshot(case.identity) == before
    assert len(case.provider.prompts) == 2


def test_old_and_new_answers_display_their_own_memory_revision_without_future_evidence(case):
    old_messages = copy.deepcopy(case.chats.messages(case.identity))
    current = case.chats.memory(case.memory)
    case.chats.update_memory(
        case.memory, current["content"] + "\n合成的第二次人工修订提醒。",
        expected_revision=current["revision"],
    )
    case.provider.content = f"NON_REAL_THIRD_REPLY [{case.memory_ref['label']}]。"
    chat.send(case.chats, case.identity, case.content, "THIRD", case.provider, allow_remote=True)
    snapshot = copy.deepcopy(case.chats.snapshot(case.identity))
    app = _app(case)
    assert not app.exception and not app.error
    captions = [value.value for value in app.caption if value.value.startswith("引用资料：")]
    assert len(captions) == 3
    assert "第1修订" in captions[1] and "第2修订" not in captions[1]
    assert "第2修订" in captions[2] and "第1修订" not in captions[2]
    assert case.chats.messages(case.identity)[:4] == old_messages
    assert case.chats.snapshot(case.identity) == snapshot
    assert case.chats.memory(case.memory)["revision"] == 2
    assert len(case.provider.prompts) == 3


@pytest.mark.parametrize("state", ["stale", "unavailable"])
def test_saved_management_keeps_inactive_memory_manageable_without_exporting_review_as_chat(
    case, monkeypatch, state,
):
    ordinary_archive = case.chats.archive(case.identity)
    assert ordinary_archive != case.extraction["archive_id"]
    if state == "stale":
        case.reviews.update_working(case.document["id"], "NON_REAL_NEW_SAVED_REVIEW")
        case.reviews.save_version(case.document["id"])
    else:
        case.reviews.delete_document(case.document["id"], confirmed=True)
    assert case.chats.memory_status(case.memory)[state]
    assert case.chats.memories("NON_REAL_COURSE_A") == []
    source_status = copy.deepcopy(case.chats.memory_status(case.memory))
    history = copy.deepcopy(case.chats.snapshot(case.identity))
    review = copy.deepcopy(case.chats.snapshot(case.document["conversation_id"]))
    versions = copy.deepcopy(case.reviews.versions(case.document["id"]))
    archives = {row["id"]: copy.deepcopy(case.chats.archived(row["id"]))
                for row in case.chats.archives()}
    study_bytes = case.store.db.read_bytes()
    bindings = []

    def no_model(*args, **kwargs):
        bindings.append(True)
        raise AssertionError("Managing saved records must never initialize a model")

    monkeypatch.setattr("src.study_loop.chat_ui.StudyChatProvider", no_model)
    app = AppTest.from_string(
        "from pathlib import Path\nimport streamlit as st\n"
        "from src.study_loop.storage import StudyStore\n"
        "from src.study_loop.chat_ui import render_saved_management\n"
        "render_saved_management(StudyStore(Path(st.session_state['test_root'])))"
    )
    app.session_state["test_root"] = str(case.store.root)
    app.run(timeout=30)
    assert not app.exception and not app.error
    assert len(app.selectbox(key="sl_manage_archive").options) == 1
    assert app.selectbox(key="sl_manage_archive").value["id"] == ordinary_archive
    assert any(button.label == "下载所选对话" for button in app.get("download_button"))
    assert app.selectbox(key="sl_manage_memory").value == case.memory
    assert any(case.content in text.value for text in app.markdown)
    captions = "\n".join(caption.value for caption in app.caption)
    assert "当前不参与自动检索" in captions
    assert all(reason in captions for reason in source_status["reasons"])
    delete = next(button for button in app.button if button.label == "删除记忆")
    assert delete.disabled
    confirm = next(box for box in app.checkbox if box.label == "确认将这条记忆移入回收站")
    confirm.check().run(timeout=30)
    app.button(key=delete.key).click().run(timeout=30)
    assert not app.exception and not app.error
    assert case.chats.memories(None, include_stale=True, include_unavailable=True) == []
    app.selectbox(key="sl_manage_trash_category").set_value("memory").run(timeout=30)
    assert app.selectbox(key="sl_manage_trash_selection_memory").value == case.memory
    app.button(key="sl_manage_trash_restore").click().run(timeout=30)
    assert not app.exception and not app.error
    restored = case.chats.memories(None, include_stale=True, include_unavailable=True)
    assert restored[0]["id"] == case.memory and restored[0]["content"] == case.content
    assert case.chats.memories("NON_REAL_COURSE_A") == []
    assert case.chats.memory_status(case.memory) == source_status
    assert case.chats.snapshot(case.identity) == history
    assert case.chats.snapshot(case.document["conversation_id"]) == review
    assert case.reviews.versions(case.document["id"]) == versions
    assert {row["id"]: case.chats.archived(row["id"])
            for row in case.chats.archives()} == archives
    assert case.store.db.read_bytes() == study_bytes
    assert bindings == [] and len(case.provider.prompts) == 2


@pytest.mark.parametrize("raw,expected", [
    ("依据（R0123456789ab，第4版）和[Mabcdef123456]。",
     "引用资料：合成资料 · 第4版；已确认知识记忆"),
    ("仅有 XR0123456789ab 与 Mabcdef123456Z。", None),
    ("未知 R9999aaaabbbb 以及 Snot_a_reference，不应捏造来源。", None),
])
def test_friendly_caption_uses_only_exact_actual_bracket_or_bare_reference_labels(raw, expected):
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from src.study_loop.chat_ui import _course_source_caption\n"
        "_course_source_caption(None, st.session_state['message'], st.session_state['evidence'])"
    )
    app.session_state["message"] = {"content": raw, "metadata": {"raw_response": raw}}
    app.session_state["evidence"] = [
        {"label": "R0123456789ab", "kind": "saved_review", "name": "合成资料 · 第4版"},
        {"label": "Mabcdef123456", "name": "已确认知识记忆"},
        {"label": "R111122223333", "kind": "saved_review", "name": "没有引用的资料"},
    ]
    app.run(timeout=30)
    assert not app.exception and not app.error
    assert [caption.value for caption in app.caption] == ([] if expected is None else [expected])
    assert app.session_state["message"]["metadata"]["raw_response"] == raw


@pytest.mark.parametrize("raw,expected", [
    ("复盘（R01854296376b，第4版）。", "复盘（第4版）。"),
    ("来源（[S0123456789ab]）保留正文。", "来源保留正文。"),
    ("依据[Mabcdef123456]计算完成。", "依据计算完成。"),
    ("未知完整键 R9999aaaabbbb 应隐藏。", "未知完整键  应隐藏。"),
    (r"$R=10\,\Omega,\ S=2\,m^2,\ M=3\,kg$；f()与普通文字（条件成立）。",
     r"$R=10\,\Omega,\ S=2\,m^2,\ M=3\,kg$；f()与普通文字（条件成立）。"),
    ("XR0123456789ab Mabcdef123456Z Snot_a_reference RZZZZZZZZ 保持原文。",
     "XR0123456789ab Mabcdef123456Z Snot_a_reference RZZZZZZZZ 保持原文。"),
    ("R" + "a" * 65, "R" + "a" * 65),
])
def test_display_hides_complete_internal_keys_and_preserves_normal_text_and_math(raw, expected):
    assert visible_answer(raw) == expected
