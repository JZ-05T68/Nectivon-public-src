"""NON_REAL_TEST_DATA: service/UI contracts, not real-model or learning acceptance."""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import zipfile
from xml.etree import ElementTree as ET

import pytest
from streamlit.testing.v1 import AppTest

from src.ai.conversation import decode_conversation
from src.ai.provider import CompletionResult
from src.study_loop import chat, chat_export
from src.study_loop.chat_store import ChatStore
from src.study_loop.components import import_source
from src.study_loop.models import StudyError
from src.study_loop.storage import StudyStore
from src.study_loop.word_import import structured_word
from src.study_loop.word_math import docx_markdown, markdown_docx


class Provider:
    """A visibly marked test transport, recording real prompts for isolation assertions."""

    def __init__(self):
        self.prompts = []

    def complete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return CompletionResult(
            text=r"NON_REAL_TEST_DATA 公式 $T=\frac{P}{\omega}$；仅为自动测试。",
            model="TEST_ONLY",
            finish_reason="stop",
        )


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.study_loop.chat_ui.configured_chat_model",
        lambda: ("deepseek", "deepseek-v4-pro"),
    )
    study = StudyStore(tmp_path / "study", create=True)
    for number in [1, 2]:
        source = study.receive(
            f"TEST_{number}.txt",
            f"1. ONLY_QUESTION_{number}\n答案：TEST_ANSWER_{number}".encode(),
            "txt",
        )
        import_source(study, source.id)
    chats = ChatStore(study)
    chats.set_course("TEST_COURSE", [s.id for s in study.sources()], "USER_EXAM_SCOPE_ONLY")
    return study, chats, Provider()


def test_three_rounds_isolation_reopen_and_legacy_database_unchanged(case):
    study, chats, provider = case
    before = hashlib.sha256(study.db.read_bytes()).hexdigest()
    first, second = study.drafts()
    identity = chats.open_question(first.id)
    for index in range(3):
        chat.send(
            chats, identity, f"QUESTION_ROUND_{index}", f"r{index}", provider, allow_remote=True
        )
    assert len(chats.messages(identity)) == 6
    assert all(f"QUESTION_ROUND_{i}" in provider.prompts[-1] for i in range(3))
    other = chats.open_question(second.id)
    chat.send(chats, other, "ISOLATED_QUERY", "isolated", provider, allow_remote=True)
    assert first.question.content not in provider.prompts[-1]
    assert "QUESTION_ROUND_0" not in provider.prompts[-1]
    reopened = ChatStore(StudyStore(study.root))
    assert reopened.open_question(first.id) == identity
    assert reopened.messages(identity) == chats.messages(identity)
    assert before == hashlib.sha256(study.db.read_bytes()).hexdigest()


def test_course_free_query_scope_and_memory_confirmation(case):
    study, chats, provider = case
    identity = chats.open_course("TEST_COURSE")
    chat.send(chats, identity, "自由提问推导 TEST", "r1", provider, allow_remote=True)
    prompt = provider.prompts[-1]
    assert "USER_EXAM_SCOPE_ONLY" in prompt
    assert all(s.name in prompt for s in study.sources())
    candidate_id = chat.propose(chats, identity, provider, allow_remote=True)
    assert chats.memories("TEST_COURSE") == []
    with pytest.raises(StudyError):
        chats.confirm(candidate_id, "edited", confirmed=False)
    content = "TEST 适用条件与边界：" + r"$T=\frac{P}{\omega}$"
    memory = chats.confirm(candidate_id, content, confirmed=True)
    assert chats.confirm(candidate_id, content, confirmed=True) == memory
    assert len(chats.memories("TEST_COURSE", "适用条件")) == 1
    other = chats.open_course("TEST_COURSE", new=True)
    chat.send(chats, other, "TEST 的适用条件是什么", "r2", provider, allow_remote=True)
    messages = decode_conversation(provider.prompts[-1])
    context = json.loads(messages[1]["content"].split("\n", 1)[1])
    assert any(ref["text"] == content and ref["memory_links"] for ref in context["evidence"])
    assert chats.memories("OTHER_COURSE") == []


def test_failure_idempotency_pending_and_no_consent(case):
    _, chats, provider = case
    identity = chats.open_course("TEST_COURSE")
    with pytest.raises(StudyError):
        chat.send(chats, identity, "first", "r1", provider, allow_remote=False)
    assert not provider.prompts and not chats.messages(identity)

    class Broken:
        def complete(self, *args, **kwargs):
            raise RuntimeError("DO_NOT_LOG_SECRET")

    chat.send(chats, identity, "first", "r1", Broken(), allow_remote=True)
    chat.send(chats, identity, "first", "r1", provider, allow_remote=True)
    assert not provider.prompts
    assert len(chats.messages(identity)) == 2
    assert "DO_NOT_LOG_SECRET" not in json.dumps(chats.snapshot(identity))
    chats.begin(identity, "pending", "second", {})
    with pytest.raises(StudyError):
        chats.begin(identity, "new", "third", {})
    with pytest.raises(StudyError):
        chats.archive(identity)
    chats.interrupt_pending(identity)
    assert chats.messages(identity)[-1]["status"] == "error"


def test_unknown_citations_and_truncation_visible(case):
    _, chats, _ = case

    class BadCitation:
        def complete(self, *args, **kwargs):
            return CompletionResult(
                "引用 [S999]、[S...]、[Munknown]", "TEST", finish_reason="length"
            )

    identity = chats.open_course("TEST_COURSE")
    chat.send(chats, identity, "test", "r1", BadCitation(), allow_remote=True)
    message = chats.messages(identity)[-1]
    assert "[S999]" not in message["content"]
    assert "来源未核验" in message["content"] and "截断" in message["content"]
    assert message["metadata"]["raw_response"] == "引用 [S999]、[S...]、[Munknown]"
    assert message["metadata"]["unverified_citation_labels"] == ["Munknown", "S...", "S999"]


def test_context_overflow_no_paid_call_and_no_silent_history_loss(case, monkeypatch):
    _, chats, provider = case
    identity = chats.open_course("TEST_COURSE")
    monkeypatch.setattr(chat, "MAX_PROMPT", 10)
    with pytest.raises(StudyError, match="未截断"):
        chat.send(chats, identity, "test", "r1", provider, allow_remote=True)
    assert not chats.messages(identity) and not provider.prompts


def test_complete_exports_native_math_tables_lists_and_word_roundtrip(case):
    _, chats, provider = case
    identity = chats.open_course("TEST_COURSE")
    for i in range(3):
        chats.begin(identity, str(i), f"ROUND_{i}", {})
        chats.finish(
            identity,
            str(i),
            (
                f"ANSWER_{i}\n\n"
                + r"$$s_m=s_N(K_T+\sqrt{K_T^2-1})$$"
                + "\n\n"
                + r"$$\frac{T}{T_m}=\frac{2}{\frac{s}{s_m}+\frac{s_m}{s}}$$"
                + "\n\n"
                "|变量|含义|\n|---|---|\n|$s_m$|TEST|\n\n- 条件一\n- 条件二\n\nLAST_MARKER"
            ),
            {"model": "TEST"},
        )
    snapshot = chats.snapshot(identity)
    md = chat_export.markdown(snapshot)
    data = chat_export.docx(snapshot)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
        assert archive.read("nectivon/transcript.md").decode() == md
        ns = {
            "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
            "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
        }
        assert len(root.findall(".//m:f", ns)) >= 12
        assert len(root.findall(".//m:rad", ns)) == 3
        assert len(root.findall(".//w:tbl", ns)) == 3
        assert root.findall(".//w:numPr", ns)
        text = "".join(root.itertext())
        for i in range(3):
            assert f"ROUND_{i}" in text and f"ANSWER_{i}" in text
        assert text.count("LAST_MARKER") == 3
    assert "\\frac" in docx_markdown(data)


def test_word_headings_preserve_formulas_and_do_not_invent_conversations(case):
    study, _, _ = case
    content = (
        "# NON_REAL_TEST_DATA\n\n# 原题\n\n输入 $x$，求 $y$。\n\n"
        "# 参考答案\n\n$y=1$\n\n# 完整解析\n\n"
        + r"$$y=\frac{x}{\sqrt{x^2}}$$"
        + "\n\n## 追问\n\n为什么？"
    )
    data = markdown_docx(content)
    source = study.receive("TEST.docx", data, "docx")
    from src.study_loop.models import Page

    page = Page("test_word", source.id, None, text=docx_markdown(data))
    drafts = structured_word(source, page, [page.id])
    assert len(drafts) == 1
    assert "\\frac" in drafts[0].question.observed_analysis
    assert "为什么" in drafts[0].question.observed_analysis
    assert drafts[0].question.answer_status == "unconfirmed"


def test_sidecar_rejects_unknown_database_and_legacy_schema_unchanged(case):
    study, chats, _ = case
    with sqlite3.connect(study.db) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1
        assert not db.execute("SELECT name FROM sqlite_master WHERE name='messages'").fetchall()
    with sqlite3.connect(chats.db) as db:
        db.execute("PRAGMA user_version=99")
    before = chats.db.read_bytes()
    with pytest.raises(StudyError, match="版本"):
        ChatStore(study)
    assert chats.db.read_bytes() == before


def test_ui_question_route_three_rounds_back_restore_and_course_entry(case, monkeypatch):
    study, chats, provider = case
    monkeypatch.setenv("EKB_STUDY_LOOP_ACCEPTANCE_CONFIRMED", "yes")
    monkeypatch.setattr("src.study_loop.chat_ui.StudyChatProvider", lambda _: provider)
    app = AppTest.from_string("from src.study_loop.ui import render\nrender()")
    app.session_state["sl_open_root"] = str(study.root)
    app.run(timeout=30)
    assert not app.exception
    item = study.drafts()[0]
    app.text_input(key="sl_draft_input_" + item.id + "_subject").set_value("UNSAVED_SUBJECT")
    app.button(key="sl_draft_chat_" + item.id).click().run(timeout=30)
    assert not app.exception and len(app.chat_input) == 1
    identity = app.session_state["sl_chat_id"]
    for i in range(3):
        app.chat_input[0].set_value(f"UI_ROUND_{i}").run(timeout=30)
        assert not app.exception
    assert len(chats.messages(identity)) == 6
    app.button(key="sl_chat_return").click().run(timeout=30)
    assert not app.exception
    assert app.selectbox(key="sl_draft_choice").value.id == item.id
    assert app.text_input(key="sl_draft_input_" + item.id + "_subject").value == "UNSAVED_SUBJECT"
    assert not study.items()
    app.button(key="sl_draft_chat_" + item.id).click().run(timeout=30)
    assert app.session_state["sl_chat_id"] == identity
    assert len(app.chat_message) == 6
    app.button(key="sl_chat_return").click().run(timeout=30)
    app.button(key="sl_chat_course_open").click().run(timeout=30)
    assert not app.exception
    course_id = app.session_state["sl_chat_id"]
    assert chats.conversation(course_id)["kind"] == "course"
    assert chats.messages(course_id) == []


def test_chat_ui_follows_settings_without_duplicate_key_or_rerun_call(case, monkeypatch):
    study, chats, provider = case
    selected = ["deepseek", "deepseek-v4-pro"]
    bound = []
    monkeypatch.setattr(
        "src.study_loop.chat_ui.configured_chat_model", lambda: tuple(selected)
    )

    def bind(selection):
        bound.append(selection)
        return provider

    monkeypatch.setattr("src.study_loop.chat_ui.StudyChatProvider", bind)
    identity = chats.open_course("TEST_COURSE")
    app = AppTest.from_string(
        "import streamlit as st\nfrom pathlib import Path\n"
        "from src.study_loop.storage import StudyStore\n"
        "from src.study_loop.chat_ui import render_chat\n"
        "render_chat(StudyStore(Path(st.session_state['test_root'])), st.session_state['test_id'])"
    )
    app.session_state["test_root"] = str(study.root)
    app.session_state["test_id"] = identity
    app.session_state["sl_chat_model"] = "qwen3.8-flash"  # Old UI state must be ignored.
    app.run(timeout=30)
    assert not app.exception
    assert any("DeepSeek · deepseek-v4-pro" in c.value for c in app.caption)
    assert not any("Key" in t.label for t in app.text_input)
    assert not any("对话模型" in s.label for s in app.selectbox)
    assert not provider.prompts
    assert not provider.prompts
    app.chat_input[0].set_value("NON_REAL_TEST_DATA first").run(timeout=30)
    assert len(provider.prompts) == 1
    assert bound[-1] == ("deepseek", "deepseek-v4-pro")
    selected[:] = ["qwen", "qwen3.8-max"]
    app.run(timeout=30)
    assert not app.exception and len(provider.prompts) == 1
    assert any("Qwen · qwen3.8-max" in c.value for c in app.caption)
    app.chat_input[0].set_value("NON_REAL_TEST_DATA second").run(timeout=30)
    assert len(provider.prompts) == 2 and bound[-1] == ("qwen", "qwen3.8-max")


def test_chat_ui_no_model_keeps_history_and_manual_memory_available(case, monkeypatch):
    from src.ai.provider import AIUnavailableError

    study, chats, provider = case
    identity = chats.open_course("TEST_COURSE")
    chat.send(chats, identity, "NON_REAL_TEST_DATA", "saved", provider, allow_remote=True)

    def unavailable():
        raise AIUnavailableError("请前往设置选择模型。")

    monkeypatch.setattr("src.study_loop.chat_ui.configured_chat_model", unavailable)
    app = AppTest.from_string(
        "import streamlit as st\nfrom pathlib import Path\n"
        "from src.study_loop.storage import StudyStore\n"
        "from src.study_loop.chat_ui import render_chat\n"
        "render_chat(StudyStore(Path(st.session_state['test_root'])), st.session_state['test_id'])"
    )
    app.session_state["test_root"] = str(study.root)
    app.session_state["test_id"] = identity
    app.run(timeout=30)
    assert not app.exception and len(app.chat_message) == 2
    assert app.chat_input[0].disabled
    assert app.button(key="sl_knowledge_extract_course_" + identity).disabled
    assert any(w.label == "下载完整 Markdown" for w in app.get("download_button"))
    assert any(w.label == "保存人工候选供核对" and not w.disabled for w in app.button)


def test_historical_invalid_citations_checked_without_rewriting_original(case):
    _, chats, _ = case
    identity = chats.open_course("TEST_COURSE")
    chats.begin(identity, "r1", "USER [S...] TEXT RETAINED", {"evidence": [{"label": "Sabc"}]})
    chats.finish(identity, "r1", "OLD ANSWER [S...] and valid [Sabc]", {})
    from src.study_loop.chat_citations import checked_messages

    original = chats.snapshot(identity)
    checked = checked_messages(original["messages"])
    assert "[S...]" not in checked[-1]["content"]
    assert "[Sabc]" in checked[-1]["content"]
    assert checked[0]["content"] == "USER [S...] TEXT RETAINED"
    assert checked[-1]["metadata"]["raw_response"] == "OLD ANSWER [S...] and valid [Sabc]"
    assert chats.snapshot(identity) == original


def test_ui_return_restores_secondary_practice_unsaved_answers_and_hint_state(case, monkeypatch):
    study, _, provider = case
    item = study.drafts()[0]
    study.confirm(item, user_confirmed=True)
    session = study.start_review([item.id], user_started=True)
    monkeypatch.setenv("EKB_STUDY_LOOP_ACCEPTANCE_CONFIRMED", "yes")
    monkeypatch.setattr("src.study_loop.chat_ui.StudyChatProvider", lambda _: provider)
    app = AppTest.from_string("from src.study_loop.ui import render\nrender()")
    app.session_state["sl_open_root"] = str(study.root)
    app.session_state["sl_main_tab"] = "主动复盘"
    app.session_state["sl_session"] = session
    app.run(timeout=30)
    assert not app.exception and not app.error
    practice = next(w for w in app.expander if w.label == "按题练习／主动回忆")
    assert not practice.proto.expanded
    answer = next(w for w in practice.text_area if w.label == "我的实际作答 / 回忆内容")
    attempted = next(
        w for w in practice.checkbox if w.label == "我已实际尝试作答，上面记录的是本人的结果"
    )
    answer.set_value("UI_ONLY_UNSAVED_ANSWER")
    attempted.check()
    practice.button(key="sl_review_chat_" + item.id).click().run(timeout=30)
    assert not app.exception and not app.error and len(app.chat_input) == 1
    app.button(key="sl_chat_return").click().run(timeout=30)
    assert not app.exception and not app.error
    assert app.session_state["sl_main_tab"] == "主动复盘"
    assert app.text_area(key=answer.key).value == "UI_ONLY_UNSAVED_ANSWER"
    assert app.checkbox(key=attempted.key).value
    prefix = f"sl_attempt_{session}_{item.id}_{item.question.id}"
    assert not app.session_state.filtered_state.get(prefix + "_shown", False)
    app.button(key="sl_review_chat_" + item.id).click().run(timeout=30)
    app.chat_input[0].set_value("UI_ONLY_HINT_QUERY").run(timeout=30)
    app.button(key="sl_chat_return").click().run(timeout=30)
    assert app.session_state[prefix + "_shown"]
    assert study.attempts() == []  # Navigation never submits a learning outcome.


def test_edited_memory_versions_can_both_show_original_links(case, monkeypatch):
    study, chats, provider = case
    identity = chats.open_course("TEST_COURSE")
    chat.send(chats, identity, "UI_ONLY", "r1", provider, allow_remote=True)
    candidate = chats.add_candidate(chats.archive(identity), "UI_ONLY_CANDIDATE")
    for version in ["first", "second"]:
        chats.confirm(candidate, "UI_ONLY_EDIT_" + version, confirmed=True)
    monkeypatch.setenv("EKB_STUDY_LOOP_ACCEPTANCE_CONFIRMED", "yes")
    app = AppTest.from_string("from src.study_loop.ui import render\nrender()")
    app.session_state["sl_open_root"] = str(study.root)
    app.session_state["sl_chat_id"] = identity
    app.run(timeout=30)
    assert not app.exception and not app.error
    downloads = app.get("download_button")
    assert sum(w.label == "下载关联的原始对话" for w in downloads) == 2


def test_word_table_breaks_and_external_media_remain_inert():
    content = "|a|b|\n|---|---|\n|$x_1$<br>$x_2$|![image](file:///private.png)|"
    with zipfile.ZipFile(io.BytesIO(markdown_docx(content))) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        assert root.findall(".//w:tbl//w:br", ns)
        assert not root.findall(".//w:drawing", ns)
        assert "<br>" not in "".join(root.itertext())
        assert archive.read("nectivon/transcript.md").decode() == content


def test_test_launcher_rejects_nonempty_unowned_and_formal_port(tmp_path):
    from scripts.phase3_candidate import environment

    root = tmp_path / "owned_elsewhere"
    root.mkdir()
    source = root / "DO_NOT_TOUCH.txt"
    source.write_text("ORIGINAL", encoding="utf-8")
    with pytest.raises(ValueError, match="非空"):
        environment(root, 8517)
    assert [p.name for p in root.iterdir()] == ["DO_NOT_TOUCH.txt"]
    assert source.read_text() == "ORIGINAL"
    with pytest.raises(ValueError, match="端口"):
        environment(tmp_path / "new", 8501)
    assert not (tmp_path / "new").exists()


def test_confirming_edited_draft_keeps_both_question_context_versions(case, monkeypatch):
    study, chats, provider = case
    item = study.drafts()[0]
    original = chats.open_question(item.id)
    chat.send(chats, original, "OLD_CONTEXT_QUERY", "r1", provider, allow_remote=True)
    item.question.content = "EDITED_CONDITIONS_AFTER_CONFIRMATION"
    item.question.reference_analysis = "UI_ONLY_USER_NOTE"
    item.question.answer_status = "user_note"
    study.confirm(item, user_confirmed=True)
    current = chats.open_question(item.id)
    assert current != original
    assert chats.open_question(item.id) == current
    assert len(chats.messages(original)) == 2 and chats.messages(current) == []
    chat.send(chats, current, "NEW_QUERY", "r2", provider, allow_remote=True)
    assert "EDITED_CONDITIONS_AFTER_CONFIRMATION" in provider.prompts[-1]
    assert "UI_ONLY_USER_NOTE" in provider.prompts[-1]
    assert "OLD_CONTEXT_QUERY" not in provider.prompts[-1]
    monkeypatch.setenv("EKB_STUDY_LOOP_ACCEPTANCE_CONFIRMED", "yes")
    app = AppTest.from_string("from src.study_loop.ui import render\nrender()")
    app.session_state["sl_open_root"] = str(study.root)
    app.session_state["sl_chat_id"] = current
    app.run(timeout=30)
    assert not app.exception and not app.error
    app.selectbox(key="sl_question_history_" + current).set_value(original).run(timeout=30)
    assert app.session_state["sl_chat_id"] == original
    assert len(app.chat_message) == 2


def test_wide_formula_table_has_larger_landscape_grid_and_exact_transcript():
    text = (
        "|名称|公式|变量|条件|边界|\n|---|---|---|---|---|\n"
        + r"|TEST|$s_m=s_N(K_T+\sqrt{K_T^2-1})$|$K_T$|TEST_CONDITION|TEST_BOUNDARY|"
    )
    with zipfile.ZipFile(io.BytesIO(markdown_docx(text))) as archive:
        body = archive.read("word/document.xml")
        root = ET.fromstring(body)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        size = root.find(".//w:pgSz", ns)
        assert size.get("{" + ns["w"] + "}orient") == "landscape"
        widths = [int(col.get("{" + ns["w"] + "}w"))
                  for col in root.findall(".//w:tblGrid/w:gridCol", ns)]
        assert sum(widths) >= 15035
        assert archive.read("nectivon/transcript.md").decode() == text


def test_memory_edit_preview_does_not_save_until_explicit_confirmation(case, monkeypatch):
    study, chats, provider = case
    identity = chats.open_course("TEST_COURSE")
    chat.send(chats, identity, "UI_ONLY", "r1", provider, allow_remote=True)
    chats.add_candidate(chats.archive(identity), "UI_ONLY_ORIGINAL_CANDIDATE")
    monkeypatch.setenv("EKB_STUDY_LOOP_ACCEPTANCE_CONFIRMED", "yes")
    app = AppTest.from_string("from src.study_loop.ui import render\nrender()")
    app.session_state["sl_open_root"] = str(study.root)
    app.session_state["sl_chat_id"] = identity
    app.run(timeout=30)
    content = r"UI_ONLY_EDITED $T=\frac{P}{\omega}$"
    next(w for w in app.text_area if w.label == "编辑知识记忆").set_value(content)
    next(w for w in app.button if w.label == "预览修改与公式").click().run(timeout=30)
    assert not app.exception and not app.error
    assert any(content in m.value for m in app.markdown)
    assert chats.memories("TEST_COURSE") == []
    next(w for w in app.checkbox if w.label.startswith("我已核对来源、公式")).check()
    next(w for w in app.button if w.label == "确认写入知识记忆").click().run(timeout=30)
    assert not app.exception and not app.error
    assert chats.memories("TEST_COURSE")[0]["content"] == content


def test_deleting_conversation_hides_its_saved_records_and_can_restore(case):
    _, chats, provider = case
    identity = chats.open_course("TEST_COURSE")
    chat.send(chats, identity, "DELETE_FIXTURE", "r1", provider, allow_remote=True)
    archive = chats.archive(identity)
    candidate = chats.add_candidate(archive, "DELETE_FIXTURE_MEMORY")
    memory = chats.confirm(candidate, "DELETE_FIXTURE_MEMORY", confirmed=True)
    before = chats.archived(archive)
    with pytest.raises(StudyError, match="确认删除"):
        chats.delete("conversation", identity, confirmed=False)
    assert chats.messages(identity)
    chats.delete("conversation", identity, confirmed=True)
    assert chats.conversations("course") == []
    assert chats.archives() == [] and chats.candidates(identity) == []
    with pytest.raises(StudyError, match="已删除"):
        chats.conversation(identity)
    assert chats.memories("TEST_COURSE")[0]["id"] == memory
    assert chats.archived(archive) == before  # Existing memory links retain their source.
    with pytest.raises(StudyError, match="已删除"):
        chats.confirm(candidate, "DO_NOT_RESURRECT_FROM_HIDDEN_CONVERSATION", confirmed=True)
    replacement = chats.open_course("TEST_COURSE")
    assert replacement != identity and chats.messages(replacement) == []
    chats.restore("conversation", identity)
    assert len(chats.conversations("course")) == 2
    assert chats.messages(identity)[0]["content"] == "DELETE_FIXTURE"
    assert chats.archives()[0]["id"] == archive
    with sqlite3.connect(chats.db) as db:
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_deleted_memory_is_excluded_from_retrieval_and_restore_is_explicit(case):
    _, chats, provider = case
    identity = chats.open_course("TEST_COURSE")
    chat.send(chats, identity, "START", "r1", provider, allow_remote=True)
    candidate = chats.add_candidate(chats.archive(identity), "UNIQUE_MEMORY_FIXTURE")
    memory = chats.confirm(candidate, "UNIQUE_MEMORY_FIXTURE", confirmed=True)
    chats.delete("memory", memory, confirmed=True)
    assert chats.memories("TEST_COURSE") == [] and chats.memories(None) == []
    separate = chats.open_course("TEST_COURSE", new=True)
    chat.send(chats, separate, "UNIQUE_MEMORY_FIXTURE", "r2", provider, allow_remote=True)
    assert '"memory_links"' not in provider.prompts[-1]
    chats.restore("memory", memory)
    assert chats.memories("TEST_COURSE")[0]["content"] == "UNIQUE_MEMORY_FIXTURE"


def test_deleted_archives_candidates_and_courses_require_explicit_recovery(case):
    _, chats, provider = case
    identity = chats.open_course("TEST_COURSE")
    chat.send(chats, identity, "START", "r1", provider, allow_remote=True)
    archive = chats.archive(identity)
    candidate = chats.add_candidate(archive, "MEMORY_FIXTURE")
    chats.delete("candidate", candidate, confirmed=True)
    assert chats.candidates(identity) == []
    with pytest.raises(StudyError, match="已删除"):
        chats.confirm(candidate, "MEMORY_FIXTURE", confirmed=True)
    chats.restore("candidate", candidate)
    assert chats.candidates(identity)[0]["id"] == candidate
    chats.delete("archive", archive, confirmed=True)
    assert chats.archives(identity) == [] and chats.candidates(identity) == []
    with pytest.raises(StudyError, match="已删除"):
        chats.confirm(candidate, "MEMORY_FIXTURE", confirmed=True)
    chats.restore("archive", archive)
    assert chats.candidates(identity)[0]["id"] == candidate
    chats.delete("course", "TEST_COURSE", confirmed=True)
    assert chats.courses() == []
    assert chats.messages(identity)  # Removing a course scope does not erase dialogue.
    chats.restore("course", "TEST_COURSE")
    assert chats.courses()[0]["subject"] == "TEST_COURSE"


def test_conversation_cannot_be_deleted_during_a_pending_request(case):
    _, chats, _ = case
    identity = chats.open_course("TEST_COURSE")
    chats.begin(identity, "pending", "PENDING_FIXTURE", {})
    with pytest.raises(StudyError, match="等待回答"):
        chats.delete("conversation", identity, confirmed=True)
    assert chats.trash() == [] and chats.messages(identity)[0]["status"] == "pending"


def test_chat_hides_internal_citations_without_rewriting_saved_answers(case, monkeypatch):
    study, chats, provider = case
    monkeypatch.setattr("src.study_loop.chat_ui.StudyChatProvider", lambda _: provider)
    identity = chats.open_course("TEST_COURSE")
    chats.begin(identity, "r1", "UI_FIXTURE", {"evidence": [{"label": "S10fe0bfd10c1"}]})
    chats.finish(identity, "r1", "VISIBLE_ANSWER [S10fe0bfd10c1] $x^2$", {})
    before = chats.messages(identity)
    app = AppTest.from_string(
        "import streamlit as st\nfrom pathlib import Path\n"
        "from src.study_loop.storage import StudyStore\n"
        "from src.study_loop.chat_ui import render_chat\n"
        "render_chat(StudyStore(Path(st.session_state['test_root'])), st.session_state['test_id'])"
    )
    # Minimal evidence fixture is sufficient for display but not full export.
    monkeypatch.setattr(chat_export, "markdown", lambda _: "UI_FIXTURE_EXPORT")
    app.session_state["test_root"] = str(study.root)
    app.session_state["test_id"] = identity
    app.run(timeout=30)
    assert not app.exception
    assert not app.chat_input[0].disabled
    assert not any(c.label.startswith("启用模型") for c in app.checkbox)
    assert not any(e.label == "本轮检索的来源依据" for e in app.expander)
    assert sum(e.label == "高级设置" for e in app.expander) == 1
    assert any("VISIBLE_ANSWER" in m.value and "$x^2$" in m.value for m in app.markdown)
    assert all("S10fe0bfd10c1" not in m.value for m in app.markdown)
    assert chats.messages(identity) == before


def test_saved_management_deletes_only_after_confirmation(case):
    study, chats, provider = case
    identity = chats.open_course("TEST_COURSE")
    chat.send(chats, identity, "MANAGEMENT_FIXTURE", "r1", provider, allow_remote=True)
    candidate = chats.add_candidate(chats.archive(identity), "MEMORY_DELETE_FIXTURE")
    memory = chats.confirm(candidate, "MEMORY_DELETE_FIXTURE", confirmed=True)
    app = AppTest.from_string(
        "import streamlit as st\nfrom pathlib import Path\n"
        "from src.study_loop.storage import StudyStore\n"
        "from src.study_loop.chat_ui import render_saved_management\n"
        "render_saved_management(StudyStore(Path(st.session_state['test_root'])))"
    )
    app.session_state["test_root"] = str(study.root)
    app.run(timeout=30)
    assert not app.exception
    delete_button = next(w for w in app.button if w.label == "删除记忆")
    assert delete_button.disabled
    next(w for w in app.checkbox if w.label == "确认将这条记忆移入回收站").check().run(timeout=30)
    assert chats.memories(None)[0]["id"] == memory
    app.button(key=delete_button.key).click().run(timeout=30)
    assert not app.exception and chats.memories(None) == []
    assert chats.messages(identity)
    app.selectbox(key="sl_manage_trash_category").set_value("memory").run(timeout=30)
    app.button(key="sl_manage_trash_restore").click().run(timeout=30)
    assert not app.exception and chats.memories(None)[0]["id"] == memory


def test_old_generic_chat_failure_is_shortened_for_display_only(case, monkeypatch):
    study, chats, provider = case
    monkeypatch.setattr("src.study_loop.chat_ui.StudyChatProvider", lambda _: provider)
    identity = chats.open_course("TEST_COURSE")
    original = ("模型调用未完成，原消息和来源已保存。请检查「设置 → AI / 模型服务」与网络；"
                "未自动重试，原请求可能已计费。")
    chats.begin(identity, "legacy", "UI_FAILURE_FIXTURE", {})
    chats.finish(identity, "legacy", original, {"error_type": "RuntimeError"}, error=True)
    chats.begin(identity, "specific", "UI_FAILURE_FIXTURE_NEW", {})
    chats.finish(identity, "specific", "服务暂时繁忙，请稍后重试。",
                 {"error_code": "rate_limited"}, error=True)
    app = AppTest.from_string(
        "import streamlit as st\nfrom pathlib import Path\n"
        "from src.study_loop.storage import StudyStore\n"
        "from src.study_loop.chat_ui import render_chat\n"
        "render_chat(StudyStore(Path(st.session_state['test_root'])), st.session_state['test_id'])"
    )
    app.session_state["test_root"] = str(study.root)
    app.session_state["test_id"] = identity
    app.run(timeout=30)
    assert not app.exception
    assert [m.value for m in app.error] == [
        "这次未取得回答，请重新发送。", "服务暂时繁忙，请稍后重试。",
    ]
    assert chats.messages(identity)[1]["content"] == original


def test_clean_exports_preserve_all_turns_and_embed_the_unmodified_snapshot(case):
    _, chats, _ = case
    identity = chats.open_course("TEST_COURSE")
    evidence = {"label": "S10fe0bfd10c1", "name": "FIXTURE_SOURCE.docx",
                "sha256": "FIXTURE_PRIVATE_HASH", "location": "word/document.xml",
                "text": "FIXTURE_PROVENANCE_ONLY"}
    for index in range(13):
        prompt = "FIXTURE_QUESTION_" + str(max(index - 1, 0))
        chats.begin(identity, str(index), prompt, {"evidence": [evidence]})
        if index == 0:
            chats.finish(identity, str(index), "模型调用未完成，原消息和来源已保存。旧版长提示。",
                         {"error_type": "RuntimeError"}, error=True)
        else:
            chats.finish(identity, str(index),
                         "FIXTURE_ANSWER_" + str(index) + " [S10fe0bfd10c1]\n\n"
                         + r"$$\frac{T}{T_m}=\frac{2}{x+\frac{1}{x}}$$"
                         + "\n\n|名称|公式|条件|边界|\n|---|---|---|---|\n"
                         + r"|FIXTURE|$s_m$|$x>0$|$T_m>0$|", {"model": "FIXTURE_MODEL"})
    snapshot = chats.snapshot(identity)
    clean = chat_export.study_markdown(snapshot)
    assert len(snapshot["messages"]) == 26
    assert len(set(m["content"] for m in snapshot["messages"] if m["role"] == "user")) == 12
    assert all(m["content"] in clean for m in snapshot["messages"] if m["role"] == "user")
    assert clean.count(". 用户") == 13 and clean.count(". AI") == 13
    assert "这次未取得回答，请重新发送。" in clean
    assert all("FIXTURE_ANSWER_" + str(index) in clean for index in range(1, 13))
    for internal in [identity, "S10fe0bfd10c1", "FIXTURE_PRIVATE_HASH", "word/document.xml",
                     "FIXTURE_PROVENANCE_ONLY", "FIXTURE_MODEL", "原始模型正文", "状态："]:
        assert internal not in clean
    with zipfile.ZipFile(io.BytesIO(chat_export.study_docx(snapshot))) as package:
        assert package.read("nectivon/transcript.md").decode("utf-8") == clean
        assert json.loads(package.read("nectivon/source-snapshot.json")) == snapshot
        body = ET.fromstring(package.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
              "m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}
        assert len(body.findall(".//m:oMath", ns)) >= 12
        assert len(body.findall(".//w:tbl", ns)) == 12
    assert chats.snapshot(identity) == snapshot


def test_hiding_a_standalone_source_key_does_not_leave_a_punctuation_paragraph():
    from src.study_loop.chat_citations import visible_answer

    original = ("公式说明。\n\n[S10fe0bfd10c1]。本题两问保持原条件。\n\n"
                "[S10fe0bfd10c1]。\n\n继续推导。")
    visible = visible_answer(original)
    assert visible == "公式说明。\n\n本题两问保持原条件。\n\n继续推导。"
