"""Local Chinese course-chat labels without mutations to model input or archived history."""

from __future__ import annotations

import copy
import re

import pytest
from test_ai_review_support import review_case

from src.study_loop import chat
from src.study_loop.conversation_titles import ConversationTitles, derive_title
from src.study_loop.models import StudyError


@pytest.fixture
def course(tmp_path):
    case = review_case(tmp_path / "NON_REAL_CHAT_TITLES")
    identity = case.chats.open_course(case.selected.subject)
    return case, identity


def _turn(chats, identity, token, question):
    chats.begin(identity, token, question, {})
    chats.finish(identity, token, "NON_REAL_ASSISTANT_REPLY", {})


def test_titles_are_local_chinese_labels_with_empty_and_non_chinese_fallbacks():
    assert derive_title("") == "新对话"
    for question in ("请解释合成课程概念之间的关系与适用条件。", "概念", "Explain local examples"):
        title = derive_title(question)
        assert 8 <= len(title) <= 24
        assert re.fullmatch(r"[\u3400-\u9fff]+", title)
    assert "frac" not in derive_title(r"请说明关系 $x=\frac{a}{b}$ 的条件。")
    prefixed = derive_title("根据我已保存的知识记忆总结合成概念的关系与适用条件。")
    assert prefixed.startswith("合成概念") and prefixed.endswith("整理")


def test_reading_old_titles_does_not_create_a_table_or_modify_any_records(course):
    case, identity = course
    titles = ConversationTitles(case.store)
    before = case.chats.db.read_bytes()
    assert titles.title(identity) == "新对话"
    assert titles.entries(case.selected.subject)[0]["title"] == "新对话"
    assert case.chats.db.read_bytes() == before
    with case.chats.connect() as db:
        assert not db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='conversation_titles'"
        ).fetchone()


def test_manual_rename_survives_new_turns_without_entering_prompt_or_archive(course):
    case, identity = course
    _turn(case.chats, identity, "FIRST", "请解释合成课程概念的适用条件。")
    titles = ConversationTitles(case.store)
    original = copy.deepcopy(case.chats.snapshot(identity))
    archive = case.chats.archive(identity)
    prompt, refs = chat.prepare(case.chats, identity, "继续说明")
    title = "人工命名的课程复习主题"
    titles.rename(identity, title)
    assert ConversationTitles(case.store).title(identity) == title
    assert case.chats.snapshot(identity) == original
    assert case.chats.archived(archive)["messages"] == original["messages"]
    assert chat.prepare(case.chats, identity, "继续说明") == (prompt, refs)
    assert title not in prompt
    assert "title" not in case.chats.conversation(identity)
    _turn(case.chats, identity, "SECOND", "第二个完全不同的合成问题。")
    assert ConversationTitles(case.store).title(identity) == title


def test_identity_and_course_filters_survive_duplicate_names_delete_and_restore(course):
    case, first = course
    second = case.chats.open_course(case.selected.subject, new=True)
    foreign = case.chats.open_course(case.foreign.subject)
    titles = ConversationTitles(case.store)
    for identity in (first, second, foreign):
        titles.rename(identity, "同一个人工课程主题")
    _turn(case.chats, second, "SEARCH", "只有第二个会话包含这个独有检索内容。")
    assert {row["id"] for row in titles.entries(case.selected.subject)} == {first, second}
    assert [row["id"] for row in titles.entries(case.selected.subject, "独有检索内容")] == [second]
    case.chats.delete("conversation", first, confirmed=True)
    assert [row["id"] for row in titles.entries(case.selected.subject)] == [second]
    deleted = [row for row in titles.entries(case.selected.subject, include_deleted=True)
               if row["deleted"]]
    assert len(deleted) == 1 and deleted[0]["id"] == first
    case.chats.restore("conversation", first)
    assert ConversationTitles(case.store).title(first) == "同一个人工课程主题"
    assert {row["id"] for row in titles.entries(case.selected.subject)} == {first, second}


def test_rename_requires_an_active_course_chat_and_nonempty_single_line(course):
    case, identity = course
    titles = ConversationTitles(case.store)
    for value in ("", " ", "甲\n乙", "字" * 49):
        with pytest.raises(StudyError):
            titles.rename(identity, value)
    question = case.chats.open_question(case.selected.id)
    with pytest.raises(StudyError, match="课程对话"):
        titles.rename(question, "题目不能冒充课程")
    case.chats.delete("conversation", identity, confirmed=True)
    with pytest.raises(StudyError):
        titles.rename(identity, "已删除的对话")


def test_first_real_question_persists_once_and_manual_title_wins_after_reopen(course):
    case, identity = course
    titles = ConversationTitles(case.store)
    titles.mark_opened(identity)
    with case.chats.connect() as db:
        placeholder = dict(db.execute("SELECT * FROM conversation_titles").fetchone())
    assert placeholder["mode"] == "auto" and placeholder["first_user_message_id"] is None
    _turn(case.chats, identity, "FIRST_REAL", "请说明合成系统参数之间的关系与适用条件。")
    original = copy.deepcopy(case.chats.snapshot(identity))
    before_prompt = chat.prepare(case.chats, identity, "继续讨论")
    derived = titles.sync_auto(identity)
    with case.chats.connect() as db:
        persisted = dict(db.execute("SELECT * FROM conversation_titles").fetchone())
    assert persisted["mode"] == "auto" and persisted["title"] == derived != "新对话"
    assert persisted["first_user_message_id"] == original["messages"][0]["id"]
    assert ConversationTitles(case.store).title(identity) == derived
    titles.sync_auto(identity)
    with case.chats.connect() as db:
        assert dict(db.execute("SELECT * FROM conversation_titles").fetchone()) == persisted
    _turn(case.chats, identity, "LATER", "另一个不应替换第一问题标题的主题。")
    assert titles.sync_auto(identity) == derived
    titles.rename(identity, "人工保存的稳定复习名称")
    assert ConversationTitles(case.store).sync_auto(identity) == "人工保存的稳定复习名称"
    assert case.chats.messages(identity)[:2] == original["messages"]
    # Title metadata never enters the immutable conversation or its model context.
    assert case.chats.conversation(identity) == original["conversation"]
    assert "人工保存的稳定复习名称" not in chat.prepare(case.chats, identity, "继续讨论")[0]
    assert "title" not in before_prompt[0]


def test_legacy_manual_table_migrates_without_treating_rename_as_automatic(course):
    case, identity = course
    _turn(case.chats, identity, "LEGACY", "一条已有真实用户课程问题。")
    with case.chats.connect() as db:
        db.execute("CREATE TABLE conversation_titles(conversation_id TEXT PRIMARY KEY,"
                   "title TEXT NOT NULL,updated_at TEXT NOT NULL)")
        db.execute("INSERT INTO conversation_titles VALUES (?,?,?)",
                   (identity, "新对话", "2026-01-01T00:00:00+00:00"))
    before = copy.deepcopy(case.chats.snapshot(identity))
    titles = ConversationTitles(case.store)
    assert titles.sync_auto(identity) == "新对话"  # An explicit legacy rename is still manual.
    titles.mark_opened(identity)
    with case.chats.connect() as db:
        row = dict(db.execute("SELECT * FROM conversation_titles").fetchone())
    assert row["mode"] == "manual" and row["title"] == "新对话"
    assert row["last_opened_at"] and case.chats.snapshot(identity) == before


def test_recent_course_navigation_does_not_return_original_empty_default(course, monkeypatch):
    case, original_empty = course
    used = case.chats.open_course(case.selected.subject, new=True)
    foreign = case.chats.open_course(case.foreign.subject)
    _turn(case.chats, used, "USED", "最近实际讨论过的合成课程问题。")
    titles = ConversationTitles(case.store)
    assert case.chats.open_course(case.selected.subject) == original_empty
    assert titles.recent(case.selected.subject) == used
    monkeypatch.setattr("src.study_loop.conversation_titles.now",
                        lambda: "2099-01-01T00:00:01+00:00")
    titles.mark_opened(original_empty)
    assert titles.recent(case.selected.subject) == original_empty
    monkeypatch.setattr("src.study_loop.conversation_titles.now",
                        lambda: "2099-01-01T00:00:02+00:00")
    titles.mark_opened(used)
    titles.mark_opened(foreign)
    assert ConversationTitles(case.store).recent(case.selected.subject) == used
    case.chats.delete("conversation", used, confirmed=True)
    assert titles.recent(case.selected.subject) == original_empty
    case.chats.restore("conversation", used)
    assert titles.recent(case.selected.subject) == used
    assert case.chats.messages(original_empty) == []


def test_local_title_removes_request_scaffolding_without_subject_specific_rules():
    a = derive_title("请根据我已保存的知识记忆，说明合成系统参数的关系与适用边界。")
    b = derive_title("说明合成系统参数的关系与适用边界。")
    assert a == b
    for prompt in ("$x=1$", "1234!", "Explain $x=1$", "\n", "概念"):
        value = derive_title(prompt)
        assert value == "新对话" or (8 <= len(value) <= 24
                                    and re.fullmatch(r"[\u3400-\u9fff]+", value))


@pytest.mark.parametrize("question,expected", [
    ("请总结异步电动机的功率关系", "异步电动机功率关系整理"),
    ("电拖常考公式有哪些", "电拖常考公式整理"),
    ("为什么转子铜耗要用气隙功率计算", "转子铜耗计算依据"),
    ("比较直流电机和异步电机的功率平衡", "直流电机和异步电机功率平衡对比"),
    ("请总结传感器的测量误差与校准方法", "传感器测量误差与校准方法整理"),
    ("为什么缓存需要使用过期机制", "缓存过期机制原因分析"),
    ("比较液压回路和气动回路的响应特性", "液压回路和气动回路响应特性对比"),
])
def test_request_topics_are_natural_across_questions_and_courses(question, expected):
    assert derive_title(question) == expected
    assert 8 <= len(expected) <= 24


def test_source_prefaces_and_course_pronouns_do_not_become_the_topic():
    course = "电机驱动与运动控制"
    prompt = ("请根据我已经导入的七份作业材料、已保存的单题复盘，以及经过我确认的长期知识记忆，"
              "系统总结这门课程需要重点掌握的计算公式。\n\n要求：按知识模块分类。")
    assert derive_title(prompt, course) == course + "计算公式整理"
    assert derive_title(prompt, "自动控制理论") == "自动控制理论计算公式整理"
    assert derive_title(
        "请从我已经确认的知识记忆中，找到4-2三相异步电动机计算题对应的个人复习提醒，复述原文。"
    ) == "三相异步电动机个人复习提醒"
    assert derive_title(
        "请回顾我之前修改并保存的4-2三相异步电动机复盘记忆。\n告诉我其中有哪些关键公式。"
    ) == "三相异步电动机复盘记忆回顾"


def test_old_automatic_title_upgrades_idempotently_while_manual_title_is_untouched(course):
    case, first = course
    second = case.chats.open_course(case.selected.subject, new=True)
    for identity in (first, second):
        _turn(case.chats, identity, "TOPIC", "请总结传感器的测量误差与校准方法。")
    titles = ConversationTitles(case.store)
    titles.sync_auto(first)
    titles.rename(second, "用户明确保留的人工名称")
    original = {identity: copy.deepcopy(case.chats.snapshot(identity))
                for identity in (first, second)}
    with case.chats.connect() as db:
        db.execute("UPDATE conversation_titles SET title=? WHERE conversation_id=?",
                   ("已保存的资料以及经过确认的知识记忆", first))
    entries = titles.entries(case.selected.subject)
    assert next(row for row in entries if row["id"] == first)["title"] == (
        "传感器测量误差与校准方法整理"
    )
    with case.chats.connect() as db:
        before = [dict(row) for row in db.execute("SELECT * FROM conversation_titles")]
    assert next(row for row in before if row["conversation_id"] == first)["title"] == (
        "传感器测量误差与校准方法整理"
    )
    titles.sync_auto(first)
    titles.entries(case.selected.subject)
    with case.chats.connect() as db:
        assert [dict(row) for row in db.execute("SELECT * FROM conversation_titles")] == before
    assert titles.title(second) == "用户明确保留的人工名称"
    assert {identity: case.chats.snapshot(identity) for identity in (first, second)} == original
