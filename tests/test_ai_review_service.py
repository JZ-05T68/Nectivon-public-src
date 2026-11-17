"""AI-review persistence and input-boundary regressions using temporary synthetic data."""

from __future__ import annotations

import copy
import json

import pytest
from test_ai_review_support import (
    FORMULA_ANSWER,
    RecordingProvider,
    record_actual_attempt,
    review_case,
)

from src.ai.conversation import decode_conversation
from src.study_loop import chat
from src.study_loop.ai_review import AIReviewService
from src.study_loop.chat_store import ChatStore
from src.study_loop.models import Draft, Page, Question, StudyError
from src.study_loop.storage import StudyStore


@pytest.fixture
def case(tmp_path):
    return review_case(tmp_path / "NON_REAL_REVIEW_LIBRARY")


def _context(provider: RecordingProvider) -> dict:
    messages = decode_conversation(provider.prompts[-1])
    assert messages[0]["role"] == "system"
    return json.loads(messages[1]["content"].split("\n", 1)[1])


def test_pending_and_saved_materials_are_available_without_confirmation(case):
    service = AIReviewService(case.store)
    original = case.store.db.read_bytes()
    options = {option["id"]: option for option in service.options()}
    assert set(options) == {case.selected.id, case.sibling.id, case.foreign.id}
    assert options[case.selected.id]["saved"] is False
    assert options[case.selected.id]["source_id"] == case.source.id
    assert options[case.selected.id]["filename"] == case.source.name
    assert options[case.selected.id]["number"] == "4-2"
    assert "SELECTED_4_2_QUESTION" in options[case.selected.id]["preview"]
    assert not case.store.items() and not case.store.sessions() and not case.store.attempts()
    assert original == case.store.db.read_bytes()
    case.store.confirm(case.selected, user_confirmed=True)
    options = {option["id"]: option for option in service.options()}
    assert options[case.selected.id]["saved"] is True
    assert options[case.sibling.id]["saved"] is False
    assert len(options) == 3  # A saved item and its extraction are never offered twice.


def test_one_generation_uses_only_selected_question_even_on_a_shared_page(case):
    service, provider = AIReviewService(case.store), RecordingProvider()
    study_before = case.store.db.read_bytes()
    original_drafts = [draft.to_dict() for draft in case.store.drafts()]
    document = service.generate([case.selected.id], provider)
    assert len(provider.prompts) == 1
    prompt = provider.prompts[0]
    assert "SELECTED_4_2_QUESTION" in prompt
    assert "USER_SCOPE_A_ONLY_NON_REAL" in prompt
    for forbidden in (
        "UNSELECTED_5_1_SENTINEL", "OTHER_ONLY_SPEED_8412", "OTHER_ONLY_R_0_987",
        "UNSELECTED_IMPORTED_AI_SENTINEL", "FOREIGN_COURSE_CONTENT_SENTINEL",
        "USER_SCOPE_B_FORBIDDEN_NON_REAL",
    ):
        assert forbidden not in prompt
    assert document["status"] == "ready"
    assert document["item_ids"] == [case.selected.id]
    assert document["source_ids"] == [case.source.id]
    assert document["raw_ai"] == document["working"] == FORMULA_ANSWER
    assert document["initial_working"] == FORMULA_ANSWER
    assert service.document(document["id"]) == document
    assert case.chats.messages(document["conversation_id"])[-1]["content"] == FORMULA_ANSWER
    assert [draft.to_dict() for draft in case.store.drafts()] == original_drafts
    assert not case.store.items() and not case.store.sessions() and not case.store.attempts()
    assert not case.chats.memories(None) and not case.chats.candidates(document["conversation_id"])
    assert study_before == case.store.db.read_bytes()


def test_review_envelope_keeps_shared_policies_once_without_losing_task_boundaries(case):
    """Deduplicate the actual outgoing envelope, not merely an unused prompt constant."""
    from src.study_loop.color_markup import STUDY_COLOR_POLICY
    from src.study_loop.formula_quality import REVIEW_QUALITY_POLICY

    provider = RecordingProvider()
    AIReviewService(case.store).generate([case.selected.id], provider)
    messages = decode_conversation(provider.prompts[0])
    assert [message["role"] for message in messages] == ["system", "user", "user"]
    for policy in (REVIEW_QUALITY_POLICY, STUDY_COLOR_POLICY):
        assert messages[0]["content"].count(policy) == 1
        assert sum(message["content"].count(policy) for message in messages) == 1
    request = messages[-1]["content"]
    assert "只使用本次选中的学习条目" in request
    assert "不推断我已经作答、答错或掌握" in request
    assert "不自动记入掌握状态或长期记忆" in request
    assert len(provider.prompts) == 1


def test_review_context_separates_imported_claims_annotations_and_real_attempts(case):
    provider = RecordingProvider()
    AIReviewService(case.store).generate([case.selected.id], provider)
    context = _context(provider)
    rendered = json.dumps(context, ensure_ascii=False)
    assert "IMPORTED_REFERENCE_CLAIM_NOT_PERSONAL_ANSWER" in rendered
    assert "HISTORICAL_AI_EXPLANATION_NOT_PERSONAL_ATTEMPT" in rendered
    assert "USER_ANNOTATION_NOT_PERSONAL_ATTEMPT" in rendered
    # A model must not be supplied fictional personal work from an imported obstacle.
    material = context["selected_materials"][0]
    assert material["actual_user_attempts"] == []
    question = material["question_context"]
    assert question["problem"]["content"] == case.selected.question.content
    assert question["material_state"]["saved"] is False
    assert question["imported_reference_claims"]["observed_answer"] == (
        "IMPORTED_REFERENCE_CLAIM_NOT_PERSONAL_ANSWER"
    )
    assert question["user_annotations"]["reference_answer"] == (
        "USER_ANNOTATION_NOT_PERSONAL_ATTEMPT"
    )
    request = decode_conversation(provider.prompts[-1])[-1]["content"]
    assert "没有证据时" in request and "不推断我已经作答、答错或掌握" in request


def test_default_review_requests_methods_connections_formulas_and_evidence_boundaries(case):
    provider = RecordingProvider()
    AIReviewService(case.store).generate([case.selected.id], provider)
    request = decode_conversation(provider.prompts[-1])[-1]["content"]
    for requirement in (
        "主动汇总", "关键公式", "符号含义", "单位与适用条件", "条件触发相应方法",
        "每一步为什么成立", "概念、公式和知识点之间的联系", "易错点与适用边界",
        "不按原解析逐段复述", "没有证据时", "不列出等待填写的空栏目",
        "表格、说明、易错提醒和边界讨论", "整体与分量", "完整定义式", "完整等价式",
        "求和、平均、比例与变换系数", "不能只凭量纲相同", "基本单位与显示单位",
        "变量取值范围与成立条件", "零值、负值或其他工况", "大小与有符号量",
        "定义或独立关系复核关键系数", "在同一次生成内完成", "不输出自检清单",
        "未经推导就等同于另一对象", "定义式、恒等关系和含除法的变形式",
        "中间新引入的符号也须定义", "记录为空或缺失时，不得声称已核对",
        "没有实质问题无需另设来源核对栏目",
    ):
        assert requirement in request


def test_default_self_check_is_shared_across_unrelated_materials_without_embedded_answers(tmp_path):
    """The actual default send must stay generic while each source claim stays unverified data."""
    store = StudyStore(tmp_path / "NON_REAL_GENERIC_REVIEW", create=True)
    service = AIReviewService(store)
    scenarios = (
        ("NON_REAL_MATH", "讨论正数变量之间的关系。", r"未核对材料声称 $r=11u/v$。"),
        ("NON_REAL_CIRCUIT", "比较某元件的输入和输出。", r"未核对材料声称 $g=13ab$。"),
        ("NON_REAL_FLUID", "说明一种流动量的定义。", r"未核对材料声称 $w=17cd$。"),
    )
    materials = []
    for index, (marker, problem, claim) in enumerate(scenarios):
        original = (marker + "\n" + problem + "\n" + claim).encode()
        source = store.receive(marker + ".txt", original, "txt")
        page = Page("NON_REAL_PAGE_" + str(index), source.id, None, original.decode())
        store.put_page(page)
        draft = Draft(
            "NON_REAL_ITEM_" + str(index), source.id, [page.id],
            Question(content=marker + " " + problem, observed_analysis=claim),
            subject=marker,
        )
        store.add_draft(draft)
        service.chats.set_course(marker, [source.id], marker + "_USER_SCOPE")
        materials.append((draft, source, original))
    before = store.db.read_bytes()
    provider, requests = RecordingProvider(), []
    for index, (draft, source, original) in enumerate(materials):
        document = service.generate([draft.id], provider)
        messages = decode_conversation(provider.prompts[-1])
        requests.append(messages[-1]["content"])
        assert [message["role"] for message in messages] == ["system", "user", "user"]
        assert document["requirements"] == ""  # No correction cue supplied by this caller.
        assert len(provider.prompts) == index + 1
        material = _context(provider)["selected_materials"][0]
        assert material["question_context"]["problem"]["content"] == draft.question.content
        assert material["question_context"]["imported_reference_claims"]["observed_analysis"] == (
            draft.question.observed_analysis
        )
        assert material["actual_user_attempts"] == []
        assert material["user_provided_exam_scope"] == draft.subject + "_USER_SCOPE"
        assert all(marker not in provider.prompts[-1] for marker, _, _ in scenarios
                   if marker != draft.subject)
        assert store.read_source(source) == original
        assert service.chats.messages(document["conversation_id"])[0]["content"] == requests[-1]
    assert len(set(requests)) == 1
    for marker, _, claim in scenarios:
        assert marker not in requests[0] and claim not in requests[0]
    # A general instruction may ask for checking equations; it must not embed a problem's solution.
    assert not any(token in requests[0] for token in ("4-2", "5-1", "$", r"\frac", r"\sqrt"))
    assert store.db.read_bytes() == before
    assert not store.items() and not store.sessions() and not store.attempts()
    assert service.chats.memories(None) == []


def test_explicit_requirements_are_sent_once_and_oversized_requirements_fail_before_creation(case):
    service, provider = AIReviewService(case.store), RecordingProvider()
    with pytest.raises(StudyError, match="过长"):
        service.generate([case.selected.id], provider, requirements="X" * 16000)
    assert not provider.prompts and not service.documents()
    requirements = "NON_REAL_USER_REQUIREMENT 仅说明本材料方法联系。"
    document = service.generate([case.selected.id], provider, requirements=requirements)
    assert document["requirements"] == requirements
    assert requirements in decode_conversation(provider.prompts[-1])[-1]["content"]
    assert len(provider.prompts) == 1


def test_personal_attempt_evidence_is_real_and_selected_item_only(case):
    record_actual_attempt(case, case.selected, "SELECTED_ACTUAL_USER_ANSWER")
    record_actual_attempt(case, case.sibling, "UNSELECTED_ACTUAL_ANSWER_FORBIDDEN")
    before = copy.deepcopy(case.store.attempts())
    study_before = case.store.db.read_bytes()
    provider = RecordingProvider()
    AIReviewService(case.store).generate([case.selected.id], provider)
    assert "SELECTED_ACTUAL_USER_ANSWER" in provider.prompts[-1]
    assert "NON_REAL_EXPLICIT_USER_OBSTACLE" in provider.prompts[-1]
    assert "UNSELECTED_ACTUAL_ANSWER_FORBIDDEN" not in provider.prompts[-1]
    assert case.store.attempts() == before
    assert case.store.db.read_bytes() == study_before
    assert case.chats.memories(None) == []


@pytest.mark.parametrize("damage", ["non_personal", "wrong_item", "unknown_unit", "empty_work"])
def test_imported_or_inconsistent_attempt_records_are_excluded_from_personal_evidence(case, damage):
    marker = "NON_REAL_DAMAGED_ATTEMPT_MUST_NOT_BE_LEARNER_EVIDENCE"
    record_actual_attempt(case, case.selected, marker)
    with case.store.connect(write=True) as database:
        row = database.execute("SELECT id,payload FROM attempts").fetchone()
        payload = json.loads(row["payload"])
        if damage == "non_personal":
            payload["source"] = "imported_ai_answer"
        elif damage == "wrong_item":
            payload["item_id"] = case.sibling.id
        elif damage == "unknown_unit":
            payload["unit_id"] = "UNKNOWN_UNIT_NOT_IN_THIS_QUESTION"
            database.execute("UPDATE attempts SET unit_id=?", (payload["unit_id"],))
        else:
            payload["answer"] = payload["note"] = ""
        database.execute(
            "UPDATE attempts SET payload=? WHERE id=?", (json.dumps(payload), row["id"])
        )
    study_before = case.store.db.read_bytes()
    provider = RecordingProvider()
    AIReviewService(case.store).generate([case.selected.id], provider)
    assert _context(provider)["selected_materials"][0]["actual_user_attempts"] == []
    assert marker not in provider.prompts[-1]
    assert case.store.db.read_bytes() == study_before


def test_multi_selection_is_explicit_and_each_course_keeps_its_own_constraint(case):
    service, provider = AIReviewService(case.store), RecordingProvider()
    document = service.generate([case.selected.id, case.sibling.id], provider)
    assert set(document["item_ids"]) == {case.selected.id, case.sibling.id}
    assert "SELECTED_4_2_QUESTION" in provider.prompts[-1]
    assert "UNSELECTED_5_1_SENTINEL" in provider.prompts[-1]
    assert "FOREIGN_COURSE_CONTENT_SENTINEL" not in provider.prompts[-1]
    service.generate([case.selected.id, case.foreign.id], provider)
    context = _context(provider)
    constraints = {
        material["item_id"]: material["user_provided_exam_scope"]
        for material in context["selected_materials"]
    }
    assert constraints == {
        case.selected.id: "USER_SCOPE_A_ONLY_NON_REAL",
        case.foreign.id: "USER_SCOPE_B_FORBIDDEN_NON_REAL",
    }
    assert context["course"] == context["user_provided_exam_scope"] == ""
    assert "UNSELECTED_5_1_SENTINEL" not in provider.prompts[-1]
    with pytest.raises(StudyError):
        service.generate([], provider)
    with pytest.raises(StudyError):
        service.generate(["NON_REAL_UNKNOWN_ID"], provider)
    assert len(provider.prompts) == 2


def test_current_course_constraint_is_used_without_expanding_selected_materials(case):
    service, provider = AIReviewService(case.store), RecordingProvider()
    first = service.generate([case.selected.id], provider)
    frozen = copy.deepcopy(case.chats.snapshot(first["conversation_id"]))
    case.chats.set_course(
        "NON_REAL_COURSE_A", [case.source.id, case.foreign.source_id], "USER_SCOPE_A_V2"
    )
    second = service.generate([case.selected.id], provider)
    assert "USER_SCOPE_A_V2" in provider.prompts[-1]
    assert "USER_SCOPE_A_ONLY_NON_REAL" not in provider.prompts[-1]
    assert "FOREIGN_COURSE_CONTENT_SENTINEL" not in provider.prompts[-1]
    assert first["id"] != second["id"]
    assert case.chats.snapshot(first["conversation_id"]) == frozen


def test_saved_manual_question_revision_wins_over_original_extraction(case):
    original = copy.deepcopy(case.selected.to_dict())
    case.selected.question.content = r"SAVED_EDITED_CONDITION $P=Fv$"
    case.selected.question.reference_analysis = "SAVED_USER_CORRECTION"
    case.store.confirm(case.selected, user_confirmed=True)
    provider = RecordingProvider()
    AIReviewService(case.store).generate([case.selected.id], provider)
    question = _context(provider)["selected_materials"][0]["question_context"]
    assert question["problem"]["content"] == case.selected.question.content
    assert question["user_annotations"]["reference_analysis"] == "SAVED_USER_CORRECTION"
    assert question["material_state"]["saved"] is True
    assert original["question"]["content"] not in provider.prompts[-1]
    with case.store.connect() as database:
        payload = database.execute(
            "SELECT payload FROM drafts WHERE id=?", (case.selected.id,)
        ).fetchone()[0]
    assert json.loads(payload) == original


def test_only_selected_question_dialogue_is_sent_as_historical_not_current_roles(case):
    for item, marker in (
        (case.selected, "SELECTED_OLD_QUESTION_DIALOGUE"),
        (case.sibling, "UNSELECTED_OLD_DIALOGUE_FORBIDDEN"),
    ):
        identity = case.chats.open_question(item.id)
        case.chats.begin(identity, "NON_REAL_OLD_TURN", marker + "_USER", {})
        case.chats.finish(identity, "NON_REAL_OLD_TURN", marker + "_AI", {})
    before = copy.deepcopy(case.chats.conversations("question"))
    provider = RecordingProvider()
    AIReviewService(case.store).generate([case.selected.id], provider)
    messages = decode_conversation(provider.prompts[-1])
    material = _context(provider)["selected_materials"][0]
    assert len(material["historical_question_dialogues"]) == 1
    history = material["historical_question_dialogues"][0]["messages"]
    assert [message["role"] for message in history] == ["user", "assistant"]
    assert history[-1]["content"] == "SELECTED_OLD_QUESTION_DIALOGUE_AI"
    assert "UNSELECTED_OLD_DIALOGUE_FORBIDDEN" not in provider.prompts[-1]
    assert len(messages) == 3  # Imported history remains context data, never live turns.
    assert case.chats.conversations("question") == before


def test_explicit_subject_overrides_ambiguous_shared_material_membership(case):
    case.chats.set_course(
        "NON_REAL_COURSE_B", [case.source.id], "AMBIGUOUS_MEMBERSHIP_SCOPE_FORBIDDEN"
    )
    provider = RecordingProvider()
    AIReviewService(case.store).generate([case.selected.id], provider)
    context = _context(provider)
    assert context["course"] == "NON_REAL_COURSE_A"
    assert context["selected_materials"][0]["user_provided_exam_scope"] == (
        "USER_SCOPE_A_ONLY_NON_REAL"
    )
    assert "AMBIGUOUS_MEMBERSHIP_SCOPE_FORBIDDEN" not in provider.prompts[-1]


def test_working_revision_and_saved_versions_survive_reopen_without_overwriting_raw(case):
    provider = RecordingProvider()
    service = AIReviewService(case.store)
    document = service.generate([case.selected.id], provider)
    revision = document["working_revision"]
    changed = service.update_working(
        document["id"], r"NON_REAL_EDIT_ONE $F=\frac{P}{v}$", expected_revision=revision
    )
    assert changed["working_revision"] > revision
    assert changed["raw_ai"] == FORMULA_ANSWER
    assert changed["initial_working"] == FORMULA_ANSWER
    with pytest.raises(StudyError):
        service.update_working(
            document["id"], "STALE_TAB_MUST_NOT_OVERWRITE", expected_revision=revision
        )
    first = service.save_version(document["id"])
    assert first["content"] == changed["working"]
    service.update_working(document["id"], "NON_REAL_EDIT_TWO")
    second = service.save_version(document["id"])
    assert first["id"] != second["id"] and second["number"] > first["number"]
    reopened = AIReviewService(StudyStore(case.store.root))
    assert reopened.document(document["id"])["working"] == "NON_REAL_EDIT_TWO"
    assert reopened.document(document["id"])["raw_ai"] == FORMULA_ANSWER
    assert reopened.document(document["id"])["initial_working"] == FORMULA_ANSWER
    assert reopened.version(first["id"]) == first
    assert {value["id"] for value in reopened.versions(document["id"])} == {
        first["id"], second["id"],
    }
    assert len(provider.prompts) == 1
    assert case.chats.memories(None) == []


def test_generating_again_appends_a_document_without_reusing_edited_working_context(case):
    # Deliberately unverified model math is preserved as evidence, not silently repaired by storage.
    raw_first = "NON_REAL_UNVERIFIED_MODEL\n\n$z=11xy$；模型未证明负值也适用。"
    raw_second = "NON_REAL_SECOND_MODEL\n\n$z=13xy$；单位与条件仍须阅读者核对。"
    service, provider = AIReviewService(case.store), RecordingProvider(raw_first)
    before = case.store.db.read_bytes()
    first = service.generate([case.selected.id], provider)
    assert first["raw_ai"] == first["working"] == raw_first
    original_messages = copy.deepcopy(case.chats.snapshot(first["conversation_id"]))
    assert original_messages["messages"][-1]["metadata"]["raw_response"] == raw_first
    manual = r"FIRST_DOCUMENT_PRIVATE_WORKING_EDIT $z=xy$；人工修订版保留。"
    service.update_working(first["id"], manual)
    version = service.save_version(first["id"])
    assert version["content"] == manual
    first_saved = copy.deepcopy(service.document(first["id"]))
    provider.content = raw_second
    second = service.generate([case.selected.id], provider)
    assert len(provider.prompts) == 2
    assert first["id"] != second["id"]
    assert first["conversation_id"] != second["conversation_id"]
    assert "FIRST_DOCUMENT_PRIVATE_WORKING_EDIT" not in provider.prompts[-1]
    assert "NON_REAL_UNVERIFIED_MODEL" not in provider.prompts[-1]
    assert second["raw_ai"] == second["working"] == raw_second
    assert service.document(first["id"])["raw_ai"] == raw_first
    assert service.document(first["id"])["working"] == manual
    assert service.document(first["id"]) == first_saved
    assert service.version(version["id"]) == version
    assert case.chats.snapshot(first["conversation_id"]) == original_messages
    assert case.chats.messages(second["conversation_id"])[-1]["metadata"]["raw_response"] == (
        raw_second
    )
    assert {value["id"] for value in service.documents(case.selected.id)} == {
        first["id"], second["id"],
    }
    assert case.store.db.read_bytes() == before
    assert not case.store.items() and not case.store.sessions() and not case.store.attempts()
    assert case.chats.memories(None) == []


def test_memory_candidate_uses_selected_immutable_revision_without_confirming_knowledge(case):
    service, provider = AIReviewService(case.store), RecordingProvider()
    document = service.generate([case.selected.id], provider)
    service.update_working(document["id"], r"SELECTED_SAVED_REVISION $F=\frac{P}{v}$")
    selected = service.save_version(document["id"])
    service.update_working(document["id"], "NEWER_WORKING_MUST_NOT_ENTER_SELECTED_MEMORY")
    service.save_version(document["id"])
    candidate = service.memory_candidate(selected["id"])
    candidates = case.chats.candidates(document["conversation_id"])
    chosen = next(value for value in candidates if value["id"] == candidate)
    assert chosen["content"] == selected["content"]
    assert chosen["content"] != document["raw_ai"]
    assert "NEWER_WORKING" not in chosen["content"]
    assert case.chats.memories(None) == []
    archive = case.chats.archived(chosen["archive_id"])
    assert archive["messages"][-1]["content"] == FORMULA_ANSWER
    assert archive["conversation"]["context"]["saved_manual_review_version"] == selected
    assert service.memory_candidate(selected["id"]) == candidate
    assert len(provider.prompts) == 1
    assert case.store.attempts() == []


def test_deleted_source_cannot_send_a_preexisting_saved_review_to_memory(case):
    service, provider = AIReviewService(case.store), RecordingProvider()
    document = service.generate([case.selected.id], provider)
    selected = service.save_version(document["id"])
    case.store.delete_source(case.source.id, user_confirmed=True)
    with pytest.raises(StudyError):
        service.memory_candidate(selected["id"])
    assert not case.chats.memories(None)
    assert not case.chats.candidates(document["conversation_id"])
    assert service.version(selected["id"]) == selected
    assert len(provider.prompts) == 1


def test_deleted_or_changed_source_is_rejected_before_model_dispatch(case):
    service, provider = AIReviewService(case.store), RecordingProvider()
    original = case.store.read_source(case.source)
    case.store.delete_source(case.source.id, user_confirmed=True)
    assert case.selected.id not in {option["id"] for option in service.options()}
    with pytest.raises(StudyError):
        service.generate([case.selected.id], provider)
    assert not provider.prompts and not service.documents()
    assert case.store.path(case.source.blob).read_bytes() == original


def test_changed_original_does_not_use_stale_extraction_or_make_a_paid_call(case):
    service, provider = AIReviewService(case.store), RecordingProvider()
    case.store.path(case.source.blob).write_bytes(b"NON_REAL_TAMPERED_TEST_FILE")
    with pytest.raises(StudyError):
        service.generate([case.selected.id], provider)
    assert not provider.prompts and not service.documents()


def test_failed_generation_is_persisted_and_browsing_does_not_retry(case):
    class FailingProvider(RecordingProvider):
        def complete(self, prompt: str, **kwargs):
            self.prompts.append(prompt)
            raise RuntimeError("NON_REAL_PRIVATE_EXCEPTION_MUST_NOT_LEAK")

    service, provider = AIReviewService(case.store), FailingProvider()
    document = service.generate([case.selected.id], provider)
    assert document["status"] == "failed"
    assert document["raw_ai"] == document["working"] == ""
    assert "NON_REAL_PRIVATE_EXCEPTION_MUST_NOT_LEAK" not in json.dumps(document)
    reopened = AIReviewService(StudyStore(case.store.root))
    assert reopened.document(document["id"])["status"] == "failed"
    assert reopened.documents() == service.documents()
    assert len(provider.prompts) == 1
    assert not case.store.items() and not case.store.sessions() and not case.store.attempts()
    assert ChatStore(case.store).memories(None) == []


def test_oversized_context_is_retained_as_failed_without_dispatch_or_silent_truncation(
    case, monkeypatch,
):
    service, provider = AIReviewService(case.store), RecordingProvider()
    monkeypatch.setattr(chat, "MAX_PROMPT", 10)
    with pytest.raises(StudyError, match="未截断"):
        service.generate([case.selected.id], provider)
    assert provider.prompts == []
    assert len(service.documents()) == 1
    document = service.documents()[0]
    assert document["status"] == "failed"
    assert case.chats.messages(document["conversation_id"]) == []
    assert not case.store.sessions() and not case.store.attempts()
