"""Full-page question/course chat using Streamlit's pinned composer and local history."""

from __future__ import annotations

import re
from uuid import uuid4

import streamlit as st

from src.ai.provider import AIUnavailableError
from src.ai.study_chat_adapter import StudyChatProvider, configured_chat_model
from src.time_display import format_beijing_time

from . import chat, chat_export
from .chat_citations import checked_messages, visible_answer
from .chat_store import ChatStore, digest
from .display import render_saved_text
from .models import StudyError
from .source_presentation import memory_title
from .storage import StudyStore


def question_entry(
    store: StudyStore, item_id: str, *, key: str, in_form: bool = False,
    shown_key: str | None = None,
) -> None:
    """A form submit captures unsaved input before leaving for the question chat."""
    def enter() -> None:
        st.session_state["sl_chat_id"] = ChatStore(store).open_question(item_id)
        if shown_key:
            st.session_state["sl_chat_attempt_origin"] = {"item_id": item_id, "key": shown_key}
        else:
            st.session_state.pop("sl_chat_attempt_origin", None)

    button = st.form_submit_button if in_form else st.button
    button("追问 AI", key=key, on_click=enter)


def course_entry(store: StudyStore) -> None:
    """Course selection is explicit; no automatic mixing of all local user materials."""
    with st.expander("自由复盘 · 按课程提问", expanded=True):
        chats = ChatStore(store)
        courses = chats.courses()
        if courses:
            subject = st.selectbox(
                "复盘课程", [c["subject"] for c in courses], key="sl_chat_course"
            )

            def enter(selected_subject: str = subject) -> None:
                from .conversation_titles import ConversationTitles
                from .course_chat_ui import open_course_chat

                recent = ConversationTitles(store).recent(selected_subject)
                open_course_chat(recent or chats.open_course(selected_subject))

            st.button("进入自由复盘", key="sl_chat_course_open", on_click=enter)
        with st.expander("设置课程与材料范围"):
            with st.form("sl_course_scope"):
                subject = st.text_input("课程名称", key="sl_course_name")
                sources = {s.id: s.name for s in store.sources()}
                selected = st.multiselect(
                    "本课程使用的材料", list(sources), format_func=sources.get
                )
                exam = st.text_area(
                    "已知考试范围及依据（选填）",
                    help="只填写本人已知信息，无法确认的命题规律请注明不确定。",
                )
                save = st.form_submit_button("保存课程范围")
            if save:
                chats.set_course(subject, selected, exam)
                st.rerun()


def _delete_control(chats: ChatStore, kind: str, identity: str, *, key: str,
                    label: str, detail: str = "") -> None:
    """Require an explicit action and confirmation before moving a record to trash."""
    key += "_" + digest([kind, identity])[:12]
    if detail:
        st.caption(detail)
    confirmed = st.checkbox("确认删除" + label, key=key + "_confirm")

    def remove() -> None:
        try:
            chats.delete(kind, identity, confirmed=bool(st.session_state.get(key + "_confirm")))
        except StudyError as exc:
            st.session_state[key + "_error"] = str(exc)
        else:
            st.session_state.pop(key + "_error", None)
            if kind == "conversation" and st.session_state.get("sl_chat_id") == identity:
                st.session_state.pop("sl_chat_id", None)
                st.session_state.pop("sl_chat_attempt_origin", None)

    st.button("删除" + label, key=key + "_delete", disabled=not confirmed, on_click=remove)
    if error := st.session_state.get(key + "_error"):
        st.warning(error)


def _memory(chats: ChatStore, identity: str, provider: StudyChatProvider | None) -> None:
    shown_memories: set[str] = set()
    if st.button("保存完整原始对话", key="sl_chat_archive"):
        chats.archive(identity)
        st.success("已保存。")
    is_course = chats.conversation(identity)["kind"] == "course"
    if is_course:
        from .knowledge_memory_ui import render_course_memory

        def current_provider() -> StudyChatProvider:
            if provider is None:
                raise AIUnavailableError("请先配置模型服务后提炼记忆。")
            return provider

        render_course_memory(
            chats.study, identity, current_provider, model_available=provider is not None,
        )
    elif st.button("AI 提炼记忆候选", key="sl_chat_propose", disabled=provider is None):
        with st.spinner("正在整理记忆…"):
            chat.propose(chats, identity, provider, allow_remote=True)
        st.success("已生成，请核对后保存。")
    with st.expander("手动补充记忆"):
        with st.form("sl_chat_manual_memory"):
            manual = st.text_area("也可以人工填写记忆候选", key="sl_chat_manual_content")
            add = st.form_submit_button("保存人工候选供核对")
        if add:
            chats.add_candidate(chats.archive(identity), manual)
    candidates = chats.candidates(identity)
    if is_course:
        from .knowledge_memory import KnowledgeMemoryService

        extractions = KnowledgeMemoryService(chats.study).extractions(identity)
        extracted = {value.get("candidate_id") for value in extractions}
        selected_extraction = st.session_state.get("sl_knowledge_select_course_" + identity)
        shown_memories.update(
            value["confirmed_memory_id"] for value in extractions
            if value["id"] == selected_extraction and value.get("confirmed_memory_id")
        )
        candidates = [candidate for candidate in candidates if candidate["id"] not in extracted]
    if candidates:
        candidate = st.selectbox(
            "待核对的候选", candidates,
            format_func=lambda c: format_beijing_time(c["created_at"]),
            key="sl_chat_candidate",
        )
        if shown := render_candidate_confirmation(chats, candidate):
            shown_memories.add(shown)
    query = st.text_input("检索本课程已确认的记忆", key="sl_memory_query")
    subject = chats.conversation(identity)["subject"]
    for memory in chats.memories(subject, query):
        with st.expander(memory_title(memory)):
            from .knowledge_memory_ui import render_memory_editor, render_memory_sources

            if memory["id"] not in shown_memories:
                render_memory_editor(chats.study, memory["id"],
                                     key_prefix="sl_chat_memory_" + identity)
                render_memory_sources(chats.study, memory, key_prefix="sl_chat_source_" + identity)
            for link in memory["links"]:
                snapshot = chats.archived(link["archive_id"])
                source = chats.memory_source(snapshot)
                if source and source.get("type") == "review_memory":
                    # A review snapshot contains review_items, not a live user's dialogue.
                    continue
                st.download_button(
                    "下载关联的原始对话", chat_export.study_markdown(snapshot), "记忆来源.md",
                    key="sl_memory_origin_" + digest([memory["id"], link]), mime="text/markdown",
                )


def render_candidate_confirmation(
    chats: ChatStore, candidate: dict, *, prefix: str = "",
) -> str | None:
    """Preview and confirm one exact candidate, shared by chat and saved AI review revisions."""
    confirmed_memories = [memory for memory in chats.memories(
        None, include_stale=True, include_unavailable=True, include_deleted=True
    ) if any(link["candidate_id"] == candidate["id"] for link in memory["links"])]
    if confirmed_memories:
        from .knowledge_memory_ui import render_memory_editor, render_memory_sources

        if st.session_state.pop(prefix + "sl_candidate_saved_" + candidate["id"], False):
            st.success("已保存知识记忆。")
        selected = confirmed_memories[0]["id"]
        if len(confirmed_memories) > 1:
            labels = {memory["id"]: f"记忆{index + 1} · "
                      + memory_title(memory)
                      for index, memory in enumerate(confirmed_memories)}
            selected = st.selectbox("已确认的记忆", list(labels), format_func=labels.get,
                                    key=prefix + "sl_candidate_memory_" + candidate["id"])
        render_memory_editor(chats.study, selected,
                             key_prefix=prefix + "sl_candidate_confirmed_" + candidate["id"])
        memory = next(value for value in confirmed_memories if value["id"] == selected)
        render_memory_sources(chats.study, memory,
                              key_prefix=prefix + "sl_candidate_source_" + candidate["id"])
        return selected
    with st.form(prefix + "sl_candidate_" + candidate["id"]):
        content = st.text_area("编辑知识记忆", candidate["content"], height=220)
        preview = st.form_submit_button("预览修改与公式")
        render_saved_text(visible_answer(content))
        if preview:
            st.caption("预览已更新。")
        confirmed = st.checkbox("我已核对来源、公式和适用边界，确认保存这份知识记忆")
        save = st.form_submit_button("确认写入知识记忆")
    if save:
        chats.confirm(candidate["id"], content, confirmed=confirmed)
        st.session_state[prefix + "sl_candidate_saved_" + candidate["id"]] = True
        st.rerun()
    _delete_control(chats, "candidate", candidate["id"],
                    key=prefix + "sl_candidate_remove_" + candidate["id"], label="此候选")


def _export(chats: ChatStore, identity: str, messages: list[dict]) -> None:
    if messages and not any(m["status"] == "pending" for m in messages):
        snapshot = chats.snapshot(identity)
        st.download_button("下载完整 Markdown", chat_export.study_markdown(snapshot),
                           f"对话-{identity[:12]}.md", mime="text/markdown")
        export_key = "sl_study_docx_" + digest(snapshot)
        if st.button("生成完整 Word", key="sl_chat_docx"):
            try:
                with st.spinner("正在生成 Word…"):
                    st.session_state[export_key] = chat_export.study_docx(snapshot)
            except (StudyError, OSError):
                st.error("Word 导出失败，可以先下载 Markdown。")
        if export_key in st.session_state:
            st.download_button(
                "下载完整 Word", st.session_state[export_key], f"对话-{identity[:12]}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
    _delete_control(chats, "conversation", identity, key="sl_chat_remove", label="本段对话",
                    detail="删除后移入回收站，已确认记忆单独保留。")


def render_saved_management(store: StudyStore) -> None:
    """Manage saved conversations and knowledge, including those outside the current course."""
    chats = ChatStore(store)
    conversations, snapshots, memories, trash = st.tabs(
        ["追问对话", "已保存对话", "记忆", "回收站"]
    )
    with conversations:
        values = chats.conversations("course") + chats.conversations("question")
        if values:
            from .conversation_titles import ConversationTitles

            titles = ConversationTitles(store)
            value = st.selectbox(
                "选择对话", values, key="sl_manage_conversation",
                format_func=lambda c: (c["subject"] + " · " + titles.title(c["id"]) if
                                       c["kind"] == "course" else
                                       (c["subject"] or "题目追问") + " · "
                                       + format_beijing_time(c["created_at"])),
            )
            if st.button("打开对话", key="sl_manage_open"):
                if value["kind"] == "course":
                    from .course_chat_ui import open_course_chat

                    open_course_chat(value["id"])
                else:
                    st.session_state["sl_chat_id"] = value["id"]
                st.rerun()
            _delete_control(chats, "conversation", value["id"],
                            key="sl_manage_chat", label="所选对话",
                            detail="删除后移入回收站，已确认记忆单独保留。")
        else:
            st.caption("暂无对话。")
        courses = chats.courses()
        if courses:
            course = st.selectbox("课程设置", courses, format_func=lambda c: c["subject"],
                                  key="sl_manage_course")
            _delete_control(chats, "course", course["subject"], key="sl_manage_course_remove",
                            label="课程设置", detail="只删除课程设置，已有对话和记忆单独保留。")
    with snapshots:
        values = [
            archive for archive in chats.archives()
            if chats.archived(archive["id"])["conversation"]["kind"] != "review"
        ]
        if values:
            value = st.selectbox("已保存的对话", values, key="sl_manage_archive",
                                 format_func=lambda a: format_beijing_time(a["created_at"]))
            st.download_button("下载所选对话",
                               chat_export.study_markdown(chats.archived(value["id"])),
                               "学习对话.md", mime="text/markdown", key="sl_manage_archive_export")
            _delete_control(chats, "archive", value["id"], key="sl_manage_archive_remove",
                            label="已保存对话", detail="已确认记忆单独保留。")
        else:
            st.caption("暂无已保存对话。")
    with memories:
        query = st.text_input("搜索记忆", key="sl_manage_memory_query")
        values = chats.memories(None, query, include_stale=True, include_unavailable=True)
        if values:
            by_id = {memory["id"]: memory for memory in values}
            if st.session_state.get("sl_manage_memory") not in by_id:
                st.session_state.pop("sl_manage_memory", None)
            selected = st.selectbox(
                "已保存的记忆", list(by_id), key="sl_manage_memory",
                format_func=lambda memory_id: memory_title(by_id[memory_id]),
            )
            from .knowledge_memory_ui import render_memory_editor, render_memory_sources

            render_memory_editor(store, selected, key_prefix="sl_manage_memory")
            render_memory_sources(store, by_id[selected], key_prefix="sl_manage_memory_source")
        else:
            st.caption("暂无记忆。")
    with trash:
        from .trash_ui import render_classified_trash

        render_classified_trash(store, chats=chats, key_prefix="sl_manage_trash")


def render_chat(
    store: StudyStore, identity: str, *, embedded: bool = False, immersive: bool = False
) -> None:
    """Sending a message is the explicit model action; rerenders never dispatch requests."""
    chats = ChatStore(store)
    conversation = chats.conversation(identity)
    kind = conversation["kind"]
    immersive = immersive and kind == "course"

    def go_back() -> None:
        st.session_state.pop("sl_chat_id", None)
        st.session_state.pop("sl_chat_attempt_origin", None)

    if not embedded and not immersive:
        left, right = st.columns([1, 4])
        left.button("返回原题 / 学习整理", key="sl_chat_return", on_click=go_back)
        right.subheader("题目追问" if kind == "question" else "课程自由复盘")
    if conversation["subject"] and not embedded and not immersive:
        st.caption(conversation["subject"])
    if kind == "course":
        if not immersive:
            histories = chats.conversations("course", conversation["subject"])
            if len(histories) > 1:
                choice = st.selectbox(
                    "已有对话", [c["id"] for c in histories],
                    index=[c["id"] for c in histories].index(identity),
                    format_func=lambda s: format_beijing_time(
                        next(c["created_at"] for c in histories if c["id"] == s)
                    ), key="sl_course_history_" + identity,
                )
                if choice != identity:
                    st.session_state["sl_chat_id"] = choice
                    st.rerun()
            if st.button("新建本课程对话", key="sl_chat_new"):
                st.session_state["sl_chat_id"] = chats.open_course(
                    conversation["subject"], new=True
                )
                st.rerun()
    else:
        histories = {c["id"]: c for c in chats.conversations("question")
                     if c["context"]["item"]["id"] == conversation["context"]["item"]["id"]}
        if len(histories) > 1:
            choice = st.selectbox(
                "本题已有对话", list(histories), index=list(histories).index(identity),
                format_func=lambda s: format_beijing_time(histories[s]["created_at"]),
                key="sl_question_history_" + identity,
            )
            if choice != identity:
                st.session_state["sl_chat_id"] = choice
                st.rerun()
    try:
        selection = configured_chat_model()
    except AIUnavailableError as exc:
        provider = None
        st.info(str(exc))
    else:
        provider = StudyChatProvider(selection)
        label = {"deepseek": "DeepSeek", "qwen": "Qwen"}.get(selection[0], selection[0])
        st.caption(f"{label} · {selection[1]}")
    messages = chats.messages(identity)
    if immersive and not messages:
        with st.container(key="sl_course_chat_welcome"):
            st.markdown("开始这次复习吧，可以直接提问，或先看看「课程资料」。")
    origin = st.session_state.get("sl_chat_attempt_origin")
    if (origin and kind == "question"
            and origin["item_id"] == conversation["context"]["item"]["id"]
            and any(m["role"] == "assistant" and m["status"] == "complete" for m in messages)):
        st.session_state[origin["key"]] = True
    evidence_by_label: dict[str, dict] = {}
    with st.container(**({} if immersive else {"height": 520})):
        for message in checked_messages(messages):
            evidence_by_label.update({ref["label"]: ref
                                      for ref in message["metadata"].get("evidence", [])})
            evidence = list(evidence_by_label.values())
            with st.chat_message(message["role"]):
                content = message["content"]
                if message["role"] == "assistant":
                    content = visible_answer(content, evidence)
                if message["status"] == "error":
                    if (not message["metadata"].get("error_code")
                            and content.startswith("模型调用未完成，原消息和来源已保存。")):
                        content = "这次未取得回答，请重新发送。"
                    st.error(content)
                else:
                    render_saved_text(content)
                    audit = message["metadata"].get("formula_audit")
                    if (message["role"] == "assistant" and isinstance(audit, dict)
                            and audit.get("formula_count", 0) > 0):
                        st.caption("公式与适用条件待人工核验。")
                    if kind == "course" and message["role"] == "assistant":
                        _course_source_caption(chats, message, evidence)
                if message["status"] == "pending":
                    st.caption("正在等待回答…")
    if any(m["status"] == "pending" for m in messages):
        if st.button("结束等待", key="sl_chat_interrupt"):
            chats.interrupt_pending(identity)
            st.rerun()
    text = st.chat_input(
        "输入问题，Enter 发送；Shift+Enter 换行", key="sl_chat_input_" + identity,
        max_chars=16000, disabled=provider is None,
    )
    if text:
        try:
            with st.spinner("正在回答…"):
                chat.send(chats, identity, text, uuid4().hex, provider, allow_remote=True)
        except StudyError as exc:
            st.warning(str(exc))
        else:
            if kind == "course":
                from .conversation_titles import ConversationTitles

                ConversationTitles(store).sync_auto(identity)
            st.rerun()
    with st.expander("高级设置"):
        export_tab, memory_tab = st.tabs(["导出对话", "保存到记忆"])
        with export_tab:
            _export(chats, identity, messages)
        with memory_tab:
            if messages:
                try:
                    _memory(chats, identity, provider)
                except StudyError as exc:
                    st.warning(str(exc))


def _course_source_caption(
    chats: ChatStore | None, message: dict, evidence: list[dict] | None = None,
) -> None:
    """Name cited sources without exposing retrieval diagnostics or internal identifiers."""
    from .source_presentation import evidence_name

    refs = evidence if evidence is not None else message["metadata"].get("evidence", [])
    raw = message["metadata"].get("raw_response", message["content"])
    names = []
    for ref in refs:
        label = ref.get("label", "")
        if not label or not re.search(
            r"(?<![A-Za-z0-9_])" + re.escape(label) + r"(?![A-Za-z0-9_])", raw,
        ):
            continue
        if ref.get("kind") == "saved_review" or label.startswith("M"):
            names.append(evidence_name(ref, chats.study if chats is not None else None))
    if names:
        st.caption("引用资料：" + "；".join(dict.fromkeys(names)))
