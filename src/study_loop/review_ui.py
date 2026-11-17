"""User-selected review and user-requested summaries, without automatic reminders."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

import streamlit as st

from src.time_display import format_beijing_time

from .components import search
from .display import render_saved_text
from .import_ui import render_source
from .material_management_ui import visible_session_items
from .models import OUTCOME_LABELS, VERIFICATION_LABELS, Draft, Question, StudyError
from .review import item_state, select_items, summary, unit_states
from .runtime import launch_once, queue, submit
from .storage import StudyStore


def label(item: Draft) -> str:
    return (
        f"{item.subject or '科目未确认'} · {item.question.number or '未编号'} · "
        f"{item.question.content[:36]}"
    )


def _render_reference_content(unit: Question) -> None:
    """Keep answer, explanation and user-note provenance visually separate."""
    if unit.answer_status == "source_confirmed":
        st.markdown("**已核对的参考答案**")
        render_saved_text(unit.reference_answer or "原材料没有可用答案。")
        st.markdown("**已核对的参考解析**")
        render_saved_text(unit.reference_analysis or "原材料没有可用解析。")
    elif unit.answer_status == "user_note":
        st.info("以下是用户补充笔记，不是已核验的原材料答案。")
        st.markdown("**用户补充答案 / 笔记**")
        render_saved_text(unit.reference_answer or "未补充答案。")
        st.markdown("**用户补充解析**")
        render_saved_text(unit.reference_analysis or "未补充解析。")
    else:
        st.info("没有已确认参考内容，信息不足。")


def _render_verification(store: StudyStore, item: Draft) -> None:
    """Show human verification and reachable evidence without rewriting prior versions."""
    st.caption("知识验证状态：" + VERIFICATION_LABELS[item.verification_status])
    if item.verification_note:
        st.markdown("**人工核对依据 / 错误或冲突说明**")
        render_saved_text(item.verification_note)
    if item.revision_of:
        previous = store.item(item.revision_of)
        st.caption("本条修订自：" + label(previous) + "；原版本与历史作答保留。")
    if item.conflict_item_ids:
        with st.expander("存在冲突的历史材料与原始依据"):
            for identity in item.conflict_item_ids:
                conflicting = store.item(identity)
                st.caption(label(conflicting))
                render_saved_text(conflicting.question.content)
                render_source(
                    store,
                    conflicting.source_id,
                    conflicting.page_ids,
                    prefix=f"conflict_{item.id}_{identity}",
                )
    if st.button("创建修订草稿（保留原版本）", key=f"sl_revise_{item.id}"):
        store.create_revision(item.id, user_requested=True)
        st.session_state["sl_revision_created"] = item.id
        st.rerun()
    if st.session_state.get("sl_revision_created") == item.id:
        st.success("已创建待验证的修订草稿，请到“待确认材料”核对并保存；原版本未改动。")


def _filters(store: StudyStore, prefix: str) -> list[Draft]:
    items, attempts = store.items(), store.attempts()
    subject = st.selectbox(
        "科目",
        ["", *sorted({i.subject for i in items if i.subject})],
        format_func=lambda x: x or "全部",
        key=f"{prefix}_subject",
    )
    scoped = [item for item in items if not subject or item.subject == subject]
    knowledge = sorted(
        {
            k
            for i in scoped
            for unit in i.question.units()
            for k in (unit.knowledge or i.question.knowledge)
        }
    )
    point = st.selectbox(
        "知识点", ["", *knowledge], format_func=lambda x: x or "全部", key=f"{prefix}_point"
    )
    attributes = {"": "全部", "error": "错题", "good": "好题", "normal": "普通题"}
    attribute = st.selectbox(
        "题目属性", list(attributes), format_func=attributes.get, key=f"{prefix}_attribute"
    )
    mastery = st.selectbox(
        "掌握情况",
        ["", *OUTCOME_LABELS],
        format_func=lambda x: OUTCOME_LABELS.get(x, "全部"),
        key=f"{prefix}_mastery",
    )
    filtered = select_items(
        items,
        attempts,
        subject=subject,
        knowledge=point,
        attribute=attribute,
        mastery=mastery,
    )
    query = st.text_input("在这些材料中搜索", key=f"{prefix}_query")
    if st.button("搜索已保存文字", key=f"{prefix}_search"):
        if not query.strip():
            raise StudyError("请输入搜索关键词。")
        st.session_state[f"{prefix}_hits"] = (query, search(store, query))
    saved = st.session_state.get(f"{prefix}_hits")
    if query and saved and saved[0] == query:
        filtered = [item for item in filtered if item.id in saved[1]]
    return filtered


def render_review(store: StudyStore) -> None:
    """Record real practice attempts; AI manuscripts use a separate primary workspace."""
    if not store.items():
        st.info("暂无已保存的练习材料。可先保存学习材料；待整理材料也可以直接进行AI复盘。")
        return
    with st.expander("筛选练习材料"):
        items = _filters(store, "sl_review")
    selected = []
    if items:
        selected = st.multiselect(
            "本次复盘内容", items, format_func=lambda item: store.source(item.source_id).name,
            key="sl_review_selected",
        )
    else:
        st.info("当前筛选没有匹配的练习材料，请调整筛选条件。")
    if st.button("主动开始本次复盘", disabled=not selected, key="sl_start_review"):
        st.session_state["sl_session"] = store.start_review(
            [i.id for i in selected], user_started=True
        )
        st.rerun()
    sessions = [
        session for session in store.sessions()
        if session["ended_at"] is None and visible_session_items(store, session)
    ]
    if not sessions:
        st.session_state.pop("sl_review_advanced_context", None)
        return
    identities = [session["id"] for session in sessions]
    current = st.session_state.get("sl_session", identities[0])
    current = current if current in identities else identities[0]
    sessions_by_id = {session["id"]: session for session in sessions}
    session_id = st.selectbox(
        "已主动开始的复盘（可恢复）",
        identities,
        index=identities.index(current),
        format_func=lambda x: format_beijing_time(sessions_by_id[x]["started_at"]),
        key="sl_active_session",
    )
    active_items = visible_session_items(store, sessions_by_id[session_id])
    item = st.selectbox(
        "正在复盘的材料", active_items,
        format_func=lambda item: store.source(item.source_id).name, key=f"sl_active_{session_id}"
    )
    from .ai_review_ui import ai_review_entry
    from .chat_ui import question_entry

    render_saved_text(item.question.content)
    ai_review_entry(store, item.id, key="sl_practice_ai_review_" + item.id)
    states = unit_states(active_items, store.attempts())
    st.caption("当前记录：" + OUTCOME_LABELS[item_state(item, states)])
    units = item.question.units()
    unit = st.selectbox(
        "选择要作答的小问",
        units,
        format_func=lambda q: q.number or "本题 / 知识点 / 笔记",
        key=f"sl_unit_{session_id}_{item.id}",
    )
    if unit.id != item.question.id:
        render_saved_text(unit.content)
    prefix = f"sl_attempt_{session_id}_{item.id}_{unit.id}"
    st.session_state["sl_review_advanced_context"] = (session_id, item.id, unit.id, prefix)
    st.session_state.setdefault(prefix + "_id", uuid4().hex)
    input_key = "sl_attempt_input_" + st.session_state[prefix + "_id"]
    with st.form(prefix + st.session_state[prefix + "_id"]):
        question_entry(
            store, item.id, key="sl_review_chat_" + item.id, in_form=True,
            shown_key=prefix + "_shown",
        )
        answer = st.text_area("我的实际作答 / 回忆内容", height=180, key=input_key + "_answer")
        note = st.text_area("作答说明或无法作答的具体障碍", key=input_key + "_note")
        outcome = st.selectbox(
            "本次结果", ["not_yet", "partial", "correct"], format_func=OUTCOME_LABELS.get,
            key=input_key + "_outcome",
        )
        independent = st.checkbox("本次作答未使用提示或查看答案", key=input_key + "_independent")
        errors = st.text_input(
            "实际出现的错因（中文分号分隔，没有则留空）", key=input_key + "_errors"
        )
        error_confirmed = st.checkbox(
            "我已核对，本次实际作答存在错误（计入错题画像）",
            help="仅记录本人已确认的作答错误；不会、未完成或模型建议本身不等于答错。",
            key=input_key + "_error_confirmed",
        )
        attempted = st.checkbox(
            "我已实际尝试作答，上面记录的是本人的结果", key=input_key + "_attempted"
        )
        save = st.form_submit_button("记录本次结果")
    if save:
        store.record_attempt(
            session_id,
            item.id,
            unit.id,
            outcome=outcome,
            answer=answer,
            note=note,
            independent=independent and not st.session_state.get(prefix + "_shown", False),
            attempted=attempted,
            error_tags=errors.split("；"),
            submission_id=st.session_state[prefix + "_id"],
            error_confirmed=error_confirmed,
        )
        st.success("已记录本次作答。再次作答请先点击“开始新一次作答”。")
    if st.button("开始新一次作答", key=prefix + "_new"):
        st.session_state[prefix + "_id"] = uuid4().hex
        st.session_state[prefix + "_shown"] = False
        st.rerun()
    if st.button("查看已核对的参考内容", key=prefix + "_show_reference"):
        st.session_state[prefix + "_shown"] = True
    if st.session_state.get(prefix + "_shown"):
        _render_reference_content(unit)
    history = [a for a in store.attempts() if a["item_id"] == item.id and a["unit_id"] == unit.id]
    with st.expander("历史作答记录"):
        for attempt in history:
            st.caption(
                f"{format_beijing_time(attempt['created_at'])}（北京时间） · "
                f"{OUTCOME_LABELS[attempt['outcome']]}"
            )
            render_saved_text(attempt["answer"] or attempt["note"])
    if st.button("结束这次复盘", key=f"sl_end_{session_id}"):
        store.end_review(session_id)
        st.rerun()


def render_review_advanced(store: StudyStore) -> None:
    """Render occasional source checks and model-review controls in the bottom settings."""
    context = st.session_state.get("sl_review_advanced_context")
    if not context:
        return
    session_id, item_id, unit_id, prefix = context
    item = next((item for item in store.items() if item.id == item_id), None)
    if item is None:
        return
    unit = next((unit for unit in item.question.units() if unit.id == unit_id), None)
    if unit is None:
        return
    _render_verification(store, item)
    render_source(store, item.source_id, item.page_ids, prefix="review")
    from .model_review_ui import render_model_review

    render_model_review(store, session_id, item, unit, prefix)


def render_summary(store: StudyStore) -> None:
    items = _filters(store, "sl_summary")
    chosen = st.multiselect(
        "总结范围", items, format_func=lambda item: store.source(item.source_id).name,
        key="sl_summary_selection",
    )
    item_ids = [item.id for item in chosen]
    if st.button("主动生成复盘总结", disabled=not chosen, key="sl_generate_summary"):
        st.session_state["sl_report"] = {"ids": item_ids, "text": summary(store, item_ids)}
    report = st.session_state.get("sl_report")
    if report:
        render_saved_text(report["text"])
        st.download_button(
            "下载这份总结（Markdown）",
            report["text"],
            "学习复盘总结.md",
            mime="text/markdown",
            key="sl_download_summary",
        )
    if st.button("主动核验所选原始材料", disabled=not chosen, key="sl_verify_sources"):
        identity = uuid4().hex
        submit(store.root, "study_verify", {"item_ids": item_ids}, f"verify:{identity}")
        launch_once(store.root)
        st.success("已提交本地来源核验，请在任务区查看范围和结果。")
    if st.button(
        "导出所选内容的 PDF 证据包及最新总结", disabled=not chosen, key="sl_export_review"
    ):
        identity = uuid4().hex
        submit(
            store.root,
            "study_export",
            {"item_ids": item_ids, "run_id": identity},
            f"export:{identity}",
        )
        launch_once(store.root)
        st.success("已提交本地导出；不覆盖已有包。")


def render_tasks(store: StudyStore) -> None:
    from nectivon_tasks import TaskError

    def task_action(action: Callable[[], object], *, start_worker: bool = False) -> None:
        try:
            action()
            if start_worker:
                launch_once(store.root)
        except TaskError as exc:
            st.warning(str(exc) + " 已保存记录保留，未自动重试或重发模型请求。")
        except OSError:
            st.warning("任务控制或 worker 启动未完成，请刷新查看实际状态；不要重复提交收费任务。")
        else:
            st.rerun()

    with st.container():
        tasks = queue(store.root)
        if st.button("刷新草稿和任务状态", key="sl_refresh"):
            st.rerun()
        states = {
            "queued": "等待处理",
            "running": "处理中",
            "completed": "已完成",
            "failed": "失败",
            "interrupted": "处理中断",
            "paused": "已暂停",
            "pause_requested": "正在暂停",
            "cancelled": "已取消",
            "cancel_requested": "正在取消",
        }
        for task in tasks.list(limit=30):
            started = format_beijing_time(datetime.fromtimestamp(task.created_at, UTC))
            state_label = states.get(task.state, task.state)
            if task.state == "completed" and task.result and "model_review_run" in task.result:
                state_label = (
                    "模型结果待人工核对"
                    if task.result.get("status") == "pending_user"
                    else "模型复盘未完成 / 信息不足"
                )
            st.caption(f"{started}（北京时间） · {state_label}")
            if task.error_message:
                st.warning(
                    task.error_message + " 原件与已保存草稿保留。收费操作未决时不得重复调用。"
                )
                if task.kind == "study_model_review":
                    from .model_review_ui import _read_history_record

                    failure = _read_history_record(store, task.payload["run_id"], "failure")
                    if failure:
                        st.warning(str(failure.get("message", "模型复盘未完成。")))
            if task.result:
                if task.result.get("failed_file_count"):
                    names = [
                        store.source(failure["source_id"]).name
                        for failure in task.result.get("failed_files", [])
                    ]
                    st.warning("以下材料未能导入：" + "、".join(names))
                if "directory" in task.result:
                    st.success("证据包已保存：" + task.result["directory"])
                elif "scope" in task.result:
                    st.info(task.result["scope"])
                elif "draft_count" in task.result:
                    st.caption(f"已形成 {task.result['draft_count']} 条待确认草稿。")
            columns = st.columns(3)
            if task.state in {"queued", "running"} and columns[0].button(
                "暂停", key=f"pause_{task.id}"
            ):
                task_action(lambda task_id=task.id: tasks.request_pause(task_id))
            if task.state in {"paused", "interrupted", "failed"} and columns[1].button(
                "主动继续",
                key=f"resume_{task.id}",
            ):
                # Unsafe uncertain steps remain rejected by Task Engine.
                def resume_selected(
                    selected_id: str = task.id, selected_kind: str = task.kind,
                ) -> None:
                    tasks.resume(selected_id)
                    if selected_kind == "study_import_batch":
                        st.session_state["sl_import_batch_task_id"] = selected_id
                        launch_once(store.root, kind=selected_kind, task_id=selected_id)

                task_action(resume_selected, start_worker=task.kind != "study_import_batch")
            if task.state not in {"completed", "cancelled"} and columns[2].button(
                "取消任务（保留材料）",
                key=f"cancel_{task.id}",
            ):
                task_action(lambda task_id=task.id: tasks.request_cancel(task_id))
        if st.button("主动处理下一项等待任务", key="sl_run_next"):
            launch_once(store.root)
            st.info("已请求处理一项等待任务；没有定时器或自动排期。")
