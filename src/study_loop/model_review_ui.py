"""Optional, explicit model review alongside the existing manual attempt workflow."""

from __future__ import annotations

from uuid import uuid4

import streamlit as st

from src.ai.model_registry import CapabilitySupport, get_capability_profile
from src.ai.provider_scope import ACTIVE_MODELS

from .display import render_saved_text
from .model_review_records import (
    CALL_STAGES,
    STAGES,
    list_runs,
    load_record,
    pending_record_names,
    record_digest,
)
from .models import Draft, Question, StudyError
from .runtime import launch_once, submit
from .storage import StudyStore

LABELS = {"deepseek": "DeepSeek", "qwen": "通义千问"}
SUMMARY_LABELS = {
    "overview": "题目复盘概况",
    "method_comparison": "方法、步骤和物理意义比较",
    "user_result_observation": "用户实际作答观察",
    "disagreements": "模型分歧",
    "evidence_limits": "证据边界与信息不足",
    "priority_review": "优先复习建议",
    "next_checks": "需要人工核验的项目",
}


def _read_history_record(store: StudyStore, run_id: str, stage: str) -> dict | None:
    """A bad local result must not prevent access to task cancellation controls."""
    try:
        record = load_record(store, run_id, stage)
        if record is None:
            return None
        if stage in CALL_STAGES or stage.endswith("_raw"):
            if type(record.get("text")) is not str:
                raise StudyError("原始正文格式无效。")
        if stage == "report" and (
            type(record.get("call_stages")) is not list
            or any(type(s) is not str or s not in CALL_STAGES for s in record["call_stages"])
        ):
            raise StudyError("阶段清单格式无效。")
        return record
    except (StudyError, OSError):
        st.warning(f"{stage} 记录无法读取，文件保留；其他回答和任务控制仍可使用。")
        return None


def render_model_review(
    store: StudyStore,
    session_id: str,
    item: Draft,
    unit: Question,
    attempt_prefix: str,
) -> None:
    """No model calls on rerun; submission, history display and downloads are explicit."""
    key = f"sl_mr_{session_id}_{item.id}_{unit.id}"
    st.subheader("可选模型复盘")
    mode = st.radio(
        "复盘方式", ["手动复盘", "单模型复盘", "双模型协同复盘"], horizontal=True, key=key + "_mode"
    )
    if mode != "手动复盘":
        providers = (
            ["deepseek", "qwen"]
            if mode == "双模型协同复盘"
            else [
                st.selectbox(
                    "供应商",
                    ["deepseek", "qwen"],
                    format_func=LABELS.get,
                    key=key + "_provider",
                )
            ]
        )
        selections = [
            {
                "provider": provider,
                "model": st.selectbox(
                    LABELS[provider] + " 型号",
                    ACTIVE_MODELS[provider],
                    key=key + "_" + provider,
                ),
            }
            for provider in providers
        ]
        input_mode = st.radio(
            "本次发送的题目形式", ["原页图和题干", "仅已核对完整的题干文本"], key=key + "_input"
        )
        text_complete = False
        if input_mode != "原页图和题干":
            text_complete = st.checkbox(
                "我已核对所有图中条件均完整包含在题干文本中，本次不需要模型读取图片",
                key=key + "_text_complete",
            )
        images = [p for p in store.pages(item.source_id) if p.id in item.page_ids and p.image]
        image_problem = input_mode == "原页图和题干" and (
            len(images) != 1
            or any(
                get_capability_profile(s["provider"], s["model"]).effective.vision
                is not CapabilitySupport.SUPPORTED
                for s in selections
            )
        )
        if image_problem:
            st.warning("本轮需要恰好一张原页图且所有所选模型都支持图片；DeepSeek Pro 不支持读图。")
        retries = st.selectbox("每次调用允许的传输重试次数", [0, 1, 2], key=key + "_retries")
        calls = 5 if len(selections) == 2 else 2
        st.caption(
            f"最多 {calls} 次模型调用，每次最多 {1 + retries} 次传输尝试。"
            f"最终由 {LABELS[providers[-1]]} 所选型号整理总结。"
            "网络失败可能已计费；无语义重试、无投票。取消会阻止后续调用，"
            "已经发送的请求须等返回或超时，不能撤销供应商计费。"
        )
        real = st.checkbox(
            "此题来自真实教材、试卷或可追溯错题/好题，题干条件和小问已核对", key=key + "_real"
        )
        consent = st.checkbox(
            "同意将本题所选文本/页图及本次实际作答记录发送给以上供应商进行分析、必要核验和总结",
            key=key + "_consent",
        )
        submitted = st.session_state.get(key + "_submitted")
        if submitted:
            st.info("此轮已提交，刷新或重复点击不会创建新收费任务。进度见任务区。")
            if st.button("明确开始另一轮模型复盘（可能重复计费）", key=key + "_another"):
                st.session_state.pop(key + "_submitted", None)
                st.session_state.pop(key + "_request_id", None)
                st.rerun()
        if st.button(
            "主动发起本题模型复盘",
            key=key + "_submit",
            disabled=not real
            or not consent
            or image_problem
            or bool(submitted)
            or (input_mode != "原页图和题干" and not text_complete),
        ):
            # Reuse identity even if enqueue committed but the response/worker
            # launch failed; a second click must not create another paid run.
            run_id = st.session_state.setdefault(key + "_request_id", uuid4().hex)
            payload = {
                "run_id": run_id,
                "session_id": session_id,
                "item_id": item.id,
                "unit_id": unit.id,
                "selections": selections,
                "input_mode": "image" if input_mode == "原页图和题干" else "text",
                "allow_remote": consent,
                "real_material_confirmed": real,
                "text_complete": text_complete,
                "max_extra_attempts": retries,
            }
            from nectivon_tasks import TaskError

            try:
                submit(store.root, "study_model_review", payload, run_id)
            except TaskError as exc:
                st.warning(str(exc) + " 请求标识已保留；未自动重复提交，请核对任务区。")
            else:
                st.session_state[key + "_submitted"] = run_id
                try:
                    launch_once(store.root)
                except OSError:
                    st.warning("任务已入队但 worker 启动失败；请在任务区主动处理，不要重复提交。")
                else:
                    st.info("已提交本题复盘。进度、暂停和取消见下方任务区；已有原始回答保留。")
    try:
        runs, warnings = list_runs(store, item.id, unit.id)
    except (StudyError, OSError):
        runs, warnings = [], ["模型记录目录暂时无法读取；文件保留，请在任务区控制现有任务。"]
    for warning in warnings:
        st.warning(warning)
    if not runs:
        return
    selected = st.selectbox(
        "已保存的模型复盘",
        runs,
        format_func=lambda r: r["created_at"],
        key=key + "_history",
    )
    if st.button("查看此轮原始回答与总结", key=key + "_show"):
        st.session_state[key + "_visible"] = selected["run_id"]
    if st.session_state.get(key + "_visible") != selected["run_id"]:
        return
    if st.button("收起本轮模型复盘", key=key + "_hide"):
        st.session_state.pop(key + "_visible", None)
        st.rerun()
    st.session_state[attempt_prefix + "_shown"] = True
    run_id = selected["run_id"]
    bundle = {stage: _read_history_record(store, run_id, stage) for stage in sorted(STAGES)}
    report = bundle["report"]
    failure = bundle["failure"]
    integrity_issues = []
    for stage in CALL_STAGES:
        processed, original = bundle[stage], bundle[stage + "_raw"]
        if (
            processed
            and processed.get("raw_record_sha256")
            and (original is None or record_digest(original) != processed["raw_record_sha256"])
        ):
            integrity_issues.append(stage)
        if report and stage in report.get("call_stages", []) and processed is None:
            integrity_issues.append(stage)
    if integrity_issues:
        st.warning("原答记录缺失或摘要不匹配，暂不展示统一结论：" + "、".join(integrity_issues))
    if failure:
        st.warning("此前保存的失败说明：" + str(failure.get("message", "信息不足")))
    try:
        pending = pending_record_names(store, run_id)
        if pending:
            st.warning("发现保留的暂存副本，未自动恢复或删除：" + "、".join(pending))
    except (StudyError, OSError):
        st.warning("暂存记录清单暂时无法读取，请保留目录并人工核对。")
    if report and not integrity_issues and isinstance(report.get("summary"), dict):
        for field, label in SUMMARY_LABELS.items():
            value = report["summary"].get(field, "信息不足")
            st.markdown("**" + label + "**")
            render_saved_text("\n".join(map(str, value)) if isinstance(value, list) else str(value))
    else:
        st.info("本轮总结尚未形成或信息不足；可查看已经保存的原始回答及调用状态。")
    for stage in ("analysis_0", "analysis_1", "verify_0", "verify_1", "summary"):
        processed = bundle[stage]
        original = bundle[stage + "_raw"]
        record = original or processed
        if record is None:
            continue
        stage_label = {
            "analysis_0": "第一份分析", "analysis_1": "第二份分析",
            "verify_0": "第一份核验", "verify_1": "第二份核验", "summary": "总结",
        }[stage]
        with st.expander(stage_label):
            if bundle[stage] is None:
                st.warning("本阶段只保存了原始响应，后续处理未完成；不能视为有效结论。")
            if (
                processed
                and processed.get("raw_record_sha256")
                and (original is None or record_digest(original) != processed["raw_record_sha256"])
            ):
                st.warning("原始副本缺失或摘要不匹配，请保留文件并人工核对，不能信任处理后结论。")
            if (processed or record).get("error_message"):
                st.warning((processed or record)["error_message"])
            text = record.get("text") or "本次没有取得可保存的回答正文。"
            display_text = str(text).encode("utf-8", errors="backslashreplace").decode("utf-8")
            render_saved_text(display_text)
