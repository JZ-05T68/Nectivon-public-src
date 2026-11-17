"""Explicit preview, validation and reversible graph projection controls."""

from __future__ import annotations

import logging

import streamlit as st

from src import runtime
from src.database import Database
from src.knowledge_graph_generation import GraphGenerationStore
from src.knowledge_graph_service import KnowledgeGraphService

LOGGER = logging.getLogger(__name__)


def render_graph_generation(database: Database) -> None:
    """Keep knowledge data updates separate from user layouts and renderer preferences."""
    settings = runtime.application_settings()
    store = GraphGenerationStore(settings.runtime_dir, database.database_path,
                                 library_uuid=database.get_knowledge_base_uuid())
    st.markdown("**星图数据生成**")
    st.caption("更新本地知识图数据；全量生成也会保留人工关系、二维/三维布局及手势和语音设置。")
    modes = {"incremental": "增量更新", "full": "全量重新生成"}
    mode = st.radio("更新方式", list(modes), format_func=modes.get,
                     key="graph_generation_mode", horizontal=True)
    key = f"graph_generation_preview_{store.scope}"
    try:
        existing = store.load()
        if existing:
            st.caption(f"上次生成：{existing['generated_at']} · "
                       f"{len(existing['nodes'])} 个节点、{len(existing['edges'])} 条关系"
                       " · 校验通过")
        if st.button("生成预览与校验", key="graph_generation_preview"):
            # Generation reads the complete library independently of the UI cap.
            graph = KnowledgeGraphService(database).generation_snapshot()
            st.session_state[key] = store.preview(graph, mode=mode)
        preview = st.session_state.get(key)
        if preview and preview["mode"] == mode:
            st.info(f"校验通过：{preview['node_count']} 个节点、{preview['edge_count']} 条关系。"
                    f"新增 {preview['added']}、更新 {preview['changed']}、"
                    f"移出 {preview['removed']} 个节点。")
            for warning in preview["warnings"]:
                st.warning(warning)
            if st.button("确认生成星图数据", key="graph_generation_commit", type="primary"):
                graph = KnowledgeGraphService(database).generation_snapshot()
                store.generate(graph, mode=mode, preview_token=preview["token"])
                st.session_state.pop(key, None)
                st.session_state["graph_relation_notice"] = (
                    "✅", "星图数据已生成，人工修改与布局已保留。",
                )
                st.rerun()
        if st.button("校验当前生成文件", key="graph_generation_validate"):
            st.success("生成文件校验通过。" if store.load() else "尚未生成文件，请先预览。")
        rollback = st.checkbox("恢复上一次生成的数据文件（保留知识库与布局）",
                                key="graph_generation_rollback_consent")
        if st.button("回滚生成文件", key="graph_generation_rollback", disabled=not rollback):
            store.rollback()
            st.session_state.pop(key, None)
            st.success("已恢复上一次有效数据文件。当前知识库与人工布局保持完整。")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        LOGGER.warning("星图生成未完成：error_type=%s", type(exc).__name__)
        st.error(f"星图生成未完成：{exc}。知识库、人工关系与布局已保留。")
