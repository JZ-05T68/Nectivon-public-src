"""Small view-only integration for the existing knowledge connection page."""

from __future__ import annotations

import logging

import streamlit as st

import src.runtime as runtime
from src.database import Database
from src.knowledge_graph_service import KnowledgeGraph
from src.knowledge_mindmap import MindmapLayoutStore, mindmap_payload
from src.knowledge_mindmap_component import render_mindmap

LOGGER = logging.getLogger(__name__)


def render_mindmap_view(
    database: Database, graph: KnowledgeGraph, *, subject: str | None = None,
) -> None:
    """Consume selections or an explicit canvas save, never relationship edits."""
    settings = runtime.application_settings()
    options = {"graph_id": f"knowledge-subject:{subject}"} if subject else {}
    if getattr(graph, "is_preview", False):
        options["graph_id"] = f"preview:{subject or 'all'}:{','.join(sorted(graph.nodes))}"
    store = MindmapLayoutStore(settings.runtime_dir, database.database_path,
                              library_uuid=database.get_knowledge_base_uuid(), **options)
    layout = store.load()
    if layout.warning:
        st.warning(layout.warning)
    st.caption("拖动节点调整位置，滚轮缩放；点击“保存布局”后可恢复。点击节点查看内容与来源。")
    try:
        event = render_mindmap(mindmap_payload(graph, layout, store.scope))
        last_key = f"graph_last_mindmap_event_{store.scope}"
        if not isinstance(event, dict) or event.get("event_id") == st.session_state.get(last_key):
            return
        st.session_state[last_key] = event.get("event_id")
        if event.get("scope") != store.scope:
            st.warning("画布所属资料库已变化，请刷新页面。")
            return
        if event.get("action") == "select":
            if event.get("node_id") in graph.nodes:
                st.session_state["graph_node_picker"] = event["node_id"]
            else:
                st.warning("所选节点已不在当前视图，请重新选择。")
        elif event.get("action") == "save":
            store.save(event, graph)
            st.rerun()
    except (OSError, ValueError, TypeError) as exc:
        LOGGER.warning("二维导图布局未保存：error_type=%s", type(exc).__name__)
        st.error("二维导图布局读取或保存失败。知识内容与关系未改动；"
                 "请检查候选目录写入权限或重新加载后再保存。")
