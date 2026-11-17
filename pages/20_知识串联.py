"""知识串联入口：共享学习资产的离线知识星图。"""

from __future__ import annotations

import logging

import streamlit as st

from src import __version__
from src.knowledge_graph_editor_ui import render_selected_knowledge_editor
from src.knowledge_graph_generation_ui import render_graph_generation
from src.knowledge_graph_preview import load_preview, preview_graph
from src.knowledge_graph_service import KnowledgeGraphService
from src.knowledge_graph_ui import (
    apply_star_relation_command,
    render_graph_details,
    render_graph_editor,
    render_material_mining,
    render_pending_relation_removal,
)
from src.knowledge_graph_views import search_graph, subject_graph
from src.knowledge_mindmap_ui import render_mindmap_view
from src.knowledge_starmap_component import render_starmap
from src.knowledge_starmap_layout import StarmapLayoutStore
from src.runtime import application_database, application_settings
from src.study_loop.knowledge_import_ui import (
    render_preview_node_details,
    render_review_preview_confirmation,
)
from src.workspace_ui import render_workspace

LOGGER = logging.getLogger(__name__)
st.set_page_config(page_title=f"知识串联 · Nectivon v{__version__}", page_icon="🌌", layout="wide")
render_workspace("pages/20_知识串联.py")
st.title("知识串联")
st.caption("从已保存复盘提炼并确认知识，在二维导图与三维星图中整理、查看和修改。")
view_mode = st.radio("知识视图", ["二维思维导图", "知识星图"], horizontal=True,
                     key="graph_view_mode")

saved_node = st.session_state.pop("graph_saved_knowledge_node", None)
if saved_node:
    # Show the saved object's conversion even when the previous search excluded it.
    st.session_state["graph_query"] = ""
filters = st.columns([6, 1])
query = filters[0].text_input("查找节点", placeholder="输入题干或知识点关键词", key="graph_query")
if filters[1].button("刷新视图", use_container_width=True):
    st.rerun()
# Render low-frequency controls after the canvas, while retaining their values across reruns.
limit = st.session_state.get("graph_node_limit", 300)
try:
    database = application_database()
    graph = KnowledgeGraphService(database).snapshot(query=query, limit=limit)
except Exception as exc:
    LOGGER.exception("知识串联初始化失败")
    st.error(f"知识串联暂时无法读取本地知识：{exc}")
    st.stop()

preview_draft = None
try:
    preview_draft = load_preview(database, application_settings().runtime_dir)
except (OSError, ValueError, TypeError, KeyError) as exc:
    LOGGER.warning("候选图谱预览不可用：error_type=%s", type(exc).__name__)
    st.warning("候选预览暂时无法读取，知识库仍可正常浏览。")
if preview_draft is not None and st.toggle("查看待确认生成预览", value=True,
                                           key="graph_show_review_preview"):
    graph = search_graph(preview_graph(preview_draft, database), query)
    st.info("当前为真实复盘生成的待确认预览，知识点和连线均未写入知识库。"
            "可先查看两种图，核对后再确认导入；关系编辑在导入后使用。")
    with st.expander("核对候选并导入知识库"):
        render_review_preview_confirmation(database, application_settings().runtime_dir,
                                           preview_draft)
else:
    preview_draft = None

st.caption(f"当前 {graph.shown} / {graph.total} 个学习与知识节点 · "
           f"{len(graph.links)} 条关联（含来源）")
if graph.shown < graph.total:
    st.info("当前按类别显示部分节点。缩小关键词范围或提高上限可查看更多。")
if not graph.nodes:
    if query:
        st.info("没有匹配当前关键词的节点，请调整或清空搜索。")
    else:
        st.info("还没有可展示的知识。可从学习整理的已保存复盘导入，或在高级设置中手动新增。")
else:
    if saved_node in graph.nodes:
        st.session_state["graph_node_picker"] = saved_node
    if view_mode == "二维思维导图":
        subjects = graph.primary_subjects()
        if any(not node.get("tags") and node["entity_kind"] not in {"page", "document"}
               for node in graph.nodes.values()):
            subjects = (*subjects, "未归类")
        if subjects:
            if len(subjects) > 1:
                subject = st.selectbox("学科思维导图", subjects, key="graph_mindmap_subject")
            else:
                subject = subjects[0]
                st.caption(f"学科思维导图：{subject}")
            graph = subject_graph(graph, subject)
        else:
            subject = None
        render_mindmap_view(database, graph, subject=subject)
    else:
        st.caption("拖动旋转，滚轮缩放；点击星星查看详情。手势、语音与关系操作说明见高级设置。")
        settings = application_settings()
        star_layout = StarmapLayoutStore(settings.runtime_dir, database.database_path,
                                         library_uuid=database.get_knowledge_base_uuid())
        try:
            event = render_starmap(star_layout.payload(graph))
        except (OSError, ValueError, TypeError) as exc:
            LOGGER.warning("星图布局读取失败：error_type=%s", type(exc).__name__)
            st.warning("三维布局暂时无法读取，使用当前知识数据浏览；原布局文件已保留。")
            event = render_starmap(graph.star_payload())
        if isinstance(event, dict) and event != st.session_state.get("graph_last_star_event"):
            st.session_state["graph_last_star_event"] = event
            request = event.get("relation")
            if event.get("action") == "save_star_layout":
                try:
                    star_layout.save(event, graph)
                except (OSError, ValueError, TypeError) as exc:
                    st.error(f"三维布局未保存：{exc}")
                else:
                    st.session_state["graph_relation_notice"] = ("✅", "三维布局已保存。")
                    st.rerun()
            elif isinstance(request, dict):
                if graph.is_preview:
                    st.warning("候选关系尚未确认入库，请先核对并导入后再编辑正式关系。")
                else:
                    apply_star_relation_command(database, request)
            elif event.get("node_id") in graph.nodes:
                st.session_state["graph_node_picker"] = event["node_id"]
    identifiers = list(graph.nodes)
    # 断开反馈紧跟星图，避免被较长的节点正文推到页面深处。
    render_pending_relation_removal(database)
    if st.session_state.get("graph_node_picker") not in graph.nodes:
        st.session_state["graph_node_picker"] = identifiers[0]
    selected = st.selectbox("查看节点", identifiers,
                            format_func=lambda key: graph.nodes[key]["title"],
                            key="graph_node_picker")
    with st.container(key="graph_details"):
        if preview_draft is not None:
            render_preview_node_details(database, preview_draft, graph.nodes[selected])
        else:
            render_graph_details(database, graph, selected)
relation_notice = st.session_state.pop("graph_relation_notice", None)
if relation_notice:
    st.toast(relation_notice[1], icon=relation_notice[0])
mining_saved = st.session_state.pop("graph_mining_saved", None)
if mining_saved:
    st.success(mining_saved)
with st.expander("高级设置"):
    edit_tab, create_tab, graph_tab = st.tabs(["编辑当前知识", "新增与关联", "视图与星图生成"])
    with edit_tab:
        if preview_draft is not None:
            st.caption("当前预览尚未导入。请先核对候选并确认，之后可修改知识正文与关系。")
        else:
            render_selected_knowledge_editor(database, st.session_state.get("graph_node_picker"))
        selected_id = st.session_state.get("graph_node_picker", "")
        if selected_id.startswith("knowledge:"):
            from src.knowledge_graph_ui import _render_subject_links

            _render_subject_links(database, int(selected_id.split(":", 1)[1]))
    with create_tab:
        if preview_draft is not None:
            st.caption("候选预览中不编辑正式知识。确认导入后可新增、修改和连接知识点。")
        else:
            render_material_mining(database)
            render_graph_editor(database, graph)
    with graph_tab:
        st.selectbox("最多显示学习与知识节点", [300, 600, 1000], key="graph_node_limit")
        if preview_draft is None:
            render_graph_generation(database)
        else:
            st.caption("候选已经生成预览；确认导入后可进行正式星图数据的增量或全量生成。")
        st.caption("筛选与空间布局只改变查看方式；知识点与关系由两种视图共用。"
                   "二维画布支持拖动、平移、缩放、适应画布、分支折叠和保存恢复。")
        st.caption("拖动旋转、滚轮缩放；点击星星选择，再次点击打开详情。下方列表也能选择节点。"
               "连线展示关联，具体方向和类型见节点详情。")
        st.caption("星图内可选启用手势与语音控制：右下角「开启手势控制」隔空浏览（画面仅在本机处理）；"
               "底部控制条或按住空格说话——直接说出节点标题或内容的一部分（题干、知识点、页面文字均可）"
               "即可选中对应节点，也说「下一个星座」「回到全景」（首次使用需允许麦克风；"
               "语音识别由浏览器提供，Chrome/Edge 需联网；打字输入随时可用）。")
        st.caption("星图动作可直接改变知识点之间的关系：说或输入「连接 A 和 B」「A 派生自 B」"
               "（也支持 支持/矛盾/依赖/实例）即可连线；「断开 A 和 B」需在页面点击确认后才会删除。"
               "只说一个名字时，配对当前选中的星。手势操作：先用食指停留选中第一颗知识点星，"
               "比 👍 连接或 ✌️ 断开；收到锁定提示后松开手势，再选第二颗星并重复相同手势，"
               "最后在星图下方确认。")
