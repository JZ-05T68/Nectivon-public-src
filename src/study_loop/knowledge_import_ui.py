"""Saved-review import, inspectable candidates and exact return navigation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import streamlit as st

from src import runtime
from src.ai.provider import AIError
from src.database import Database, DatabaseError
from src.knowledge_mining_service import KnowledgeMiningAIService, KnowledgeMiningError
from src.knowledge_object_service import KnowledgeObjectService
from src.study_knowledge_bridge import (
    SavedReviewKnowledgeDraft,
    StudyKnowledgeBridge,
    knowledge_review_refs,
)

from .display import render_saved_text
from .models import StudyError
from .storage import StudyStore


def render_review_knowledge_import(
    store: StudyStore, document: dict, version: dict, *, location: str,
) -> None:
    """Import only the supplied saved revision; never inspect the live editor body."""
    key = f"sl_knowledge_import_{location}_{version['id']}_"
    # Do not initialize platform runtime merely by browsing an old review.
    if st.button("导入知识串联", key=key + "open"):
        st.session_state[key + "opened"] = True
    if not st.session_state.get(key + "opened"):
        return
    try:
        database = runtime.application_database()
        bridge = StudyKnowledgeBridge(database, store)
        imports = bridge.imports_for_review(document["id"])
        current = next((ref for ref in imports
                        if ref["study_review_revision_id"] == version["id"]), None)
        if current:
            st.success(f"第{version['number']}版已导入，重复操作不会新增节点。")
            _render_imported_nodes(database, current["knowledge_ids"])
            st.page_link("pages/20_知识串联.py", label="打开二维思维导图 / 三维知识星图")
            return
        st.caption(f"仅提炼已保存的第{version['number']}版；未保存的编辑内容不会参与。")
        st.caption("独立学习库来源会保留原题、材料与页码；未建立平台资料身份映射，"
                   "不会按标题或相似文字自动绑定。")
        allowed = st.checkbox("允许将这个已保存版本发送给当前配置的 AI 提炼一次",
                              key=key + "allow")
        if st.button("生成知识候选", disabled=not allowed, key=key + "generate"):
            from src.ai.knowledge_generation import configured_knowledge_generation_provider

            provider = configured_knowledge_generation_provider()
            if provider is None:
                raise StudyError("AI 尚未启用，请先在设置中选择可用模型；已保存复盘不受影响。")
            with st.spinner("正在提炼已保存复盘…"):
                st.session_state[key + "draft"] = bridge.mine_saved_review(
                    version["id"], KnowledgeMiningAIService(provider),
                )
        draft = st.session_state.get(key + "draft")
        if draft is None:
            return
        if _render_candidate_confirmation(database, bridge, draft, key=key,
                                          has_history=bool(imports)):
            st.session_state.pop(key + "draft", None)
            st.rerun()
    except (StudyError, KnowledgeMiningError, AIError, DatabaseError, OSError) as exc:
        st.warning(str(exc))


def _render_candidate_confirmation(
    database: Database, bridge: StudyKnowledgeBridge, draft: SavedReviewKnowledgeDraft, *,
    key: str, has_history: bool,
) -> bool:
    """Share one inspection and service-backed confirmation path for all candidate previews."""
    if draft.mining.truncated:
        st.warning("复盘超过单次提炼范围，本次只处理前段，未声称完整覆盖。")
    selected_points = []
    bindings = {}
    existing = KnowledgeObjectService(database).list(lifecycle="active", limit=500)
    options = {obj.id: obj for obj in existing}
    if has_history:
        st.info("这是同一复盘的新版本，请逐项选择复用已有节点或明确新增。"
                "复用只增加来源版本关联，不覆盖已有知识正文。")
    for index, point in enumerate(draft.mining.points):
        with st.container(border=True):
            selected = st.checkbox(point.title, value=True, key=key + f"point_{index}")
            st.caption("复盘明确写出" if point.origin == "explicit" else "AI 推导候选")
            render_saved_text(point.content)
            st.caption("复盘依据：" + point.evidence)
            if selected:
                selected_points.append(point)
                binding = st.selectbox(
                    "保存方式", [-1, None, *options], key=key + f"binding_{index}",
                    index=0 if has_history else 1,
                    format_func=lambda identity: (
                        "请选择保存方式" if identity == -1 else "新增知识点" if identity is None
                        else f"复用：{options[identity].title}（{identity}）"
                    ),
                )
                if binding != -1:
                    bindings[point.title] = binding
    titles = {point.title for point in selected_points}
    selected_relations = []
    st.markdown("**候选关系与依据**")
    for index, relation in enumerate(draft.mining.relations):
        if relation.source_title not in titles or relation.target_title not in titles:
            continue
        if st.checkbox(
            f"{relation.source_title} → {relation.target_title} · {relation.relation_type.label}",
            value=True, key=key + f"relation_{index}",
        ):
            selected_relations.append(relation)
        st.caption(relation.description)
    if not draft.mining.relations:
        st.caption("没有可靠关系依据的知识点将独立保存，不按关键词强行连线。")
    confirmed = st.checkbox("我已核对所选知识点、推导内容和关系，确认写入知识库",
                            key=key + "confirm")
    if st.button("确认导入知识串联", key=key + "save",
                 disabled=not confirmed or not selected_points
                 or len(bindings) != len(selected_points)):
        result = bridge.confirm(
            replace(draft, mining=replace(draft.mining, points=tuple(selected_points),
                                          relations=tuple(selected_relations))),
            user_confirmed=confirmed, bindings=bindings,
        )
        st.session_state["graph_saved_knowledge_node"] = f"knowledge:{result.knowledge_ids[0]}"
        return True
    return False


def render_graph_review_preview(database: Database, runtime_dir: Path) -> None:
    """Inspect a persisted visual preview and promote only explicit user selections.

    No provider is bound and no model is called. Merely rendering this panel
    never creates formal knowledge, including after an app restart.
    """
    from src.knowledge_graph_preview import load_preview

    try:
        draft = load_preview(database, runtime_dir)
        if draft is None:
            return
        render_review_preview_confirmation(database, runtime_dir, draft)
    except (StudyError, KnowledgeMiningError, AIError, DatabaseError, OSError,
            ValueError, KeyError, TypeError) as exc:
        st.warning("复盘知识预览暂不能确认：" + str(exc))


def render_review_preview_confirmation(
    database: Database, runtime_dir: Path, draft: SavedReviewKnowledgeDraft,
) -> None:
    """Confirm a displayed persisted draft through the same inspected-import service."""
    from src.knowledge_graph_preview import clear_preview, preview_identity

    try:
        source = draft.source
        candidate_id = preview_identity(draft)
        key = f"graph_review_preview_{candidate_id}_"
        st.markdown("**检查复盘知识候选并确认**")
        st.caption(f"{source['course_id'] or '未分类'} · 已保存第{source['revision_number']}版。"
                   "两种图中的预览内容尚未写入正式知识库；确认前不会成为正式知识关系。")
        store = StudyStore(Path(source["study_root"]))
        bridge = StudyKnowledgeBridge(database, store)
        verified, _ = bridge.saved_source(source["study_review_revision_id"])
        if verified != source:
            raise StudyError("预览对应的来源身份已变化，请重新核对已保存复盘。")
        imports = bridge.imports_for_review(source["study_review_id"])
        current = next((ref for ref in imports if ref["study_review_revision_id"]
                        == source["study_review_revision_id"]), None)
        if current:
            st.success("这个复盘版本已导入，预览不会重复新增知识。")
            _render_imported_nodes(database, current["knowledge_ids"])
            if st.button("关闭已导入预览", key=key + "close_imported"):
                clear_preview(database, runtime_dir, candidate_id)
                st.rerun()
            return
        if _render_candidate_confirmation(database, bridge, draft, key=key,
                                          has_history=bool(imports)):
            # Keep saved preview history recoverable. Only its active pointer
            # is cleared once the same atomic bridge confirmation succeeds.
            clear_preview(database, runtime_dir, candidate_id)
            st.session_state["graph_mining_saved"] = "知识候选已确认入库，两种图已同步到正式节点。"
            st.rerun()
    except (StudyError, KnowledgeMiningError, AIError, DatabaseError, OSError,
            ValueError, KeyError, TypeError) as exc:
        st.warning("复盘知识预览暂不能确认：" + str(exc))


def render_preview_node_details(
    database: Database, draft: SavedReviewKnowledgeDraft, node: dict,
) -> None:
    """Read an unconfirmed visual node and its saved source, without formal graph writes."""
    source = draft.source
    index = node.get("meta", {}).get("preview_point_index")
    if not isinstance(index, int) or not 0 <= index < len(draft.mining.points):
        # An exact title in this one validated candidate is an internal lookup,
        # never an identity binding to an existing knowledge object or source.
        index = next((i for i, item in enumerate(draft.mining.points)
                      if item.title == node.get("title")), None)
    if index is None:
        st.warning("预览节点与当前候选不一致，请重新选择。")
        return
    point = draft.mining.points[index]
    st.subheader(point.title.replace("![", "&#33;["))
    st.caption("AI 候选 · 尚未确认入库 · "
               + ("复盘明确写出" if point.origin == "explicit" else "根据复盘推导"))
    render_saved_text(point.content)
    st.caption("复盘依据：" + point.evidence)
    st.caption(f"{source['course_id'] or '未分类'} · 已保存复盘第{source['revision_number']}版")
    for material in source["materials"]:
        st.caption("原材料：" + material["filename"])
    st.info("预览只用于检查布局、知识点和候选关系。人工内容编辑与语音/手势关系写入"
            "需先在高级设置确认所选候选；未确认前不会改变正式知识。")
    with st.expander("查看已保存复盘、原题与原始材料"):
        try:
            from .ai_review import AIReviewService
            from .import_ui import render_source

            store = StudyStore(Path(source["study_root"]))
            verified, content = StudyKnowledgeBridge(database, store).saved_source(
                source["study_review_revision_id"],
            )
            if verified != source:
                raise StudyError("预览来源身份已变化，请重新核对。")
            st.markdown("**已保存复盘正文**")
            render_saved_text(content)
            reviews = AIReviewService(store)
            document = reviews.document(source["study_review_id"])
            context = reviews.chats.conversation(document["conversation_id"])["context"]
            for selected_index, entry in enumerate(context["review_items"]):
                item = entry["item"]
                st.markdown("**原题（该复盘的材料快照）**")
                render_saved_text(item["question"]["content"])
                render_source(
                    store, item["source_id"], item["page_ids"],
                    prefix=f"preview_source_{source['study_review_revision_id']}_{selected_index}",
                )
        except (StudyError, OSError, ValueError) as exc:
            st.warning("原题或材料暂不可用：" + str(exc))


def _render_imported_nodes(database: Database, identities: list[int]) -> None:
    for identity in identities:
        obj = KnowledgeObjectService(database).get(identity)
        st.caption(obj.title if obj else f"知识点 {identity} 当前不可用，来源历史保留。")


def render_study_review_sources(database: Database, knowledge_id: int) -> None:
    """Navigate a knowledge node to its exact review revision and original question."""
    refs = knowledge_review_refs(database, knowledge_id)
    if not refs:
        return
    st.markdown("**单题复盘来源与版本**")
    for index, ref in enumerate(refs):
        st.caption(f"{ref['course_id'] or '未分类'} · 已保存第{ref['revision_number']}版 · "
                   + ("复盘明确写出" if ref["origin"] == "explicit" else "由复盘推导"))
        for material in ref["materials"]:
            pages = [str(page["page_number"]) for page in ref["source_page_refs"]
                     if page["material_id"] == material["material_id"]
                     and page["page_number"] is not None]
            st.caption("原材料：" + material["filename"]
                       + (" · 第" + "、".join(pages) + "页" if pages else " · 无物理页码"))
        with st.expander(f"查看原题与原始材料（复盘第{ref['revision_number']}版）"):
            try:
                from .ai_review import AIReviewService
                from .import_ui import render_source

                source_store = StudyStore(Path(ref["study_root"]))
                verified, _ = StudyKnowledgeBridge(database, source_store).saved_source(
                    ref["study_review_revision_id"],
                )
                if verified["content_sha256"] != ref["content_sha256"]:
                    raise StudyError("来源版本指纹不一致，未展示其他材料。")
                reviews = AIReviewService(source_store)
                document = reviews.document(ref["study_review_id"])
                context = reviews.chats.conversation(document["conversation_id"])["context"]
                for selected_index, entry in enumerate(context["review_items"]):
                    item = entry["item"]
                    st.markdown("**原题（该复盘的材料快照）**")
                    render_saved_text(item["question"]["content"])
                    render_source(
                        source_store, item["source_id"], item["page_ids"],
                        prefix=f"graph_review_source_{knowledge_id}_{index}_{selected_index}",
                    )
            except (StudyError, OSError) as exc:
                st.warning("原题或材料暂不可用：" + str(exc))
        if st.button("返回此版复盘与原题", key=f"graph_study_review_{knowledge_id}_{index}"):
            try:
                store = StudyStore(Path(ref["study_root"]))
                source, _ = StudyKnowledgeBridge(database, store).saved_source(
                    ref["study_review_revision_id"],
                )
                if source["content_sha256"] != ref["content_sha256"]:
                    raise StudyError("来源版本指纹不一致，未跳转到其他材料。")
                from .ai_review_ui import open_ai_review

                previous_root = st.session_state.get("sl_open_root")
                if previous_root and Path(previous_root).resolve() != store.root:
                    # Avoid applying an active editor identity from another library
                    # to this source library. Persist its existing local working
                    # draft before using the same explicit library-switch semantics.
                    from .ai_review_ui import persist_ai_review

                    if not persist_ai_review(StudyStore(Path(previous_root))):
                        raise StudyError("当前学习库的编辑尚未同步，请先返回保存后再切换来源库。")
                    for key in list(st.session_state):
                        if key.startswith("sl_"):
                            del st.session_state[key]
                st.session_state["sl_open_root"] = str(store.root)
                open_ai_review(ref["questions"][0]["item_id"], ref["study_review_id"],
                               ref["study_review_revision_id"])
                st.switch_page("pages/18_学习整理.py")
            except (StudyError, OSError) as exc:
                st.warning("无法关联原复盘：" + str(exc))
