"""Material-bound AI review manuscripts with durable local editing and explicit generation."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import streamlit as st

from src.ai.provider import AIError, CompletionProvider
from src.ai.study_chat_adapter import StudyChatProvider, configured_chat_model
from src.time_display import format_beijing_time

from .chat_citations import visible_answer
from .color_editor import (
    confirmed_color_body,
    deactivate_color_editor,
    defer_active_color_action,
    defer_color_action,
    render_color_editor,
)
from .display import render_saved_text
from .models import StudyError
from .storage import StudyStore

if TYPE_CHECKING:
    from .ai_review import AIReviewService

PREFIX = "sl_ai_review_"


def review_service(store: StudyStore) -> AIReviewService:
    """Resolve local storage without making the model a startup dependency."""
    from .ai_review import AIReviewService

    return AIReviewService(store)


def model_provider() -> CompletionProvider:
    """Bind the configured model only after an explicit generation action."""
    return StudyChatProvider(configured_chat_model())


def open_ai_review(
    item_id: str, document_id: str | None = None, version_id: str | None = None
) -> None:
    """Request a focused editor without creating a model request or changing materials."""
    st.session_state[PREFIX + "return_tab"] = st.session_state.get(
        "sl_main_tab", "待确认材料"
    )
    st.session_state[PREFIX + "focus"] = True
    st.session_state[PREFIX + "requested"] = {
        "kind": "document" if document_id else "item",
        "id": document_id or item_id,
        "version_id": version_id,
    }


def ai_review_entry(store: StudyStore, item_id: str, *, key: str) -> None:
    """Open this exact imported question, including a readable unconfirmed draft."""
    st.button("AI复盘", key=key, on_click=open_ai_review, args=(item_id,))


def _active(service: AIReviewService) -> dict | None:
    identity = st.session_state.get(PREFIX + "document_id")
    return service.document(identity) if identity else None


def _dirty(service: AIReviewService, document: dict | None) -> bool:
    if not document:
        return False
    saved = document.get("last_saved_version_id")
    baseline = (service.version(saved)["content"] if saved else
                document.get("initial_working", visible_answer(document.get("raw_ai", ""))))
    return document["working"] != baseline


def _cache(document: dict) -> None:
    identity = document["id"]
    st.session_state[PREFIX + "body_" + identity] = document["working"]
    st.session_state[PREFIX + "revision_" + identity] = document["working_revision"]
    st.session_state[PREFIX + "loaded_" + identity] = document["working"]


def _activate(service: AIReviewService, document: dict) -> None:
    service.remember_view(document["id"], document.get("viewed_version_id"))
    st.session_state["sl_review_route_guard"] = persist_review_route
    st.session_state[PREFIX + "document_id"] = document["id"]
    st.session_state[PREFIX + "item_id"] = document["item_ids"][0]
    st.session_state[PREFIX + "source_id"] = document["source_ids"][0]
    st.session_state[PREFIX + "material"] = document["source_ids"][0]
    st.session_state[PREFIX + "question"] = document["item_ids"][0]
    st.session_state[PREFIX + "requirements"] = document.get("requirements", "")
    if document.get("viewed_version_id"):
        st.session_state[PREFIX + "selected_version"] = document["viewed_version_id"]
    else:
        st.session_state.pop(PREFIX + "selected_version", None)
    st.session_state[PREFIX + "view_" + document["id"]] = "阅读"
    _cache(document)


def _persist(store: StudyStore, *, user_edit: bool = False,
             document_id: str | None = None) -> bool:
    """Persist the editor before navigation; retain concurrent input as a separate local draft."""
    service = review_service(store)
    active = _active(service)
    document = service.document(document_id) if document_id else active
    if not document:
        return True
    identity = document["id"]
    key = PREFIX + "body_" + identity
    if key not in st.session_state:
        return True
    content = st.session_state[key]
    expected = st.session_state.get(PREFIX + "revision_" + identity)
    if (not user_edit and content == st.session_state.get(PREFIX + "loaded_" + identity)
            and expected != document["working_revision"]):
        # A read-view cache is not a new edit. Another tab can advance the draft;
        # only an actual input callback may conflict or create a recovery copy.
        _cache(document)
        return True
    if content == document["working"]:
        st.session_state[PREFIX + "revision_" + identity] = document["working_revision"]
        return True
    try:
        updated = service.update_working(identity, content, expected_revision=expected)
    except StudyError as exc:
        if expected is not None and expected != document["working_revision"]:
            try:
                recovered = service.fork_working(identity, content)
            except StudyError as recovery_error:
                st.session_state[PREFIX + "error"] = str(recovery_error)
                return False
            if active and active["id"] == identity:
                _activate(service, recovered)
            st.session_state[PREFIX + "flash"] = "本次输入已保留为独立草稿。"
        else:
            st.session_state[PREFIX + "error"] = str(exc)
            return False
    else:
        # The editor widget already owns its text; only advance the optimistic revision.
        st.session_state[PREFIX + "revision_" + identity] = updated["working_revision"]
        st.session_state[PREFIX + "loaded_" + identity] = content
    return True


def persist_ai_review(store: StudyStore) -> bool:
    """Flush local review input before a learning-route button selects another screen."""
    return _persist(store)


def _finish_view_change(store: StudyStore, identity: str, target: str) -> None:
    st.session_state[PREFIX + "view_" + identity] = target
    _view_changed(store, identity, flushed=True)


def _view_changed(store: StudyStore, identity: str, *, flushed: bool = False) -> None:
    """Continue the latest saved manuscript in its retained working draft."""
    target = st.session_state.get(PREFIX + "view_" + identity)
    if (not flushed and target != "编辑"
            and st.session_state.get(PREFIX + "editor_live_" + identity)
            and defer_color_action(
            PREFIX + "body_" + identity, _finish_view_change, args=(store, identity, target))):
        st.session_state[PREFIX + "view_" + identity] = "编辑"
        return
    if not _persist(store):
        return
    service = review_service(store)
    document = service.document(identity)
    if (st.session_state.get(PREFIX + "view_" + identity) == "编辑"
            and st.session_state.get(PREFIX + "selected_version")
            == document.get("last_saved_version_id")):
        st.session_state.pop(PREFIX + "selected_version", None)
        service.remember_view(identity)


def _page_after_flush(target: str) -> None:
    st.session_state[PREFIX + "page_after_flush"] = target


def _pending_page_editor() -> dict | None:
    """Keep editor identity outside widgets while a cross-page acknowledgement is pending."""
    active_key = st.session_state.get("sl_active_color_editor")
    memory_route = st.session_state.get("sl_memory_editor_route")
    if isinstance(memory_route, dict) and memory_route.get("body_key") == active_key:
        return dict(memory_route)
    identity = st.session_state.get(PREFIX + "document_id")
    key = PREFIX + "body_" + identity if isinstance(identity, str) else None
    if not key or active_key != key:
        return None
    return {"document_id": identity, "body_key": key,
            "focused": bool(st.session_state.get(PREFIX + "focus"))}


def _remount_page_editor(binding: dict) -> bool:
    """Restore only view metadata, never replace the pending editor's canonical input."""
    root = binding.get("study_root", st.session_state.get("sl_open_root"))
    if not isinstance(root, str):
        return False
    from pathlib import Path

    store = StudyStore(Path(root))
    if binding.get("kind") == "memory":
        from .memory_editor_ui import _prefix, render_memory_editor

        prefix = _prefix(store, binding["memory_id"], binding["key_prefix"])
        if binding["body_key"] != prefix + "_body":
            raise StudyError("待同步的记忆入口不一致，输入仍保留，未切换页面。")
        st.session_state[prefix + "_mode"] = "编辑"
        if binding["key_prefix"] == "sl_memory_home":
            st.session_state["sl_memory_home_selection"] = binding["memory_id"]
        st.caption("正在同步编辑输入…")
        render_memory_editor(store, binding["memory_id"], key_prefix=binding["key_prefix"])
        return True
    document = review_service(store).document(binding["document_id"])
    identity = document["id"]
    st.session_state[PREFIX + "document_id"] = identity
    st.session_state[PREFIX + "item_id"] = document["item_ids"][0]
    st.session_state[PREFIX + "source_id"] = document["source_ids"][0]
    st.session_state[PREFIX + "material"] = document["source_ids"][0]
    st.session_state[PREFIX + "question"] = document["item_ids"][0]
    st.session_state[PREFIX + "focus"] = binding["focused"]
    if not binding["focused"]:
        st.session_state["sl_main_tab"] = "主动复盘"
    st.session_state.pop(PREFIX + "selected_version", None)
    st.session_state[PREFIX + "view_" + identity] = "编辑"
    st.session_state[PREFIX + "editor_live_" + identity] = True
    return False


def persist_review_route(current: str) -> None:
    """Capture page departure and rehydrate from SQLite before stale page widgets can return."""
    route_key = PREFIX + "page_route"
    previous = st.session_state.get(route_key, "pages/18_学习整理.py")
    learning = "pages/18_学习整理.py"
    destination = st.session_state.pop(PREFIX + "page_after_flush", None)
    if destination:
        st.session_state.pop(PREFIX + "page_editor_binding", None)
        if current == learning:
            st.session_state[route_key] = destination
            st.switch_page(destination)
    if previous == learning and current != learning:
        binding = _pending_page_editor()
        if defer_active_color_action(_page_after_flush, args=(current,)):
            if binding:
                st.session_state[PREFIX + "page_editor_binding"] = binding
            # Do not let destination-widget cleanup turn the return radio into
            # a read view: that would unmount the only frame able to acknowledge.
            st.switch_page(learning)
    binding = st.session_state.get(PREFIX + "page_editor_binding")
    if binding and current == learning:
        if st.session_state.get(binding["body_key"] + "_color_pending"):
            if _remount_page_editor(binding):
                st.stop()
        else:
            st.session_state.pop(PREFIX + "page_editor_binding", None)
    if previous == current:
        return
    root = st.session_state.get("sl_open_root")
    if not root:
        return
    from pathlib import Path

    store = StudyStore(Path(root))
    # text_area's change callback owns writes. A different page may replay an
    # old widget value, so routing must never treat that replay as a new edit.
    if current == learning:
        service = review_service(store)
        document = _active(service) or service.resume_document()
        if document and not document.get("deleted_at"):
            _activate(service, document)
    st.session_state[route_key] = current


def _apply(store: StudyStore, action: dict) -> None:
    service = review_service(store)
    kind = action["kind"]
    if kind == "generate":
        st.session_state[PREFIX + "execute"] = action
    elif kind == "document":
        document = service.document(action["id"])
        if action.get("version_id"):
            document = {**document, "viewed_version_id": action["version_id"]}
        _activate(service, document)
    elif kind == "item":
        identity = action["id"]
        previous = service.resume_document(identity)
        if previous:
            _activate(service, previous)
        else:
            st.session_state[PREFIX + "item_id"] = identity
            option = next((v for v in service.options() if v["id"] == identity), None)
            if not option:
                st.session_state[PREFIX + "error"] = "所选材料已删除或不可用，请重新选择。"
                return
            st.session_state[PREFIX + "source_id"] = option["source_id"]
            st.session_state[PREFIX + "material"] = option["source_id"]
            st.session_state[PREFIX + "question"] = identity
            st.session_state.pop(PREFIX + "document_id", None)
            st.session_state.pop(PREFIX + "selected_version", None)
            st.session_state[PREFIX + "requirements"] = ""
    elif kind in {"leave", "import"}:
        st.session_state[PREFIX + "focus"] = False
        st.session_state["sl_main_tab"] = (
            "待确认材料" if kind == "import" else
            st.session_state.get(PREFIX + "return_tab", "待确认材料")
        )


def _defer_editor(callback, *, args: tuple = (), kwargs: dict | None = None) -> bool:
    identity = st.session_state.get(PREFIX + "document_id")
    return bool(identity and st.session_state.get(PREFIX + "editor_live_" + identity)
                and defer_color_action(PREFIX + "body_" + identity, callback,
                                       args=args, kwargs=kwargs))


def _request(store: StudyStore, action: dict, *, flushed: bool = False) -> None:
    if (action["kind"] == "item" and action["id"] ==
            st.session_state.get(PREFIX + "item_id")):
        return
    if (action["kind"] == "document" and action["id"] ==
            st.session_state.get(PREFIX + "document_id") and
            action.get("version_id") in (None, st.session_state.get(PREFIX + "selected_version"))):
        return
    if not flushed and _defer_editor(_request, args=(store, action), kwargs={"flushed": True}):
        return
    if not _persist(store):
        return
    service = review_service(store)
    if action["kind"] == "generate" and _dirty(service, _active(service)):
        st.session_state[PREFIX + "pending"] = action
    else:
        _apply(store, action)


def _material_changed(store: StudyStore) -> None:
    source_id = st.session_state[PREFIX + "material"]
    previous = st.session_state.get(PREFIX + "source_id")
    if source_id is None or source_id == previous:
        return
    options = review_service(store).options()
    st.session_state[PREFIX + "material"] = (
        previous if any(o["source_id"] == previous for o in options) else None
    )
    target = next((option for option in options if option["source_id"] == source_id), None)
    if target:
        _request(store, {"kind": "item", "id": target["id"]})


def _question_changed(store: StudyStore) -> None:
    target = st.session_state[PREFIX + "question"]
    previous = st.session_state.get(PREFIX + "item_id")
    st.session_state[PREFIX + "question"] = previous or target
    if target != previous:
        _request(store, {"kind": "item", "id": target})


def _finish_switch(store: StudyStore, *, save: bool, flushed: bool = False) -> None:
    if not flushed and _defer_editor(_finish_switch, args=(store,),
                                     kwargs={"save": save, "flushed": True}):
        return
    try:
        intended = _intended_body(store, flushed=flushed)
    except StudyError as exc:
        st.session_state[PREFIX + "error"] = str(exc)
        return
    if not _persist(store, user_edit=save):
        return
    action = st.session_state.get(PREFIX + "pending")
    if not action:
        return
    if save:
        service = review_service(store)
        document = _active(service)
        if document:
            try:
                _save_exact_version(service, document, intended)
            except StudyError as exc:
                st.session_state[PREFIX + "error"] = str(exc)
                return
    st.session_state.pop(PREFIX + "pending", None)
    _apply(store, action)


def _save(store: StudyStore, *, flushed: bool = False) -> None:
    if not flushed and _defer_editor(_save, args=(store,), kwargs={"flushed": True}):
        return
    try:
        intended = _intended_body(store, flushed=flushed)
    except StudyError as exc:
        st.session_state[PREFIX + "error"] = str(exc)
        return
    if _persist(store, user_edit=True):
        service = review_service(store)
        document = _active(service)
        if document:
            try:
                _save_exact_version(service, document, intended)
            except StudyError as exc:
                st.session_state[PREFIX + "error"] = str(exc)
            else:
                st.session_state[PREFIX + "flash"] = "复盘已保存。"


def _intended_body(store: StudyStore, *, flushed: bool = False) -> str | None:
    """Capture the acknowledged editor body before persistence can update its local cache."""
    document = _active(review_service(store))
    if document is None:
        return None
    key = PREFIX + "body_" + document["id"]
    return confirmed_color_body(key, flushed=flushed)


def _save_exact_version(service: AIReviewService, document: dict, intended: str | None) -> dict:
    """A save may not silently substitute another body between capture and commit."""
    if intended is None or document["working"] != intended:
        raise StudyError("编辑正文尚未完成同步，未保存旧稿，请保留当前输入并重新核对。")
    return service.save_version(
        document["id"], expected_revision=document["working_revision"], expected_content=intended,
    )


def _render_pending(store: StudyStore) -> None:
    if PREFIX + "pending" not in st.session_state:
        return
    st.info("当前稿件尚未保存为版本，编辑内容已保留在本地。")
    keep, save, cancel = st.columns(3)
    keep.button("保留草稿并继续", key=PREFIX + "keep_switch",
                on_click=_finish_switch, args=(store,), kwargs={"save": False})
    save.button("保存后继续", key=PREFIX + "save_switch",
                on_click=_finish_switch, args=(store,), kwargs={"save": True})
    cancel.button("取消切换", key=PREFIX + "cancel_switch",
                  on_click=lambda: st.session_state.pop(PREFIX + "pending", None))


def render_ai_review(store: StudyStore, *, focused: bool = False, advanced: bool = True) -> None:
    """Read, edit, preview and save locally; only the generation button can queue a model call."""
    service = review_service(store)
    requested = st.session_state.pop(PREFIX + "requested", None)
    if requested:
        _request(store, requested)
    action = st.session_state.pop(PREFIX + "execute", None)
    if action:
        try:
            with st.spinner("正在生成AI复盘…"):
                document = service.generate(
                    [action["id"]], model_provider(), requirements=action.get("requirements", "")
                )
            _activate(service, document)
        except (StudyError, AIError) as exc:
            st.session_state[PREFIX + "error"] = str(exc)
    if focused:
        st.button("返回学习整理", key=PREFIX + "back", on_click=_request,
                  args=(store, {"kind": "leave"}))
    st.subheader("AI复盘")
    if error := st.session_state.pop(PREFIX + "error", None):
        st.warning(error)
    if flash := st.session_state.pop(PREFIX + "flash", None):
        st.success(flash)
    options = service.options()
    if not options:
        st.info("暂无可用于AI复盘的文字材料。请先导入材料；图片或扫描件需先完成文字识别。")
        st.button("前往导入材料", key=PREFIX + "go_import", on_click=_request,
                  args=(store, {"kind": "import"}))
        _render_pending(store)
        document = _active(service)
        if document:
            _render_manuscript(store, service, document)
            if advanced:
                with st.expander("高级设置"):
                    render_ai_review_advanced(store)
        return
    by_id = {option["id"]: option for option in options}
    selected = st.session_state.get(PREFIX + "item_id")
    if selected is None:
        previous = service.resume_document()
        _apply(store, {"kind": "document", "id": previous["id"]} if previous else
               {"kind": "item", "id": options[0]["id"]})
        selected = st.session_state.get(PREFIX + "item_id")
    missing = selected not in by_id
    sources = {option["source_id"]: option["filename"] for option in options}
    if missing:
        st.warning("当前选择的材料已删除或不可用，请重新选择后生成；已有稿件已保留。")
        st.session_state[PREFIX + "material"] = None
    st.selectbox("当前材料", ([None] if missing else []) + list(sources),
                 format_func=lambda i: sources[i] if i else "请选择材料",
                 key=PREFIX + "material", on_change=_material_changed, args=(store,))
    same_source = [option for option in options if option["source_id"] ==
                   st.session_state.get(PREFIX + "source_id")]
    if len(same_source) > 1 and not missing:
        questions = {option["id"]: option for option in same_source}
        labels = {identity: ((option["number"] + " · ") if option["number"] else "")
                  + option["preview"][:40] for identity, option in questions.items()}
        st.selectbox("题目", list(questions), format_func=labels.get,
                     key=PREFIX + "question", on_change=_question_changed, args=(store,))
    st.text_area("可选要求", key=PREFIX + "requirements", height=80, max_chars=16000)
    generate = st.button("AI复盘", key=PREFIX + "generate", type="primary", disabled=missing)
    if generate and not missing:
        _request(store, {"kind": "generate", "id": st.session_state[PREFIX + "item_id"],
                         "requirements": st.session_state.get(PREFIX + "requirements", "")})
        if PREFIX + "execute" in st.session_state:
            st.rerun()
    _render_pending(store)
    document = _active(service)
    if document:
        _render_manuscript(store, service, document)
    if advanced:
        with st.expander("高级设置"):
            render_ai_review_advanced(store)


def _render_manuscript(store: StudyStore, service: AIReviewService, document: dict) -> None:
    identity = document["id"]
    if document["status"] == "failed":
        st.warning(document["error"] or "这次未取得复盘正文，请重新生成。")
    elif document["status"] == "generating":
        st.info("复盘请求尚未结束，已有稿件已保留。")
    if not document["working"]:
        saved_id = (st.session_state.get(PREFIX + "selected_version")
                    or document.get("last_saved_version_id"))
        if saved_id:
            _render_version_export(store, service, document, service.version(saved_id),
                                   location="current")
        return
    body_key, loaded_key = PREFIX + "body_" + identity, PREFIX + "loaded_" + identity
    if body_key not in st.session_state or st.session_state.get(body_key) == st.session_state.get(
        loaded_key
    ):
        _cache(document)
    version_id = st.session_state.get(PREFIX + "selected_version")
    if document.get("viewed_version_id") != version_id:
        service.remember_view(identity, version_id)
    version = service.version(version_id) if version_id else None
    content = version["content"] if version else document["working"]
    if version:
        st.caption("已保存修订版")
    elif _dirty(service, document):
        st.caption("本地编辑草稿")
    else:
        st.caption("已保存修订版" if document.get("last_saved_version_id") else "AI初稿")
    if not version and not document.get("last_saved_version_id"):
        from .formula_quality import formula_inventory

        if formula_inventory(content):
            st.caption("公式与适用条件待人工核验。")
    mode = st.radio("正文视图", ["阅读", "编辑", "预览"], horizontal=True,
                    key=PREFIX + "view_" + identity, on_change=_view_changed,
                    args=(store, identity))
    st.session_state[PREFIX + "editor_live_" + identity] = mode == "编辑" and not version
    deactivate_color_editor(body_key)
    if mode == "编辑" and version:
        st.info("当前查看的是历史版本。可在高级设置中将此版本另存为草稿编辑。")
        render_saved_text(content)
    elif mode == "编辑":
        render_color_editor("复盘正文", key=body_key, height=460, max_chars=200000,
                            on_change=partial(_persist, user_edit=True, document_id=identity),
                            args=(store,))
    else:
        render_saved_text(content)
    if not version:
        st.button("保存复盘", key=PREFIX + "save_" + identity, on_click=_save, args=(store,))
    export_version = version or (
        service.version(document["last_saved_version_id"])
        if document.get("last_saved_version_id") else None
    )
    if export_version:
        _render_version_export(store, service, document, export_version, location="current")
    else:
        st.caption("保存复盘后可导出 Markdown 和 Word。")


def _render_version_export(
    store: StudyStore, service: AIReviewService, document: dict, version: dict, *, location: str
) -> None:
    """Offer downloads bound to this saved revision, even when the working text has changed."""
    from . import ai_review_export
    from .knowledge_import_ui import render_review_knowledge_import

    render_review_knowledge_import(store, document, version, location=location)

    names = []
    for source_id in document["source_ids"]:
        try:
            names.append(store.source(source_id).name)
        except StudyError:
            continue
    st.caption(f"导出第{version['number']}版（已保存内容）")
    md, word = st.columns(2)
    key = PREFIX + "export_" + location + "_"
    md.download_button(
        "下载 Markdown", ai_review_export.markdown(service, version["id"]),
        ai_review_export.filename(names, version["number"], "md"), mime="text/markdown",
        key=key + "md_" + version["id"], on_click="ignore",
    )
    word.download_button(
        "下载 Word", partial(ai_review_export.docx, service, version["id"]),
        ai_review_export.filename(names, version["number"], "docx"),
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        key=key + "docx_" + version["id"], on_click="ignore",
    )


def _open_version(document_id: str, version_id: str, *, flushed: bool = False) -> None:
    if (not flushed and st.session_state.get(PREFIX + "editor_live_" + document_id)
            and defer_color_action(PREFIX + "body_" + document_id, _open_version,
                                   args=(document_id, version_id), kwargs={"flushed": True})):
        return
    st.session_state[PREFIX + "selected_version"] = version_id
    st.session_state[PREFIX + "view_" + document_id] = "阅读"


def _edit_version(store: StudyStore, document_id: str, version_id: str,
                  *, flushed: bool = False) -> None:
    if not flushed and _defer_editor(_edit_version, args=(store, document_id, version_id),
                                     kwargs={"flushed": True}):
        return
    if _persist(store):
        service = review_service(store)
        recovered = service.fork_working(document_id, service.version(version_id)["content"])
        _activate(service, recovered)


def _restore_document(store: StudyStore, document_id: str) -> None:
    service = review_service(store)
    service.restore_document(document_id)
    _activate(service, service.document(document_id))


def _delete_document(store: StudyStore, document_id: str, *, confirmed: bool,
                     flushed: bool = False) -> None:
    if not flushed and _defer_editor(_delete_document, args=(store, document_id),
                                     kwargs={"confirmed": confirmed, "flushed": True}):
        return
    if _persist(store):
        review_service(store).delete_document(document_id, confirmed=confirmed)
        st.session_state.pop(PREFIX + "document_id", None)
        st.session_state.pop(PREFIX + "selected_version", None)


def render_ai_review_advanced(store: StudyStore) -> None:
    """Open exact saved revisions and confirm memory candidates without calling a model."""
    service = review_service(store)
    item_id = st.session_state.get(PREFIX + "item_id")
    deleted = [value for value in service.documents(item_id, include_deleted=True)
               if value.get("deleted_at")]
    if deleted:
        with st.expander("已删除的复盘"):
            chosen = st.selectbox("恢复复盘", deleted,
                                  format_func=lambda v: format_beijing_time(v["created_at"]),
                                  key=PREFIX + "deleted_document")
            st.button("恢复这份复盘", key=PREFIX + "restore_document",
                      on_click=_restore_document, args=(store, chosen["id"]))
    documents = service.documents(item_id) if item_id else []
    if not documents:
        st.caption("暂无AI复盘稿件。")
        return
    selected = st.selectbox(
        "已有复盘稿件", documents, key=PREFIX + "history_document",
        format_func=lambda d: format_beijing_time(d["created_at"])
        + (" · 已保存" if d.get("last_saved_version_id") else " · 草稿"),
    )
    st.button("打开此稿件", key=PREFIX + "open_document", on_click=_request,
              args=(store, {"kind": "document", "id": selected["id"]}))
    document = _active(service)
    if not document:
        return
    versions = service.versions(document["id"])
    if versions:
        version = st.selectbox(
            "已保存版本", versions, key=PREFIX + "history_version_" + document["id"],
            format_func=lambda v: f"第{v['number']}版 · {format_beijing_time(v['created_at'])}",
        )
        st.button("查看此版本", key=PREFIX + "read_version", on_click=_open_version,
                  args=(document["id"], version["id"]))
        st.button("另存为草稿编辑", key=PREFIX + "edit_version", on_click=_edit_version,
                  args=(store, document["id"], version["id"]))
        _render_version_export(store, service, document, version, location="history")
        _render_version_memory(store, service, document, version)
    else:
        st.button("保存到记忆", disabled=True, key=PREFIX + "memory_requires_saved")
        st.caption("先保存复盘，再提炼可编辑的记忆候选。")
    with st.expander("删除这份复盘"):
        confirmed = st.checkbox("确认将这份复盘移入回收站", key=PREFIX + "delete_confirm")
        st.button("删除复盘", disabled=not confirmed, key=PREFIX + "delete_document",
                  on_click=_delete_document, args=(store, document["id"]),
                  kwargs={"confirmed": confirmed})
    with st.expander("记录信息"):
        st.text("稿件编号：" + document["id"])
        if document.get("raw_ai"):
            st.text_area("最初AI稿", document["raw_ai"], disabled=True, height=160,
                         key=PREFIX + "raw_" + document["id"])


def _render_version_memory(
    store: StudyStore, service: AIReviewService, document: dict, version: dict
) -> None:
    from .knowledge_memory_ui import render_review_memory

    render_review_memory(store, document, version, provider_factory=model_provider)


def render_recent_ai_reviews(store: StudyStore) -> None:
    """Reopen saved AI manuscripts without confusing them with the user's practice attempts."""
    service = review_service(store)
    documents = service.documents()
    if not documents:
        st.caption("暂无AI复盘记录。")
        return
    options = {option["id"]: option for option in service.options()}
    for document in documents[:10]:
        name = "、".join(options[i]["filename"] for i in document["item_ids"] if i in options)
        state = ("编辑草稿" if _dirty(service, document) else
                 "已保存修订版" if document.get("last_saved_version_id") else "AI初稿")
        if document["status"] == "failed":
            state = "生成未完成"
        st.text((name or "原材料已不可用") + " · AI复盘 · " + state)
        st.caption(format_beijing_time(document["updated_at"]))
        st.button("打开AI复盘", key="sl_recent_ai_review_" + document["id"],
                  on_click=open_ai_review, args=(document["item_ids"][0], document["id"]))
