"""One explicitly launched task in the process main thread; never an autonomous agent loop."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from .storage import StudyStore


def registry(root: Path, *, only: str | None = None):
    """Register fixed handlers; task payloads cannot select Python modules or paths."""
    from nectivon_tasks import Registry

    from .components import export_review, import_source, verify_sources

    handlers = Registry()

    def require_active_sources(store: StudyStore, source_ids: list[str]) -> None:
        from .models import StudyError

        if any(store.is_source_deleted(source_id) for source_id in source_ids):
            raise StudyError("材料已删除，任务已停止。")

    def require_active_items(store: StudyStore, item_ids: list[str]) -> None:
        require_active_sources(store, [store.item(item_id).source_id for item_id in item_ids])

    def import_handler(context, payload: dict) -> dict:
        store = StudyStore(root)
        require_active_sources(store, [payload["source_id"]])
        return context.step(
            "import",
            lambda _: import_source(store, payload["source_id"]),
            inputs=payload,
            policy="idempotent",
            idempotency_key=payload["source_id"],
        )

    def import_batch_handler(context, payload: dict) -> dict:
        from .models import StudyError

        store = StudyStore(root)
        results = []
        for index, source_id in enumerate(payload["source_ids"]):
            context.check_control()
            if store.is_source_deleted(source_id):
                raise StudyError("材料已删除，导入任务已停止。")

            def import_one(_: str, selected: str = source_id) -> dict:
                try:
                    return import_source(store, selected)
                except Exception as exc:  # noqa: BLE001 - isolate local file parsing only
                    logging.getLogger(__name__).warning(
                        "study_import_batch file_failed type=%s", type(exc).__name__,
                    )
                    return {
                        "draft_count": 0, "status": "failed", "source_id": selected,
                        "error_type": type(exc).__name__,
                    }

            result = context.step(
                f"import_{index}",
                import_one,
                inputs={"source_id": source_id},
                policy="idempotent",
                idempotency_key=source_id,
            )
            results.append(result)
        result = {
            "file_count": len(results), "draft_count": sum(r["draft_count"] for r in results),
        }
        failures = [record for record in results if record.get("status") == "failed"]
        if failures:
            result.update(failed_file_count=len(failures), failed_files=failures)
        return result

    def extraction_handler(context, payload: dict) -> dict:
        from src.ai.study_model_adapter import configured_extractor

        from .extraction import extract_page, save_extraction
        from .models import StudyError

        store = StudyStore(root)
        if store.is_source_deleted(payload["source_id"]):
            raise StudyError("材料已删除，识别任务已停止。")
        provider = configured_extractor(payload["model"])

        # The sole paid operation is journaled unsafe; a lost response is never replayed.
        def call(_: str) -> dict:
            return extract_page(
                store,
                payload["source_id"],
                payload["page_id"],
                provider,
                allow_remote=payload["allow_remote"],
                use_image=payload["use_image"],
            )

        result = context.step("extract", call, inputs=payload, policy="unsafe")
        return save_extraction(
            store,
            payload["source_id"],
            payload["page_id"],
            result,
            payload["run_id"],
        )

    def verify_handler(context, payload: dict) -> dict:
        store = StudyStore(root)
        require_active_items(store, payload["item_ids"])
        return context.step(
            "verify",
            lambda _: verify_sources(store, payload["item_ids"]),
            inputs=payload,
            policy="pure",
        )

    def export_handler(context, payload: dict) -> dict:
        store = StudyStore(root)
        require_active_items(store, payload["item_ids"])
        return context.step(
            "export",
            lambda _: export_review(store, payload["item_ids"], payload["run_id"]),
            inputs=payload,
            policy="idempotent",
            idempotency_key=payload["run_id"],
        )

    def model_review_handler(context, payload: dict) -> dict:
        from nectivon_model_adapter import RequestBudget

        from src.ai.model_registry import CapabilitySupport, get_capability_profile
        from src.ai.provider_scope import require_active_model
        from src.ai.study_review_adapter import configured_review_provider

        from .model_review import run_review
        from .models import StudyError

        selections = payload["selections"]
        store = StudyStore(root)
        item = store.item(payload["item_id"])
        if store.is_source_deleted(item.source_id):
            raise StudyError("材料已删除，模型复盘任务已停止。")
        budget = RequestBudget(
            (2 if len(selections) == 1 else 5) * (1 + payload["max_extra_attempts"])
        )
        providers = []
        # Validate every selection/credential before paying for either independent answer.
        for selection in selections:
            provider, model = selection["provider"], selection["model"]
            require_active_model(provider, model)
            if (
                payload["input_mode"] == "image"
                and get_capability_profile(
                    provider,
                    model,
                ).effective.vision
                is not CapabilitySupport.SUPPORTED
            ):
                raise StudyError("所选模型不支持图片输入，未发送任何题目；请重新选择。")
            providers.append(
                configured_review_provider(
                    provider,
                    model,
                    control=context.check_control,
                    budget=budget,
                    max_extra_attempts=payload["max_extra_attempts"],
                )
            )
        return run_review(store, context, payload, providers)

    def safe_model_review_handler(context, payload: dict) -> dict:
        from nectivon_tasks import TaskError

        from src.ai.credential_store import CredentialStoreError
        from src.ai.provider import AIUnavailableError

        from .model_review_records import load_record, save_record
        from .models import StudyError

        try:
            return model_review_handler(context, payload)
        except (
            AIUnavailableError,
            StudyError,
            CredentialStoreError,
            OSError,
            ValueError,
            ImportError,
        ) as exc:
            # Keep the engine's failed/unsafe-intent state, plus a safe local
            # explanation. Never convert an unfinished intent into task success.
            message = (
                str(exc)
                if isinstance(exc, (AIUnavailableError, StudyError, CredentialStoreError))
                else "本地依赖、配置或文件处理失败；已存原答和暂存文件保留，请人工核对。"
            )
            try:
                store = StudyStore(root)
                if load_record(store, payload["run_id"], "failure") is None:
                    save_record(
                        store,
                        payload["run_id"],
                        "failure",
                        {
                            "status": "blocked",
                            "message": "模型复盘未完成：" + message,
                        },
                    )
            except (StudyError, OSError, ValueError):
                # A second disk/config failure must not hide the original failed
                # task or cause any paid replay. Never log material/exception text.
                logging.getLogger(__name__).warning("model_review failure_record_unavailable")
            raise TaskError("HANDLER") from None

    def has_source(payload: dict) -> bool:
        return type(payload.get("source_id")) is str and bool(payload["source_id"])

    def selection(payload: dict) -> bool:
        return (
            type(payload.get("item_ids")) is list
            and 0 < len(payload["item_ids"]) <= 1000
            and all(type(item) is str and item for item in payload["item_ids"])
        )

    handlers.register("study_import", 1, import_handler, validate=has_source)
    handlers.register(
        "study_import_batch", 1, import_batch_handler,
        validate=lambda p: (
            type(p.get("source_ids")) is list
            and 0 < len(p["source_ids"]) <= 100
            and all(type(value) is str and value for value in p["source_ids"])
            and len(set(p["source_ids"])) == len(p["source_ids"])
        ),
    )
    handlers.register(
        "study_extract",
        1,
        extraction_handler,
        validate=lambda p: (
            has_source(p)
            and type(p.get("page_id")) is str
            and type(p.get("model")) is str
            and bool(p["model"])
            and p.get("allow_remote") is True
            and type(p.get("use_image")) is bool
            and type(p.get("run_id")) is str
            and len(p["run_id"]) == 32
        ),
    )
    handlers.register("study_verify", 1, verify_handler, validate=selection)
    from .model_review import validate_request

    handlers.register("study_model_review", 1, safe_model_review_handler, validate=validate_request)
    handlers.register(
        "study_export",
        1,
        export_handler,
        validate=lambda p: (
            selection(p)
            and type(p.get("run_id")) is str
            and len(p["run_id"]) == 32
            and all(c in "0123456789abcdef" for c in p["run_id"])
        ),
    )
    if only is None:
        return handlers
    if only != "study_import_batch":
        raise ValueError("Only the local import batch may use a restricted worker.")
    restricted = Registry()
    handler = handlers.get(only, 1)
    restricted.register(only, 1, handler.run, validate=handler.validate)
    return restricted


def run_import_batch(tasks, root: Path, task_id: str, *, timeout_seconds: float = 300) -> None:
    """Wait for this local batch only, with a finite wait for an occupied queue slot."""
    from nectivon_tasks import Worker

    from .models import StudyError

    worker = Worker(tasks, registry(root, only="study_import_batch"))
    deadline = time.monotonic() + timeout_seconds
    while True:
        task = tasks.get(task_id)
        if task.kind != "study_import_batch":
            raise StudyError("导入任务类型不匹配。")
        if task.state not in {"queued", "running", "pause_requested", "cancel_requested"}:
            return
        if task.state == "queued" and worker.run_once():
            continue
        if time.monotonic() >= deadline:
            # Do not leave an unclaimed batch indefinitely queued, and never
            # restart an in-flight operation owned by another worker.
            if tasks.get(task_id).state == "queued":
                tasks.request_cancel(task_id)
            logging.getLogger(__name__).warning("study_import_batch queue_wait_expired")
            return
        time.sleep(0.25)


def main() -> None:
    """A bounded child process, started only by the user-facing task controls."""
    from nectivon_tasks import Worker

    from .runtime import queue, require_released

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kind", choices=["study_import_batch"])
    parser.add_argument("--task-id")
    arguments = parser.parse_args()
    require_released()
    store = StudyStore(arguments.root)
    tasks = queue(store.root)
    tasks.recover()  # Interrupted tasks remain interrupted; no implicit resume.
    if arguments.kind:
        if not arguments.task_id:
            parser.error("--task-id is required for a local import batch")
        run_import_batch(tasks, store.root, arguments.task_id)
    else:
        Worker(tasks, registry(store.root)).run_once()


if __name__ == "__main__":
    main()
