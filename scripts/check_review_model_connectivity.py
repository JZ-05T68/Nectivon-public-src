"""Explicit connectivity only: no database, materials, fabricated questions or fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.provider import AIUnavailableError  # noqa: E402
from src.ai.provider_scope import ACTIVE_MODELS, DOCUMENTATION_CHECKED  # noqa: E402
from src.ai.study_review_adapter import configured_review_provider  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-api-calls", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.confirm_api_calls:
        raise SystemExit("未明确授权 API 连通性调用，未读取密钥、发送请求或运行测试。")
    from nectivon_model_adapter import RequestBudget

    budget = RequestBudget(4)
    results = []
    for provider, models in ACTIVE_MODELS.items():
        for model in models:
            try:
                adapter = configured_review_provider(
                    provider.value,
                    model,
                    control=lambda: None,
                    budget=budget,
                    timeout_seconds=45,
                    max_extra_attempts=0,
                    max_output_tokens=64,
                    connectivity=True,
                )
                result = adapter.review("Connectivity check only. Reply with exactly OK.")
                result["connectivity"] = (
                    "RESPONSE_RECEIVED" if result["status"] == "ok" else "CALL_FAILED"
                )
            except AIUnavailableError as exc:
                result = {
                    "provider": provider.value,
                    "requested_model": model,
                    "connectivity": "BLOCKED",
                    "attempts": 0,
                    "reason": str(exc),
                }
            except Exception:
                result = {
                    "provider": provider.value,
                    "requested_model": model,
                    "connectivity": "BLOCKED",
                    "attempts": 0,
                    "reason": "本地依赖或安全配置不可用；未打印异常对象或凭据。",
                }
            results.append(result)
            print(
                json.dumps(
                    {
                        k: result.get(k)
                        for k in (
                            "provider",
                            "requested_model",
                            "connectivity",
                            "attempts",
                            "error_code",
                            "reason",
                        )
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    report = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "documentation_checked": DOCUMENTATION_CHECKED,
        "scope": "仅真实 API 连通性；固定中性 OK 指令，不涉及学科题目、用户材料或数据库。",
        "functional_tests": "NOT_RUN",
        "real_question_acceptance": "BLOCKED_NO_AUTHORIZED_MATERIAL",
        "calls_limit": 4,
        "automatic_retries": 0,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if any(r["connectivity"] != "RESPONSE_RECEIVED" for r in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
