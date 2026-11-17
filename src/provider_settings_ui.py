"""Streamlit component for local multi-provider model configuration."""

from __future__ import annotations

import logging

import streamlit as st

from src.ai.model_registry import (
    CapabilitySupport,
    ModelPurpose,
    ProviderId,
    get_capability_profile,
    get_provider_definition,
    list_model_presets,
)
from src.ai.provider import AIUnavailableError, ProviderCallError
from src.ai.provider_config import ModelSelectionKind
from src.ai.provider_settings_service import (
    ProviderConfigurationState,
    ProviderSettingsService,
    safe_settings_error,
)

LOGGER = logging.getLogger(__name__)
_PROVIDER_LABELS = {
    ProviderId.DEEPSEEK: "DeepSeek",
    ProviderId.QWEN: "Qwen",
    ProviderId.KIMI: "Kimi",
    ProviderId.HUNYUAN: "Hunyuan / 腾讯混元",
    ProviderId.GLM: "GLM / 智谱",
}
_AI_SESSION_KEYS = (
    "agent_live_client",
    "rag_answer_output",
    "rag_answer_mock",
    "experience_output",
    "experience_is_mock",
    "experience_context_items",
)


def render_provider_settings() -> None:
    """Render settings without ever reading a stored credential value."""

    from src.runtime import application_provider_settings_service

    service = application_provider_settings_service()
    st.divider()
    st.subheader("AI / 模型服务")
    try:
        state, active_provider, active_model = service.current_state()
    except (AIUnavailableError, ProviderCallError) as exc:
        st.error(safe_settings_error(exc))
        return

    status_label = {
        ProviderConfigurationState.CONFIGURED: "已配置",
        ProviderConfigurationState.UNCONFIGURED: "未配置",
        ProviderConfigurationState.LEGACY: "legacy",
        ProviderConfigurationState.DISABLED: "已停用（历史配置保留）",
    }[state]
    current_provider = _PROVIDER_LABELS.get(active_provider, "—")
    st.markdown(
        f"**当前模型（讲题/知识问答）**　Provider：`{current_provider}`　"
        f"Model：`{active_model or '—'}`　状态：**{status_label}**"
    )
    if state is ProviderConfigurationState.LEGACY:
        st.info(
            "当前正在使用旧环境变量配置。尚未迁移到安全存储；"
            "可在下方重新输入 API Key 并保存为新配置。"
        )

    image_provider, image_model = service.current_image_state()
    st.markdown(
        f"**当前读图模型**　Provider：`{_PROVIDER_LABELS.get(image_provider, '—')}`　"
        f"Model：`{image_model or '—'}`"
    )
    image_tab, text_tab = st.tabs(["读图 / 切题", "讲题 / 知识问答"])
    with image_tab:
        st.caption("直接读取原图，生成题目、数学公式和原图裁块；无网络时提示无法识别。")
        _render_purpose_settings(service, ModelPurpose.IMAGE)
        st.caption("本轮仅保留 DeepSeek 和通义千问；旧供应商记录可读取，正式调用已停用。")
    with text_tab:
        st.caption("用于讲题、知识问答和人工输入的数学内容排版。")
        _render_purpose_settings(service, ModelPurpose.TEXT)
    _render_knowledge_generation_settings()


def _render_knowledge_generation_settings() -> None:
    """Separate explicitly authorized domestic API use from verified free credits."""
    from datetime import UTC, datetime

    from src.ai.knowledge_generation import (
        DEFAULT_KNOWLEDGE_TASK_TOKENS,
        KNOWLEDGE_MODEL_ROLES,
        VerifiedFreeCreditGrant,
    )
    from src.ai.knowledge_generation_settings import (
        KnowledgeGenerationSettings,
        load_knowledge_generation_settings,
        save_knowledge_generation_settings,
        set_knowledge_openai_credential,
    )
    from src.ai.provider_scope import ACTIVE_MODELS

    with st.expander("高级设置 · 知识生成与免费额度", expanded=False):
        st.caption(
            "仅用于知识提炼和关系候选。GPT-6.1 Sol：xhigh；DeepSeek：max；"
            "Qwen3.8：xhigh。模型审查仍是候选，需要你确认后入库。"
        )
        st.caption(
            "知识生成默认输出上限 65,536 Token，推理与可见正文共用此上限；"
            "任务预算同时计入输入、推理和正文。单次超时上限不少于 300 秒，不自动重试。"
        )
        try:
            saved = load_knowledge_generation_settings()
        except AIUnavailableError as exc:
            st.warning(str(exc))
            saved = None
        providers = list(KNOWLEDGE_MODEL_ROLES)
        selected = st.selectbox(
            "知识生成服务",
            providers,
            index=providers.index(saved.provider if saved else "deepseek"),
            key="knowledge_generation_provider",
        )
        models = (
            [KNOWLEDGE_MODEL_ROLES[selected][0]]
            if selected == "openai"
            else list(ACTIVE_MODELS[ProviderId(selected)])
        )
        preferred = (
            saved.model
            if saved and saved.provider == selected
            else KNOWLEDGE_MODEL_ROLES[selected][0]
        )
        model = st.selectbox(
            "知识生成模型",
            models,
            index=models.index(preferred) if preferred in models else 0,
            key=f"knowledge_generation_model_{selected}",
        )
        st.caption(
            f"建议职责：{KNOWLEDGE_MODEL_ROLES[selected][1]}。不自动多模型投票或切换供应商。"
        )
        provider_label = {"deepseek": "DeepSeek", "qwen": "Qwen"}.get(selected, selected)
        configured_api = selected in {"deepseek", "qwen"} and st.checkbox(
            f"使用已配置的 {provider_label} API（可能按供应商计费）",
            value=bool(saved and saved.provider == selected and saved.configured_api),
            key=f"knowledge_generation_{selected}_configured_api",
        )
        grant = (
            saved.grant if saved and saved.provider == selected and saved.model == model else None
        )
        if configured_api:
            st.info(
                f"保存后沿用上方 {provider_label} 凭据；每次生成须主动确认，受任务与全局预算限制，"
                "不自动重试。此模式不代表拥有免费额度，实际费用以供应商账单为准。"
            )
        else:
            st.info(
                "当前额度状态：UNKNOWN / BLOCKED，未自动查询账户。"
                if grant is None
                else "已保存人工核验记录；每次请求检查账期、模型权限及累计预留。"
            )
            st.caption(
                "免费额度模式须在供应商端核验组织、账期、适用模型并启用额度用尽即停。"
                "仅有本地估算余额不能保证不扣费；未核验时不会发送请求。"
            )
        with st.form(f"knowledge_generation_credit_form_{selected}_{model}", clear_on_submit=False):
            api_key = (
                st.text_input(
                    "OpenAI API Key（仅本次进程内存保存；也可通过 EKB_OPENAI_API_KEY 配置）",
                    type="password",
                    key="knowledge_generation_openai_key",
                )
                if selected == "openai"
                else ""
            )
            st.caption("国内 API Key 使用上方现有安全存储，不复制旧供应商凭据。")
            organization = st.text_input(
                "已核对的组织 / 业务空间",
                value=grant.organization if grant else "",
                disabled=configured_api,
            )
            evidence = st.text_input(
                "核验凭据说明（控制台页面 / 本地证据编号，不填写 Key）",
                value=grant.evidence_ref if grant else "",
                disabled=configured_api,
            )
            start = st.text_input(
                "额度账期开始（含时区的 ISO 时间）",
                value=grant.period_start.isoformat() if grant else "",
                placeholder="2026-11-15T00:00:00+08:00",
                disabled=configured_api,
            )
            expiry = st.text_input(
                "额度到期（含时区的 ISO 时间）",
                value=grant.expires_at.isoformat() if grant else "",
                placeholder="2026-12-15T00:00:00+08:00",
                disabled=configured_api,
            )
            token_allowance = st.number_input(
                "本账期授权使用的免费 Token 上限（含输入、推理和输出，按实际额度保守填写）",
                min_value=0,
                value=grant.remaining_tokens if grant else 0,
                step=1000,
                disabled=configured_api,
            )
            task_tokens = st.number_input(
                "单任务 Token 上限（输入、推理与正文合计）",
                min_value=1024,
                max_value=1000000,
                value=saved.task_tokens if saved else DEFAULT_KNOWLEDGE_TASK_TOKENS,
                step=1024,
            )
            max_calls = st.number_input(
                "单任务调用上限",
                min_value=1,
                max_value=3,
                value=saved.max_calls if saved else 1,
            )
            access = st.checkbox(
                "我已在该组织核验所选准确模型的访问权限",
                value=bool(grant and grant.model_access_verified),
                disabled=configured_api,
            )
            free_only = st.checkbox(
                "我已在供应商端确认到账免费额度，并设置额度用尽即停、不转付费、不自动充值",
                value=bool(grant and grant.free_only_enforced),
                disabled=configured_api,
            )
            save = st.form_submit_button("保存知识生成设置")
        if save:
            try:
                verified = None
                if not configured_api and (access or free_only):
                    period_start, expires_at = (
                        datetime.fromisoformat(start),
                        datetime.fromisoformat(expiry),
                    )
                    if (
                        not organization.strip()
                        or not evidence.strip()
                        or not access
                        or not free_only
                        or token_allowance <= 0
                        or period_start.tzinfo is None
                        or expires_at.tzinfo is None
                        or not period_start <= datetime.now(UTC) < expires_at
                    ):
                        raise ValueError(
                            "请完整核验有效账期、组织、证据、免费额度及禁止额外付费状态。"
                        )
                    verified = VerifiedFreeCreditGrant(
                        selected,
                        organization.strip(),
                        model,
                        period_start,
                        expires_at,
                        int(token_allowance),
                        evidence.strip(),
                        free_only,
                        access,
                    )
                save_knowledge_generation_settings(
                    KnowledgeGenerationSettings(
                        selected,
                        model,
                        verified,
                        int(max_calls),
                        int(task_tokens),
                        configured_api=configured_api,
                    )
                )
                if api_key:
                    set_knowledge_openai_credential(api_key)
            except (ValueError, AIUnavailableError, OSError) as exc:
                st.error(f"知识生成设置未保存：{exc}")
            else:
                st.success(
                    "已保存；未发送任何模型请求。"
                    + (
                        f"已授权使用配置的 {provider_label} API，可能产生供应商费用。"
                        if configured_api
                        else ("额度仍为 UNKNOWN，生成保持阻止。" if verified is None else "")
                    )
                )
        st.markdown(
            "[OpenAI 模型契约](https://developers.openai.com/api/docs/models/gpt-6.1-sol) · "
            "[OpenAI 额度](https://developers.openai.com/api/docs/guides/chatgpt-api-credits) · "
            "[DeepSeek 参数](https://api-docs.deepseek.com/api/create-chat-completion/) · "
            "[Qwen 参数](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions)"
        )


def _render_purpose_settings(service: ProviderSettingsService, purpose: ModelPurpose) -> None:
    """Render independent selections while sharing each provider's saved key."""

    image_mode = purpose is ModelPurpose.IMAGE
    key_prefix = "ai_image_settings" if image_mode else "ai_settings"
    provider_ids = [item for item in ProviderId if list_model_presets(item, purpose=purpose)]
    current_provider = (
        service.current_image_state()[0] if image_mode else service.current_state()[1]
    )
    preferred = current_provider or (ProviderId.QWEN if image_mode else ProviderId.DEEPSEEK)
    selected_provider = st.selectbox(
        "读图服务商" if image_mode else "服务商",
        options=provider_ids,
        format_func=lambda item: _PROVIDER_LABELS[item],
        index=provider_ids.index(preferred) if preferred in provider_ids else 0,
        key=f"{key_prefix}_provider",
    )
    try:
        view = (
            service.image_view(selected_provider) if image_mode else service.view(selected_provider)
        )
    except (AIUnavailableError, ProviderCallError) as exc:
        st.error(safe_settings_error(exc))
        return
    definition = get_provider_definition(selected_provider)
    preset_ids = [
        preset.model_id for preset in list_model_presets(selected_provider, purpose=purpose)
    ]
    current_model_id = view.settings.model_id if view.settings is not None else ""
    current_is_preset = current_model_id in preset_ids
    model_choice = st.selectbox(
        "读图模型" if image_mode else "模型",
        options=preset_ids,
        index=preset_ids.index(current_model_id) if current_is_preset else 0,
        key=f"{key_prefix}_model_{selected_provider.value}",
    )
    model_id = model_choice
    model_selection = ModelSelectionKind.PRESET

    profile = get_capability_profile(selected_provider, model_id)
    st.caption(
        "支持图片输入和文字问答。"
        if profile.effective.vision is CapabilitySupport.SUPPORTED
        else "用于文字问答；读图请在「读图 / 切题」中单独选择视觉模型。"
    )
    if view.credential_saved:
        st.success("API Key 已安全保存")
    else:
        st.caption("尚未保存 API Key")

    form_key = f"ai_provider_form_{purpose.value}_{selected_provider.value}"
    with st.form(form_key, clear_on_submit=True):
        api_key = st.text_input(
            "API Key",
            value="",
            type="password",
            help="已保存的 Key 永不回填；留空会保留现有 Key。",
        )
        with st.expander("高级设置"):
            base_url = st.text_input(
                "Base URL",
                value=(
                    view.settings.base_url
                    if view.settings is not None
                    else definition.default_base_url
                ),
            )
        st.caption(
            "测试图片连接会发送一张本地生成的小图，可能产生少量 API 费用。"
            if image_mode
            else "测试连接会发送一次极小的模型请求，可能产生少量 API 费用。"
        )
        buttons = st.columns(2 if image_mode else 3)
        save = (
            False
            if image_mode
            else buttons[0].form_submit_button("保存配置", use_container_width=True)
        )
        save_active = buttons[0 if image_mode else 1].form_submit_button(
            "保存并用于读图" if image_mode else "保存并设为当前",
            type="primary",
            use_container_width=True,
        )
        test = buttons[1 if image_mode else 2].form_submit_button(
            "测试图片连接" if image_mode else "测试连接", use_container_width=True
        )

    if save or save_active or test:
        try:
            provider_settings = service.make_settings(
                provider_id=selected_provider,
                base_url=base_url,
                model_id=model_id,
                model_selection=model_selection,
            )
            if test:
                result = service.test_connection(
                    provider_settings, api_key_override=api_key, purpose=purpose
                )
                st.success(
                    (
                        "✓ 图片请求可以调用\n\n"
                        if image_mode
                        else "✓ API Key 有效　✓ 服务可访问　✓ 当前模型可以调用\n\n"
                    )
                    + f"响应模型：{result.resolved_model}"
                )
            else:
                service.save(
                    provider_settings,
                    new_api_key=api_key,
                    make_active=save_active,
                    purpose=purpose,
                )
                _clear_model_session_state()
                st.success("模型服务配置已安全保存。")
        except (AIUnavailableError, ProviderCallError) as exc:
            st.error(safe_settings_error(exc))
        except Exception as exc:  # UI boundary: never expose raw diagnostics
            LOGGER.error("模型服务设置操作失败：error_type=%s", type(exc).__name__)
            st.error(safe_settings_error(exc))

    if not image_mode and st.button(
        "清除该 Provider 配置",
        key=f"ai_settings_delete_{selected_provider.value}",
        use_container_width=True,
    ):
        try:
            service.delete(selected_provider)
            _clear_model_session_state()
            st.success("该 Provider 的配置与凭据已清除。")
        except (AIUnavailableError, ProviderCallError) as exc:
            st.error(safe_settings_error(exc))
        except Exception as exc:
            LOGGER.error("清除模型服务配置失败：error_type=%s", type(exc).__name__)
            st.error(safe_settings_error(exc))


def _effective_capability_caption(capabilities: object) -> str:
    labels = (
        ("流式", capabilities.streaming),
        ("推理参数", capabilities.reasoning),
        ("原生工具", capabilities.tool_calling),
        ("视觉", capabilities.vision),
    )
    rendered = []
    for label, support in labels:
        value = {
            CapabilitySupport.SUPPORTED: "可用",
            CapabilitySupport.UNSUPPORTED: "当前未实现",
            CapabilitySupport.UNKNOWN: "未知",
        }[support]
        rendered.append(f"{label}：{value}")
    return "Nectivon 当前有效能力｜" + "　".join(rendered)


def _clear_model_session_state() -> None:
    for key in _AI_SESSION_KEYS:
        st.session_state.pop(key, None)
