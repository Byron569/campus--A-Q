"""大模型供应商注册表。

统一走 OpenAI 兼容协议，用 base_url 切换供应商——换模型只改 .env，不侵入业务代码。

设计依据：docs/02-架构设计.md §4.1
"""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI

from config.settings import Settings, get_settings
from src.errors import ProviderError

logger = logging.getLogger(__name__)

# 供应商 → OpenAI 兼容地址
PROVIDER_BASE_URLS: dict[str, str] = {
    "deepseek": "https://api.deepseek.com/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "openai": "https://api.openai.com/v1",
}

# 供应商 → 默认模型
DEFAULT_MODELS: dict[str, str] = {
    "deepseek": "deepseek-chat",
    "dashscope": "qwen-max",
    "zhipu": "glm-4-plus",
    "openai": "gpt-4o",
}

# 请求超时（秒）与重试次数
REQUEST_TIMEOUT = 60
MAX_RETRIES = 1


def resolve_base_url(provider: str, custom_base_url: str = "") -> str:
    """把 provider 名称解析为可用的 base_url。"""
    name = (provider or "").strip().lower()

    if name == "custom":
        if not custom_base_url.strip():
            raise ProviderError(
                "LLM_PROVIDER=custom 时必须提供 LLM_BASE_URL"
            )
        return custom_base_url.strip()

    if name not in PROVIDER_BASE_URLS:
        supported = "、".join(PROVIDER_BASE_URLS) + "、custom"
        raise ProviderError(f"未知的 LLM_PROVIDER：{provider}（可选：{supported}）")

    return PROVIDER_BASE_URLS[name]


def resolve_model(provider: str, configured_model: str = "") -> str:
    """决定使用哪个模型名：配置优先，否则用该供应商的默认模型。"""
    if configured_model and configured_model.strip():
        return configured_model.strip()

    name = (provider or "").strip().lower()
    default = DEFAULT_MODELS.get(name, "")
    if not default and name != "custom":
        raise ProviderError(f"供应商 {provider} 没有默认模型，请在 .env 中显式指定 LLM_MODEL")
    if not default:
        raise ProviderError("LLM_PROVIDER=custom 时必须提供 LLM_MODEL")
    return default


def get_llm(settings: Settings | None = None, *, streaming: bool = False) -> ChatOpenAI:
    """构造大模型实例。

    Args:
        settings: 配置，默认取全局单例。
        streaming: 是否流式输出。生成答案用 True，Query 改写用 False。

    Raises:
        ProviderError: 未配置 LLM_API_KEY，或 provider / 模型名无法解析。
    """
    s = settings or get_settings()

    if not s.llm_api_key.strip():
        raise ProviderError(
            "未配置 LLM_API_KEY。请复制 .env.example 为 .env 并填入 API Key。"
        )

    provider = (s.llm_provider or "").strip().lower()
    base_url = resolve_base_url(provider, s.llm_base_url)
    model = resolve_model(provider, s.llm_model)

    logger.debug("初始化 LLM：provider=%s model=%s streaming=%s", provider, model, streaming)

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=s.llm_api_key,
        temperature=s.llm_temperature,
        streaming=streaming,
        timeout=REQUEST_TIMEOUT,
        max_retries=MAX_RETRIES,
    )


def response_text(response) -> str:
    """从模型响应里取出纯文本。

    OpenAI 兼容协议下 `content` 通常是字符串，但部分供应商会返回分段内容
    （`list[dict]`），这里统一成字符串，避免上层各写一遍。
    """
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)
