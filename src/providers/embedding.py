"""向量化（Embedding）工厂：本地 BGE 与云端 API 可插拔。

设计依据：docs/02-架构设计.md §4.1

注意：本地模型相关依赖（sentence-transformers / torch）较重，因此延迟到函数内部导入，
避免在只需要配置或 LLM 的场景下付出加载成本。
"""

from __future__ import annotations

import logging
from typing import Any

from config.settings import Settings, get_settings
from src.errors import ProviderError

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("local", "cloud")


def get_embeddings(settings: Settings | None = None) -> Any:
    """构造向量化实例。

    Args:
        settings: 配置，默认取全局单例。

    Returns:
        LangChain Embeddings 实例（本地 BGE 或云端 API），两者接口一致，上层无感知。

    Raises:
        ProviderError: provider 取值未知，或云端模式下缺少 API Key。
    """
    s = settings or get_settings()
    provider = (s.embedding_provider or "").strip().lower()

    if provider == "local":
        # 本地模式：模型在进程内加载，向量不出本机
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.debug("初始化本地 Embedding：%s", s.embedding_model)
        return HuggingFaceEmbeddings(
            model_name=s.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )

    if provider == "cloud":
        from langchain_openai import OpenAIEmbeddings

        if not s.embedding_api_key.strip():
            raise ProviderError(
                "EMBEDDING_PROVIDER=cloud 时必须配置 EMBEDDING_API_KEY"
            )
        if not s.embedding_model.strip():
            raise ProviderError("EMBEDDING_PROVIDER=cloud 时必须配置 EMBEDDING_MODEL")

        kwargs: dict[str, Any] = {
            "model": s.embedding_model,
            "api_key": s.embedding_api_key,
        }
        if s.embedding_base_url.strip():
            kwargs["base_url"] = s.embedding_base_url

        logger.debug("初始化云端 Embedding：%s", s.embedding_model)
        return OpenAIEmbeddings(**kwargs)

    supported = "、".join(SUPPORTED_PROVIDERS)
    raise ProviderError(f"未知的 EMBEDDING_PROVIDER：{provider}（可选：{supported}）")
