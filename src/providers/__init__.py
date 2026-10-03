"""模型供应层：大模型与向量化的可插拔抽象。

设计依据：docs/02-架构设计.md §4.1、docs/06-接口文档.md §1.2
"""

from src.providers.embedding import get_embeddings
from src.providers.llm import (
    DEFAULT_MODELS,
    PROVIDER_BASE_URLS,
    get_llm,
    resolve_base_url,
    resolve_model,
)

__all__ = [
    "DEFAULT_MODELS",
    "PROVIDER_BASE_URLS",
    "get_llm",
    "get_embeddings",
    "resolve_base_url",
    "resolve_model",
]
