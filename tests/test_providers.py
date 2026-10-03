"""模型供应层单元测试。

覆盖 docs/03 §6.1 的 TC-U07 / TC-U08 / TC-U09。
注意：不实例化本地 Embedding（会触发模型下载），本地模式的加载验证放在集成验收中做。
"""

from __future__ import annotations

import pytest

from config.settings import Settings
from src.errors import ProviderError
from src.providers.embedding import get_embeddings
from src.providers.llm import (
    DEFAULT_MODELS,
    PROVIDER_BASE_URLS,
    get_llm,
    resolve_base_url,
    resolve_model,
)


def make_settings(**overrides) -> Settings:
    """构造不受本机 .env 与环境变量影响的配置。"""
    defaults = {"_env_file": None, "llm_api_key": "test-key"}
    defaults.update(overrides)
    return Settings(**defaults)


# ---------------- TC-U07：各供应商 base_url 解析 ----------------


@pytest.mark.parametrize("provider", list(PROVIDER_BASE_URLS))
def test_resolve_base_url_known_providers(provider: str) -> None:
    assert resolve_base_url(provider) == PROVIDER_BASE_URLS[provider]


def test_resolve_base_url_is_case_insensitive() -> None:
    assert resolve_base_url("DeepSeek") == PROVIDER_BASE_URLS["deepseek"]


def test_resolve_base_url_custom_uses_configured_url() -> None:
    assert resolve_base_url("custom", "http://localhost:11434/v1") == "http://localhost:11434/v1"


# ---------------- TC-U08：custom 缺少 base_url ----------------


def test_resolve_base_url_custom_without_url_raises() -> None:
    with pytest.raises(ProviderError, match="LLM_BASE_URL"):
        resolve_base_url("custom", "")


def test_resolve_base_url_custom_with_blank_url_raises() -> None:
    with pytest.raises(ProviderError, match="LLM_BASE_URL"):
        resolve_base_url("custom", "   ")


# ---------------- TC-U09：未知 provider ----------------


def test_resolve_base_url_unknown_provider_raises_with_options() -> None:
    with pytest.raises(ProviderError) as excinfo:
        resolve_base_url("ollama")
    message = str(excinfo.value)
    assert "ollama" in message
    # 异常信息必须列出可选值，便于排错
    assert "deepseek" in message and "custom" in message


# ---------------- 默认模型 ----------------


@pytest.mark.parametrize("provider", list(DEFAULT_MODELS))
def test_resolve_model_falls_back_to_provider_default(provider: str) -> None:
    assert resolve_model(provider, "") == DEFAULT_MODELS[provider]


def test_resolve_model_prefers_configured_value() -> None:
    assert resolve_model("deepseek", "deepseek-reasoner") == "deepseek-reasoner"


def test_resolve_model_custom_without_model_raises() -> None:
    with pytest.raises(ProviderError, match="LLM_MODEL"):
        resolve_model("custom", "")


# ---------------- get_llm ----------------


def test_get_llm_requires_api_key() -> None:
    settings = make_settings(llm_api_key="")
    with pytest.raises(ProviderError, match="LLM_API_KEY"):
        get_llm(settings)


def test_get_llm_requires_api_key_when_only_whitespace() -> None:
    settings = make_settings(llm_api_key="   ")
    with pytest.raises(ProviderError, match="LLM_API_KEY"):
        get_llm(settings)


def test_get_llm_builds_client_without_network() -> None:
    """构造实例不应发起网络请求，只校验参数装配正确。"""
    settings = make_settings(llm_provider="deepseek", llm_model="deepseek-chat")
    llm = get_llm(settings, streaming=True)

    assert llm.model_name == "deepseek-chat"
    assert str(llm.openai_api_base) == PROVIDER_BASE_URLS["deepseek"]
    assert llm.streaming is True


def test_get_llm_unknown_provider_raises() -> None:
    settings = make_settings(llm_provider="not-exist")
    with pytest.raises(ProviderError, match="未知的 LLM_PROVIDER"):
        get_llm(settings)


# ---------------- get_embeddings ----------------


def test_get_embeddings_unknown_provider_raises() -> None:
    settings = make_settings(embedding_provider="magic")
    with pytest.raises(ProviderError, match="未知的 EMBEDDING_PROVIDER"):
        get_embeddings(settings)


def test_get_embeddings_cloud_without_key_raises() -> None:
    settings = make_settings(embedding_provider="cloud", embedding_api_key="")
    with pytest.raises(ProviderError, match="EMBEDDING_API_KEY"):
        get_embeddings(settings)


def test_get_embeddings_cloud_without_model_raises() -> None:
    settings = make_settings(
        embedding_provider="cloud", embedding_api_key="k", embedding_model=""
    )
    with pytest.raises(ProviderError, match="EMBEDDING_MODEL"):
        get_embeddings(settings)
