"""Query 改写：把带指代的追问改成可独立检索的问题。

设计依据：
- docs/02-架构设计.md §4.4（多轮对话与 Query 改写）
- docs/06-接口文档.md §1.6（`rewrite_query(question, history) -> str`）

契约（三条退化路径都必须回到原问题，绝不因改写失败而检索不到）：
- 无历史（首问）→ 不调 LLM，直接用原问题；
- 调 LLM 抛错（超时 / 401 / 未配置 Key）→ 用原问题；
- 模型返回空 → 用原问题。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from config.settings import Settings, get_settings
from src.providers.llm import get_llm, response_text

if TYPE_CHECKING:
    from src.repository import Message

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """你是校园问答系统的查询改写助手。请根据对话历史，把用户的最新提问改写成一个**不依赖上下文、可独立检索**的完整问题。

要求：
1. 补全历史中省略的主语、宾语与指代（如「它」「这个」「那边」「那个学院」）。
2. 保持原意，不得添加历史与提问中都没有出现的信息。
3. 只输出改写后的问题本身，不要解释、不要加引号、不要输出其他内容。

对话历史：
{history}

用户最新提问：{question}

改写结果："""


def format_history(history: list["Message"], *, limit: int) -> str:
    """把最近若干条消息拼成「用户：… / 助手：…」的文本。"""
    recent = history[-limit:] if limit > 0 else []
    lines = [
        f"{'用户' if message.role == 'user' else '助手'}：{message.content}"
        for message in recent
    ]
    return "\n".join(lines)


def rewrite_query(
    question: str,
    history: list["Message"] | None = None,
    *,
    llm=None,
    settings: Settings | None = None,
) -> str:
    """改写追问。任何失败都退化为原问题，保证检索一定拿得到查询串。

    Args:
        question: 用户本轮原始提问。
        history: 该会话历史消息（时间正序）。
        llm: 可注入的聊天模型，便于测试；默认按配置构造。
        settings: 配置，默认取全局单例。
    """
    original = (question or "").strip()
    if not original:
        return original

    s = settings or get_settings()
    recent = list(history or [])[-(s.history_rounds * 2):]
    if not recent:
        # 首问没有上下文可补，省一次 LLM 调用
        return original

    prompt = REWRITE_PROMPT.format(
        history=format_history(recent, limit=len(recent)), question=original
    )
    try:
        model = llm or get_llm(s, streaming=False)
        response = model.invoke(prompt)
        rewritten = response_text(response).strip()
    except Exception as exc:  # 超时 / 401 / 欠费 / 网络中断 / 未配置 Key
        logger.warning("Query 改写失败，退化为原问题：%s", exc)
        return original

    if not rewritten:
        logger.warning("Query 改写返回空结果，退化为原问题")
        return original
    return rewritten
