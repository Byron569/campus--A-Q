"""问答主链路：改写 → 检索 → 拒答判定 → 生成 → 引用解析 → 降级 → 落库。

设计依据：
- docs/02-架构设计.md §4.5（防幻觉与引用解析、DR-11 越界剔除）、§4.6（故障降级）、§6.2（问答流）
- docs/06-接口文档.md §1.6（`answer(question, user_id, conversation_id, category=None) -> AnswerResult`）、§3.1
- docs/09-迭代开发计划.md §7 C-06（拒答不调 LLM；降级能出原文；引用落库与展示一致）

三条分支的判定顺序（顺序不能换，否则会白调一次 LLM）：

1. 检索为空 → **不调用 LLM**，直接输出固定拒答文案（FR-10）；
2. 检索有结果 → 组装上下文交给 LLM；
3. LLM 抛异常（超时 / 401 / 欠费 / 断网 / 未配置 Key）→ **降级**：展示检索原文片段。

引用处理遵循 DR-11：模型写出的编号若超出本次实际提供的资料条数，直接剔除，
既不渲染引用卡片、也不写入 `message_sources`，绝不伪造来源。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import Settings, get_settings
from src.providers.llm import get_llm, response_text
from src.rag.prompts import (
    build_context,
    build_refusal_text,
    build_system_prompt,
    parse_source_numbers,
    strip_invalid_sources,
)
from src.rag.retriever import build_retriever
from src.rag.rewrite import rewrite_query
from src.repository import (
    ROLE_ASSISTANT,
    ROLE_USER,
    add_message,
    add_metric,
    add_sources,
    list_messages,
)
from src.store.chroma import SearchHit

if TYPE_CHECKING:
    from src.rag.retriever import HybridRetriever
    from src.store.chroma import VectorStore

logger = logging.getLogger(__name__)

# 降级提示（docs/02 §11）
DEGRADE_NOTICE = "模型服务暂不可用，以下为原始资料："

USER_PROMPT_TEMPLATE = """【资料】
{context}

【问题】
{question}

请严格依据上述资料回答，并在每个结论后按「【来源N】」标注来源。"""


@dataclass
class AnswerResult:
    """一次问答的结构化结果（docs/06 §3.1）。"""

    text: str
    sources: list[SearchHit]
    refused: bool
    degraded: bool
    latency_ms: int


def answer(
    question: str,
    user_id: int | None,
    conversation_id: int,
    *,
    category: str | None = None,
    store: VectorStore | None = None,
    retriever: HybridRetriever | None = None,
    llm=None,
    settings: Settings | None = None,
    db_path=None,
) -> AnswerResult:
    """跑完一轮问答，并把消息、引用与指标落库。

    Args:
        question: 用户本轮提问。
        user_id: 用户身份。None 表示无身份，只检索公共文档（FR-31）。
        conversation_id: 所属会话，由上层保证存在（M2-06 负责会话管理）。
        category: 可选的分类限定（FR-14）。
        store / retriever / llm: 可注入，便于测试；生产环境按配置构造。
        settings: 配置，默认取全局单例。
        db_path: 数据库路径，默认取配置。
    """
    started = time.perf_counter()
    s = settings or get_settings()
    resolved_db = db_path or s.database_path

    history = list_messages(conversation_id, limit=s.history_rounds * 2, db_path=resolved_db)
    rewritten = rewrite_query(question, history, llm=llm, settings=s)

    if retriever is None:
        retriever = build_retriever(user_id, category, store=store, settings=s)
    hits = retriever.search(rewritten)

    if not hits:
        # 检索无结果：不调用 LLM，直接拒答（FR-10）
        logger.info("检索未命中，直接拒答：conversation_id=%s", conversation_id)
        result = AnswerResult(
            text=build_refusal_text(),
            sources=[],
            refused=True,
            degraded=False,
            latency_ms=0,
        )
    else:
        result = _generate(rewritten, hits, llm=llm, settings=s)

    result = replace(result, latency_ms=_elapsed_ms(started))
    _persist(question, result, conversation_id=conversation_id, db_path=resolved_db)
    return result


def _generate(question: str, hits: list[SearchHit], *, llm, settings: Settings) -> AnswerResult:
    """调用 LLM 生成答案；失败则降级为展示检索原文。"""
    messages = [
        SystemMessage(content=build_system_prompt()),
        HumanMessage(
            content=USER_PROMPT_TEMPLATE.format(
                context=build_context(hits), question=question
            )
        ),
    ]

    try:
        model = llm or get_llm(settings, streaming=False)
        response = model.invoke(messages)
        raw = response_text(response)
    except Exception:
        # 超时 / 401 / 欠费 / 断网 / 未配置 Key 都走同一条降级路径（docs/02 §4.6）
        logger.exception("LLM 调用失败，降级为展示检索原文")
        return AnswerResult(
            text=_degrade_text(hits),
            sources=list(hits),
            refused=False,
            degraded=True,
            latency_ms=0,
        )

    text = strip_invalid_sources(raw, max_index=len(hits))
    numbers = parse_source_numbers(text)
    cited = [hits[number - 1] for number in numbers if 1 <= number <= len(hits)]
    if not cited:
        # 有资料却没标出任何有效来源：不伪造来源，如实按无引用返回
        logger.warning("模型回答未包含有效引用编号（问题片段：%s）", question[:30])

    return AnswerResult(text=text, sources=cited, refused=False, degraded=False, latency_ms=0)


def _degrade_text(hits: list[SearchHit]) -> str:
    return "\n\n".join([DEGRADE_NOTICE, build_context(hits)])


def _persist(question: str, result: AnswerResult, *, conversation_id: int, db_path) -> None:
    """落库：用户消息 → 助手消息 → 引用 → 指标（docs/02 §6.2）。"""
    add_message(conversation_id, ROLE_USER, question, db_path=db_path)
    assistant_id = add_message(
        conversation_id, ROLE_ASSISTANT, result.text, db_path=db_path
    )
    add_sources(assistant_id, result.sources, db_path=db_path)

    top = result.sources[0] if result.sources else None
    add_metric(
        question_len=len(question),
        answerable=not result.refused,
        top_doc_id=top.doc_id if top else None,
        top_score=top.score if top else None,
        latency_ms=result.latency_ms,
        degraded=result.degraded,
        db_path=db_path,
    )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
