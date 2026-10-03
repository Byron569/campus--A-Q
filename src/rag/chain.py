"""问答主链路：改写 → 检索 → 拒答判定 → 生成 → 引用解析 → 降级 → 落库。

设计依据：
- docs/02-架构设计.md §4.5（防幻觉与引用解析、DR-11 越界剔除）、§4.6（故障降级）、§6.2（问答流）
- docs/06-接口文档.md §1.6（`answer(...) -> AnswerResult`）、§3.1
- docs/09-迭代开发计划.md §7 C-06（拒答不调 LLM；降级能出原文；引用落库与展示一致）

三条分支的判定顺序（顺序不能换，否则会白调一次 LLM）：

1. 检索为空 → **不调用 LLM**，直接输出固定拒答文案（FR-10）；
2. 检索有结果 → 组装上下文交给 LLM；
3. LLM 抛异常（超时 / 401 / 欠费 / 断网 / 未配置 Key）→ **降级**：展示检索原文片段。

**流式与 DR-11 的冲突与处理**：M2-05 要求流式输出，而 DR-11 要求剔除越界的
`【来源N】`。流出去的文字收不回来，所以这里用 `_split_safe` 做**增量清洗**：
完整且越界的引用标记在输出前就被丢掉，疑似未写完的标记（如 `【来源1`）先攒在
缓冲区里，等补全后再判定。这样界面与落库文本始终一致，不会出现「卡片没有来源、
正文却写着【来源9】」。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
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


@dataclass
class StreamedTurn:
    """流式问答的一轮。

    `tokens` 消费完之后 `result` 才会被回填——界面必须先把 tokens 交给
    `st.write_stream`，再读 `result` 渲染引用卡片与反馈按钮。
    """

    tokens: Iterator[str]
    result: AnswerResult | None = None


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
    """跑完一轮问答并落库，直接返回结构化结果（内部即把流式结果收完）。

    参数含义见 `stream_answer`。
    """
    turn = stream_answer(
        question,
        user_id,
        conversation_id,
        category=category,
        store=store,
        retriever=retriever,
        llm=llm,
        settings=settings,
        db_path=db_path,
    )
    for _ in turn.tokens:  # 消费掉流，触发落库与 result 回填
        pass
    assert turn.result is not None  # 生成器必然回填
    return turn.result


def stream_answer(
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
) -> StreamedTurn:
    """跑完一轮问答的前半段（改写 + 检索），返回可迭代的流式结果。

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

    turn = StreamedTurn(tokens=iter(()))
    turn.tokens = _run(turn, question, rewritten, hits, conversation_id, resolved_db, s, llm, started)
    return turn


# ==================== 内部实现 ====================


def _run(
    turn: StreamedTurn,
    question: str,
    rewritten: str,
    hits: list[SearchHit],
    conversation_id: int,
    db_path,
    settings: Settings,
    llm,
    started: float,
) -> Iterator[str]:
    """生成器：逐步产出答案文本，结束时回填 result 并落库。"""
    if not hits:
        # 检索无结果：不调用 LLM，直接拒答（FR-10）
        logger.info("检索未命中，直接拒答：conversation_id=%s", conversation_id)
        result = AnswerResult(
            text=build_refusal_text(), sources=[], refused=True, degraded=False,
            latency_ms=_elapsed_ms(started),
        )
        turn.result = result
        _persist(question, result, conversation_id=conversation_id, db_path=db_path)
        yield result.text
        return

    messages = [
        SystemMessage(content=build_system_prompt()),
        HumanMessage(
            content=USER_PROMPT_TEMPLATE.format(context=build_context(hits), question=rewritten)
        ),
    ]

    parts: list[str] = []
    buffer = ""
    degraded = False
    try:
        for token in _iter_tokens(messages, llm=llm, settings=settings):
            buffer += token
            safe, buffer = _split_safe(buffer, max_index=len(hits))
            if safe:
                parts.append(safe)
                yield safe
    except Exception:
        # 超时 / 401 / 欠费 / 断网 / 未配置 Key 都走同一条降级路径（docs/02 §4.6）
        logger.exception("LLM 调用失败，降级为展示检索原文")
        degraded = True
        suffix = _degraded_suffix(hits, has_partial=bool(parts))
        parts.append(suffix)
        buffer = ""
        yield suffix

    if buffer:
        # 流结束时仍留着的片段：做最后一次清洗后补出去
        tail = strip_invalid_sources(buffer, max_index=len(hits))
        if tail:
            parts.append(tail)
            yield tail

    text = "".join(parts)
    sources = (
        list(hits)
        if degraded
        else [hits[number - 1] for number in parse_source_numbers(text) if 1 <= number <= len(hits)]
    )
    if not degraded and not sources:
        # 有资料却没标出任何有效来源：不伪造来源，如实按无引用返回
        logger.warning("模型回答未包含有效引用编号（问题片段：%s）", rewritten[:30])

    result = AnswerResult(
        text=text,
        sources=sources,
        refused=False,
        degraded=degraded,
        latency_ms=_elapsed_ms(started),
    )
    turn.result = result
    _persist(question, result, conversation_id=conversation_id, db_path=db_path)


def _iter_tokens(messages, *, llm, settings: Settings) -> Iterator[str]:
    """流式产出模型文本。注入的替身若只有 `invoke`，则一次性产出全文。"""
    model = llm or get_llm(settings, streaming=True)
    stream = getattr(model, "stream", None)
    if stream is None:
        yield response_text(model.invoke(messages))
        return
    for chunk in stream(messages):
        text = response_text(chunk)
        if text:
            yield text


def _split_safe(buffer: str, *, max_index: int) -> tuple[str, str]:
    """把缓冲区拆成「可直接输出」与「需要继续等待」两段。

    未闭合的 `【…` 可能是不完整引用，先留在缓冲区；已闭合的引用则立即剔除越界者。
    """
    tail_start = buffer.rfind("【")
    if tail_start != -1 and "】" not in buffer[tail_start:]:
        head, tail = buffer[:tail_start], buffer[tail_start:]
    else:
        head, tail = buffer, ""
    return strip_invalid_sources(head, max_index=max_index), tail


def _degraded_suffix(hits: list[SearchHit], *, has_partial: bool) -> str:
    """降级文案。已经输出过片段时只追加提示，避免前面的内容白流。"""
    body = build_context(hits)
    if has_partial:
        return f"\n\n{DEGRADE_NOTICE}"
    return f"{DEGRADE_NOTICE}\n\n{body}"


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
