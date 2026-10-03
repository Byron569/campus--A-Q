"""问答主链路单元测试（M2-04）。

覆盖 docs/09 §7 C-06：**拒答不调 LLM**、降级能出原文、引用落库与展示一致；
以及 DR-11（越界引用剔除）与 docs/02 §6.2 的落库顺序。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from config.settings import Settings
from src.rag.chain import DEGRADE_NOTICE, AnswerResult, answer, stream_answer
from src.rag.prompts import build_refusal_text
from src.rag.retriever import build_retriever
from src.repository import (
    ROLE_USER,
    add_message,
    create_conversation,
    list_messages,
    list_sources_by_message,
)
from src.store.chroma import SearchHit, VectorStore
from src.store.db import get_conn


class StubLLM:
    """按队列返回内容；exc 非空时模拟调用失败，并记录每次调用。"""

    def __init__(self, responses: list[str] | None = None, exc: Exception | None = None) -> None:
        self.responses = list(responses or [])
        self.exc = exc
        self.calls: list[object] = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(content=self.responses.pop(0) if self.responses else "")


class StreamLLM:
    """按块流式返回；`exc_after` 指定在第几块之前抛错，用于验证流中途降级。"""

    def __init__(self, chunks: list[str], exc_after: int | None = None) -> None:
        self.chunks = list(chunks)
        self.exc_after = exc_after
        self.calls: list[object] = []

    def stream(self, messages):
        self.calls.append(messages)
        for index, chunk in enumerate(self.chunks):
            if self.exc_after is not None and index >= self.exc_after:
                raise RuntimeError("connection reset")
            yield SimpleNamespace(content=chunk)
        if self.exc_after is not None and self.exc_after >= len(self.chunks):
            raise RuntimeError("connection reset")


class SpyRetriever:
    """记录收到的查询串，并返回预设命中。"""

    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        self.hits = list(hits or [])
        self.queries: list[str] = []

    def search(self, query: str) -> list[SearchHit]:
        self.queries.append(query)
        return list(self.hits)


def make_hit(*, doc_id: int = 1, chunk_index: int = 0, score: float | None = 0.8) -> SearchHit:
    return SearchHit(
        text="搬迁申请：需提前三个工作日向辅导员提交书面申请。",
        score=score,
        filename="宿舍搬迁通知.docx",
        doc_id=doc_id,
        chunk_index=chunk_index,
        category="freshman",
        matched_by="both",
    )


def new_conversation(db: Path, user_id: int = 1) -> int:
    return create_conversation(user_id, title="测试会话", db_path=db)


def metric_rows(db: Path) -> list[dict]:
    with get_conn(db) as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM qa_metrics ORDER BY id").fetchall()]


# ==================== 拒答（检索为空，不调 LLM）====================


def test_refusal_when_no_hit_and_llm_not_called(
    store: VectorStore, settings: Settings, db: Path
) -> None:
    """C-06：两路皆空时直接拒答，绝不调用 LLM。"""
    llm = StubLLM(["不该被使用"])
    conversation = new_conversation(db)

    result = answer(
        "食堂今天中午吃什么",
        1,
        conversation,
        retriever=build_retriever(1, store=store, settings=settings),
        llm=llm,
        settings=settings,
        db_path=db,
    )

    assert isinstance(result, AnswerResult)
    assert result.refused is True
    assert result.degraded is False
    assert result.sources == []
    assert result.text == build_refusal_text()
    assert llm.calls == []


def test_refusal_is_persisted_and_counted_as_unanswerable(
    store: VectorStore, settings: Settings, db: Path
) -> None:
    conversation = new_conversation(db)

    answer(
        "完全不相关的问题",
        1,
        conversation,
        retriever=build_retriever(1, store=store, settings=settings),
        llm=StubLLM(),
        settings=settings,
        db_path=db,
    )

    messages = list_messages(conversation, db_path=db)
    assert [m.role for m in messages] == [ROLE_USER, "assistant"]
    assert messages[1].content == build_refusal_text()

    metrics = metric_rows(db)
    assert len(metrics) == 1
    assert metrics[0]["answerable"] == 0
    assert metrics[0]["top_doc_id"] is None


# ==================== 生成与引用（含 DR-11 越界剔除）====================


def test_answer_keeps_valid_citation_and_strips_out_of_range(
    settings: Settings, db: Path
) -> None:
    hits = [make_hit(doc_id=7, chunk_index=2)]
    llm = StubLLM(["搬迁需提前三个工作日申请【来源1】。另有说法【来源9】。"])
    conversation = new_conversation(db)

    result = answer(
        "搬宿舍要提前几天",
        1,
        conversation,
        retriever=SpyRetriever(hits),
        llm=llm,
        settings=settings,
        db_path=db,
    )

    assert result.refused is False
    assert result.degraded is False
    assert "【来源1】" in result.text
    assert "【来源9】" not in result.text  # DR-11：越界编号被剔除
    assert [hit.doc_id for hit in result.sources] == [7]


def test_sources_are_persisted_and_match_display(settings: Settings, db: Path) -> None:
    """C-06：引用落库与返回给界面的一致。"""
    hits = [make_hit(doc_id=7, chunk_index=2)]
    conversation = new_conversation(db)

    result = answer(
        "搬宿舍要提前几天",
        1,
        conversation,
        retriever=SpyRetriever(hits),
        llm=StubLLM(["结论【来源1】"]),
        settings=settings,
        db_path=db,
    )

    assistant = list_messages(conversation, db_path=db)[1]
    saved = list_sources_by_message(assistant.id, db_path=db)

    assert len(saved) == 1
    assert saved[0]["doc_id"] == hits[0].doc_id
    assert saved[0]["filename"] == hits[0].filename
    assert saved[0]["matched_by"] == "both"
    assert [hit.doc_id for hit in result.sources] == [saved[0]["doc_id"]]

    metrics = metric_rows(db)
    assert metrics[0]["answerable"] == 1
    assert metrics[0]["top_doc_id"] == 7
    assert metrics[0]["degraded"] == 0


def test_answer_without_citation_returns_no_sources(settings: Settings, db: Path) -> None:
    """模型没标来源时不伪造来源，如实返回空引用。"""
    conversation = new_conversation(db)

    result = answer(
        "搬宿舍要提前几天",
        1,
        conversation,
        retriever=SpyRetriever([make_hit()]),
        llm=StubLLM(["搬迁需提前三个工作日申请。"]),
        settings=settings,
        db_path=db,
    )

    assert result.sources == []
    assert result.refused is False


# ==================== 降级（LLM 异常）====================


def test_degraded_when_llm_raises(settings: Settings, db: Path) -> None:
    """C-06：LLM 抛异常时不崩，展示检索原文片段。"""
    hits = [make_hit(doc_id=7, chunk_index=2)]
    conversation = new_conversation(db)

    result = answer(
        "搬宿舍要提前几天",
        1,
        conversation,
        retriever=SpyRetriever(hits),
        llm=StubLLM(exc=RuntimeError("401 unauthorized")),
        settings=settings,
        db_path=db,
    )

    assert result.degraded is True
    assert result.refused is False
    assert result.text.startswith(DEGRADE_NOTICE)
    assert hits[0].text in result.text
    assert [hit.doc_id for hit in result.sources] == [7]

    metrics = metric_rows(db)
    assert metrics[0]["degraded"] == 1


# ==================== 改写与检索的衔接（M2-03 → M2-02）====================


def test_rewritten_query_is_used_for_retrieval(settings: Settings, db: Path) -> None:
    """多轮追问：检索用的是改写后的问题，不是原问题。"""
    conversation = new_conversation(db)
    add_message(conversation, ROLE_USER, "搬宿舍要怎么弄", db_path=db)
    add_message(conversation, "assistant", "需提前三个工作日申请", db_path=db)

    llm = StubLLM(["搬宿舍需要提前几天申请？", "需提前三个工作日【来源1】。"])
    retriever = SpyRetriever([make_hit()])

    answer(
        "提前几天",
        1,
        conversation,
        retriever=retriever,
        llm=llm,
        settings=settings,
        db_path=db,
    )

    assert retriever.queries == ["搬宿舍需要提前几天申请？"]
    assert len(llm.calls) == 2  # 一次改写、一次生成


def test_history_is_passed_to_rewrite_and_messages_accumulate(
    settings: Settings, db: Path
) -> None:
    conversation = new_conversation(db)
    retriever = SpyRetriever([])

    for question in ("第一问", "第二问"):
        answer(
            question,
            1,
            conversation,
            retriever=retriever,
            llm=StubLLM(),
            settings=settings,
            db_path=db,
        )

    messages = list_messages(conversation, db_path=db)
    # 两轮 = 4 条（用户 + 助手 × 2），历史随轮次累积
    assert [m.role for m in messages] == [ROLE_USER, "assistant", ROLE_USER, "assistant"]
    assert messages[0].content == "第一问"


# ==================== 流式输出（M2-05 + DR-11）====================


def test_stream_fills_result_only_after_consumption(settings: Settings, db: Path) -> None:
    conversation = new_conversation(db)
    turn = stream_answer(
        "搬宿舍要提前几天",
        1,
        conversation,
        retriever=SpyRetriever([make_hit()]),
        llm=StreamLLM(["搬迁", "需提前三个工作日【来源1】。"]),
        settings=settings,
        db_path=db,
    )

    assert turn.result is None  # 还没消费，结果未生成

    streamed = "".join(turn.tokens)

    assert turn.result is not None
    assert streamed == turn.result.text
    assert [hit.doc_id for hit in turn.result.sources] == [1]
    # 流式路径也落库
    assert len(list_messages(conversation, db_path=db)) == 2


def test_stream_strips_out_of_range_citation_split_across_chunks(
    settings: Settings, db: Path
) -> None:
    """越界编号被拆成多块时也要被剔除，且界面文本与落库文本一致（DR-11）。"""
    conversation = new_conversation(db)
    llm = StreamLLM(
        ["搬迁需", "提前三个工作日【来源1】。", "另有说法【来源", "9】。"]
    )

    turn = stream_answer(
        "搬宿舍要提前几天",
        1,
        conversation,
        retriever=SpyRetriever([make_hit()]),
        llm=llm,
        settings=settings,
        db_path=db,
    )
    streamed = "".join(turn.tokens)

    assert "【来源1】" in streamed
    assert "【来源9】" not in streamed
    assert streamed == turn.result.text
    assert [hit.doc_id for hit in turn.result.sources] == [1]

    saved = list_sources_by_message(list_messages(conversation, db_path=db)[1].id, db_path=db)
    assert len(saved) == 1  # 越界编号没有写进 message_sources


def test_stream_degrades_midway_and_keeps_partial_output(
    settings: Settings, db: Path
) -> None:
    hits = [make_hit(doc_id=7)]
    conversation = new_conversation(db)

    turn = stream_answer(
        "搬宿舍要提前几天",
        1,
        conversation,
        retriever=SpyRetriever(hits),
        llm=StreamLLM(["已经输出的半句"], exc_after=1),
        settings=settings,
        db_path=db,
    )
    streamed = "".join(turn.tokens)

    assert streamed.startswith("已经输出的半句")
    assert DEGRADE_NOTICE in streamed
    assert turn.result.degraded is True
    assert [hit.doc_id for hit in turn.result.sources] == [7]


def test_stream_degrades_before_any_output(settings: Settings, db: Path) -> None:
    hits = [make_hit(doc_id=7)]
    conversation = new_conversation(db)

    turn = stream_answer(
        "搬宿舍要提前几天",
        1,
        conversation,
        retriever=SpyRetriever(hits),
        llm=StreamLLM([], exc_after=0),
        settings=settings,
        db_path=db,
    )
    streamed = "".join(turn.tokens)

    assert streamed.startswith(DEGRADE_NOTICE)
    assert hits[0].text in streamed
    assert turn.result.degraded is True


def test_stream_refusal_path(store: VectorStore, settings: Settings, db: Path) -> None:
    conversation = new_conversation(db)

    turn = stream_answer(
        "完全不相关的问题",
        1,
        conversation,
        retriever=build_retriever(1, store=store, settings=settings),
        llm=StreamLLM(["不该被使用"]),
        settings=settings,
        db_path=db,
    )
    streamed = "".join(turn.tokens)

    assert streamed == build_refusal_text()
    assert turn.result.refused is True
    assert turn.result.sources == []
