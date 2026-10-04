"""课件总结 / 复习提纲（二期 2.4）单元测试。

覆盖：分批与归并（map-reduce）、空输入与模型异常、产物落库的覆盖语义与按用户隔离、
删除文档 / 注销的级联清理、备份纳入，以及切片按文档过滤的权限。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from config.settings import Settings
from src import summarize
from src.errors import SummarizeError
from src.repository import (
    SUMMARY_KIND_OUTLINE,
    SUMMARY_KIND_SUMMARY,
    create_document,
    delete_user,
    export_tables,
    get_summary,
    import_tables,
    list_summaries_by_doc,
    soft_delete_document,
    upsert_summary,
)
from src.store.chroma import SearchHit, VectorStore, build_metadata
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


def _hits(*texts: str) -> list[SearchHit]:
    return [
        SearchHit(
            text=text,
            score=None,
            filename="课件.pdf",
            doc_id=1,
            chunk_index=index,
            category="course",
            matched_by="vector",
        )
        for index, text in enumerate(texts)
    ]


def _add(
    store: VectorStore,
    text: str,
    *,
    doc_id: int,
    user_id: int | None = None,
    is_public: bool = False,
    chunk_index: int = 0,
) -> None:
    metadata = build_metadata(
        doc_id=doc_id,
        user_id=user_id,
        is_public=is_public,
        category="course",
        filename="课件.pdf",
        chunk_index=chunk_index,
    )
    store.add_chunks([(text, metadata)])


# ==================== 分批与纯逻辑 ====================


def test_chunk_batches() -> None:
    assert summarize.chunk_batches(["a", "b", "c"], 2) == [["a", "b"], ["c"]]
    assert summarize.chunk_batches([], 3) == []
    assert summarize.chunk_batches(["a"], 0) == [["a"]]  # size 下限为 1


def test_usable_chunks_skips_blank() -> None:
    assert summarize.usable_chunks(_hits("内容", "   ", "")) == ["内容"]


# ==================== 生成 ====================


def test_generate_single_batch_uses_direct_prompt(settings: Settings) -> None:
    llm = StubLLM(["# 总结\n要点"])
    text = summarize.generate(
        _hits("第一章 概述", "第二章 原理"),
        filename="课件.pdf",
        kind=SUMMARY_KIND_SUMMARY,
        llm=llm,
        settings=settings,
    )

    assert text == "# 总结\n要点"
    assert len(llm.calls) == 1  # 只有一批 → 一次生成，无归并
    prompt = llm.calls[0][-1].content
    assert "内容总结" in prompt
    assert "课件.pdf" in prompt


def test_generate_multi_batch_runs_map_then_reduce(settings: Settings) -> None:
    small = settings.model_copy(update={"summarize_batch_chunks": 1})
    llm = StubLLM(["摘要A", "摘要B", "最终整合"])

    text = summarize.generate(
        _hits("第一章", "第二章"),
        filename="课件.pdf",
        kind=SUMMARY_KIND_OUTLINE,
        llm=llm,
        settings=small,
    )

    assert text == "最终整合"
    assert len(llm.calls) == 3  # 2 批摘要 + 1 次归并
    reduce_prompt = llm.calls[-1][-1].content
    assert "摘要A" in reduce_prompt and "摘要B" in reduce_prompt
    assert "复习提纲" in reduce_prompt


def test_generate_unknown_kind_raises(settings: Settings) -> None:
    with pytest.raises(SummarizeError):
        summarize.generate(
            _hits("内容"), filename="a.pdf", kind="bad", llm=StubLLM(), settings=settings
        )


def test_generate_without_text_raises(settings: Settings) -> None:
    with pytest.raises(SummarizeError):
        summarize.generate(
            _hits("   "),
            filename="a.pdf",
            kind=SUMMARY_KIND_SUMMARY,
            llm=StubLLM(),
            settings=settings,
        )


def test_generate_llm_failure_hides_internal_details(settings: Settings) -> None:
    llm = StubLLM(exc=RuntimeError("connection reset by peer"))
    with pytest.raises(SummarizeError) as err:
        summarize.generate(
            _hits("内容"),
            filename="a.pdf",
            kind=SUMMARY_KIND_SUMMARY,
            llm=llm,
            settings=settings,
        )
    assert "connection reset" not in str(err.value)


def test_generate_empty_model_output_raises(settings: Settings) -> None:
    with pytest.raises(SummarizeError):
        summarize.generate(
            _hits("内容"),
            filename="a.pdf",
            kind=SUMMARY_KIND_SUMMARY,
            llm=StubLLM(["   "]),
            settings=settings,
        )


# ==================== 落库与隔离 ====================


def test_upsert_summary_overwrites_same_kind(db: Path) -> None:
    first = upsert_summary(
        user_id=1, doc_id=10, kind=SUMMARY_KIND_SUMMARY, content="v1",
        source_chunks=3, db_path=db,
    )
    second = upsert_summary(
        user_id=1, doc_id=10, kind=SUMMARY_KIND_SUMMARY, content="v2",
        source_chunks=5, db_path=db,
    )

    assert first == second  # 覆盖而非新增
    item = get_summary(1, 10, SUMMARY_KIND_SUMMARY, db_path=db)
    assert item is not None
    assert item.content == "v2"
    assert item.source_chunks == 5
    assert len(list_summaries_by_doc(1, 10, db_path=db)) == 1


def test_summaries_isolated_by_user(db: Path) -> None:
    upsert_summary(user_id=1, doc_id=10, kind=SUMMARY_KIND_OUTLINE, content="我的", db_path=db)
    upsert_summary(user_id=2, doc_id=10, kind=SUMMARY_KIND_OUTLINE, content="别人的", db_path=db)

    assert get_summary(1, 10, SUMMARY_KIND_OUTLINE, db_path=db).content == "我的"
    assert [s.content for s in list_summaries_by_doc(1, 10, db_path=db)] == ["我的"]


def test_soft_delete_document_clears_summaries(db: Path) -> None:
    doc_id = create_document(
        filename="课件.pdf", filetype="pdf", category="course", user_id=1, db_path=db
    )
    upsert_summary(
        user_id=1, doc_id=doc_id, kind=SUMMARY_KIND_SUMMARY, content="x", db_path=db
    )

    assert soft_delete_document(doc_id, db_path=db)
    assert get_summary(1, doc_id, SUMMARY_KIND_SUMMARY, db_path=db) is None


def test_delete_user_cascades_summaries(db: Path) -> None:
    upsert_summary(user_id=1, doc_id=10, kind=SUMMARY_KIND_SUMMARY, content="x", db_path=db)
    upsert_summary(user_id=2, doc_id=10, kind=SUMMARY_KIND_SUMMARY, content="y", db_path=db)

    delete_user(1, db_path=db)

    with get_conn(db) as conn:
        remaining = [row["user_id"] for row in conn.execute("SELECT user_id FROM summaries")]
    assert remaining == [2]


def test_backup_includes_summaries(db: Path) -> None:
    upsert_summary(user_id=1, doc_id=10, kind=SUMMARY_KIND_SUMMARY, content="x", db_path=db)

    tables = export_tables(db_path=db)
    assert "summaries" in tables
    assert len(tables["summaries"]) == 1

    import_tables(tables, db_path=db)
    assert get_summary(1, 10, SUMMARY_KIND_SUMMARY, db_path=db).content == "x"


# ==================== 切片按文档过滤（权限） ====================


def test_list_chunks_filters_by_doc_id(store: VectorStore) -> None:
    _add(store, "公共通知一段", doc_id=1, is_public=True, chunk_index=0)
    _add(store, "公共通知二段", doc_id=1, is_public=True, chunk_index=1)
    _add(store, "我的笔记", doc_id=2, user_id=1)
    _add(store, "别人的笔记", doc_id=3, user_id=2)

    hits = store.list_chunks(user_id=1, doc_id=1)
    assert {hit.doc_id for hit in hits} == {1}
    assert [hit.chunk_index for hit in hits] == [0, 1]  # 顺序稳定，便于拼接
    # 加了 doc_id 也不能越过权限：他人文档取不到
    assert store.list_chunks(user_id=1, doc_id=3) == []
